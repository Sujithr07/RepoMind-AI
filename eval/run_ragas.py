"""
RAG Pipeline Evaluation using RAGAS

This script evaluates the quality of the RAG pipeline using the RAGAS framework
with metrics: faithfulness, answer_relevancy, and context_recall.
"""

import os
import sys
import asyncio
import uuid
import json
import asyncpg
import redis.asyncio as aioredis
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel
from typing import Dict, List, Any

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.query.retriever import retrieve
from app.query.answerer import stream_answer

load_dotenv()

# Test cases - 15 realistic questions about a Python/FastAPI codebase
TEST_CASES = [
    {
        "question": "How do I add a new API endpoint?",
        "ground_truth": "To add a new API endpoint, define a new function with the @app.post(), @app.get(), @app.put(), or @app.delete() decorator. The function should accept a Pydantic model as input if it's a POST/PUT request, and return the response data."
    },
    {
        "question": "What does the chunker return?",
        "ground_truth": "The chunker returns a list of code chunks, where each chunk contains metadata including file_path, language, chunk_type, name, start_line, end_line, content, and context_prefix. These chunks are used for embedding and retrieval."
    },
    {
        "question": "How is Redis used in this project?",
        "ground_truth": "Redis is used for caching query results to improve performance. The retrieve function caches query results for 1 hour using a cache key based on the query, repo_id, and top_k parameters."
    },
    {
        "question": "What is the purpose of Multi-Query Fusion?",
        "ground_truth": "Multi-Query Fusion expands the user's question into several semantically diverse search queries in a single LLM call (via Groq). Each reformulation is retrieved independently and the result sets are combined with Reciprocal Rank Fusion. This surfaces relevant code that a single phrasing of the question would miss, improving recall."
    },
    {
        "question": "How does the retrieval pipeline work?",
        "ground_truth": "The retrieval pipeline expands the question into several diverse queries via Multi-Query Fusion (one Groq call), embeds each query with Cohere embed-english-v3.0 and searches Qdrant, runs BM25 sparse search in parallel, fuses all result sets with Reciprocal Rank Fusion, then reranks the candidates with Cohere rerank-english-v3.0 to select the top_k most relevant chunks."
    },
    {
        "question": "What models are used for embedding?",
        "ground_truth": "The project uses Cohere's embed-english-v3.0 model (1024-dimensional) to embed code snippets and queries. For reranking it uses Cohere's rerank-english-v3.0, with a local cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) as an automatic fallback when the Cohere API is unavailable or rate-limited."
    },
    {
        "question": "How is the database connection managed?",
        "ground_truth": "The database connection is managed using asyncpg with a connection pool. The pool is created in the FastAPI lifespan function and closed when the app shuts down. Connections are acquired using async with db_pool.acquire()."
    },
    {
        "question": "What is the purpose of the reflection node?",
        "ground_truth": "The reflection node checks whether the retrieved chunks are relevant to the question. It uses an LLM to evaluate relevance and decides whether to retry retrieval or proceed to answer generation."
    },
    {
        "question": "How does the answer generation work?",
        "ground_truth": "The answer generation uses the Groq API with the llama-3.3-70b-versatile model. It streams the response token-by-token, providing the retrieved code chunks as context. The system prompt instructs the model to answer using only the provided context."
    },
    {
        "question": "What is the chunk size and overlap strategy?",
        "ground_truth": "The chunker uses tree-sitter to parse code and create chunks based on AST nodes rather than fixed sizes. This ensures chunks are semantically meaningful, typically corresponding to functions, classes, or other code blocks."
    },
    {
        "question": "How are webhooks handled?",
        "ground_truth": "The GitHub webhook endpoint re-indexes repositories when a push event occurs on the default branch. It verifies the ref matches the default branch, looks up the repo by clone_url, and enqueues an indexing task."
    },
    {
        "question": "What is the retry mechanism in the query graph?",
        "ground_truth": "The query graph has a retry mechanism controlled by the reflection node. If chunks are deemed irrelevant and retry_count < 2, it retries retrieval. Otherwise, it proceeds to answer generation."
    },
    {
        "question": "How is conversation history managed?",
        "ground_truth": "Conversation history is stored in Redis using the session_id as the key. The history is retrieved before query execution and updated after the answer is generated, allowing for multi-turn conversations."
    },
    {
        "question": "What is the role of LangGraph in this project?",
        "ground_truth": "LangGraph orchestrates the RAG pipeline as a state machine with a retrieve node and a reflect node. It manages the flow between them and handles conditional logic for retries; answer generation is streamed separately by the caller."
    },
    {
        "question": "How are embeddings stored and indexed?",
        "ground_truth": "Embeddings are stored in Qdrant vector database. Each chunk is stored with its embedding vector, metadata (file_path, content, context_prefix, etc.), and repo_id for filtering during search."
    }
]


async def get_database_connection() -> asyncpg.Pool:
    """Create a database connection pool."""
    raw_url = os.getenv("DATABASE_URL", "")
    db_url = raw_url.replace("postgresql+asyncpg://", "postgresql://").replace("postgres://", "postgresql://")
    db_url = db_url.split("?")[0]

    use_ssl = not ("localhost" in db_url or "127.0.0.1" in db_url)
    pool = await asyncpg.create_pool(db_url, ssl="require" if use_ssl else False)
    return pool


async def get_redis_client() -> aioredis.Redis:
    """Create a Redis client."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = aioredis.from_url(redis_url, decode_responses=True)
    return client


async def get_ready_repo_id(conn: asyncpg.Connection) -> str:
    """Get the first ready repo_id from the database."""
    row = await conn.fetchrow(
        "SELECT id FROM repos WHERE status = 'ready' LIMIT 1"
    )
    if not row:
        raise ValueError("No ready repositories found in database. Please index a repository first.")
    return str(row["id"])


async def run_rag_pipeline(question: str, repo_id: str, conn: asyncpg.Connection, redis_client: aioredis.Redis) -> Dict[str, Any]:
    """
    Run the full RAG pipeline for a single question.
    
    Returns:
        Dict with 'answer' (str) and 'contexts' (list of chunk texts)
    """
    # Step 1: Retrieve chunks (Multi-Query Fusion expansion happens inside `retrieve`)
    chunks = await retrieve(
        query=question,
        repo_id=uuid.UUID(repo_id),
        conn=conn,
        redis_client=redis_client,
        top_k=5
    )

    # Step 2: Generate answer. stream_answer yields (text_delta, citations|None);
    # the final yield carries the fully parsed answer.
    answer = ""
    async for delta, cit in stream_answer(question, chunks, history=None):
        if cit is not None:
            answer = cit.get("answer") or answer
        elif delta:
            answer += delta

    # Extract context texts for evaluation
    contexts = [chunk["content"] for chunk in chunks]
    
    return {
        "answer": answer,
        "contexts": contexts
    }


async def main():
    """Main evaluation function."""
    print("🚀 Starting RAG Pipeline Evaluation with RAGAS\n")
    
    # Setup connections
    print("📡 Connecting to database and Redis...")
    db_pool = await get_database_connection()
    redis_client = await get_redis_client()
    
    try:
        # Get a ready repo_id
        async with db_pool.acquire() as conn:
            repo_id = await get_ready_repo_id(conn)
            print(f"✅ Using repository: {repo_id}\n")
        
        # Run pipeline for all test cases
        print("🔄 Running RAG pipeline for test cases...")
        results = []
        
        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"  [{i}/{len(TEST_CASES)}] Processing: {test_case['question'][:50]}...")
            
            async with db_pool.acquire() as conn:
                rag_result = await run_rag_pipeline(
                    question=test_case["question"],
                    repo_id=repo_id,
                    conn=conn,
                    redis_client=redis_client
                )
            
            results.append({
                "question": test_case["question"],
                "ground_truth": test_case["ground_truth"],
                "answer": rag_result["answer"],
                "contexts": rag_result["contexts"]
            })
            print(f"    ✓ Answer generated ({len(rag_result['answer'])} chars, {len(rag_result['contexts'])} contexts)")
        
        print(f"\n✅ Pipeline completed for all {len(TEST_CASES)} test cases\n")
        
        # Build HuggingFace Dataset
        print("📊 Building HuggingFace Dataset...")
        dataset_dict = {
            "question": [r["question"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results]
        }
        dataset = Dataset.from_dict(dataset_dict)
        print(f"✅ Dataset created with {len(dataset)} examples\n")
        
        # Setup Groq as LLM judge
        print("🧠 Setting up Groq as LLM judge...")
        groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0
        )
        print("✅ Groq LLM initialized\n")
        
        # Run evaluation
        print("📈 Running RAGAS evaluation...")
        print("   Metrics: faithfulness, answer_relevancy, context_recall")
        print("   This may take several minutes...\n")
        
        eval_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
            llm=groq_llm
        )
        
        # Print results as table
        print("\n" + "="*80)
        print("📊 EVALUATION RESULTS")
        print("="*80)
        print(eval_result.to_pandas())
        print("="*80 + "\n")
        
        # Save results to JSON
        os.makedirs("eval", exist_ok=True)
        results_path = "eval/results.json"
        
        results_data = {
            "scores": eval_result.to_pandas().to_dict(orient="records"),
            "average_scores": {
                "faithfulness": float(eval_result.to_pandas()["faithfulness"].mean()),
                "answer_relevancy": float(eval_result.to_pandas()["answer_relevancy"].mean()),
                "context_recall": float(eval_result.to_pandas()["context_recall"].mean())
            },
            "test_cases": results
        }
        
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to {results_path}\n")
        
        # Print summary
        print("📋 SUMMARY")
        print("-" * 40)
        print(f"Faithfulness:      {results_data['average_scores']['faithfulness']:.4f}")
        print(f"Answer Relevancy:  {results_data['average_scores']['answer_relevancy']:.4f}")
        print(f"Context Recall:    {results_data['average_scores']['context_recall']:.4f}")
        print("-" * 40 + "\n")
        
    finally:
        await db_pool.close()
        await redis_client.aclose()
        print("🔌 Connections closed")


if __name__ == "__main__":
    asyncio.run(main())
