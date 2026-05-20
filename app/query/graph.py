import os
import uuid
import asyncpg
from typing import TypedDict, Annotated, Literal
from typing_extensions import Annotated as TypingAnnotated
from groq import Groq
from dotenv import load_dotenv

from app.query.hyde import generate_hyde
from app.query.retriever import retrieve
from app.query.answerer import stream_answer

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


class QueryState(TypedDict):
    question: str
    repo_id: str
    hyde_snippet: str
    chunks: list
    answer: str
    citations: list
    retry_count: int
    conn: asyncpg.Connection
    redis_client: object
    history: list


async def hyde_node(state: QueryState) -> dict:
    """Generate HyDE snippet from the question."""
    result = await generate_hyde(state["question"])
    return {"hyde_snippet": result}


async def retrieve_node(state: QueryState) -> dict:
    """Retrieve chunks using the HyDE snippet."""
    real_repo_id = uuid.UUID(state["repo_id"])
    result = await retrieve(
        state["hyde_snippet"],
        real_repo_id,
        state["conn"],
        state["redis_client"]
    )
    return {"chunks": result}


async def reflect_node(state: QueryState) -> dict:
    """Reflect on whether the retrieved chunks are relevant."""
    question = state["question"]
    chunks = state["chunks"]
    
    # Take first 2 chunks for reflection
    chunks_preview = chunks[:2] if chunks else []
    chunks_text = "\n\n".join([c.get("content", "") for c in chunks_preview])
    
    prompt = f"Given the question: {question} and these retrieved chunks: {chunks_text}, are the chunks relevant? Reply with only YES or NO."
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"Reflection failed: {e}")
        result = "YES"  # Default to YES on error
    
    return {"retry_count": state["retry_count"] + 1, "_reflection_result": result}


async def answer_node(state: QueryState) -> dict:
    """Generate the final answer using the retrieved chunks."""
    result = ""
    citations = []
    async for token, cit in stream_answer(state["question"], state["chunks"], state.get("history")):
        result += token
        if cit is not None:
            citations = cit.get("citations", [])
    return {"answer": result, "citations": citations}


def should_retry(state: QueryState) -> Literal["retrieve", "answer"]:
    """Decide whether to retry retrieval or proceed to answer."""
    reflection = state.get("_reflection_result", "YES")
    retry_count = state["retry_count"]
    
    if reflection == "NO" and retry_count < 2:
        return "retrieve"
    return "answer"


from langgraph.graph import StateGraph, END

builder = StateGraph(QueryState)

# Add nodes
builder.add_node("hyde", hyde_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("reflect", reflect_node)
builder.add_node("answer", answer_node)

# Set entry point
builder.set_entry_point("hyde")

# Add edges
builder.add_edge("hyde", "retrieve")
builder.add_edge("retrieve", "reflect")
builder.add_conditional_edges(
    "reflect",
    should_retry,
    {
        "retrieve": "retrieve",
        "answer": "answer"
    }
)
builder.add_edge("answer", END)

# Compile the graph
query_graph = builder.compile()
