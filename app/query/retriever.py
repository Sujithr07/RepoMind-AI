import asyncio
import asyncpg
import voyageai
import hashlib
import json
import os
import uuid
from app.query.hyde import generate_hyde
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

voyage_client = voyageai.AsyncClient(api_key=os.getenv("VOYAGE_API_KEY"))


async def _embed_query(text: str) -> list[float]:
    """Embed a query string, retrying on Voyage free-tier (3 RPM) rate limits."""
    delays = [0, 21, 42, 63]
    last_error = None
    for delay in delays:
        if delay:
            await asyncio.sleep(delay)
        try:
            result = await voyage_client.embed(
                [text],
                model="voyage-code-3",
                input_type="query",
            )
            return result.embeddings[0]
        except voyageai.error.RateLimitError as e:
            last_error = e
    raise last_error

# Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_client = QdrantClient(url=QDRANT_URL)
COLLECTION_NAME = "codebase_chunks"


async def retrieve(
    query: str,
    repo_id: uuid.UUID,           # Fix: use UUID type, not str
    conn: asyncpg.Connection,
    redis_client,
    top_k: int = 5,
) -> list[dict]:

    # Fix: include top_k in the cache key so different top_k values don't collide
    cache_key = f"query:{hashlib.sha256(f'{query}{repo_id}{top_k}'.encode()).hexdigest()}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # HyDE: embed a hypothetical answer, not the raw question.
    # Must use the same model as the embedder (voyage-code-3) so the query
    # vector lives in the same space as the indexed document vectors.
    hypothetical = await generate_hyde(query)
    vec = await _embed_query(hypothetical)

    # Vector search: top 20 candidates from Qdrant
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="repo_id",
                    match=MatchValue(value=str(repo_id))
                )
            ]
        ),
        limit=20
    ).points
    
    rows = [
        {
            "id": result.id,
            "content": result.payload["content"],
            "context_prefix": result.payload["context_prefix"],
            "file_path": result.payload["file_path"],
            "start_line": result.payload["start_line"],
            "end_line": result.payload["end_line"],
            "similarity": result.score,
        }
        for result in search_results
    ]

    if not rows:
        return []

    # Take top_k directly from Qdrant results (no reranking for now)
    top_chunks = rows[:top_k]

    results = [
        {
            "content": chunk["content"],
            "context_prefix": chunk["context_prefix"],
            "file_path": chunk["file_path"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "relevance_score": chunk["similarity"],
        }
        for chunk in top_chunks
    ]

    # Cache for 1 hour; guard against non-serialisable floats (NaN/inf)
    await redis_client.setex(
        cache_key, 3600,
        json.dumps(results, default=lambda x: str(x))
    )
    return results
