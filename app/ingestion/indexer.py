# app/ingestion/indexer.py
import uuid
import asyncpg
import os
import redis.asyncio as aioredis
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from app.ingestion.chunker import CodeChunk
from app.query.retriever import build_bm25_index

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_client = QdrantClient(url=QDRANT_URL)

# Create collection if it doesn't exist
COLLECTION_NAME = "codebase_chunks"
try:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
except Exception:
    # Collection already exists
    pass


async def upsert_chunks(
    conn: asyncpg.Connection,
    repo_id: str,
    chunks: list[CodeChunk],
    embeddings: list[list[float]],
) -> None:
    """Batch upsert chunks + embeddings into Qdrant, then build a BM25 index
    over the same chunks and persist it in Redis for hybrid retrieval.

    PostgreSQL keeps repo metadata only; chunk content lives in Qdrant payloads
    (dense) and the Redis-backed BM25 corpus (sparse).
    """

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "repo_id": repo_id,
                "file_path": c.file_path,
                "content": c.content,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "context_prefix": c.context_prefix,
            }
        )
        for c, emb in zip(chunks, embeddings)
    ]

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"[indexer] upserted {len(points)} chunks for repo {repo_id}")

    # Build & cache BM25 index for the same corpus. Use a short-lived async
    # Redis client because the indexer runs in a Celery worker process that
    # does not share FastAPI's connection pool.
    redis_client = aioredis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
    )
    try:
        await build_bm25_index(chunks, repo_id, redis_client)
    except Exception as e:
        # BM25 is best-effort: if it fails, retrieval falls back to vector-only.
        print(f"[indexer] BM25 index build failed for repo {repo_id}: {e}")
    finally:
        await redis_client.aclose()