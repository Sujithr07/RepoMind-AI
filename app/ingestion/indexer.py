# app/ingestion/indexer.py
import uuid
import asyncpg
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from app.ingestion.chunker import CodeChunk

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
    """Batch upsert chunks + embeddings into Qdrant.
    Keeps PostgreSQL for repo metadata only."""

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