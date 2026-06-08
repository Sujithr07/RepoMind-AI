# app/ingestion/indexer.py
import asyncio
import uuid
import asyncpg
import os
import redis.asyncio as aioredis
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)
from app.ingestion.chunker import CodeChunk
from app.query.retriever import build_bm25_index, merge_bm25_index

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)

COLLECTION_NAME = "codebase_chunks"
UPSERT_BATCH_SIZE = 500  # cap points per request so large repos don't send one massive payload

# Guard so the collection is only created once per process.
_collection_ready: asyncio.Lock = asyncio.Lock()
_collection_created = False


async def _ensure_collection() -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    global _collection_created
    if _collection_created:
        return
    async with _collection_ready:
        if _collection_created:
            return
        try:
            await qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
        except Exception:
            # Collection already exists
            pass
        _collection_created = True


async def list_indexed_files(repo_id: str) -> list[str]:
    """Return the sorted, distinct file paths indexed for a repo.

    Chunk content lives only in Qdrant payloads (the Postgres ``chunks`` table is
    unused), so we scroll the collection filtering by ``repo_id`` and collect the
    distinct ``file_path`` values. Only that one payload field is fetched.
    """
    await _ensure_collection()
    paths: set[str] = set()
    flt = Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))])
    next_offset = None
    while True:
        points, next_offset = await qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=flt,
            with_payload=["file_path"],
            with_vectors=False,
            limit=512,
            offset=next_offset,
        )
        for p in points:
            fp = (p.payload or {}).get("file_path")
            if fp:
                # Stored paths are OS-native (backslashes on Windows); normalize
                # to forward slashes so clients get consistent POSIX-style paths.
                paths.add(fp.replace("\\", "/"))
        if next_offset is None:
            break
    return sorted(paths)


async def delete_chunks_for_files(
    conn: asyncpg.Connection,
    repo_id: str,
    file_paths: set[str],
) -> None:
    """Delete existing Qdrant points and DB rows for specific file paths only.

    Used by delta re-indexing to evict the stale chunks of changed/removed files
    without wiping the whole repo. A no-op when ``file_paths`` is empty.
    """
    if not file_paths:
        return

    await _ensure_collection()
    paths = list(file_paths)

    await qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(key="repo_id", match=MatchValue(value=repo_id)),
                FieldCondition(key="file_path", match=MatchAny(any=paths)),
            ]
        ),
    )
    await conn.execute(
        "DELETE FROM chunks WHERE repo_id = $1::uuid AND file_path = ANY($2::text[])",
        repo_id,
        paths,
    )
    print(f"[indexer] deleted stale chunks for {len(paths)} files in repo {repo_id}")


async def upsert_chunks(
    conn: asyncpg.Connection,
    repo_id: str,
    chunks: list[CodeChunk],
    embeddings: list[list[float]],
    replaced_file_paths: set[str] | None = None,
) -> None:
    """Batch upsert chunks + embeddings into Qdrant, then refresh the BM25 index
    in Redis for hybrid retrieval.

    PostgreSQL keeps repo metadata only; chunk content lives in Qdrant payloads
    (dense) and the Redis-backed BM25 corpus (sparse).

    ``replaced_file_paths`` distinguishes the two indexing modes:
      * ``None``  -> full index: rebuild the BM25 corpus from ``chunks`` alone.
      * a set     -> delta index: merge ``chunks`` into the existing BM25 corpus,
                     dropping any prior records for these file paths so unchanged
                     files survive.
    """

    await _ensure_collection()

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

    batches = [
        points[i : i + UPSERT_BATCH_SIZE]
        for i in range(0, len(points), UPSERT_BATCH_SIZE)
    ]
    await asyncio.gather(
        *(
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch)
            for batch in batches
        )
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
        if replaced_file_paths is None:
            await build_bm25_index(chunks, repo_id, redis_client)
        else:
            await merge_bm25_index(chunks, replaced_file_paths, repo_id, redis_client)
    except Exception as e:
        # BM25 is best-effort: if it fails, retrieval falls back to vector-only.
        print(f"[indexer] BM25 index build failed for repo {repo_id}: {e}")
    finally:
        await redis_client.aclose()