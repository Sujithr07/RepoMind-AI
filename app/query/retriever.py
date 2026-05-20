import asyncio
import asyncpg
import voyageai
import hashlib
import json
import os
import re
import uuid
from app.query.hyde import generate_hyde
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi

load_dotenv()

voyage_client = voyageai.AsyncClient(api_key=os.getenv("VOYAGE_API_KEY"))

# Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_client = QdrantClient(url=QDRANT_URL)
COLLECTION_NAME = "codebase_chunks"

# Redis key prefix for the per-repo BM25 index payload
BM25_KEY_PREFIX = "bm25_index"
_TOKEN_RE = re.compile(r"[\W_]+")


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


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-word characters and underscores.
    Matches the spec: text.lower() split by `[\\W_]+`.
    """
    if not text:
        return []
    return [tok for tok in _TOKEN_RE.split(text.lower()) if tok]


# ---------------------------------------------------------------------------
# BM25 index lifecycle
# ---------------------------------------------------------------------------
def _chunk_to_record(chunk) -> dict:
    """Coerce a CodeChunk dataclass or dict into the storage shape used at query time."""
    if isinstance(chunk, dict):
        return {
            "content": chunk["content"],
            "context_prefix": chunk["context_prefix"],
            "file_path": chunk["file_path"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
        }
    return {
        "content": chunk.content,
        "context_prefix": chunk.context_prefix,
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
    }


async def build_bm25_index(chunks: list, repo_id, redis_client) -> BM25Okapi:
    """Build a BM25 index over chunk content and persist it in Redis.

    The pickled BM25Okapi state is non-portable, so we instead store the
    tokenised corpus + chunk metadata as JSON and rebuild BM25Okapi on load.
    Building BM25 is O(N * tokens) and fast enough to do per query.
    """
    records = [_chunk_to_record(c) for c in chunks]
    tokenized = [_tokenize(r["content"]) for r in records]

    payload = json.dumps({"chunks": records, "tokenized": tokenized})
    await redis_client.set(f"{BM25_KEY_PREFIX}:{repo_id}", payload)

    print(f"[bm25] indexed {len(records)} chunks for repo {repo_id}")
    return BM25Okapi(tokenized) if tokenized else None


async def _load_bm25(repo_id, redis_client) -> tuple[BM25Okapi, list[dict]] | None:
    """Fetch the BM25 corpus for a repo from Redis and reconstruct BM25Okapi.
    Returns None when no index exists (repo indexed before BM25 was added).
    """
    raw = await redis_client.get(f"{BM25_KEY_PREFIX}:{repo_id}")
    if not raw:
        return None
    try:
        data = json.loads(raw)
        tokenized = data["tokenized"]
        chunks = data["chunks"]
        if not tokenized:
            return None
        return BM25Okapi(tokenized), chunks
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[bm25] failed to load index for {repo_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Sparse (BM25) and dense (vector) search
# ---------------------------------------------------------------------------
async def search_bm25(
    query: str,
    repo_id,
    redis_client,
    top_k: int = 20,
) -> list[dict]:
    """Return top_k chunks ranked by BM25 over the raw query tokens.
    Falls back to an empty list if no BM25 index exists for the repo.
    """
    loaded = await _load_bm25(str(repo_id), redis_client)
    if loaded is None:
        return []
    bm25, chunks = loaded

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    ranked = sorted(
        zip(chunks, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [chunk for chunk, score in ranked[:top_k] if score > 0]


async def _vector_search(
    query: str,
    repo_id: uuid.UUID,
    top_k: int = 20,
) -> list[dict]:
    """HyDE-expanded dense retrieval against Qdrant."""
    hypothetical = await generate_hyde(query)
    vec = await _embed_query(hypothetical)

    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="repo_id",
                    match=MatchValue(value=str(repo_id)),
                )
            ]
        ),
        limit=top_k,
    ).points

    return [
        {
            "content": r.payload["content"],
            "context_prefix": r.payload["context_prefix"],
            "file_path": r.payload["file_path"],
            "start_line": r.payload["start_line"],
            "end_line": r.payload["end_line"],
            "relevance_score": r.score,
        }
        for r in search_results
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def _chunk_key(chunk: dict) -> tuple:
    """Stable identity for a chunk across vector / BM25 result lists."""
    return (
        chunk.get("file_path"),
        chunk.get("start_line"),
        chunk.get("end_line"),
    )


def fuse_rankings(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over rankers of 1 / (k + rank + 1).
    `k=60` is the canonical constant from Cormack et al. (2009).
    """
    fused: dict[tuple, dict] = {}

    for rank, chunk in enumerate(vector_results):
        key = _chunk_key(chunk)
        entry = fused.setdefault(key, {"chunk": chunk, "score": 0.0})
        entry["score"] += 1.0 / (k + rank + 1)

    for rank, chunk in enumerate(bm25_results):
        key = _chunk_key(chunk)
        # Prefer the vector-side chunk record (carries relevance_score) when present.
        entry = fused.setdefault(key, {"chunk": chunk, "score": 0.0})
        entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)
    return [
        {**entry["chunk"], "fused_score": entry["score"]}
        for entry in ranked
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
async def retrieve(
    query: str,
    repo_id: uuid.UUID,
    conn: asyncpg.Connection,
    redis_client,
    top_k: int = 5,
    candidate_pool: int = 20,
) -> list[dict]:
    """Hybrid retrieval: dense (vector) + sparse (BM25) fused via RRF.

    Falls back to vector-only when no BM25 index exists for the repo.
    """
    cache_key = f"query:{hashlib.sha256(f'{query}{repo_id}{top_k}'.encode()).hexdigest()}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Run dense + sparse retrieval concurrently.
    vector_results, bm25_results = await asyncio.gather(
        _vector_search(query, repo_id, top_k=candidate_pool),
        search_bm25(query, repo_id, redis_client, top_k=candidate_pool),
    )

    if not vector_results and not bm25_results:
        return []

    fused = fuse_rankings(vector_results, bm25_results)
    results = fused[:top_k]

    # Cache for 1 hour; guard against non-serialisable floats (NaN/inf)
    await redis_client.setex(
        cache_key, 3600,
        json.dumps(results, default=lambda x: str(x)),
    )
    return results
