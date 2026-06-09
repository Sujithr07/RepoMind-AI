import asyncio
import asyncpg
import hashlib
import json
import os
import re
import uuid
import cohere
from app.query.fusion import generate_fusion_queries
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi

load_dotenv()

cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
_async_cohere_client = cohere.AsyncClient(api_key=os.getenv("COHERE_API_KEY"))

# Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_client = QdrantClient(url=QDRANT_URL)
COLLECTION_NAME = "codebase_chunks"

# Redis key prefix for the per-repo BM25 index payload
BM25_KEY_PREFIX = "bm25_index"
_TOKEN_RE = re.compile(r"[\W_]+")

EMBED_MODEL = "embed-english-v3.0"
# Query embeddings are deterministic per (text, model), so they cache well. 24h
# bounds Redis growth while still saving repeated Cohere calls within a session.
QUERY_EMBED_CACHE_TTL = 86400


async def _embed_query(text: str, redis_client=None) -> list[float]:
    """Embed a query string via Cohere, caching the result in Redis.

    Caching matters now that Multi-Query Fusion embeds several queries per search:
    recurring reformulations (or repeated questions) are served from cache instead
    of re-spending Cohere quota. Caching is best-effort — any Redis error falls
    through to a live embed call.

    Note: there is deliberately no fallback embedding provider. Embeddings from a
    different model live in a different vector space and are not comparable to the
    Cohere vectors already stored in Qdrant, so silently swapping models would
    corrupt retrieval. The embedding provider must stay singular.
    """
    cache_key = None
    if redis_client is not None:
        cache_key = f"qembed:{EMBED_MODEL}:{hashlib.sha256(text.encode()).hexdigest()}"
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"[retriever] query-embed cache read failed: {e}")

    response = await _async_cohere_client.embed(
        texts=[text],
        model=EMBED_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    vec = response.embeddings.float_[0]

    if cache_key is not None:
        try:
            await redis_client.setex(cache_key, QUERY_EMBED_CACHE_TTL, json.dumps(vec))
        except Exception as e:
            print(f"[retriever] query-embed cache write failed: {e}")
    return vec


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


async def merge_bm25_index(
    new_chunks: list,
    replaced_file_paths,
    repo_id,
    redis_client,
) -> BM25Okapi:
    """Merge ``new_chunks`` into the repo's existing BM25 corpus and persist it.

    For delta re-indexing: load the current corpus, drop every record whose
    ``file_path`` is in ``replaced_file_paths`` (the changed/removed files), then
    append the freshly chunked records. This keeps records for unchanged files
    intact, so sparse retrieval stays correct over the whole repo.

    Falls back to a plain build when no prior corpus exists.
    """
    replaced = set(replaced_file_paths)

    raw = await redis_client.get(f"{BM25_KEY_PREFIX}:{repo_id}")
    kept: list[dict] = []
    if raw:
        try:
            existing = json.loads(raw).get("chunks", [])
            kept = [r for r in existing if r.get("file_path") not in replaced]
        except Exception as e:
            # Corrupt/legacy payload: rebuild from scratch rather than fail.
            print(f"[bm25] could not read existing corpus for {repo_id}: {e}")
            kept = []

    new_records = [_chunk_to_record(c) for c in new_chunks]
    all_records = kept + new_records
    tokenized = [_tokenize(r["content"]) for r in all_records]

    payload = json.dumps({"chunks": all_records, "tokenized": tokenized})
    await redis_client.set(f"{BM25_KEY_PREFIX}:{repo_id}", payload)

    print(
        f"[bm25] merged {len(new_records)} new + {len(kept)} kept "
        f"= {len(all_records)} chunks for repo {repo_id}"
    )
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
    redis_client=None,
    top_k: int = 20,
) -> list[dict]:
    """Dense retrieval against Qdrant for a single (already-expanded) query.

    Query expansion now happens upstream in ``retrieve`` via Multi-Query Fusion,
    so this just embeds the given query (cached via ``redis_client``) and searches.
    """
    vec = await _embed_query(query, redis_client)

    try:
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
    except Exception as e:
        print(f"[retriever] Qdrant search failed (collection may not exist): {e}")
        return []

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
    *result_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion over any number of ranked result lists:
    score = sum over rankers of 1 / (k + rank + 1).
    `k=60` is the canonical constant from Cormack et al. (2009).

    Accepts a variable number of lists so Multi-Query Fusion can fuse the dense +
    sparse results of every reformulation in one pass. The two-argument hybrid
    call ``fuse_rankings(vector_results, bm25_results)`` remains valid.
    The first list a chunk appears in wins the stored record, so put the
    relevance_score-carrying vector lists ahead of BM25 lists.
    """
    fused: dict[tuple, dict] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            key = _chunk_key(chunk)
            entry = fused.setdefault(key, {"chunk": chunk, "score": 0.0})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)
    return [
        {**entry["chunk"], "fused_score": entry["score"]}
        for entry in ranked
    ]


# ---------------------------------------------------------------------------
# Reranking: Cohere primary, local cross-encoder fallback
# ---------------------------------------------------------------------------
LOCAL_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None  # lazily loaded singleton; only touched on the fallback path


def _get_cross_encoder():
    """Lazily load the local cross-encoder. Imported and instantiated on first
    use only (the model weights download once, ~80MB), so the heavy
    sentence-transformers import never runs unless the Cohere path is unavailable."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(LOCAL_RERANK_MODEL)
    return _cross_encoder


def _local_rerank(
    query: str,
    results: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """Rerank locally with a cross-encoder when Cohere is unavailable or
    rate-limited. Runs fully offline, so it never consumes API quota. Falls back
    to the original fused order if the model can't be loaded or scored."""
    try:
        model = _get_cross_encoder()
        scores = model.predict([(query, r.get("content", "")) for r in results])
        ranked = sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)
        reranked = []
        for chunk, score in ranked[: top_k or len(results)]:
            chunk = chunk.copy()
            chunk["rerank_score"] = float(score)
            reranked.append(chunk)
        print(f"[rerank] used local cross-encoder for {len(results)} candidates")
        return reranked
    except Exception as e:
        print(f"[rerank] local cross-encoder failed, using fused order: {e}")
        return results


def rerank_with_cohere(
    query: str,
    results: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """Rerank results, preferring Cohere's hosted reranker and degrading to a
    local cross-encoder when Cohere is unavailable.

    Resolution order:
      1. Cohere ``rerank-english-v3.0`` (best quality) when an API key is set.
      2. Local cross-encoder fallback on any Cohere error/rate-limit, or when no
         API key is configured — keeps reranking working without spending quota.
      3. Original fused order if the local model is unavailable too.

    Returns a list of chunk dicts with a ``rerank_score`` field added.
    """
    if not results:
        return []

    api_key = os.getenv("COHERE_API_KEY")
    if api_key and api_key != "your_cohere_api_key_here":
        try:
            documents = [r.get("content", "") for r in results]
            response = cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=top_k or len(results),
            )
            reranked = []
            for result in response.results:
                chunk = results[result.index].copy()
                chunk["rerank_score"] = result.relevance_score
                reranked.append(chunk)
            return reranked
        except Exception as e:
            print(f"[cohere] rerank failed, falling back to local cross-encoder: {e}")
    else:
        print("[cohere] no API key, using local cross-encoder rerank")

    # Fallback: local, quota-free reranking.
    return _local_rerank(query, results, top_k)


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
    use_rerank: bool = True,
) -> list[dict]:
    """Multi-Query RAG Fusion: expand the question into several diverse queries
    (one LLM call), run dense (vector) + sparse (BM25) retrieval for each, fuse
    every result list via RRF, then optionally rerank with Cohere.

    Falls back to single-query retrieval when expansion fails, and to vector-only
    when no BM25 index exists for the repo.
    """
    cache_key = f"query:{hashlib.sha256(f'{query}{repo_id}{top_k}{use_rerank}'.encode()).hexdigest()}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # One LLM call -> diverse reformulations (the original question included).
    queries = await generate_fusion_queries(query)

    # Fan out: dense + sparse retrieval for every query variant, all concurrently.
    # Vector lists are gathered ahead of BM25 lists so RRF keeps the vector-side
    # chunk record (which carries relevance_score) on ties.
    vector_coros = [
        _vector_search(q, repo_id, redis_client, top_k=candidate_pool) for q in queries
    ]
    bm25_coros = [
        search_bm25(q, repo_id, redis_client, top_k=candidate_pool) for q in queries
    ]
    all_results = await asyncio.gather(*vector_coros, *bm25_coros)
    result_lists = list(all_results)  # vector lists first, then BM25 lists

    if not any(result_lists):
        return []

    fused = fuse_rankings(*result_lists)

    # Rerank with Cohere if enabled and API key is configured
    if use_rerank:
        # Rerank a larger pool (e.g., 20) and then slice to top_k
        rerank_pool = min(candidate_pool, len(fused))
        reranked = rerank_with_cohere(query, fused[:rerank_pool], top_k=top_k)
        results = reranked[:top_k] if reranked else fused[:top_k]
    else:
        results = fused[:top_k]

    # Cache for 1 hour; guard against non-serialisable floats (NaN/inf)
    await redis_client.setex(
        cache_key, 3600,
        json.dumps(results, default=lambda x: str(x)),
    )
    return results
