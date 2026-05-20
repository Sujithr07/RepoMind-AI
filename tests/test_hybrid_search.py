"""Unit tests for hybrid search (BM25 + RRF fusion).

These tests intentionally avoid hitting Qdrant / Voyage / Redis. They exercise
the pure ranking logic and the BM25 build/search round-trip against an
in-memory fake Redis client that mirrors the small async API surface we use.
"""
from __future__ import annotations

import asyncio

import pytest

from app.query.retriever import (
    _tokenize,
    build_bm25_index,
    fuse_rankings,
    search_bm25,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeAsyncRedis:
    """Tiny stand-in for redis.asyncio.Redis covering get/set/setex/aclose."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: str):
        self.store[key] = value

    async def setex(self, key: str, ttl: int, value: str):
        self.store[key] = value

    async def aclose(self):
        pass


def _chunk(file_path: str, content: str, start_line: int = 0, end_line: int = 5) -> dict:
    return {
        "content": content,
        "context_prefix": file_path,
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
    }


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def test_tokenize_lowercases_and_splits_on_punctuation():
    tokens = _tokenize("Async DEF fetch_user(id: int) -> User")
    assert tokens == ["async", "def", "fetch", "user", "id", "int", "user"]


def test_tokenize_handles_empty_and_whitespace():
    assert _tokenize("") == []
    assert _tokenize("   \n\t ") == []


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------
def test_fuse_rankings_promotes_chunks_in_both_lists():
    a = _chunk("a.py", "...", 0, 1)
    b = _chunk("b.py", "...", 0, 1)
    c = _chunk("c.py", "...", 0, 1)

    vector_results = [a, b, c]   # ranks 0,1,2
    bm25_results = [c, a, b]     # ranks 0,1,2

    fused = fuse_rankings(vector_results, bm25_results, k=60)

    # Every input chunk shows up exactly once.
    assert {f["file_path"] for f in fused} == {"a.py", "b.py", "c.py"}

    # `a` is rank 0 dense + rank 1 sparse  -> 1/61 + 1/62
    # `c` is rank 2 dense + rank 0 sparse  -> 1/63 + 1/61
    # `b` is rank 1 dense + rank 2 sparse  -> 1/62 + 1/63
    # so a > c > b.
    assert [f["file_path"] for f in fused] == ["a.py", "c.py", "b.py"]

    # Fused score is monotonically non-increasing.
    scores = [f["fused_score"] for f in fused]
    assert scores == sorted(scores, reverse=True)


def test_fuse_rankings_handles_disjoint_lists():
    a = _chunk("a.py", "...", 0, 1)
    b = _chunk("b.py", "...", 0, 1)

    fused = fuse_rankings([a], [b])
    assert {f["file_path"] for f in fused} == {"a.py", "b.py"}


def test_fuse_rankings_empty_inputs():
    assert fuse_rankings([], []) == []


def test_fuse_rankings_uses_constant_k():
    a = _chunk("a.py", "...", 0, 1)
    fused = fuse_rankings([a], [], k=60)
    # Only ranker, rank 0 -> 1/(60 + 0 + 1) == 1/61.
    assert fused[0]["fused_score"] == pytest.approx(1 / 61)


# ---------------------------------------------------------------------------
# BM25 round-trip via a fake async Redis
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_bm25_returns_empty_when_no_index():
    redis = FakeAsyncRedis()
    results = await search_bm25("anything", "missing-repo", redis, top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_search_bm25_finds_exact_keyword_matches():
    redis = FakeAsyncRedis()
    chunks = [
        _chunk("auth.py", "def login(user, password):\n    return True", 0, 2),
        _chunk("utils.py", "def add(a, b):\n    return a + b", 0, 2),
        _chunk("api.py", "async def fetch_user(id):\n    return await db.get(id)", 0, 2),
    ]
    await build_bm25_index(chunks, "repo-1", redis)

    # Hit on exact identifier name -> auth.py wins.
    results = await search_bm25("login", "repo-1", redis, top_k=3)
    assert results, "expected at least one BM25 hit for 'login'"
    assert results[0]["file_path"] == "auth.py"


@pytest.mark.asyncio
async def test_search_bm25_async_def_retrieves_async_function():
    """Spec example: the query 'async def' should rank async functions first."""
    redis = FakeAsyncRedis()
    chunks = [
        _chunk("sync.py", "def parse(data):\n    return data.strip()", 0, 2),
        _chunk("async_io.py", "async def fetch(url):\n    return await client.get(url)", 0, 2),
        _chunk("math_utils.py", "def add(a, b):\n    return a + b", 0, 2),
    ]
    await build_bm25_index(chunks, "repo-async", redis)

    results = await search_bm25("async def", "repo-async", redis, top_k=3)

    assert results, "expected BM25 to retrieve at least one chunk for 'async def'"
    assert results[0]["file_path"] == "async_io.py"
    assert "async" in results[0]["content"]


@pytest.mark.asyncio
async def test_build_bm25_index_persists_payload():
    redis = FakeAsyncRedis()
    chunks = [_chunk("a.py", "hello world", 0, 1)]
    await build_bm25_index(chunks, "repo-x", redis)
    assert "bm25_index:repo-x" in redis.store


# ---------------------------------------------------------------------------
# End-to-end integration of fuse + BM25 on the spec example
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_hybrid_fusion_recovers_keyword_only_match():
    """Vector retrieval misses the keyword-perfect chunk entirely; BM25 finds
    it. After RRF, the BM25-only hit must appear in the fused result set so
    hybrid recall is strictly better than vector-only.
    """
    redis = FakeAsyncRedis()

    fetch = _chunk("io.py", "async def fetch(url):\n    return await client.get(url)", 0, 2)
    parse_user = _chunk("user.py", "def parse_user(data):\n    return User(**data)", 0, 2)
    add = _chunk("math.py", "def add(a, b):\n    return a + b", 0, 2)

    await build_bm25_index([fetch, parse_user, add], "repo-h", redis)

    # Vector ranker returns chunks unrelated to the keyword query.
    vector_results = [parse_user, add]
    bm25_results = await search_bm25("async def fetch", "repo-h", redis, top_k=3)

    # BM25 must rank the exact match first.
    assert bm25_results[0]["file_path"] == "io.py"

    fused_paths = [f["file_path"] for f in fuse_rankings(vector_results, bm25_results)]
    vector_only_paths = [c["file_path"] for c in vector_results]

    # Hybrid recovers the keyword-only chunk that vector-only retrieval missed.
    assert "io.py" in fused_paths
    assert "io.py" not in vector_only_paths
