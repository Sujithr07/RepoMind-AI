"""Multi-Query RAG Fusion: expand one user question into several diverse search
queries in a *single* LLM call, so we can fan out retrieval and fuse the results
without multiplying our LLM quota usage.

This replaces the single HyDE expansion (see app/query/hyde.py): instead of one
hypothetical snippet we generate a handful of reformulations that approach the
question from different angles. The expansion model is isolated here so it can
later be pointed at a separate/free provider (e.g. Gemini Flash, Cerebras) by
swapping this one client.
"""
import asyncio
import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)

# Number of reformulations to request. The original question is always added on
# top, so the effective query set is NUM_QUERIES + 1 (here: 3 queries total).
# Kept deliberately small: each query is embedded separately at search time, so
# more reformulations means more embedding-API calls — 3 total balances recall
# against free-tier embedding quota.
NUM_QUERIES = 2

SYSTEM_PROMPT = (
    "You are a search query generator for a code search engine. Given a user's "
    "question about a codebase, produce {n} semantically diverse reformulations "
    "that approach it from different angles — vary the phrasing, the likely "
    "function/class/identifier names, and the specific code constructs involved. "
    "Each reformulation must be a self-contained search query that could retrieve "
    "relevant code on its own. Do not number them or add commentary.\n"
    'Return ONLY a JSON object of this exact shape: {{"queries": ["...", "..."]}}'
)


async def generate_fusion_queries(query: str, n: int = NUM_QUERIES) -> list[str]:
    """Expand ``query`` into a diverse set of search queries via one Groq call.

    Returns the original question followed by the generated reformulations, with
    case-insensitive duplicates removed and original order preserved. On any
    failure it degrades gracefully to ``[query]`` (i.e. plain single-query
    retrieval), so a flaky expansion never breaks search.
    """

    def _call() -> str:
        # Groq's SDK is synchronous; run it off the event loop.
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
                {"role": "user", "content": query},
            ],
            temperature=0.4,  # a little heat for diverse phrasings
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    reformulations: list[str] = []
    try:
        raw = await asyncio.to_thread(_call)
        data = json.loads(raw)
        reformulations = [
            q.strip()
            for q in data.get("queries", [])
            if isinstance(q, str) and q.strip()
        ]
    except Exception as e:
        print(f"[fusion] query generation failed, falling back to single query: {e}")

    # Original question first (anchors BM25 keyword matching and is a safety net),
    # then reformulations; dedupe case-insensitively while preserving order.
    seen: set[str] = set()
    queries: list[str] = []
    for q in [query, *reformulations]:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            queries.append(q)
    return queries
