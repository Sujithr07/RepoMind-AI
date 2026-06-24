"""Query-expansion ablation: single-query vs HyDE vs RAG Fusion.

Runs the live RAG pipeline over the curated test set (eval/run_ragas.py's
TEST_CASES) three times — once per retrieval ``strategy`` — and scores each with
RAGAS Faithfulness and (LLM) Context Recall, using Groq llama-3.3-70b as the
judge. Both metrics are LLM-only, so no OpenAI embedding key is required.

The point is a *relative* comparison on the same indexed repo: which query
expansion retrieves context that best supports the ground-truth answer
(context recall) and best grounds the generated answer (faithfulness).

Usage:
    python eval/run_ablation.py [REPO_ID]

REPO_ID must be a 'ready' repo whose code the TEST_CASES describe (RepoMind
itself). Defaults to the env var ABLATION_REPO_ID, else the first ready repo.
Writes eval/ablation_results.json and prints a Markdown table.
"""
import asyncio
import importlib.util
import json
import os
import sys
import uuid

from dotenv import load_dotenv

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
load_dotenv(os.path.join(_ROOT, ".env"))
os.environ.setdefault(
    "DATABASE_URL", "postgresql://postgres:pass0407@localhost:5432/repomind"
)

import asyncpg
import redis.asyncio as aioredis

from app.query.retriever import retrieve, invalidate_query_cache
from app.query.answerer import stream_answer

# Load TEST_CASES from run_ragas.py by path (the 'eval' dir isn't a package).
_spec = importlib.util.spec_from_file_location(
    "_run_ragas", os.path.join(os.path.dirname(__file__), "run_ragas.py")
)
_run_ragas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_ragas)
TEST_CASES = _run_ragas.TEST_CASES

STRATEGIES = ["single", "hyde", "fusion"]


def _dsn() -> str:
    raw = os.getenv("DATABASE_URL", "")
    url = raw.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgres://", "postgresql://"
    )
    return url.split("?")[0]


def _use_ssl(dsn: str) -> bool:
    return not ("localhost" in dsn or "127.0.0.1" in dsn)


async def _resolve_repo_id(conn: asyncpg.Connection) -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    env_id = os.getenv("ABLATION_REPO_ID")
    if env_id:
        return env_id
    row = await conn.fetchrow("SELECT id FROM repos WHERE status='ready' LIMIT 1")
    if not row:
        raise SystemExit("No 'ready' repo found — index the RepoMind repo first.")
    return str(row["id"])


async def _answer(question: str, chunks: list) -> str:
    answer = ""
    async for delta, cit in stream_answer(question, chunks, history=None):
        if cit is not None:
            answer = cit.get("answer") or answer
        elif delta:
            answer += delta
    return answer


async def _collect_all() -> tuple[str, dict]:
    """Run retrieval+generation for every strategy x test case in one event loop.

    Returns (repo_id, {strategy: [ {question, answer, contexts, ground_truth} ]}).
    A single loop is reused on purpose: the async Cohere client binds its pool to
    the loop on first use, so spawning a fresh loop per strategy would orphan it.
    """
    dsn = _dsn()
    conn = await asyncpg.connect(dsn=dsn, ssl="require" if _use_ssl(dsn) else False)
    redis_client = aioredis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
    )
    out: dict[str, list] = {}
    try:
        repo_id = await _resolve_repo_id(conn)
        print(f"[ablation] target repo_id={repo_id}")
        # Clean slate so cached results from a prior run can't leak in.
        await invalidate_query_cache(repo_id, redis_client)

        for strategy in STRATEGIES:
            print(f"\n[ablation] === strategy: {strategy} ===")
            samples = []
            for i, tc in enumerate(TEST_CASES, 1):
                q = tc["question"]
                # Reranking is disabled on purpose: it re-scores candidates
                # against the *original* query and collapses the final top-k to
                # (nearly) the same chunks regardless of how they were retrieved,
                # which masks the very variable under study. Turning it off
                # isolates the effect of the query-expansion strategy on
                # retrieval. (In production the Cohere reranker sits on top.)
                chunks = await retrieve(
                    q, uuid.UUID(repo_id), conn, redis_client,
                    top_k=5, strategy=strategy, use_rerank=False,
                )
                contexts = [c["content"] for c in chunks]
                answer = await _answer(q, chunks)
                samples.append({
                    "question": q,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": tc["ground_truth"],
                })
                print(f"  [{i:2}/{len(TEST_CASES)}] {q[:48]:48}  "
                      f"ctx={len(contexts)} ans={len(answer)}c")
            out[strategy] = samples
    finally:
        await conn.close()
        await redis_client.aclose()
    return repo_id, out


def _score(samples: list, llm) -> dict:
    """Score one strategy's samples with RAGAS Faithfulness + Context Recall."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import Faithfulness, LLMContextRecall
    from ragas.run_config import RunConfig

    ds = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s["ground_truth"],
        )
        for s in samples
    ])
    metrics = [Faithfulness(), LLMContextRecall()]
    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=llm,
        run_config=RunConfig(max_workers=2, max_retries=5, timeout=180),
        show_progress=True,
    )
    df = result.to_pandas()

    def col_mean(*cands):
        for c in cands:
            if c in df.columns:
                return float(df[c].mean(skipna=True)), int(df[c].notna().sum())
        return float("nan"), 0

    faith, faith_n = col_mean("faithfulness")
    recall, recall_n = col_mean("context_recall", "llm_context_recall")
    return {
        "faithfulness": faith,
        "faithfulness_scored": faith_n,
        "context_recall": recall,
        "context_recall_scored": recall_n,
        "n": len(samples),
    }


def main():
    # Phase 1 (async): collect answers + contexts in ONE event loop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        repo_id, data = loop.run_until_complete(_collect_all())
    finally:
        loop.close()

    # Phase 2 (sync): RAGAS evaluate() outside any running loop to avoid nesting.
    from ragas.llms import LangchainLLMWrapper
    from langchain_groq import ChatGroq

    judge = LangchainLLMWrapper(ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
    ))

    scores = {}
    for strategy in STRATEGIES:
        print(f"\n[ablation] scoring strategy: {strategy} ...")
        try:
            scores[strategy] = _score(data[strategy], judge)
        except Exception as e:
            print(f"[ablation] scoring failed for {strategy}: {e}")
            scores[strategy] = {"faithfulness": float("nan"),
                                "context_recall": float("nan"), "error": str(e)}

    label = {"single": "Single query (baseline)", "hyde": "HyDE",
             "fusion": "RAG Fusion"}
    print("\n" + "=" * 64)
    print("ABLATION RESULTS  (repo_id=%s)" % repo_id)
    print("=" * 64)
    header = f"| {'Query strategy':24} | {'Context Recall':14} | {'Faithfulness':12} |"
    print(header)
    print(f"|{'-'*26}|{'-'*16}|{'-'*14}|")
    for s in STRATEGIES:
        sc = scores[s]
        cr = sc.get("context_recall", float("nan"))
        fa = sc.get("faithfulness", float("nan"))
        print(f"| {label[s]:24} | {cr:14.3f} | {fa:12.3f} |")
    print("=" * 64)

    out_path = os.path.join(os.path.dirname(__file__), "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump({"repo_id": repo_id, "scores": scores,
                   "test_set": [t["question"] for t in TEST_CASES]}, f, indent=2)
    print(f"\n[ablation] saved {out_path}")


if __name__ == "__main__":
    main()
