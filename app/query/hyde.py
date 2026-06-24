"""HyDE (Hypothetical Document Embeddings) query expansion.

HyDE asks an LLM for a hypothetical code snippet that *would* answer the
question, then embeds that snippet instead of the raw question — code embeddings
are trained on code-to-code similarity, so a fake code snippet often sits closer
to the real implementation in vector space than an English question does.

In production this project uses Multi-Query RAG Fusion (see app/query/fusion.py)
as its default query-expansion strategy. HyDE is kept here as a selectable
alternative so the two can be compared head-to-head on the RAGAS eval set
(strategy="hyde" vs strategy="fusion" in app/query/retriever.retrieve; see
eval/run_ablation.py). Unlike Fusion, HyDE only reshapes the *dense* query — the
sparse/BM25 leg keeps using the original question, since a hallucinated snippet's
identifiers are noisy keyword signal.
"""
import asyncio
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a code generation assistant. Given a natural language question "
    "about a codebase, generate a short hypothetical code snippet (Python, "
    "JavaScript, or whatever language fits) that would plausibly be the answer "
    "to the question. Output only the code, no explanation or fences."
)


async def generate_hyde(query: str) -> str:
    """Return a hypothetical code snippet that would answer ``query``.

    The caller embeds this snippet (not the raw question) for dense retrieval.
    Degrades gracefully to the original question on any failure, so a flaky
    expansion never breaks search.
    """

    def _call() -> str:
        # Groq's SDK is synchronous; run it off the event loop.
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    try:
        snippet = await asyncio.to_thread(_call)
        return snippet or query
    except Exception as e:
        print(f"[hyde] expansion failed, falling back to raw query: {e}")
        return query
