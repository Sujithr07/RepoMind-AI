"""HyDE (Hypothetical Document Embeddings) — RETIRED. Nothing imports this module.

HyDE asks an LLM for a hypothetical code snippet that *would* answer the
question, then embeds that snippet instead of the raw question, on the theory
that a fake code snippet sits closer to real code in embedding space than an
English question does.

It was compared head-to-head against Multi-Query RAG Fusion in a one-off
ablation (see the "Query-Expansion Ablation" table in the README). HyDE did not
beat RAG Fusion, so it has been retired: the live retrieval path uses RAG Fusion
only, and nothing in the running app references this code. The implementation is
preserved below, commented out, purely as a record of the approach.

Unlike Fusion, HyDE only reshaped the *dense* query — the sparse/BM25 leg kept
the original question, since a hallucinated snippet's identifiers are noisy
keyword signal.

To revive it (e.g. to re-run the ablation): uncomment below, re-add
``from app.query.hyde import generate_hyde`` plus a ``strategy == "hyde"`` branch
in app/query/retriever.retrieve, and add "hyde" back to STRATEGIES in
eval/run_ablation.py.
"""

# import asyncio
# import os
# from dotenv import load_dotenv
# from groq import Groq
#
# load_dotenv()
#
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# client = Groq(api_key=GROQ_API_KEY)
#
# SYSTEM_PROMPT = (
#     "You are a code generation assistant. Given a natural language question "
#     "about a codebase, generate a short hypothetical code snippet (Python, "
#     "JavaScript, or whatever language fits) that would plausibly be the answer "
#     "to the question. Output only the code, no explanation or fences."
# )
#
#
# async def generate_hyde(query: str) -> str:
#     """Return a hypothetical code snippet that would answer ``query``.
#
#     The caller embeds this snippet (not the raw question) for dense retrieval.
#     Degrades gracefully to the original question on any failure.
#     """
#
#     def _call() -> str:
#         # Groq's SDK is synchronous; run it off the event loop.
#         response = client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": query},
#             ],
#             temperature=0.2,
#             max_tokens=300,
#         )
#         return response.choices[0].message.content.strip()
#
#     try:
#         snippet = await asyncio.to_thread(_call)
#         return snippet or query
#     except Exception as e:
#         print(f"[hyde] expansion failed, falling back to raw query: {e}")
#         return query
