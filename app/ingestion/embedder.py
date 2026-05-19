import os
import asyncio
import voyageai
from app.ingestion.chunker import CodeChunk
from dotenv import load_dotenv

load_dotenv()

# Use the same client initialization pattern as the retriever.
# This prevents embedding calls from failing due to missing/implicit credentials.
client = voyageai.AsyncClient(api_key=os.getenv("VOYAGE_API_KEY"))

# Free-tier Voyage limits are 3 RPM / 10K TPM. Keep each batch well under the
# token cap and pace requests so indexing survives without a paid plan.
MAX_BATCH_TOKENS = 8000
SECONDS_BETWEEN_REQUESTS = 21


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — good enough for batch sizing."""
    return max(1, len(text) // 4)


def _make_batches(texts: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = _estimate_tokens(text)
        if current and (current_tokens + tokens > MAX_BATCH_TOKENS or len(current) >= 128):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


async def embed_chunks(chunks: list[CodeChunk]) -> list[list[float]]:
    """Embed code chunks with Voyage AI, token-aware batching + rate limiting."""
    texts = [f"{c.context_prefix}\n\n{c.content}" for c in chunks]
    batches = _make_batches(texts)
    all_embeddings: list[list[float]] = []

    for i, batch in enumerate(batches):
        if i > 0:
            await asyncio.sleep(SECONDS_BETWEEN_REQUESTS)
        result = await client.embed(
            batch,
            model="voyage-code-3",
            input_type="document",  ## "document" for indexing, "query" for quering
        )
        all_embeddings.extend(result.embeddings)

    return all_embeddings
