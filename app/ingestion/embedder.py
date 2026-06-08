import asyncio
import itertools
import os
from typing import Awaitable, Callable, Optional
import cohere
from dotenv import load_dotenv
from app.ingestion.chunker import CodeChunk

load_dotenv()

EMBED_DIM = 1024
EMBED_MODEL = "embed-english-v3.0"

_cohere_client = cohere.AsyncClient(api_key=os.getenv("COHERE_API_KEY"))

BATCH_SIZE = 96  # Cohere's max per request
MAX_CONCURRENCY = 5  # cap in-flight requests to avoid hammering the API


async def embed_chunks(
    chunks: list[CodeChunk],
    progress_cb: Optional[Callable[[int, int], Awaitable[None]]] = None,
) -> list[list[float]]:
    """Embed code chunks via Cohere Embed API, batches sent concurrently.

    ``progress_cb`` (optional) is awaited after each batch completes with
    ``(done, total)`` chunk counts, so callers can stream embedding progress.
    """
    texts = [f"{c.context_prefix}\n\n{c.content}" for c in chunks]
    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    done = 0

    async def embed_batch(batch: list[str]) -> list[list[float]]:
        nonlocal done
        async with semaphore:
            response = await _cohere_client.embed(
                texts=batch,
                model=EMBED_MODEL,
                input_type="search_document",
                embedding_types=["float"],
            )
        done += len(batch)
        print(f"[embedder] embedded {done}/{len(texts)} chunks")
        if progress_cb is not None:
            try:
                await progress_cb(done, len(texts))
            except Exception as e:
                # Progress reporting is best-effort; never fail embedding over it.
                print(f"[embedder] progress callback failed: {e}")
        return response.embeddings.float_

    results = await asyncio.gather(*(embed_batch(batch) for batch in batches))
    return list(itertools.chain.from_iterable(results))
