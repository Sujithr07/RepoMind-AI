import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.ingestion.chunker import CodeChunk

# Single thread executor keeps the model loaded in one process
_executor = ThreadPoolExecutor(max_workers=1)

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_DIM = 1024


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    print(f"[embedder] loading model {EMBED_MODEL} (first call only)")
    return SentenceTransformer(EMBED_MODEL)


def _encode_sync(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


async def embed_chunks(chunks: list[CodeChunk]) -> list[list[float]]:
    """Embed code chunks locally — no API keys, no rate limits."""
    texts = [f"{c.context_prefix}\n\n{c.content}" for c in chunks]
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _encode_sync, texts)
