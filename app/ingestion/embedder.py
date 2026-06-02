import os
import cohere
from dotenv import load_dotenv
from app.ingestion.chunker import CodeChunk

load_dotenv()

EMBED_DIM = 1024
EMBED_MODEL = "embed-english-v3.0"

_cohere_client = cohere.AsyncClient(api_key=os.getenv("COHERE_API_KEY"))

BATCH_SIZE = 96  # Cohere's max per request


async def embed_chunks(chunks: list[CodeChunk]) -> list[list[float]]:
    """Embed code chunks via Cohere Embed API in batches."""
    texts = [f"{c.context_prefix}\n\n{c.content}" for c in chunks]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = await _cohere_client.embed(
            texts=batch,
            model=EMBED_MODEL,
            input_type="search_document",
            embedding_types=["float"],
        )
        all_embeddings.extend(response.embeddings.float_)
        print(f"[embedder] embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks")

    return all_embeddings
