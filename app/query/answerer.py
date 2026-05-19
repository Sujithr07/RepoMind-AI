# app/query/answerer.py
from typing import AsyncIterator
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a senior engineer answering questions about a codebase. "
    "Answer using ONLY the provided code context. "
    "Always cite the file path and function name when referencing code. "
    "If the answer is not in the context, say so clearly — do not guess."
)

async def stream_answer(query: str, chunks: list[dict], history: list = None) -> AsyncIterator[str]:
    """Stream token-by-token answer from Groq API."""

    context = "\n\n---\n\n".join(
        f"# {c['context_prefix']} (line {c['start_line']})\n"
        f"```python\n{c['content']}\n```"
        for c in chunks
    )

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Code context:\n{context}\n\n"
        f"Question: {query}"
    )

    # Build messages array with history prepended
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history if provided
    if history:
        messages.extend(history)
    
    # Add current query
    messages.append({"role": "user", "content": full_prompt})

    try:
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
            stream=True,
        )
        for chunk in stream:
            if token := chunk.choices[0].delta.content:
                yield token
    except Exception as e:
        print(f"Streaming failed: {e}")
        yield ""