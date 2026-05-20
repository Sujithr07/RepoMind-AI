# app/query/answerer.py
from typing import AsyncIterator, Tuple
import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a senior engineer answering questions about a codebase. "
    "Answer using ONLY the provided code context. "
    "If the answer is not in the context, say so clearly — do not guess.\n\n"
    "Return a JSON object with this exact structure:\n"
    "{\n"
    '  "answer": "Your detailed answer here",\n'
    '  "citations": [\n'
    '    {\n'
    '      "file": "filename.py",\n'
    '      "line": 42,\n'
    '      "snippet": "the exact line or code snippet from the context"\n'
    '    }\n'
    '  ]\n'
    "}\n"
    "Always include citations. Reference the line numbers from the provided context."
)

async def stream_answer(query: str, chunks: list[dict], history: list = None) -> AsyncIterator[Tuple[str, list[dict] | None]]:
    """Stream answer from Groq API with JSON response containing answer and citations.

    Yields:
        Tuple of (answer_chunk, citations) where citations is None until the final chunk.
    """

    context = "\n\n---\n\n".join(
        f"File: {c['file_path']} (line {c['start_line']})\n{c['content']}"
        for c in chunks
    )

    full_prompt = (
        f"Context chunks from the repository:\n{context}\n\n"
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
            response_format={"type": "json_object"},
        )

        accumulated = ""
        for chunk in stream:
            if token := chunk.choices[0].delta.content:
                accumulated += token
                yield (token, None)

        # Parse the complete JSON from accumulated response
        try:
            parsed = json.loads(accumulated.strip())
            answer = parsed.get("answer", "")
            citations = parsed.get("citations", [])
            # Yield final parsed data
            yield ("", {"answer": answer, "citations": citations})
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            # Fallback: treat accumulated as plain answer
            yield ("", {"answer": accumulated, "citations": []})

    except Exception as e:
        print(f"Streaming failed: {e}")
        yield ("", {"answer": "", "citations": []})