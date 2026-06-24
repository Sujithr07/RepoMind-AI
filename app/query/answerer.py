# app/query/answerer.py
from typing import AsyncIterator, Tuple
import json
import os
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
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

_ANSWER_KEY_RE = re.compile(r'"answer"\s*:\s*"')
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}


def _extract_answer_prefix(raw: str) -> str | None:
    """Best-effort decode of the (possibly partial) JSON string value of the
    "answer" key from a streaming JSON buffer.

    Returns the decoded text streamed so far, or None if the "answer" key has
    not started yet. Stops before any incomplete trailing escape sequence so we
    never emit a half-decoded character.
    """
    m = _ANSWER_KEY_RE.search(raw)
    if not m:
        return None

    i = m.end()
    n = len(raw)
    out: list[str] = []
    while i < n:
        c = raw[i]
        if c == "\\":
            if i + 1 >= n:
                break  # incomplete escape — wait for more input
            nxt = raw[i + 1]
            if nxt == "u":
                if i + 6 > n:
                    break  # incomplete \uXXXX
                try:
                    out.append(chr(int(raw[i + 2 : i + 6], 16)))
                except ValueError:
                    out.append(raw[i + 2 : i + 6])
                i += 6
                continue
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
            continue
        if c == '"':
            break  # unescaped quote terminates the answer value
        out.append(c)
        i += 1
    return "".join(out)


async def stream_answer(query: str, chunks: list[dict], history: list = None) -> AsyncIterator[Tuple[str, list[dict] | None]]:
    """Stream answer from Groq API with JSON response containing answer and citations.

    Yields:
        Tuple of (answer_text_delta, citations) where citations is None for every
        streamed text delta and a dict {"answer", "citations"} on the final yield.
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
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
            stream=True,
            response_format={"type": "json_object"},
        )

        accumulated = ""
        emitted = ""
        for chunk in stream:
            if token := chunk.choices[0].delta.content:
                accumulated += token
                prefix = _extract_answer_prefix(accumulated)
                if prefix is not None and len(prefix) > len(emitted):
                    yield (prefix[len(emitted):], None)
                    emitted = prefix

        # Parse the complete JSON from accumulated response
        try:
            parsed = json.loads(accumulated.strip())
            answer = parsed.get("answer", emitted)
            citations = parsed.get("citations", [])
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            # Fallback: stream whatever answer text we decoded, else the raw text
            answer = emitted or accumulated
            citations = []

        # Flush any answer text not yet streamed (e.g. trailing held-back escape)
        if len(answer) > len(emitted):
            yield (answer[len(emitted):], None)

        yield ("", {"answer": answer, "citations": citations})

    except Exception as e:
        print(f"Streaming failed: {e}")
        yield ("", {"answer": "", "citations": []})