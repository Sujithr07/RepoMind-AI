import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a code generation assistant. Given a natural language question about a codebase, "
    "generate a short hypothetical Python/JavaScript code snippet that would be the answer to the question. "
    "Output only the code, no explanation."
)

async def generate_hyde(query:str) -> str:
    """Generate a hypothetical code chunk that would answer the query.
    Embed this instead of the raw question — code embeddings are
    trained on code-to-code similarity, not NL-to-code."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"HyDE expansion failed: {e}")
        return query