import os
import uuid
import asyncio
import asyncpg
from typing import TypedDict, Literal
from groq import Groq
from dotenv import load_dotenv

from app.query.retriever import retrieve

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


class QueryState(TypedDict):
    question: str
    repo_id: str
    chunks: list
    retry_count: int
    conn: asyncpg.Connection
    redis_client: object
    history: list
    _reflection_result: str


async def retrieve_node(state: QueryState) -> dict:
    """Retrieve chunks for the question.

    HyDE expansion happens inside `retrieve` (dense path only); BM25 uses the
    raw question. Passing the raw question here avoids embedding a HyDE-of-HyDE.
    """
    real_repo_id = uuid.UUID(state["repo_id"])
    result = await retrieve(
        state["question"],
        real_repo_id,
        state["conn"],
        state["redis_client"],
    )
    return {"chunks": result}


async def reflect_node(state: QueryState) -> dict:
    """Reflect on whether the retrieved chunks are relevant."""
    question = state["question"]
    chunks = state["chunks"]

    # Take first 2 chunks for reflection
    chunks_preview = chunks[:2] if chunks else []
    chunks_text = "\n\n".join([c.get("content", "") for c in chunks_preview])

    prompt = f"Given the question: {question} and these retrieved chunks: {chunks_text}, are the chunks relevant? Reply with only YES or NO."

    def _call() -> str:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return response.choices[0].message.content.strip().upper()

    try:
        # Groq's client is sync; run it off the event loop so it doesn't block.
        result = await asyncio.to_thread(_call)
    except Exception as e:
        print(f"Reflection failed: {e}")
        result = "YES"  # Default to YES on error

    return {"retry_count": state["retry_count"] + 1, "_reflection_result": result}


from langgraph.graph import StateGraph, END


def should_retry(state: QueryState) -> Literal["retrieve", "__end__"]:
    """Retry retrieval when chunks were judged irrelevant, else finish.

    The graph stops after producing chunks; answer generation is streamed by
    the caller so tokens can reach the client as they're generated.
    """
    reflection = state.get("_reflection_result", "YES")
    retry_count = state["retry_count"]

    if reflection == "NO" and retry_count < 2:
        return "retrieve"
    return END


builder = StateGraph(QueryState)

# Add nodes
builder.add_node("retrieve", retrieve_node)
builder.add_node("reflect", reflect_node)

# Set entry point
builder.set_entry_point("retrieve")

# Add edges
builder.add_edge("retrieve", "reflect")
builder.add_conditional_edges(
    "reflect",
    should_retry,
    {
        "retrieve": "retrieve",
        END: END,
    },
)

# Compile the graph
query_graph = builder.compile()
