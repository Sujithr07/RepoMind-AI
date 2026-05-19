import os
import json
import uuid
import asyncpg
import asyncio
import redis.asyncio as aioredis
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.query.graph import query_graph
from app.workers.tasks import index_repo_task
from app.utils.tracing import get_langfuse
from app.utils.memory import get_history, save_history

load_dotenv()

# Connection pool
db_pool: asyncpg.Pool = None
redis_pool: aioredis.Redis = None

def get_dsn():
    raw_url = os.getenv("DATABASE_URL", "")
    db_url = raw_url.replace("postgresql+asyncpg://", "postgresql://").replace("postgres://", "postgresql://")
    return db_url.split("?")[0]

def should_use_ssl(dsn: str) -> bool:
    host = urlparse(dsn).hostname or ""
    return host not in {"localhost", "127.0.0.1"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_pool
    
    dsn = get_dsn()
    db_pool = await asyncpg.create_pool(dsn, ssl="require" if should_use_ssl(dsn) else False)
    
    redis_pool = aioredis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True
    )
    yield 
    await db_pool.close()
    await redis_pool.aclose()
    
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Serve the HTML UI."""
    html_path = Path(__file__).parent.parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return {"error": "index.html not found"}

# Request/response models
class AddRequest(BaseModel):
    github_url: str
    
class QueryRequest(BaseModel):
    repo_id: str
    question: str
    session_id: str | None = None

# Endpoints
@app.post("/repos")
async def add_repo(body: AddRequest):
    """Enqueue indexing job. Returns immediately - indexing runs in background.""" 
    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id, status FROM repos WHERE github_url = $1", body.github_url
        )
        if existing:
            return {"repo_id": str(existing["id"]), "status": existing["status"]}
        
        repo_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO repos (id, github_url, status)
            VALUES ($1, $2, 'pending')
            """,
            repo_id, body.github_url
        )
        
    index_repo_task.delay(str(repo_id), body.github_url)
    return {"repo_id": str(repo_id), "status": "pending"}

@app.get("/repos/{repo_id}/status")
async def repo_status(repo_id: str):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, indexed_at FROM repos WHERE id = $1::uuid", repo_id
        )        
    if not row:
        raise HTTPException(status_code=404, detail="Repo not found")
    return {
        "status": row["status"],
        "indexed_at": str(row["indexed_at"]) if row["indexed_at"] else None
    }

@app.post("/query") # Added slash
async def query_repo(body: QueryRequest):
    """Streaming SSE endpoint - token stream back as they're generated. """
    
    # Verify repo is ready
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM repos WHERE id = $1::uuid", body.repo_id # Fixed table name to repos
        )
    if not row:
        raise HTTPException(status_code=404, detail="Repo not found")
    if row["status"] != "ready":
        raise HTTPException(status_code=400, detail=f"Repo status is '{row['status']}' — wait for indexing to finish")
    
    # Get or create session_id
    session_id = body.session_id or str(uuid.uuid4())
    
    # Get conversation history
    history = get_history(session_id)
    
    async def token_stream():
        langfuse = None
        trace = None
        
        try:
            langfuse = get_langfuse()
            trace = langfuse.start_observation(
                name="rag-query",
                as_type="span",
                input={"question": body.question, "repo_id": body.repo_id},
            )
        except Exception as e:
            print(f"Langfuse initialization failed: {e}")
        
        try:
            async with db_pool.acquire() as conn:
                # Initialize state for LangGraph
                initial_state = {
                    "question": body.question,
                    "repo_id": body.repo_id,
                    "hyde_snippet": "",
                    "chunks": [],
                    "answer": "",
                    "retry_count": 0,
                    "conn": conn,
                    "redis_client": redis_pool,
                    "history": history
                }
                
                # Invoke the LangGraph
                result = await query_graph.ainvoke(initial_state)

                # Stream the answer
                final_answer = result.get("answer", "")

                # Include session_id in first SSE event
                yield f"data: {json.dumps({'session_id': session_id})}\n\n"

                for token in final_answer:
                    yield f"data: {token}\n\n"

                # Emit the source chunks used so the client can show citations
                sources = [
                    {
                        "file_path": c.get("file_path"),
                        "start_line": c.get("start_line"),
                        "end_line": c.get("end_line"),
                        "context_prefix": c.get("context_prefix"),
                        "relevance_score": c.get("relevance_score"),
                    }
                    for c in result.get("chunks", [])
                ]
                yield f"data: {json.dumps({'sources': sources})}\n\n"
                yield "data: [DONE]\n\n"
                
                # Save updated history
                updated_history = history + [
                    {"role": "user", "content": body.question},
                    {"role": "assistant", "content": final_answer}
                ]
                save_history(session_id, updated_history)
                
                if trace:
                    trace.update(output={"answer": final_answer})
        except Exception as e:
            print(f"Error in query pipeline: {e}")
            raise
        finally:
            try:
                if trace:
                    trace.end()
                if langfuse:
                    langfuse.flush()
            except Exception as e:
                print(f"Langfuse flush failed: {e}")

    return StreamingResponse(token_stream(), media_type="text/event-stream")

@app.post("/webhooks/github")
async def github_webhook(payload: dict):
    """
    Re-index on push to default branch.
    """
    default_branch = payload.get("repository", {}).get("default_branch", "main")
    if payload.get("ref") == f"refs/heads/{default_branch}":
        github_url = payload["repository"]["clone_url"]
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM repos WHERE github_url = $1", github_url
            )
            if row:
                index_repo_task.delay(str(row["id"]), github_url)
    return {"ok": True}