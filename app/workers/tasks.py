import asyncio
import os
import subprocess
import asyncpg
import redis.asyncio as aioredis
from celery import Celery
from dotenv import load_dotenv

from app.ingestion.cloner import clone_repo, walk_code_files, cleanup_repo
from app.ingestion.chunker import chunk_file
from app.ingestion.embedder import embed_chunks
from app.ingestion.indexer import upsert_chunks, delete_chunks_for_files
from app.workers.progress import ProgressPublisher

load_dotenv()


def _git(repo_path, *args) -> str:
    """Run a git command inside repo_path and return stripped stdout.

    Sync/blocking — call via asyncio.to_thread from the event loop.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()

celery = Celery(
    "codeqa",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)

_DEFAULT_DSN = "postgresql://postgres:pass0407@localhost:5432/repomind"

def get_dsn():
    raw_url = os.getenv("DATABASE_URL", _DEFAULT_DSN)
    db_url = raw_url.replace("postgresql+asyncpg://", "postgresql://").replace("postgres://", "postgresql://")
    return db_url.split("?")[0]

# Reuse a single event loop for the lifetime of the worker process. The async
# API clients (Cohere, Qdrant) are module-level singletons that bind their
# connection pools to the loop on first use. asyncio.run() created a fresh loop
# per task and closed it on exit, orphaning those pools and raising
# "Event loop is closed" on every task after the first. --pool=solo runs tasks
# serially, so one shared loop is safe.
_worker_loop: asyncio.AbstractEventLoop | None = None


def _run_async(coro):
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop.run_until_complete(coro)


@celery.task(bind=True, max_retries=3, default_retry_delay=10)
def index_repo_task(self, repo_id: str, github_url: str):
    try:
        _run_async(_index_repo(repo_id, github_url))
    except Exception as exc:
        # Status is already set to 'error' in _index_repo; retry transient failures.
        raise self.retry(exc=exc)


async def _index_repo(repo_id: str, github_url: str):
    conn = None
    repo_path = None
    # Short-lived async Redis client for streaming progress. The worker runs its
    # own event loop and does not share FastAPI's connection pool.
    redis_client = aioredis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
    )
    progress = ProgressPublisher(repo_id, redis_client)
    try:
        dsn = get_dsn()
        use_ssl = not ("localhost" in dsn or "127.0.0.1" in dsn)
        conn = await asyncpg.connect(dsn=dsn, ssl="require" if use_ssl else False)

        # Mark as indexing
        await conn.execute(
            "UPDATE repos SET status = 'indexing' WHERE id = $1::uuid", repo_id
        )

        await progress.publish("cloning", "Cloning repository…")
        repo_path = await clone_repo(github_url)

        new_sha = await asyncio.to_thread(_git, repo_path, "rev-parse", "HEAD")

        old_sha = await conn.fetchval(
            "SELECT last_commit_sha FROM repos WHERE id = $1::uuid", repo_id
        )

        # changed_files is None => full index; a set => delta index over those files.
        changed_files = None
        if old_sha:
            try:
                await progress.publish("diffing", "Computing changed files…")
                # Shallow clone only has HEAD; pull just the old commit so diff has
                # both trees. diff compares trees directly and needs no shared history.
                await asyncio.to_thread(
                    _git, repo_path, "fetch", "--depth=1", "origin", old_sha
                )
                diff_out = await asyncio.to_thread(
                    _git, repo_path, "diff", "--name-only", old_sha, new_sha
                )
                changed_files = {
                    line.strip().replace("\\", "/")
                    for line in diff_out.splitlines()
                    if line.strip()
                }
                print(
                    f"[indexer] delta: {len(changed_files)} changed files "
                    f"since {old_sha[:7]} for repo {repo_id}"
                )
            except Exception as e:
                # Force-push / GC'd commit / fetch refused -> safe full reindex.
                print(f"[indexer] delta diff failed ({e}); falling back to full index")
                changed_files = None

        files = list(walk_code_files(repo_path))
        if changed_files is not None:
            files = [f for f in files if f[0].replace("\\", "/") in changed_files]

        total_files = len(files)
        await progress.publish(
            "chunking", f"Parsing {total_files} files…", current=0, total=total_files
        )

        semaphore = asyncio.Semaphore(10)
        parsed = 0
        # Throttle: avoid one pub/sub message per file on large repos.
        parse_step = max(1, total_files // 50)

        async def chunk_one(file_path, source, language):
            nonlocal parsed
            async with semaphore:
                # chunk_file is sync/CPU-bound; offload to a thread.
                result = await asyncio.to_thread(chunk_file, file_path, source, language)
            parsed += 1
            if parsed % parse_step == 0 or parsed == total_files:
                await progress.publish(
                    "chunking",
                    f"Parsed {parsed}/{total_files} files",
                    current=parsed,
                    total=total_files,
                )
            return result

        per_file_chunks = await asyncio.gather(
            *(chunk_one(file_path, source, language) for file_path, source, language in files)
        )

        all_chunks = []
        for chunks in per_file_chunks:
            all_chunks.extend(chunks)

        print(f"[indexer] {len(all_chunks)} chunks from {github_url}")

        total_chunks = len(all_chunks)

        async def embed_progress(done, total):
            await progress.publish(
                "embedding",
                f"Embedded {done}/{total} chunks",
                current=done,
                total=total,
            )

        await progress.publish(
            "embedding", f"Embedding {total_chunks} chunks…", current=0, total=total_chunks
        )

        if changed_files is not None:
            # Delta: evict stale chunks for changed/removed files, then upsert the
            # rebuilt ones, merging into (not replacing) the existing BM25 corpus.
            await delete_chunks_for_files(conn, repo_id, changed_files)
            embeddings = await embed_chunks(all_chunks, progress_cb=embed_progress)
            await progress.publish("upserting", "Storing vectors & search index…")
            await upsert_chunks(
                conn, repo_id, all_chunks, embeddings,
                replaced_file_paths=changed_files,
            )
        else:
            embeddings = await embed_chunks(all_chunks, progress_cb=embed_progress)
            await progress.publish("upserting", "Storing vectors & search index…")
            await upsert_chunks(conn, repo_id, all_chunks, embeddings)

        await conn.execute(
            """
            UPDATE repos
            SET status = 'ready', indexed_at = now(), last_commit_sha = $2
            WHERE id = $1::uuid
            """,
            repo_id,
            new_sha,
        )
        await progress.publish(
            "done", f"Indexed {total_chunks} chunks from {total_files} files"
        )
        print(f"[indexer] done - repo {repo_id} is ready")

    except Exception as e:
        print(f"[indexer] error indexing {github_url}: {e}")
        if conn:
            await conn.execute(
                "UPDATE repos SET status = 'error' WHERE id = $1::uuid", repo_id
            )
        await progress.publish("error", f"Indexing failed: {e}")
        raise e

    finally:
        if repo_path:
            cleanup_repo(repo_path)
        if conn:
            await conn.close()
        await redis_client.aclose()