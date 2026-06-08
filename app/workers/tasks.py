import asyncio
import os
import subprocess
import asyncpg
from celery import Celery
from dotenv import load_dotenv

from app.ingestion.cloner import clone_repo, walk_code_files, cleanup_repo
from app.ingestion.chunker import chunk_file
from app.ingestion.embedder import embed_chunks
from app.ingestion.indexer import upsert_chunks, delete_chunks_for_files

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

def get_dsn():
    raw_url = os.getenv("DATABASE_URL", "")
    db_url = raw_url.replace("postgresql+asyncpg://", "postgresql://").replace("postgres://", "postgresql://")
    return db_url.split("?")[0]

@celery.task(bind=True, max_retries=3, default_retry_delay=10)
def index_repo_task(self, repo_id: str, github_url: str):
    try:
        asyncio.run(_index_repo(repo_id, github_url))
    except Exception as exc:
        # Status is already set to 'error' in _index_repo; retry transient failures.
        raise self.retry(exc=exc)


async def _index_repo(repo_id: str, github_url: str):
    conn = None
    repo_path = None
    try:
        dsn = get_dsn()
        use_ssl = not ("localhost" in dsn or "127.0.0.1" in dsn)
        conn = await asyncpg.connect(dsn=dsn, ssl="require" if use_ssl else False)

        # Mark as indexing
        await conn.execute(
            "UPDATE repos SET status = 'indexing' WHERE id = $1::uuid", repo_id
        )

        repo_path = await clone_repo(github_url)

        new_sha = await asyncio.to_thread(_git, repo_path, "rev-parse", "HEAD")

        old_sha = await conn.fetchval(
            "SELECT last_commit_sha FROM repos WHERE id = $1::uuid", repo_id
        )

        # changed_files is None => full index; a set => delta index over those files.
        changed_files = None
        if old_sha:
            try:
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

        semaphore = asyncio.Semaphore(10)

        async def chunk_one(file_path, source, language):
            async with semaphore:
                # chunk_file is sync/CPU-bound; offload to a thread.
                return await asyncio.to_thread(chunk_file, file_path, source, language)

        per_file_chunks = await asyncio.gather(
            *(chunk_one(file_path, source, language) for file_path, source, language in files)
        )

        all_chunks = []
        for chunks in per_file_chunks:
            all_chunks.extend(chunks)

        print(f"[indexer] {len(all_chunks)} chunks from {github_url}")

        if changed_files is not None:
            # Delta: evict stale chunks for changed/removed files, then upsert the
            # rebuilt ones, merging into (not replacing) the existing BM25 corpus.
            await delete_chunks_for_files(conn, repo_id, changed_files)
            embeddings = await embed_chunks(all_chunks)
            await upsert_chunks(
                conn, repo_id, all_chunks, embeddings,
                replaced_file_paths=changed_files,
            )
        else:
            embeddings = await embed_chunks(all_chunks)
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
        print(f"[indexer] done - repo {repo_id} is ready")

    except Exception as e:
        print(f"[indexer] error indexing {github_url}: {e}")
        if conn:
            await conn.execute(
                "UPDATE repos SET status = 'error' WHERE id = $1::uuid", repo_id
            )
        raise e

    finally:
        if repo_path:
            cleanup_repo(repo_path)
        if conn:
            await conn.close()