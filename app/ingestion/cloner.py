import os
import tempfile
from pathlib import Path
from typing import Iterator
import git
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".rb": "ruby",
    ".cs": "csharp",
    ".php": "php",
    ".scala": "scala",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".pytest_cache",
    "migrations",
}


async def clone_repo(github_url: str) -> Path:
    """Clone a GitHub repo to a temp directory. Return the path."""
    tmp_dir = tempfile.mkdtemp(prefix="codeqa_")
    print(f"[cloner] cloning {github_url} -> {tmp_dir}")

    # Inject token if set — handles private repos
    token = os.getenv("GITHUB_TOKEN")
    if token and "github.com" in github_url:
        github_url = github_url.replace("https://", f"https://{token}@")

    repo = git.Repo.clone_from(
        github_url,
        tmp_dir,
        depth=1,           # Shallow clone — only latest commit
        single_branch=True,
    )
    # Release GitPython's file handles so the temp dir can be removed later;
    # otherwise Windows holds a lock on .git pack files and cleanup fails.
    repo.close()
    print("[cloner] clone complete")
    return Path(tmp_dir)


def walk_code_files(repo_path: Path) -> Iterator[tuple[str, str, str]]:
    """
    Walk the repo directory tree and yield (relative_path, source_code, language)
    for every supported file, skipping irrelevant directories.
    """
    for root, dirs, files in os.walk(repo_path):
        # Prune skip dirs in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in files:
            file_path = Path(root) / filename
            ext = file_path.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                continue

            # Fix: correct attribute is st_size, not st.size
            # Fix: this block was dedented out of the for-loop — now correctly indented
            if file_path.stat().st_size > 500_000:
                print(f"[cloner] skipping large file: {file_path.name}")
                continue

            try:
                source = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"[cloner] could not read {file_path}: {e}")
                continue

            # Relative path from repo root — used as context prefix
            relative_path = str(file_path.relative_to(repo_path))
            language = SUPPORTED_EXTENSIONS[ext]

            yield relative_path, source, language


def cleanup_repo(repo_path: Path) -> None:
    """Remove the temp directory created by clone_repo."""
    import shutil
    import stat

    def _on_rm_error(func, path, _exc):
        # Windows marks .git pack files read-only; clear the bit and retry.
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    try:
        shutil.rmtree(repo_path, onerror=_on_rm_error)
        print(f"[cloner] cleaned up {repo_path}")
    except Exception as e:
        print(f"[cloner] could not clean up {repo_path}: {e}")