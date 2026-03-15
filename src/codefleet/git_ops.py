import shutil
import subprocess
from pathlib import Path


class GitError(Exception):
    pass


def _run_git(repo_path: Path, *args: str, timeout: int = 10):
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def is_git_repo(path: Path) -> bool:
    """Check if path is inside a git repository."""
    try:
        result = _run_git(path, "rev-parse", "--is-inside-work-tree")
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def resolve_ref(repo_path: Path, ref: str = "HEAD") -> str:
    """Resolve a git ref to a commit hash."""
    result = _run_git(repo_path, "rev-parse", "--", ref)
    if result.returncode != 0:
        raise GitError(f"Failed to resolve ref '{ref}': {result.stderr.strip()}")
    return result.stdout.strip()


def create_worktree(
    repo_path: Path,
    worktree_path: Path,
    branch_name: str,
    base_ref: str = "HEAD",
) -> None:
    """Create a git worktree with a new branch."""
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    result = _run_git(
        repo_path,
        "worktree", "add",
        "-b", branch_name,
        str(worktree_path),
        base_ref,
        timeout=30,
    )
    if result.returncode != 0:
        raise GitError(f"Failed to create worktree: {result.stderr.strip()}")


def remove_worktree(repo_path: Path, worktree_path: Path) -> None:
    """Remove a git worktree."""
    result = _run_git(
        repo_path, "worktree", "remove", "--force", str(worktree_path), timeout=30
    )
    if result.returncode != 0:
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
        _run_git(repo_path, "worktree", "prune")


def delete_branch(repo_path: Path, branch_name: str) -> None:
    """Delete a git branch."""
    result = _run_git(repo_path, "branch", "-D", branch_name)
    if result.returncode != 0:
        raise GitError(
            f"Failed to delete branch '{branch_name}': {result.stderr.strip()}"
        )


def get_git_path() -> str | None:
    """Return the path to git, or None."""
    return shutil.which("git")


def get_repo_root(path: Path) -> Path | None:
    """Return the root of the git repo containing path."""
    result = _run_git(path, "rev-parse", "--show-toplevel")
    if result.returncode == 0:
        return Path(result.stdout.strip())
    return None
