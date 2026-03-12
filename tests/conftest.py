import subprocess
from pathlib import Path

import pytest

from codex_fleet_supervisor.store import WorkerStore
from codex_fleet_supervisor.supervisor import FleetSupervisor


@pytest.fixture
def db_path(tmp_path):
    """Provide a path for a test database."""
    return tmp_path / "test_fleet.db"


@pytest.fixture
def store(db_path):
    """Create a WorkerStore with a test database."""
    s = WorkerStore(db_path)
    yield s
    s.close()


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository with an initial commit."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    subprocess.run(
        ["git", "init", str(repo_path)], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(repo_path), "config", "user.email", "test@test.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_path), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
    )
    # Create an initial commit
    (repo_path / "README.md").write_text("# Test Repo")
    subprocess.run(
        ["git", "-C", str(repo_path), "add", "."],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_path), "commit", "-m", "Initial commit"],
        capture_output=True,
        check=True,
    )
    return repo_path


@pytest.fixture
def supervisor(tmp_path, git_repo):
    """Create a FleetSupervisor with test configuration."""
    base_dir = tmp_path / "fleet"
    sup = FleetSupervisor(
        base_dir=base_dir,
        allowed_repos=[str(git_repo)],
        default_model="gpt-5.4",
        default_timeout=60,
        max_concurrent=5,
    )
    yield sup
    sup.close()
