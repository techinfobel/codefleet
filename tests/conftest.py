import subprocess
import sys
import time
from pathlib import Path

import pytest

from codefleet.models import WorkerRecord, WorkerStatus
from codefleet.store import WorkerStore
from codefleet.supervisor import FleetSupervisor


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
        default_model="gpt-5.5",
        default_timeout=60,
        max_concurrent=5,
    )
    yield sup
    sup.close()


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def fake_build_success(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
    """Return a command that writes a valid result.json and exits 0.

    Declares one file changed so the supervisor's silent-failure check
    treats this as a legitimate completion.
    """
    script = (
        "import json; "
        "json.dump("
        '{"summary":"done","status":"completed",'
        '"files_changed":["fake.py"],"commits":[],"next_steps":[]}, '
        f"open('{result_json_path}', 'w'))"
    )
    return [sys.executable, "-c", script]


def fake_build_fail(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
    """Return a command that exits non-zero."""
    return [sys.executable, "-c", "import sys; sys.exit(1)"]


def fake_build_sleep(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
    """Return a command that sleeps for 300 seconds."""
    return [sys.executable, "-c", "import time; time.sleep(300)"]


def make_worker_record(worker_id="w_test001", **overrides):
    """Create a WorkerRecord with sensible defaults for tests."""
    defaults = dict(
        worker_id=worker_id,
        task_name="test task",
        repo_path="/tmp/repo",
        branch_name=f"codex/test/{worker_id}",
        worktree_path=f"/tmp/fleet/workers/{worker_id}/worktree",
        worker_dir=f"/tmp/fleet/workers/{worker_id}",
        model="gpt-5.5",
        status=WorkerStatus.PENDING,
        created_at=time.time(),
        timeout_seconds=600,
        command_json='["codex", "exec", "test"]',
        prompt="Do something useful",
        result_json_path=f"/tmp/fleet/workers/{worker_id}/result.json",
        stdout_path=f"/tmp/fleet/workers/{worker_id}/stdout.log",
        stderr_path=f"/tmp/fleet/workers/{worker_id}/stderr.log",
        prompt_path=f"/tmp/fleet/workers/{worker_id}/prompt.txt",
        meta_path=f"/tmp/fleet/workers/{worker_id}/meta.json",
    )
    defaults.update(overrides)
    return WorkerRecord(**defaults)


def wait_for_terminal(supervisor, worker_id, timeout=15):
    """Poll until a worker reaches a terminal status."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = supervisor.get_worker_status(worker_id)
        if status.status.is_terminal:
            return status
        time.sleep(0.2)
    return supervisor.get_worker_status(worker_id)


def wait_workflow_terminal(supervisor, workflow_id, timeout=30):
    """Poll until a workflow reaches a terminal status."""
    from codefleet.models import WorkflowStatus

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        wf = supervisor.get_workflow_status(workflow_id)
        if wf.status in {WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}:
            return wf
        time.sleep(0.2)
    return supervisor.get_workflow_status(workflow_id)
