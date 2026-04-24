import os
import shutil
import time

import pytest

from codefleet.models import WorkerStatus
from codefleet.supervisor import FleetSupervisor


pytestmark = pytest.mark.smoke


def _real_claude_smoke_enabled() -> bool:
    return os.environ.get("FLEET_RUN_REAL_CLAUDE_SMOKE") == "1"


@pytest.mark.skipif(
    not _real_claude_smoke_enabled(),
    reason="Set FLEET_RUN_REAL_CLAUDE_SMOKE=1 to run the real Claude smoke test.",
)
def test_real_claude_worker_smoke(tmp_path, git_repo):
    claude_path = shutil.which("claude")
    if claude_path is None:
        pytest.skip("claude CLI is not installed")

    model = os.environ.get("FLEET_REAL_CLAUDE_SMOKE_MODEL", "claude-sonnet-4-6")
    timeout_seconds = int(os.environ.get("FLEET_REAL_CLAUDE_SMOKE_TIMEOUT", "180"))

    base_dir = tmp_path / "fleet"
    supervisor = FleetSupervisor(
        base_dir=base_dir,
        allowed_repos=[str(git_repo)],
        default_claude_model=model,
        default_timeout=timeout_seconds,
        max_concurrent=1,
        heartbeat_interval=5.0,
    )

    payload = supervisor.create_worker(
        repo_path=str(git_repo),
        task_name="claude-smoke",
        prompt=(
            "Inspect the repository and produce a completed result. "
            "Do not modify repository files. "
            "Do not create commits. "
            "Do not run tests. "
            "Set files_changed, tests, commits, and next_steps to empty arrays."
        ),
        executor="claude",
        model=model,
        reasoning_effort="high",
        timeout_seconds=timeout_seconds,
    )

    try:
        deadline = payload.created_at + payload.timeout_seconds + 30
        saw_heartbeat = False
        while True:
            status = supervisor.get_worker_status(payload.worker_id)
            if status.last_heartbeat_at is not None and status.heartbeat_message:
                saw_heartbeat = True
            if status.status.is_terminal:
                break
            if deadline <= time.time():
                pytest.fail("real Claude smoke test did not reach a terminal state")
            time.sleep(1.0)

        result = supervisor.collect_worker_result(payload.worker_id, include_logs=True)

        assert saw_heartbeat
        assert status.last_heartbeat_at is not None
        assert status.heartbeat_message is not None
        assert status.status == WorkerStatus.SUCCEEDED, result.get("stderr_tail", "")
        assert result["result"] is not None
        assert result["result"]["status"] in ("completed", "completed_no_changes")
        assert result["result"]["files_changed"] == []
        assert result["result"]["tests"] == []
        assert result["result"]["commits"] == []
        assert result["result"]["next_steps"] == []
    finally:
        supervisor.close()
