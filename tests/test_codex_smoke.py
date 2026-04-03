import os
import shutil
import time

import pytest

from codefleet.models import WorkerStatus
from codefleet.supervisor import FleetSupervisor


pytestmark = pytest.mark.smoke


def _real_codex_smoke_enabled() -> bool:
    return os.environ.get("FLEET_RUN_REAL_CODEX_SMOKE") == "1"


@pytest.mark.skipif(
    not _real_codex_smoke_enabled(),
    reason="Set FLEET_RUN_REAL_CODEX_SMOKE=1 to run the real Codex smoke test.",
)
def test_real_codex_worker_smoke(tmp_path, git_repo):
    codex_path = shutil.which("codex")
    if codex_path is None:
        pytest.skip("codex CLI is not installed")

    base_dir = tmp_path / "fleet"
    supervisor = FleetSupervisor(
        base_dir=base_dir,
        allowed_repos=[str(git_repo)],
        default_model=os.environ.get("FLEET_REAL_CODEX_SMOKE_MODEL", "gpt-5.4-mini"),
        default_timeout=int(os.environ.get("FLEET_REAL_CODEX_SMOKE_TIMEOUT", "180")),
        max_concurrent=1,
    )

    payload = supervisor.create_worker(
        repo_path=str(git_repo),
        task_name="codex-smoke",
        prompt=(
            "Inspect the repository and produce a completed result. "
            "Do not modify repository files. "
            "Do not create commits. "
            "Do not run tests. "
            "Set files_changed, tests, commits, and next_steps to empty arrays."
        ),
        executor="codex",
        model=os.environ.get("FLEET_REAL_CODEX_SMOKE_MODEL", "gpt-5.4-mini"),
        timeout_seconds=int(os.environ.get("FLEET_REAL_CODEX_SMOKE_TIMEOUT", "180")),
    )

    try:
        deadline = payload.created_at + payload.timeout_seconds + 30
        while True:
            status = supervisor.get_worker_status(payload.worker_id)
            if status.status.is_terminal:
                break
            if deadline <= time.time():
                pytest.fail("real Codex smoke test did not reach a terminal state")
            time.sleep(1.0)

        result = supervisor.collect_worker_result(payload.worker_id, include_logs=True)

        assert status.status == WorkerStatus.SUCCEEDED, result.get("stderr_tail", "")
        assert result["result"] is not None
        assert result["result"]["status"] == "completed"
        assert result["result"]["files_changed"] == []
        assert result["result"]["tests"] == []
        assert result["result"]["commits"] == []
        assert result["result"]["next_steps"] == []
    finally:
        supervisor.close()
