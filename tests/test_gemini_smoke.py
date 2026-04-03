import os
import shutil
import time

import pytest

from codefleet.models import WorkerStatus
from codefleet.supervisor import FleetSupervisor


pytestmark = pytest.mark.smoke


def _real_gemini_smoke_enabled() -> bool:
    return os.environ.get("FLEET_RUN_REAL_GEMINI_SMOKE") == "1"


@pytest.mark.skipif(
    not _real_gemini_smoke_enabled(),
    reason="Set FLEET_RUN_REAL_GEMINI_SMOKE=1 to run the real Gemini smoke test.",
)
def test_real_gemini_worker_smoke(tmp_path, git_repo):
    gemini_path = shutil.which("gemini")
    if gemini_path is None:
        pytest.skip("gemini CLI is not installed")

    model = os.environ.get("FLEET_REAL_GEMINI_SMOKE_MODEL", "gemini-2.5-pro")
    timeout_seconds = int(os.environ.get("FLEET_REAL_GEMINI_SMOKE_TIMEOUT", "180"))

    base_dir = tmp_path / "fleet"
    supervisor = FleetSupervisor(
        base_dir=base_dir,
        allowed_repos=[str(git_repo)],
        default_gemini_model=model,
        default_timeout=timeout_seconds,
        max_concurrent=1,
    )

    payload = supervisor.create_worker(
        repo_path=str(git_repo),
        task_name="gemini-smoke",
        prompt=(
            "Inspect the repository and produce a completed result. "
            "Do not modify repository files. "
            "Do not create commits. "
            "Do not run tests. "
            "Set files_changed, tests, commits, and next_steps to empty arrays."
        ),
        executor="gemini",
        model=model,
        timeout_seconds=timeout_seconds,
    )

    try:
        deadline = payload.created_at + payload.timeout_seconds + 30
        while True:
            status = supervisor.get_worker_status(payload.worker_id)
            if status.status.is_terminal:
                break
            if deadline <= time.time():
                pytest.fail("real Gemini smoke test did not reach a terminal state")
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
