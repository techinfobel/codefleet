"""Tests for supervisor.py - integration tests with real git, SQLite, and subprocesses."""

import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from codefleet.models import ExecutorType, WorkerRecord, WorkerStatus, WorkerStatusPayload
from codefleet.store import WorkerStore
from codefleet.supervisor import (
    FleetSupervisor,
    _sanitize_task_name,
    _generate_worker_id,
)


class TestSanitizeTaskName:
    def test_simple_name(self):
        assert _sanitize_task_name("add-login") == "add-login"

    def test_spaces_and_special(self):
        result = _sanitize_task_name("Add login feature!")
        assert " " not in result
        assert "!" not in result

    def test_long_name_truncated(self):
        long_name = "a" * 100
        result = _sanitize_task_name(long_name)
        assert len(result) <= 50

    def test_empty_name(self):
        assert _sanitize_task_name("") == "task"

    def test_only_special_chars(self):
        assert _sanitize_task_name("@#$%") == "task"

    def test_consecutive_dashes_collapsed(self):
        result = _sanitize_task_name("hello   world   foo")
        assert "--" not in result


class TestGenerateWorkerId:
    def test_format(self):
        wid = _generate_worker_id()
        assert wid.startswith("w_")
        assert len(wid) == 14  # w_ + 12 hex chars

    def test_uniqueness(self):
        ids = {_generate_worker_id() for _ in range(100)}
        assert len(ids) == 100


class TestHealthcheck:
    def test_healthcheck(self, supervisor):
        result = supervisor.healthcheck()
        assert result["app"] == "codefleet"
        assert result["git_found"] is True
        assert isinstance(result["codex_found"], bool)
        assert isinstance(result["gemini_found"], bool)
        assert isinstance(result["claude_found"], bool)
        assert result["default_model"] == "gpt-5.5"
        assert "db_path" in result
        assert "base_dir" in result
        assert "max_spawn_depth" in result
        assert result["supported_models"]["codex"] == ["gpt-5.5"]
        assert result["supported_models"]["gemini"] == ["gemini-3.1-pro-preview"]
        assert result["supported_models"]["claude"] == [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
        ]

    def test_healthcheck_includes_versions_and_auth_fields(self, supervisor):
        result = supervisor.healthcheck()
        assert "codex_version" in result
        assert "gemini_version" in result
        assert "claude_version" in result
        assert "codex_auth_status" in result
        assert "gemini_auth_status" in result
        assert "claude_auth_status" in result
        assert result["auth_check_mode"] == "local_artifact"
        assert "claude_auth_note" in result


class TestCreateWorkerValidation:
    def test_nonexistent_repo(self, supervisor):
        with pytest.raises(ValueError, match="does not exist"):
            supervisor.create_worker(
                repo_path="/tmp/nonexistent_repo_xyz",
                task_name="test",
                prompt="do something",
            )

    def test_not_a_git_repo(self, supervisor, tmp_path):
        plain_dir = tmp_path / "not_a_repo"
        plain_dir.mkdir()
        # Need to add to allowlist since supervisor has one
        supervisor.allowed_repos.append(plain_dir.resolve())
        with pytest.raises(ValueError, match="Not a git repository"):
            supervisor.create_worker(
                repo_path=str(plain_dir),
                task_name="test",
                prompt="do something",
            )

    def test_repo_not_in_allowlist(self, supervisor, tmp_path):
        # Create a valid git repo but don't add to allowlist
        other_repo = tmp_path / "other_repo"
        other_repo.mkdir()
        import subprocess
        subprocess.run(["git", "init", str(other_repo)], capture_output=True, check=True)
        with pytest.raises(ValueError, match="not in allowlist"):
            supervisor.create_worker(
                repo_path=str(other_repo),
                task_name="test",
                prompt="do something",
            )

    def test_invalid_model_for_executor(self, supervisor, git_repo):
        with pytest.raises(ValueError, match="Unsupported model 'o3'"):
            supervisor.create_worker(
                repo_path=str(git_repo),
                task_name="bad-model",
                prompt="do something",
                executor="codex",
                model="o3",
            )


class TestCreateWorkerIntegration:
    """Integration tests that create real workers using Python subprocesses."""

    def _create_worker_with_script(self, supervisor, git_repo, script, task_name="test-task"):
        """Helper to create a worker using a Python script instead of codex."""
        # Patch build_codex_command to use our Python script
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            # We'll dynamically set the command based on result_json_path
            def fake_build(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
                full_script = script.replace("__RESULT_PATH__", str(result_json_path))
                return [sys.executable, "-c", full_script]

            mock_build.side_effect = fake_build
            return supervisor.create_worker(
                repo_path=str(git_repo),
                task_name=task_name,
                prompt="Test prompt",
                tags=["test"],
                metadata={"test": True},
            )

    def test_successful_worker_lifecycle(self, supervisor, git_repo):
        """Full lifecycle: create -> running -> succeeded -> collect -> cleanup."""
        script = (
            "import json; "
            "open('fake.py', 'a').write('worker output\\n'); "
            "json.dump("
            '{"summary":"done","status":"completed",'
            '"files_changed":["fake.py"],"commits":[],"tests":[]}, '
            "open('__RESULT_PATH__', 'w'))"
        )
        payload = self._create_worker_with_script(supervisor, git_repo, script)

        assert payload.status == WorkerStatus.RUNNING
        assert payload.worker_id.startswith("w_")
        assert "codex/test-task/" in payload.branch_name
        assert payload.tags == ["test"]
        assert payload.metadata == {"test": True}

        # Wait for completion
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = supervisor.get_worker_status(payload.worker_id)
            if status.status.is_terminal:
                break
            time.sleep(0.2)

        status = supervisor.get_worker_status(payload.worker_id)
        assert status.status == WorkerStatus.SUCCEEDED
        assert status.exit_code == 0

        # Collect result
        result = supervisor.collect_worker_result(payload.worker_id)
        assert result["result"] is not None
        assert result["result"]["summary"] == "done"
        assert result["result"]["status"] == "completed"

        # Collect with logs
        result_with_logs = supervisor.collect_worker_result(
            payload.worker_id, include_logs=True
        )
        assert "stdout_tail" in result_with_logs
        assert "stderr_tail" in result_with_logs

        # Cleanup
        cleanup = supervisor.cleanup_worker(payload.worker_id)
        assert cleanup["worktree_removed"] is True
        assert cleanup["branch_removed"] is True
        assert cleanup["worker_dir_removed"] is True
        assert cleanup["errors"] == []

    def test_success_auto_commits_and_normalizes_result(self, supervisor, git_repo):
        """Valid results are still checked against actual git state."""
        script = (
            "import json; "
            "open('actual.py', 'a').write('real work\\n'); "
            "json.dump("
            '{"summary":"done","status":"completed",'
            '"files_changed":["hallucinated.py"],"commits":[],"tests":[]}, '
            "open('__RESULT_PATH__', 'w'))"
        )
        payload = self._create_worker_with_script(
            supervisor, git_repo, script, task_name="normalize-result"
        )

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = supervisor.get_worker_status(payload.worker_id)
            if status.status.is_terminal:
                break
            time.sleep(0.2)

        status = supervisor.get_worker_status(payload.worker_id)
        assert status.status == WorkerStatus.SUCCEEDED

        result = supervisor.collect_worker_result(payload.worker_id)["result"]
        assert result["files_changed"] == ["actual.py"]
        assert len(result["commits"]) == 1

        log = subprocess.run(
            ["git", "-C", status.worktree_path, "log", "--oneline", "-1"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "auto-commit: uncommitted agent work" in log.stdout

    def test_fake_self_reported_changes_fail(self, supervisor, git_repo):
        """A completed result cannot claim changed files that git cannot see."""
        script = (
            "import json; "
            "json.dump("
            '{"summary":"claimed work","status":"completed",'
            '"files_changed":["fake.py"],"commits":[],"tests":[]}, '
            "open('__RESULT_PATH__', 'w'))"
        )
        payload = self._create_worker_with_script(
            supervisor, git_repo, script, task_name="fake-self-report"
        )

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = supervisor.get_worker_status(payload.worker_id)
            if status.status.is_terminal:
                break
            time.sleep(0.2)

        status = supervisor.get_worker_status(payload.worker_id)
        assert status.status == WorkerStatus.FAILED
        assert "no file changes in git" in (status.error_message or "")

    def test_failed_worker_no_result(self, supervisor, git_repo):
        """Worker that exits 0 but writes no result.json should fail."""
        script = "print('did nothing useful')"
        payload = self._create_worker_with_script(
            supervisor, git_repo, script, task_name="no-result"
        )

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = supervisor.get_worker_status(payload.worker_id)
            if status.status.is_terminal:
                break
            time.sleep(0.2)

        status = supervisor.get_worker_status(payload.worker_id)
        assert status.status == WorkerStatus.FAILED
        assert "Result validation failed" in (status.error_message or "")

    def test_running_worker_persists_heartbeat(self, tmp_path, git_repo):
        """A running worker updates heartbeat fields before it completes."""
        base_dir = tmp_path / "fleet-heartbeat"
        supervisor = FleetSupervisor(
            base_dir=base_dir,
            allowed_repos=[str(git_repo)],
            default_model="gpt-5.5",
            default_timeout=30,
            max_concurrent=1,
            heartbeat_interval=1.0,
            stale_timeout=0,
        )
        script = (
            "import json, time; "
            "time.sleep(2.5); "
            "open('fake.py', 'a').write('heartbeat output\\n'); "
            "json.dump("
            '{"summary":"done","status":"completed",'
            '"files_changed":["fake.py"],"commits":[],"tests":[]}, '
            "open('__RESULT_PATH__', 'w'))"
        )
        try:
            payload = self._create_worker_with_script(
                supervisor, git_repo, script, task_name="heartbeat"
            )

            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                status = supervisor.get_worker_status(payload.worker_id)
                if (
                    status.last_heartbeat_at is not None
                    and status.heartbeat_message
                    and "running" in status.heartbeat_message.lower()
                ):
                    break
                time.sleep(0.2)

            status = supervisor.get_worker_status(payload.worker_id)
            assert status.last_heartbeat_at is not None
            assert status.last_activity_at is not None
            assert status.heartbeat_message is not None

            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                status = supervisor.get_worker_status(payload.worker_id)
                if status.status.is_terminal:
                    break
                time.sleep(0.2)

            assert status.status == WorkerStatus.SUCCEEDED
        finally:
            supervisor.close()

    def test_failed_worker_nonzero_exit(self, supervisor, git_repo):
        """Worker that exits non-zero should fail."""
        script = "import sys; sys.exit(42)"
        payload = self._create_worker_with_script(
            supervisor, git_repo, script, task_name="exit-fail"
        )

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = supervisor.get_worker_status(payload.worker_id)
            if status.status.is_terminal:
                break
            time.sleep(0.2)

        status = supervisor.get_worker_status(payload.worker_id)
        assert status.status == WorkerStatus.FAILED
        assert status.exit_code == 42

    def test_worker_invalid_result_json(self, supervisor, git_repo):
        """Worker that writes invalid result.json should fail."""
        script = (
            "f = open('__RESULT_PATH__', 'w'); "
            "f.write('{\"bad\": \"schema\"}'); "
            "f.close()"
        )
        payload = self._create_worker_with_script(
            supervisor, git_repo, script, task_name="bad-result"
        )

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = supervisor.get_worker_status(payload.worker_id)
            if status.status.is_terminal:
                break
            time.sleep(0.2)

        status = supervisor.get_worker_status(payload.worker_id)
        assert status.status == WorkerStatus.FAILED
        assert "Result validation failed" in (status.error_message or "")

    @pytest.mark.parametrize("executor", ["gemini", "claude"])
    def test_executor_stdout_result_materialized(self, supervisor, git_repo, executor):
        """Gemini and Claude should succeed without writing result.json directly."""
        payload_json = json.dumps(
            {
                "summary": f"{executor} ok",
                "files_changed": ["src/app.py"],
                "tests": [],
                "commits": [],
                "next_steps": ["review"],
                "status": "completed",
            }
        )
        if executor == "gemini":
            script = (
                "import json, os; "
                "os.makedirs('src', exist_ok=True); "
                "open('src/app.py', 'a').write('stdout materialized\\n'); "
                "print('Loaded cached credentials.', flush=True); "
                f"payload = {payload_json!r}; "
                "print(json.dumps({"
                "'type': 'message', 'role': 'assistant', 'content': payload"
                "}), flush=True)"
            )
        else:
            script = (
                "import json, os; "
                "os.makedirs('src', exist_ok=True); "
                "open('src/app.py', 'a').write('stdout materialized\\n'); "
                f"payload = {payload_json!r}; "
                "print(json.dumps({"
                "'type': 'assistant', "
                "'message': {'content': [{'type': 'text', 'text': payload}]}"
                "}), flush=True); "
                "print(json.dumps({'type': 'result', 'result': payload}), flush=True)"
            )

        with patch("codefleet.supervisor.build_worker_command") as mock_build:
            mock_build.return_value = [sys.executable, "-c", script]
            worker = supervisor.create_worker(
                repo_path=str(git_repo),
                task_name=f"{executor}-stdout-result",
                prompt="Return final JSON in stdout",
                executor=executor,
            )

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = supervisor.get_worker_status(worker.worker_id)
            if status.status.is_terminal:
                break
            time.sleep(0.2)

        status = supervisor.get_worker_status(worker.worker_id)
        assert status.status == WorkerStatus.SUCCEEDED
        assert status.heartbeat_message == (
            "Worker completed successfully via parsed stdout result"
        )

        result = supervisor.collect_worker_result(worker.worker_id)
        assert result["result"]["summary"] == f"{executor} ok"
        assert result["result"]["status"] == "completed"


class TestCancelWorker:
    def test_cancel_running_worker(self, supervisor, git_repo):
        """Cancel a worker that is sleeping."""
        script = "import time; time.sleep(300)"
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            mock_build.return_value = [sys.executable, "-c", script]
            payload = supervisor.create_worker(
                repo_path=str(git_repo),
                task_name="cancel-test",
                prompt="Sleep forever",
            )

        time.sleep(0.5)
        result = supervisor.cancel_worker(payload.worker_id)
        assert result.status == WorkerStatus.CANCELLED

    def test_cancel_terminal_worker_fails(self, supervisor, git_repo):
        """Cannot cancel a worker that has already finished."""
        script = "pass"
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            mock_build.return_value = [sys.executable, "-c", script]
            payload = supervisor.create_worker(
                repo_path=str(git_repo),
                task_name="cancel-terminal",
                prompt="Finish immediately",
            )

        # Wait for it to finish
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = supervisor.get_worker_status(payload.worker_id)
            if status.status.is_terminal:
                break
            time.sleep(0.1)

        with pytest.raises(ValueError, match="already terminal"):
            supervisor.cancel_worker(payload.worker_id)


class TestCleanupWorker:
    def test_cleanup_non_terminal_fails(self, supervisor, git_repo):
        """Cannot cleanup a running worker."""
        script = "import time; time.sleep(300)"
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            mock_build.return_value = [sys.executable, "-c", script]
            payload = supervisor.create_worker(
                repo_path=str(git_repo),
                task_name="cleanup-running",
                prompt="Sleep",
            )

        time.sleep(0.3)
        with pytest.raises(ValueError, match="Cannot cleanup non-terminal"):
            supervisor.cleanup_worker(payload.worker_id)

        # Clean up the worker for test teardown
        supervisor.cancel_worker(payload.worker_id)

    def test_cleanup_preserves_branch(self, supervisor, git_repo):
        """Cleanup with remove_branch=False should keep the branch."""
        import subprocess

        script = (
            "import json; "
            "json.dump("
            '{"summary":"done","status":"completed"}, '
            "open('__RESULT_PATH__', 'w'))"
        )
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            def fake_build(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
                full_script = script.replace("__RESULT_PATH__", str(result_json_path))
                return [sys.executable, "-c", full_script]
            mock_build.side_effect = fake_build

            payload = supervisor.create_worker(
                repo_path=str(git_repo),
                task_name="keep-branch",
                prompt="Test",
            )

        # Wait for completion
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = supervisor.get_worker_status(payload.worker_id)
            if status.status.is_terminal:
                break
            time.sleep(0.2)

        cleanup = supervisor.cleanup_worker(
            payload.worker_id,
            remove_branch=False,
            remove_worktree_dir=True,
        )
        assert cleanup["branch_removed"] is False

        # Branch should still exist
        result = subprocess.run(
            ["git", "-C", str(git_repo), "branch", "--list", payload.branch_name],
            capture_output=True,
            text=True,
        )
        assert payload.branch_name in result.stdout


class TestListWorkers:
    def test_list_empty(self, supervisor):
        workers = supervisor.list_workers()
        assert workers == []

    def test_list_with_workers(self, supervisor, git_repo):
        """Create multiple workers and list them."""
        script = "import time; time.sleep(300)"
        workers_created = []
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            mock_build.return_value = [sys.executable, "-c", script]
            for i in range(3):
                payload = supervisor.create_worker(
                    repo_path=str(git_repo),
                    task_name=f"list-test-{i}",
                    prompt=f"Task {i}",
                )
                workers_created.append(payload)
                time.sleep(0.1)  # ensure different created_at

        workers = supervisor.list_workers()
        assert len(workers) == 3

        # Cleanup
        for w in workers_created:
            supervisor.cancel_worker(w.worker_id)

    def test_list_filtered_by_status(self, supervisor, git_repo):
        """Filter workers by status."""
        script_sleep = "import time; time.sleep(300)"
        script_done = "pass"
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            mock_build.return_value = [sys.executable, "-c", script_sleep]
            running_payload = supervisor.create_worker(
                repo_path=str(git_repo),
                task_name="filter-running",
                prompt="Running",
            )

            mock_build.return_value = [sys.executable, "-c", script_done]
            done_payload = supervisor.create_worker(
                repo_path=str(git_repo),
                task_name="filter-done",
                prompt="Done",
            )

        # Wait for done worker to finish
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = supervisor.get_worker_status(done_payload.worker_id)
            if status.status.is_terminal:
                break
            time.sleep(0.1)

        running = supervisor.list_workers(statuses=["running"])
        assert len(running) >= 1
        assert all(w.status == WorkerStatus.RUNNING for w in running)

        # Cleanup
        supervisor.cancel_worker(running_payload.worker_id)


class TestGetWorkerStatus:
    def test_nonexistent_worker(self, supervisor):
        with pytest.raises(ValueError, match="Worker not found"):
            supervisor.get_worker_status("w_nonexistent")


class TestCollectWorkerResult:
    def test_nonexistent_worker(self, supervisor):
        with pytest.raises(ValueError, match="Worker not found"):
            supervisor.collect_worker_result("w_nonexistent")

    def test_collect_with_logs(self, supervisor, git_repo):
        """Collect result with log tails."""
        script = (
            "import json, sys; "
            "print('stdout line 1'); "
            "print('stdout line 2'); "
            "print('stderr info', file=sys.stderr); "
            "json.dump("
            '{"summary":"done","status":"completed"}, '
            "open('__RESULT_PATH__', 'w'))"
        )
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            def fake_build(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
                full_script = script.replace("__RESULT_PATH__", str(result_json_path))
                return [sys.executable, "-c", full_script]
            mock_build.side_effect = fake_build

            payload = supervisor.create_worker(
                repo_path=str(git_repo),
                task_name="collect-logs",
                prompt="Test",
            )

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = supervisor.get_worker_status(payload.worker_id)
            if status.status.is_terminal:
                break
            time.sleep(0.2)

        result = supervisor.collect_worker_result(
            payload.worker_id, include_logs=True, log_tail_lines=10
        )
        assert "stdout_tail" in result
        assert "stdout line 1" in result["stdout_tail"]
        assert "stderr_tail" in result
        assert "stderr info" in result["stderr_tail"]


class TestConcurrencyLimit:
    def test_exceeds_limit(self, tmp_path, git_repo):
        """Creating workers beyond the limit should fail."""
        sup = FleetSupervisor(
            base_dir=tmp_path / "fleet_limit",
            allowed_repos=[str(git_repo)],
            default_model="gpt-5.5",
            default_timeout=60,
            max_concurrent=2,
        )

        script = "import time; time.sleep(300)"
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            mock_build.return_value = [sys.executable, "-c", script]

            sup.create_worker(
                repo_path=str(git_repo), task_name="limit-1", prompt="1"
            )
            sup.create_worker(
                repo_path=str(git_repo), task_name="limit-2", prompt="2"
            )

            with pytest.raises(RuntimeError, match="Concurrency limit"):
                sup.create_worker(
                    repo_path=str(git_repo), task_name="limit-3", prompt="3"
                )

        # Cleanup
        for w in sup.list_workers():
            if not w.status.is_terminal:
                sup.cancel_worker(w.worker_id)
        sup.close()


class TestRepoAllowlist:
    def test_no_allowlist_allows_all(self, tmp_path, git_repo):
        sup = FleetSupervisor(
            base_dir=tmp_path / "fleet_noallow",
            allowed_repos=None,
            default_timeout=10,
        )
        assert sup._is_repo_allowed(git_repo)
        sup.close()

    def test_allowlist_blocks_unlisted(self, tmp_path, git_repo):
        sup = FleetSupervisor(
            base_dir=tmp_path / "fleet_allow",
            allowed_repos=["/some/other/repo"],
            default_timeout=10,
        )
        assert not sup._is_repo_allowed(git_repo)
        sup.close()

    def test_allowlist_allows_listed(self, tmp_path, git_repo):
        sup = FleetSupervisor(
            base_dir=tmp_path / "fleet_allow2",
            allowed_repos=[str(git_repo)],
            default_timeout=10,
        )
        assert sup._is_repo_allowed(git_repo)
        sup.close()

    def test_allowlist_allows_subdirectory(self, tmp_path, git_repo):
        sup = FleetSupervisor(
            base_dir=tmp_path / "fleet_allow3",
            allowed_repos=[str(git_repo.parent)],
            default_timeout=10,
        )
        assert sup._is_repo_allowed(git_repo)
        sup.close()


class TestTailFile:
    def test_tail_existing_file(self, tmp_path):
        f = tmp_path / "test.log"
        lines = [f"line {i}" for i in range(100)]
        f.write_text("\n".join(lines))
        result = FleetSupervisor._tail_file(f, 10)
        assert "line 90" in result
        assert "line 99" in result
        assert "line 0" not in result

    def test_tail_short_file(self, tmp_path):
        f = tmp_path / "short.log"
        f.write_text("only one line")
        result = FleetSupervisor._tail_file(f, 10)
        assert result == "only one line"

    def test_tail_nonexistent_file(self, tmp_path):
        result = FleetSupervisor._tail_file(tmp_path / "missing.log", 10)
        assert result == ""

    def test_tail_empty_file(self, tmp_path):
        f = tmp_path / "empty.log"
        f.write_text("")
        result = FleetSupervisor._tail_file(f, 10)
        assert result == ""


class TestStatePersistence:
    def test_state_survives_restart(self, tmp_path, git_repo):
        """Worker state should survive supervisor restart (SQLite durability)."""
        base_dir = tmp_path / "fleet_persist"

        # Create supervisor, add a worker record manually
        sup1 = FleetSupervisor(
            base_dir=base_dir,
            allowed_repos=[str(git_repo)],
            default_timeout=10,
        )
        # Insert a worker record directly into the store
        from codefleet.models import WorkerRecord
        import time as t

        record = WorkerRecord(
            worker_id="w_persist_test",
            task_name="persistence test",
            repo_path=str(git_repo),
            branch_name="codex/persist/w_persist_test",
            worktree_path=str(base_dir / "workers/w_persist_test/worktree"),
            worker_dir=str(base_dir / "workers/w_persist_test"),
            model="gpt-5.5",
            status=WorkerStatus.SUCCEEDED,
            created_at=t.time(),
            timeout_seconds=60,
            command_json="[]",
            prompt="test",
            result_json_path=str(base_dir / "workers/w_persist_test/result.json"),
            stdout_path=str(base_dir / "workers/w_persist_test/stdout.log"),
            stderr_path=str(base_dir / "workers/w_persist_test/stderr.log"),
            prompt_path=str(base_dir / "workers/w_persist_test/prompt.txt"),
            meta_path=str(base_dir / "workers/w_persist_test/meta.json"),
        )
        sup1.store.insert_worker(record)
        sup1.close()

        # Create new supervisor with same base_dir
        sup2 = FleetSupervisor(
            base_dir=base_dir,
            allowed_repos=[str(git_repo)],
            default_timeout=10,
        )
        status = sup2.get_worker_status("w_persist_test")
        assert status.worker_id == "w_persist_test"
        assert status.status == WorkerStatus.SUCCEEDED
        assert status.task_name == "persistence test"
        sup2.close()

    def test_recover_orphaned_worker_from_stdout_result(self, tmp_path, git_repo):
        """A previously running Gemini worker should recover from stdout on restart."""
        base_dir = tmp_path / "fleet_recover_stdout"
        worker_id = "w_recover_stdout"
        worker_dir = base_dir / "workers" / worker_id
        worktree = worker_dir / "worktree"
        codefleet_dir = worktree / ".codefleet"
        codefleet_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = worker_dir / "stdout.log"
        stderr_path = worker_dir / "stderr.log"
        prompt_path = codefleet_dir / "prompt.txt"
        result_path = codefleet_dir / "result.json"
        meta_path = worker_dir / "meta.json"

        prompt_path.write_text("Recover me", encoding="utf-8")
        meta_path.write_text("{}", encoding="utf-8")
        stdout_path.write_text(
            '{"type":"message","role":"assistant","content":"'
            '{\\"summary\\":\\"recovered\\",\\"files_changed\\":[\\"a.py\\"],'
            '\\"tests\\":[],\\"commits\\":[],\\"next_steps\\":[\\"ship\\"],'
            '\\"status\\":\\"completed\\"}"}\n',
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")

        store = WorkerStore(base_dir / "fleet.db")
        try:
            store.insert_worker(
                WorkerRecord(
                    worker_id=worker_id,
                    task_name="recover stdout",
                    repo_path=str(git_repo),
                    branch_name=f"gemini/recover/{worker_id}",
                    worktree_path=str(worktree),
                    worker_dir=str(worker_dir),
                    model="gemini-3.1-pro-preview",
                    executor=ExecutorType.GEMINI,
                    status=WorkerStatus.RUNNING,
                    created_at=time.time() - 15,
                    started_at=time.time() - 10,
                    last_activity_at=time.time() - 5,
                    timeout_seconds=60,
                    pid=999999,
                    command_json='["gemini", "-p", "test"]',
                    prompt="Recover me",
                    result_json_path=str(result_path),
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                    prompt_path=str(prompt_path),
                    meta_path=str(meta_path),
                )
            )
        finally:
            store.close()

        sup = FleetSupervisor(
            base_dir=base_dir,
            allowed_repos=[str(git_repo)],
            default_timeout=60,
        )
        try:
            status = sup.get_worker_status(worker_id)
            assert status.status == WorkerStatus.SUCCEEDED
            assert status.heartbeat_message == (
                "Recovered completed worker after supervisor restart via parsed stdout result"
            )
            assert result_path.exists()

            result = sup.collect_worker_result(worker_id)
            assert result["result"]["summary"] == "recovered"
            assert result["result"]["status"] == "completed"
        finally:
            sup.close()

    def test_recover_running_worker_monitor_after_restart(self, tmp_path, git_repo):
        """A live worker PID should be reattached and complete under the new supervisor."""
        base_dir = tmp_path / "fleet_recover_live"
        worker_id = "w_recover_live"
        worker_dir = base_dir / "workers" / worker_id
        worktree = worker_dir / "worktree"
        codefleet_dir = worktree / ".codefleet"
        codefleet_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = worker_dir / "stdout.log"
        stderr_path = worker_dir / "stderr.log"
        prompt_path = codefleet_dir / "prompt.txt"
        result_path = codefleet_dir / "result.json"
        meta_path = worker_dir / "meta.json"

        prompt_path.write_text("Recover live worker", encoding="utf-8")
        meta_path.write_text("{}", encoding="utf-8")

        script = (
            "import json, time; "
            "print('still running', flush=True); "
            f"time.sleep(2); json.dump({{'summary':'done','files_changed':['fake.py'],"
            f"'tests':[],'commits':[],'next_steps':[],'status':'completed'}}, "
            f"open({str(result_path)!r}, 'w'))"
        )
        stdout_file = stdout_path.open("w")
        stderr_file = stderr_path.open("w")
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=str(worktree),
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        stdout_file.close()
        stderr_file.close()

        store = WorkerStore(base_dir / "fleet.db")
        try:
            store.insert_worker(
                WorkerRecord(
                    worker_id=worker_id,
                    task_name="recover live",
                    repo_path=str(git_repo),
                    branch_name=f"codex/recover/{worker_id}",
                    worktree_path=str(worktree),
                    worker_dir=str(worker_dir),
                    model="gpt-5.5",
                    executor=ExecutorType.CODEX,
                    status=WorkerStatus.RUNNING,
                    created_at=time.time() - 5,
                    started_at=time.time() - 5,
                    last_activity_at=time.time() - 5,
                    timeout_seconds=60,
                    pid=proc.pid,
                    command_json=json.dumps([sys.executable, "-c", script]),
                    prompt="Recover live worker",
                    result_json_path=str(result_path),
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                    prompt_path=str(prompt_path),
                    meta_path=str(meta_path),
                )
            )
        finally:
            store.close()

        sup = FleetSupervisor(
            base_dir=base_dir,
            allowed_repos=[str(git_repo)],
            default_timeout=60,
            heartbeat_interval=0.5,
            stale_timeout=0,
        )
        try:
            deadline = time.monotonic() + 10
            recovered = False
            reaped = False
            while time.monotonic() < deadline:
                status = sup.get_worker_status(worker_id)
                if status.heartbeat_message == "Recovered worker monitor after supervisor restart":
                    recovered = True
                if not reaped and proc.poll() is not None:
                    proc.wait(timeout=5)
                    reaped = True
                if status.status.is_terminal:
                    break
                time.sleep(0.2)

            status = sup.get_worker_status(worker_id)
            assert recovered
            assert status.status == WorkerStatus.SUCCEEDED
            assert sup.collect_worker_result(worker_id)["result"]["summary"] == "done"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            sup.close()


class TestSpawnDepth:
    """Test the max_spawn_depth enforcement."""

    def test_root_worker_can_spawn_child(self, tmp_path, git_repo):
        """With max_spawn_depth=1, a root worker can spawn one child."""
        sup = FleetSupervisor(
            base_dir=tmp_path / "fleet_depth",
            allowed_repos=[str(git_repo)],
            default_timeout=60,
            max_concurrent=5,
            max_spawn_depth=1,
        )
        script = "import time; time.sleep(300)"
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            mock_build.return_value = [sys.executable, "-c", script]

            # Create root worker
            root = sup.create_worker(
                repo_path=str(git_repo),
                task_name="root",
                prompt="Root task",
            )

            # Root spawns a child — should succeed (depth of root = 1, 1 <= 1)
            child = sup.create_worker(
                repo_path=str(git_repo),
                task_name="child",
                prompt="Child task",
                parent_worker_id=root.worker_id,
            )
            assert child.worker_id != root.worker_id

            # Child tries to spawn grandchild — should fail (depth of child = 2, 2 > 1)
            with pytest.raises(RuntimeError, match="Spawn depth limit"):
                sup.create_worker(
                    repo_path=str(git_repo),
                    task_name="grandchild",
                    prompt="Too deep",
                    parent_worker_id=child.worker_id,
                )

        # Cleanup
        for w in sup.list_workers():
            if not w.status.is_terminal:
                sup.cancel_worker(w.worker_id)
        sup.close()

    def test_depth_zero_blocks_all_children(self, tmp_path, git_repo):
        """With max_spawn_depth=0, no worker can spawn children."""
        sup = FleetSupervisor(
            base_dir=tmp_path / "fleet_depth0",
            allowed_repos=[str(git_repo)],
            default_timeout=60,
            max_spawn_depth=0,
        )
        script = "import time; time.sleep(300)"
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            mock_build.return_value = [sys.executable, "-c", script]

            root = sup.create_worker(
                repo_path=str(git_repo),
                task_name="root",
                prompt="Root",
            )

            with pytest.raises(RuntimeError, match="Spawn depth limit"):
                sup.create_worker(
                    repo_path=str(git_repo),
                    task_name="child",
                    prompt="Blocked",
                    parent_worker_id=root.worker_id,
                )

        for w in sup.list_workers():
            if not w.status.is_terminal:
                sup.cancel_worker(w.worker_id)
        sup.close()

    def test_depth_two_allows_grandchildren(self, tmp_path, git_repo):
        """With max_spawn_depth=2, root -> child -> grandchild is allowed."""
        sup = FleetSupervisor(
            base_dir=tmp_path / "fleet_depth2",
            allowed_repos=[str(git_repo)],
            default_timeout=60,
            max_concurrent=10,
            max_spawn_depth=2,
        )
        script = "import time; time.sleep(300)"
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            mock_build.return_value = [sys.executable, "-c", script]

            root = sup.create_worker(
                repo_path=str(git_repo), task_name="root", prompt="R"
            )
            child = sup.create_worker(
                repo_path=str(git_repo), task_name="child", prompt="C",
                parent_worker_id=root.worker_id,
            )
            grandchild = sup.create_worker(
                repo_path=str(git_repo), task_name="grandchild", prompt="G",
                parent_worker_id=child.worker_id,
            )
            assert grandchild.worker_id != child.worker_id

            # Great-grandchild blocked (depth 3 > 2)
            with pytest.raises(RuntimeError, match="Spawn depth limit"):
                sup.create_worker(
                    repo_path=str(git_repo), task_name="great-gc", prompt="X",
                    parent_worker_id=grandchild.worker_id,
                )

        for w in sup.list_workers():
            if not w.status.is_terminal:
                sup.cancel_worker(w.worker_id)
        sup.close()


class TestClaudeExecutor:
    def test_create_claude_worker_branch_prefix(self, supervisor, git_repo):
        """Claude workers should get claude/ branch prefix."""
        script = "import time; time.sleep(300)"
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            mock_build.return_value = [sys.executable, "-c", script]
            payload = supervisor.create_worker(
                repo_path=str(git_repo),
                task_name="claude-test",
                prompt="Test",
                executor="claude",
            )
        assert payload.branch_name.startswith("claude/")
        assert payload.executor == ExecutorType.CLAUDE
        supervisor.cancel_worker(payload.worker_id)

    def test_healthcheck_claude_found(self, supervisor):
        result = supervisor.healthcheck()
        assert "claude_found" in result
        assert "claude_path" in result


class TestSalvageResult:
    """Tests for _salvage_result auto-commit safety net."""

    def _setup_worktree_worker(self, supervisor, git_repo, tmp_path):
        """Create a real worktree and a WorkerRecord pointing at it.

        Returns (record, worktree_path, base_commit).
        """
        # Get the current HEAD as base_commit
        base_commit = subprocess.run(
            ["git", "-C", str(git_repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

        worker_id = "w_salvage_test"
        worker_dir = tmp_path / "workers" / worker_id
        worker_dir.mkdir(parents=True)

        branch_name = f"codex/salvage/{worker_id}"
        worktree_path = worker_dir / "worktree"

        # Create a real worktree
        subprocess.run(
            ["git", "-C", str(git_repo), "worktree", "add",
             "-b", branch_name, str(worktree_path), "HEAD"],
            capture_output=True, check=True,
        )

        # Write meta.json with base_commit
        meta_path = worker_dir / "meta.json"
        meta_path.write_text(json.dumps({"base_commit": base_commit}))

        # Set up .codefleet dir in worktree
        codefleet_dir = worktree_path / ".codefleet"
        codefleet_dir.mkdir(exist_ok=True)

        result_json_path = codefleet_dir / "result.json"

        record = WorkerRecord(
            worker_id=worker_id,
            task_name="salvage test",
            repo_path=str(git_repo),
            branch_name=branch_name,
            worktree_path=str(worktree_path),
            worker_dir=str(worker_dir),
            model="gpt-5.5",
            status=WorkerStatus.FAILED,
            created_at=time.time(),
            timeout_seconds=60,
            command_json='["codex", "exec", "test"]',
            prompt="Salvage me",
            result_json_path=str(result_json_path),
            stdout_path=str(worker_dir / "stdout.log"),
            stderr_path=str(worker_dir / "stderr.log"),
            prompt_path=str(codefleet_dir / "prompt.txt"),
            meta_path=str(meta_path),
        )

        return record, worktree_path, base_commit

    def test_auto_commits_uncommitted_changes(self, supervisor, git_repo, tmp_path):
        """Uncommitted changes in the worktree are auto-committed during salvage."""
        record, worktree_path, base_commit = self._setup_worktree_worker(
            supervisor, git_repo, tmp_path,
        )

        # Create an uncommitted file in the worktree
        (worktree_path / "new_feature.py").write_text("print('hello')")

        result = supervisor._salvage_result(record)
        assert result is True

        # Verify the file was committed (not just staged)
        status = subprocess.run(
            ["git", "-C", str(worktree_path), "status", "--porcelain"],
            capture_output=True, text=True,
        )
        assert "new_feature.py" not in status.stdout

        # Verify the commit exists on the branch
        log = subprocess.run(
            ["git", "-C", str(worktree_path), "log", "--oneline", "-1"],
            capture_output=True, text=True,
        )
        assert "auto-commit" in log.stdout

        # Verify salvaged result.json lists the file
        result_data = json.loads(Path(record.result_json_path).read_text())
        assert "new_feature.py" in result_data["files_changed"]

    def test_salvage_includes_already_committed_changes(self, supervisor, git_repo, tmp_path):
        """Changes the agent already committed are included in the salvaged result."""
        record, worktree_path, base_commit = self._setup_worktree_worker(
            supervisor, git_repo, tmp_path,
        )

        # Simulate an agent that committed properly
        (worktree_path / "committed.py").write_text("x = 1")
        subprocess.run(
            ["git", "-C", str(worktree_path), "add", "committed.py"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(worktree_path), "commit", "-m", "agent commit"],
            capture_output=True, check=True,
        )

        result = supervisor._salvage_result(record)
        assert result is True

        result_data = json.loads(Path(record.result_json_path).read_text())
        assert "committed.py" in result_data["files_changed"]

    def test_salvage_excludes_codefleet_dir(self, supervisor, git_repo, tmp_path):
        """Files under .codefleet/ are not included in salvaged results."""
        record, worktree_path, base_commit = self._setup_worktree_worker(
            supervisor, git_repo, tmp_path,
        )

        # Only create a .codefleet file (no real work)
        (worktree_path / ".codefleet" / "internal.txt").write_text("meta")

        result = supervisor._salvage_result(record)
        # No real changes to salvage
        assert result is False

    def test_salvage_no_changes_returns_false(self, supervisor, git_repo, tmp_path):
        """If there are no changes at all, salvage returns False."""
        record, worktree_path, base_commit = self._setup_worktree_worker(
            supervisor, git_repo, tmp_path,
        )

        result = supervisor._salvage_result(record)
        assert result is False

    def test_salvage_mixed_committed_and_uncommitted(self, supervisor, git_repo, tmp_path):
        """Both committed and uncommitted files appear in the salvaged result."""
        record, worktree_path, base_commit = self._setup_worktree_worker(
            supervisor, git_repo, tmp_path,
        )

        # Agent committed one file
        (worktree_path / "done.py").write_text("done = True")
        subprocess.run(
            ["git", "-C", str(worktree_path), "add", "done.py"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(worktree_path), "commit", "-m", "partial work"],
            capture_output=True, check=True,
        )

        # Agent forgot to commit this one
        (worktree_path / "forgot.py").write_text("forgot = True")

        result = supervisor._salvage_result(record)
        assert result is True

        result_data = json.loads(Path(record.result_json_path).read_text())
        assert "done.py" in result_data["files_changed"]
        assert "forgot.py" in result_data["files_changed"]

        # Both should now be committed (only .codefleet/ remains untracked)
        status = subprocess.run(
            ["git", "-C", str(worktree_path), "status", "--porcelain"],
            capture_output=True, text=True,
        )
        remaining = [
            line for line in status.stdout.strip().split("\n")
            if line.strip() and not line.endswith(".codefleet/")
        ]
        assert remaining == []
