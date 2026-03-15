"""Tests for supervisor.py - integration tests with real git, SQLite, and subprocesses."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from codefleet.models import ExecutorType, WorkerStatus, WorkerStatusPayload
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
        assert result["default_model"] == "gpt-5.4"
        assert "db_path" in result
        assert "base_dir" in result
        assert "max_spawn_depth" in result


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


class TestCreateWorkerIntegration:
    """Integration tests that create real workers using Python subprocesses."""

    def _create_worker_with_script(self, supervisor, git_repo, script, task_name="test-task"):
        """Helper to create a worker using a Python script instead of codex."""
        # Patch build_codex_command to use our Python script
        with patch(
            "codefleet.supervisor.build_worker_command"
        ) as mock_build:
            # We'll dynamically set the command based on result_json_path
            def fake_build(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None):
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
            "json.dump("
            '{"summary":"done","status":"completed","files_changed":[],"tests":[]}, '
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
            def fake_build(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None):
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
            def fake_build(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None):
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
            default_model="gpt-5.4",
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
            model="gpt-5.4",
            status=WorkerStatus.SUCCEEDED,
            created_at=t.time(),
            timeout_seconds=60,
            codex_command="[]",
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
