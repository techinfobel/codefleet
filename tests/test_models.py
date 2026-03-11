"""Tests for models.py - Pydantic models and enums."""

import time

import pytest

from codex_fleet_supervisor.models import (
    ResultStatus,
    TestResult,
    TestStatus,
    WorkerRecord,
    WorkerResult,
    WorkerStatus,
    WorkerStatusPayload,
)


class TestWorkerStatus:
    def test_terminal_statuses(self):
        terminal = {
            WorkerStatus.SUCCEEDED,
            WorkerStatus.FAILED,
            WorkerStatus.CANCELLED,
            WorkerStatus.TIMED_OUT,
            WorkerStatus.CLEANUP_FAILED,
        }
        for status in WorkerStatus:
            if status in terminal:
                assert status.is_terminal, f"{status} should be terminal"
            else:
                assert not status.is_terminal, f"{status} should not be terminal"

    def test_non_terminal_statuses(self):
        assert not WorkerStatus.PENDING.is_terminal
        assert not WorkerStatus.RUNNING.is_terminal

    def test_string_values(self):
        assert WorkerStatus.PENDING.value == "pending"
        assert WorkerStatus.RUNNING.value == "running"
        assert WorkerStatus.SUCCEEDED.value == "succeeded"
        assert WorkerStatus.FAILED.value == "failed"
        assert WorkerStatus.CANCELLED.value == "cancelled"
        assert WorkerStatus.TIMED_OUT.value == "timed_out"
        assert WorkerStatus.CLEANUP_FAILED.value == "cleanup_failed"

    def test_from_string(self):
        assert WorkerStatus("running") == WorkerStatus.RUNNING
        assert WorkerStatus("succeeded") == WorkerStatus.SUCCEEDED


class TestResultStatus:
    def test_values(self):
        assert ResultStatus.COMPLETED.value == "completed"
        assert ResultStatus.BLOCKED.value == "blocked"


class TestTestStatus:
    def test_values(self):
        assert TestStatus.PASSED.value == "passed"
        assert TestStatus.FAILED.value == "failed"
        assert TestStatus.NOT_RUN.value == "not_run"


class TestTestResult:
    def test_create(self):
        tr = TestResult(command="pytest", status=TestStatus.PASSED, details="2 passed")
        assert tr.command == "pytest"
        assert tr.status == TestStatus.PASSED
        assert tr.details == "2 passed"

    def test_default_details(self):
        tr = TestResult(command="pytest", status=TestStatus.FAILED)
        assert tr.details == ""


class TestWorkerResult:
    def test_valid_completed(self):
        result = WorkerResult(
            summary="Added login feature",
            files_changed=["src/auth.py"],
            tests=[
                TestResult(
                    command="pytest tests/test_auth.py",
                    status=TestStatus.PASSED,
                    details="3 passed",
                )
            ],
            commits=["abc123"],
            next_steps=["Add rate limiting"],
            status=ResultStatus.COMPLETED,
        )
        assert result.summary == "Added login feature"
        assert len(result.files_changed) == 1
        assert len(result.tests) == 1
        assert result.status == ResultStatus.COMPLETED

    def test_valid_blocked(self):
        result = WorkerResult(
            summary="Blocked on missing dependency",
            status=ResultStatus.BLOCKED,
        )
        assert result.status == ResultStatus.BLOCKED
        assert result.files_changed == []
        assert result.tests == []

    def test_missing_required_fields(self):
        with pytest.raises(Exception):
            WorkerResult(files_changed=[])  # missing summary and status

    def test_invalid_status(self):
        with pytest.raises(Exception):
            WorkerResult(summary="test", status="invalid")

    def test_serialization_roundtrip(self):
        result = WorkerResult(
            summary="Test",
            files_changed=["a.py"],
            status=ResultStatus.COMPLETED,
        )
        data = result.model_dump()
        restored = WorkerResult.model_validate(data)
        assert restored == result


class TestWorkerRecord:
    def _make_record(self, **overrides):
        defaults = dict(
            worker_id="w_test123",
            task_name="test task",
            repo_path="/tmp/repo",
            branch_name="codex/test/w_test123",
            worktree_path="/tmp/fleet/workers/w_test123/worktree",
            worker_dir="/tmp/fleet/workers/w_test123",
            model="gpt-5.3-codex",
            status=WorkerStatus.PENDING,
            created_at=time.time(),
            timeout_seconds=600,
            codex_command='["codex", "exec", "test"]',
            prompt="Do something",
            result_json_path="/tmp/fleet/workers/w_test123/result.json",
            stdout_path="/tmp/fleet/workers/w_test123/stdout.log",
            stderr_path="/tmp/fleet/workers/w_test123/stderr.log",
            prompt_path="/tmp/fleet/workers/w_test123/prompt.txt",
            meta_path="/tmp/fleet/workers/w_test123/meta.json",
        )
        defaults.update(overrides)
        return WorkerRecord(**defaults)

    def test_create_minimal(self):
        record = self._make_record()
        assert record.worker_id == "w_test123"
        assert record.profile is None
        assert record.tags == []
        assert record.metadata == {}
        assert record.retry_count == 0

    def test_create_with_optionals(self):
        record = self._make_record(
            profile="reviewer",
            tags=["urgent", "backend"],
            metadata={"priority": 1},
            parent_worker_id="w_parent",
        )
        assert record.profile == "reviewer"
        assert record.tags == ["urgent", "backend"]
        assert record.metadata == {"priority": 1}
        assert record.parent_worker_id == "w_parent"


class TestWorkerStatusPayload:
    def test_from_record(self):
        now = time.time()
        record = WorkerRecord(
            worker_id="w_abc",
            task_name="my task",
            repo_path="/repo",
            branch_name="codex/my-task/w_abc",
            worktree_path="/fleet/workers/w_abc/worktree",
            worker_dir="/fleet/workers/w_abc",
            model="gpt-5.3-codex",
            profile="coder",
            status=WorkerStatus.RUNNING,
            created_at=now,
            started_at=now + 1,
            timeout_seconds=300,
            pid=12345,
            codex_command='["codex"]',
            prompt="Do work",
            result_json_path="/fleet/workers/w_abc/result.json",
            stdout_path="/fleet/workers/w_abc/stdout.log",
            stderr_path="/fleet/workers/w_abc/stderr.log",
            prompt_path="/fleet/workers/w_abc/prompt.txt",
            meta_path="/fleet/workers/w_abc/meta.json",
            tags=["test"],
            metadata={"key": "val"},
        )
        payload = WorkerStatusPayload.from_record(record)
        assert payload.worker_id == "w_abc"
        assert payload.task_name == "my task"
        assert payload.status == WorkerStatus.RUNNING
        assert payload.pid == 12345
        assert payload.profile == "coder"
        assert payload.tags == ["test"]
        assert payload.metadata == {"key": "val"}

    def test_payload_serialization(self):
        now = time.time()
        record = WorkerRecord(
            worker_id="w_ser",
            task_name="serialize",
            repo_path="/repo",
            branch_name="codex/serialize/w_ser",
            worktree_path="/wt",
            worker_dir="/wd",
            model="gpt-5.3-codex",
            status=WorkerStatus.SUCCEEDED,
            created_at=now,
            timeout_seconds=60,
            codex_command="[]",
            prompt="test",
            result_json_path="/r.json",
            stdout_path="/o.log",
            stderr_path="/e.log",
            prompt_path="/p.txt",
            meta_path="/m.json",
        )
        payload = WorkerStatusPayload.from_record(record)
        data = payload.model_dump()
        assert data["worker_id"] == "w_ser"
        assert data["status"] == "succeeded"
