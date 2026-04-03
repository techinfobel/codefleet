"""Tests for models.py - Pydantic models and enums."""

import time

import pytest

from codefleet.models import (
    ExecutorType,
    ResultStatus,
    StageDefinition,
    StageState,
    SupportedModel,
    WorkerRecord,
    WorkerResult,
    WorkerStatus,
    WorkerStatusPayload,
    WorkflowRecord,
    WorkflowStatus,
    WorkflowStatusPayload,
    WorktreeStrategy,
    supported_models_for_executor,
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


class TestWorkerResult:
    def test_valid_completed(self):
        result = WorkerResult(
            summary="Added login feature",
            files_changed=["src/auth.py"],
            next_steps=["Add rate limiting"],
            status=ResultStatus.COMPLETED,
        )
        assert result.summary == "Added login feature"
        assert len(result.files_changed) == 1
        assert result.status == ResultStatus.COMPLETED

    def test_valid_blocked(self):
        result = WorkerResult(
            summary="Blocked on missing dependency",
            status=ResultStatus.BLOCKED,
        )
        assert result.status == ResultStatus.BLOCKED
        assert result.files_changed == []

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
            model="gpt-5.4",
            status=WorkerStatus.PENDING,
            created_at=time.time(),
            timeout_seconds=600,
            command_json='["codex", "exec", "test"]',
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
            model="gpt-5.4",
            profile="coder",
            status=WorkerStatus.RUNNING,
            created_at=now,
            started_at=now + 1,
            timeout_seconds=300,
            pid=12345,
            command_json='["codex"]',
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
            model="gpt-5.4",
            status=WorkerStatus.SUCCEEDED,
            created_at=now,
            timeout_seconds=60,
            command_json="[]",
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


class TestExecutorType:
    def test_values(self):
        assert ExecutorType.CODEX.value == "codex"
        assert ExecutorType.GEMINI.value == "gemini"

    def test_from_string(self):
        assert ExecutorType("codex") == ExecutorType.CODEX
        assert ExecutorType("gemini") == ExecutorType.GEMINI

    def test_worker_record_default(self):
        record = WorkerRecord(
            worker_id="w_ex",
            task_name="t",
            repo_path="/r",
            branch_name="b",
            worktree_path="/w",
            worker_dir="/d",
            model="m",
            status=WorkerStatus.PENDING,
            created_at=time.time(),
            timeout_seconds=60,
            command_json="[]",
            prompt="p",
            result_json_path="/rj",
            stdout_path="/o",
            stderr_path="/e",
            prompt_path="/pp",
            meta_path="/mp",
        )
        assert record.executor == ExecutorType.CODEX

    def test_worker_record_gemini(self):
        record = WorkerRecord(
            worker_id="w_gem",
            task_name="t",
            repo_path="/r",
            branch_name="b",
            worktree_path="/w",
            worker_dir="/d",
            model="gemini-3.1-pro-preview",
            executor=ExecutorType.GEMINI,
            status=WorkerStatus.PENDING,
            created_at=time.time(),
            timeout_seconds=60,
            command_json="[]",
            prompt="p",
            result_json_path="/rj",
            stdout_path="/o",
            stderr_path="/e",
            prompt_path="/pp",
            meta_path="/mp",
        )
        assert record.executor == ExecutorType.GEMINI


class TestSupportedModel:
    def test_values(self):
        assert SupportedModel.GPT_5_4.value == "gpt-5.4"
        assert SupportedModel.GEMINI_3_1_PRO_PREVIEW.value == "gemini-3.1-pro-preview"
        assert SupportedModel.CLAUDE_OPUS_4_6.value == "claude-opus-4-6"
        assert SupportedModel.CLAUDE_SONNET_4_6.value == "claude-sonnet-4-6"

    def test_from_string(self):
        assert SupportedModel("gpt-5.4") == SupportedModel.GPT_5_4
        assert (
            SupportedModel("claude-sonnet-4-6")
            == SupportedModel.CLAUDE_SONNET_4_6
        )

    def test_payload_includes_executor(self):
        record = WorkerRecord(
            worker_id="w_ex2",
            task_name="t",
            repo_path="/r",
            branch_name="b",
            worktree_path="/w",
            worker_dir="/d",
            model="m",
            executor=ExecutorType.GEMINI,
            status=WorkerStatus.PENDING,
            created_at=time.time(),
            timeout_seconds=60,
            command_json="[]",
            prompt="p",
            result_json_path="/rj",
            stdout_path="/o",
            stderr_path="/e",
            prompt_path="/pp",
            meta_path="/mp",
        )
        payload = WorkerStatusPayload.from_record(record)
        assert payload.executor == ExecutorType.GEMINI

    def test_supported_models_for_executor(self):
        assert supported_models_for_executor(ExecutorType.CODEX) == ("gpt-5.4",)
        assert supported_models_for_executor(ExecutorType.GEMINI) == (
            "gemini-3.1-pro-preview",
        )
        assert supported_models_for_executor(ExecutorType.CLAUDE) == (
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        )


class TestWorktreeStrategy:
    def test_values(self):
        assert WorktreeStrategy.NEW.value == "new"
        assert WorktreeStrategy.INHERIT.value == "inherit"


class TestWorkflowModels:
    def test_stage_definition(self):
        sd = StageDefinition(
            name="impl",
            executor=ExecutorType.CODEX,
            prompt_template="{task_prompt}",
        )
        assert sd.worktree_strategy == WorktreeStrategy.NEW
        assert sd.depends_on == []
        assert sd.model is None

    def test_stage_definition_accepts_compatible_model(self):
        sd = StageDefinition(
            name="review",
            executor=ExecutorType.CLAUDE,
            prompt_template="{task_prompt}",
            model=SupportedModel.CLAUDE_OPUS_4_6,
        )
        assert sd.model == SupportedModel.CLAUDE_OPUS_4_6

    def test_stage_definition_rejects_incompatible_model(self):
        with pytest.raises(Exception, match="Unsupported model 'gpt-5.4' for executor 'gemini'"):
            StageDefinition(
                name="bad",
                executor=ExecutorType.GEMINI,
                prompt_template="{task_prompt}",
                model=SupportedModel.GPT_5_4,
            )

    def test_workflow_status_values(self):
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.SUCCEEDED.value == "succeeded"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"

    def test_stage_state_defaults(self):
        ss = StageState()
        assert ss.worker_id is None
        assert ss.status == WorkerStatus.PENDING

    def test_workflow_record(self):
        now = time.time()
        record = WorkflowRecord(
            workflow_id="wf_test",
            name="test-wf",
            status=WorkflowStatus.RUNNING,
            repo_path="/repo",
            base_ref="HEAD",
            task_prompt="Do stuff",
            stages=[
                StageDefinition(name="s0", executor=ExecutorType.CODEX, prompt_template="x"),
            ],
            stage_states={0: StageState()},
            created_at=now,
        )
        assert record.workflow_id == "wf_test"
        assert len(record.stages) == 1

    def test_workflow_status_payload_from_record(self):
        now = time.time()
        record = WorkflowRecord(
            workflow_id="wf_p",
            name="payload-test",
            status=WorkflowStatus.SUCCEEDED,
            repo_path="/repo",
            base_ref="HEAD",
            task_prompt="task",
            stages=[
                StageDefinition(name="s0", executor=ExecutorType.CODEX, prompt_template="x"),
                StageDefinition(name="s1", executor=ExecutorType.GEMINI, prompt_template="y", depends_on=[0]),
            ],
            stage_states={
                0: StageState(worker_id="w_1", status=WorkerStatus.SUCCEEDED),
                1: StageState(worker_id="w_2", status=WorkerStatus.SUCCEEDED),
            },
            created_at=now,
            completed_at=now + 10,
        )
        payload = WorkflowStatusPayload.from_record(record)
        assert payload.workflow_id == "wf_p"
        assert payload.status == WorkflowStatus.SUCCEEDED
        assert len(payload.stage_summary) == 2
        assert payload.stage_summary[0]["name"] == "s0"
        assert payload.stage_summary[0]["executor"] == "codex"
        assert payload.stage_summary[1]["executor"] == "gemini"
