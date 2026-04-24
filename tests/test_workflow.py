"""Tests for workflow.py - WorkflowEngine: DAG validation, stage advancement,
template rendering, error propagation, fan-out/fan-in, cleanup."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from codefleet.models import (
    ExecutorType,
    StageDefinition,
    StageState,
    SupportedModel,
    WorkerStatus,
    WorkerStatusPayload,
    WorkflowRecord,
    WorkflowStatus,
    WorkflowStatusPayload,
    WorktreeStrategy,
)
from codefleet.supervisor import FleetSupervisor
from codefleet.workflow import WorkflowEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_build(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
    """Build a Python script that writes a valid result.json and exits."""
    script = (
        "import json; "
        "json.dump("
        '{"summary":"stage done","status":"completed",'
        '"files_changed":["a.py"],"tests":[],"next_steps":["review"]}, '
        f"open('{result_json_path}', 'w'))"
    )
    return [sys.executable, "-c", script]


def _fake_build_fail(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
    """Build a Python script that exits non-zero."""
    return [sys.executable, "-c", "import sys; sys.exit(1)"]


def _fake_build_sleep(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
    """Build a Python script that sleeps forever."""
    return [sys.executable, "-c", "import time; time.sleep(300)"]


def _wait_workflow_terminal(supervisor, workflow_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        wf = supervisor.get_workflow_status(workflow_id)
        if wf.status in {WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}:
            return wf
        time.sleep(0.2)
    return supervisor.get_workflow_status(workflow_id)


# Decorator-style patch that covers the entire test method
_patch_build = patch(
    "codefleet.supervisor.build_worker_command",
    side_effect=_fake_build,
)
_patch_build_fail = patch(
    "codefleet.supervisor.build_worker_command",
    side_effect=_fake_build_fail,
)
_patch_build_sleep = patch(
    "codefleet.supervisor.build_worker_command",
    side_effect=_fake_build_sleep,
)


# ---------------------------------------------------------------------------
# Stage Schema Validation (common LLM mistakes)
# ---------------------------------------------------------------------------

class TestStageSchemaValidation:
    """Test that common mistakes when passing stage dicts produce clear errors."""

    def test_missing_prompt_template(self, supervisor, git_repo):
        """Omitting prompt_template should fail with a clear error."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="prompt_template"):
            WorkflowEngine(supervisor).create_workflow(
                name="bad", repo_path=str(git_repo), task_prompt="test",
                stages=[{"name": "s0", "executor": "codex"}],
            )

    def test_invalid_executor_value(self, supervisor, git_repo):
        """Wrong executor value like 'Codex' or 'GPT' should fail."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            WorkflowEngine(supervisor).create_workflow(
                name="bad", repo_path=str(git_repo), task_prompt="test",
                stages=[{"name": "s0", "executor": "Codex", "prompt_template": "x"}],
            )

    def test_root_stage_without_worktree_strategy_succeeds(self, supervisor, git_repo):
        """Root stage with default worktree_strategy (now 'new') should work."""
        # This was the #1 source of confusion: default was INHERIT which broke root stages.
        # After fix, omitting worktree_strategy on a root stage should succeed.
        stage = StageDefinition.model_validate({
            "name": "s0", "executor": "codex", "prompt_template": "x",
        })
        assert stage.worktree_strategy == WorktreeStrategy.NEW
        assert stage.depends_on == []
        # Should pass DAG validation
        WorkflowEngine._validate_dag([stage])

    def test_inherit_on_root_stage_rejected(self, supervisor, git_repo):
        """Explicitly using inherit on a root stage should fail validation."""
        with pytest.raises(ValueError, match="INHERIT.*no dependencies"):
            WorkflowEngine(supervisor).create_workflow(
                name="bad", repo_path=str(git_repo), task_prompt="test",
                stages=[{
                    "name": "s0", "executor": "codex", "prompt_template": "x",
                    "worktree_strategy": "inherit", "depends_on": [],
                }],
            )

    def test_incompatible_stage_model_rejected(self):
        with pytest.raises(Exception, match="Unsupported model 'gpt-5.5' for executor 'gemini'"):
            StageDefinition.model_validate({
                "name": "bad-model",
                "executor": "gemini",
                "prompt_template": "x",
                "model": "gpt-5.5",
            })

    def test_stage_model_forwarded_to_create_worker_as_string(self, supervisor, git_repo):
        payload = WorkerStatusPayload(
            worker_id="w_stage_model",
            task_name="wf/review",
            status=WorkerStatus.RUNNING,
            repo_path=str(git_repo),
            branch_name="claude/review/w_stage_model",
            worktree_path=str(git_repo),
            worker_dir=str(git_repo / ".worker"),
            model="claude-opus-4-7",
            executor=ExecutorType.CLAUDE,
            created_at=time.time(),
            timeout_seconds=60,
            prompt_path=str(git_repo / "prompt.txt"),
            result_json_path=str(git_repo / "result.json"),
            stdout_path=str(git_repo / "stdout.log"),
            stderr_path=str(git_repo / "stderr.log"),
            meta_path=str(git_repo / "meta.json"),
        )

        with patch.object(supervisor, "create_worker", return_value=payload) as mock_create:
            WorkflowEngine(supervisor).create_workflow(
                name="wf-stage-model",
                repo_path=str(git_repo),
                task_prompt="Review this",
                stages=[
                    {
                        "name": "review",
                        "executor": "claude",
                        "model": SupportedModel.CLAUDE_OPUS_4_7.value,
                        "prompt_template": "{task_prompt}",
                        "worktree_strategy": "new",
                        "depends_on": [],
                    }
                ],
            )

        assert mock_create.call_args.kwargs["model"] == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# DAG Validation
# ---------------------------------------------------------------------------

class TestDAGValidation:
    def test_empty_stages_rejected(self):
        with pytest.raises(ValueError, match="at least one stage"):
            WorkflowEngine._validate_dag([])

    def test_self_dependency_rejected(self):
        stages = [
            StageDefinition(
                name="self-ref", executor=ExecutorType.CODEX,
                prompt_template="x", depends_on=[0], worktree_strategy=WorktreeStrategy.NEW,
            )
        ]
        with pytest.raises(ValueError, match="cannot depend on itself"):
            WorkflowEngine._validate_dag(stages)

    def test_invalid_index_rejected(self):
        stages = [
            StageDefinition(
                name="bad-dep", executor=ExecutorType.CODEX,
                prompt_template="x", depends_on=[5], worktree_strategy=WorktreeStrategy.NEW,
            )
        ]
        with pytest.raises(ValueError, match="invalid index"):
            WorkflowEngine._validate_dag(stages)

    def test_cycle_detected(self):
        stages = [
            StageDefinition(
                name="a", executor=ExecutorType.CODEX,
                prompt_template="x", depends_on=[1], worktree_strategy=WorktreeStrategy.NEW,
            ),
            StageDefinition(
                name="b", executor=ExecutorType.CODEX,
                prompt_template="x", depends_on=[0], worktree_strategy=WorktreeStrategy.NEW,
            ),
        ]
        with pytest.raises(ValueError, match="cycle"):
            WorkflowEngine._validate_dag(stages)

    def test_inherit_without_deps_rejected(self):
        stages = [
            StageDefinition(
                name="orphan", executor=ExecutorType.CODEX,
                prompt_template="x", depends_on=[],
                worktree_strategy=WorktreeStrategy.INHERIT,
            ),
        ]
        with pytest.raises(ValueError, match="INHERIT.*no dependencies"):
            WorkflowEngine._validate_dag(stages)

    def test_valid_linear_dag(self):
        stages = [
            StageDefinition(
                name="a", executor=ExecutorType.CODEX,
                prompt_template="x", depends_on=[], worktree_strategy=WorktreeStrategy.NEW,
            ),
            StageDefinition(
                name="b", executor=ExecutorType.GEMINI,
                prompt_template="x", depends_on=[0],
            ),
            StageDefinition(
                name="c", executor=ExecutorType.CODEX,
                prompt_template="x", depends_on=[1],
            ),
        ]
        WorkflowEngine._validate_dag(stages)  # Should not raise

    def test_valid_diamond_dag(self):
        stages = [
            StageDefinition(name="root", executor=ExecutorType.CODEX, prompt_template="x", depends_on=[], worktree_strategy=WorktreeStrategy.NEW),
            StageDefinition(name="left", executor=ExecutorType.CODEX, prompt_template="x", depends_on=[0], worktree_strategy=WorktreeStrategy.NEW),
            StageDefinition(name="right", executor=ExecutorType.CODEX, prompt_template="x", depends_on=[0], worktree_strategy=WorktreeStrategy.NEW),
            StageDefinition(name="join", executor=ExecutorType.GEMINI, prompt_template="x", depends_on=[1, 2], worktree_strategy=WorktreeStrategy.NEW),
        ]
        WorkflowEngine._validate_dag(stages)


# ---------------------------------------------------------------------------
# Template Rendering
# ---------------------------------------------------------------------------

class TestPromptRendering:
    def test_render_with_task_prompt(self, supervisor):
        engine = WorkflowEngine(supervisor)
        record = WorkflowRecord(
            workflow_id="wf_test",
            name="test",
            status=WorkflowStatus.RUNNING,
            repo_path="/tmp",
            base_ref="HEAD",
            task_prompt="Add validation",
            stages=[
                StageDefinition(
                    name="impl", executor=ExecutorType.CODEX,
                    prompt_template="Do this: {task_prompt}",
                    worktree_strategy=WorktreeStrategy.NEW,
                ),
            ],
            stage_states={0: StageState()},
            created_at=time.time(),
        )
        rendered = engine._render_prompt(record, 0)
        assert rendered == "Do this: Add validation"

    def test_render_missing_vars_safe(self, supervisor):
        engine = WorkflowEngine(supervisor)
        record = WorkflowRecord(
            workflow_id="wf_test2",
            name="test2",
            status=WorkflowStatus.RUNNING,
            repo_path="/tmp",
            base_ref="HEAD",
            task_prompt="task",
            stages=[
                StageDefinition(
                    name="s", executor=ExecutorType.CODEX,
                    prompt_template="Summary: {stage_0_summary}, Files: {stage_0_files}",
                    worktree_strategy=WorktreeStrategy.NEW,
                ),
            ],
            stage_states={0: StageState()},
            created_at=time.time(),
        )
        rendered = engine._render_prompt(record, 0)
        assert rendered == "Summary: , Files: "


# ---------------------------------------------------------------------------
# Integration: Linear Workflow (3-stage pipeline)
# The patch MUST cover the entire test because stages start asynchronously
# from worker completion callbacks.
# ---------------------------------------------------------------------------

class TestLinearWorkflow:
    @_patch_build
    def test_three_stage_pipeline(self, mock_build, supervisor, git_repo):
        """Stages: implement -> review -> refine, all succeed sequentially."""
        stages = [
            {
                "name": "implement",
                "executor": "codex",
                "prompt_template": "{task_prompt}",
                "worktree_strategy": "new",
                "depends_on": [],
            },
            {
                "name": "review",
                "executor": "gemini",
                "prompt_template": "Review: {stage_0_summary}",
                "worktree_strategy": "inherit",
                "depends_on": [0],
            },
            {
                "name": "refine",
                "executor": "codex",
                "prompt_template": "Refine: {stage_1_summary}",
                "worktree_strategy": "inherit",
                "depends_on": [1],
            },
        ]
        result = supervisor.create_workflow(
            name="linear-test",
            repo_path=str(git_repo),
            task_prompt="Add feature X",
            stages=stages,
            base_ref="HEAD",
        )

        assert result.status == WorkflowStatus.RUNNING
        assert result.workflow_id.startswith("wf_")

        wf = _wait_workflow_terminal(supervisor, result.workflow_id)
        assert wf.status == WorkflowStatus.SUCCEEDED
        assert len(wf.stage_summary) == 3
        assert all(s["status"] == "succeeded" for s in wf.stage_summary)


# ---------------------------------------------------------------------------
# Integration: Parallel Fan-Out + Fan-In
# ---------------------------------------------------------------------------

class TestParallelWorkflow:
    @_patch_build
    def test_fan_out_fan_in(self, mock_build, supervisor, git_repo):
        """Two root stages run in parallel, one downstream waits for both."""
        stages = [
            {
                "name": "module-a",
                "executor": "codex",
                "prompt_template": "Implement module A. {task_prompt}",
                "worktree_strategy": "new",
                "depends_on": [],
            },
            {
                "name": "module-b",
                "executor": "codex",
                "prompt_template": "Implement module B. {task_prompt}",
                "worktree_strategy": "new",
                "depends_on": [],
            },
            {
                "name": "review-all",
                "executor": "gemini",
                "prompt_template": "Review A: {stage_0_summary} B: {stage_1_summary}",
                "worktree_strategy": "new",
                "depends_on": [0, 1],
            },
        ]
        result = supervisor.create_workflow(
            name="fanout-test",
            repo_path=str(git_repo),
            task_prompt="Refactor auth",
            stages=stages,
            base_ref="HEAD",
        )

        wf = _wait_workflow_terminal(supervisor, result.workflow_id)
        assert wf.status == WorkflowStatus.SUCCEEDED
        assert len(wf.stage_summary) == 3


# ---------------------------------------------------------------------------
# Integration: Failure Propagation
# ---------------------------------------------------------------------------

class TestFailurePropagation:
    @_patch_build_fail
    def test_stage_failure_stops_workflow(self, mock_build, supervisor, git_repo):
        """If stage 0 fails, stage 1 should never start and workflow fails."""
        stages = [
            {
                "name": "fail-stage",
                "executor": "codex",
                "prompt_template": "This will fail",
                "worktree_strategy": "new",
                "depends_on": [],
            },
            {
                "name": "never-runs",
                "executor": "gemini",
                "prompt_template": "Should not run",
                "worktree_strategy": "new",
                "depends_on": [0],
            },
        ]
        result = supervisor.create_workflow(
            name="fail-test",
            repo_path=str(git_repo),
            task_prompt="Will fail",
            stages=stages,
            base_ref="HEAD",
        )

        wf = _wait_workflow_terminal(supervisor, result.workflow_id)
        assert wf.status == WorkflowStatus.FAILED
        # Stage 1 should still be pending (never started)
        assert wf.stage_summary[1]["status"] in ("pending", "cancelled")


# ---------------------------------------------------------------------------
# Cancel Workflow
# ---------------------------------------------------------------------------

class TestCancelWorkflow:
    @_patch_build_sleep
    def test_cancel_running_workflow(self, mock_build, supervisor, git_repo):
        """Cancel a workflow with a long-running stage."""
        stages = [
            {
                "name": "slow-stage",
                "executor": "codex",
                "prompt_template": "Sleep forever",
                "worktree_strategy": "new",
                "depends_on": [],
            },
        ]
        result = supervisor.create_workflow(
            name="cancel-test",
            repo_path=str(git_repo),
            task_prompt="Slow task",
            stages=stages,
            base_ref="HEAD",
        )

        time.sleep(0.5)
        wf = supervisor.cancel_workflow(result.workflow_id)
        assert wf.status == WorkflowStatus.CANCELLED

    @_patch_build
    def test_cancel_terminal_workflow_fails(self, mock_build, supervisor, git_repo):
        stages = [
            {
                "name": "quick",
                "executor": "codex",
                "prompt_template": "x",
                "worktree_strategy": "new",
                "depends_on": [],
            },
        ]
        result = supervisor.create_workflow(
            name="done-wf",
            repo_path=str(git_repo),
            task_prompt="Quick",
            stages=stages,
        )

        _wait_workflow_terminal(supervisor, result.workflow_id)
        with pytest.raises(ValueError, match="already terminal"):
            supervisor.cancel_workflow(result.workflow_id)


# ---------------------------------------------------------------------------
# Collect Workflow Result
# ---------------------------------------------------------------------------

class TestCollectWorkflowResult:
    @_patch_build
    def test_collect_final_result(self, mock_build, supervisor, git_repo):
        stages = [
            {
                "name": "only-stage",
                "executor": "codex",
                "prompt_template": "{task_prompt}",
                "worktree_strategy": "new",
                "depends_on": [],
            },
        ]
        result = supervisor.create_workflow(
            name="collect-test",
            repo_path=str(git_repo),
            task_prompt="Test",
            stages=stages,
        )

        _wait_workflow_terminal(supervisor, result.workflow_id)
        collected = supervisor.collect_workflow_result(workflow_id=result.workflow_id)
        assert collected["status"] == "succeeded"
        assert collected["final_result"] is not None
        assert collected["final_result"]["result"]["summary"] == "stage done"

    @_patch_build
    def test_collect_all_stages(self, mock_build, supervisor, git_repo):
        stages = [
            {
                "name": "s0",
                "executor": "codex",
                "prompt_template": "{task_prompt}",
                "worktree_strategy": "new",
                "depends_on": [],
            },
        ]
        result = supervisor.create_workflow(
            name="collect-all-test",
            repo_path=str(git_repo),
            task_prompt="Test",
            stages=stages,
        )

        _wait_workflow_terminal(supervisor, result.workflow_id)
        collected = supervisor.collect_workflow_result(
            workflow_id=result.workflow_id, include_all_stages=True
        )
        assert "stage_results" in collected
        assert len(collected["stage_results"]) == 1
        assert collected["stage_results"][0]["name"] == "s0"

    def test_collect_nonexistent_workflow(self, supervisor):
        with pytest.raises(ValueError, match="Workflow not found"):
            supervisor.collect_workflow_result(workflow_id="wf_nonexistent")


# ---------------------------------------------------------------------------
# List Workflows
# ---------------------------------------------------------------------------

class TestListWorkflows:
    def test_list_empty(self, supervisor):
        results = supervisor.list_workflows()
        assert results == []

    @_patch_build
    def test_list_after_create(self, mock_build, supervisor, git_repo):
        stages = [
            {
                "name": "s0",
                "executor": "codex",
                "prompt_template": "x",
                "worktree_strategy": "new",
                "depends_on": [],
            },
        ]
        supervisor.create_workflow(
            name="list-test",
            repo_path=str(git_repo),
            task_prompt="Test",
            stages=stages,
        )

        results = supervisor.list_workflows()
        assert len(results) == 1
        assert results[0].name == "list-test"


# ---------------------------------------------------------------------------
# Cleanup Workflow
# ---------------------------------------------------------------------------

class TestCleanupWorkflow:
    @_patch_build
    def test_cleanup_completed_workflow(self, mock_build, supervisor, git_repo):
        stages = [
            {
                "name": "impl",
                "executor": "codex",
                "prompt_template": "{task_prompt}",
                "worktree_strategy": "new",
                "depends_on": [],
            },
        ]
        result = supervisor.create_workflow(
            name="cleanup-test",
            repo_path=str(git_repo),
            task_prompt="Test",
            stages=stages,
        )

        _wait_workflow_terminal(supervisor, result.workflow_id)
        cleanup = supervisor.cleanup_workflow(result.workflow_id)
        assert cleanup["stages_cleaned"] == 1
        assert cleanup["errors"] == []

    @_patch_build_sleep
    def test_cleanup_running_workflow_fails(self, mock_build, supervisor, git_repo):
        stages = [
            {
                "name": "slow",
                "executor": "codex",
                "prompt_template": "x",
                "worktree_strategy": "new",
                "depends_on": [],
            },
        ]
        result = supervisor.create_workflow(
            name="cleanup-running",
            repo_path=str(git_repo),
            task_prompt="x",
            stages=stages,
        )

        time.sleep(0.3)
        with pytest.raises(ValueError, match="Cannot cleanup non-terminal"):
            supervisor.cleanup_workflow(result.workflow_id)

        # Cleanup for test teardown
        supervisor.cancel_workflow(result.workflow_id)


# ---------------------------------------------------------------------------
# Stage isolation: INHERIT gives a fresh worktree off the parent's branch
# ---------------------------------------------------------------------------


class TestInheritIsolation:
    @_patch_build
    def test_inherit_creates_own_worktree_and_branch(
        self, mock_build, supervisor, git_repo
    ):
        """INHERIT stages must not share .codefleet/ or .git/index with parent."""
        stages = [
            {
                "name": "parent",
                "executor": "codex",
                "prompt_template": "{task_prompt}",
                "worktree_strategy": "new",
                "depends_on": [],
            },
            {
                "name": "child",
                "executor": "codex",
                "prompt_template": "Refine: {stage_0_summary}",
                "worktree_strategy": "inherit",
                "depends_on": [0],
            },
        ]
        result = supervisor.create_workflow(
            name="inherit-isolation",
            repo_path=str(git_repo),
            task_prompt="x",
            stages=stages,
        )
        wf = _wait_workflow_terminal(supervisor, result.workflow_id)
        assert wf.status == WorkflowStatus.SUCCEEDED

        # Fetch workers via the store to inspect worktree + branch paths
        parent_wid = wf.stage_summary[0]["worker_id"]
        child_wid = wf.stage_summary[1]["worker_id"]
        parent_rec = supervisor.store.get_worker(parent_wid)
        child_rec = supervisor.store.get_worker(child_wid)
        assert parent_rec.worktree_path != child_rec.worktree_path
        assert parent_rec.branch_name != child_rec.branch_name


# ---------------------------------------------------------------------------
# Silent-failure detection: agent claims completed but produced no work
# ---------------------------------------------------------------------------


def _fake_build_empty_success(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
    """Write a result.json claiming completed with no commits and no files."""
    script = (
        "import json; "
        "json.dump("
        '{"summary":"I did nothing","status":"completed",'
        '"files_changed":[],"commits":[],"tests":[],"next_steps":[]}, '
        f"open('{result_json_path}', 'w'))"
    )
    return [sys.executable, "-c", script]


def _fake_build_honestly_blocked(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
    """Write a result.json with status=blocked — the honest failure path."""
    script = (
        "import json; "
        "json.dump("
        '{"summary":"sandbox denied git access","status":"blocked",'
        '"files_changed":[],"commits":[],"tests":[],"next_steps":[]}, '
        f"open('{result_json_path}', 'w'))"
    )
    return [sys.executable, "-c", script]


class TestSilentFailureDetection:
    def test_empty_completed_is_flagged_as_failed(self, supervisor, git_repo):
        with patch(
            "codefleet.supervisor.build_worker_command",
            side_effect=_fake_build_empty_success,
        ):
            stages = [
                {
                    "name": "silent",
                    "executor": "codex",
                    "prompt_template": "{task_prompt}",
                    "worktree_strategy": "new",
                    "depends_on": [],
                },
            ]
            result = supervisor.create_workflow(
                name="silent-fail",
                repo_path=str(git_repo),
                task_prompt="x",
                stages=stages,
            )
            wf = _wait_workflow_terminal(supervisor, result.workflow_id)

        assert wf.status == WorkflowStatus.FAILED
        worker_id = wf.stage_summary[0]["worker_id"]
        rec = supervisor.store.get_worker(worker_id)
        assert rec.status == WorkerStatus.FAILED
        assert "silent failure" in (rec.error_message or "").lower()

    def test_blocked_status_is_flagged_as_failed(self, supervisor, git_repo):
        with patch(
            "codefleet.supervisor.build_worker_command",
            side_effect=_fake_build_honestly_blocked,
        ):
            stages = [
                {
                    "name": "blocked",
                    "executor": "codex",
                    "prompt_template": "{task_prompt}",
                    "worktree_strategy": "new",
                    "depends_on": [],
                },
            ]
            result = supervisor.create_workflow(
                name="blocked-honest",
                repo_path=str(git_repo),
                task_prompt="x",
                stages=stages,
            )
            wf = _wait_workflow_terminal(supervisor, result.workflow_id)

        assert wf.status == WorkflowStatus.FAILED
        worker_id = wf.stage_summary[0]["worker_id"]
        rec = supervisor.store.get_worker(worker_id)
        assert rec.status == WorkerStatus.FAILED
        assert "blocked" in (rec.error_message or "").lower()
        assert "sandbox denied" in (rec.error_message or "").lower()
