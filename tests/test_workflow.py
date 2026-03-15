"""Tests for workflow.py - WorkflowEngine: DAG validation, stage advancement,
template rendering, error propagation, fan-out/fan-in, cleanup."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_fleet_supervisor.models import (
    ExecutorType,
    StageDefinition,
    StageState,
    WorkerStatus,
    WorkflowRecord,
    WorkflowStatus,
    WorkflowStatusPayload,
    WorktreeStrategy,
)
from agent_fleet_supervisor.supervisor import FleetSupervisor
from agent_fleet_supervisor.workflow import WorkflowEngine, _SafeDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_build(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None):
    """Build a Python script that writes a valid result.json and exits."""
    script = (
        "import json; "
        "json.dump("
        '{"summary":"stage done","status":"completed",'
        '"files_changed":["a.py"],"tests":[],"next_steps":["review"]}, '
        f"open('{result_json_path}', 'w'))"
    )
    return [sys.executable, "-c", script]


def _fake_build_fail(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None):
    """Build a Python script that exits non-zero."""
    return [sys.executable, "-c", "import sys; sys.exit(1)"]


def _fake_build_sleep(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None):
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
    "agent_fleet_supervisor.supervisor.build_worker_command",
    side_effect=_fake_build,
)
_patch_build_fail = patch(
    "agent_fleet_supervisor.supervisor.build_worker_command",
    side_effect=_fake_build_fail,
)
_patch_build_sleep = patch(
    "agent_fleet_supervisor.supervisor.build_worker_command",
    side_effect=_fake_build_sleep,
)


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
# SafeDict
# ---------------------------------------------------------------------------

class TestSafeDict:
    def test_missing_key_returns_empty(self):
        d = _SafeDict(a="hello")
        assert "{a} {b}".format_map(d) == "hello "

    def test_present_key_works(self):
        d = _SafeDict(x="val")
        assert d["x"] == "val"


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
