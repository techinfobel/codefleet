"""Tests for store.py - SQLite worker store with real database operations."""

import json
import time

import pytest

from codefleet.models import (
    ExecutorType,
    StageDefinition,
    StageState,
    WorkerRecord,
    WorkerStatus,
    WorkflowRecord,
    WorkflowStatus,
)
from codefleet.store import WorkerStore


def _make_record(worker_id="w_test001", **overrides):
    defaults = dict(
        worker_id=worker_id,
        task_name="test task",
        repo_path="/tmp/repo",
        branch_name=f"codex/test/{worker_id}",
        worktree_path=f"/tmp/fleet/workers/{worker_id}/worktree",
        worker_dir=f"/tmp/fleet/workers/{worker_id}",
        model="gpt-5.4",
        status=WorkerStatus.PENDING,
        created_at=time.time(),
        timeout_seconds=600,
        codex_command='["codex", "exec", "test"]',
        prompt="Do something useful",
        result_json_path=f"/tmp/fleet/workers/{worker_id}/result.json",
        stdout_path=f"/tmp/fleet/workers/{worker_id}/stdout.log",
        stderr_path=f"/tmp/fleet/workers/{worker_id}/stderr.log",
        prompt_path=f"/tmp/fleet/workers/{worker_id}/prompt.txt",
        meta_path=f"/tmp/fleet/workers/{worker_id}/meta.json",
    )
    defaults.update(overrides)
    return WorkerRecord(**defaults)


class TestWorkerStore:
    def test_insert_and_get(self, store):
        record = _make_record()
        store.insert_worker(record)
        retrieved = store.get_worker("w_test001")
        assert retrieved is not None
        assert retrieved.worker_id == "w_test001"
        assert retrieved.task_name == "test task"
        assert retrieved.status == WorkerStatus.PENDING
        assert retrieved.model == "gpt-5.4"
        assert retrieved.prompt == "Do something useful"

    def test_get_nonexistent(self, store):
        result = store.get_worker("w_nonexistent")
        assert result is None

    def test_update_status(self, store):
        record = _make_record()
        store.insert_worker(record)
        store.update_worker("w_test001", status=WorkerStatus.RUNNING, pid=12345)
        updated = store.get_worker("w_test001")
        assert updated.status == WorkerStatus.RUNNING
        assert updated.pid == 12345

    def test_update_multiple_fields(self, store):
        record = _make_record()
        store.insert_worker(record)
        now = time.time()
        store.update_worker(
            "w_test001",
            status=WorkerStatus.SUCCEEDED,
            started_at=now - 10,
            ended_at=now,
            exit_code=0,
        )
        updated = store.get_worker("w_test001")
        assert updated.status == WorkerStatus.SUCCEEDED
        assert updated.exit_code == 0
        assert updated.ended_at is not None

    def test_update_tags(self, store):
        record = _make_record(tags=["initial"])
        store.insert_worker(record)
        store.update_worker("w_test001", tags=["updated", "tags"])
        updated = store.get_worker("w_test001")
        assert updated.tags == ["updated", "tags"]

    def test_update_metadata(self, store):
        record = _make_record(metadata={"key": "old"})
        store.insert_worker(record)
        store.update_worker("w_test001", metadata={"key": "new", "extra": 42})
        updated = store.get_worker("w_test001")
        assert updated.metadata == {"key": "new", "extra": 42}

    def test_update_error_message(self, store):
        record = _make_record()
        store.insert_worker(record)
        store.update_worker(
            "w_test001",
            status=WorkerStatus.FAILED,
            error_message="Something went wrong",
        )
        updated = store.get_worker("w_test001")
        assert updated.error_message == "Something went wrong"

    def test_list_all_workers(self, store):
        for i in range(5):
            store.insert_worker(
                _make_record(
                    worker_id=f"w_list{i:03d}",
                    created_at=time.time() + i,
                )
            )
        workers = store.list_workers()
        assert len(workers) == 5
        # Should be ordered by created_at DESC
        assert workers[0].worker_id == "w_list004"
        assert workers[4].worker_id == "w_list000"

    def test_list_with_limit(self, store):
        for i in range(10):
            store.insert_worker(
                _make_record(
                    worker_id=f"w_lim{i:03d}",
                    created_at=time.time() + i,
                )
            )
        workers = store.list_workers(limit=3)
        assert len(workers) == 3

    def test_list_with_status_filter(self, store):
        store.insert_worker(
            _make_record(worker_id="w_run1", status=WorkerStatus.RUNNING)
        )
        store.insert_worker(
            _make_record(worker_id="w_done1", status=WorkerStatus.SUCCEEDED)
        )
        store.insert_worker(
            _make_record(worker_id="w_fail1", status=WorkerStatus.FAILED)
        )
        store.insert_worker(
            _make_record(worker_id="w_run2", status=WorkerStatus.RUNNING)
        )

        running = store.list_workers(statuses=["running"])
        assert len(running) == 2
        assert all(w.status == WorkerStatus.RUNNING for w in running)

        terminal = store.list_workers(statuses=["succeeded", "failed"])
        assert len(terminal) == 2

    def test_list_empty_store(self, store):
        workers = store.list_workers()
        assert workers == []

    def test_insert_with_optional_fields(self, store):
        record = _make_record(
            profile="reviewer",
            parent_worker_id="w_parent",
            tags=["backend", "urgent"],
            metadata={"priority": 1, "team": "core"},
        )
        store.insert_worker(record)
        retrieved = store.get_worker("w_test001")
        assert retrieved.profile == "reviewer"
        assert retrieved.parent_worker_id == "w_parent"
        assert retrieved.tags == ["backend", "urgent"]
        assert retrieved.metadata == {"priority": 1, "team": "core"}

    def test_insert_duplicate_fails(self, store):
        record = _make_record()
        store.insert_worker(record)
        with pytest.raises(Exception):
            store.insert_worker(record)

    def test_persistence_across_connections(self, db_path):
        """Verify data survives closing and reopening the store."""
        store1 = WorkerStore(db_path)
        store1.insert_worker(_make_record(worker_id="w_persist"))
        store1.close()

        store2 = WorkerStore(db_path)
        retrieved = store2.get_worker("w_persist")
        assert retrieved is not None
        assert retrieved.worker_id == "w_persist"
        store2.close()

    def test_db_directory_creation(self, tmp_path):
        """Store should create parent directories for the db file."""
        deep_path = tmp_path / "a" / "b" / "c" / "fleet.db"
        store = WorkerStore(deep_path)
        store.insert_worker(_make_record())
        assert deep_path.exists()
        store.close()

    def test_full_lifecycle(self, store):
        """Test a complete worker lifecycle through the store."""
        now = time.time()
        record = _make_record(worker_id="w_lifecycle", created_at=now)
        store.insert_worker(record)

        # pending -> running
        store.update_worker(
            "w_lifecycle",
            status=WorkerStatus.RUNNING,
            started_at=now + 1,
            pid=9999,
        )
        r = store.get_worker("w_lifecycle")
        assert r.status == WorkerStatus.RUNNING
        assert r.pid == 9999

        # running -> succeeded
        store.update_worker(
            "w_lifecycle",
            status=WorkerStatus.SUCCEEDED,
            ended_at=now + 60,
            exit_code=0,
        )
        r = store.get_worker("w_lifecycle")
        assert r.status == WorkerStatus.SUCCEEDED
        assert r.exit_code == 0
        assert r.ended_at > r.started_at


class TestExecutorField:
    def test_default_executor(self, store):
        record = _make_record(worker_id="w_exec_default")
        store.insert_worker(record)
        retrieved = store.get_worker("w_exec_default")
        assert retrieved.executor == ExecutorType.CODEX

    def test_gemini_executor(self, store):
        record = _make_record(worker_id="w_exec_gemini", executor=ExecutorType.GEMINI)
        store.insert_worker(record)
        retrieved = store.get_worker("w_exec_gemini")
        assert retrieved.executor == ExecutorType.GEMINI

    def test_workflow_fields(self, store):
        record = _make_record(
            worker_id="w_wf_test",
            workflow_id="wf_abc",
            stage_index=2,
        )
        store.insert_worker(record)
        retrieved = store.get_worker("w_wf_test")
        assert retrieved.workflow_id == "wf_abc"
        assert retrieved.stage_index == 2


class TestWorkflowStore:
    def _make_workflow(self, workflow_id="wf_test001", **overrides):
        defaults = dict(
            workflow_id=workflow_id,
            name="test workflow",
            status=WorkflowStatus.PENDING,
            repo_path="/tmp/repo",
            base_ref="HEAD",
            task_prompt="Do something",
            stages=[
                StageDefinition(
                    name="s0",
                    executor=ExecutorType.CODEX,
                    prompt_template="{task_prompt}",
                ),
            ],
            stage_states={0: StageState()},
            created_at=time.time(),
        )
        defaults.update(overrides)
        return WorkflowRecord(**defaults)

    def test_insert_and_get(self, store):
        record = self._make_workflow()
        store.insert_workflow(record)
        retrieved = store.get_workflow("wf_test001")
        assert retrieved is not None
        assert retrieved.workflow_id == "wf_test001"
        assert retrieved.name == "test workflow"
        assert retrieved.status == WorkflowStatus.PENDING
        assert len(retrieved.stages) == 1
        assert retrieved.stages[0].name == "s0"

    def test_get_nonexistent(self, store):
        assert store.get_workflow("wf_nope") is None

    def test_update_status(self, store):
        store.insert_workflow(self._make_workflow())
        store.update_workflow("wf_test001", status=WorkflowStatus.RUNNING)
        retrieved = store.get_workflow("wf_test001")
        assert retrieved.status == WorkflowStatus.RUNNING

    def test_update_stage_states(self, store):
        store.insert_workflow(self._make_workflow())
        new_states = {0: StageState(worker_id="w_abc", status=WorkerStatus.SUCCEEDED)}
        store.update_workflow("wf_test001", stage_states=new_states)
        retrieved = store.get_workflow("wf_test001")
        assert retrieved.stage_states[0].worker_id == "w_abc"
        assert retrieved.stage_states[0].status == WorkerStatus.SUCCEEDED

    def test_list_workflows(self, store):
        for i in range(3):
            store.insert_workflow(
                self._make_workflow(
                    workflow_id=f"wf_list{i:03d}",
                    created_at=time.time() + i,
                )
            )
        workflows = store.list_workflows()
        assert len(workflows) == 3
        assert workflows[0].workflow_id == "wf_list002"

    def test_list_with_status_filter(self, store):
        store.insert_workflow(
            self._make_workflow(workflow_id="wf_run", status=WorkflowStatus.RUNNING)
        )
        store.insert_workflow(
            self._make_workflow(workflow_id="wf_done", status=WorkflowStatus.SUCCEEDED)
        )
        running = store.list_workflows(statuses=["running"])
        assert len(running) == 1
        assert running[0].workflow_id == "wf_run"

    def test_multi_stage_roundtrip(self, store):
        stages = [
            StageDefinition(name="impl", executor=ExecutorType.CODEX, prompt_template="{task_prompt}"),
            StageDefinition(name="review", executor=ExecutorType.GEMINI, prompt_template="Review: {stage_0_summary}", depends_on=[0]),
        ]
        record = self._make_workflow(
            workflow_id="wf_multi",
            stages=stages,
            stage_states={
                0: StageState(worker_id="w_1", status=WorkerStatus.SUCCEEDED),
                1: StageState(),
            },
        )
        store.insert_workflow(record)
        retrieved = store.get_workflow("wf_multi")
        assert len(retrieved.stages) == 2
        assert retrieved.stages[1].executor == ExecutorType.GEMINI
        assert retrieved.stages[1].depends_on == [0]
        assert retrieved.stage_states[0].worker_id == "w_1"
