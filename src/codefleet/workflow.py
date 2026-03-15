from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .models import (
    StageDefinition,
    StageState,
    WorkerStatus,
    WorkflowRecord,
    WorkflowStatus,
    WorkflowStatusPayload,
    WorktreeStrategy,
)
from .result_schema import parse_result_file

if TYPE_CHECKING:
    from .supervisor import FleetSupervisor


class _SafeDict(dict):
    """dict subclass that returns '' for missing keys in str.format_map()."""

    def __missing__(self, key: str) -> str:
        return ""


class WorkflowEngine:
    def __init__(self, supervisor: FleetSupervisor):
        self.supervisor = supervisor
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_workflow(
        self,
        name: str,
        repo_path: str,
        task_prompt: str,
        stages: list[dict],
        base_ref: str = "HEAD",
        timeout_seconds: Optional[int] = None,
    ) -> WorkflowStatusPayload:
        stage_defs = [StageDefinition.model_validate(s) for s in stages]
        self._validate_dag(stage_defs)

        workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
        now = time.time()

        stage_states: dict[int, StageState] = {
            i: StageState() for i in range(len(stage_defs))
        }

        if timeout_seconds is not None:
            for sd in stage_defs:
                if sd.timeout_seconds is None:
                    sd.timeout_seconds = timeout_seconds

        record = WorkflowRecord(
            workflow_id=workflow_id,
            name=name,
            status=WorkflowStatus.PENDING,
            repo_path=repo_path,
            base_ref=base_ref,
            task_prompt=task_prompt,
            stages=stage_defs,
            stage_states=stage_states,
            created_at=now,
        )

        self.supervisor.store.insert_workflow(record)

        root_indices = [i for i, s in enumerate(stage_defs) if not s.depends_on]
        with self._lock:
            for idx in root_indices:
                self._start_stage(workflow_id, idx)

            self.supervisor.store.update_workflow(
                workflow_id, status=WorkflowStatus.RUNNING
            )
            record.status = WorkflowStatus.RUNNING

        return WorkflowStatusPayload.from_record(record)

    def get_workflow_status(self, workflow_id: str) -> WorkflowStatusPayload:
        record = self.supervisor.store.get_workflow(workflow_id)
        if record is None:
            raise ValueError(f"Workflow not found: {workflow_id}")
        return WorkflowStatusPayload.from_record(record)

    def list_workflows(
        self,
        statuses: Optional[list[str]] = None,
        limit: int = 25,
    ) -> list[WorkflowStatusPayload]:
        records = self.supervisor.store.list_workflows(statuses=statuses, limit=limit)
        return [WorkflowStatusPayload.from_record(r) for r in records]

    def cancel_workflow(self, workflow_id: str) -> WorkflowStatusPayload:
        record = self.supervisor.store.get_workflow(workflow_id)
        if record is None:
            raise ValueError(f"Workflow not found: {workflow_id}")
        if record.status in {WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}:
            raise ValueError(f"Workflow {workflow_id} is already terminal ({record.status.value})")

        for idx, state in record.stage_states.items():
            if not state.status.is_terminal and state.worker_id:
                try:
                    self.supervisor.cancel_worker(state.worker_id)
                except (ValueError, RuntimeError):
                    pass
                state.status = WorkerStatus.CANCELLED

        self.supervisor.store.update_workflow(
            workflow_id,
            status=WorkflowStatus.CANCELLED,
            stage_states=record.stage_states,
            completed_at=time.time(),
            error_message="Cancelled by user",
        )

        record.status = WorkflowStatus.CANCELLED
        record.completed_at = time.time()
        record.error_message = "Cancelled by user"
        return WorkflowStatusPayload.from_record(record)

    def collect_workflow_result(
        self,
        workflow_id: str,
        include_all_stages: bool = False,
        include_logs: bool = False,
    ) -> dict:
        record = self.supervisor.store.get_workflow(workflow_id)
        if record is None:
            raise ValueError(f"Workflow not found: {workflow_id}")

        payload = WorkflowStatusPayload.from_record(record).model_dump()

        if include_all_stages:
            stage_results = []
            for i, stage in enumerate(record.stages):
                state = record.stage_states.get(i, StageState())
                stage_data = {"index": i, "name": stage.name, "status": state.status.value}
                if state.worker_id:
                    stage_data["result"] = self._collect_worker_result(
                        state.worker_id, include_logs
                    )
                stage_results.append(stage_data)
            payload["stage_results"] = stage_results
        else:
            last_idx = len(record.stages) - 1
            last_state = record.stage_states.get(last_idx, StageState())
            payload["final_result"] = self._collect_worker_result(
                last_state.worker_id, include_logs
            )

        return payload

    def cleanup_workflow(self, workflow_id: str) -> dict:
        record = self.supervisor.store.get_workflow(workflow_id)
        if record is None:
            raise ValueError(f"Workflow not found: {workflow_id}")
        if record.status not in {WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}:
            raise ValueError(
                f"Cannot cleanup non-terminal workflow {workflow_id} "
                f"(status: {record.status.value})"
            )

        cleanup_summary = {
            "workflow_id": workflow_id,
            "stages_cleaned": 0,
            "errors": [],
        }

        for i, state in record.stage_states.items():
            if state.worker_id:
                try:
                    stage_def = record.stages[i]
                    remove_branch = stage_def.worktree_strategy == WorktreeStrategy.NEW
                    self.supervisor.cleanup_worker(
                        state.worker_id,
                        remove_branch=remove_branch,
                        remove_worktree_dir=True,
                    )
                    cleanup_summary["stages_cleaned"] += 1
                except (ValueError, RuntimeError) as e:
                    cleanup_summary["errors"].append(f"Stage {i}: {e}")

        return cleanup_summary

    def _collect_worker_result(
        self, worker_id: Optional[str], include_logs: bool
    ) -> Optional[dict]:
        if not worker_id:
            return None
        try:
            return self.supervisor.collect_worker_result(
                worker_id, include_logs=include_logs
            )
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Callback from supervisor when a workflow worker completes
    # ------------------------------------------------------------------

    def on_stage_complete(self, worker_id: str, workflow_id: str, stage_index: int):
        with self._lock:
            self._on_stage_complete_locked(worker_id, workflow_id, stage_index)

    def _on_stage_complete_locked(self, worker_id: str, workflow_id: str, stage_index: int):
        record = self.supervisor.store.get_workflow(workflow_id)
        if record is None:
            return

        worker_record = self.supervisor.store.get_worker(worker_id)
        if worker_record is None:
            return

        state = record.stage_states.get(stage_index, StageState())
        state.status = worker_record.status
        record.stage_states[stage_index] = state

        if worker_record.status in {WorkerStatus.FAILED, WorkerStatus.TIMED_OUT, WorkerStatus.CANCELLED}:
            for idx, s in record.stage_states.items():
                if idx != stage_index and not s.status.is_terminal and s.worker_id:
                    try:
                        self.supervisor.cancel_worker(s.worker_id)
                    except (ValueError, RuntimeError):
                        pass
                    s.status = WorkerStatus.CANCELLED

            self.supervisor.store.update_workflow(
                workflow_id,
                status=WorkflowStatus.FAILED,
                stage_states=record.stage_states,
                completed_at=time.time(),
                error_message=f"Stage {stage_index} ({record.stages[stage_index].name}) "
                              f"{worker_record.status.value}: {worker_record.error_message or ''}",
            )
            return

        self.supervisor.store.update_workflow(
            workflow_id, stage_states=record.stage_states
        )

        for i, stage_def in enumerate(record.stages):
            state = record.stage_states.get(i)
            if state is None or state.status != WorkerStatus.PENDING:
                continue
            if stage_index not in stage_def.depends_on:
                continue

            all_deps_done = all(
                record.stage_states.get(dep, StageState()).status == WorkerStatus.SUCCEEDED
                for dep in stage_def.depends_on
            )
            if all_deps_done:
                self._start_stage(workflow_id, i)

        record = self.supervisor.store.get_workflow(workflow_id)
        statuses = [
            record.stage_states.get(i, StageState()).status
            for i in range(len(record.stages))
        ]
        if all(s.is_terminal for s in statuses):
            final_status = (
                WorkflowStatus.SUCCEEDED
                if all(s == WorkerStatus.SUCCEEDED for s in statuses)
                else WorkflowStatus.FAILED
            )
            self.supervisor.store.update_workflow(
                workflow_id,
                status=final_status,
                completed_at=time.time(),
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_stage(self, workflow_id: str, stage_index: int) -> str:
        record = self.supervisor.store.get_workflow(workflow_id)
        if record is None:
            raise ValueError(f"Workflow not found: {workflow_id}")
        stage_def = record.stages[stage_index]

        rendered_prompt = self._render_prompt(record, stage_index)

        existing_worktree_path = None
        existing_branch_name = None
        if stage_def.worktree_strategy == WorktreeStrategy.INHERIT and stage_def.depends_on:
            first_dep = stage_def.depends_on[0]
            dep_state = record.stage_states.get(first_dep, StageState())
            if dep_state.worker_id:
                dep_worker = self.supervisor.store.get_worker(dep_state.worker_id)
                if dep_worker:
                    existing_worktree_path = dep_worker.worktree_path
                    existing_branch_name = dep_worker.branch_name

        payload = self.supervisor.create_worker(
            repo_path=record.repo_path,
            task_name=f"{record.name}/{stage_def.name}",
            prompt=rendered_prompt,
            base_ref=record.base_ref,
            model=stage_def.model,
            executor=stage_def.executor.value,
            reasoning_effort=stage_def.reasoning_effort,
            timeout_seconds=stage_def.timeout_seconds,
            extra_args=stage_def.extra_args,
            workflow_id=workflow_id,
            stage_index=stage_index,
            existing_worktree_path=existing_worktree_path,
            existing_branch_name=existing_branch_name,
        )

        state = record.stage_states.get(stage_index, StageState())
        state.worker_id = payload.worker_id
        state.status = WorkerStatus.RUNNING
        state.worktree_path = payload.worktree_path
        record.stage_states[stage_index] = state
        self.supervisor.store.update_workflow(
            workflow_id, stage_states=record.stage_states
        )

        return payload.worker_id

    def _render_prompt(self, workflow: WorkflowRecord, stage_index: int) -> str:
        template = workflow.stages[stage_index].prompt_template
        variables = _SafeDict(task_prompt=workflow.task_prompt)

        worker_ids = [
            workflow.stage_states[i].worker_id
            for i in range(len(workflow.stages))
            if workflow.stage_states.get(i, StageState()).status == WorkerStatus.SUCCEEDED
            and workflow.stage_states.get(i, StageState()).worker_id
        ]

        workers = {w.worker_id: w for w in self.supervisor.store.get_workers(worker_ids)} if worker_ids else {}

        for i, stage_def in enumerate(workflow.stages):
            state = workflow.stage_states.get(i, StageState())
            if state.status != WorkerStatus.SUCCEEDED or not state.worker_id:
                continue

            worker = workers.get(state.worker_id)
            if worker is None:
                continue

            result_path = Path(worker.result_json_path)
            if not result_path.exists():
                continue

            try:
                result = parse_result_file(result_path)
                variables.update({
                    f"stage_{i}_summary": result.summary,
                    f"stage_{i}_files": ", ".join(result.files_changed),
                    f"stage_{i}_result": result_path.read_text(encoding="utf-8"),
                    f"stage_{i}_next_steps": "\n".join(result.next_steps),
                    f"stage_{i}_status": result.status.value,
                })
            except Exception:
                pass

        return template.format_map(variables)

    @staticmethod
    def _validate_dag(stages: list[StageDefinition]) -> None:
        n = len(stages)
        if n == 0:
            raise ValueError("Workflow must have at least one stage")

        for i, stage in enumerate(stages):
            for dep in stage.depends_on:
                if dep < 0 or dep >= n:
                    raise ValueError(
                        f"Stage {i} ({stage.name}) depends on invalid index {dep}"
                    )
                if dep == i:
                    raise ValueError(
                        f"Stage {i} ({stage.name}) cannot depend on itself"
                    )

        in_degree = [0] * n
        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        for i, stage in enumerate(stages):
            for dep in stage.depends_on:
                adj[dep].append(i)
                in_degree[i] += 1

        queue = deque(i for i in range(n) if in_degree[i] == 0)
        if not queue:
            raise ValueError("Workflow DAG has no root stages (cycle detected)")

        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited != n:
            raise ValueError("Workflow DAG contains a cycle")

        for i, stage in enumerate(stages):
            if stage.worktree_strategy == WorktreeStrategy.INHERIT and not stage.depends_on:
                raise ValueError(
                    f"Stage {i} ({stage.name}) uses INHERIT strategy but has no dependencies"
                )
