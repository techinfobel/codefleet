from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
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
    parse_result_file,
)

if TYPE_CHECKING:
    from .supervisor import FleetSupervisor

logger = logging.getLogger(__name__)

_TERMINAL_WORKFLOW_STATUSES = {
    WorkflowStatus.SUCCEEDED,
    WorkflowStatus.FAILED,
    WorkflowStatus.CANCELLED,
}


class WorkflowStartAborted(Exception):
    """Raised internally when a stage should no longer be launched."""


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
        if not self._lock.acquire(timeout=60):
            raise RuntimeError(
                "Timed out waiting for workflow engine lock — another "
                "workflow operation may be in progress"
            )
        try:
            for idx in root_indices:
                try:
                    self._start_stage(workflow_id, idx, record=record)
                except Exception as e:
                    logger.exception(
                        "Failed to start root stage %d of workflow %s",
                        idx, workflow_id,
                    )
                    record = self.supervisor.store.update_workflow(
                        workflow_id,
                        status=WorkflowStatus.FAILED,
                        completed_at=time.time(),
                        error_message=f"Failed to start stage {idx}: {e}",
                    )
                    return WorkflowStatusPayload.from_record(record)

            record = self.supervisor.store.update_workflow(
                workflow_id, status=WorkflowStatus.RUNNING
            )
        finally:
            self._lock.release()

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
        with self._lock:
            record = self.supervisor.store.get_workflow(workflow_id)
            if record is None:
                raise ValueError(f"Workflow not found: {workflow_id}")
            if record.status in _TERMINAL_WORKFLOW_STATUSES:
                raise ValueError(
                    f"Workflow {workflow_id} is already terminal "
                    f"({record.status.value})"
                )

            for idx, state in record.stage_states.items():
                if not state.status.is_terminal:
                    if state.worker_id:
                        try:
                            self.supervisor.cancel_worker(state.worker_id)
                        except (ValueError, RuntimeError) as e:
                            logger.debug(
                                "Could not cancel worker %s for workflow %s: %s",
                                state.worker_id,
                                workflow_id,
                                e,
                            )
                    state.status = WorkerStatus.CANCELLED

            record = self.supervisor.store.update_workflow(
                workflow_id,
                status=WorkflowStatus.CANCELLED,
                stage_states=record.stage_states,
                completed_at=time.time(),
                error_message="Cancelled by user",
            )

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
                    self.supervisor.cleanup_worker(
                        state.worker_id,
                        remove_branch=True,
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
        # Determine state changes and which stages to start under the lock,
        # then release the lock before doing heavy I/O (worktree creation,
        # process spawning) to avoid blocking create_workflow callers.
        stages_to_start: list[int] = []
        with self._lock:
            result = self._evaluate_stage_completion(
                worker_id, workflow_id, stage_index
            )
            if result is None:
                return
            stages_to_start = result

        # Start downstream stages outside the lock — each may create a
        # worktree (30 s subprocess) and spawn a worker process.
        for idx in stages_to_start:
            try:
                self._start_stage(workflow_id, idx)
            except WorkflowStartAborted:
                logger.info(
                    "Skipped stage %d of workflow %s because the workflow is terminal",
                    idx,
                    workflow_id,
                )
                return
            except Exception as e:
                logger.exception(
                    "Failed to start stage %d of workflow %s", idx, workflow_id
                )
                with self._lock:
                    record = self.supervisor.store.get_workflow(workflow_id)
                    if record is None:
                        return
                    state = record.stage_states.get(idx, StageState())
                    state.status = WorkerStatus.FAILED
                    record.stage_states[idx] = state
                    self.supervisor.store.update_workflow(
                        workflow_id,
                        status=WorkflowStatus.FAILED,
                        stage_states=record.stage_states,
                        completed_at=time.time(),
                        error_message=(
                            f"Failed to start stage {idx} "
                            f"({record.stages[idx].name}): {e}"
                        ),
                    )
                return

        # Check whether all stages are now terminal.
        with self._lock:
            record = self.supervisor.store.get_workflow(workflow_id)
            if record is None:
                return
            if record.status in _TERMINAL_WORKFLOW_STATUSES:
                return
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

    def _evaluate_stage_completion(
        self, worker_id: str, workflow_id: str, stage_index: int
    ) -> Optional[list[int]]:
        """Evaluate a completed stage and return indices of stages to start.

        Must be called while holding ``self._lock``.  Returns ``None`` if the
        workflow/worker is missing or the workflow already failed, otherwise
        returns a (possibly empty) list of stage indices whose dependencies
        are now satisfied.
        """
        record = self.supervisor.store.get_workflow(workflow_id)
        if record is None:
            return None
        if record.status in _TERMINAL_WORKFLOW_STATUSES:
            return None

        worker_record = self.supervisor.store.get_worker(worker_id)
        if worker_record is None:
            return None

        state = record.stage_states.get(stage_index, StageState())
        state.status = worker_record.status
        record.stage_states[stage_index] = state

        if worker_record.status in {WorkerStatus.FAILED, WorkerStatus.TIMED_OUT, WorkerStatus.CANCELLED}:
            for idx, s in record.stage_states.items():
                if idx != stage_index and not s.status.is_terminal and s.worker_id:
                    try:
                        self.supervisor.cancel_worker(s.worker_id)
                    except (ValueError, RuntimeError) as e:
                        logger.debug("Could not cancel worker %s during stage failure: %s",
                                     s.worker_id, e)
                    s.status = WorkerStatus.CANCELLED

            self.supervisor.store.update_workflow(
                workflow_id,
                status=WorkflowStatus.FAILED,
                stage_states=record.stage_states,
                completed_at=time.time(),
                error_message=f"Stage {stage_index} ({record.stages[stage_index].name}) "
                              f"{worker_record.status.value}: {worker_record.error_message or ''}",
            )
            return None

        self.supervisor.store.update_workflow(
            workflow_id, stage_states=record.stage_states
        )

        ready: list[int] = []
        for i, stage_def in enumerate(record.stages):
            s = record.stage_states.get(i)
            if s is None or s.status != WorkerStatus.PENDING:
                continue
            if stage_index not in stage_def.depends_on:
                continue

            all_deps_done = all(
                record.stage_states.get(dep, StageState()).status == WorkerStatus.SUCCEEDED
                for dep in stage_def.depends_on
            )
            if all_deps_done:
                # Mark RUNNING now so no other callback tries to start the
                # same stage concurrently.
                s.status = WorkerStatus.RUNNING
                record.stage_states[i] = s
                ready.append(i)

        if ready:
            self.supervisor.store.update_workflow(
                workflow_id, stage_states=record.stage_states
            )

        return ready

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_stage(
        self, workflow_id: str, stage_index: int, *, record: Optional[WorkflowRecord] = None
    ) -> str:
        if record is None:
            record = self.supervisor.store.get_workflow(workflow_id)
            if record is None:
                raise ValueError(f"Workflow not found: {workflow_id}")
        if record.status in _TERMINAL_WORKFLOW_STATUSES:
            raise WorkflowStartAborted(
                f"Workflow {workflow_id} is already terminal ({record.status.value})"
            )
        state = record.stage_states.get(stage_index, StageState())
        if state.status == WorkerStatus.CANCELLED:
            raise WorkflowStartAborted(
                f"Stage {stage_index} of workflow {workflow_id} was cancelled"
            )
        stage_def = record.stages[stage_index]

        rendered_prompt = self._render_prompt(record, stage_index)

        # INHERIT: branch off the parent's branch tip so this stage sees the
        # parent's commits, but get its own fresh worktree (no shared
        # .codefleet/ artifacts, no .git/index.lock contention). NEW:
        # branch off the workflow's base_ref.
        base_ref = record.base_ref
        if stage_def.worktree_strategy == WorktreeStrategy.INHERIT and stage_def.depends_on:
            first_dep = stage_def.depends_on[0]
            dep_state = record.stage_states.get(first_dep, StageState())
            if dep_state.worker_id:
                dep_worker = self.supervisor.store.get_worker(dep_state.worker_id)
                if dep_worker:
                    base_ref = dep_worker.branch_name

        payload = self.supervisor.create_worker(
            repo_path=record.repo_path,
            task_name=f"{record.name}/{stage_def.name}",
            prompt=rendered_prompt,
            base_ref=base_ref,
            model=stage_def.model.value if stage_def.model else None,
            executor=stage_def.executor.value,
            reasoning_effort=stage_def.reasoning_effort,
            timeout_seconds=stage_def.timeout_seconds,
            extra_args=stage_def.extra_args,
            workflow_id=workflow_id,
            stage_index=stage_index,
        )

        state = record.stage_states.get(stage_index, StageState())
        state.worker_id = payload.worker_id
        state.status = WorkerStatus.RUNNING
        record.stage_states[stage_index] = state
        self.supervisor.store.update_workflow(
            workflow_id, stage_states=record.stage_states
        )

        return payload.worker_id

    def _render_prompt(self, workflow: WorkflowRecord, stage_index: int) -> str:
        template = workflow.stages[stage_index].prompt_template
        variables = defaultdict(str, task_prompt=workflow.task_prompt)

        worker_ids = []
        for i in range(len(workflow.stages)):
            state = workflow.stage_states.get(i, StageState())
            if state.status == WorkerStatus.SUCCEEDED and state.worker_id:
                worker_ids.append(state.worker_id)

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
            except Exception as e:
                logger.error(
                    "Failed to parse result for stage %d (worker %s): %s",
                    i, state.worker_id, e, exc_info=True,
                )
                variables[f"stage_{i}_summary"] = (
                    f"[ERROR: stage {i} result unavailable]"
                )

        # Use regex substitution instead of str.format_map so that literal
        # braces in prompts (JSON examples, code, etc.) don't crash.
        # Known variable prefixes get empty-string defaults; anything else
        # is left untouched.
        import re
        _KNOWN_PREFIXES = ("task_prompt", "stage_")
        def _replace(m: re.Match) -> str:
            key = m.group(1)
            if key in variables:
                return str(variables[key])
            if any(key.startswith(p) for p in _KNOWN_PREFIXES):
                return ""  # known template var not yet populated
            return m.group(0)  # leave truly unknown {keys} as-is
        return re.sub(r"\{(\w+)\}", _replace, template)

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
