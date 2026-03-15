from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field


class ExecutorType(str, enum.Enum):
    CODEX = "codex"
    GEMINI = "gemini"
    CLAUDE = "claude"


class WorkerStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    CLEANUP_FAILED = "cleanup_failed"

    @property
    def is_terminal(self) -> bool:
        return self in {
            WorkerStatus.SUCCEEDED,
            WorkerStatus.FAILED,
            WorkerStatus.CANCELLED,
            WorkerStatus.TIMED_OUT,
            WorkerStatus.CLEANUP_FAILED,
        }


class ResultStatus(str, enum.Enum):
    COMPLETED = "completed"
    BLOCKED = "blocked"


class TestStatus(str, enum.Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


class TestResult(BaseModel):
    command: str
    status: TestStatus
    details: str = ""


class WorkerResult(BaseModel):
    summary: str
    files_changed: list[str] = Field(default_factory=list)
    tests: list[TestResult] = Field(default_factory=list)
    commits: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    status: ResultStatus


class WorkerRecord(BaseModel):
    worker_id: str
    task_name: str
    repo_path: str
    branch_name: str
    worktree_path: str
    worker_dir: str
    model: str
    executor: ExecutorType = ExecutorType.CODEX
    profile: Optional[str] = None
    status: WorkerStatus
    created_at: float
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    timeout_seconds: int
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    codex_command: str
    prompt: str
    result_json_path: str
    stdout_path: str
    stderr_path: str
    prompt_path: str
    meta_path: str
    retry_count: int = 0
    parent_worker_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    error_message: Optional[str] = None
    workflow_id: Optional[str] = None
    stage_index: Optional[int] = None


class WorkerStatusPayload(BaseModel):
    """Payload returned by create_worker, get_worker_status, etc."""

    worker_id: str
    task_name: str
    status: WorkerStatus
    repo_path: str
    branch_name: str
    worktree_path: str
    worker_dir: str
    model: str
    executor: ExecutorType = ExecutorType.CODEX
    profile: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    timeout_seconds: int
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    retry_count: int = 0
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    error_message: Optional[str] = None
    prompt_path: str
    result_json_path: str
    stdout_path: str
    stderr_path: str
    meta_path: str

    @classmethod
    def from_record(cls, record: WorkerRecord) -> WorkerStatusPayload:
        return cls(
            worker_id=record.worker_id,
            task_name=record.task_name,
            status=record.status,
            repo_path=record.repo_path,
            branch_name=record.branch_name,
            worktree_path=record.worktree_path,
            worker_dir=record.worker_dir,
            model=record.model,
            executor=record.executor,
            profile=record.profile,
            created_at=record.created_at,
            started_at=record.started_at,
            ended_at=record.ended_at,
            timeout_seconds=record.timeout_seconds,
            pid=record.pid,
            exit_code=record.exit_code,
            retry_count=record.retry_count,
            tags=record.tags,
            metadata=record.metadata,
            error_message=record.error_message,
            prompt_path=record.prompt_path,
            result_json_path=record.result_json_path,
            stdout_path=record.stdout_path,
            stderr_path=record.stderr_path,
            meta_path=record.meta_path,
        )


# --- Workflow Models ---


class WorktreeStrategy(str, enum.Enum):
    NEW = "new"
    INHERIT = "inherit"


class StageDefinition(BaseModel):
    name: str
    executor: ExecutorType
    prompt_template: str
    model: Optional[str] = None
    worktree_strategy: WorktreeStrategy = WorktreeStrategy.INHERIT
    depends_on: list[int] = Field(default_factory=list)
    timeout_seconds: Optional[int] = None
    reasoning_effort: Optional[str] = None
    extra_args: Optional[list[str]] = None


class WorkflowStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageState(BaseModel):
    worker_id: Optional[str] = None
    status: WorkerStatus = WorkerStatus.PENDING
    worktree_path: Optional[str] = None


class WorkflowRecord(BaseModel):
    workflow_id: str
    name: str
    status: WorkflowStatus
    repo_path: str
    base_ref: str
    task_prompt: str
    stages: list[StageDefinition]
    stage_states: dict[int, StageState] = Field(default_factory=dict)
    created_at: float
    completed_at: Optional[float] = None
    error_message: Optional[str] = None


class WorkflowStatusPayload(BaseModel):
    workflow_id: str
    name: str
    status: WorkflowStatus
    repo_path: str
    stage_summary: list[dict]
    created_at: float
    completed_at: Optional[float] = None
    error_message: Optional[str] = None

    @classmethod
    def from_record(cls, record: WorkflowRecord) -> WorkflowStatusPayload:
        stage_summary = []
        for i, stage in enumerate(record.stages):
            state = record.stage_states.get(i, StageState())
            stage_summary.append({
                "index": i,
                "name": stage.name,
                "executor": stage.executor.value,
                "status": state.status.value,
                "worker_id": state.worker_id,
            })
        return cls(
            workflow_id=record.workflow_id,
            name=record.name,
            status=record.status,
            repo_path=record.repo_path,
            stage_summary=stage_summary,
            created_at=record.created_at,
            completed_at=record.completed_at,
            error_message=record.error_message,
        )
