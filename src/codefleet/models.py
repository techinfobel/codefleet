from __future__ import annotations

import enum
import json
from pathlib import Path
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


class WorkerResult(BaseModel):
    summary: str
    files_changed: list[str] = Field(default_factory=list)
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
    timeout_seconds: int = Field(gt=0)
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    command_json: str
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
        data = record.model_dump(include=set(cls.model_fields))
        return cls.model_validate(data)


# --- Workflow Models ---


class WorktreeStrategy(str, enum.Enum):
    NEW = "new"
    INHERIT = "inherit"


class StageDefinition(BaseModel):
    name: str
    executor: ExecutorType
    prompt_template: str
    model: Optional[str] = None
    worktree_strategy: WorktreeStrategy = WorktreeStrategy.NEW
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


def parse_result_file(path: Path) -> WorkerResult:
    """Parse and validate a result.json file."""
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"Result file is empty: {path}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in result file: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Result JSON must be an object")

    return WorkerResult.model_validate(data)
