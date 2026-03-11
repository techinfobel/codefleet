from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field


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
