from __future__ import annotations

import enum
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ExecutorType(str, enum.Enum):
    CODEX = "codex"
    GEMINI = "gemini"
    CLAUDE = "claude"


class SupportedModel(str, enum.Enum):
    GPT_5_5 = "gpt-5.5"
    GEMINI_3_1_PRO_PREVIEW = "gemini-3.1-pro-preview"
    CLAUDE_OPUS_4_7 = "claude-opus-4-7"
    CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"


_SUPPORTED_MODELS_BY_EXECUTOR = {
    ExecutorType.CODEX: (SupportedModel.GPT_5_5.value,),
    ExecutorType.GEMINI: (SupportedModel.GEMINI_3_1_PRO_PREVIEW.value,),
    ExecutorType.CLAUDE: (
        SupportedModel.CLAUDE_OPUS_4_7.value,
        SupportedModel.CLAUDE_SONNET_4_6.value,
    ),
}


def supported_models_for_executor(executor: ExecutorType) -> tuple[str, ...]:
    return _SUPPORTED_MODELS_BY_EXECUTOR[executor]


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
    COMPLETED_NO_CHANGES = "completed_no_changes"
    BLOCKED = "blocked"


class TestResult(BaseModel):
    command: str
    status: str
    details: str


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
    last_heartbeat_at: Optional[float] = None
    last_activity_at: Optional[float] = None
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
    heartbeat_message: Optional[str] = None
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
    last_heartbeat_at: Optional[float] = None
    last_activity_at: Optional[float] = None
    timeout_seconds: int
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    retry_count: int = 0
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    error_message: Optional[str] = None
    heartbeat_message: Optional[str] = None
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
    model: Optional[SupportedModel] = None
    worktree_strategy: WorktreeStrategy = WorktreeStrategy.NEW
    depends_on: list[int] = Field(default_factory=list)
    timeout_seconds: Optional[int] = None
    reasoning_effort: Optional[str] = None
    extra_args: Optional[list[str]] = None

    @model_validator(mode="after")
    def validate_model_executor_compatibility(self) -> StageDefinition:
        if self.model is None:
            return self
        allowed_models = supported_models_for_executor(self.executor)
        if self.model.value not in allowed_models:
            raise ValueError(
                f"Unsupported model '{self.model.value}' for executor "
                f"'{self.executor.value}'. Allowed models: {', '.join(allowed_models)}"
            )
        return self


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
