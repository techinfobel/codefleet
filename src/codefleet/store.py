import enum
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .models import (
    ExecutorType,
    StageDefinition,
    StageState,
    WorkerRecord,
    WorkerStatus,
    WorkflowRecord,
    WorkflowStatus,
)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,
    task_name TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    worktree_path TEXT NOT NULL,
    worker_dir TEXT NOT NULL,
    model TEXT NOT NULL,
    profile TEXT,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    ended_at REAL,
    timeout_seconds INTEGER NOT NULL,
    pid INTEGER,
    exit_code INTEGER,
    codex_command TEXT NOT NULL,
    prompt TEXT NOT NULL,
    result_json_path TEXT NOT NULL,
    stdout_path TEXT NOT NULL,
    stderr_path TEXT NOT NULL,
    prompt_path TEXT NOT NULL,
    meta_path TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    parent_worker_id TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT
);
"""

_CREATE_WORKFLOWS_TABLE = """
CREATE TABLE IF NOT EXISTS workflows (
    workflow_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    repo_path TEXT NOT NULL,
    base_ref TEXT NOT NULL,
    task_prompt TEXT NOT NULL,
    stages_json TEXT NOT NULL,
    stage_states_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    completed_at REAL,
    error_message TEXT
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status);",
    "CREATE INDEX IF NOT EXISTS idx_workers_created_at ON workers(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);",
    "CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON workflows(created_at);",
]

_MIGRATIONS = [
    "ALTER TABLE workers ADD COLUMN executor TEXT NOT NULL DEFAULT 'codex'",
    "ALTER TABLE workers ADD COLUMN workflow_id TEXT",
    "ALTER TABLE workers ADD COLUMN stage_index INTEGER",
]

_INSERT_SQL = """
INSERT INTO workers (
    worker_id, task_name, repo_path, branch_name, worktree_path,
    worker_dir, model, profile, status, created_at, started_at,
    ended_at, timeout_seconds, pid, exit_code, codex_command,
    prompt, result_json_path, stdout_path, stderr_path,
    prompt_path, meta_path, retry_count, parent_worker_id,
    tags_json, metadata_json, error_message, executor,
    workflow_id, stage_index
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_WORKFLOW_SQL = """
INSERT INTO workflows (
    workflow_id, name, status, repo_path, base_ref, task_prompt,
    stages_json, stage_states_json, created_at, completed_at, error_message
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _dump_model(value):
    return value.model_dump() if hasattr(value, "model_dump") else value


class WorkerStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_WORKFLOWS_TABLE)
        for idx in _CREATE_INDEXES:
            conn.execute(idx)
        for migration in _MIGRATIONS:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # Column already exists
        conn.commit()

    # --- Worker CRUD ---

    def insert_worker(self, record: WorkerRecord) -> None:
        conn = self._get_conn()
        conn.execute(
            _INSERT_SQL,
            (
                record.worker_id,
                record.task_name,
                record.repo_path,
                record.branch_name,
                record.worktree_path,
                record.worker_dir,
                record.model,
                record.profile,
                record.status.value,
                record.created_at,
                record.started_at,
                record.ended_at,
                record.timeout_seconds,
                record.pid,
                record.exit_code,
                record.codex_command,
                record.prompt,
                record.result_json_path,
                record.stdout_path,
                record.stderr_path,
                record.prompt_path,
                record.meta_path,
                record.retry_count,
                record.parent_worker_id,
                json.dumps(record.tags),
                json.dumps(record.metadata),
                record.error_message,
                record.executor.value,
                record.workflow_id,
                record.stage_index,
            ),
        )
        conn.commit()

    def get_worker(self, worker_id: str) -> Optional[WorkerRecord]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM workers WHERE worker_id = ?", (worker_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_workers(self, worker_ids: list[str]) -> list[WorkerRecord]:
        if not worker_ids:
            return []
        conn = self._get_conn()
        placeholders = ", ".join("?" for _ in worker_ids)
        rows = conn.execute(
            f"SELECT * FROM workers WHERE worker_id IN ({placeholders})",
            tuple(worker_ids)
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def update_worker(self, worker_id: str, **kwargs) -> None:
        if not kwargs:
            return
        conn = self._get_conn()
        _JSON_COLUMNS = {"tags": "tags_json", "metadata": "metadata_json"}
        sets = []
        vals = []
        for key, value in kwargs.items():
            col = _JSON_COLUMNS.get(key, key)
            if col in _JSON_COLUMNS.values():
                value = json.dumps(value)
            elif isinstance(value, enum.Enum):
                value = value.value
            sets.append(f"{col} = ?")
            vals.append(value)
        vals.append(worker_id)
        conn.execute(
            f"UPDATE workers SET {', '.join(sets)} WHERE worker_id = ?",
            vals,
        )
        conn.commit()

    def list_workers(
        self,
        statuses: Optional[list[str]] = None,
        limit: int = 25,
    ) -> list[WorkerRecord]:
        conn = self._get_conn()
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            rows = conn.execute(
                f"SELECT * FROM workers WHERE status IN ({placeholders}) "
                f"ORDER BY created_at DESC LIMIT ?",
                (*statuses, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM workers ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> WorkerRecord:
        data = dict(row)
        data["executor"] = ExecutorType(data.get("executor") or "codex")
        data["status"] = WorkerStatus(data["status"])
        data["tags"] = json.loads(data.pop("tags_json", "[]"))
        data["metadata"] = json.loads(data.pop("metadata_json", "{}"))
        return WorkerRecord.model_validate(data)

    # --- Workflow CRUD ---

    def insert_workflow(self, record: WorkflowRecord) -> None:
        conn = self._get_conn()
        stages_json = json.dumps([_dump_model(stage) for stage in record.stages])
        states_json = json.dumps(
            {str(k): _dump_model(state) for k, state in record.stage_states.items()}
        )
        conn.execute(
            _INSERT_WORKFLOW_SQL,
            (
                record.workflow_id,
                record.name,
                record.status.value,
                record.repo_path,
                record.base_ref,
                record.task_prompt,
                stages_json,
                states_json,
                record.created_at,
                record.completed_at,
                record.error_message,
            ),
        )
        conn.commit()

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowRecord]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_workflow(row)

    def update_workflow(self, workflow_id: str, **kwargs) -> None:
        if not kwargs:
            return
        conn = self._get_conn()
        sets = []
        vals = []
        for key, value in kwargs.items():
            if key == "status" and isinstance(value, WorkflowStatus):
                value = value.value
            elif key == "stage_states" and isinstance(value, dict):
                key = "stage_states_json"
                value = json.dumps(
                    {str(k): _dump_model(state) for k, state in value.items()}
                )
            elif key == "stages" and isinstance(value, list):
                key = "stages_json"
                value = json.dumps([_dump_model(stage) for stage in value])
            sets.append(f"{key} = ?")
            vals.append(value)
        vals.append(workflow_id)
        conn.execute(
            f"UPDATE workflows SET {', '.join(sets)} WHERE workflow_id = ?",
            vals,
        )
        conn.commit()

    def list_workflows(
        self,
        statuses: Optional[list[str]] = None,
        limit: int = 25,
    ) -> list[WorkflowRecord]:
        conn = self._get_conn()
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            rows = conn.execute(
                f"SELECT * FROM workflows WHERE status IN ({placeholders}) "
                f"ORDER BY created_at DESC LIMIT ?",
                (*statuses, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM workflows ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_workflow(row) for row in rows]

    @staticmethod
    def _row_to_workflow(row: sqlite3.Row) -> WorkflowRecord:
        data = dict(row)
        data["status"] = WorkflowStatus(data["status"])
        data["stages"] = [
            StageDefinition.model_validate(stage)
            for stage in json.loads(data.pop("stages_json"))
        ]
        data["stage_states"] = {
            int(index): StageState.model_validate(state)
            for index, state in json.loads(data.pop("stage_states_json")).items()
        }
        return WorkflowRecord.model_validate(data)

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
