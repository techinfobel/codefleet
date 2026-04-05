import enum
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

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
    last_heartbeat_at REAL,
    last_activity_at REAL,
    timeout_seconds INTEGER NOT NULL,
    pid INTEGER,
    exit_code INTEGER,
    command_json TEXT NOT NULL,
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
    error_message TEXT,
    heartbeat_message TEXT
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
    "ALTER TABLE workers RENAME COLUMN codex_command TO command_json",
    "ALTER TABLE workers ADD COLUMN last_heartbeat_at REAL",
    "ALTER TABLE workers ADD COLUMN last_activity_at REAL",
    "ALTER TABLE workers ADD COLUMN heartbeat_message TEXT",
]

_INSERT_SQL = """
INSERT INTO workers (
    worker_id, task_name, repo_path, branch_name, worktree_path,
    worker_dir, model, profile, status, created_at, started_at,
    ended_at, last_heartbeat_at, last_activity_at, timeout_seconds, pid, exit_code, command_json,
    prompt, result_json_path, stdout_path, stderr_path,
    prompt_path, meta_path, retry_count, parent_worker_id,
    tags_json, metadata_json, error_message, heartbeat_message, executor,
    workflow_id, stage_index
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            self._local.conn = sqlite3.connect(str(self.db_path), timeout=10)
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
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "duplicate column" in msg or "already exists" in msg or "no such column" in msg:
                    logger.debug("Migration already applied: %s", migration[:60])
                else:
                    raise
        conn.commit()

    # --- Worker CRUD ---

    def insert_worker(self, record: WorkerRecord) -> None:
        conn = self._get_conn()
        try:
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
                    record.last_heartbeat_at,
                    record.last_activity_at,
                    record.timeout_seconds,
                    record.pid,
                    record.exit_code,
                    record.command_json,
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
                    record.heartbeat_message,
                    record.executor.value,
                    record.workflow_id,
                    record.stage_index,
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

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

    def update_worker(self, worker_id: str, **kwargs) -> Optional[WorkerRecord]:
        if not kwargs:
            return self.get_worker(worker_id)
        json_columns = {"tags": "tags_json", "metadata": "metadata_json"}
        self._execute_update("workers", "worker_id", worker_id, kwargs, json_columns)
        return self.get_worker(worker_id)

    def list_workers(
        self,
        statuses: Optional[list[str]] = None,
        limit: int = 25,
    ) -> list[WorkerRecord]:
        conn = self._get_conn()
        sql, params = self._build_list_query("workers", statuses, limit)
        rows = conn.execute(sql, params).fetchall()
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
        try:
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
        except Exception:
            conn.rollback()
            raise

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowRecord]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_workflow(row)

    def update_workflow(self, workflow_id: str, **kwargs) -> Optional[WorkflowRecord]:
        if not kwargs:
            return self.get_workflow(workflow_id)
        json_columns = {
            "stage_states": "stage_states_json",
            "stages": "stages_json",
        }
        self._execute_update("workflows", "workflow_id", workflow_id, kwargs, json_columns)
        return self.get_workflow(workflow_id)

    def list_workflows(
        self,
        statuses: Optional[list[str]] = None,
        limit: int = 25,
    ) -> list[WorkflowRecord]:
        conn = self._get_conn()
        sql, params = self._build_list_query("workflows", statuses, limit)
        rows = conn.execute(sql, params).fetchall()
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

    # --- Shared helpers ---

    def _execute_update(
        self,
        table: str,
        pk_col: str,
        pk_val: str,
        updates: dict,
        json_columns: Optional[dict[str, str]] = None,
    ) -> None:
        """Build and execute an UPDATE statement.

        *json_columns* maps logical field names (e.g. ``"tags"``) to their
        ``_json`` column counterparts (e.g. ``"tags_json"``).  Values for
        these columns are serialised with ``json.dumps`` and, for dicts /
        lists whose items may be Pydantic models, via ``_dump_model``.
        """
        if json_columns is None:
            json_columns = {}
        conn = self._get_conn()
        sets: list[str] = []
        vals: list = []
        for key, value in updates.items():
            col = json_columns.get(key, key)
            if col in json_columns.values():
                # Serialise JSON columns, handling nested Pydantic models
                if isinstance(value, dict):
                    value = json.dumps(
                        {str(k): _dump_model(v) for k, v in value.items()}
                    )
                elif isinstance(value, list):
                    value = json.dumps([_dump_model(v) for v in value])
                else:
                    value = json.dumps(value)
            elif isinstance(value, enum.Enum):
                value = value.value
            sets.append(f"{col} = ?")
            vals.append(value)
        vals.append(pk_val)
        conn.execute(
            f"UPDATE {table} SET {', '.join(sets)} WHERE {pk_col} = ?",
            vals,
        )
        conn.commit()

    @staticmethod
    def _build_list_query(
        table: str,
        statuses: Optional[list[str]],
        limit: int,
    ) -> tuple[str, tuple]:
        """Return (sql, params) for a status-filtered list query."""
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            sql = (
                f"SELECT * FROM {table} WHERE status IN ({placeholders}) "
                f"ORDER BY created_at DESC LIMIT ?"
            )
            return sql, (*statuses, limit)
        sql = f"SELECT * FROM {table} ORDER BY created_at DESC LIMIT ?"
        return sql, (limit,)

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
