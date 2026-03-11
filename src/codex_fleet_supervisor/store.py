import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .models import WorkerRecord, WorkerStatus

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

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status);",
    "CREATE INDEX IF NOT EXISTS idx_workers_created_at ON workers(created_at);",
]

_INSERT_SQL = """
INSERT INTO workers (
    worker_id, task_name, repo_path, branch_name, worktree_path,
    worker_dir, model, profile, status, created_at, started_at,
    ended_at, timeout_seconds, pid, exit_code, codex_command,
    prompt, result_json_path, stdout_path, stderr_path,
    prompt_path, meta_path, retry_count, parent_worker_id,
    tags_json, metadata_json, error_message
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


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
        for idx in _CREATE_INDEXES:
            conn.execute(idx)
        conn.commit()

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

    def update_worker(self, worker_id: str, **kwargs) -> None:
        conn = self._get_conn()
        column_map = {
            "tags": "tags_json",
            "metadata": "metadata_json",
        }
        sets = []
        vals = []
        for key, value in kwargs.items():
            col = column_map.get(key, key)
            if col == "tags_json":
                value = json.dumps(value)
            elif col == "metadata_json":
                value = json.dumps(value)
            elif col == "status" and isinstance(value, WorkerStatus):
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
        return WorkerRecord(
            worker_id=row["worker_id"],
            task_name=row["task_name"],
            repo_path=row["repo_path"],
            branch_name=row["branch_name"],
            worktree_path=row["worktree_path"],
            worker_dir=row["worker_dir"],
            model=row["model"],
            profile=row["profile"],
            status=WorkerStatus(row["status"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            timeout_seconds=row["timeout_seconds"],
            pid=row["pid"],
            exit_code=row["exit_code"],
            codex_command=row["codex_command"],
            prompt=row["prompt"],
            result_json_path=row["result_json_path"],
            stdout_path=row["stdout_path"],
            stderr_path=row["stderr_path"],
            prompt_path=row["prompt_path"],
            meta_path=row["meta_path"],
            retry_count=row["retry_count"],
            parent_worker_id=row["parent_worker_id"],
            tags=json.loads(row["tags_json"]),
            metadata=json.loads(row["metadata_json"]),
            error_message=row["error_message"],
        )

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
