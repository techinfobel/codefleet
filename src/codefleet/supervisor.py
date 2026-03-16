import json
import logging
import re
import shutil
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

from .models import (
    ExecutorType,
    WorkerRecord,
    WorkerStatus,
    WorkerStatusPayload,
    parse_result_file,
)
from .store import WorkerStore
from .git_ops import (
    GitError,
    is_git_repo,
    create_worktree,
    remove_worktree,
    delete_branch,
    get_git_path,
)
from .worker_runtime import (
    WorkerProcess,
    build_worker_command,
    get_codex_path,
    get_gemini_path,
    get_claude_path,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_CLAUDE_MODEL = "claude-opus-4-6"
DEFAULT_REASONING_EFFORT = "max"
DEFAULT_TIMEOUT = 600  # 10 minutes
DEFAULT_BASE_DIR = Path.home() / ".codex-fleet"
MAX_CONCURRENT_WORKERS = 50
DEFAULT_MAX_SPAWN_DEPTH = 2
DEFAULT_RATE_LIMIT_RETRIES = 3
DEFAULT_RATE_LIMIT_BASE_DELAY = 4.0
DEFAULT_RATE_LIMIT_MAX_DELAY = 60.0
DEFAULT_STALE_TIMEOUT = 120.0  # 2 minutes
DEFAULT_STALE_MAX_RESTARTS = 2

_EXECUTOR_REASONING_DEFAULTS = {
    ExecutorType.CODEX: "xhigh",      # OpenAI: low, medium, high, xhigh
    ExecutorType.GEMINI: None,         # Gemini CLI has no effort flag
    ExecutorType.CLAUDE: "max",        # Claude: low, medium, high, max
}


def _sanitize_task_name(name: str) -> str:
    """Sanitize task name for use in branch names."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    return sanitized[:50] or "task"


def _generate_worker_id() -> str:
    return f"w_{uuid.uuid4().hex[:12]}"


class FleetSupervisor:
    def __init__(
        self,
        base_dir: Optional[Path | str] = None,
        db_path: Optional[Path | str] = None,
        allowed_repos: Optional[list[str]] = None,
        default_model: str = DEFAULT_MODEL,
        default_gemini_model: str = DEFAULT_GEMINI_MODEL,
        default_claude_model: str = DEFAULT_CLAUDE_MODEL,
        default_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        default_timeout: int = DEFAULT_TIMEOUT,
        max_concurrent: int = MAX_CONCURRENT_WORKERS,
        default_executor: str = "codex",
        max_spawn_depth: int = DEFAULT_MAX_SPAWN_DEPTH,
        rate_limit_max_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
        rate_limit_base_delay: float = DEFAULT_RATE_LIMIT_BASE_DELAY,
        rate_limit_max_delay: float = DEFAULT_RATE_LIMIT_MAX_DELAY,
        stale_timeout: float = DEFAULT_STALE_TIMEOUT,
        stale_max_restarts: int = DEFAULT_STALE_MAX_RESTARTS,
    ):
        if default_timeout <= 0:
            raise ValueError(f"default_timeout must be positive, got {default_timeout}")
        if max_spawn_depth < 0:
            raise ValueError(f"max_spawn_depth must be non-negative, got {max_spawn_depth}")

        self.base_dir = Path(base_dir or DEFAULT_BASE_DIR).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = Path(db_path or self.base_dir / "fleet.db")
        self.store = WorkerStore(self.db_path)

        self.allowed_repos: Optional[list[Path]] = None
        if allowed_repos:
            self.allowed_repos = [Path(r).resolve() for r in allowed_repos]

        self.default_model = default_model
        self.default_gemini_model = default_gemini_model
        self.default_claude_model = default_claude_model
        self.default_reasoning_effort = default_reasoning_effort
        self.default_timeout = default_timeout
        self.max_concurrent = max_concurrent
        self.default_executor = ExecutorType(default_executor)
        self.max_spawn_depth = max_spawn_depth
        self.rate_limit_max_retries = rate_limit_max_retries
        self.rate_limit_base_delay = rate_limit_base_delay
        self.rate_limit_max_delay = rate_limit_max_delay
        self.stale_timeout = stale_timeout
        self.stale_max_restarts = stale_max_restarts

        self._active_workers: dict[str, WorkerProcess] = {}
        self._workflow_engine = None

    @property
    def workflow_engine(self):
        if self._workflow_engine is None:
            from .workflow import WorkflowEngine
            self._workflow_engine = WorkflowEngine(self)
        return self._workflow_engine

    def _is_repo_allowed(self, repo_path: Path) -> bool:
        if self.allowed_repos is None:
            return True
        resolved = repo_path.resolve()
        return any(
            resolved == allowed or resolved.is_relative_to(allowed)
            for allowed in self.allowed_repos
        )

    def healthcheck(self) -> dict:
        from importlib.metadata import version
        try:
            app_version = version("codefleet")
        except Exception:
            app_version = "unknown"
        executor_paths = {
            "codex": get_codex_path(),
            "gemini": get_gemini_path(),
            "claude": get_claude_path(),
        }
        git_path = get_git_path()
        return {
            "ok": bool(any(executor_paths.values()) and git_path),
            "app": "codefleet",
            "version": app_version,
            "db_path": str(self.db_path),
            "base_dir": str(self.base_dir),
            **{f"{name}_found": path is not None for name, path in executor_paths.items()},
            **{f"{name}_path": path or "" for name, path in executor_paths.items()},
            "git_found": git_path is not None,
            "git_path": git_path or "",
            "default_model": self.default_model,
            "default_gemini_model": self.default_gemini_model,
            "default_claude_model": self.default_claude_model,
            "default_executor": self.default_executor.value,
            "max_spawn_depth": self.max_spawn_depth,
        }

    def create_worker(
        self,
        repo_path: str,
        task_name: str,
        prompt: str,
        base_ref: str = "HEAD",
        model: Optional[str] = None,
        executor: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        profile: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        extra_args: Optional[list[str]] = None,
        extra_codex_args: Optional[list[str]] = None,
        parent_worker_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        stage_index: Optional[int] = None,
        existing_worktree_path: Optional[str] = None,
        existing_branch_name: Optional[str] = None,
    ) -> WorkerStatusPayload:
        repo = Path(repo_path).resolve()

        if not repo.exists():
            raise ValueError(f"Repo path does not exist: {repo}")
        if not is_git_repo(repo):
            raise ValueError(f"Not a git repository: {repo}")
        if not self._is_repo_allowed(repo):
            raise ValueError(f"Repo not in allowlist: {repo}")

        running = self.store.list_workers(
            statuses=["running"], limit=self.max_concurrent + 1
        )
        if len(running) >= self.max_concurrent:
            raise RuntimeError(
                f"Concurrency limit reached ({self.max_concurrent} workers running)"
            )

        if parent_worker_id:
            depth = self._compute_spawn_depth(parent_worker_id)
            if depth > self.max_spawn_depth:
                raise RuntimeError(
                    f"Spawn depth limit reached ({self.max_spawn_depth}). "
                    f"Worker {parent_worker_id} is at depth {depth}."
                )

        executor_type = ExecutorType(executor) if executor else self.default_executor

        if model is None:
            model = {
                ExecutorType.CODEX: self.default_model,
                ExecutorType.GEMINI: self.default_gemini_model,
                ExecutorType.CLAUDE: self.default_claude_model,
            }[executor_type]

        if reasoning_effort is None:
            reasoning_effort = _EXECUTOR_REASONING_DEFAULTS.get(
                executor_type, self.default_reasoning_effort
            )

        timeout_seconds = timeout_seconds or self.default_timeout
        merged_extra_args = extra_args or extra_codex_args

        worker_id = _generate_worker_id()
        sanitized = _sanitize_task_name(task_name)
        branch_name = f"{executor_type.value}/{sanitized}/{worker_id}"

        worker_dir = self.base_dir / "workers" / worker_id
        worker_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = worker_dir / "stdout.log"
        stderr_path = worker_dir / "stderr.log"
        meta_path = worker_dir / "meta.json"

        meta = {
            "worker_id": worker_id,
            "task_name": task_name,
            "repo_path": str(repo),
            "base_ref": base_ref,
            "model": model,
            "executor": executor_type.value,
            "profile": profile,
            "tags": tags or [],
            "metadata": metadata or {},
            "parent_worker_id": parent_worker_id,
            "workflow_id": workflow_id,
            "stage_index": stage_index,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if existing_worktree_path and existing_branch_name:
            worktree_path = Path(existing_worktree_path)
            branch_name = existing_branch_name
        else:
            worktree_path = worker_dir / "worktree"
            create_worktree(repo, worktree_path, branch_name, base_ref)

        # Place prompt and result inside the worktree so sandboxed executors
        # (especially Codex) can access them without path violations.
        codefleet_dir = worktree_path / ".codefleet"
        codefleet_dir.mkdir(exist_ok=True)
        gitignore = codefleet_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n")
        prompt_path = codefleet_dir / "prompt.txt"
        result_json_path = codefleet_dir / "result.json"
        prompt_path.write_text(prompt, encoding="utf-8")

        cmd = build_worker_command(
            executor=executor_type.value,
            prompt_path=prompt_path,
            result_json_path=result_json_path,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=merged_extra_args,
        )

        now = time.time()

        record = WorkerRecord(
            worker_id=worker_id,
            task_name=task_name,
            repo_path=str(repo),
            branch_name=branch_name,
            worktree_path=str(worktree_path),
            worker_dir=str(worker_dir),
            model=model,
            executor=executor_type,
            profile=profile,
            status=WorkerStatus.PENDING,
            created_at=now,
            timeout_seconds=timeout_seconds,
            command_json=json.dumps(cmd),
            prompt=prompt,
            result_json_path=str(result_json_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            prompt_path=str(prompt_path),
            meta_path=str(meta_path),
            retry_count=0,
            parent_worker_id=parent_worker_id,
            tags=tags or [],
            metadata=metadata or {},
            workflow_id=workflow_id,
            stage_index=stage_index,
        )

        self.store.insert_worker(record)

        worker_proc = WorkerProcess(
            worker_id=worker_id,
            command=cmd,
            cwd=worktree_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=timeout_seconds,
            on_complete=self._on_worker_complete,
            max_retries=self.rate_limit_max_retries,
            retry_base_delay=self.rate_limit_base_delay,
            retry_max_delay=self.rate_limit_max_delay,
            stale_timeout=self.stale_timeout,
            stale_max_restarts=self.stale_max_restarts,
        )

        pid = worker_proc.start()
        self._active_workers[worker_id] = worker_proc

        record = self.store.update_worker(
            worker_id,
            status=WorkerStatus.RUNNING,
            started_at=time.time(),
            pid=pid,
        )

        return WorkerStatusPayload.from_record(record)

    def _compute_spawn_depth(self, worker_id: str) -> int:
        """Walk the parent_worker_id chain to compute spawn depth."""
        depth = 0
        current_id = worker_id
        while current_id:
            record = self.store.get_worker(current_id)
            if record is None:
                break
            depth += 1
            current_id = record.parent_worker_id
        return depth

    def _on_worker_complete(
        self, worker_id: str, exit_code: int, error: Optional[str]
    ):
        """Callback when a worker process completes."""
        record = self.store.get_worker(worker_id)
        if record is None:
            return

        now = time.time()
        update: dict = {"ended_at": now, "exit_code": exit_code}

        # Persist retry count from the process
        proc = self._active_workers.get(worker_id)
        if proc and proc.retry_count > 0:
            update["retry_count"] = proc.retry_count

        if error and "timed out" in error.lower():
            update["status"] = WorkerStatus.TIMED_OUT
            update["error_message"] = error
        elif exit_code != 0:
            update["status"] = WorkerStatus.FAILED
            update["error_message"] = error or f"Process exited with code {exit_code}"
        else:
            result_path = Path(record.result_json_path)
            try:
                parse_result_file(result_path)
            except (FileNotFoundError, ValueError) as e:
                update["status"] = WorkerStatus.FAILED
                update["error_message"] = f"Result validation failed: {e}"
            else:
                update["status"] = WorkerStatus.SUCCEEDED

        record = self.store.update_worker(worker_id, **update)

        self._active_workers.pop(worker_id, None)
        if record and record.workflow_id is not None and record.stage_index is not None:
            try:
                self.workflow_engine.on_stage_complete(
                    worker_id, record.workflow_id, record.stage_index
                )
            except Exception:
                logger.exception("Error in workflow stage completion callback")

    def get_worker_status(self, worker_id: str) -> WorkerStatusPayload:
        record = self.store.get_worker(worker_id)
        if record is None:
            raise ValueError(f"Worker not found: {worker_id}")
        return WorkerStatusPayload.from_record(record)

    def list_workers(
        self,
        statuses: Optional[list[str]] = None,
        limit: int = 25,
    ) -> list[WorkerStatusPayload]:
        records = self.store.list_workers(statuses=statuses, limit=limit)
        return [WorkerStatusPayload.from_record(r) for r in records]

    def collect_worker_result(
        self,
        worker_id: str,
        include_logs: bool = False,
        log_tail_lines: int = 80,
    ) -> dict:
        record = self.store.get_worker(worker_id)
        if record is None:
            raise ValueError(f"Worker not found: {worker_id}")

        payload = WorkerStatusPayload.from_record(record).model_dump()

        result_path = Path(record.result_json_path)
        if result_path.exists():
            try:
                result = parse_result_file(result_path)
                payload["result"] = result.model_dump()
            except (FileNotFoundError, ValueError) as e:
                payload["result"] = None
                payload["result_error"] = str(e)
        else:
            payload["result"] = None

        if include_logs:
            payload["stdout_tail"] = self._tail_file(
                Path(record.stdout_path), log_tail_lines
            )
            payload["stderr_tail"] = self._tail_file(
                Path(record.stderr_path), log_tail_lines
            )

        return payload

    def cancel_worker(self, worker_id: str) -> WorkerStatusPayload:
        record = self.store.get_worker(worker_id)
        if record is None:
            raise ValueError(f"Worker not found: {worker_id}")

        if record.status.is_terminal:
            raise ValueError(
                f"Worker {worker_id} is already terminal ({record.status.value})"
            )

        proc = self._active_workers.pop(worker_id, None)
        if proc:
            proc.cancel()

        record = self.store.update_worker(
            worker_id,
            status=WorkerStatus.CANCELLED,
            ended_at=time.time(),
            error_message="Cancelled by user",
        )

        return WorkerStatusPayload.from_record(record)

    def cleanup_worker(
        self,
        worker_id: str,
        remove_branch: bool = True,
        remove_worktree_dir: bool = True,
    ) -> dict:
        record = self.store.get_worker(worker_id)
        if record is None:
            raise ValueError(f"Worker not found: {worker_id}")

        if not record.status.is_terminal:
            raise ValueError(
                f"Cannot cleanup non-terminal worker {worker_id} "
                f"(status: {record.status.value}). "
                f"Cancel or wait for completion first."
            )

        cleanup_summary = {
            "worker_id": worker_id,
            "worktree_removed": False,
            "branch_removed": False,
            "worker_dir_removed": False,
            "errors": [],
        }

        repo_path = Path(record.repo_path)
        worktree_path = Path(record.worktree_path)

        try:
            if worktree_path.exists():
                remove_worktree(repo_path, worktree_path)
            cleanup_summary["worktree_removed"] = True
        except GitError as e:
            logger.warning("Git error removing worktree for worker %s: %s", worker_id, e)
            cleanup_summary["errors"].append(f"Worktree removal: {e}")
        except Exception as e:
            cleanup_summary["errors"].append(f"Worktree removal: {e}")

        if remove_branch:
            try:
                delete_branch(repo_path, record.branch_name)
                cleanup_summary["branch_removed"] = True
            except GitError as e:
                logger.warning("Git error deleting branch %s for worker %s: %s",
                               record.branch_name, worker_id, e)
                cleanup_summary["errors"].append(f"Branch deletion: {e}")
            except Exception as e:
                cleanup_summary["errors"].append(f"Branch deletion: {e}")

        if remove_worktree_dir:
            try:
                worker_dir = Path(record.worker_dir)
                if worker_dir.exists():
                    shutil.rmtree(worker_dir)
                cleanup_summary["worker_dir_removed"] = True
            except Exception as e:
                cleanup_summary["errors"].append(f"Worker dir removal: {e}")

        if cleanup_summary["errors"]:
            self.store.update_worker(
                worker_id,
                status=WorkerStatus.CLEANUP_FAILED,
                error_message="; ".join(cleanup_summary["errors"]),
            )

        return cleanup_summary

    # --- Workflow delegation ---

    def create_workflow(self, **kwargs):
        return self.workflow_engine.create_workflow(**kwargs)

    def get_workflow_status(self, workflow_id: str):
        return self.workflow_engine.get_workflow_status(workflow_id)

    def list_workflows(self, **kwargs):
        return self.workflow_engine.list_workflows(**kwargs)

    def cancel_workflow(self, workflow_id: str):
        return self.workflow_engine.cancel_workflow(workflow_id)

    def collect_workflow_result(self, **kwargs):
        return self.workflow_engine.collect_workflow_result(**kwargs)

    def cleanup_workflow(self, workflow_id: str):
        return self.workflow_engine.cleanup_workflow(workflow_id)

    @staticmethod
    def _tail_file(path: Path, lines: int) -> str:
        if not path.exists():
            return ""
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                tail = deque(f, maxlen=lines)
            return "".join(tail).rstrip("\n")
        except Exception:
            return ""

    def close(self):
        self.store.close()
