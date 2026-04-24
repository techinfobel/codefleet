import json
import logging
import os
import re
import shutil
import subprocess
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
    WorkflowStatus,
    parse_result_file,
    supported_models_for_executor,
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
    materialize_result_from_stdout,
    write_result_schema,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_REASONING_EFFORT = "max"
DEFAULT_TIMEOUT = 600  # 10 minutes
DEFAULT_BASE_DIR = Path.home() / ".codex-fleet"
MAX_CONCURRENT_WORKERS = 50
DEFAULT_MAX_SPAWN_DEPTH = 2
DEFAULT_RATE_LIMIT_RETRIES = 3
DEFAULT_RATE_LIMIT_BASE_DELAY = 4.0
DEFAULT_RATE_LIMIT_MAX_DELAY = 60.0
DEFAULT_STALE_TIMEOUT = 180.0  # 3 minutes
DEFAULT_STALE_MAX_RESTARTS = 2
DEFAULT_HEARTBEAT_INTERVAL = 30.0

_EXECUTOR_REASONING_DEFAULTS = {
    ExecutorType.CODEX: "xhigh",      # OpenAI: low, medium, high, xhigh
    ExecutorType.GEMINI: None,         # Gemini CLI has no effort flag
    ExecutorType.CLAUDE: "high",       # Claude: low, medium, high, max
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
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
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
        self.heartbeat_interval = heartbeat_interval

        self._active_workers: dict[str, WorkerProcess] = {}
        self._workflow_engine = None
        self._recover_running_workers()

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

    @staticmethod
    def _read_cli_version(command_name: str) -> str:
        path = shutil.which(command_name)
        if path is None:
            return ""
        try:
            result = subprocess.run(
                [command_name, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return ""

        output = (result.stdout or result.stderr or "").strip()
        if not output:
            return ""
        return output.splitlines()[0].strip()

    @staticmethod
    def _auth_status_from_artifact(path: Path) -> str:
        try:
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                return "configured"
        except OSError:
            pass
        return "missing"

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
        executor_versions = {
            "codex": self._read_cli_version("codex"),
            "gemini": self._read_cli_version("gemini"),
            "claude": self._read_cli_version("claude"),
        }
        auth_status = {
            "codex": self._auth_status_from_artifact(Path.home() / ".codex" / "auth.json"),
            "gemini": self._auth_status_from_artifact(Path.home() / ".gemini" / "oauth_creds.json"),
            "claude": "unknown",
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
            **{f"{name}_version": executor_versions[name] for name in executor_versions},
            **{f"{name}_auth_status": auth_status[name] for name in auth_status},
            "auth_check_mode": "local_artifact",
            "claude_auth_note": (
                "Claude CLI login does not expose a reliable local auth artifact; "
                "use the live Claude smoke test for end-to-end verification."
            ),
            "git_found": git_path is not None,
            "git_path": git_path or "",
            "default_model": self.default_model,
            "default_gemini_model": self.default_gemini_model,
            "default_claude_model": self.default_claude_model,
            "default_executor": self.default_executor.value,
            "max_spawn_depth": self.max_spawn_depth,
            "heartbeat_interval": self.heartbeat_interval,
            "supported_models": {
                executor.value: list(supported_models_for_executor(executor))
                for executor in ExecutorType
            },
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
        else:
            model = str(model)

        allowed_models = supported_models_for_executor(executor_type)
        if model not in allowed_models:
            raise ValueError(
                f"Unsupported model '{model}' for executor '{executor_type.value}'. "
                f"Allowed models: {', '.join(allowed_models)}"
            )

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

        worktree_path = worker_dir / "worktree"
        try:
            create_worktree(repo, worktree_path, branch_name, base_ref)
        except Exception:
            self._cleanup_failed_creation(
                repo, worktree_path, branch_name, worker_dir
            )
            raise

        # Resolve base commit hash. Every stage gets its own fresh worktree
        # off its base_ref, so this is also the commit the agent's work
        # builds on. Used later for silent-failure detection and salvage.
        base_commit = None
        try:
            base_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(worktree_path),
                capture_output=True, text=True, timeout=10,
            )
            if base_result.returncode == 0:
                base_commit = base_result.stdout.strip()
        except Exception:
            logger.debug(
                "Failed to resolve base commit for worker %s",
                worker_id,
                exc_info=True,
            )

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
            "base_commit": base_commit,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Place prompt and result inside the worktree so sandboxed executors
        # (especially Codex) can access them without path violations.
        codefleet_dir = worktree_path / ".codefleet"
        codefleet_dir.mkdir(exist_ok=True)
        gitignore = codefleet_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n")
        prompt_path = codefleet_dir / "prompt.txt"
        result_json_path = codefleet_dir / "result.json"
        result_schema_path = codefleet_dir / "result_schema.json"
        prompt_path.write_text(prompt, encoding="utf-8")
        write_result_schema(result_schema_path)

        cmd = build_worker_command(
            executor=executor_type.value,
            prompt_path=prompt_path,
            result_json_path=result_json_path,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_args=merged_extra_args,
            base_commit=base_commit,
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
            last_heartbeat_at=None,
            last_activity_at=None,
            timeout_seconds=timeout_seconds,
            command_json=json.dumps(cmd),
            # Full prompt is on disk at prompt_path; store only a preview
            # in the DB to avoid slow inserts with large prompts.
            prompt=prompt[:500] + "… [truncated]" if len(prompt) > 500 else prompt,
            result_json_path=str(result_json_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            prompt_path=str(prompt_path),
            meta_path=str(meta_path),
            retry_count=0,
            parent_worker_id=parent_worker_id,
            tags=tags or [],
            metadata=metadata or {},
            heartbeat_message=None,
            workflow_id=workflow_id,
            stage_index=stage_index,
        )

        try:
            self.store.insert_worker(record)
        except Exception:
            self._cleanup_failed_creation(
                repo, worktree_path, branch_name, worker_dir
            )
            raise

        try:
            worker_proc = WorkerProcess(
                worker_id=worker_id,
                command=cmd,
                cwd=worktree_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout_seconds=timeout_seconds,
                on_complete=self._on_worker_complete,
                on_heartbeat=self._on_worker_heartbeat,
                started_at_wall=now,
                last_activity_wall=now,
                max_retries=self.rate_limit_max_retries,
                retry_base_delay=self.rate_limit_base_delay,
                retry_max_delay=self.rate_limit_max_delay,
                stale_timeout=self.stale_timeout,
                stale_max_restarts=self.stale_max_restarts,
                heartbeat_interval=self.heartbeat_interval,
            )
            pid = worker_proc.start()
        except Exception:
            self._cleanup_failed_creation(
                repo, worktree_path, branch_name, worker_dir
            )
            raise

        self._active_workers[worker_id] = worker_proc

        record = self.store.update_worker(
            worker_id,
            status=WorkerStatus.RUNNING,
            started_at=time.time(),
            last_heartbeat_at=time.time(),
            last_activity_at=time.time(),
            heartbeat_message="Worker launching",
            pid=pid,
        )

        return WorkerStatusPayload.from_record(record)

    def _recover_running_workers(self) -> None:
        """Reattach monitors to workers that were running before restart."""
        try:
            running_workers = self.store.list_workers(statuses=["running"], limit=10000)
        except Exception:
            logger.exception("Failed to load running workers during supervisor recovery")
            return

        for record in running_workers:
            try:
                self._recover_running_worker(record)
            except Exception:
                logger.exception(
                    "Failed to recover running worker %s during supervisor startup",
                    record.worker_id,
                )

    def _recover_running_worker(self, record: WorkerRecord) -> None:
        if record.pid and WorkerProcess.pid_exists(record.pid):
            cmd = json.loads(record.command_json)
            worker_proc = WorkerProcess(
                worker_id=record.worker_id,
                command=cmd,
                cwd=Path(record.worktree_path),
                stdout_path=Path(record.stdout_path),
                stderr_path=Path(record.stderr_path),
                timeout_seconds=record.timeout_seconds,
                on_complete=self._on_worker_complete,
                on_heartbeat=self._on_worker_heartbeat,
                started_at_wall=record.started_at,
                last_activity_wall=record.last_activity_at,
                max_retries=self.rate_limit_max_retries,
                retry_base_delay=self.rate_limit_base_delay,
                retry_max_delay=self.rate_limit_max_delay,
                stale_timeout=self.stale_timeout,
                stale_max_restarts=self.stale_max_restarts,
                heartbeat_interval=self.heartbeat_interval,
            )
            worker_proc.attach(record.pid)
            self._active_workers[record.worker_id] = worker_proc
            self.store.update_worker(
                record.worker_id,
                last_heartbeat_at=time.time(),
                heartbeat_message="Recovered worker monitor after supervisor restart",
            )
            return

        self._finalize_orphaned_running_worker(record)

    def _finalize_orphaned_running_worker(self, record: WorkerRecord) -> None:
        """Reconcile a worker left running in the DB with no live process."""
        now = time.time()
        update: dict = {
            "ended_at": now,
            "last_heartbeat_at": now,
        }
        result_path = Path(record.result_json_path)

        try:
            parsed_result = parse_result_file(result_path)
        except (FileNotFoundError, ValueError) as e:
            recovery_mode = self._materialize_or_salvage_result(record, e)
            if recovery_mode == "materialized":
                parsed_result = parse_result_file(result_path)
                update.update(
                    self._result_completion_update(
                        record,
                        parsed_result,
                        (
                            "Recovered completed worker after supervisor restart "
                            "via parsed stdout result"
                        ),
                    )
                )
            elif recovery_mode == "salvaged":
                update["status"] = WorkerStatus.SUCCEEDED
                update["heartbeat_message"] = (
                    "Recovered completed worker after supervisor restart via worktree salvage"
                )
                update["error_message"] = (
                    f"Recovered terminal result by salvaging the worktree after supervisor restart: {e}"
                )
            else:
                update["status"] = WorkerStatus.FAILED
                update["error_message"] = (
                    "Worker was marked running before supervisor restart, "
                    "but no live process or valid result was found."
                )
                update["heartbeat_message"] = update["error_message"]
        else:
            update.update(
                self._result_completion_update(
                    record,
                    parsed_result,
                    "Recovered completed worker after supervisor restart",
                )
            )

        record = self.store.update_worker(record.worker_id, **update)

        if record and record.workflow_id is not None and record.stage_index is not None:
            try:
                self.workflow_engine.on_stage_complete(
                    record.worker_id, record.workflow_id, record.stage_index
                )
            except Exception:
                logger.exception(
                    "Error in workflow stage completion callback for "
                    "recovered worker %s",
                    record.worker_id,
                )
                try:
                    self.store.update_workflow(
                        record.workflow_id,
                        status=WorkflowStatus.FAILED,
                        completed_at=time.time(),
                        error_message=(
                            f"Internal error in stage completion callback "
                            f"for recovered worker {record.worker_id}"
                        ),
                    )
                except Exception:
                    logger.exception(
                        "Failed to mark workflow %s as failed after "
                        "recovery callback error",
                        record.workflow_id,
                    )

    def _cleanup_failed_creation(
        self, repo: Path, worktree_path: Path, branch_name: str, worker_dir: Path
    ):
        """Clean up worktree and branch after a failed worker creation."""
        try:
            if worktree_path.exists():
                remove_worktree(repo, worktree_path)
        except Exception:
            logger.debug(
                "Cleanup: failed to remove worktree %s", worktree_path, exc_info=True
            )
        try:
            delete_branch(repo, branch_name)
        except Exception:
            logger.debug(
                "Cleanup: failed to delete branch %s", branch_name, exc_info=True
            )
        try:
            if worker_dir.exists():
                shutil.rmtree(worker_dir)
        except Exception:
            logger.debug(
                "Cleanup: failed to remove worker dir %s", worker_dir, exc_info=True
            )

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

    @staticmethod
    def _is_codefleet_internal_path(path: str) -> bool:
        return path == ".codefleet" or path.startswith(".codefleet/")

    @classmethod
    def _status_user_paths(cls, status_output: str) -> list[str]:
        """Extract non-codefleet paths from git status --porcelain output."""
        paths: list[str] = []
        for line in status_output.splitlines():
            if not line.strip():
                continue
            raw_path = line[3:].strip() if len(line) > 3 else line.strip()
            candidates = (
                raw_path.split(" -> ", 1) if " -> " in raw_path else [raw_path]
            )
            for candidate in candidates:
                path = candidate.strip().strip('"')
                if path and not cls._is_codefleet_internal_path(path):
                    paths.append(path)
        return paths

    def _commit_uncommitted_changes(
        self, record: WorkerRecord, message: str
    ) -> tuple[bool, Optional[str]]:
        """Commit non-internal worktree changes so they survive cleanup.

        Returns ``(committed, error_message)``.  A missing/clean worktree is not
        an error; git command failures are returned so completion can fail
        loudly instead of silently losing work.
        """
        worktree = Path(record.worktree_path)
        if not worktree.exists():
            return False, None
        if not is_git_repo(worktree):
            return False, None

        try:
            status_result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=str(worktree),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            return False, f"git status failed while checking uncommitted work: {e}"

        if status_result.returncode != 0:
            detail = (status_result.stderr or status_result.stdout).strip()
            return False, f"git status failed while checking uncommitted work: {detail}"

        if not self._status_user_paths(status_result.stdout):
            return False, None

        add_result = subprocess.run(
            ["git", "add", "-A", "--", "."],
            cwd=str(worktree),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if add_result.returncode != 0:
            detail = (add_result.stderr or add_result.stdout).strip()
            return False, f"git add failed while preserving worker changes: {detail}"

        # .codefleet contains prompts/results and must never be committed.
        subprocess.run(
            ["git", "reset", "-q", "--", ".codefleet"],
            cwd=str(worktree),
            capture_output=True,
            text=True,
            timeout=10,
        )

        staged_result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(worktree),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if staged_result.returncode != 0:
            detail = (staged_result.stderr or staged_result.stdout).strip()
            return False, f"git diff --cached failed before auto-commit: {detail}"

        staged_user_paths = [
            p.strip()
            for p in staged_result.stdout.splitlines()
            if p.strip() and not self._is_codefleet_internal_path(p.strip())
        ]
        if not staged_user_paths:
            subprocess.run(
                ["git", "reset", "-q"],
                cwd=str(worktree),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return False, None

        env = os.environ.copy()
        env.setdefault("GIT_AUTHOR_NAME", "codefleet")
        env.setdefault("GIT_AUTHOR_EMAIL", "codefleet@example.invalid")
        env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
        env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])

        commit_result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(worktree),
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if commit_result.returncode != 0:
            detail = (commit_result.stderr or commit_result.stdout).strip()
            return False, f"git commit failed while preserving worker changes: {detail}"

        logger.info(
            "Auto-committed %d uncommitted files for worker %s",
            len(staged_user_paths),
            record.worker_id,
        )
        return True, None

    def _git_changes_since_base(
        self, record: WorkerRecord
    ) -> tuple[bool, list[str], list[str], Optional[str]]:
        """Return (verified, commits, files, error) for work after base_commit."""
        worktree = Path(record.worktree_path)
        base_commit = self._read_base_commit(record)
        if not worktree.exists():
            return False, [], [], f"worktree not found: {worktree}"
        if not base_commit:
            return False, [], [], "base commit unavailable"

        try:
            log_res = subprocess.run(
                ["git", "log", "--format=%H", f"{base_commit}..HEAD"],
                cwd=str(worktree),
                capture_output=True,
                text=True,
                timeout=10,
            )
            diff_res = subprocess.run(
                ["git", "diff", "--name-only", base_commit, "HEAD"],
                cwd=str(worktree),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            return False, [], [], f"git verification failed: {e}"

        if log_res.returncode != 0:
            detail = (log_res.stderr or log_res.stdout).strip()
            return False, [], [], f"git log failed: {detail}"
        if diff_res.returncode != 0:
            detail = (diff_res.stderr or diff_res.stdout).strip()
            return False, [], [], f"git diff failed: {detail}"

        commits = [h.strip() for h in log_res.stdout.splitlines() if h.strip()]
        files = [
            f.strip()
            for f in diff_res.stdout.splitlines()
            if f.strip() and not self._is_codefleet_internal_path(f.strip())
        ]
        return True, commits, files, None

    def _normalize_result_git_metadata(
        self, record: WorkerRecord, result, commits: list[str], files: list[str]
    ) -> None:
        """Persist verified git metadata back into result.json for downstream stages."""
        if result.commits == commits and result.files_changed == files:
            return
        try:
            data = result.model_dump(mode="json")
            data["commits"] = commits
            data["files_changed"] = files
            Path(record.result_json_path).write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.debug(
                "Failed to normalize result git metadata for worker %s",
                record.worker_id,
                exc_info=True,
            )

    def _result_completion_update(
        self, record: WorkerRecord, result, success_message: str
    ) -> dict:
        verdict, verdict_msg = self._verify_worker_produced_work(record, result)
        if verdict == "ok":
            return {
                "status": WorkerStatus.SUCCEEDED,
                "heartbeat_message": success_message,
                "error_message": None,
            }
        return {
            "status": WorkerStatus.FAILED,
            "heartbeat_message": verdict_msg,
            "error_message": verdict_msg,
        }

    def _salvage_result(self, record: WorkerRecord) -> bool:
        """Check if a worker produced meaningful changes despite invalid result.json.

        If changes are detected, writes a synthetic result.json so downstream
        workflow stages can still access the work. Returns True if salvage succeeded.
        """
        worktree = Path(record.worktree_path)
        if not worktree.exists():
            return False

        _, commit_error = self._commit_uncommitted_changes(
            record, "auto-commit: uncommitted agent work (salvaged)"
        )
        if commit_error:
            logger.warning(
                "Could not salvage worker %s because auto-commit failed: %s",
                record.worker_id,
                commit_error,
            )
            return False

        verified, commits, files_changed, verify_error = (
            self._git_changes_since_base(record)
        )
        if not verified:
            logger.debug(
                "Could not verify salvaged git changes for worker %s: %s",
                record.worker_id,
                verify_error,
            )
            return False

        if not files_changed and not commits:
            return False

        result_path = Path(record.result_json_path)
        salvaged = {
            "summary": (
                "Worker completed but did not write a valid result.json. "
                "Changes were detected and salvaged from the worktree."
            ),
            "files_changed": files_changed,
            "tests": [],
            "commits": commits,
            "status": "completed",
            "next_steps": ["Review salvaged changes manually"],
        }
        try:
            result_path.write_text(
                json.dumps(salvaged, indent=2), encoding="utf-8"
            )
            logger.info(
                "Salvaged result for worker %s: %d files changed",
                record.worker_id,
                len(files_changed),
            )
            return True
        except Exception:
            logger.warning(
                "Failed to write salvaged result for %s",
                record.worker_id,
                exc_info=True,
            )
            return False

    def _verify_worker_produced_work(
        self, record: WorkerRecord, result
    ) -> tuple[str, str]:
        """Cross-check an agent's self-reported result against git state.

        Returns ("ok", message) if the worker legitimately completed, or
        ("failed", message) if the worker claimed completed but produced no
        visible work, or honestly reported blocked.
        """
        if result.status.value == "blocked":
            reason = (result.summary or "").strip() or "no reason given"
            return (
                "failed",
                f"Agent reported status=blocked: {reason[:500]}",
            )

        _, commit_error = self._commit_uncommitted_changes(
            record, "auto-commit: uncommitted agent work"
        )
        if commit_error:
            return (
                "failed",
                "Agent produced uncommitted changes, but codefleet could not "
                f"commit them safely: {commit_error}",
            )

        verified, git_commits, git_files, verify_error = self._git_changes_since_base(
            record
        )
        if verified:
            if not git_commits and not git_files:
                return (
                    "failed",
                    "Agent claimed status=completed but produced no commits "
                    "and no file changes in git — likely a silent failure. "
                    "Check stdout/stderr for sandbox, permission, or auth errors.",
                )
            self._normalize_result_git_metadata(
                record, result, git_commits, git_files
            )
            return ("ok", "Worker completed successfully")

        logger.debug(
            "Falling back to self-reported result metadata for worker %s: %s",
            record.worker_id,
            verify_error,
        )
        produced_commits = bool(result.commits)
        produced_files = bool(result.files_changed)
        if not produced_commits and not produced_files:
            return (
                "failed",
                "Agent claimed status=completed but produced no commits "
                "and no file changes — likely a silent failure. "
                "Check stdout/stderr for sandbox, permission, or auth errors.",
            )

        return ("ok", "Worker completed successfully")

    @staticmethod
    def _read_base_commit(record: WorkerRecord) -> Optional[str]:
        """Read base_commit from a worker's meta.json, if present."""
        meta_path = Path(record.meta_path)
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = meta.get("base_commit")
        return value if isinstance(value, str) and value else None

    def _on_worker_complete(
        self, worker_id: str, exit_code: int, error: Optional[str]
    ):
        """Callback when a worker process completes."""
        record = self.store.get_worker(worker_id)
        if record is None:
            return

        now = time.time()
        update: dict = {
            "ended_at": now,
            "exit_code": exit_code,
            "last_heartbeat_at": now,
        }

        # Persist retry count from the process
        proc = self._active_workers.get(worker_id)
        if proc and proc.retry_count > 0:
            update["retry_count"] = proc.retry_count

        if error and "timed out" in error.lower():
            update["status"] = WorkerStatus.TIMED_OUT
            update["error_message"] = error
            update["heartbeat_message"] = error
        elif exit_code != 0:
            update["status"] = WorkerStatus.FAILED
            update["error_message"] = error or f"Process exited with code {exit_code}"
            update["heartbeat_message"] = (
                error or f"Process exited with code {exit_code}"
            )
        else:
            result_path = Path(record.result_json_path)
            try:
                result = parse_result_file(result_path)
            except (FileNotFoundError, ValueError) as e:
                recovery_mode = self._materialize_or_salvage_result(record, e)
                if recovery_mode == "materialized":
                    result = parse_result_file(result_path)
                    update.update(
                        self._result_completion_update(
                            record,
                            result,
                            "Worker completed successfully via parsed stdout result",
                        )
                    )
                elif recovery_mode == "salvaged":
                    update["status"] = WorkerStatus.SUCCEEDED
                    update["error_message"] = (
                        f"Result file invalid ({e}), but changes detected "
                        f"in worktree. Salvaged result written."
                    )
                    update["heartbeat_message"] = (
                        "Worker completed with salvaged result"
                    )
                else:
                    update["status"] = WorkerStatus.FAILED
                    update["error_message"] = f"Result validation failed: {e}"
                    update["heartbeat_message"] = f"Result validation failed: {e}"
            else:
                update.update(
                    self._result_completion_update(
                        record,
                        result,
                        "Worker completed successfully",
                    )
                )

        record = self.store.update_worker(worker_id, **update)

        self._active_workers.pop(worker_id, None)
        if record and record.workflow_id is not None and record.stage_index is not None:
            try:
                self.workflow_engine.on_stage_complete(
                    worker_id, record.workflow_id, record.stage_index
                )
            except Exception:
                logger.exception(
                    "Error in workflow stage completion callback for worker %s",
                    worker_id,
                )
                # Mark workflow as FAILED so it doesn't hang forever
                try:
                    self.store.update_workflow(
                        record.workflow_id,
                        status=WorkflowStatus.FAILED,
                        completed_at=time.time(),
                        error_message=(
                            f"Internal error in stage completion callback "
                            f"for worker {worker_id}"
                        ),
                    )
                except Exception:
                    logger.exception(
                        "Failed to mark workflow %s as failed after callback error",
                        record.workflow_id,
                )

    def _on_worker_heartbeat(self, worker_id: str, payload: dict) -> None:
        """Persist liveness information from the worker monitor."""
        record = self.store.get_worker(worker_id)
        if record is None or record.status.is_terminal:
            return
        self.store.update_worker(worker_id, **payload)

    def _materialize_or_salvage_result(
        self,
        record: WorkerRecord,
        original_error: Exception,
    ) -> Optional[str]:
        """Try executor-native stdout parsing before falling back to salvage."""
        if record.executor in {ExecutorType.GEMINI, ExecutorType.CLAUDE}:
            try:
                materialize_result_from_stdout(
                    executor=record.executor.value,
                    stdout_path=Path(record.stdout_path),
                    result_json_path=Path(record.result_json_path),
                )
                parse_result_file(Path(record.result_json_path))
                return "materialized"
            except Exception:
                logger.info(
                    "Could not materialize %s result for worker %s after %s",
                    record.executor.value,
                    record.worker_id,
                    original_error,
                    exc_info=True,
                )

        if self._salvage_result(record):
            return "salvaged"
        return None

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
