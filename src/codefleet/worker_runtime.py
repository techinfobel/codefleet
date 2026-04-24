import json
import logging
import os
import signal
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def write_result_schema(path: Path) -> None:
    """Write the shared result schema used for structured final output."""
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "files_changed": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
            "tests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["passed", "failed", "not_run"],
                        },
                        "details": {"type": "string"},
                    },
                    "required": ["command", "status", "details"],
                    "additionalProperties": False,
                },
                "default": [],
            },
            "commits": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
            "next_steps": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
            "status": {
                "type": "string",
                "enum": ["completed", "completed_no_changes", "blocked"],
            },
        },
        "required": [
            "summary",
            "files_changed",
            "tests",
            "commits",
            "next_steps",
            "status",
        ],
        "additionalProperties": False,
    }
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def get_codex_path() -> str | None:
    return shutil.which("codex")


def get_gemini_path() -> str | None:
    return shutil.which("gemini")


def get_claude_path() -> str | None:
    return shutil.which("claude")


_SHARED_INSTRUCTIONS = (
    "Read the task prompt from {prompt_path}.\n"
    "Work in the current directory (a git worktree). Your worktree started at "
    "base commit {base_ref}.\n"
    "Do NOT commit or modify anything in the .codefleet/ directory.\n"
    "Run relevant tests where appropriate.\n"
    "\n"
    "When you finish the task:\n"
    "  1. Stage and commit all changed files with a descriptive commit message.\n"
    "  2. Run `git log --oneline {base_ref}..HEAD` and include its output in "
    "`summary`.\n"
    "  3. Run `git diff --name-only {base_ref} HEAD` and put the paths in "
    "`files_changed`.\n"
    "  4. Put the full commit hashes in `commits`.\n"
    "  5. Set `status` to \"completed\".\n"
    "\n"
    "If the task is already satisfied and genuinely requires no code changes "
    "(e.g. an audit that found nothing, an idempotent refactor already applied, "
    "a requirement already met):\n"
    "  - Set `status` to \"completed_no_changes\".\n"
    "  - In `summary`, describe what you checked and why no changes were "
    "needed.\n"
    "  - Leave `commits` and `files_changed` as [].\n"
    "\n"
    "If you cannot complete the task — sandbox denial, missing tool, permission "
    "error, authentication failure, unclear requirements, or any other blocker — "
    "DO NOT exit with an empty success. Instead:\n"
    "  - Set `status` to \"blocked\".\n"
    "  - In `summary`, describe exactly what blocked you (the command that "
    "failed, the error message, what you tried).\n"
    "  - Leave `commits` and `files_changed` as [] if you made none.\n"
    "\n"
    "A `completed` status with empty `commits` and empty `files_changed` is "
    "treated as a silent failure by the supervisor. If you truly did no work, "
    "use `completed_no_changes` (success) or `blocked` (failure) — never a "
    "bare `completed` with nothing to show."
)


def _codex_result_instruction(prompt_path: Path, base_ref: str) -> str:
    """Build the Codex-specific instruction string."""
    return (
        _SHARED_INSTRUCTIONS.format(prompt_path=prompt_path, base_ref=base_ref)
        + "\n\nReturn a single JSON object as your final response matching the "
          "provided output schema. Do not wrap the JSON in markdown."
    )


def _stream_result_instruction(prompt_path: Path, base_ref: str) -> str:
    """Build the Gemini/Claude instruction string for JSON final responses."""
    return (
        _SHARED_INSTRUCTIONS.format(prompt_path=prompt_path, base_ref=base_ref)
        + "\n\nReturn exactly one JSON object as your final response with this "
          'schema: {"summary": "string", "files_changed": ["path"], '
          '"tests": [{"command": "string", "status": "passed|failed|not_run", '
          '"details": "string"}], '
          '"commits": ["hash"], "next_steps": ["string"], '
          '"status": "completed|completed_no_changes|blocked"}. '
          "Do not wrap the JSON in markdown."
    )


def build_codex_command(
    prompt_path: Path,
    result_json_path: Path,
    model: str = "gpt-5.5",
    reasoning_effort: str = "xhigh",
    extra_args: Optional[list[str]] = None,
    base_commit: Optional[str] = None,
) -> list[str]:
    """Build the codex exec command."""
    instruction = _codex_result_instruction(prompt_path, base_commit or "HEAD")
    result_schema_path = result_json_path.with_name("result_schema.json")

    cmd = [
        "codex",
        "exec",
        "--full-auto",
        "--output-schema",
        str(result_schema_path),
        "--output-last-message",
        str(result_json_path),
    ]
    if model:
        cmd.extend(["--model", model])
    if reasoning_effort:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(instruction)
    return cmd


def build_gemini_command(
    prompt_path: Path,
    result_json_path: Path,
    model: str = "gemini-3.1-pro-preview",
    extra_args: Optional[list[str]] = None,
    base_commit: Optional[str] = None,
) -> list[str]:
    """Build the gemini CLI command."""
    instruction = _stream_result_instruction(prompt_path, base_commit or "HEAD")

    cmd = [
        "gemini", "-p", instruction,
        "--approval-mode", "yolo",
        "--sandbox", "false",
        "--output-format", "stream-json",
    ]
    if model:
        cmd.extend(["-m", model])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def build_claude_command(
    prompt_path: Path,
    result_json_path: Path,
    model: str = "claude-sonnet-4-6",
    effort: str = "high",
    extra_args: Optional[list[str]] = None,
    base_commit: Optional[str] = None,
) -> list[str]:
    """Build the claude CLI command."""
    instruction = _stream_result_instruction(prompt_path, base_commit or "HEAD")

    cmd = [
        "claude", "-p", instruction,
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--verbose",
    ]
    if model:
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["--effort", effort])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def build_worker_command(
    executor: str,
    prompt_path: Path,
    result_json_path: Path,
    model: str,
    reasoning_effort: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    base_commit: Optional[str] = None,
) -> list[str]:
    """Dispatch to the correct command builder based on executor type."""
    if executor == "gemini":
        return build_gemini_command(
            prompt_path=prompt_path,
            result_json_path=result_json_path,
            model=model,
            extra_args=extra_args,
            base_commit=base_commit,
        )
    if executor == "claude":
        return build_claude_command(
            prompt_path=prompt_path,
            result_json_path=result_json_path,
            model=model,
            effort=reasoning_effort or "high",
            extra_args=extra_args,
            base_commit=base_commit,
        )
    # Default: codex
    return build_codex_command(
        prompt_path=prompt_path,
        result_json_path=result_json_path,
        model=model,
        reasoning_effort=reasoning_effort or "xhigh",
        extra_args=extra_args,
        base_commit=base_commit,
    )


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1] == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_json_object(text: str) -> dict:
    candidates = [text.strip(), _strip_json_fence(text)]
    stripped = _strip_json_fence(text)
    if "{" in stripped and "}" in stripped:
        candidates.append(stripped[stripped.find("{"): stripped.rfind("}") + 1].strip())

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError("Could not extract a JSON object from executor output")


def parse_executor_result_from_stdout(executor: str, stdout_path: Path) -> dict:
    """Parse a final JSON result from Gemini/Claude stream-json stdout."""
    if not stdout_path.exists():
        raise FileNotFoundError(f"Worker stdout not found: {stdout_path}")

    raw = stdout_path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        raise ValueError("Worker stdout is empty")

    assistant_chunks: list[str] = []
    assistant_messages: list[str] = []
    result_messages: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        if executor == "gemini":
            if payload.get("type") == "message" and payload.get("role") == "assistant":
                content = payload.get("content")
                if isinstance(content, str):
                    if payload.get("delta"):
                        assistant_chunks.append(content)
                    else:
                        assistant_messages.append(content)
            continue

        if executor == "claude":
            if payload.get("type") == "result" and isinstance(payload.get("result"), str):
                result_messages.append(payload["result"])
            elif payload.get("type") == "assistant":
                message = payload.get("message", {})
                content = message.get("content", [])
                if isinstance(content, list):
                    text = "".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                    if text:
                        assistant_messages.append(text)
            continue

    if executor == "gemini":
        candidate = assistant_messages[-1] if assistant_messages else "".join(assistant_chunks)
    elif executor == "claude":
        candidate = result_messages[-1] if result_messages else (
            assistant_messages[-1] if assistant_messages else ""
        )
    else:
        raise ValueError(f"Unsupported executor for stdout parsing: {executor}")

    if not candidate:
        raise ValueError(f"No assistant JSON result found in {executor} stdout")
    return _extract_json_object(candidate)


def materialize_result_from_stdout(
    executor: str,
    stdout_path: Path,
    result_json_path: Path,
) -> None:
    """Write the parsed final executor JSON output to result.json."""
    data = parse_executor_result_from_stdout(executor, stdout_path)
    result_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class WorkerProcess:
    """Manages a single worker subprocess with automatic rate-limit retry."""

    _RATE_LIMIT_PATTERNS = [
        "429",
        "rate limit",
        "rate_limit",
        "resource_exhausted",
        "too many requests",
        "quota exceeded",
    ]
    _AUTH_REQUIRED_PATTERNS = [
        "opening authentication page in your browser",
        "login required",
        "please login",
        "please log in",
        "authentication required",
        "not logged in",
        "please run /login",
    ]

    def __init__(
        self,
        worker_id: str,
        command: list[str],
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        timeout_seconds: int,
        on_complete: Optional[Callable[[str, int, Optional[str]], None]] = None,
        on_heartbeat: Optional[Callable[[str, dict], None]] = None,
        started_at_wall: Optional[float] = None,
        last_activity_wall: Optional[float] = None,
        max_retries: int = 0,
        retry_base_delay: float = 4.0,
        retry_max_delay: float = 60.0,
        stale_timeout: float = 120.0,
        stale_max_restarts: int = 2,
        heartbeat_interval: float = 30.0,
    ):
        self.worker_id = worker_id
        self.command = command
        self.cwd = cwd
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.timeout_seconds = timeout_seconds
        self.on_complete = on_complete
        self.on_heartbeat = on_heartbeat
        self.started_at_wall = started_at_wall
        self.last_activity_wall = last_activity_wall
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.stale_timeout = stale_timeout
        self.stale_max_restarts = stale_max_restarts
        self.heartbeat_interval = heartbeat_interval
        self._process: Optional[subprocess.Popen] = None
        self._attached_pid: Optional[int] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._retry_count = 0
        self._stale_restarts = 0
        self._cancelled = threading.Event()

    @property
    def pid(self) -> Optional[int]:
        if self._process is not None:
            return self._process.pid
        return self._attached_pid

    @property
    def retry_count(self) -> int:
        return self._retry_count

    def start(self) -> int:
        """Start the worker process. Returns the PID."""
        stdout_f = open(self.stdout_path, "w")
        stderr_f = open(self.stderr_path, "w")

        try:
            self._process = subprocess.Popen(
                self.command,
                cwd=str(self.cwd),
                stdin=subprocess.DEVNULL,
                stdout=stdout_f,
                stderr=stderr_f,
                start_new_session=True,
            )
            if self.started_at_wall is None:
                self.started_at_wall = time.time()
            if self.last_activity_wall is None:
                self.last_activity_wall = self.started_at_wall
        except Exception:
            stdout_f.close()
            stderr_f.close()
            raise

        self._monitor_thread = threading.Thread(
            target=self._monitor,
            args=(stdout_f, stderr_f),
            daemon=True,
        )
        self._monitor_thread.start()

        return self._process.pid

    @staticmethod
    def pid_exists(pid: int) -> bool:
        """Check whether a process currently exists."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def attach(self, pid: int) -> int:
        """Attach monitoring to an already-running worker process by PID."""
        self._attached_pid = pid
        if self.started_at_wall is None:
            self.started_at_wall = time.time()
        if self.last_activity_wall is None:
            self.last_activity_wall = self.started_at_wall
        stdout_f = open(self.stdout_path, "a")
        stderr_f = open(self.stderr_path, "a")
        self._monitor_thread = threading.Thread(
            target=self._monitor,
            args=(stdout_f, stderr_f),
            daemon=True,
        )
        self._monitor_thread.start()
        return pid

    def _is_rate_limited(self) -> bool:
        """Check stderr for rate-limit indicators."""
        try:
            text = self.stderr_path.read_text(
                encoding="utf-8", errors="replace"
            ).lower()
            return any(p in text for p in self._RATE_LIMIT_PATTERNS)
        except Exception:
            return False

    def _output_size(self) -> int:
        """Return combined size of stdout + stderr files."""
        total = 0
        for p in (self.stdout_path, self.stderr_path):
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    def _detect_auth_required(self) -> bool:
        """Check stdout/stderr for authentication prompts that need user action."""
        try:
            text = []
            for path in (self.stdout_path, self.stderr_path):
                if path.exists():
                    text.append(path.read_text(encoding="utf-8", errors="replace"))
            combined = "\n".join(text).lower()
            return any(pattern in combined for pattern in self._AUTH_REQUIRED_PATTERNS)
        except Exception:
            return False

    def _restart_process(self, stdout_f, stderr_f, reason: str):
        """Kill current process and restart in the same worktree.

        Returns new (stdout_f, stderr_f) file handles.
        """
        self._terminate()
        stdout_f.close()
        stderr_f.close()
        with open(self.stderr_path, "a") as f:
            f.write(f"\n--- [codefleet] {reason} ---\n")
        stdout_f = open(self.stdout_path, "w")
        stderr_f = open(self.stderr_path, "a")
        self._attached_pid = None
        self._process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            stdin=subprocess.DEVNULL,
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,
        )
        self.started_at_wall = time.time()
        self.last_activity_wall = self.started_at_wall
        return stdout_f, stderr_f

    def _emit_heartbeat(
        self,
        *,
        now_wall: float,
        last_activity_wall: float,
        message: str,
    ) -> None:
        """Report worker liveness without writing into child stdout/stderr."""
        if self.on_heartbeat is None:
            return
        try:
            self.on_heartbeat(
                self.worker_id,
                {
                    "last_heartbeat_at": now_wall,
                    "last_activity_at": last_activity_wall,
                    "heartbeat_message": message,
                    "retry_count": self._retry_count,
                },
            )
        except Exception:
            logger.debug(
                "Heartbeat callback failed for worker %s",
                self.worker_id,
                exc_info=True,
            )

    def _monitor(self, stdout_f, stderr_f):
        """Monitor the process for completion, stale detection, or rate-limit retry."""
        exit_code = -1
        error = None
        last_output_size = self._output_size()
        last_activity = time.monotonic()
        now_wall = time.time()
        started_wall = self.started_at_wall or now_wall
        last_activity_wall = self.last_activity_wall or started_wall
        last_heartbeat = time.monotonic()
        self._emit_heartbeat(
            now_wall=now_wall,
            last_activity_wall=last_activity_wall,
            message=(
                "Recovered worker monitor after supervisor restart"
                if self._attached_pid is not None
                else "Worker started"
            ),
        )
        try:
            while True:
                try:
                    exit_code = self._wait_for_exit(timeout=1.0)

                    # Check for rate limiting and retry
                    if (
                        exit_code != 0
                        and self._retry_count < self.max_retries
                        and not self._cancelled.is_set()
                    ):
                        stdout_f.close()
                        stderr_f.close()
                        if self._is_rate_limited():
                            self._retry_count += 1
                            delay = min(
                                self.retry_base_delay
                                * (2 ** (self._retry_count - 1)),
                                self.retry_max_delay,
                            )
                            logger.info(
                                "Worker %s rate-limited, retry %d/%d in %.0fs",
                                self.worker_id,
                                self._retry_count,
                                self.max_retries,
                                delay,
                            )
                            with open(self.stderr_path, "a") as f:
                                f.write(
                                    f"\n--- [codefleet] Rate limited. "
                                    f"Retry {self._retry_count}/{self.max_retries} "
                                    f"after {delay:.0f}s backoff ---\n"
                                )
                            if self._cancelled.wait(timeout=delay):
                                exit_code = -1
                                error = "Cancelled during retry backoff"
                                return

                            stdout_f = open(self.stdout_path, "w")
                            stderr_f = open(self.stderr_path, "a")
                            self._process = subprocess.Popen(
                                self.command,
                                cwd=str(self.cwd),
                                stdin=subprocess.DEVNULL,
                                stdout=stdout_f,
                                stderr=stderr_f,
                                start_new_session=True,
                            )
                            last_output_size = self._output_size()
                            last_activity = time.monotonic()
                            last_activity_wall = time.time()
                            self._emit_heartbeat(
                                now_wall=last_activity_wall,
                                last_activity_wall=last_activity_wall,
                                message=(
                                    f"Retrying after rate limit "
                                    f"({self._retry_count}/{self.max_retries})"
                                ),
                            )
                            continue

                    return
                except subprocess.TimeoutExpired:
                    # Check for activity (output growth)
                    current_size = self._output_size()
                    now = time.monotonic()
                    now_wall = time.time()
                    if now_wall - started_wall >= self.timeout_seconds:
                        self._terminate()
                        error = f"Worker timed out after {self.timeout_seconds}s"
                        return
                    if current_size != last_output_size:
                        last_output_size = current_size
                        last_activity = now
                        last_activity_wall = now_wall
                    if self._detect_auth_required():
                        self._terminate()
                        error = (
                            "Executor authentication required. "
                            "Run the CLI interactively once to complete login, "
                            "then retry the worker."
                        )
                        return

                    # Stale detection: no output for stale_timeout seconds
                    if (
                        self.stale_timeout > 0
                        and now - last_activity >= self.stale_timeout
                        and not self._cancelled.is_set()
                    ):
                        if self._stale_restarts < self.stale_max_restarts:
                            self._stale_restarts += 1
                            logger.info(
                                "Worker %s stale (no output for %.0fs), "
                                "restart %d/%d",
                                self.worker_id,
                                now - last_activity,
                                self._stale_restarts,
                                self.stale_max_restarts,
                            )
                            stdout_f, stderr_f = self._restart_process(
                                stdout_f,
                                stderr_f,
                                f"Stale (no output for {int(now - last_activity)}s). "
                                f"Restart {self._stale_restarts}/{self.stale_max_restarts}",
                            )
                            last_output_size = self._output_size()
                            last_activity = time.monotonic()
                            last_activity_wall = time.time()
                            self._emit_heartbeat(
                                now_wall=last_activity_wall,
                                last_activity_wall=last_activity_wall,
                                message=(
                                    f"Restarted after stale detection "
                                    f"({self._stale_restarts}/{self.stale_max_restarts})"
                                ),
                            )
                        else:
                            self._terminate()
                            error = (
                                f"Worker stale after {self._stale_restarts} restarts "
                                f"(no output for {int(now - last_activity)}s)"
                            )
                            return
                    if (
                        self.heartbeat_interval > 0
                        and now - last_heartbeat >= self.heartbeat_interval
                    ):
                        silence_seconds = int(max(0.0, now - last_activity))
                        self._emit_heartbeat(
                            now_wall=now_wall,
                            last_activity_wall=last_activity_wall,
                            message=(
                                f"Worker running; last output {silence_seconds}s ago"
                            ),
                        )
                        last_heartbeat = now
        except Exception as e:
            error = f"Monitor error: {e}"
        finally:
            stdout_f.close()
            stderr_f.close()
            if self.on_complete:
                self.on_complete(self.worker_id, exit_code, error)

    def _terminate(self):
        """Terminate the process group."""
        pid = self.pid
        if pid is None:
            return
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            if self._process is not None:
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    self._process.wait(timeout=5)
            else:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and self.pid_exists(pid):
                    time.sleep(0.1)
                if self.pid_exists(pid):
                    os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        finally:
            self._attached_pid = None

    def cancel(self):
        """Cancel the running worker (also interrupts retry backoff)."""
        self._cancelled.set()
        self._terminate()

    def is_running(self) -> bool:
        if self._process is not None:
            return self._process.poll() is None
        if self._attached_pid is not None:
            return self.pid_exists(self._attached_pid)
        return False

    def _wait_for_exit(self, timeout: float) -> int:
        """Wait for either a managed or attached process to exit."""
        if self._process is not None:
            return self._process.wait(timeout=timeout)
        if self._attached_pid is None:
            return 0

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.pid_exists(self._attached_pid):
                self._attached_pid = None
                return 0
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        raise subprocess.TimeoutExpired(self.command, timeout)
