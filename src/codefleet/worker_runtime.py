import logging
import os
import signal
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)


def get_codex_path() -> str | None:
    return shutil.which("codex")


def get_gemini_path() -> str | None:
    return shutil.which("gemini")


def get_claude_path() -> str | None:
    return shutil.which("claude")


def _result_instruction(prompt_path: Path, result_json_path: Path) -> str:
    """Build the shared instruction string for any executor."""
    return (
        f"Read the task prompt from {prompt_path}. "
        f"Work in the current directory (a git worktree). "
        f"Do NOT commit or modify anything in the .codefleet/ directory. "
        f"Run relevant tests where appropriate. "
        f"Write a single JSON object to {result_json_path} with this schema: "
        f'{{"summary": "string", "files_changed": ["path"], '
        f'"tests": [{{"command": "string", "status": "passed|failed|not_run", '
        f'"details": "string"}}], '
        f'"commits": ["hash"], "next_steps": ["string"], '
        f'"status": "completed|blocked"}}'
    )


def build_codex_command(
    prompt_path: Path,
    result_json_path: Path,
    model: str = "gpt-5.4",
    reasoning_effort: str = "xhigh",
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """Build the codex exec command."""
    instruction = _result_instruction(prompt_path, result_json_path)

    cmd = ["codex", "exec", "--full-auto"]
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
) -> list[str]:
    """Build the gemini CLI command."""
    instruction = _result_instruction(prompt_path, result_json_path)

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
    model: str = "claude-opus-4-6",
    effort: str = "high",
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """Build the claude CLI command."""
    instruction = _result_instruction(prompt_path, result_json_path)

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
) -> list[str]:
    """Dispatch to the correct command builder based on executor type."""
    if executor == "gemini":
        return build_gemini_command(
            prompt_path=prompt_path,
            result_json_path=result_json_path,
            model=model,
            extra_args=extra_args,
        )
    if executor == "claude":
        return build_claude_command(
            prompt_path=prompt_path,
            result_json_path=result_json_path,
            model=model,
            effort=reasoning_effort or "max",
            extra_args=extra_args,
        )
    # Default: codex
    return build_codex_command(
        prompt_path=prompt_path,
        result_json_path=result_json_path,
        model=model,
        reasoning_effort=reasoning_effort or "xhigh",
        extra_args=extra_args,
    )


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

    def __init__(
        self,
        worker_id: str,
        command: list[str],
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        timeout_seconds: int,
        on_complete: Optional[Callable[[str, int, Optional[str]], None]] = None,
        max_retries: int = 0,
        retry_base_delay: float = 4.0,
        retry_max_delay: float = 60.0,
        stale_timeout: float = 120.0,
        stale_max_restarts: int = 2,
    ):
        self.worker_id = worker_id
        self.command = command
        self.cwd = cwd
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.timeout_seconds = timeout_seconds
        self.on_complete = on_complete
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.stale_timeout = stale_timeout
        self.stale_max_restarts = stale_max_restarts
        self._process: Optional[subprocess.Popen] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._retry_count = 0
        self._stale_restarts = 0
        self._cancelled = threading.Event()

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None

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
                stdout=stdout_f,
                stderr=stderr_f,
                start_new_session=True,
            )
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
        self._process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,
        )
        return stdout_f, stderr_f

    def _monitor(self, stdout_f, stderr_f):
        """Monitor the process for completion, stale detection, or rate-limit retry."""
        exit_code = -1
        error = None
        last_output_size = self._output_size()
        last_activity = time.monotonic()
        try:
            while True:
                try:
                    exit_code = self._process.wait(timeout=1.0)

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
                                stdout=stdout_f,
                                stderr=stderr_f,
                                start_new_session=True,
                            )
                            last_output_size = self._output_size()
                            last_activity = time.monotonic()
                            continue

                    return
                except subprocess.TimeoutExpired:
                    # Check for activity (output growth)
                    current_size = self._output_size()
                    now = time.monotonic()
                    if current_size != last_output_size:
                        last_output_size = current_size
                        last_activity = now

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
                        else:
                            self._terminate()
                            error = (
                                f"Worker stale after {self._stale_restarts} restarts "
                                f"(no output for {int(now - last_activity)}s)"
                            )
                            return
        except Exception as e:
            error = f"Monitor error: {e}"
        finally:
            stdout_f.close()
            stderr_f.close()
            if self.on_complete:
                self.on_complete(self.worker_id, exit_code, error)

    def _terminate(self):
        """Terminate the process group."""
        if self._process is None:
            return
        try:
            pgid = os.getpgid(self._process.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self._process.wait(timeout=5)
        except (ProcessLookupError, OSError):
            pass

    def cancel(self):
        """Cancel the running worker (also interrupts retry backoff)."""
        self._cancelled.set()
        self._terminate()

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None
