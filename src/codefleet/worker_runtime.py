import os
import signal
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Callable


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
        "--output-format", "json",
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
        "--output-format", "json",
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
    """Manages a single worker subprocess."""

    def __init__(
        self,
        worker_id: str,
        command: list[str],
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        timeout_seconds: int,
        on_complete: Optional[Callable[[str, int, Optional[str]], None]] = None,
    ):
        self.worker_id = worker_id
        self.command = command
        self.cwd = cwd
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.timeout_seconds = timeout_seconds
        self.on_complete = on_complete
        self._process: Optional[subprocess.Popen] = None
        self._monitor_thread: Optional[threading.Thread] = None

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None

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

    def _monitor(self, stdout_f, stderr_f):
        """Monitor the process for completion or timeout."""
        exit_code = -1
        error = None
        try:
            start_time = time.monotonic()
            while True:
                try:
                    exit_code = self._process.wait(timeout=1.0)
                    return
                except subprocess.TimeoutExpired:
                    if time.monotonic() - start_time >= self.timeout_seconds:
                        self._terminate()
                        error = "Worker timed out"
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
        """Cancel the running worker."""
        self._terminate()

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None
