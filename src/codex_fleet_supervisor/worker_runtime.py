import os
import signal
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Callable


def get_codex_path() -> str | None:
    """Return the path to codex, or None."""
    return shutil.which("codex")


def build_codex_command(
    prompt_path: Path,
    result_json_path: Path,
    model: str = "gpt-5.4",
    reasoning_effort: str = "xhigh",
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """Build the codex exec command."""
    instruction = (
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

    cmd = ["codex", "exec", "--full-auto"]
    if model:
        cmd.extend(["--model", model])
    if reasoning_effort:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(instruction)
    return cmd


class WorkerProcess:
    """Manages a single codex worker subprocess."""

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

        self._process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,
            env={**os.environ},
        )

        # Start monitor thread
        self._monitor_thread = threading.Thread(
            target=self._monitor,
            args=(stdout_f, stderr_f),
            daemon=True,
        )
        self._monitor_thread.start()

        return self._process.pid

    def _monitor(self, stdout_f, stderr_f):
        """Monitor the process for completion or timeout."""
        try:
            start_time = time.monotonic()
            while True:
                try:
                    exit_code = self._process.wait(timeout=1.0)
                    # Process completed
                    stdout_f.close()
                    stderr_f.close()
                    if self.on_complete:
                        self.on_complete(self.worker_id, exit_code, None)
                    return
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= self.timeout_seconds:
                        self._terminate()
                        stdout_f.close()
                        stderr_f.close()
                        if self.on_complete:
                            self.on_complete(
                                self.worker_id, -1, "Worker timed out"
                            )
                        return
        except Exception as e:
            try:
                stdout_f.close()
                stderr_f.close()
            except Exception:
                pass
            if self.on_complete:
                self.on_complete(self.worker_id, -1, f"Monitor error: {e}")

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
        if self._process is None:
            return False
        return self._process.poll() is None
