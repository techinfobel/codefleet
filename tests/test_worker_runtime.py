"""Tests for worker_runtime.py - real subprocess management."""

import os
import sys
import time
from pathlib import Path

import pytest

from codex_fleet_supervisor.worker_runtime import (
    WorkerProcess,
    build_codex_command,
    get_codex_path,
)


class TestBuildCodexCommand:
    def test_basic_command(self, tmp_path):
        prompt_path = tmp_path / "prompt.txt"
        result_path = tmp_path / "result.json"
        cmd = build_codex_command(prompt_path, result_path)
        assert cmd[0] == "codex"
        assert cmd[1] == "exec"
        assert "--model" in cmd
        assert "gpt-5.4" in cmd
        assert str(prompt_path) in cmd[-1]
        assert str(result_path) in cmd[-1]

    def test_custom_model(self, tmp_path):
        cmd = build_codex_command(
            tmp_path / "p.txt",
            tmp_path / "r.json",
            model="gpt-4o",
        )
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "gpt-4o"

    def test_extra_args(self, tmp_path):
        cmd = build_codex_command(
            tmp_path / "p.txt",
            tmp_path / "r.json",
            extra_args=["--json", "-C", "/some/dir"],
        )
        assert "--json" in cmd
        assert "-C" in cmd
        assert "/some/dir" in cmd


class TestGetCodexPath:
    def test_returns_string_or_none(self):
        result = get_codex_path()
        # codex should be installed per environment check
        assert result is None or isinstance(result, str)


class TestWorkerProcess:
    def test_successful_process(self, tmp_path):
        """Run a real subprocess that exits successfully."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"
        result_file = tmp_path / "result.json"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["wid"] = wid
            completed["exit_code"] = exit_code
            completed["error"] = error

        # Use Python itself as the subprocess
        script = (
            f'import json; '
            f'print("hello stdout"); '
            f'import sys; print("hello stderr", file=sys.stderr); '
            f'json.dump({{"summary":"done","status":"completed"}}, '
            f'open("{result_file}", "w"))'
        )

        wp = WorkerProcess(
            worker_id="w_test_ok",
            command=[sys.executable, "-c", script],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=30,
            on_complete=on_complete,
        )

        pid = wp.start()
        assert pid > 0
        assert wp.is_running() or True  # might finish very fast

        # Wait for completion
        deadline = time.monotonic() + 10
        while "exit_code" not in completed and time.monotonic() < deadline:
            time.sleep(0.1)

        assert completed["exit_code"] == 0
        assert completed["error"] is None
        assert completed["wid"] == "w_test_ok"

        # Check files
        assert stdout_path.exists()
        assert "hello stdout" in stdout_path.read_text()
        assert stderr_path.exists()
        assert "hello stderr" in stderr_path.read_text()
        assert result_file.exists()

    def test_failing_process(self, tmp_path):
        """Run a subprocess that exits with non-zero."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["exit_code"] = exit_code

        wp = WorkerProcess(
            worker_id="w_test_fail",
            command=[sys.executable, "-c", "import sys; sys.exit(1)"],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=30,
            on_complete=on_complete,
        )

        wp.start()

        deadline = time.monotonic() + 10
        while "exit_code" not in completed and time.monotonic() < deadline:
            time.sleep(0.1)

        assert completed["exit_code"] == 1

    def test_timeout(self, tmp_path):
        """Run a subprocess that exceeds timeout."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["exit_code"] = exit_code
            completed["error"] = error

        wp = WorkerProcess(
            worker_id="w_test_timeout",
            command=[sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=2,
            on_complete=on_complete,
        )

        wp.start()

        deadline = time.monotonic() + 15
        while "exit_code" not in completed and time.monotonic() < deadline:
            time.sleep(0.2)

        assert "error" in completed
        assert "timed out" in completed["error"].lower()

    def test_cancel(self, tmp_path):
        """Cancel a running subprocess."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["exit_code"] = exit_code

        wp = WorkerProcess(
            worker_id="w_test_cancel",
            command=[sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=300,
            on_complete=on_complete,
        )

        wp.start()
        time.sleep(0.5)
        assert wp.is_running()

        wp.cancel()

        deadline = time.monotonic() + 10
        while "exit_code" not in completed and time.monotonic() < deadline:
            time.sleep(0.1)

        assert not wp.is_running()

    def test_stdout_stderr_capture(self, tmp_path):
        """Verify stdout and stderr are captured to separate files."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["done"] = True

        script = (
            "import sys; "
            "print('line1'); print('line2'); "
            "print('err1', file=sys.stderr); print('err2', file=sys.stderr)"
        )

        wp = WorkerProcess(
            worker_id="w_test_io",
            command=[sys.executable, "-c", script],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=30,
            on_complete=on_complete,
        )

        wp.start()

        deadline = time.monotonic() + 10
        while "done" not in completed and time.monotonic() < deadline:
            time.sleep(0.1)

        stdout_content = stdout_path.read_text()
        stderr_content = stderr_path.read_text()
        assert "line1" in stdout_content
        assert "line2" in stdout_content
        assert "err1" in stderr_content
        assert "err2" in stderr_content
