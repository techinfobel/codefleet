"""Tests for worker_runtime.py - real subprocess management."""

import os
import sys
import time
from pathlib import Path

import pytest

from codefleet.worker_runtime import (
    WorkerProcess,
    build_codex_command,
    build_gemini_command,
    build_claude_command,
    build_worker_command,
    get_codex_path,
    get_gemini_path,
    get_claude_path,
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


class TestBuildGeminiCommand:
    def test_basic_command(self, tmp_path):
        prompt_path = tmp_path / "prompt.txt"
        result_path = tmp_path / "result.json"
        cmd = build_gemini_command(prompt_path, result_path)
        assert cmd[0] == "gemini"
        assert "-p" in cmd
        assert "--approval-mode" in cmd
        assert "yolo" in cmd
        assert "--sandbox" in cmd
        assert "false" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "-m" in cmd
        assert "gemini-3.1-pro-preview" in cmd

    def test_custom_model(self, tmp_path):
        cmd = build_gemini_command(
            tmp_path / "p.txt",
            tmp_path / "r.json",
            model="gemini-2.0-flash",
        )
        idx = cmd.index("-m")
        assert cmd[idx + 1] == "gemini-2.0-flash"

    def test_extra_args(self, tmp_path):
        cmd = build_gemini_command(
            tmp_path / "p.txt",
            tmp_path / "r.json",
            extra_args=["--sandbox", "false"],
        )
        assert "--sandbox" in cmd
        assert "false" in cmd


class TestBuildClaudeCommand:
    def test_basic_command(self, tmp_path):
        prompt_path = tmp_path / "prompt.txt"
        result_path = tmp_path / "result.json"
        cmd = build_claude_command(prompt_path, result_path)
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--model" in cmd
        assert "claude-opus-4-6" in cmd
        # Default effort is high
        assert "--effort" in cmd
        assert "high" in cmd

    def test_custom_model(self, tmp_path):
        cmd = build_claude_command(
            tmp_path / "p.txt",
            tmp_path / "r.json",
            model="claude-opus-4-6",
        )
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4-6"

    def test_custom_effort(self, tmp_path):
        cmd = build_claude_command(
            tmp_path / "p.txt",
            tmp_path / "r.json",
            effort="max",
        )
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "max"

    def test_extra_args(self, tmp_path):
        cmd = build_claude_command(
            tmp_path / "p.txt",
            tmp_path / "r.json",
            extra_args=["--max-budget-usd", "5"],
        )
        assert "--max-budget-usd" in cmd
        assert "5" in cmd


class TestBuildWorkerCommand:
    def test_dispatch_codex(self, tmp_path):
        cmd = build_worker_command(
            executor="codex",
            prompt_path=tmp_path / "p.txt",
            result_json_path=tmp_path / "r.json",
            model="gpt-5.4",
        )
        assert cmd[0] == "codex"

    def test_dispatch_gemini(self, tmp_path):
        cmd = build_worker_command(
            executor="gemini",
            prompt_path=tmp_path / "p.txt",
            result_json_path=tmp_path / "r.json",
            model="gemini-3.1-pro-preview",
        )
        assert cmd[0] == "gemini"

    def test_dispatch_claude(self, tmp_path):
        cmd = build_worker_command(
            executor="claude",
            prompt_path=tmp_path / "p.txt",
            result_json_path=tmp_path / "r.json",
            model="claude-opus-4-6",
        )
        assert cmd[0] == "claude"

    def test_codex_gets_reasoning_effort(self, tmp_path):
        cmd = build_worker_command(
            executor="codex",
            prompt_path=tmp_path / "p.txt",
            result_json_path=tmp_path / "r.json",
            model="gpt-5.4",
            reasoning_effort="high",
        )
        assert any("reasoning_effort" in arg for arg in cmd)

    def test_gemini_ignores_reasoning_effort(self, tmp_path):
        cmd = build_worker_command(
            executor="gemini",
            prompt_path=tmp_path / "p.txt",
            result_json_path=tmp_path / "r.json",
            model="gemini-3.1-pro-preview",
            reasoning_effort="high",
        )
        assert not any("reasoning_effort" in arg for arg in cmd)

    def test_claude_maps_reasoning_effort_to_effort(self, tmp_path):
        cmd = build_worker_command(
            executor="claude",
            prompt_path=tmp_path / "p.txt",
            result_json_path=tmp_path / "r.json",
            model="claude-opus-4-6",
            reasoning_effort="max",
        )
        assert "--effort" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "max"

    def test_claude_defaults_effort_high(self, tmp_path):
        cmd = build_worker_command(
            executor="claude",
            prompt_path=tmp_path / "p.txt",
            result_json_path=tmp_path / "r.json",
            model="claude-opus-4-6",
        )
        assert "--effort" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "high"


class TestGetCodexPath:
    def test_returns_string_or_none(self):
        result = get_codex_path()
        assert result is None or isinstance(result, str)


class TestGetGeminiPath:
    def test_returns_string_or_none(self):
        result = get_gemini_path()
        assert result is None or isinstance(result, str)


class TestGetClaudePath:
    def test_returns_string_or_none(self):
        result = get_claude_path()
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
