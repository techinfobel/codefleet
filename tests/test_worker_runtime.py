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
        assert "stream-json" in cmd
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
        assert "stream-json" in cmd
        assert "--verbose" in cmd
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

    def test_claude_defaults_effort_max(self, tmp_path):
        cmd = build_worker_command(
            executor="claude",
            prompt_path=tmp_path / "p.txt",
            result_json_path=tmp_path / "r.json",
            model="claude-opus-4-6",
        )
        assert "--effort" in cmd
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "max"


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
        assert pid > 0

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

    def test_stale_kills_silent_process(self, tmp_path):
        """A silent process is killed by stale detection."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["exit_code"] = exit_code
            completed["error"] = error

        wp = WorkerProcess(
            worker_id="w_test_stale",
            command=[sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=600,
            on_complete=on_complete,
            stale_timeout=2.0,
            stale_max_restarts=0,  # kill immediately, no restarts
        )

        wp.start()

        deadline = time.monotonic() + 15
        while "exit_code" not in completed and time.monotonic() < deadline:
            time.sleep(0.2)

        assert "error" in completed
        assert "stale" in completed["error"].lower()

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


class TestRateLimitRetry:
    """Test automatic retry on rate-limit (429) errors."""

    def test_retry_on_rate_limit(self, tmp_path):
        """Worker retries when stderr contains rate-limit indicator."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"
        marker = tmp_path / "attempt"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["exit_code"] = exit_code
            completed["error"] = error

        # Script file: first run writes "429" to stderr and exits 1,
        # second run succeeds (uses a marker file to track attempts).
        script_path = tmp_path / "retry_script.py"
        script_path.write_text(
            "import sys, os\n"
            f"marker = {str(marker)!r}\n"
            "if not os.path.exists(marker):\n"
            "    open(marker, 'w').close()\n"
            "    print('429 Too Many Requests', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "print('ok')\n"
        )

        wp = WorkerProcess(
            worker_id="w_retry_429",
            command=[sys.executable, str(script_path)],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=30,
            on_complete=on_complete,
            max_retries=2,
            retry_base_delay=0.1,  # fast for tests
            retry_max_delay=1.0,
        )

        wp.start()

        deadline = time.monotonic() + 15
        while "exit_code" not in completed and time.monotonic() < deadline:
            time.sleep(0.1)

        assert completed["exit_code"] == 0
        assert wp.retry_count == 1
        stderr_content = stderr_path.read_text()
        assert "Rate limited" in stderr_content
        assert "Retry 1/2" in stderr_content

    def test_no_retry_without_rate_limit(self, tmp_path):
        """Worker does NOT retry for non-rate-limit failures."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["exit_code"] = exit_code

        script = "import sys; print('some other error', file=sys.stderr); sys.exit(1)"

        wp = WorkerProcess(
            worker_id="w_no_retry",
            command=[sys.executable, "-c", script],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=30,
            on_complete=on_complete,
            max_retries=2,
            retry_base_delay=0.1,
        )

        wp.start()

        deadline = time.monotonic() + 10
        while "exit_code" not in completed and time.monotonic() < deadline:
            time.sleep(0.1)

        assert completed["exit_code"] == 1
        assert wp.retry_count == 0

    def test_retry_exhausted(self, tmp_path):
        """Worker fails after exhausting all retries."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["exit_code"] = exit_code

        # Always fails with 429
        script = "import sys; print('429 rate limit exceeded', file=sys.stderr); sys.exit(1)"

        wp = WorkerProcess(
            worker_id="w_exhaust",
            command=[sys.executable, "-c", script],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=30,
            on_complete=on_complete,
            max_retries=2,
            retry_base_delay=0.1,
            retry_max_delay=0.5,
        )

        wp.start()

        deadline = time.monotonic() + 15
        while "exit_code" not in completed and time.monotonic() < deadline:
            time.sleep(0.1)

        assert completed["exit_code"] == 1
        assert wp.retry_count == 2
        stderr_content = stderr_path.read_text()
        assert "Retry 1/2" in stderr_content
        assert "Retry 2/2" in stderr_content

    def test_cancel_during_retry_backoff(self, tmp_path):
        """Cancel interrupts retry backoff sleep."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["exit_code"] = exit_code
            completed["error"] = error

        # Always fails with 429, long backoff to give time to cancel
        script = "import sys; print('429', file=sys.stderr); sys.exit(1)"

        wp = WorkerProcess(
            worker_id="w_cancel_retry",
            command=[sys.executable, "-c", script],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=30,
            on_complete=on_complete,
            max_retries=3,
            retry_base_delay=30.0,  # long delay so we can cancel during it
        )

        wp.start()

        # Wait for the first attempt to fail and backoff to start
        deadline = time.monotonic() + 10
        while wp.retry_count < 1 and time.monotonic() < deadline:
            time.sleep(0.1)

        wp.cancel()

        deadline = time.monotonic() + 5
        while "exit_code" not in completed and time.monotonic() < deadline:
            time.sleep(0.1)

        assert "error" in completed
        assert "cancelled" in completed["error"].lower()


class TestStaleDetection:
    """Test stale process detection and restart."""

    def test_stale_restart_recovers(self, tmp_path):
        """A stale process is killed and restarted, second attempt succeeds."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"
        marker = tmp_path / "attempt"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["exit_code"] = exit_code
            completed["error"] = error

        # First run: write one line then go silent (stale).
        # Second run (after restart): marker exists, succeed immediately.
        script_path = tmp_path / "stale_script.py"
        script_path.write_text(
            "import sys, os, time\n"
            f"marker = {str(marker)!r}\n"
            "if not os.path.exists(marker):\n"
            "    open(marker, 'w').close()\n"
            "    print('starting...', file=sys.stderr)\n"
            "    time.sleep(300)  # go silent\n"
            "else:\n"
            "    print('recovered', file=sys.stderr)\n"
            "    print('ok')\n"
        )

        wp = WorkerProcess(
            worker_id="w_stale_ok",
            command=[sys.executable, str(script_path)],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=600,
            on_complete=on_complete,
            stale_timeout=2.0,  # fast for tests
            stale_max_restarts=2,
        )

        wp.start()

        deadline = time.monotonic() + 20
        while "exit_code" not in completed and time.monotonic() < deadline:
            time.sleep(0.2)

        assert completed["exit_code"] == 0
        assert wp._stale_restarts == 1
        stderr_content = stderr_path.read_text()
        assert "Stale" in stderr_content
        assert "Restart 1/2" in stderr_content

    def test_stale_exhausted(self, tmp_path):
        """Worker fails after exhausting stale restarts."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["exit_code"] = exit_code
            completed["error"] = error

        # Always goes silent after initial output
        script = (
            "import sys, time; "
            "print('start', file=sys.stderr); "
            "time.sleep(300)"
        )

        wp = WorkerProcess(
            worker_id="w_stale_exhaust",
            command=[sys.executable, "-c", script],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=600,
            on_complete=on_complete,
            stale_timeout=2.0,
            stale_max_restarts=1,
        )

        wp.start()

        deadline = time.monotonic() + 20
        while "exit_code" not in completed and time.monotonic() < deadline:
            time.sleep(0.2)

        assert "error" in completed
        assert "stale" in completed["error"].lower()
        assert wp._stale_restarts == 1

    def test_active_process_not_stale(self, tmp_path):
        """A process that keeps writing output is never considered stale."""
        stdout_path = tmp_path / "stdout.log"
        stderr_path = tmp_path / "stderr.log"

        completed = {}

        def on_complete(wid, exit_code, error):
            completed["exit_code"] = exit_code

        # Writes output every 0.5s for 3s then exits — should never trigger
        # stale detection even with a 2s stale_timeout.
        script = (
            "import sys, time\n"
            "for i in range(6):\n"
            "    print(f'tick {i}', file=sys.stderr)\n"
            "    time.sleep(0.5)\n"
        )

        wp = WorkerProcess(
            worker_id="w_active",
            command=[sys.executable, "-c", script],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=600,
            on_complete=on_complete,
            stale_timeout=2.0,
            stale_max_restarts=2,
        )

        wp.start()

        deadline = time.monotonic() + 15
        while "exit_code" not in completed and time.monotonic() < deadline:
            time.sleep(0.2)

        assert completed["exit_code"] == 0
        assert wp._stale_restarts == 0
