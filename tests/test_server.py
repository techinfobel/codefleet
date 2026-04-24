"""Tests for server.py - MCP server creation and tool registration."""

import sys
import time
from unittest.mock import patch

import pytest

from codefleet.server import create_server
from codefleet.supervisor import FleetSupervisor


def _fake_build(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
    script = (
        "import json; "
        "open('server_output.txt', 'a').write('server worker output\\n'); "
        "json.dump("
        '{"summary":"done","status":"completed",'
        '"files_changed":["server_output.txt"],"tests":[]}, '
        f"open('{result_json_path}', 'w'))"
    )
    return [sys.executable, "-c", script]


def _fake_build_sleep(executor, prompt_path, result_json_path, model, reasoning_effort=None, extra_args=None, base_commit=None):
    return [sys.executable, "-c", "import time; time.sleep(300)"]


class TestCreateServer:
    def test_server_creation(self, supervisor):
        """Server should be created with all tools registered."""
        server = create_server(supervisor=supervisor)
        assert server is not None
        assert server.name == "codefleet"

    def test_server_has_tools(self, supervisor):
        """Server should expose all required MCP tools."""
        server = create_server(supervisor=supervisor)
        tool_names = {tool.name for tool in server._tool_manager.list_tools()}
        expected_tools = {
            "healthcheck",
            "executor_guide",
            "create_worker",
            "get_worker_status",
            "list_workers",
            "collect_worker_result",
            "cancel_worker",
            "cleanup_worker",
            "create_workflow",
            "get_workflow_status",
            "list_workflows",
            "cancel_workflow",
            "collect_workflow_result",
            "cleanup_workflow",
        }
        assert expected_tools.issubset(tool_names)

    def test_server_default_creation(self, tmp_path, monkeypatch):
        """Server should create with env vars when no supervisor is passed."""
        monkeypatch.setenv("FLEET_BASE_DIR", str(tmp_path / "fleet"))
        monkeypatch.setenv("FLEET_DEFAULT_MODEL", "gpt-5.5")
        monkeypatch.setenv("FLEET_DEFAULT_TIMEOUT", "300")
        monkeypatch.setenv("FLEET_MAX_CONCURRENT", "5")
        monkeypatch.setenv("FLEET_ALLOWED_REPOS", "")
        server = create_server()
        assert server is not None


class TestServerToolCalls:
    """Call the actual MCP tool functions through the server to cover server.py bodies."""

    def _get_tools(self, supervisor):
        server = create_server(supervisor=supervisor)
        return {t.name: t for t in server._tool_manager.list_tools()}

    @pytest.mark.asyncio
    async def test_healthcheck_tool(self, supervisor):
        server = create_server(supervisor=supervisor)
        result = await server.call_tool("healthcheck", {})
        assert len(result) > 0
        text = result[0].text
        assert "supported_models" in text
        assert "gpt-5.5" in text
        assert "gemini-3.1-pro-preview" in text
        assert "claude-sonnet-4-6" in text

    @pytest.mark.asyncio
    async def test_executor_guide_tool(self, supervisor):
        server = create_server(supervisor=supervisor)
        result = await server.call_tool("executor_guide", {})
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_list_workers_tool(self, supervisor):
        server = create_server(supervisor=supervisor)
        result = await server.call_tool("list_workers", {"limit": 5})
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_list_workflows_tool(self, supervisor):
        server = create_server(supervisor=supervisor)
        result = await server.call_tool("list_workflows", {"limit": 5})
        assert len(result) > 0

    @pytest.mark.asyncio
    @patch("codefleet.supervisor.build_worker_command", side_effect=_fake_build)
    async def test_create_worker_tool(self, mock_build, supervisor, git_repo):
        server = create_server(supervisor=supervisor)
        result = await server.call_tool("create_worker", {
            "repo_path": str(git_repo),
            "task_name": "server-test",
            "prompt": "Test prompt",
        })
        assert len(result) > 0

        # Wait for completion
        deadline = time.monotonic() + 15
        workers = supervisor.list_workers()
        wid = workers[0].worker_id
        while time.monotonic() < deadline:
            s = supervisor.get_worker_status(wid)
            if s.status.is_terminal:
                break
            time.sleep(0.2)

        # Test get_worker_status tool
        result = await server.call_tool("get_worker_status", {"worker_id": wid})
        assert len(result) > 0

        # Test collect_worker_result tool
        result = await server.call_tool("collect_worker_result", {
            "worker_id": wid, "include_logs": True, "log_tail_lines": 10,
        })
        assert len(result) > 0

        # Test cleanup_worker tool
        result = await server.call_tool("cleanup_worker", {"worker_id": wid})
        assert len(result) > 0

    @pytest.mark.asyncio
    @patch("codefleet.supervisor.build_worker_command", side_effect=_fake_build_sleep)
    async def test_cancel_worker_tool(self, mock_build, supervisor, git_repo):
        server = create_server(supervisor=supervisor)
        result = await server.call_tool("create_worker", {
            "repo_path": str(git_repo),
            "task_name": "cancel-server-test",
            "prompt": "Sleep",
        })
        time.sleep(0.5)
        wid = supervisor.list_workers()[0].worker_id
        result = await server.call_tool("cancel_worker", {"worker_id": wid})
        assert len(result) > 0

    @pytest.mark.asyncio
    @patch("codefleet.supervisor.build_worker_command", side_effect=_fake_build)
    async def test_workflow_tools(self, mock_build, supervisor, git_repo):
        server = create_server(supervisor=supervisor)

        # create_workflow
        result = await server.call_tool("create_workflow", {
            "name": "server-wf-test",
            "repo_path": str(git_repo),
            "task_prompt": "Test",
            "stages": [
                {"name": "s0", "executor": "codex", "prompt_template": "{task_prompt}",
                 "worktree_strategy": "new", "depends_on": []},
            ],
        })
        assert len(result) > 0

        # Wait for workflow to finish
        workflows = supervisor.list_workflows()
        wf_id = workflows[0].workflow_id
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            wf = supervisor.get_workflow_status(wf_id)
            if wf.status.value in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.2)

        # get_workflow_status
        result = await server.call_tool("get_workflow_status", {"workflow_id": wf_id})
        assert len(result) > 0

        # collect_workflow_result
        result = await server.call_tool("collect_workflow_result", {
            "workflow_id": wf_id, "include_all_stages": True,
        })
        assert len(result) > 0

        # cleanup_workflow
        result = await server.call_tool("cleanup_workflow", {"workflow_id": wf_id})
        assert len(result) > 0

    @pytest.mark.asyncio
    @patch("codefleet.supervisor.build_worker_command", side_effect=_fake_build_sleep)
    async def test_cancel_workflow_tool(self, mock_build, supervisor, git_repo):
        server = create_server(supervisor=supervisor)
        await server.call_tool("create_workflow", {
            "name": "cancel-wf-test",
            "repo_path": str(git_repo),
            "task_prompt": "Slow",
            "stages": [
                {"name": "s0", "executor": "codex", "prompt_template": "x",
                 "worktree_strategy": "new", "depends_on": []},
            ],
        })
        time.sleep(0.5)
        wf_id = supervisor.list_workflows()[0].workflow_id
        result = await server.call_tool("cancel_workflow", {"workflow_id": wf_id})
        assert len(result) > 0
