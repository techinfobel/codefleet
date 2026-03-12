"""Tests for server.py - MCP server creation and tool registration."""

import pytest

from codex_fleet_supervisor.server import create_server
from codex_fleet_supervisor.supervisor import FleetSupervisor


class TestCreateServer:
    def test_server_creation(self, supervisor):
        """Server should be created with all tools registered."""
        server = create_server(supervisor=supervisor)
        assert server is not None
        assert server.name == "codex-fleet-supervisor"

    def test_server_has_tools(self, supervisor):
        """Server should expose all required MCP tools."""
        server = create_server(supervisor=supervisor)
        tool_names = {tool.name for tool in server._tool_manager.list_tools()}
        expected_tools = {
            "healthcheck",
            "create_worker",
            "get_worker_status",
            "list_workers",
            "collect_worker_result",
            "cancel_worker",
            "cleanup_worker",
        }
        assert expected_tools.issubset(tool_names)

    def test_server_default_creation(self, tmp_path, monkeypatch):
        """Server should create with env vars when no supervisor is passed."""
        monkeypatch.setenv("FLEET_BASE_DIR", str(tmp_path / "fleet"))
        monkeypatch.setenv("FLEET_DEFAULT_MODEL", "gpt-5.4")
        monkeypatch.setenv("FLEET_DEFAULT_TIMEOUT", "300")
        monkeypatch.setenv("FLEET_MAX_CONCURRENT", "5")
        monkeypatch.setenv("FLEET_ALLOWED_REPOS", "")
        server = create_server()
        assert server is not None
