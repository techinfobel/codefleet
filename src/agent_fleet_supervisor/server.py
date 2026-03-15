import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .supervisor import FleetSupervisor


def create_server(supervisor: Optional[FleetSupervisor] = None) -> FastMCP:
    """Create the MCP server with all tools registered."""
    mcp = FastMCP("agent-fleet-supervisor")

    if supervisor is None:
        allowed_repos_str = os.environ.get("FLEET_ALLOWED_REPOS", "")
        allowed_repos = (
            [r.strip() for r in allowed_repos_str.split(",") if r.strip()] or None
        )

        supervisor = FleetSupervisor(
            base_dir=os.environ.get("FLEET_BASE_DIR"),
            default_model=os.environ.get("FLEET_DEFAULT_MODEL", "gpt-5.4"),
            default_gemini_model=os.environ.get(
                "FLEET_GEMINI_DEFAULT_MODEL", "gemini-3.1-pro-preview"
            ),
            default_claude_model=os.environ.get(
                "FLEET_CLAUDE_DEFAULT_MODEL", "claude-sonnet-4-6"
            ),
            default_reasoning_effort=os.environ.get("FLEET_REASONING_EFFORT", "xhigh"),
            default_timeout=int(os.environ.get("FLEET_DEFAULT_TIMEOUT", "600")),
            max_concurrent=int(os.environ.get("FLEET_MAX_CONCURRENT", "10")),
            allowed_repos=allowed_repos,
            default_executor=os.environ.get("FLEET_DEFAULT_EXECUTOR", "codex"),
            max_spawn_depth=int(os.environ.get("FLEET_MAX_SPAWN_DEPTH", "2")),
        )

    # --- Worker tools ---

    @mcp.tool()
    def healthcheck() -> dict:
        """Return a basic capability report. Lets Claude verify the MCP server is reachable and the local environment is sane."""
        return supervisor.healthcheck()

    @mcp.tool()
    def create_worker(
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
    ) -> dict:
        """Launch a new worker in an isolated git worktree. Supports 'codex', 'gemini', and 'claude' executors."""
        result = supervisor.create_worker(
            repo_path=repo_path,
            task_name=task_name,
            prompt=prompt,
            base_ref=base_ref,
            model=model,
            executor=executor,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            profile=profile,
            tags=tags,
            metadata=metadata,
            extra_args=extra_args,
            extra_codex_args=extra_codex_args,
            parent_worker_id=parent_worker_id,
        )
        return result.model_dump()

    @mcp.tool()
    def get_worker_status(worker_id: str) -> dict:
        """Return current status and metadata for a worker."""
        result = supervisor.get_worker_status(worker_id)
        return result.model_dump()

    @mcp.tool()
    def list_workers(
        statuses: Optional[list[str]] = None,
        limit: int = 25,
    ) -> dict:
        """List recent workers, optionally filtered by status."""
        results = supervisor.list_workers(statuses=statuses, limit=limit)
        return {"workers": [r.model_dump() for r in results], "count": len(results)}

    @mcp.tool()
    def collect_worker_result(
        worker_id: str,
        include_logs: bool = False,
        log_tail_lines: int = 80,
    ) -> dict:
        """Return worker metadata plus parsed result.json and optional log tails."""
        return supervisor.collect_worker_result(
            worker_id=worker_id,
            include_logs=include_logs,
            log_tail_lines=log_tail_lines,
        )

    @mcp.tool()
    def cancel_worker(worker_id: str) -> dict:
        """Cancel a running worker."""
        result = supervisor.cancel_worker(worker_id)
        return result.model_dump()

    @mcp.tool()
    def cleanup_worker(
        worker_id: str,
        remove_branch: bool = True,
        remove_worktree_dir: bool = True,
    ) -> dict:
        """Remove worktree and optional branch for a terminal worker."""
        return supervisor.cleanup_worker(
            worker_id=worker_id,
            remove_branch=remove_branch,
            remove_worktree_dir=remove_worktree_dir,
        )

    # --- Workflow tools ---

    @mcp.tool()
    def create_workflow(
        name: str,
        repo_path: str,
        task_prompt: str,
        stages: list[dict],
        base_ref: str = "HEAD",
        timeout_seconds: Optional[int] = None,
    ) -> dict:
        """Define and start a multi-stage workflow. Stages form a DAG where each stage uses any executor (codex/gemini) and results flow between stages via prompt templates."""
        result = supervisor.create_workflow(
            name=name,
            repo_path=repo_path,
            task_prompt=task_prompt,
            stages=stages,
            base_ref=base_ref,
            timeout_seconds=timeout_seconds,
        )
        return result.model_dump()

    @mcp.tool()
    def get_workflow_status(workflow_id: str) -> dict:
        """Return workflow state with per-stage worker statuses."""
        result = supervisor.get_workflow_status(workflow_id)
        return result.model_dump()

    @mcp.tool()
    def list_workflows(
        statuses: Optional[list[str]] = None,
        limit: int = 25,
    ) -> dict:
        """List workflows with optional status filter."""
        results = supervisor.list_workflows(statuses=statuses, limit=limit)
        return {"workflows": [r.model_dump() for r in results], "count": len(results)}

    @mcp.tool()
    def cancel_workflow(workflow_id: str) -> dict:
        """Cancel all running stages and mark workflow cancelled."""
        result = supervisor.cancel_workflow(workflow_id)
        return result.model_dump()

    @mcp.tool()
    def collect_workflow_result(
        workflow_id: str,
        include_all_stages: bool = False,
        include_logs: bool = False,
    ) -> dict:
        """Get the final stage's result (or all stages' results)."""
        return supervisor.collect_workflow_result(
            workflow_id=workflow_id,
            include_all_stages=include_all_stages,
            include_logs=include_logs,
        )

    @mcp.tool()
    def cleanup_workflow(workflow_id: str) -> dict:
        """Clean up all worktrees, branches, and worker dirs for a terminal workflow."""
        return supervisor.cleanup_workflow(workflow_id)

    return mcp


def main():
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
