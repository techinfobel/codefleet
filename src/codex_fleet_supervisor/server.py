import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .supervisor import FleetSupervisor


def create_server(supervisor: Optional[FleetSupervisor] = None) -> FastMCP:
    """Create the MCP server with all tools registered."""
    mcp = FastMCP("codex-fleet-supervisor")

    if supervisor is None:
        allowed_repos_str = os.environ.get("FLEET_ALLOWED_REPOS", "")
        allowed_repos = (
            [r.strip() for r in allowed_repos_str.split(",") if r.strip()] or None
        )

        supervisor = FleetSupervisor(
            base_dir=os.environ.get("FLEET_BASE_DIR"),
            default_model=os.environ.get("FLEET_DEFAULT_MODEL", "gpt-5.3-codex"),
            default_reasoning_effort=os.environ.get("FLEET_REASONING_EFFORT", "xhigh"),
            default_timeout=int(os.environ.get("FLEET_DEFAULT_TIMEOUT", "600")),
            max_concurrent=int(os.environ.get("FLEET_MAX_CONCURRENT", "10")),
            allowed_repos=allowed_repos,
        )

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
        reasoning_effort: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        profile: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        extra_codex_args: Optional[list[str]] = None,
        parent_worker_id: Optional[str] = None,
    ) -> dict:
        """Launch a new Codex worker in an isolated git worktree."""
        result = supervisor.create_worker(
            repo_path=repo_path,
            task_name=task_name,
            prompt=prompt,
            base_ref=base_ref,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            profile=profile,
            tags=tags,
            metadata=metadata,
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

    return mcp


def main():
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
