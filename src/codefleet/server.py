import logging
import os
import time as _time
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .supervisor import FleetSupervisor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response formatting helpers
# ---------------------------------------------------------------------------

_STATUS_LABELS = {
    "pending": "PENDING",
    "running": "RUNNING",
    "succeeded": "OK",
    "failed": "FAILED",
    "cancelled": "CANCELLED",
    "timed_out": "TIMED OUT",
    "cleanup_failed": "CLEANUP FAILED",
}


def _fmt_duration(seconds: float | None) -> str:
    """Format seconds as human-readable duration."""
    if seconds is None or seconds < 0:
        return "-"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _worker_elapsed(d: dict) -> float | None:
    started = d.get("started_at")
    ended = d.get("ended_at")
    if started and ended:
        return ended - started
    if started:
        return _time.time() - started
    return None


def _enrich_worker(d: dict) -> dict:
    """Add human-readable display fields to a worker status dict."""
    status = d.get("status", "")
    elapsed = _worker_elapsed(d)
    d["elapsed"] = _fmt_duration(elapsed)
    d["status_label"] = f"[{_STATUS_LABELS.get(status, status.upper())}]"
    task = d.get("task_name", "")
    executor = d.get("executor", "")
    model = d.get("model", "")
    d["summary_line"] = (
        f"{d['status_label']} {task} ({executor}/{model}) — {d['elapsed']}"
    )
    return d


def _enrich_workflow(d: dict, supervisor: FleetSupervisor) -> dict:
    """Add progress display fields to a workflow status dict."""
    stages = d.get("stage_summary", [])
    created = d.get("created_at")
    completed = d.get("completed_at")

    if created and completed:
        total = completed - created
    elif created:
        total = _time.time() - created
    else:
        total = None
    d["elapsed"] = _fmt_duration(total)

    # Look up worker records for per-stage timing
    worker_ids = [s["worker_id"] for s in stages if s.get("worker_id")]
    workers = {}
    if worker_ids:
        for rec in supervisor.store.get_workers(worker_ids):
            workers[rec.worker_id] = rec

    for s in stages:
        st = s.get("status", "pending")
        s["status_label"] = f"[{_STATUS_LABELS.get(st, st.upper())}]"
        wid = s.get("worker_id")
        if wid and wid in workers:
            rec = workers[wid]
            if rec.started_at and rec.ended_at:
                s["elapsed"] = _fmt_duration(rec.ended_at - rec.started_at)
            elif rec.started_at:
                s["elapsed"] = _fmt_duration(_time.time() - rec.started_at)
            else:
                s["elapsed"] = "-"
            s["model"] = rec.model
        else:
            s["elapsed"] = "-"

    # Progress summary
    counts: dict[str, int] = {}
    for s in stages:
        st = s.get("status", "pending")
        counts[st] = counts.get(st, 0) + 1

    n = len(stages)
    ok = counts.get("succeeded", 0)
    running = counts.get("running", 0)
    pending = counts.get("pending", 0)
    failed_n = sum(
        counts.get(k, 0) for k in ("failed", "timed_out", "cancelled", "cleanup_failed")
    )

    parts = []
    if ok:
        parts.append(f"{ok} done")
    if running:
        parts.append(f"{running} running")
    if pending:
        parts.append(f"{pending} pending")
    if failed_n:
        parts.append(f"{failed_n} failed")

    detail = f" ({', '.join(parts)})" if parts else ""
    d["progress"] = f"{ok}/{n} stages complete{detail}"

    status = d.get("status", "")
    name = d.get("name", "")
    label = _STATUS_LABELS.get(status, status.upper())
    d["summary_line"] = f"[{label}] {name} — {d['progress']} ({d['elapsed']})"

    # Enrich nested stage results / final result if present
    if "stage_results" in d:
        for sr in d["stage_results"]:
            if sr.get("result"):
                _enrich_worker(sr["result"])
    if d.get("final_result"):
        _enrich_worker(d["final_result"])

    return d


def _list_summary(items: list[dict], kind: str) -> str:
    """Build a summary line like '5 workers: 2 running, 2 ok, 1 failed'."""
    counts: dict[str, int] = {}
    for item in items:
        st = item.get("status", "pending")
        counts[st] = counts.get(st, 0) + 1
    order = ("running", "succeeded", "failed", "pending", "cancelled", "timed_out", "cleanup_failed")
    parts = []
    for st in order:
        c = counts.get(st, 0)
        if c:
            parts.append(f"{c} {_STATUS_LABELS.get(st, st).lower()}")
    detail = f": {', '.join(parts)}" if parts else ""
    return f"{len(items)} {kind}{detail}"


def _default_supervisor() -> FleetSupervisor:
    allowed_repos = [
        repo.strip()
        for repo in os.environ.get("FLEET_ALLOWED_REPOS", "").split(",")
        if repo.strip()
    ]
    return FleetSupervisor(
        base_dir=os.environ.get("FLEET_BASE_DIR"),
        default_model=os.environ.get("FLEET_DEFAULT_MODEL", "gpt-5.4"),
        default_gemini_model=os.environ.get(
            "FLEET_GEMINI_DEFAULT_MODEL", "gemini-2.5-pro"
        ),
        default_claude_model=os.environ.get(
            "FLEET_CLAUDE_DEFAULT_MODEL", "claude-opus-4-6"
        ),
        default_reasoning_effort=os.environ.get("FLEET_REASONING_EFFORT", "max"),
        default_timeout=int(os.environ.get("FLEET_DEFAULT_TIMEOUT", "600")),
        max_concurrent=int(os.environ.get("FLEET_MAX_CONCURRENT", "50")),
        allowed_repos=allowed_repos or None,
        default_executor=os.environ.get("FLEET_DEFAULT_EXECUTOR", "codex"),
        max_spawn_depth=int(os.environ.get("FLEET_MAX_SPAWN_DEPTH", "2")),
        rate_limit_max_retries=int(os.environ.get("FLEET_RATE_LIMIT_MAX_RETRIES", "3")),
        rate_limit_base_delay=float(os.environ.get("FLEET_RATE_LIMIT_BASE_DELAY", "4.0")),
        rate_limit_max_delay=float(os.environ.get("FLEET_RATE_LIMIT_MAX_DELAY", "60.0")),
        stale_timeout=float(os.environ.get("FLEET_STALE_TIMEOUT", "180")),
        stale_max_restarts=int(os.environ.get("FLEET_STALE_MAX_RESTARTS", "2")),
    )


def create_server(supervisor: Optional[FleetSupervisor] = None) -> FastMCP:
    """Create the MCP server with all tools registered."""
    mcp = FastMCP("codefleet")

    if supervisor is None:
        supervisor = _default_supervisor()

    def _dump(payload):
        """Call .model_dump() if available, otherwise return as-is."""
        return payload.model_dump() if hasattr(payload, "model_dump") else payload

    def _error_response(exc: Exception) -> dict:
        """Build a structured error dict from an exception."""
        error_type = type(exc).__name__
        if not isinstance(exc, (ValueError, RuntimeError, FileNotFoundError)):
            logger.warning("Unexpected error in MCP tool: %s", exc, exc_info=True)
        return {"error": error_type, "details": str(exc)}

    # --- Worker tools ---

    @mcp.tool()
    def healthcheck() -> dict:
        """Return a basic capability report. Lets Claude verify the MCP server is reachable and the local environment is sane."""
        return supervisor.healthcheck()

    @mcp.tool()
    def executor_guide() -> dict:
        """Return benchmark-backed guidance on executor strengths and weaknesses.

        Only call this when the user explicitly asks which executor to use, or when
        you genuinely don't know which executor fits the task. For most tasks the
        inline hints in create_worker/create_workflow are sufficient."""
        return {
            "codex": {
                "model": "gpt-5.4",
                "best_for": [
                    "Terminal/CLI debugging and DevOps scripts",
                    "Code review and PR review (found 500+ zero-days in testing)",
                    "Git operations and build system troubleshooting",
                    "Token-efficient execution (~4x fewer tokens than Claude)",
                    "Quick targeted fixes and single-file changes",
                ],
                "weaker_at": [
                    "Frontend/React (frequent mistakes on basic React tasks)",
                    "Large-scale multi-file refactoring",
                    "Extended sessions (can become erratic)",
                ],
                "benchmarks": "SWE-Bench Pro 57.7%, Terminal-Bench 77.3%",
            },
            "gemini": {
                "model": "gemini-2.5-pro",
                "best_for": [
                    "Scientific and research code (SciCode leader at 59.0%)",
                    "UI/frontend generation from mockups, PDFs, or sketches",
                    "Competitive programming tasks (LiveCodeBench Elo 2887)",
                    "Large codebases that need full context (1M token window)",
                    "Test template generation and standard validation patterns",
                    "Budget-conscious tasks (generous free tier)",
                ],
                "weaker_at": [
                    "First-pass correctness (50-60% vs Claude's 95%)",
                    "Complex architectural refactoring",
                    "Speed on complex projects (slower than Claude)",
                ],
                "benchmarks": "SWE-Bench 80.6%, Terminal-Bench 78.4%, SciCode 59.0%",
            },
            "claude": {
                "model": "claude-opus-4-6",
                "best_for": [
                    "Multi-file architecture and complex refactoring",
                    "First-pass code correctness (95% on first try)",
                    "Security analysis and code review",
                    "Issue-to-PR workflows (implementation + tests + docs)",
                    "Codebase analysis and dependency mapping",
                    "Orchestrating multi-agent workflows",
                ],
                "weaker_at": [
                    "Token consumption (~4x more than Codex)",
                    "Terminal-native tasks (65.4% on Terminal-Bench)",
                    "Cost (Opus is expensive; use Sonnet for routine tasks)",
                ],
                "benchmarks": "SWE-Bench 80.8%, GPQA Diamond 91.3%",
            },
            "routing_rules": [
                "For implement-then-review workflows: Codex implements, Claude reviews",
                "For frontend/UI: Gemini implements, Claude reviews",
                "For refactoring/architecture: Claude implements and reviews",
                "For terminal/DevOps scripts: Codex only",
                "For scientific code: Gemini only",
                "For competitive/parallel implementation: Codex vs Claude, then Claude evaluates",
                "For budget-sensitive batch work: Gemini",
            ],
        }

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
        """Launch a worker in an isolated git worktree.

        Prefer create_workflow for multi-step tasks. Use create_worker for simple one-off tasks
        or when you want to launch multiple workers in parallel yourself.

        Executor hints (no need to call executor_guide for these):
        - 'codex': terminal/CLI, code review, DevOps, quick fixes. Token-efficient.
        - 'gemini': frontend/UI, scientific code, large codebases, budget tasks.
        - 'claude': multi-file refactoring, architecture, security, first-pass correctness."""
        try:
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
            return _enrich_worker(_dump(result))
        except (ValueError, RuntimeError, FileNotFoundError, Exception) as exc:
            return _error_response(exc)

    @mcp.tool()
    def get_worker_status(worker_id: str) -> dict:
        """Return current status and metadata for a worker."""
        try:
            result = supervisor.get_worker_status(worker_id)
            return _enrich_worker(_dump(result))
        except (ValueError, RuntimeError, FileNotFoundError, Exception) as exc:
            return _error_response(exc)

    @mcp.tool()
    def list_workers(
        statuses: Optional[list[str]] = None,
        limit: int = 25,
    ) -> dict:
        """List recent workers, optionally filtered by status."""
        try:
            results = supervisor.list_workers(statuses=statuses, limit=limit)
            workers = [_enrich_worker(_dump(r)) for r in results]
            return {
                "workers": workers,
                "count": len(workers),
                "summary": _list_summary(workers, "workers"),
            }
        except (ValueError, RuntimeError, FileNotFoundError, Exception) as exc:
            return _error_response(exc)

    @mcp.tool()
    def collect_worker_result(
        worker_id: str,
        include_logs: bool = False,
        log_tail_lines: int = 80,
    ) -> dict:
        """Return worker metadata plus parsed result.json and optional log tails."""
        try:
            result = supervisor.collect_worker_result(
                worker_id=worker_id,
                include_logs=include_logs,
                log_tail_lines=log_tail_lines,
            )
            return _enrich_worker(result)
        except (ValueError, RuntimeError, FileNotFoundError, Exception) as exc:
            return _error_response(exc)

    @mcp.tool()
    def cancel_worker(worker_id: str) -> dict:
        """Cancel a running worker."""
        try:
            result = supervisor.cancel_worker(worker_id)
            return _enrich_worker(_dump(result))
        except (ValueError, RuntimeError, FileNotFoundError, Exception) as exc:
            return _error_response(exc)

    @mcp.tool()
    def cleanup_worker(
        worker_id: str,
        remove_branch: bool = True,
        remove_worktree_dir: bool = True,
    ) -> dict:
        """Remove worktree and optional branch for a terminal worker."""
        try:
            return supervisor.cleanup_worker(
                worker_id=worker_id,
                remove_branch=remove_branch,
                remove_worktree_dir=remove_worktree_dir,
            )
        except (ValueError, RuntimeError, FileNotFoundError, Exception) as exc:
            return _error_response(exc)

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
        """Define and start a multi-stage DAG workflow with cross-executor collaboration.

        Each stage is a dict with these fields:
          name (str, required): Stage name.
          executor (str, required): One of "codex", "gemini", "claude" (lowercase).
          prompt_template (str, required): Prompt text. Use {task_prompt} for the workflow task,
              {stage_N_summary}, {stage_N_files}, {stage_N_next_steps} for prior stage results.
              Literal curly braces (JSON, code examples) are safe — only known variables are substituted.
          depends_on (list[int], default=[]): Stage indices this stage waits for. [] = run immediately.
          worktree_strategy (str, default="new"): "new" = fresh worktree, "inherit" = reuse
              the first dependency's worktree (requires depends_on to be non-empty).
          model (str, optional): Override the executor's default model.
          timeout_seconds (int, optional): Override default timeout for this stage.

        Example stages:
          [{"name": "implement", "executor": "codex", "prompt_template": "{task_prompt}"},
           {"name": "review", "executor": "claude", "worktree_strategy": "inherit",
            "prompt_template": "Review: {stage_0_summary}", "depends_on": [0]}]

        IMPORTANT: Maximize parallelism. Break work into the smallest independent units and
        run them as parallel stages (depends_on=[]) rather than one big sequential stage.

        Common patterns:
        - Backend: Codex implements -> Claude reviews -> Codex refines
        - Frontend/UI: Gemini implements -> Claude reviews
        - Competitive: Codex + Claude in parallel -> Claude evaluates
        - Fan-out: N parallel Codex workers (one per file/module) -> Claude reviews all"""
        try:
            result = supervisor.create_workflow(
                name=name,
                repo_path=repo_path,
                task_prompt=task_prompt,
                stages=stages,
                base_ref=base_ref,
                timeout_seconds=timeout_seconds,
            )
            return _enrich_workflow(_dump(result), supervisor)
        except (ValueError, RuntimeError, FileNotFoundError, Exception) as exc:
            return _error_response(exc)

    @mcp.tool()
    def get_workflow_status(workflow_id: str) -> dict:
        """Return workflow state with per-stage worker statuses."""
        try:
            result = supervisor.get_workflow_status(workflow_id)
            return _enrich_workflow(_dump(result), supervisor)
        except (ValueError, RuntimeError, FileNotFoundError, Exception) as exc:
            return _error_response(exc)

    @mcp.tool()
    def list_workflows(
        statuses: Optional[list[str]] = None,
        limit: int = 25,
    ) -> dict:
        """List workflows with optional status filter."""
        try:
            results = supervisor.list_workflows(statuses=statuses, limit=limit)
            workflows = [_enrich_workflow(_dump(r), supervisor) for r in results]
            return {
                "workflows": workflows,
                "count": len(workflows),
                "summary": _list_summary(workflows, "workflows"),
            }
        except (ValueError, RuntimeError, FileNotFoundError, Exception) as exc:
            return _error_response(exc)

    @mcp.tool()
    def cancel_workflow(workflow_id: str) -> dict:
        """Cancel all running stages and mark workflow cancelled."""
        try:
            result = supervisor.cancel_workflow(workflow_id)
            return _enrich_workflow(_dump(result), supervisor)
        except (ValueError, RuntimeError, FileNotFoundError, Exception) as exc:
            return _error_response(exc)

    @mcp.tool()
    def collect_workflow_result(
        workflow_id: str,
        include_all_stages: bool = False,
        include_logs: bool = False,
    ) -> dict:
        """Get the final stage's result (or all stages' results)."""
        try:
            result = supervisor.collect_workflow_result(
                workflow_id=workflow_id,
                include_all_stages=include_all_stages,
                include_logs=include_logs,
            )
            return _enrich_workflow(result, supervisor)
        except (ValueError, RuntimeError, FileNotFoundError, Exception) as exc:
            return _error_response(exc)

    @mcp.tool()
    def cleanup_workflow(workflow_id: str) -> dict:
        """Clean up all worktrees, branches, and worker dirs for a terminal workflow."""
        try:
            return supervisor.cleanup_workflow(workflow_id)
        except (ValueError, RuntimeError, FileNotFoundError, Exception) as exc:
            return _error_response(exc)

    return mcp


def main():
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
