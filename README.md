# Codex Fleet Supervisor

A local MCP server that lets **Claude Code** orchestrate a fleet of **OpenAI Codex** workers. Claude plans and reviews, Codex executes — each worker runs in its own isolated git worktree with full process supervision, durable state, and structured results.

```
Claude Code  -->  MCP Supervisor  -->  Codex Worker 1 (worktree + branch)
                                  -->  Codex Worker 2 (worktree + branch)
                                  -->  Codex Worker N ...
```

## What This Does

- **Claude Code** decomposes work into tasks and delegates them
- **The supervisor** launches Codex workers, tracks state in SQLite, manages git worktrees, enforces timeouts, and captures logs
- **Codex workers** execute in isolated worktrees and write structured `result.json` files
- **Claude Code** reviews results and decides to accept, reject, retry, or merge

This is not a demo script. It is designed for ongoing daily use across repos and sessions.

## Prerequisites

- **Python 3.11+**
- **Git** (with worktree support)
- **Claude Code** — [Install from Anthropic](https://docs.anthropic.com/en/docs/claude-code)
- **OpenAI Codex CLI** — Install via npm:
  ```bash
  npm install -g @openai/codex
  ```
  Verify it works:
  ```bash
  codex --version
  ```
- **OpenAI API key** — Codex needs this set in your environment:
  ```bash
  export OPENAI_API_KEY="your-key-here"
  ```

## Installation

1. **Clone and set up:**
   ```bash
   git clone <this-repo-url>
   cd codex-fleet-supervisor
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

2. **Verify it works:**
   ```bash
   python -m pytest tests/ -q
   ```
   You should see 115 tests passing.

## Register with Claude Code

Run this one command to register the MCP server:

```bash
claude mcp add --transport stdio --scope user codex-fleet -- \
  /path/to/codex-fleet-supervisor/.venv/bin/python \
  -m codex_fleet_supervisor.server
```

Replace `/path/to/codex-fleet-supervisor` with the actual path where you cloned the repo.

**With environment variables** (optional — to restrict which repos can be used):

```bash
claude mcp add --transport stdio --scope user \
  --env FLEET_ALLOWED_REPOS=/Users/you/projects/repo-a,/Users/you/projects/repo-b \
  codex-fleet -- \
  /path/to/codex-fleet-supervisor/.venv/bin/python \
  -m codex_fleet_supervisor.server
```

**Scope options:**
- `--scope user` — available in all your Claude Code sessions (recommended)
- `--scope project` — only in the current project (saved to `.mcp.json`)
- `--scope local` — only in the current project, not shared with team

**Verify registration:**
```bash
claude mcp list
```

Then start a new Claude Code session and use `/mcp` to confirm the server connects.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLEET_BASE_DIR` | `~/.codex-fleet` | Where worker data and the SQLite DB are stored |
| `FLEET_DEFAULT_MODEL` | `gpt-5.4` | Default Codex model for workers |
| `FLEET_DEFAULT_TIMEOUT` | `600` | Worker timeout in seconds (10 minutes) |
| `FLEET_MAX_CONCURRENT` | `10` | Maximum simultaneous workers |
| `FLEET_ALLOWED_REPOS` | _(empty = allow all)_ | Comma-separated list of allowed repo paths |

## MCP Tools

Once registered, Claude Code gets access to these tools:

| Tool | Description |
|---|---|
| `healthcheck` | Verify the supervisor, Codex, and Git are all available |
| `create_worker` | Launch a Codex worker in an isolated git worktree |
| `get_worker_status` | Check the current status of a worker |
| `list_workers` | List recent workers, optionally filtered by status |
| `collect_worker_result` | Get parsed results and optional log tails from a worker |
| `cancel_worker` | Cancel a running worker |
| `cleanup_worker` | Remove worktree, branch, and artifacts for a finished worker |

## Example Workflow

Start a Claude Code session and interact naturally:

```
You: I need to add input validation to the user registration endpoint
     and write tests for it. Use codex workers.

Claude: I'll break this into two tasks and delegate to Codex workers.

        [calls healthcheck — confirms environment is ready]

        [calls create_worker — task: "add-input-validation"
         with a detailed prompt about what to validate]

        [calls create_worker — task: "write-registration-tests"
         with a prompt about test cases to cover]

        [polls get_worker_status until both complete]

        [calls collect_worker_result for each — reviews the code,
         checks that tests pass, verifies result.json]

        Worker 1 succeeded — validation logic looks good.
        Worker 2 succeeded — 8 test cases covering edge cases.

        I'll merge both branches into your working branch now.

        [calls cleanup_worker for each]
```

## How Workers Run

Each worker created by `create_worker` goes through this lifecycle:

1. **Validation** — repo path, git status, allowlist, and concurrency checks
2. **Setup** — unique worker ID, git worktree, branch (`codex/<task>/<id>`), prompt file
3. **Launch** — `codex exec` runs in the worktree with stdout/stderr captured
4. **Monitor** — background thread watches for completion or timeout
5. **Evaluate** — process exit code checked, then `result.json` parsed and validated
6. **Terminal state** — worker marked `succeeded`, `failed`, `timed_out`, or `cancelled`

A worker is only `succeeded` if it exits 0 **and** writes a valid `result.json`.

### Worker Result Contract

Every Codex worker is instructed to write a `result.json`:

```json
{
  "summary": "Added email and password validation to /api/register",
  "files_changed": ["src/routes/register.py", "src/validators.py"],
  "tests": [
    {
      "command": "pytest tests/test_register.py",
      "status": "passed",
      "details": "5 passed"
    }
  ],
  "commits": ["a1b2c3d"],
  "next_steps": ["Add rate limiting"],
  "status": "completed"
}
```

## Filesystem Layout

All worker data lives under `~/.codex-fleet/` by default:

```
~/.codex-fleet/
  fleet.db                          # SQLite database (survives restarts)
  workers/
    w_a1b2c3d4e5f6/
      prompt.txt                    # Task prompt sent to Codex
      result.json                   # Structured output from Codex
      stdout.log                    # Captured stdout
      stderr.log                    # Captured stderr
      meta.json                     # Worker metadata
      worktree/                     # Isolated git worktree
```

## Troubleshooting

**"codex not found" in healthcheck**
- Make sure `codex` is installed globally: `npm install -g @openai/codex`
- Verify it's on your PATH: `which codex`

**"Repo not in allowlist"**
- If you set `FLEET_ALLOWED_REPOS`, make sure the repo path is included
- Use absolute paths, comma-separated
- Or leave it empty to allow all repos

**Worker stuck in "running"**
- Workers have a default 10-minute timeout
- Use `cancel_worker` to stop a stuck worker
- Check logs with `collect_worker_result(worker_id, include_logs=True)`

**"Concurrency limit reached"**
- Default is 10 concurrent workers
- Increase with `FLEET_MAX_CONCURRENT` env var
- Or clean up finished workers with `cleanup_worker`

**MCP server not showing in Claude Code**
- MCP servers load on session start — restart Claude Code after registering
- Verify with `claude mcp list`
- Check `/mcp` inside a session for connection status

**Workers fail with "Result validation failed"**
- Codex didn't write a valid `result.json`
- Use `collect_worker_result(worker_id, include_logs=True)` to see what happened
- The stderr log often contains the reason

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=codex_fleet_supervisor --cov-report=term-missing

# Run the server directly (stdio mode)
python -m codex_fleet_supervisor.server
```

## Architecture

```
src/codex_fleet_supervisor/
  models.py          # Pydantic models — WorkerStatus, WorkerResult, WorkerRecord
  result_schema.py   # result.json parsing and validation
  store.py           # SQLite store — thread-safe, WAL mode, durable
  git_ops.py         # Git worktree create/remove, branch management
  worker_runtime.py  # Subprocess launching with timeout monitoring
  supervisor.py      # Core FleetSupervisor tying everything together
  server.py          # FastMCP stdio server exposing all tools
```

## License

MIT
