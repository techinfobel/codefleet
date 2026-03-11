# Claude Code → Codex Fleet Supervisor

## Implementation Brief for Claude Code

This document describes a concrete, ongoing-use architecture for letting **Claude Code orchestrate a fleet of OpenAI Codex workers** through a **local Python MCP supervisor**.

The goal is not a demo script. The goal is a system that is:
- reusable across repos
- durable across sessions
- observable
- easy for Claude Code to call as tools
- safe enough for ongoing daily use

---

## 1) Executive summary

### Recommended architecture

Use:
- **Claude Code** as the planner, orchestrator, reviewer, and merge authority
- **A local Python MCP supervisor** as the control plane
- **Codex workers** launched via `codex exec`
- **Git worktrees** for isolation
- **SQLite** for durable worker state

### Why this is the best ongoing setup

A shell-only approach is fine for quick prototyping, but becomes messy once you need:
- worker IDs
- status polling
- cancellation
- retries
- structured outputs
- durable logs
- persistent job history
- per-task worktree tracking

A Python MCP supervisor gives Claude Code a clean tool surface such as:
- `create_worker`
- `get_worker_status`
- `collect_worker_result`
- `cancel_worker`
- `cleanup_worker`

This lets Claude reason in terms of **tasks and workers**, not shell flags and temp directories.

---

## 2) System roles

### Claude Code
Claude Code should be responsible for:
- decomposing work into task-sized units
- deciding which tasks to delegate
- spawning Codex workers through MCP tools
- reviewing results
- deciding whether to retry, reject, or merge
- optionally using subagents for planning/review specialization

### Python MCP supervisor
The supervisor should be responsible for:
- launching Codex workers
- creating isolated git worktrees
- tracking lifecycle state in SQLite
- capturing stdout/stderr and result artifacts
- enforcing timeouts
- supporting cancellation and cleanup
- presenting stable MCP tool interfaces to Claude Code

### Codex workers
Each Codex worker should be responsible for:
- operating in a single isolated worktree
- executing a sharply scoped implementation/review/test task
- writing a structured `result.json`
- optionally running tests
- leaving code changes in the worktree branch for later review

---

## 3) Design principles

1. **Claude should manage abstractions, not shell details**
   - Claude should call tools like `create_worker`, not manually build `codex exec` commands repeatedly.

2. **One worker = one worktree = one branch**
   - Avoid shared mutable state.
   - Every worker gets its own branch and worktree.

3. **Results must be structured**
   - Every worker must write a deterministic JSON result contract.

4. **Supervisor owns state**
   - State must survive Claude session loss.
   - Use SQLite, not in-memory tracking only.

5. **Logs are first-class artifacts**
   - stdout/stderr must be kept for debugging failures and blocked runs.

6. **Terminal cleanup is explicit**
   - Workers are not auto-destroyed until Claude has reviewed or merged the output.

7. **Safety by default**
   - Default worker execution should not assume full unsafe autonomy.

---

## 4) Recommended project shape

Start with a minimal but real project structure:

```text
codex-fleet-supervisor/
  README.md
  pyproject.toml
  .env.example
  src/
    codex_fleet_supervisor/
      __init__.py
      server.py
      store.py
      supervisor.py
      models.py
      worker_runtime.py
      git_ops.py
      result_schema.py
  tests/
    test_store.py
    test_result_schema.py
    test_supervisor_smoke.py
```

### Initial shortcut
A single-file server is acceptable for the first working version.

### Ongoing-use target
Refactor into multiple modules once the behavior is validated.

---

## 5) Core workflow

### End-to-end lifecycle

1. Claude Code decides to delegate a task.
2. Claude calls `create_worker(...)` on the local MCP supervisor.
3. The supervisor:
   - validates the repo path
   - creates a unique worker ID
   - creates a git worktree and branch
   - writes the prompt to disk
   - launches `codex exec`
   - persists the worker record in SQLite
4. Codex runs in the worktree.
5. Codex writes a `result.json` artifact.
6. Supervisor marks the worker as terminal when done.
7. Claude polls using `get_worker_status(...)`.
8. Claude reads `collect_worker_result(...)`.
9. Claude decides to:
   - accept
   - reject
   - retry
   - spawn follow-up workers
   - merge externally
10. After review/merge, Claude calls `cleanup_worker(...)`.

---

## 6) Required MCP tools

Implement the following tools for Claude Code.

### 6.1 `healthcheck`
Return a basic capability report.

#### Purpose
Lets Claude verify the MCP server is reachable and the local environment is sane.

#### Expected output
```json
{
  "ok": true,
  "app": "codex-fleet-supervisor",
  "db_path": "...",
  "base_dir": "...",
  "codex_found": true,
  "codex_path": "...",
  "git_found": true,
  "git_path": "...",
  "default_model": "gpt-5.3-codex"
}
```

---

### 6.2 `create_worker`
Launch a new Codex worker in an isolated git worktree.

#### Inputs
- `repo_path: str`
- `task_name: str`
- `prompt: str`
- `base_ref: str = "HEAD"`
- `model: str = DEFAULT_MODEL`
- `timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS`
- `permission_mode: Literal["default", "safe", "yolo"] = "default"`
- `profile: Optional[str] = None`
- `tags: Optional[list[str]] = None`
- `metadata: Optional[dict] = None`
- `extra_codex_args: Optional[list[str]] = None`
- `parent_worker_id: Optional[str] = None`

#### Behavior
- validate repo exists and is a git repo
- generate worker ID
- generate unique branch name such as:
  - `codex/<sanitized-task-name>/<worker-id>`
- create worker directory
- create git worktree
- persist prompt and metadata to disk
- launch `codex exec`
- persist worker record in SQLite
- start monitor thread/process for timeout and status updates

#### Return value
A worker status payload.

---

### 6.3 `get_worker_status`
Return current status and metadata for a worker.

#### Input
- `worker_id: str`

#### Return fields
- worker id
- task name
- status
- repo path
- branch name
- worktree path
- worker dir
- model
- profile
- created/started/ended timestamps
- timeout
- pid
- exit code
- retry count
- tags
- metadata
- error message
- file paths for logs/result/prompt/meta

---

### 6.4 `list_workers`
List recent workers, optionally filtered by status.

#### Inputs
- `statuses: Optional[list[str]] = None`
- `limit: int = 25`

#### Useful statuses
- `running`
- `succeeded`
- `failed`
- `cancelled`
- `timed_out`
- `cleanup_failed`

---

### 6.5 `collect_worker_result`
Return the worker record plus parsed `result.json`.

#### Inputs
- `worker_id: str`
- `include_logs: bool = False`
- `log_tail_lines: int = 80`

#### Behavior
- read worker metadata from SQLite
- parse `result.json` if present
- optionally return tail of stdout/stderr for debugging

#### Return value
Should include all status fields plus:
- `result`
- optionally `stdout_tail`
- optionally `stderr_tail`

---

### 6.6 `cancel_worker`
Cancel a running worker.

#### Input
- `worker_id: str`

#### Behavior
- terminate the subprocess or process group
- mark worker as cancelled
- record terminal timestamp

---

### 6.7 `cleanup_worker`
Clean up a terminal worker’s worktree and optionally delete its branch.

#### Inputs
- `worker_id: str`
- `remove_branch: bool = True`
- `remove_worktree_dir: bool = True`

#### Behavior
- refuse cleanup if worker is not terminal
- remove git worktree
- optionally delete branch
- optionally delete worker artifact directory
- return cleanup summary

---

## 7) Worker result contract

Every Codex worker must write a `result.json` file.

### Required schema
```json
{
  "summary": "string",
  "files_changed": ["path/to/file.py"],
  "tests": [
    {
      "command": "pytest tests/test_x.py",
      "status": "passed",
      "details": "2 passed"
    }
  ],
  "commits": ["optional git commit hashes or empty"],
  "next_steps": ["optional follow-up suggestions"],
  "status": "completed"
}
```

### Allowed `status` values
- `completed`
- `blocked`

### Allowed test status values
- `passed`
- `failed`
- `not_run`

### Important rule
The supervisor should not rely only on process exit code.
A worker is only a clean success if:
- process exits successfully, and
- `result.json` exists and validates

---

## 8) Suggested SQLite schema

Use a `workers` table.

```sql
CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,
    task_name TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    worktree_path TEXT NOT NULL,
    worker_dir TEXT NOT NULL,
    model TEXT NOT NULL,
    profile TEXT,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    ended_at REAL,
    timeout_seconds INTEGER NOT NULL,
    pid INTEGER,
    exit_code INTEGER,
    codex_command TEXT NOT NULL,
    prompt TEXT NOT NULL,
    result_json_path TEXT NOT NULL,
    stdout_path TEXT NOT NULL,
    stderr_path TEXT NOT NULL,
    prompt_path TEXT NOT NULL,
    meta_path TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    parent_worker_id TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT
);
```

Recommended indexes:
```sql
CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status);
CREATE INDEX IF NOT EXISTS idx_workers_created_at ON workers(created_at);
```

---

## 9) Filesystem layout

Recommended base directory:

```text
~/.codex-fleet/
  fleet.db
  workers/
    w_abcd1234ef56/
      prompt.txt
      result.json
      stdout.log
      stderr.log
      meta.json
      worktree/
```

### Required artifacts per worker
- `prompt.txt`
- `result.json`
- `stdout.log`
- `stderr.log`
- `meta.json`
- `worktree/`

---

## 10) Git/worktree behavior

### Required behavior
- use `git worktree add -b <branch> <path> <base_ref>`
- each worker gets a unique branch
- worker runs from its worktree directory

### Cleanup behavior
- `git worktree remove --force <worktree>`
- optionally `git branch -D <branch>`

### Rule
Do not let multiple workers share the same working tree.

---

## 11) Codex execution model

The supervisor should launch Codex via `codex exec`.

### General pattern
- pass an instruction telling Codex to:
  - read the full task prompt from a file path
  - do the task in the current git worktree
  - run relevant tests where appropriate
  - write a strict JSON object to `result.json`

### Example execution idea
```bash
codex exec --model gpt-5.3-codex "Read the task prompt from /path/prompt.txt. Work in the current git worktree. Run relevant tests where appropriate. Write a single JSON object to /path/result.json with schema: {...}."
```

### Important implementation notes
- capture stdout/stderr separately
- launch in its own process group if possible
- preserve environment variables unless you intentionally sandbox further
- support optional extra Codex CLI args

---

## 12) Process supervision

### Recommended approach
- launch worker process with `subprocess.Popen`
- keep PID in SQLite
- maintain an in-memory map of active `worker_id -> Popen` for the current server process
- also rely on SQLite as durable truth

### Timeout handling
- monitor running workers
- if timeout exceeded:
  - terminate process group
  - escalate to kill if needed
  - mark status `timed_out`
  - write error message

### Terminal statuses
Use these terminal statuses:
- `succeeded`
- `failed`
- `cancelled`
- `timed_out`
- `cleanup_failed`

---

## 13) Validation and error handling

### Required validations
- repo path exists
- repo path is a git repository
- worker ID exists before reads/cancel/cleanup
- cleanup only allowed for terminal workers
- `result.json` must be parsed safely

### Failure behavior
A worker should be marked `failed` if:
- Codex exits non-zero
- `result.json` is missing
- `result.json` is invalid JSON
- the result fails schema validation

### Diagnostic behavior
`collect_worker_result(include_logs=true)` should help Claude diagnose failures without reading raw log files directly.

---

## 14) Security and guardrails

Implement these guardrails as early as possible.

### Strong recommendations
1. **Repo allowlist**
   - Only permit worker creation in explicitly allowed repos or repo roots.

2. **Path normalization**
   - Resolve and normalize all repo/worktree paths.

3. **Default-safe execution**
   - Do not default to the loosest execution mode.

4. **No raw secret injection in prompts**
   - Avoid placing credentials in `prompt.txt`.

5. **Explicit cleanup**
   - Preserve artifacts until Claude has reviewed them.

6. **Schema validation**
   - Validate `result.json` with Pydantic or equivalent.

7. **Optional concurrency limit**
   - Prevent runaway worker spawning.

### Later improvement
Add optional containerized worker execution if stronger isolation is required.

---

## 15) Suggested Python implementation details

### Libraries
Recommended:
- `mcp`
- `pydantic`
- standard library: `sqlite3`, `subprocess`, `threading`, `pathlib`, `json`, `uuid`, `signal`, `time`, `shutil`

### Server style
Use a local **stdio MCP server** compatible with Claude Code.

### Important note
Avoid unnecessary multiprocessing complexity in the MCP server.
Keep the control plane simple and predictable.

---

## 16) Example Claude-facing tool semantics

Claude Code should treat this as an orchestration interface.

### Example usage pattern
1. `healthcheck()`
2. `create_worker(...)`
3. periodically `get_worker_status(...)`
4. when terminal, `collect_worker_result(...)`
5. decide whether to retry/accept/reject
6. once done, `cleanup_worker(...)`

### Example worker types Claude might spawn
- implement a feature
- refactor a module
- add tests
- investigate failing tests
- review a patch
- prepare a migration plan

---

## 17) Suggested MCP tool docstrings

Keep tool docstrings concise and operational.

### Example
- `create_worker`: “Launch a new Codex worker in an isolated git worktree.”
- `get_worker_status`: “Return current status and metadata for a worker.”
- `collect_worker_result`: “Return worker metadata plus parsed result.json and optional log tails.”
- `cancel_worker`: “Cancel a running worker.”
- `cleanup_worker`: “Remove worktree and optional branch for a terminal worker.”

---

## 18) Phased build plan

### Phase 1 — working MVP
Implement:
- stdio MCP server
- SQLite store
- create/status/list/result/cancel/cleanup tools
- worktree creation/removal
- Codex launch via `codex exec`
- timeout monitoring
- result artifact parsing

### Phase 2 — hardening
Add:
- result schema validation
- repo allowlist
- concurrency limits
- retry support
- more robust error taxonomy
- smoke tests

### Phase 3 — ongoing-use ergonomics
Add:
- merge helpers
- queueing
- priority scheduling
- reviewer workers
- metrics/dashboard output
- optional container sandboxing

---

## 19) Acceptance criteria

Claude Code implementation should be considered successful when all of the following work:

1. Claude can connect to the supervisor over local stdio MCP.
2. `healthcheck()` reports Codex and Git availability.
3. `create_worker()` creates:
   - a unique worker record
   - a unique worktree
   - a unique branch
   - persisted prompt/log/result artifact paths
4. A successful Codex run produces a valid `result.json`.
5. `get_worker_status()` reflects live and terminal states correctly.
6. `collect_worker_result()` returns parsed results and log tails when requested.
7. `cancel_worker()` reliably stops a running worker.
8. `cleanup_worker()` removes worktree and optionally branch only after terminal state.
9. State survives MCP server restart because SQLite is durable.
10. Multiple workers can be created sequentially without path collisions.

---

## 20) Nice-to-have extensions

Not required for v1, but recommended later:
- `retry_worker(worker_id)`
- `create_review_worker(...)`
- `merge_worker_branch(...)`
- `get_worker_diff(...)`
- `tail_worker_logs(...)`
- `pause_worker_queue()` / `resume_worker_queue()`
- per-repo concurrency caps

---

## 21) Example implementation prompt for Claude Code

Use the following instruction as the starting prompt for Claude Code:

---

Build a local Python stdio MCP server named `codex-fleet-supervisor`.

Requirements:
- It must expose these MCP tools:
  - `healthcheck`
  - `create_worker`
  - `get_worker_status`
  - `list_workers`
  - `collect_worker_result`
  - `cancel_worker`
  - `cleanup_worker`
- It must launch Codex workers using `codex exec`.
- Each worker must run in its own git worktree and branch.
- It must persist worker state in SQLite.
- It must write and track these worker artifacts:
  - `prompt.txt`
  - `result.json`
  - `stdout.log`
  - `stderr.log`
  - `meta.json`
- It must monitor workers for timeout and terminal state.
- A worker is only a clean success if the process exits successfully and `result.json` exists and validates.
- Implement the project in clean Python with sensible module separation.
- Use Pydantic for the result schema.
- Add a small test suite for the SQLite store and result schema validation.
- Add a README with local setup and Claude Code MCP registration instructions.

Behavioral requirements:
- `cleanup_worker` must refuse cleanup on non-terminal workers.
- `collect_worker_result` must optionally include tail snippets of stdout/stderr.
- Use normalized paths.
- Add a simple repo allowlist mechanism.
- Keep the server suitable for ongoing local use, not just a demo.

---

## 22) Recommended README content for the eventual implementation

Ask Claude Code to include:
- overview
- prerequisites
- installation
- environment variables
- local run instructions
- Claude Code MCP registration example
- example workflow
- troubleshooting section

---

## 23) Final recommendation

### Best authoring setup
Use **Claude Code** to implement and iterate on this MCP supervisor.

### Best runtime setup
Use **Codex** as the delegated execution worker managed by the supervisor.

### Best overall architecture
**Claude Code → Python MCP Supervisor → Codex workers in isolated worktrees**

That gives the cleanest long-term balance of:
- orchestration quality
- maintainability
- observability
- ongoing usability

