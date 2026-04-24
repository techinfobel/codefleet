#!/usr/bin/env python3
"""
Simulated demo for asciinema/terminal recording.

Run: python demo/demo.py
Record: asciinema rec demo/demo.cast -c "python demo/demo.py"
Convert: agg demo/demo.cast demo/demo.gif --cols 100 --rows 32
"""

import sys
import time

# ANSI
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
BLUE = "\033[34m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"
RESET = "\033[0m"
CHECK = f"{GREEN}\u2713{RESET}"
CROSS = f"{RED}\u2717{RESET}"
WARN = f"{YELLOW}\u26a0{RESET}"
SPIN = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
BAR_FILL = "\u2588"
BAR_EMPTY = "\u2591"


def out(text=""):
    sys.stdout.write(text)
    sys.stdout.flush()


def outln(text=""):
    out(text + "\n")


def typewrite(text, delay=0.025):
    for ch in text:
        out(ch)
        time.sleep(delay)
    outln()


def spin(msg, duration=1.5, result=None, color=CYAN):
    end = time.monotonic() + duration
    i = 0
    while time.monotonic() < end:
        out(f"\r    {color}{SPIN[i % len(SPIN)]}{RESET} {msg}")
        time.sleep(0.08)
        i += 1
    if result:
        out(f"\r    {CHECK} {msg} {DIM}({result}){RESET}\n")
    else:
        out(f"\r    {CHECK} {msg}{RESET}\n")


def progress_bar(done, total, width=20):
    filled = int(width * done / total) if total else 0
    return f"{BAR_FILL * filled}{BAR_EMPTY * (width - filled)}"


def status_line(done, total, running, pending, elapsed):
    bar = progress_bar(done, total)
    parts = []
    if done:
        parts.append(f"{done} done")
    if running:
        parts.append(f"{running} running")
    if pending:
        parts.append(f"{pending} pending")
    detail = ", ".join(parts)
    return f"  {bar} {done}/{total} stages complete ({detail}) \u2014 {elapsed}"


def main():
    outln()
    outln(f"  {BOLD}codefleet{RESET} {DIM}v0.3.5{RESET} \u2014 orchestrate fleets of AI coding agents")
    outln()
    time.sleep(0.6)

    # Workflow definition
    outln(f"  {BOLD}Workflow:{RESET} refactor-auth-module")
    outln(f"  {DIM}Task: Refactor auth module \u2014 backend, frontend, tests, and review{RESET}")
    outln()
    time.sleep(0.4)

    # DAG visualization
    outln(f"  {DIM}\u250c\u2500 DAG \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510{RESET}")
    outln(f"  {DIM}\u2502{RESET}  {YELLOW}[0]{RESET} backend-api  {DIM}(codex){RESET}  \u2500\u2500\u252c\u2500\u2500> {BLUE}[2]{RESET} tests   {DIM}(codex){RESET} \u2500\u2510  {DIM}\u2502{RESET}")
    outln(f"  {DIM}\u2502{RESET}  {CYAN}[1]{RESET} frontend-ui  {DIM}(gemini){RESET} \u2500\u2500\u2534\u2500\u2500> {MAGENTA}[3]{RESET} review  {DIM}(claude){RESET} \u2500\u2518  {DIM}\u2502{RESET}")
    outln(f"  {DIM}\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518{RESET}")
    outln()
    time.sleep(0.6)

    # --- Parallel stages 0 + 1 ---
    outln(status_line(0, 4, 2, 2, "0s"))
    outln()

    outln(f"  {YELLOW}[0]{RESET} {BOLD}backend-api{RESET}  {DIM}codex / gpt-5.5{RESET}")
    spin("Creating worktree", 0.6, "codex/backend-api/w_8f3a21")
    spin("Codex writing code", 1.8, "5 files changed")
    spin("Running tests", 0.8, "18 passed")
    outln(f"      {CHECK} {GREEN}[OK]{RESET} backend-api \u2014 1m 05s")
    outln()

    # Parallel: stage 1 starts alongside
    outln(f"  {CYAN}[1]{RESET} {BOLD}frontend-ui{RESET}  {DIM}gemini / gemini-3.1-pro-preview{RESET}")
    spin("Creating worktree", 0.5, "gemini/frontend-ui/w_c4d912")

    # Rate limit retry
    spin("Gemini writing code", 0.8)
    outln(f"      {WARN}  {YELLOW}429 rate limited{RESET}")
    outln(f"      {DIM}--- [codefleet] Rate limited. Retry 1/3 after 4s backoff ---{RESET}")

    # Backoff animation
    for i in range(4, 0, -1):
        out(f"\r    {YELLOW}{SPIN[i % len(SPIN)]}{RESET} Waiting {i}s...")
        time.sleep(0.4)
    out(f"\r    {CHECK} Retrying...             \n")

    spin("Gemini writing code", 1.5, "3 files changed, retry 1")
    outln(f"      {CHECK} {GREEN}[OK]{RESET} frontend-ui \u2014 0m 48s {DIM}(1 retry){RESET}")
    outln()

    outln(status_line(2, 4, 0, 2, "1m 05s"))
    time.sleep(0.3)
    outln()

    # --- Stage 2: tests (depends on 0) ---
    outln(status_line(2, 4, 1, 1, "1m 08s"))
    outln()
    outln(f"  {BLUE}[2]{RESET} {BOLD}tests{RESET}  {DIM}codex / gpt-5.5 (depends on: 0){RESET}")
    spin("Inheriting worktree from backend-api", 0.4, "same branch")
    spin("Codex writing tests", 1.5, "2 files changed")
    spin("Running tests", 0.6, "24 passed")
    outln(f"      {CHECK} {GREEN}[OK]{RESET} tests \u2014 0m 32s")
    outln()

    # --- Stage 3: review (depends on 0, 1, 2) ---
    outln(status_line(3, 4, 1, 0, "1m 40s"))
    outln()
    outln(f"  {MAGENTA}[3]{RESET} {BOLD}review{RESET}  {DIM}claude / claude-opus-4-7 / effort: high (depends on: 0, 1, 2){RESET}")
    spin("Inheriting worktree from tests", 0.4, "same branch")
    spin("Claude reviewing all changes", 1.8, "review complete")
    outln(f"      {CHECK} {GREEN}[OK]{RESET} review \u2014 0m 15s")
    outln()

    # --- Final summary ---
    outln(status_line(4, 4, 0, 0, "1m 55s"))
    outln()
    time.sleep(0.3)

    outln(f"  {BOLD}{GREEN}\u2501\u2501\u2501 Workflow succeeded{RESET} {DIM}(4 stages, 3 executors, 1m 55s){RESET}")
    outln()
    outln(f"  {DIM}Stage      Executor   Status   Elapsed   Files{RESET}")
    outln(f"  {YELLOW}[0]{RESET} backend  codex      {GREEN}[OK]{RESET}      1m 05s    5 changed")
    outln(f"  {CYAN}[1]{RESET} frontend gemini     {GREEN}[OK]{RESET}      0m 48s    3 changed {DIM}(1 retry){RESET}")
    outln(f"  {BLUE}[2]{RESET} tests    codex      {GREEN}[OK]{RESET}      0m 32s    2 changed")
    outln(f"  {MAGENTA}[3]{RESET} review   claude     {GREEN}[OK]{RESET}      0m 15s    All approved")
    outln()
    outln(f"  {DIM}Branch:{RESET}  codex/backend-api/w_8f3a21")
    outln(f"  {DIM}Commits:{RESET} 4 (backend + frontend + tests + review-fixes)")
    outln()


if __name__ == "__main__":
    main()
