#!/usr/bin/env python3
"""
Simulated demo for asciinema/terminal recording.

Run: python demo/demo.py
Record: asciinema rec demo.cast -c "python demo/demo.py"
Convert: agg demo.cast demo.gif  (or svg-term --in demo.cast --out demo.svg)
"""

import sys
import time

# ANSI colors
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
CHECK = f"{GREEN}\u2713{RESET}"
SPIN = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]


def typewrite(text, delay=0.03):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def spin_wait(msg, duration=1.5, result="done"):
    frames = SPIN
    end_time = time.monotonic() + duration
    i = 0
    while time.monotonic() < end_time:
        sys.stdout.write(f"\r  {CYAN}{frames[i % len(frames)]}{RESET} {msg}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write(f"\r  {CHECK} {msg} {DIM}({result}){RESET}\n")
    sys.stdout.flush()


def main():
    print()
    print(f"  {BOLD}codefleet{RESET} {DIM}v0.3.0{RESET} \u2014 orchestrate fleets of AI coding agents")
    print()
    time.sleep(0.5)

    # Show the workflow being created
    print(f"  {BOLD}Workflow:{RESET} write \u2192 review \u2192 refine")
    print(f"  {DIM}Task: Add input validation to the registration endpoint{RESET}")
    print()
    time.sleep(0.3)

    # Stage 1: Implement (Codex)
    print(f"  {YELLOW}Stage 1/3{RESET} \u2502 {BOLD}implement{RESET} {DIM}(codex / gpt-5.4){RESET}")
    spin_wait("Creating worktree", 0.8, "codex/implement/w_a1b2c3")
    spin_wait("Codex writing code", 2.0, "4 files changed")
    spin_wait("Running tests", 1.0, "12 passed")
    print(f"         {CHECK} {GREEN}completed{RESET} \u2014 Added validation to register(), login(), update_profile()")
    print()
    time.sleep(0.3)

    # Stage 2: Review (Claude)
    print(f"  {MAGENTA}Stage 2/3{RESET} \u2502 {BOLD}review{RESET} {DIM}(claude / claude-opus-4-6 / effort: high){RESET}")
    spin_wait("Inheriting worktree from stage 1", 0.5, "same branch")
    spin_wait("Claude reviewing changes", 2.5, "review complete")
    print(f"         {CHECK} {GREEN}completed{RESET} \u2014 3 issues found: missing email format check, no rate limit, weak password regex")
    print()
    time.sleep(0.3)

    # Stage 3: Refine (Codex)
    print(f"  {BLUE}Stage 3/3{RESET} \u2502 {BOLD}refine{RESET} {DIM}(codex / gpt-5.4){RESET}")
    spin_wait("Inheriting worktree from stage 2", 0.5, "same branch")
    spin_wait("Codex addressing review feedback", 2.0, "3 files changed")
    spin_wait("Running tests", 1.0, "15 passed")
    print(f"         {CHECK} {GREEN}completed{RESET} \u2014 All 3 review issues addressed, tests updated")
    print()
    time.sleep(0.5)

    # Summary
    print(f"  {BOLD}{GREEN}\u2501\u2501\u2501 Workflow succeeded{RESET} {DIM}(3 stages, 2 executors, 38s){RESET}")
    print()
    print(f"  {DIM}Files changed:{RESET} src/auth/validators.py, src/auth/routes.py,")
    print(f"                src/auth/tests/test_validators.py, src/auth/rate_limit.py")
    print(f"  {DIM}Branch:{RESET}        codex/implement/w_a1b2c3")
    print(f"  {DIM}Commits:{RESET}       3 (implement + review-fixes + refinement)")
    print()


if __name__ == "__main__":
    main()
