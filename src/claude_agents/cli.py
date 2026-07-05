"""Command-line entrypoint: run any loop against a task and print its trace.

Usage:
    python -m claude_agents.cli react "What is 128 * 47, and is it prime?"
    python -m claude_agents.cli orchestrator "Compare the areas of two circles..."
    python -m claude_agents.cli --list
"""
from __future__ import annotations

import argparse
import sys

from .core.tools import default_registry
from .loops import LOOPS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claude-agents",
        description="Run a custom autonomous Claude agent loop.",
    )
    parser.add_argument(
        "loop",
        nargs="?",
        choices=sorted(LOOPS),
        help="Which loop to run.",
    )
    parser.add_argument("task", nargs="?", help="The task/goal for the agent.")
    parser.add_argument(
        "--list", action="store_true", help="List available loops and exit."
    )
    parser.add_argument(
        "--trace", action="store_true", help="Print the full step-by-step trace."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list or not args.loop:
        print("Available loops:")
        for name in sorted(LOOPS):
            print(f"  {name}")
        return 0

    if not args.task:
        print("error: a task is required", file=sys.stderr)
        return 2

    agent_cls = LOOPS[args.loop]
    agent = agent_cls(tools=default_registry())
    result = agent.run(args.task)

    if args.trace:
        print("── trace ──")
        for step in result.steps:
            print(f"  [{step.label}] {step.detail}")
        print("───────────")

    print(result.answer)
    return 0 if result.completed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
