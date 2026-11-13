"""Command line interface: ``simplicity-chain "your task"``."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from .agents import CodeAgent
from .errors import AgentError
from .models import OpenAIServerModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simplicity-chain",
        description="Run a code-writing agent against any OpenAI-compatible model.",
    )
    parser.add_argument("task", help="The task for the agent to solve.")
    parser.add_argument(
        "--model",
        default=os.environ.get("SIMPLICITY_CHAIN_MODEL", "gpt-4o-mini"),
        help="Model id (default: %(default)s).",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="OpenAI-compatible API base, e.g. http://localhost:11434/v1.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key; defaults to $OPENAI_API_KEY (use 'none' for local servers).",
    )
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum agent steps.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--authorized-imports",
        default="",
        help="Comma-separated extra modules the agent may import.",
    )
    parser.add_argument("--instructions", default=None, help="Extra system-prompt instructions.")
    parser.add_argument("--quiet", action="store_true", help="Hide step-by-step output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extra_imports = [
        module.strip() for module in args.authorized_imports.split(",") if module.strip()
    ]
    try:
        model = OpenAIServerModel(
            args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            temperature=args.temperature,
        )
        agent = CodeAgent(
            tools=[],
            model=model,
            max_steps=args.max_steps,
            instructions=args.instructions,
            additional_authorized_imports=extra_imports,
            verbose=not args.quiet,
        )
        output = agent.run(args.task)
    except (AgentError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"\nFinal answer: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
