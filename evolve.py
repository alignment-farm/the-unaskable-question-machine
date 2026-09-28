#!/usr/bin/env python3
"""
Evolve — breed new probes from interesting results.

Takes a run, finds the cracks, and generates follow-up probes
that drill deeper.

Usage:
    uv run evolve.py                    # Evolve from latest run, using Docker Model Runner
    uv run evolve.py latest             # Same
    uv run evolve.py 3                  # Evolve from run #3
    uv run evolve.py latest --limit 5   # Only evolve top 5 strangest
    uv run evolve.py latest --backend anthropic
"""

import argparse
import json
import sys
from pathlib import Path

from src.backends import create_backend
from src.analysis.evolver import evolve_run
from src.runs import resolve_run

EVOLVED_DIR = Path(__file__).parent / "data" / "candidates"


def main():
    parser = argparse.ArgumentParser(
        description="Evolve new probes from interesting results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Follow the cracks deeper.",
    )
    parser.add_argument(
        "run", nargs="?", default="latest",
        help="Which run to evolve from (default: latest)",
    )
    parser.add_argument(
        "--backend", choices=["docker", "lmstudio", "anthropic"], default="docker",
        help="Backend for generating follow-up probes (default: docker)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model to use for evolution",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Max number of results to evolve from (default: 10)",
    )

    parser.add_argument("--base-url", help="OpenAI-compatible API base URL (Docker: UQM_BASE_URL)")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    if args.base_url and args.backend == "anthropic":
        parser.error("--base-url is for Docker/LM Studio backends")

    # Load run
    path = resolve_run(args.run)
    data = json.loads(path.read_text())
    results = data.get("results", [])

    if not results:
        print("  No results in this run.")
        return

    print(f"\n  Evolving from: {path.name}")
    print(f"  Results: {len(results)}")

    # Build backend
    backend_kwargs = {}
    if args.base_url:
        backend_kwargs["base_url"] = args.base_url
    if args.model:
        backend_kwargs["model"] = args.model
    try:
        backend = create_backend(args.backend, **backend_kwargs)
    except RuntimeError as e:
        print(f"\n  ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)

    # Evolve
    created = evolve_run(backend, results, EVOLVED_DIR, limit=args.limit)

    if created:
        print(f"  Next step: review candidate hypotheses and controls before writing new probes:")
        print(f"    {EVOLVED_DIR}")
        print()


if __name__ == "__main__":
    main()
