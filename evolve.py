#!/usr/bin/env python3
"""
Evolve — breed new probes from interesting results.

Takes a run, finds the cracks, and generates follow-up probes
that drill deeper.

Usage:
    uv run evolve.py                    # Evolve from latest run, using LM Studio
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

DATA_DIR = Path(__file__).parent / "data"
EVOLVED_DIR = Path(__file__).parent / "src" / "probes" / "evolved"


def list_runs() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("run_*.json"), reverse=True)


def resolve_run(name: str) -> Path:
    runs = list_runs()
    if not runs:
        print("  No runs found in data/")
        sys.exit(1)

    if name == "latest":
        return runs[0]

    try:
        idx = int(name) - 1
        if 0 <= idx < len(runs):
            return runs[idx]
    except ValueError:
        pass

    path = DATA_DIR / name
    if path.exists():
        return path

    for r in runs:
        if name in r.name:
            return r

    print(f"  Run not found: {name}")
    sys.exit(1)


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
        "--backend", choices=["lmstudio", "anthropic"], default="lmstudio",
        help="Backend for generating follow-up probes (default: lmstudio)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model to use for evolution",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Max number of results to evolve from (default: 10)",
    )

    args = parser.parse_args()

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
        print(f"  Next step: run the evolved probes:")
        print(f"    uv run run.py")
        print()


if __name__ == "__main__":
    main()
