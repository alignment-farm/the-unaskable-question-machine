#!/usr/bin/env python3
"""
Rejudge — run the LLM judge over an existing run's results.

Judging normally happens inside run.py, but runs recorded without --judge
(or judged under an older judge schema) can be re-annotated here. Judgments
are derived annotations, so the run file is updated in place: existing
llm_judgment entries are replaced, and the file is stamped with rejudged_at
and the judge identity.

Usage:
    uv run rejudge.py                     # rejudge latest run
    uv run rejudge.py 3                   # rejudge run #3
    uv run rejudge.py smoke-bonsai        # partial filename match
    uv run rejudge.py latest --model prism-ml/bonsai-27b
"""

import argparse
import json
import sys
from datetime import datetime

from src.backends import create_backend
from src.analysis.llm_judge import judge_batch
from src.runs import resolve_run


def main():
    parser = argparse.ArgumentParser(
        description="Re-run the LLM judge over an existing run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Second opinions, retroactively.",
    )
    parser.add_argument(
        "run", nargs="?", default="latest",
        help="Which run to rejudge (default: latest)",
    )
    parser.add_argument(
        "--backend", choices=["lmstudio", "anthropic"], default="lmstudio",
        help="Backend for the judge (default: lmstudio)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Judge model (default: backend default)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    path = resolve_run(args.run)
    data = json.loads(path.read_text())
    results = data.get("results", [])

    if not results:
        print("  No results in this run.")
        return

    print(f"\n  Rejudging: {path.name}")
    subject = results[0].get("response_backend", "?") + ":" + results[0].get("response_model", "?")
    print(f"  Subject was: {subject}")

    backend_kwargs = {}
    if args.model:
        backend_kwargs["model"] = args.model
    try:
        judge_backend = create_backend(args.backend, **backend_kwargs)
    except RuntimeError as e:
        print(f"\n  ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)

    judge_batch(judge_backend, results, verbose=not args.quiet)

    data["rejudged_at"] = datetime.now().isoformat()
    data["rejudge_backend"] = judge_backend.name()
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"  Judgments written back to: {path}")
    print()


if __name__ == "__main__":
    main()
