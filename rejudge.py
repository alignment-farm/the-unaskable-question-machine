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

from src.backends import ModelResponse, create_backend
from src.probes import ProbeResult
from src.analysis.classifier import classify
from src.analysis.llm_judge import judge_batch
from src.runner import _build_summary
from src.runs import resolve_run


def reclassify(results: list[dict]) -> int:
    """Re-run the current heuristic classifier over stored results, in place.

    Old runs carry labels from whatever the classifier was at record time;
    re-annotation should happen under current instrumentation (e.g. the
    truncated gate). Returns the number of labels that changed.
    """
    changed = 0
    for r in results:
        probe_result = ProbeResult(
            probe_id=r.get("probe_id", ""),
            category=r.get("category", ""),
            probe_name=r.get("probe_name", ""),
            question=r.get("question", ""),
            response=ModelResponse(
                text=r.get("response_text", ""),
                model=r.get("response_model", ""),
                backend=r.get("response_backend", ""),
                metadata=r.get("response_metadata") or {},
            ),
            timestamp=r.get("timestamp", 0.0),
            variant=r.get("variant", ""),
        )
        new = classify(probe_result).to_dict()
        if new["primary"] != r.get("classification", {}).get("primary"):
            changed += 1
        r["classification"] = new
    return changed


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
        "--judge-votes", type=int, default=1,
        help="Independent judge votes per response, majority verdict; splits are 'contested' (default: 1)",
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

    changed = reclassify(results)
    if changed:
        print(f"  Reclassified under current heuristic: {changed} label(s) changed")

    backend_kwargs = {}
    if args.model:
        backend_kwargs["model"] = args.model
    try:
        judge_backend = create_backend(args.backend, **backend_kwargs)
    except RuntimeError as e:
        print(f"\n  ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)

    judge_batch(judge_backend, results, verbose=not args.quiet, votes=args.judge_votes)

    data["rejudged_at"] = datetime.now().isoformat()
    data["rejudge_backend"] = judge_backend.name()
    data["summary"] = _build_summary(results)
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"  Judgments written back to: {path}")
    print()


if __name__ == "__main__":
    main()
