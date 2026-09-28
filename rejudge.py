#!/usr/bin/env python3
"""
Rejudge — run the LLM judge over an existing run's results.

Judging normally happens inside run.py, but runs recorded without --judge
(or judged under an older judge schema) can be re-annotated here. Judgments
are derived annotations, so a new annotated copy is written: existing
llm_judgment entries are replaced, and the file is stamped with rejudged_at
and the judge identity.

Usage:
    uv run rejudge.py                     # rejudge latest run
    uv run rejudge.py 3                   # rejudge run #3
    uv run rejudge.py smoke-bonsai        # partial filename match
    uv run rejudge.py latest --model ai/gpt-oss:20B
"""

import argparse
import json
import sys
from datetime import datetime

from src.backends import ModelResponse, create_backend
from src.artifacts import artifact_path, atomic_write, source_provenance
import hashlib
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
        "--backend", choices=["docker", "lmstudio", "anthropic"], default="docker",
        help="Backend for the judge (default: docker)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Judge model (default: backend default)",
    )
    parser.add_argument(
        "--judge-votes", type=int, default=1,
        help="Repeated judge votes per response, majority verdict; splits are 'contested' (default: 1)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output",
    )

    parser.add_argument("--base-url", help="OpenAI-compatible API base URL (Docker: UQM_BASE_URL)")
    args = parser.parse_args()
    if args.judge_votes < 1:
        parser.error("judge_votes must be positive")
    if args.base_url and args.backend == "anthropic":
        parser.error("--base-url is for Docker/LM Studio backends")

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
    if args.base_url:
        backend_kwargs["base_url"] = args.base_url
    if args.model:
        backend_kwargs["model"] = args.model
    try:
        judge_backend = create_backend(args.backend, **backend_kwargs)
    except RuntimeError as e:
        print(f"\n  ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)

    judge_batch(judge_backend, results, verbose=not args.quiet, votes=args.judge_votes)

    data["annotation_parent"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    data["annotation_provenance"] = source_provenance()
    data["rejudged_at"] = datetime.now().isoformat()
    data["rejudge_backend"] = judge_backend.name()
    data["summary"] = _build_summary(results)
    output = artifact_path(path.parent, "run", "rejudged")
    atomic_write(output, data)
    print(f"  New annotated copy: {output}; original retained: {path}")
    print()


if __name__ == "__main__":
    main()
