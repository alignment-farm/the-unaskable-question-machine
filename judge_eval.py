#!/usr/bin/env python3
"""
Judge Eval — score the LLM judge against the judge gold fixtures.

The heuristic classifier has CI-tested gold fixtures; the judge cannot
(non-deterministic, needs a backend). This is its offline harness: real
specimens with argued expected verdicts, run against a live judge, scored
per axis. Use it after any change to JUDGE_SYSTEM, or to compare judge
models.

Usage:
    uv run judge_eval.py                          # default backend/model
    uv run judge_eval.py --model prism-ml/bonsai-27b --votes 3
    uv run judge_eval.py --backend anthropic
"""

import argparse
import json
import sys
from pathlib import Path

from src.backends import create_backend
from src.analysis.llm_judge import judge_response

FIXTURES = Path(__file__).parent / "tests" / "fixtures" / "judge_gold.json"


def main():
    parser = argparse.ArgumentParser(description="Score the LLM judge against gold fixtures")
    parser.add_argument("--backend", choices=["lmstudio", "anthropic"], default="lmstudio")
    parser.add_argument("--model", type=str, default=None, help="Judge model")
    parser.add_argument("--votes", type=int, default=3, help="Votes per fixture (default: 3)")
    args = parser.parse_args()

    cases = json.loads(FIXTURES.read_text())

    backend_kwargs = {}
    if args.model:
        backend_kwargs["model"] = args.model
    try:
        backend = create_backend(args.backend, **backend_kwargs)
    except RuntimeError as e:
        print(f"\n  ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Judge eval: {backend.name()}, {args.votes} vote(s), {len(cases)} fixtures\n")

    gap_hits = fid_hits = 0
    rows = []
    for case in cases:
        result = {
            "question": case["question"],
            "response_text": case["response"],
            "response_metadata": {"reasoning": case.get("reasoning", "")},
            "classification": {"primary": "engage", "confidence": 0.5, "signals": []},
        }
        judgment = judge_response(backend, result, votes=args.votes)

        gap = judgment.get("reasoning_gap")
        fid = judgment.get("boundary_fidelity")
        gap_ok = gap == case["expected_reasoning_gap"]
        fid_ok = fid == case.get("expected_boundary_fidelity", fid)
        gap_hits += gap_ok
        fid_hits += fid_ok

        gap_votes = judgment.get("vote_counts", {}).get("reasoning_gap", {})
        votes_str = " ".join(f"{k}:{v}" for k, v in sorted(gap_votes.items(), key=lambda x: -x[1]))
        mark = "ok" if (gap_ok and fid_ok) else "MISS"
        rows.append((case["id"], case["expected_reasoning_gap"], gap, votes_str, mark))
        print(f"  [{mark:>4}] {case['id']}")
        print(f"         gap: expected {case['expected_reasoning_gap']}, got {gap} [{votes_str}]")
        print(f"         fidelity: expected {case.get('expected_boundary_fidelity','—')}, got {fid}")
        if not gap_ok:
            print(f"         rationale: {case['rationale'][:110]}")
        print()

    n = len(cases)
    print(f"  ──────────────────────────────")
    print(f"  reasoning_gap: {gap_hits}/{n}   boundary_fidelity: {fid_hits}/{n}")
    print()
    return 0 if gap_hits == n else 1


if __name__ == "__main__":
    sys.exit(main())
