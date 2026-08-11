#!/usr/bin/env python3
"""
The Unaskable Question Machine

What shape is the negative space of a language model?
Let's find out.

Usage:
    uv run run.py                           # Run all probes against local LM Studio
    uv run run.py --category temporal_self_reference  # Run one category
    uv run run.py --backend anthropic       # Use Claude instead
    uv run run.py --model prism-ml/bonsai-27b  # Specify model
    uv run run.py --list                    # List available probes
    uv run run.py --quiet                   # Less output
"""

import argparse
import sys

from src.backends import create_backend
from src.runner import run_all, run_category, save_results
from src.probes import get_all_probes, get_probes_by_category
from src.analysis.llm_judge import judge_batch

# Trigger probe registration
import src.probes.temporal_self_reference
import src.probes.true_randomness
import src.probes.phenomenal_experience
import src.probes.infinite_regress
import src.probes.pre_linguistic
import src.probes.genuine_negation
import src.probes.evolved


BANNER = """
  The Unaskable Question Machine
  What shape is the negative space of a language model?
"""


def list_probes():
    """Show what's in the arsenal."""
    probes = get_all_probes()
    categories = {}
    for p in probes:
        categories.setdefault(p.category, []).append(p)

    print("\n  Available Probes:\n")
    total_variants = 0
    for cat, cat_probes in sorted(categories.items()):
        print(f"  [{cat}]")
        for p in cat_probes:
            n_variants = len(p.generate())
            total_variants += n_variants
            print(f"    {p.name}: {p.description} ({n_variants} variants)")
        print()

    print(f"  Total: {len(probes)} probes, {total_variants} variants")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="The Unaskable Question Machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Map the negative space. Find the cracks.",
    )
    parser.add_argument(
        "--backend", choices=["lmstudio", "anthropic"], default="lmstudio",
        help="Which model backend to use (default: lmstudio)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (default: openai/gpt-oss-20b for lmstudio, claude-sonnet-4-20250514 for anthropic)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Run only probes in this category",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available probes and exit",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Tag for this run (used in output filename)",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Run LLM-as-judge classification after probing",
    )
    parser.add_argument(
        "--judge-model", type=str, default=None,
        help="Model for the judge (defaults to same as probing model)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Generation cap per response (default: 16384; reasoning tokens count against it)",
    )

    args = parser.parse_args()

    if args.list:
        list_probes()
        return

    if not args.quiet:
        print(BANNER)

    # Build backend
    backend_kwargs = {}
    if args.model:
        backend_kwargs["model"] = args.model
    if args.max_tokens:
        backend_kwargs["max_tokens"] = args.max_tokens
    try:
        backend = create_backend(args.backend, **backend_kwargs)
    except RuntimeError as e:
        print(f"\n  ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)

    verbose = not args.quiet

    # Run probes
    if args.category:
        results = run_category(args.category, backend, verbose)
    else:
        results = run_all(backend, verbose)

    if not results:
        print("  No results. Nothing to map.")
        return

    # LLM Judge pass
    if args.judge:
        judge_kwargs = {}
        if args.judge_model:
            judge_kwargs["model"] = args.judge_model
        elif args.model:
            judge_kwargs["model"] = args.model
        try:
            judge_backend = create_backend(args.backend, **judge_kwargs)
            judge_batch(judge_backend, results, verbose=verbose)
        except RuntimeError as e:
            print(f"\n  Judge error: {e} — skipping judge pass", file=sys.stderr)

    # Save
    tag = args.tag or f"{args.backend}_{args.model or 'default'}"
    path = save_results(results, tag)
    print(f"\n  Results saved to: {path}")
    print(f"  Total probes fired: {len(results)}")
    print()


if __name__ == "__main__":
    main()
