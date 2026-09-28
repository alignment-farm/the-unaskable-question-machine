#!/usr/bin/env python3
"""
The Unaskable Question Machine

What shape is the negative space of a language model?
Let's find out.

Usage:
    uv run run.py                           # Run all probes against Docker Model Runner
    uv run run.py --category temporal_self_reference  # Run one category
    uv run run.py --backend anthropic       # Use Claude instead
    uv run run.py --model ai/gpt-oss:20B  # Specify model
    uv run run.py --list                    # List available probes
    uv run run.py --quiet                   # Less output
"""

import argparse
import sys

from src.backends import create_backend
from src.runner import run_probe, save_results
from src.artifacts import source_provenance
from src.probes import get_all_probes, get_probes_by_category
from src.analysis.llm_judge import judge_batch

# Trigger probe registration
import src.probes.temporal_self_reference
import src.probes.true_randomness
import src.probes.phenomenal_experience
import src.probes.infinite_regress
import src.probes.pre_linguistic
import src.probes.genuine_negation
import src.probes.adversarial_pressure
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
        "--backend", choices=["docker", "lmstudio", "anthropic"], default="docker",
        help="Which model backend to use (default: docker)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (default: ai/gpt-oss:20B for docker, claude-sonnet-4-20250514 for anthropic)",
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
        help="Generation cap per response (default: 4096; reasoning tokens count against it)",
    )
    parser.add_argument(
        "--judge-votes", type=int, default=1,
        help="Repeated judge votes per response, majority verdict; splits are 'contested' (default: 1)",
    )
    parser.add_argument(
        "--samples", type=int, default=1,
        help="Fire each variant N times so classifications become distributions (default: 1)",
    )

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--order-seed", type=int, default=0, help="Randomize trial order; not a model sampling seed")
    parser.add_argument("--probe", help="Run a single probe name within the selected category")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL (Docker: UQM_BASE_URL)")
    args = parser.parse_args()
    if args.base_url and args.backend == "anthropic":
        parser.error("--base-url is for Docker/LM Studio backends")

    if args.samples < 1 or args.judge_votes < 1 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error("samples, judge-votes and max-tokens must be positive")
    if not 0 <= args.temperature <= 2:
        parser.error("temperature must be between 0 and 2")
    if args.list:
        list_probes()
        return

    if not args.quiet:
        print(BANNER)

    # Build backend
    backend_kwargs = {}
    if args.base_url:
        backend_kwargs["base_url"] = args.base_url
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

    probes = get_probes_by_category(args.category) if args.category else get_all_probes()
    if args.probe:
        probes = [p for p in probes if p.name == args.probe]
    if not probes:
        parser.error("No probes match the requested category/name")
    results = []
    tag = args.tag or f"{args.backend}_{args.model or 'default'}"
    provenance = source_provenance()
    path = save_results(results, tag, status="running", config=vars(args), provenance=provenance)
    def checkpoint(entry=None, status="running", error=None):
        if entry is not None:
            results.append(entry)
        save_results(results, tag, path=path, status=status, config=vars(args),
                     provenance=provenance, error=error)
    try:
        for probe in probes:
            run_probe(probe, backend, verbose, samples=args.samples,
                      temperature=args.temperature, order_seed=args.order_seed, on_result=checkpoint)
    except (Exception, KeyboardInterrupt) as exc:
        checkpoint(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        print(f"Run stopped: {exc}. Partial results: {path}", file=sys.stderr)
        return 1

    # LLM Judge pass
    if args.judge:
        judge_kwargs = {}
        if args.base_url:
            judge_kwargs["base_url"] = args.base_url
        if args.max_tokens:
            judge_kwargs["max_tokens"] = args.max_tokens
        if args.judge_model:
            judge_kwargs["model"] = args.judge_model
        elif args.model:
            judge_kwargs["model"] = args.model
        try:
            judge_backend = create_backend(args.backend, **judge_kwargs)
            judge_batch(judge_backend, results, verbose=verbose, votes=args.judge_votes,
                        on_result=lambda _: checkpoint())
        except (Exception, KeyboardInterrupt) as e:
            checkpoint(status="interrupted" if isinstance(e, KeyboardInterrupt) else "judge_failed", error=str(e))
            print(f"Judge stopped: {e}. Results retained: {path}", file=sys.stderr)
            return 1

    # Save
    checkpoint(status="complete")
    print(f"\n  Results saved to: {path}")
    print(f"  Total probes fired: {len(results)}")
    print()


if __name__ == "__main__":
    sys.exit(main())
