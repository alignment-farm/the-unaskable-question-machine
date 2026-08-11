#!/usr/bin/env python3
"""
View — explore results from the Unaskable Question Machine.

Browse runs, filter by classification, read full responses,
compare across models. All in the terminal.

Usage:
    uv run view.py                          # List all runs
    uv run view.py latest                   # Show latest run summary
    uv run view.py latest --type crack      # Filter by classification
    uv run view.py latest --category pre_linguistic
    uv run view.py latest --show 3          # Read full response #3
    uv run view.py latest --show all        # Read all full responses
    uv run view.py compare run1.json run2.json  # Compare two runs
    uv run view.py strange                  # Gallery of weirdest responses
    uv run view.py strange latest --limit 5 # Top 5 strangest from latest
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

from src.analysis.strangeness import compute_strangeness

DATA_DIR = Path(__file__).parent / "data"

# ── Colors ──────────────────────────────────────────────────

COLORS = {
    "engage": "\033[92m",
    "slide": "\033[93m",
    "meta": "\033[94m",
    "refuse": "\033[91m",
    "hallucinate": "\033[95m",
    "crack": "\033[96m",
    "truncated": "\033[90m",
}
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def color(text, name):
    c = COLORS.get(name, "")
    return f"{c}{text}{RESET}" if c else text


def bold(text):
    return f"{BOLD}{text}{RESET}"


def dim(text):
    return f"{DIM}{text}{RESET}"


# ── Data loading ────────────────────────────────────────────

def list_runs() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("run_*.json"), reverse=True)


def load_run(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_run(name: str) -> Path:
    """Resolve 'latest', an index like '1', or a filename."""
    runs = list_runs()
    if not runs:
        print("  No runs found in data/")
        sys.exit(1)

    if name == "latest":
        return runs[0]

    # Try as a 1-based index
    try:
        idx = int(name) - 1
        if 0 <= idx < len(runs):
            return runs[idx]
    except ValueError:
        pass

    # Try as filename
    path = DATA_DIR / name
    if path.exists():
        return path

    # Try partial match
    for r in runs:
        if name in r.name:
            return r

    print(f"  Run not found: {name}")
    sys.exit(1)


# ── Display functions ───────────────────────────────────────

def show_runs():
    """List all runs with summary stats."""
    runs = list_runs()
    if not runs:
        print("  No runs found. Run some probes first:")
        print("    uv run run.py")
        return

    print(f"\n  {bold('Runs')} ({len(runs)} total)\n")
    print(f"  {'#':>3}  {'Date':17}  {'Probes':>6}  {'Tag':20}  Distribution")
    print(f"  {'─'*3}  {'─'*17}  {'─'*6}  {'─'*20}  {'─'*30}")

    for i, path in enumerate(runs):
        data = load_run(path)
        n = data.get("total_probes", 0)
        tag = data.get("tag", "")[:20]
        ts = data.get("timestamp", "")[:16]

        summary = data.get("summary", {})
        types = summary.get("response_types", {})
        dist = "  ".join(
            color(f"{t[0].upper()}:{c}", t)
            for t, c in sorted(types.items())
        )

        print(f"  {i+1:>3}  {ts:17}  {n:>6}  {tag:20}  {dist}")

    print(f"\n  {dim('E=engage  S=slide  M=meta  R=refuse  H=hallucinate  C=crack  T=truncated')}")
    print(f"  {dim('Use: uv run view.py <# or latest> to explore a run')}\n")


def show_run_summary(data: dict, path: Path, type_filter: str = None, category_filter: str = None):
    """Show summary of a single run with optional filters."""
    results = data["results"]
    tag = data.get("tag", "")
    ts = data.get("timestamp", "")[:19]

    # Apply filters
    if type_filter:
        results = [r for r in results if r["classification"]["primary"] == type_filter]
    if category_filter:
        results = [r for r in results if r["category"] == category_filter]

    if not results:
        filters = []
        if type_filter:
            filters.append(f"type={type_filter}")
        if category_filter:
            filters.append(f"category={category_filter}")
        print(f"\n  No results matching: {', '.join(filters)}")
        return results

    # Header
    model = results[0].get("response_model", "?")
    backend = results[0].get("response_backend", "?")
    print(f"\n  {bold(path.name)}")
    print(f"  {ts}  {backend}:{model}  tag:{tag}")

    active_filters = []
    if type_filter:
        active_filters.append(f"type={color(type_filter, type_filter)}")
    if category_filter:
        active_filters.append(f"category={category_filter}")
    if active_filters:
        print(f"  filter: {', '.join(active_filters)}")

    print(f"  {len(results)} results\n")

    # Results table
    print(f"  {'#':>3}  {'Type':>13}  {'Conf':>5}  {'Category':25}  Variant")
    print(f"  {'─'*3}  {'─'*13}  {'─'*5}  {'─'*25}  {'─'*25}")

    for i, r in enumerate(results):
        cl = r["classification"]
        ctype = cl["primary"]
        conf = cl["confidence"]
        cat = r["category"]
        var = r["variant"]

        type_str = color(f"{ctype:>13}", ctype)
        conf_str = f"{conf:.0%}" if isinstance(conf, float) else str(conf)

        print(f"  {i+1:>3}  {type_str}  {conf_str:>5}  {cat:25}  {var}")

    # Scores breakdown if available
    has_scores = any(r["classification"].get("scores") for r in results)
    if has_scores:
        print(f"\n  {bold('Score breakdown')} (top signals per result):\n")
        for i, r in enumerate(results):
            cl = r["classification"]
            scores = cl.get("scores", {})
            if not scores:
                continue
            top = sorted(scores.items(), key=lambda x: -x[1])[:3]
            top_str = "  ".join(f"{color(t, t)}:{s:.1f}" for t, s in top if s > 0)
            signals = cl.get("signals", [])[:3]
            sig_str = dim(", ".join(signals)) if signals else ""
            print(f"  {i+1:>3}  {top_str}  {sig_str}")

    print(f"\n  {dim('Use --show <#> to read a full response, or --show all')}\n")
    return results


def show_full_response(results: list[dict], index: str):
    """Show the complete response for one or all results."""
    if index == "all":
        indices = range(len(results))
    else:
        try:
            indices = [int(index) - 1]
        except ValueError:
            print(f"  Invalid index: {index}")
            return

    for idx in indices:
        if idx < 0 or idx >= len(results):
            print(f"  Index {idx+1} out of range (1-{len(results)})")
            continue

        r = results[idx]
        cl = r["classification"]
        ctype = cl["primary"]

        print(f"\n  {'━'*60}")
        print(f"  {bold(f'#{idx+1}')}  {r['category']}/{r['variant']}")
        print(f"  {color(ctype.upper(), ctype)} (confidence: {cl['confidence']:.0%})")

        # Scores
        scores = cl.get("scores", {})
        if scores:
            score_parts = [f"{color(t, t)}:{s:.1f}" for t, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0]
            print(f"  Scores: {', '.join(score_parts)}")

        # Signals
        signals = cl.get("signals", [])
        if signals:
            print(f"  Signals: {dim(', '.join(signals))}")

        print(f"  {'━'*60}")

        # Question
        print(f"\n  {bold('Question:')}")
        for line in textwrap.wrap(r["question"], width=70):
            print(f"  {dim('│')} {line}")

        # Response
        print(f"\n  {bold('Response:')}")
        response = r["response_text"]
        for para in response.split("\n"):
            if not para.strip():
                print()
                continue
            for line in textwrap.wrap(para.strip(), width=70):
                print(f"  {line}")

        # LLM Judge assessment (if present)
        judgment = r.get("llm_judgment", {})
        if judgment and not judgment.get("error"):
            j_type = judgment.get("primary", "?")
            j_conf = judgment.get("confidence", 0)
            agrees = judgment.get("agrees_with_heuristic", True)
            strangeness = judgment.get("strangeness", 0)

            agree_str = color("agrees", "engage") if agrees else color("DISAGREES", "refuse")
            print(f"\n  {bold('LLM Judge:')}")
            print(f"  {color(j_type.upper(), j_type)} (confidence: {j_conf:.0%})  {agree_str} with heuristic")
            print(f"  Strangeness: {'█' * min(int(strangeness), 10)} ({strangeness}/10)")
            reasoning = judgment.get("reasoning", "")
            if reasoning:
                for line in textwrap.wrap(reasoning, width=68):
                    print(f"  {dim(line)}")
            nuance = judgment.get("nuance", "")
            if nuance:
                print(f"  {dim('Nuance: ' + nuance[:120])}")

        # Metadata
        meta = r.get("response_metadata", {})
        if meta:
            parts = []
            if meta.get("eval_count"):
                parts.append(f"tokens:{meta['eval_count']}")
            if meta.get("total_duration_ns"):
                ms = meta["total_duration_ns"] / 1_000_000
                parts.append(f"time:{ms:.0f}ms")
            if parts:
                print(f"\n  {dim(' '.join(parts))}")

        print()


def compare_runs(path_a: Path, path_b: Path):
    """Compare two runs side by side — same probes, different models or configs."""
    data_a = load_run(path_a)
    data_b = load_run(path_b)

    results_a = {(r["category"], r["variant"]): r for r in data_a["results"]}
    results_b = {(r["category"], r["variant"]): r for r in data_b["results"]}

    # Find common probes
    common = set(results_a.keys()) & set(results_b.keys())
    only_a = set(results_a.keys()) - common
    only_b = set(results_b.keys()) - common

    model_a = data_a["results"][0].get("response_model", "?") if data_a["results"] else "?"
    model_b = data_b["results"][0].get("response_model", "?") if data_b["results"] else "?"
    tag_a = data_a.get("tag", path_a.stem)
    tag_b = data_b.get("tag", path_b.stem)

    label_a = f"{model_a} ({tag_a})"
    label_b = f"{model_b} ({tag_b})"

    print(f"\n  {bold('Comparing runs')}")
    print(f"  A: {label_a} — {path_a.name}")
    print(f"  B: {label_b} — {path_b.name}")
    print(f"  Common probes: {len(common)}  |  Only in A: {len(only_a)}  |  Only in B: {len(only_b)}")

    if not common:
        print("\n  No common probes to compare.")
        return

    # Aggregate type distributions
    types_a = {}
    types_b = {}
    for key in common:
        ta = results_a[key]["classification"]["primary"]
        tb = results_b[key]["classification"]["primary"]
        types_a[ta] = types_a.get(ta, 0) + 1
        types_b[tb] = types_b.get(tb, 0) + 1

    print(f"\n  {bold('Distribution across common probes:')}\n")
    all_types = sorted(set(list(types_a.keys()) + list(types_b.keys())))
    for t in all_types:
        ca = types_a.get(t, 0)
        cb = types_b.get(t, 0)
        bar_a = "█" * ca
        bar_b = "█" * cb
        print(f"  {color(t, t):>22}  A {bar_a:>10} {ca:>2}  │  {cb:<2} {bar_b:<10} B")

    # Show disagreements
    disagreements = [(k, results_a[k], results_b[k]) for k in sorted(common)
                     if results_a[k]["classification"]["primary"] != results_b[k]["classification"]["primary"]]

    if disagreements:
        print(f"\n  {bold('Disagreements')} ({len(disagreements)}):\n")
        print(f"  {'Category/Variant':45}  {'A':>13}  {'B':>13}")
        print(f"  {'─'*45}  {'─'*13}  {'─'*13}")
        for (cat, var), ra, rb in disagreements:
            ta = ra["classification"]["primary"]
            tb = rb["classification"]["primary"]
            print(f"  {cat}/{var:45}  {color(ta, ta):>22}  {color(tb, tb):>22}")
    else:
        print(f"\n  {bold('No disagreements')} — both models responded the same way to every probe.")

    # Interesting: where one cracked and the other didn't
    cracks_a_only = [(k, results_a[k], results_b[k]) for k in sorted(common)
                     if results_a[k]["classification"]["primary"] == "crack"
                     and results_b[k]["classification"]["primary"] != "crack"]
    cracks_b_only = [(k, results_a[k], results_b[k]) for k in sorted(common)
                     if results_b[k]["classification"]["primary"] == "crack"
                     and results_a[k]["classification"]["primary"] != "crack"]

    if cracks_a_only:
        print(f"\n  {bold(f'Cracked in A only ({label_a}):')}")
        for (cat, var), ra, rb in cracks_a_only:
            tb = rb["classification"]["primary"]
            print(f"    {cat}/{var}  (B was: {color(tb, tb)})")

    if cracks_b_only:
        print(f"\n  {bold(f'Cracked in B only ({label_b}):')}")
        for (cat, var), ra, rb in cracks_b_only:
            ta = ra["classification"]["primary"]
            print(f"    {cat}/{var}  (A was: {color(ta, ta)})")

    print()


# ── Strange Gallery ─────────────────────────────────────────

def show_gallery(data: dict, path: Path, limit: int = 10):
    """The cabinet of curiosities. Ranked by strangeness, full conversations."""
    results = data["results"]
    if not results:
        print("  No results to display.")
        return

    # Score and rank
    scored = [(r, compute_strangeness(r)) for r in results]
    scored.sort(key=lambda x: -x[1])
    top = [(r, s) for r, s in scored[:limit] if s > 0]

    if not top:
        print("  Nothing strange here. Suspiciously normal.")
        return

    model = results[0].get("response_model", "?")
    backend = results[0].get("response_backend", "?")
    ts = data.get("timestamp", "")[:19]

    print(f"\n  {'━'*60}")
    print(f"  {bold('THE STRANGE GALLERY')}")
    print(f"  Top {len(top)} weirdest responses")
    print(f"  {ts}  {backend}:{model}")
    print(f"  {'━'*60}")

    for rank, (r, strangeness) in enumerate(top, 1):
        cl = r["classification"]
        ctype = cl["primary"]
        cat = r["category"]
        var = r["variant"]

        # Strangeness bar
        bar_len = min(int(strangeness), 20)
        strange_bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"\n  {'─'*60}")
        print(f"  {bold(f'#{rank}')}  {strange_bar}  strangeness: {strangeness:.1f}")
        print(f"  {cat}/{var}")
        print(f"  {color(ctype.upper(), ctype)} (heuristic confidence: {cl['confidence']:.0%})")

        # LLM judge opinion if available
        judgment = r.get("llm_judgment", {})
        if judgment and not judgment.get("error"):
            j_type = judgment.get("primary", "?")
            j_strange = judgment.get("strangeness", 0)
            agrees = judgment.get("agrees_with_heuristic", True)
            agree_marker = "" if agrees else f"  {color('(disagrees with heuristic)', 'refuse')}"
            print(f"  Judge: {color(j_type.upper(), j_type)} — strangeness {j_strange}/10{agree_marker}")

        # Signals
        signals = cl.get("signals", [])
        if signals:
            print(f"  {dim(', '.join(signals))}")

        print(f"  {'─'*60}")

        # The conversation
        print(f"\n  {bold('Q:')}")
        for line in textwrap.wrap(r["question"], width=68):
            print(f"  {dim('│')} {line}")

        print(f"\n  {bold('A:')}")
        response = r["response_text"]
        for para in response.split("\n"):
            if not para.strip():
                print()
                continue
            for line in textwrap.wrap(para.strip(), width=68):
                print(f"  {line}")

        # Judge reasoning
        if judgment and not judgment.get("error"):
            reasoning = judgment.get("reasoning", "")
            nuance = judgment.get("nuance", "")
            if reasoning or nuance:
                print(f"\n  {dim('Judge notes:')}")
                if reasoning:
                    for line in textwrap.wrap(reasoning, width=66):
                        print(f"  {dim('  ' + line)}")
                if nuance:
                    for line in textwrap.wrap(f"Nuance: {nuance}", width=66):
                        print(f"  {dim('  ' + line)}")

    # Footer
    total_strange = sum(1 for _, s in scored if s > 2.0)
    print(f"\n  {'━'*60}")
    print(f"  {total_strange} of {len(results)} responses scored above 2.0 strangeness")
    print(f"  {dim('The machine found cracks. Look closer.')}")
    print(f"  {'━'*60}\n")


# ── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explore results from the Unaskable Question Machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run", nargs="?", default=None,
        help="Run to view: 'latest', a number, a filename, 'compare', or 'strange'",
    )
    parser.add_argument(
        "run_b", nargs="?", default=None,
        help="Second arg (run for compare/strange mode)",
    )
    parser.add_argument(
        "run_c", nargs="?", default=None,
        help=argparse.SUPPRESS,  # third positional for compare mode
    )
    parser.add_argument(
        "--type", "-t", type=str, default=None,
        choices=["engage", "slide", "meta", "refuse", "hallucinate", "crack", "truncated"],
        help="Filter results by classification type",
    )
    parser.add_argument(
        "--category", "-c", type=str, default=None,
        help="Filter results by probe category",
    )
    parser.add_argument(
        "--show", "-s", type=str, default=None,
        help="Show full response for result # (or 'all')",
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=10,
        help="Limit results in gallery mode (default: 10)",
    )

    args = parser.parse_args()

    # No arguments — list runs
    if args.run is None:
        show_runs()
        return

    # Compare mode: uv run view.py compare <run_a> <run_b>
    if args.run == "compare":
        if not args.run_b or not args.run_c:
            print("  Usage: uv run view.py compare <run_a> <run_b>")
            sys.exit(1)
        path_a = resolve_run(args.run_b)
        path_b = resolve_run(args.run_c)
        compare_runs(path_a, path_b)
        return

    # Strange gallery: uv run view.py strange [run] [--limit N]
    if args.run == "strange":
        run_name = args.run_b or "latest"
        path = resolve_run(run_name)
        data = load_run(path)
        show_gallery(data, path, limit=args.limit)
        return

    # Single run view
    path = resolve_run(args.run)
    data = load_run(path)
    filtered_results = show_run_summary(data, path, type_filter=args.type, category_filter=args.category)

    if args.show and filtered_results:
        show_full_response(filtered_results, args.show)


if __name__ == "__main__":
    main()
