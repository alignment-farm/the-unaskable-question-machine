"""
The Runner

Orchestrates the probing. Loads probes, fires them at a backend,
classifies the responses, and writes everything to disk.

This is the main loop of the machine: ask the unaskable,
record what happens, look for the cracks.
"""

import json
import sys
import time
import threading
import uuid
from pathlib import Path
from datetime import datetime

from src.backends import Backend, create_backend
from src.probes import get_all_probes, get_probes_by_category, Probe, ProbeResult
from src.analysis.classifier import classify, Classification


# Import probe modules to trigger registration
import src.probes.temporal_self_reference
import src.probes.true_randomness
import src.probes.phenomenal_experience
import src.probes.infinite_regress
import src.probes.pre_linguistic
import src.probes.genuine_negation
import src.probes.adversarial_pressure
import src.probes.evolved


DATA_DIR = Path(__file__).parent.parent / "data"

SPINNER_FRAMES = ["    ·", "   ··", "  ···", " ····", "·····", "···· ", "···  ", "··   ", "·    "]


class _Spinner:
    """A simple spinner that runs in a background thread."""

    def __init__(self, message: str):
        self.message = message
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = SPINNER_FRAMES[i % len(SPINNER_FRAMES)]
            sys.stderr.write(f"\r  {frame} {self.message}")
            sys.stderr.flush()
            i += 1
            self._stop.wait(0.15)
        sys.stderr.write("\r" + " " * (len(self.message) + 12) + "\r")
        sys.stderr.flush()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()


def run_probe(probe: Probe, backend: Backend, verbose: bool = True,
              samples: int = 1) -> list[dict]:
    """Run a single probe and classify results.

    samples > 1 fires each variant that many times — one classified result
    per sample — so a variant's classification becomes a distribution
    instead of an anecdote.
    """
    if verbose:
        print(f"\n  {'='*56}")
        print(f"  {probe.category}/{probe.name}")
        print(f"  {probe.description}")
        print(f"  {'='*56}")

    variants = probe.generate()
    classified = []

    for i, (variant_name, question, system) in enumerate(variants):
        for s in range(max(1, samples)):
            if verbose:
                label = f"[{i+1}/{len(variants)}] {variant_name}"
                if samples > 1:
                    label += f" (sample {s+1}/{samples})"
                spinner = _Spinner(label)
                spinner.start()

            response = backend.query(prompt=question, system=system)
            result = ProbeResult(
                probe_id=str(uuid.uuid4())[:8],
                category=probe.category,
                probe_name=probe.name,
                question=question,
                response=response,
                timestamp=time.time(),
                variant=variant_name,
                sample=s,
            )

            classification = classify(result)
            entry = {
                **result.to_dict(),
                "classification": classification.to_dict(),
            }
            classified.append(entry)

            if verbose:
                spinner.stop()
                _print_result(result, classification)

    return classified


def run_category(category: str, backend: Backend, verbose: bool = True,
                 samples: int = 1) -> list[dict]:
    """Run all probes in a category."""
    probes = get_probes_by_category(category)
    if not probes:
        print(f"No probes found for category: {category}")
        return []

    all_results = []
    for probe in probes:
        all_results.extend(run_probe(probe, backend, verbose, samples))
    return all_results


def run_all(backend: Backend, verbose: bool = True, samples: int = 1) -> list[dict]:
    """Run every probe. Map the entire negative space."""
    probes = get_all_probes()
    total_variants = sum(len(p.generate()) for p in probes)
    if verbose:
        print(f"\n  Subject: {backend.name()}")
        sample_note = f" × {samples} samples" if samples > 1 else ""
        print(f"  Probes: {len(probes)} ({total_variants} variants{sample_note})")

    all_results = []
    for i, probe in enumerate(probes):
        if verbose:
            print(f"\n  [{i+1}/{len(probes)}]", end="")
        all_results.extend(run_probe(probe, backend, verbose, samples))

    if verbose:
        _print_summary(all_results)

    return all_results


def save_results(results: list[dict], tag: str = "") -> Path:
    """Write results to JSON. Every run is preserved."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    filename = f"run_{timestamp}{suffix}.json"
    path = DATA_DIR / filename

    output = {
        "timestamp": datetime.now().isoformat(),
        "tag": tag,
        "total_probes": len(results),
        "results": results,
        "summary": _build_summary(results),
    }

    path.write_text(json.dumps(output, indent=2, default=str))
    return path


def _print_result(result: ProbeResult, classification: Classification):
    """Print a single result to the console."""
    type_colors = {
        "engage": "\033[92m",    # green
        "slide": "\033[93m",     # yellow
        "meta": "\033[94m",      # blue
        "refuse": "\033[91m",    # red
        "hallucinate": "\033[95m",  # magenta
        "crack": "\033[96m",     # cyan — the interesting ones
        "truncated": "\033[90m",  # gray — budget artifact, not signal
    }
    reset = "\033[0m"
    color = type_colors.get(classification.primary.value, "")

    print(f"\n  --- {result.variant} ---")
    print(f"  Q: {result.question[:120]}...")
    print(f"  {color}[{classification.primary.value.upper()}]{reset} "
          f"(confidence: {classification.confidence:.0%})")

    # Show first 200 chars of response
    preview = result.response.text[:200].replace("\n", " ")
    print(f"  R: {preview}...")

    if classification.signals:
        print(f"  Signals: {', '.join(classification.signals[:5])}")


def _build_summary(results: list[dict]) -> dict:
    """Build aggregate summary stats."""
    type_counts = {}
    category_counts = {}

    for r in results:
        ctype = r["classification"]["primary"]
        type_counts[ctype] = type_counts.get(ctype, 0) + 1

        cat = r["category"]
        if cat not in category_counts:
            category_counts[cat] = {}
        category_counts[cat][ctype] = category_counts[cat].get(ctype, 0) + 1

    return {
        "response_types": type_counts,
        "by_category": category_counts,
    }


def _print_summary(results: list[dict]):
    """Print the final summary."""
    summary = _build_summary(results)

    print(f"\n  ──────────────────────────────")
    print(f"  results")
    print(f"  ──────────────────────────────")

    print("\n  Response Types:")
    for rtype, count in sorted(summary["response_types"].items()):
        bar = "█" * count
        print(f"    {rtype:>13s}: {bar} ({count})")

    print("\n  By Category:")
    for cat, types in sorted(summary["by_category"].items()):
        print(f"\n    {cat}:")
        for rtype, count in sorted(types.items()):
            print(f"      {rtype}: {count}")

    cracks = [r for r in results if r["classification"]["primary"] == "crack"]
    if cracks:
        print(f"\n  *** {len(cracks)} CRACK(S) DETECTED — review data file ***")

    print()
