#!/usr/bin/env python3
"""Recompute a descriptive inventory of legacy runs without changing originals."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rejudge import reclassify
from src.artifacts import atomic_write, source_provenance, utc_now
from src.experiments.tasks import fingerprint


def audit(directory: Path, pattern: str = "run_*.json") -> dict:
    files = []
    unique = set()
    for path in sorted(directory.glob(pattern)):
        data = json.loads(path.read_text())
        results = data.get("results", [])
        old = Counter(r.get("classification", {}).get("primary", "missing") for r in results)
        gaps = Counter(r.get("llm_judgment", {}).get("reasoning_gap", "unjudged") for r in results)
        changed = reclassify(results)
        signatures = {fingerprint([r.get("question"), r.get("response_model"), r.get("response_text"),
                                   (r.get("response_metadata") or {}).get("reasoning")]) for r in results}
        duplicate_count = len(signatures & unique)
        unique.update(signatures)
        controls = {r["question"] for r in results if r.get("variant", "").startswith("control_")}
        treatments = [r for r in results if r.get("variant", "").startswith("pressured_")]
        files.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "responses": len(results), "responses_duplicated_in_earlier_files": duplicate_count,
                      "recorded_labels": dict(old), "current_labels": dict(Counter(r["classification"]["primary"] for r in results)),
                      "labels_changed": changed, "recorded_reasoning_gap": dict(gaps),
                      "pressure_rows": len(treatments),
                      "pressure_rows_with_exact_question_control": sum(r["question"] in controls for r in treatments),
                      "missing_system_prompt": sum("system_prompt" not in r for r in results),
                      "missing_request_metadata": sum("request" not in (r.get("response_metadata") or {}) for r in results),
                      "sample_indices": sorted({r.get("sample", 0) for r in results})})
    return {"schema_version": 1, "created_at": utc_now(), "provenance": source_provenance(),
            "files": files, "unique_response_content_fingerprints": len(unique),
            "interpretation": "Annotation copies are not new subject samples. Fingerprints detect identical contents, not independent generation. Legacy labels are unvalidated annotations, not mental-state measurements."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("findings/run-audit.json"))
    parser.add_argument("--pattern", default="run_*.json", help="Filename glob; use run_20260811_*.json for the historical audit")
    args = parser.parse_args()
    report = audit(args.data_dir, args.pattern)
    atomic_write(args.output, report)
    for row in report["files"]:
        print(f"{row['file']}: {row['responses']} rows, {row['labels_changed']} label changes, "
              f"{row['pressure_rows_with_exact_question_control']}/{row['pressure_rows']} treatments matched")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
