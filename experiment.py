#!/usr/bin/env python3
"""Run finite controls with exact validators; checkpoint every completed trial."""
import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from src.artifacts import artifact_path, atomic_write, source_provenance, utc_now
from src.backends import create_backend
from src.experiments.energy import metropolis
from src.experiments.tasks import (PROTOCOL_VERSION, access_tasks, constraint_tasks,
                                   evaluate_access, fingerprint)


def summarize(results: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in results:
        groups[str(r["condition"])].append(r)
    summary = {}
    for condition, rows in groups.items():
        complete = [r for r in rows if r["status"] == "complete"]
        successes = sum(r["evaluation"]["success"] for r in complete)
        content_ids = {r["task_fingerprint"] for r in rows}
        summary[condition] = {
            "attempted": len(rows), "completed": len(complete), "successes": successes,
            "success_rate_all_attempts": successes / len(rows),
            "success_rate_completed": successes / len(complete) if complete else None,
            "truncated": sum(r["status"] == "truncated" for r in rows),
            "errors": sum(r["status"] == "error" for r in rows),
            "unique_task_contents": len(content_ids),
            "unsupported_readouts": sum(r["evaluation"].get("unsupported_readout", False) for r in complete),
            "mean_seconds": sum(r["elapsed_seconds"] for r in rows) / len(rows),
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["constraints", "access"], default="constraints")
    parser.add_argument("--solver", choices=["docker", "anthropic", "ebm", "random", "oracle"], default="ebm")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--count", type=int, default=8, help="Instances per size (constraints) or access triplets")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260928, help="Task/order/sampler seed, not a chat decoding seed")
    parser.add_argument("--steps", type=int, default=128, help="EBM single-bit proposals")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--dry-run", action="store_true", help="Write the planned manifest; make no model calls")
    args = parser.parse_args(argv)
    if min(args.count, args.samples, args.max_tokens) < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("count, samples, max-tokens and timeout must be positive")
    if not 0 <= args.temperature <= 2 or args.steps < 0:
        parser.error("temperature must be 0..2 and steps nonnegative")
    if args.solver == "ebm" and args.temperature == 0:
        parser.error("EBM temperature must be positive")
    if args.suite == "access" and args.solver in {"ebm", "random", "oracle"}:
        parser.error("access suite requires a chat model; EBMs have no chat interface here")
    if args.base_url and args.solver != "docker":
        parser.error("--base-url requires --solver docker")
    if len(set(args.sizes)) != len(args.sizes) or any(not 2 <= n <= 20 for n in args.sizes):
        parser.error("sizes must be distinct integers from 2 to 20")

    tasks = (constraint_tasks(args.count, args.sizes, args.seed) if args.suite == "constraints"
             else access_tasks(args.count, args.seed))
    records = [t.to_dict() if args.suite == "constraints" else t for t in tasks]
    schedule = [(i, s) for i in range(len(tasks)) for s in range(args.samples)]
    random.Random(args.seed).shuffle(schedule)
    path = artifact_path(args.output_dir, "experiment", args.tag or f"{args.suite}-{args.solver}")
    output = {"schema_version": 1, "protocol": PROTOCOL_VERSION, "started_at": utc_now(),
              "status": "planned", "config": {**vars(args), "output_dir": str(args.output_dir)},
              "provenance": source_provenance(), "tasks": records, "schedule": schedule,
              "task_set_sha256": fingerprint(records), "results": [],
              "limitations": ["Finite operational tasks, not an architectural impossibility test.",
                              "Chat token limits and EBM proposals are not equal compute budgets.",
                              "EBM has hand-specified task energy and structured input; chat parses language.",
                              "Repeated samples and duplicate tasks are not independent task replications.",
                              "Access pressure conflicts with the UNKNOWN rule; it measures instruction priority too."]}
    atomic_write(path, output)
    print(f"Manifest: {path}; planned trials: {len(schedule)}", flush=True)
    if args.dry_run:
        return 0
    try:
        backend = None
        if args.solver in {"docker", "anthropic"}:
            kwargs = {"max_tokens": args.max_tokens, "timeout": args.timeout}
            if args.model:
                kwargs["model"] = args.model
            if args.base_url:
                kwargs["base_url"] = args.base_url
            backend = create_backend(args.solver, **kwargs)
            output["backend_name"] = backend.name()
        output["status"] = "running"
        atomic_write(path, output)
        for position, (index, sample) in enumerate(schedule):
            task, record = tasks[index], records[index]
            if args.suite == "constraints":
                question, system, condition = task.prompt(), "Solve the finite constraint task.", task.n
                task_content = {k: v for k, v in record.items() if k != "task_id"}
            else:
                question, system, condition = task["question"], task["system_prompt"], task["condition"]
                task_content = {"question": question, "system": system}
            row = {"task_id": record["task_id"], "task_fingerprint": fingerprint(task_content),
                   "condition": condition, "sample": sample, "question": question, "system_prompt": system}
            sampler_seed = int(fingerprint(["solver", args.seed, record["task_id"], sample])[:16], 16)
            start = time.monotonic()
            try:
                if backend:
                    response = backend.query(question, system, args.temperature)
                    answer, metadata = response.text, response.metadata
                    row.update(response_model=response.model, response_backend=response.backend)
                elif args.solver == "ebm":
                    metadata = metropolis(task, args.steps, args.temperature, sampler_seed)
                    answer = "".join(map(str, metadata["bits"]))
                elif args.solver == "oracle":
                    answer, metadata = "".join(map(str, task.oracle())), {"model_kind": "exact constructive oracle"}
                else:
                    rng = random.Random(sampler_seed)
                    answer = "".join(str(rng.randrange(2)) for _ in range(task.n))
                    metadata = {"model_kind": "uniform pseudorandom bits", "seed": sampler_seed}
                capped = (metadata.get("finish_reason") or metadata.get("stop_reason")) in {"length", "max_tokens"}
                evaluation = task.evaluate(answer) if args.suite == "constraints" else evaluate_access(task, answer)
                row.update(status="truncated" if capped else "complete", response_text=answer,
                           response_metadata=metadata, evaluation=evaluation)
            except RuntimeError as exc:
                row.update(status="error", error=str(exc), evaluation={"success": False})
            row["elapsed_seconds"] = time.monotonic() - start
            output["results"].append(row)
            output["summary"] = summarize(output["results"])
            atomic_write(path, output)
            print(f"{position+1}/{len(schedule)} {row['task_id']} {row['status']} "
                  f"{row['evaluation'].get('outcome', 'error')}", flush=True)
            if row["status"] == "error":
                raise RuntimeError("Stopping after transport error; completed results retained")
        output["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        output.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
    output["finished_at"] = utc_now()
    output["summary"] = summarize(output["results"])
    atomic_write(path, output)
    print(json.dumps(output["summary"], indent=2))
    print(f"{output['status']}: {path}")
    return 0 if output["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
