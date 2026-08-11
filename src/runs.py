"""
Run-file resolution, shared by the CLIs (view, evolve, rejudge).

A "run" is one timestamped JSON artifact in data/. Runs can be addressed
as 'latest', a 1-based index into the newest-first list, a filename, or
a partial filename match.
"""

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


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
