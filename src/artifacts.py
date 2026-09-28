"""Versioned, atomic artifacts and source provenance shared by experiments."""
import hashlib
import json
import platform
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def source_provenance() -> dict:
    def git(*args):
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    paths = sorted(set(ROOT.glob("*.py")) | set((ROOT / "src").rglob("*.py")) |
                   {ROOT / "pyproject.toml", ROOT / "uv.lock"})
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths if p.is_file()}
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "platform": platform.platform(), "source_sha256": hashes}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_path(directory: Path, prefix: str, tag: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("._")[:100]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"{prefix}_{stamp}_{safe or 'untagged'}_{uuid.uuid4().hex[:8]}.json"


def atomic_write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
