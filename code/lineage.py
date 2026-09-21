"""Where every artifact came from.

Each file the pipeline produces gets an entry in data/lineage.json naming
its inputs, the seed that shaped them, and the commit that was checked out
at the time. The entries chain: the splits carry a `split_id` derived from
the source csv and the split seed, and everything downstream inherits it.

That id is the point. Re-running split_data.py with a different seed
produces files with the same names and the same shapes, and every number
computed from them afterwards is quietly incomparable with what came
before. Comparing split_ids turns that into an error instead of a mystery.

    from lineage import record, get, split_id_of, require_same_split
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR, ROOT

LINEAGE_PATH = DATA_DIR / "lineage.json"

# Hashing 16 GB to catch a swapped file is a poor trade. Above this size the
# digest covers the head, middle and tail plus the exact byte count, which
# separates "a different file" from "the same file" in every way this
# pipeline can actually go wrong -- but it is a fingerprint, not a proof,
# and a doctored middle section would slip past it.
FULL_HASH_LIMIT = 256 * 1024 * 1024
SAMPLE_BYTES = 1024 * 1024


def fingerprint(path):
    """Identify a file by content, cheaply enough to do on every run."""
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())

    with path.open("rb") as handle:
        if size <= FULL_HASH_LIMIT:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
            kind = "sha256"
        else:
            for offset in (0, size // 2, max(0, size - SAMPLE_BYTES)):
                handle.seek(offset)
                digest.update(handle.read(SAMPLE_BYTES))
            kind = "sampled"

    return {"name": path.name, "bytes": size,
            "digest": digest.hexdigest()[:16], "digest_kind": kind}


def digest_of(obj):
    """Short stable hash of a json-able value, for things not stored as files."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def git_commit():
    """The checked-out revision, and whether it had uncommitted edits."""
    def run(*args):
        result = subprocess.run(args, cwd=ROOT, capture_output=True,
                                text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        sha = run("git", "rev-parse", "HEAD")
        if sha is None:
            return None
        status = run("git", "status", "--porcelain")
        return {"commit": sha[:12], "dirty": bool(status)}
    except (OSError, subprocess.SubprocessError):
        return None


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load():
    if not LINEAGE_PATH.exists():
        return {}
    return json.loads(LINEAGE_PATH.read_text())


def record(artifact, **fields):
    """Store the provenance of one produced file, keyed by its filename."""
    entries = load()
    entries[Path(artifact).name] = {"written": _now(),
                                    "git": git_commit(), **fields}
    LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LINEAGE_PATH.write_text(json.dumps(entries, indent=2, sort_keys=True))
    return entries[Path(artifact).name]


def get(artifact):
    return load().get(Path(artifact).name)


def make_split_id(source, seed, ratios):
    """A stable name for one particular way of cutting the data."""
    payload = json.dumps({"source": fingerprint(source), "seed": seed,
                          "ratios": ratios}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def split_id_of(artifact):
    entry = get(artifact)
    return entry.get("split_id") if entry else None


def require_same_split(*artifacts):
    """Refuse to mix files that came from different cuts of the data."""
    seen = {Path(a).name: split_id_of(a) for a in artifacts}
    unknown = [name for name, sid in seen.items() if sid is None]
    if unknown:
        print(f"  note: no lineage recorded for {', '.join(unknown)} -- "
              f"rebuild them to have their origin checked")
        return None

    ids = set(seen.values())
    if len(ids) > 1:
        raise ValueError(
            "these files came from different splits and cannot be compared: "
            + ", ".join(f"{name}={sid}" for name, sid in seen.items())
            + " -- rebuild them from one run of split_data.py")
    return ids.pop()
