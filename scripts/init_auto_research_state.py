#!/usr/bin/env python
"""Initialize auto-research state files for a fresh clone.

Both `src/auto_research/experiments.jsonl` and
`src/auto_research/sweep_state.yaml` are intentionally gitignored
(they are RUNTIME state, not source code). On a fresh clone or after
a state-reset, they don't exist, and the agent's first iteration
would fail to read them.

This script:
  - Creates the parent directory if missing.
  - Touches `experiments.jsonl` to an empty file (0 records).
  - Writes an initial `sweep_state.yaml` representing the empty-history
    state (n_completed=0, current_best=null, halted=false).

Idempotent: safe to run multiple times. Will NOT overwrite an existing
non-empty state file unless --force is passed.

Review 008 finding #6 — this script previously did not exist; the
agent loop assumed the files were always present.

CLI:
    python scripts/init_auto_research_state.py
    python scripts/init_auto_research_state.py --force
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


EMPTY_SWEEP_STATE = {
    "schema_version": 1,
    "n_completed": 0,
    "n_failed": 0,
    "n_xattn_runs": 0,
    "gpu_hours_used": 0.0,
    "current_best": None,
    "top_3": [],
    "halted": False,
    "halt_reason": None,
    "last_updated": None,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing state files even if non-empty",
    )
    args = parser.parse_args()

    import yaml

    root = _repo_root()
    state_dir = root / "src" / "auto_research"
    runs_dir = state_dir / "runs"
    state_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = state_dir / "experiments.jsonl"
    yaml_path = state_dir / "sweep_state.yaml"

    # experiments.jsonl: touch if missing, leave alone if present.
    if jsonl_path.exists() and not args.force:
        if jsonl_path.stat().st_size > 0:
            print(f"  skip {jsonl_path} (has {jsonl_path.stat().st_size} bytes)")
        else:
            print(f"  ok   {jsonl_path} (already empty)")
    else:
        jsonl_path.write_text("")
        print(f"  created {jsonl_path} (empty)")

    # sweep_state.yaml: write empty-state if missing, --force, or n_completed=0.
    if yaml_path.exists() and not args.force:
        try:
            existing = yaml.safe_load(yaml_path.read_text())
        except Exception:
            existing = None
        n_completed = existing.get("n_completed") if isinstance(existing, dict) else None
        if n_completed is None or n_completed == 0:
            new_state = dict(EMPTY_SWEEP_STATE)
            new_state["last_updated"] = dt.datetime.utcnow().isoformat() + "Z"
            yaml_path.write_text(yaml.safe_dump(new_state, sort_keys=False))
            print(f"  refreshed {yaml_path}")
        else:
            print(f"  skip {yaml_path} (n_completed={n_completed}); "
                  f"use --force to reset")
    else:
        new_state = dict(EMPTY_SWEEP_STATE)
        new_state["last_updated"] = dt.datetime.utcnow().isoformat() + "Z"
        yaml_path.write_text(yaml.safe_dump(new_state, sort_keys=False))
        print(f"  created {yaml_path}")

    print("auto-research state initialized")
    return 0


if __name__ == "__main__":
    sys.exit(main())
