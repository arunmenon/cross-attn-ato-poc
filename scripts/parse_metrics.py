#!/usr/bin/env python
"""Parse a train.log + W&B outputs into a single metrics.json.

Robust to log format variation. Extracts:
- final train loss
- final eval loss
- final AUC under three eval modes (if logged)
- gate-activation magnitude trajectory (one entry per logged step)
- wall-clock duration
- status (ok | nan | oom | other_error)

CLI:
    python scripts/parse_metrics.py --run-dir runs/exp_NNN
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Standard HF Trainer log lines look like:
#   {'loss': 1.234, 'learning_rate': 1e-4, 'epoch': 0.01}
# or W&B-style:
#   step 1500 train_loss 1.234 eval_loss 1.789 ...
TRAIN_LOSS_RE = re.compile(r"'loss':\s*([0-9.eE+-]+|nan)")
EVAL_LOSS_RE = re.compile(r"'eval_loss':\s*([0-9.eE+-]+|nan)")
EVAL_AUC_RE = re.compile(r"'eval_auc_(stripped|opaque|full)':\s*([0-9.eE+-]+|nan)")
STEP_RE = re.compile(r"'step':\s*(\d+)|'global_step':\s*(\d+)")
GATE_RE = re.compile(r"gate_magnitude(?:_layer_\d+)?[:=]\s*([0-9.eE+-]+|nan)")
NAN_MARKER_RE = re.compile(r"\bnan\b|loss is nan|gradient overflow", re.IGNORECASE)
OOM_MARKER_RE = re.compile(r"CUDA out of memory|OutOfMemoryError|OOM", re.IGNORECASE)
DURATION_RE = re.compile(r"train_runtime[\"']?\s*[:=]\s*([0-9.eE+-]+)")


def _try_float(s: str) -> float:
    s = s.lower()
    if s == "nan":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {"status": "missing_log", "log_path": str(log_path)}

    text = log_path.read_text(errors="replace")

    train_losses = [_try_float(m.group(1)) for m in TRAIN_LOSS_RE.finditer(text)]
    eval_losses = [_try_float(m.group(1)) for m in EVAL_LOSS_RE.finditer(text)]
    auc_by_mode: dict[str, list[float]] = {"stripped": [], "opaque": [], "full": []}
    for m in EVAL_AUC_RE.finditer(text):
        mode = m.group(1)
        auc_by_mode[mode].append(_try_float(m.group(2)))

    gate_trajectory: list[dict] = []
    for line in text.splitlines():
        sm = STEP_RE.search(line)
        gm = GATE_RE.search(line)
        if sm and gm:
            step = int(sm.group(1) or sm.group(2))
            mag = _try_float(gm.group(1))
            gate_trajectory.append({"step": step, "magnitude": mag})

    duration = None
    dm = DURATION_RE.search(text)
    if dm:
        duration = _try_float(dm.group(1))

    status = "ok"
    if OOM_MARKER_RE.search(text):
        status = "oom"
    elif NAN_MARKER_RE.search(text):
        # Distinguish "loss is nan" from incidental "nan" in unrelated context
        if "loss is nan" in text.lower() or "gradient overflow" in text.lower():
            status = "nan"
        elif train_losses and any(v != v for v in train_losses):  # NaN check
            status = "nan"

    out: dict = {
        "status": status,
        "n_train_loss_records": len(train_losses),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "final_eval_auc": {
            mode: (vals[-1] if vals else None) for mode, vals in auc_by_mode.items()
        },
        "gate_trajectory_n_records": len(gate_trajectory),
        "max_gate_magnitude": max((g["magnitude"] for g in gate_trajectory
                                   if g["magnitude"] == g["magnitude"]),  # filter NaN
                                  default=None),
        "train_runtime_seconds": duration,
    }
    return out, gate_trajectory


def _atomic_write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    args = p.parse_args()

    log_path = args.run_dir / "train.log"
    metrics, trajectory = parse_log(log_path)

    _atomic_write_json(args.run_dir / "metrics.json", metrics)
    _atomic_write_json(args.run_dir / "gate_trajectory.json", trajectory)
    print(f"wrote {args.run_dir / 'metrics.json'} (status={metrics['status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
