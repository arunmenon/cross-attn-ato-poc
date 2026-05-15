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


def _merge_metrics(parser_metrics: dict, run_dir: Path) -> dict:
    """Merge parser output with any trainer-written metrics.json.

    Contract (review 007 finding #2): the trainer's `metrics.json` is
    canonical. Parser output may ONLY fill in fields the trainer did
    not write (e.g., log-derived diagnostics like `train_runtime_seconds`,
    `n_train_loss_records`, or `status` on failed runs where the trainer
    didn't reach the end). Trainer-written fields are never overwritten.
    """
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return parser_metrics

    try:
        trainer_metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return parser_metrics

    if not isinstance(trainer_metrics, dict):
        return parser_metrics

    merged = dict(trainer_metrics)
    for k, v in parser_metrics.items():
        if k not in merged or merged[k] is None:
            merged[k] = v
    return merged


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    args = p.parse_args()

    log_path = args.run_dir / "train.log"
    metrics, trajectory = parse_log(log_path)

    # Merge with trainer-written metrics.json (trainer wins on conflict).
    metrics = _merge_metrics(metrics, args.run_dir)
    _atomic_write_json(args.run_dir / "metrics.json", metrics)

    # gate_trajectory.json: trainer's version is authoritative if it
    # exists. Parser-derived trajectory is only written when the trainer
    # didn't produce one (e.g., non-xattn arms, or failed runs).
    gt_path = args.run_dir / "gate_trajectory.json"
    if not gt_path.exists():
        _atomic_write_json(gt_path, trajectory)

    print(f"wrote {args.run_dir / 'metrics.json'} (status={metrics.get('status', 'unknown')})")
    return 0


# ---------------------------------------------------------------------------
# Self-test (review 007 finding #2 — fail-fast on the merge contract)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Simulate a trainer-written run dir and confirm parse_metrics merges
    instead of overwriting."""
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        # Trainer-canonical metrics (with the canonical fields the
        # x-attn trainer writes)
        trainer_metrics = {
            "status": "ok",
            "arm": "xattn",
            "final_train_loss": 0.234,
            "n_steps": 1500,
            "wall_clock_sec": 1234.5,
            "n_trainable": 214_000_000,
            "max_gate_magnitude": 0.18,
            "predictions": {"stripped": 5000, "opaque": 5000, "full": 5000},
        }
        (run_dir / "metrics.json").write_text(json.dumps(trainer_metrics))
        # Trainer-canonical gate trajectory (non-empty)
        trainer_traj = [{"step": 0, "loss": 1.0, "gates": [[12, 0.01, 0.01]]}]
        (run_dir / "gate_trajectory.json").write_text(json.dumps(trainer_traj))
        # Synthetic train.log that produces parser metrics
        (run_dir / "train.log").write_text(
            "{'loss': 0.5, 'step': 100}\n"
            "{'loss': 0.4, 'step': 200}\n"
            "train_runtime: 1234.5\n"
        )

        # Run the main() flow programmatically
        log_path = run_dir / "train.log"
        parser_metrics, parser_traj = parse_log(log_path)
        merged = _merge_metrics(parser_metrics, run_dir)

        # Trainer fields must survive
        assert merged["max_gate_magnitude"] == 0.18, (
            f"trainer's max_gate_magnitude was overwritten: "
            f"{merged.get('max_gate_magnitude')}"
        )
        assert merged["n_steps"] == 1500
        assert merged["arm"] == "xattn"
        assert merged["predictions"]["stripped"] == 5000

        # Parser fields fill in only missing entries
        # (n_train_loss_records was not in trainer's dict, so parser's
        # value should be present)
        assert "n_train_loss_records" in merged
        assert merged["n_train_loss_records"] >= 2

        # gate_trajectory.json: parser must NOT overwrite the trainer's
        # version. Simulate the main() flow:
        gt_path = run_dir / "gate_trajectory.json"
        if not gt_path.exists():
            _atomic_write_json(gt_path, parser_traj)
        # ...the trainer's trajectory is preserved:
        final_traj = json.loads(gt_path.read_text())
        assert len(final_traj) == 1 and final_traj[0]["step"] == 0, (
            f"trainer's gate_trajectory was overwritten: {final_traj}"
        )

        # Negative case: NO trainer metrics.json. Parser writes its own.
        run_dir_2 = Path(td) / "no_trainer"
        run_dir_2.mkdir()
        (run_dir_2 / "train.log").write_text(
            "{'loss': 0.5, 'step': 100}\n"
        )
        parser_m, _ = parse_log(run_dir_2 / "train.log")
        merged_2 = _merge_metrics(parser_m, run_dir_2)
        # No trainer fields → parser metrics survive
        assert "status" in merged_2

    print("parse_metrics merge contract OK (trainer metrics survive)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
        sys.exit(0)
    sys.exit(main())
