#!/usr/bin/env python
"""Deterministic experiment launcher.

The agent proposes an experiment by writing a `config.yaml` to a new
`src/auto_research/runs/exp_NNN/` directory. The agent then runs:

    python scripts/run_next_experiment.py src/auto_research/runs/exp_NNN/config.yaml

This script owns:
  - config schema validation (whitelist of keys, sanity ranges)
  - dedup against experiments.jsonl (SHA-256 of canonical config)
  - halt-condition check against budget.yaml
  - GPU lockfile acquisition (O_EXCL, atomic)
  - subprocess launch of `accelerate launch src/train/train_xattn.py` (or the
    appropriate trainer for non-xattn arms)
  - log streaming to train.log
  - post-run: parse_metrics.py + bootstrap_ci.py invocations
  - atomic append of summary entry to experiments.jsonl
  - sweep_state.yaml updates

The agent does NOT call accelerate launch directly. It does NOT modify
metrics.json. It does NOT release the lockfile manually.

Exit codes:
    0  success (experiment launched, completed, recorded)
    1  config validation failure
    2  duplicate config (already in history)
    3  halt condition active — refusing to launch
    4  lockfile contention (another run in progress)
    5  trainer crashed (NaN / OOM / other)
    6  post-run parsing/scoring failure

CLI:
    python scripts/run_next_experiment.py PATH_TO_CONFIG.yaml
    python scripts/run_next_experiment.py --mark-failed runs/exp_NNN
    python scripts/run_next_experiment.py --halt-check  (returns 0 if can launch, 3 if halted)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "src" / "auto_research" / "runs"
EXPERIMENTS_JSONL = REPO_ROOT / "src" / "auto_research" / "experiments.jsonl"
SWEEP_STATE = REPO_ROOT / "src" / "auto_research" / "sweep_state.yaml"
BUDGET_YAML = REPO_ROOT / "src" / "auto_research" / "configs" / "budget.yaml"
SWEEP_SPACE_YAML = REPO_ROOT / "src" / "auto_research" / "configs" / "sweep_space.yaml"
LOCKFILE = Path("/workspace/.gpu.lock") if Path("/workspace").exists() else REPO_ROOT / ".gpu.lock"


# ---------------------------------------------------------------------------
# Schema (manual whitelist — Minor Patch 2)
# ---------------------------------------------------------------------------

ALLOWED_ARMS = {"xattn", "cpt_light", "lora_text", "structured_as_text", "event_only"}

# Per-arm allowed keys. Unknown keys = reject.
ARM_SCHEMA: dict[str, dict[str, type]] = {
    "xattn": {
        "exp_id": str,
        "arm": str,
        "xattn": dict,
        "training": dict,
        "data": dict,
        "eval": dict,
        "rationale": str,
    },
    "cpt_light": {
        "exp_id": str, "arm": str, "training": dict, "data": dict, "eval": dict, "rationale": str,
    },
    "lora_text": {
        "exp_id": str, "arm": str, "training": dict, "data": dict, "eval": dict, "rationale": str,
    },
    "structured_as_text": {
        "exp_id": str, "arm": str, "training": dict, "data": dict, "eval": dict, "rationale": str,
    },
    "event_only": {
        "exp_id": str, "arm": str, "training": dict, "data": dict, "eval": dict, "rationale": str,
    },
}

# Allowed sub-keys (one level deep) — defensive against shell-injection-like config values.
XATTN_ALLOWED = {"insertion_pattern", "gate_init", "resampler_slots", "encoder", "lora_r_on_q"}
TRAINING_ALLOWED = {
    "base_checkpoint", "steps", "seq_len", "micro_batch", "grad_accum",
    "lr", "warmup_steps", "precision", "optimizer", "stress_run",
}
DATA_ALLOWED = {"train_path", "eval_fast_path", "eval_mode_dropout"}
EVAL_ALLOWED = {
    "modes", "primary_mode", "bootstrap_resamples",
    "per_journey_breakdown", "per_actor_breakdown", "hard_negative_fpr",
}


def _validate_dict(d: dict, allowed: set[str], where: str) -> list[str]:
    return [f"{where}.{k}: unknown key" for k in d.keys() if k not in allowed]


def validate_config(cfg: dict) -> list[str]:
    """Return list of validation errors. Empty list = OK."""
    errs: list[str] = []
    if "arm" not in cfg:
        errs.append("missing top-level 'arm'")
        return errs
    arm = cfg["arm"]
    if arm not in ALLOWED_ARMS:
        errs.append(f"arm: unknown value {arm!r}")
        return errs

    schema = ARM_SCHEMA[arm]
    for key in cfg:
        if key not in schema:
            errs.append(f"top-level: unknown key {key!r}")
    for key, typ in schema.items():
        if key in cfg and not isinstance(cfg[key], typ):
            errs.append(f"{key}: expected {typ.__name__}, got {type(cfg[key]).__name__}")

    if arm == "xattn" and isinstance(cfg.get("xattn"), dict):
        errs += _validate_dict(cfg["xattn"], XATTN_ALLOWED, "xattn")

    if isinstance(cfg.get("training"), dict):
        errs += _validate_dict(cfg["training"], TRAINING_ALLOWED, "training")
    if isinstance(cfg.get("data"), dict):
        errs += _validate_dict(cfg["data"], DATA_ALLOWED, "data")
    if isinstance(cfg.get("eval"), dict):
        errs += _validate_dict(cfg["eval"], EVAL_ALLOWED, "eval")

    # Sanity ranges
    if isinstance(cfg.get("training"), dict):
        steps = cfg["training"].get("steps")
        if steps is not None and not (50 <= steps <= 10000):
            errs.append(f"training.steps {steps} out of [50, 10000]")
        seq_len = cfg["training"].get("seq_len")
        if seq_len is not None and seq_len not in {1024, 2048, 4096}:
            errs.append(f"training.seq_len {seq_len} not in {{1024, 2048, 4096}}")
        lr = cfg["training"].get("lr")
        if lr is not None and not (1e-6 <= lr <= 1e-2):
            errs.append(f"training.lr {lr} out of [1e-6, 1e-2]")

    # No shell-meta in any string value (defensive)
    forbidden = ("`", "$(", ";", "&&", "||", ">", "<", "\n")
    def _walk(node: Any, where: str) -> None:
        if isinstance(node, str):
            for ch in forbidden:
                if ch in node:
                    errs.append(f"{where}: forbidden shell char {ch!r} in string value")
                    return
        elif isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{where}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _walk(v, f"{where}[{i}]")
    _walk(cfg, "$")

    return errs


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def canonical_hash(cfg: dict) -> str:
    """SHA-256 of the canonical config form, excluding exp_id and rationale."""
    canonical = {k: v for k, v in cfg.items() if k not in ("exp_id", "rationale")}
    blob = json.dumps(canonical, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def already_run(config_hash: str) -> str | None:
    """Returns the exp_id of a prior run with this hash, if any."""
    if not EXPERIMENTS_JSONL.exists():
        return None
    with EXPERIMENTS_JSONL.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("config_hash") == config_hash and rec.get("status") != "failed":
                return rec.get("exp_id")
    return None


# ---------------------------------------------------------------------------
# Halt conditions
# ---------------------------------------------------------------------------

def _load_history() -> list[dict]:
    if not EXPERIMENTS_JSONL.exists():
        return []
    out: list[dict] = []
    with EXPERIMENTS_JSONL.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def check_halt_conditions() -> str | None:
    """Returns reason string if halted, None if clear to launch."""
    if not BUDGET_YAML.exists():
        return None
    budget = yaml.safe_load(BUDGET_YAML.read_text())
    history = _load_history()

    # Budget caps
    valid = [h for h in history if h.get("status") == "ok"]
    if len(valid) >= budget.get("max_experiments", 999):
        return f"max_experiments ({budget['max_experiments']}) reached"

    gpu_hours = sum(h.get("wall_clock_min", 0) for h in history) / 60.0
    if gpu_hours >= budget.get("max_gpu_hours", 999):
        return f"max_gpu_hours ({budget['max_gpu_hours']}) reached"

    halt = budget.get("halt", {})

    # NaN cascade
    if halt.get("nan_cascade", {}).get("enabled"):
        n = halt["nan_cascade"]["consecutive_threshold"]
        xattn_hist = [h for h in history if h.get("arm") == "xattn"][-n:]
        if len(xattn_hist) == n and all(h.get("status") == "nan" for h in xattn_hist):
            return f"nan_cascade: last {n} x-attn runs were NaN"

    # Zero gates
    if halt.get("zero_gate_activation", {}).get("enabled"):
        n = halt["zero_gate_activation"]["consecutive_threshold"]
        mag_thresh = halt["zero_gate_activation"]["magnitude_threshold"]
        xattn_valid = [h for h in history if h.get("arm") == "xattn" and h.get("status") == "ok"][-n:]
        if len(xattn_valid) == n and all(
            (h.get("max_gate_magnitude") or 0) < mag_thresh for h in xattn_valid
        ):
            return f"zero_gate_activation: last {n} x-attn runs had max gate < {mag_thresh}"

    # Convergence
    conv = halt.get("convergence", {})
    if conv.get("enabled"):
        min_runs = conv.get("min_valid_runs_before_halt", 6)
        window = conv.get("window", 4)
        thresh = conv.get("auc_lift_threshold", 0.005)
        xattn_valid = [h for h in history if h.get("arm") == "xattn" and h.get("status") == "ok"]
        if len(xattn_valid) >= min_runs:
            recent = xattn_valid[-window:]
            if len(recent) == window:
                aucs = [h.get("auc_stripped") for h in recent]
                if all(a is not None for a in aucs):
                    if max(aucs) - aucs[0] < thresh:
                        return f"convergence: no AUC-stripped lift >= {thresh} over last {window} runs"

    return None


# ---------------------------------------------------------------------------
# Lockfile (atomic, PID-stamped)
# ---------------------------------------------------------------------------

def acquire_lock() -> int | None:
    try:
        fd = os.open(str(LOCKFILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, f"{os.getpid()}\n".encode())
        return fd
    except FileExistsError:
        return None


def release_lock(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        LOCKFILE.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Trainer dispatch
# ---------------------------------------------------------------------------

ARM_TO_TRAINER = {
    "xattn": "src/train/train_xattn.py",
    "cpt_light": "src/train/train_cpt_light.py",
    "lora_text": "src/train/train_lora_text_only.py",
    "structured_as_text": "src/train/train_structured_as_text.py",
    "event_only": "src/train/train_event_only_classifier.py",
}


def launch_trainer(arm: str, config_path: Path, run_dir: Path) -> int:
    """Run `accelerate launch <trainer>.py --config <cfg>` and stream to train.log.

    Returns the trainer's exit code.
    """
    trainer = ARM_TO_TRAINER[arm]
    accel_cfg = REPO_ROOT / "src" / "train" / "accelerate_configs" / "single_h100.yaml"
    log_path = run_dir / "train.log"

    cmd = [
        "accelerate", "launch",
        "--config_file", str(accel_cfg),
        str(REPO_ROOT / trainer),
        "--config", str(config_path),
    ]
    print(f"launching: {' '.join(cmd)}")
    with log_path.open("w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
        rc = proc.wait()
    return rc


# ---------------------------------------------------------------------------
# Post-run: parse + score + CI
# ---------------------------------------------------------------------------

def run_post_processing(run_dir: Path, eval_modes: list[str]) -> dict:
    """Invoke parse_metrics.py, score_risk.py (per mode), bootstrap_ci.py.

    The trainer is expected to have written `predictions_<mode>.jsonl` per
    eval mode into the run dir. score_risk and bootstrap_ci read those.

    Side effects (post-review fix #4):
      - Writes per-mode files: metrics_{mode}.json, ci_report_{mode}.json
      - Writes aggregate ci_report.json with sections per mode (canonical file
        referenced by PLAN.md and AGENT_INSTRUCTIONS.md).
    """
    # 1. Parse training log → metrics.json + gate_trajectory.json
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "parse_metrics.py"),
         "--run-dir", str(run_dir)],
        check=True,
    )

    # 2. Score risk for each eval mode that produced predictions
    summary: dict[str, dict] = {}
    per_mode_ci: dict[str, dict] = {}
    for mode in eval_modes:
        preds = run_dir / f"predictions_{mode}.jsonl"
        if not preds.exists():
            summary[mode] = {"missing_predictions": True}
            continue
        metrics_out = run_dir / f"metrics_{mode}.json"
        ci_out = run_dir / f"ci_report_{mode}.json"
        subprocess.run(
            [sys.executable, "-m", "eval.score_risk",
             "--predictions", str(preds), "--out", str(metrics_out)],
            check=True, cwd=str(REPO_ROOT),
        )
        subprocess.run(
            [sys.executable, "-m", "eval.bootstrap_ci",
             "--predictions", str(preds), "--out", str(ci_out)],
            check=True, cwd=str(REPO_ROOT),
        )
        with metrics_out.open() as f:
            summary[mode] = json.load(f)
        with ci_out.open() as f:
            per_mode_ci[mode] = json.load(f)

    # 3. Aggregate per-mode CI files into a single canonical ci_report.json
    if per_mode_ci:
        agg_path = run_dir / "ci_report.json"
        tmp = agg_path.with_suffix(agg_path.suffix + ".tmp")
        tmp.write_text(json.dumps(per_mode_ci, indent=2))
        tmp.rename(agg_path)

    return summary


# ---------------------------------------------------------------------------
# Atomic experiments.jsonl append + sweep_state.yaml update
# Launcher owns BOTH state files (post-review fix #3).
# ---------------------------------------------------------------------------

def append_experiment(record: dict) -> None:
    """Append a single experiment record to experiments.jsonl atomically.

    Read existing → append → write to .tmp → rename. Concurrent writers are
    serialized by the GPU lockfile (we never get here without holding it).
    """
    EXPERIMENTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    existing = EXPERIMENTS_JSONL.read_text() if EXPERIMENTS_JSONL.exists() else ""
    new = existing + (json.dumps(record) + "\n")
    tmp = EXPERIMENTS_JSONL.with_suffix(EXPERIMENTS_JSONL.suffix + ".tmp")
    tmp.write_text(new)
    tmp.rename(EXPERIMENTS_JSONL)


def update_sweep_state() -> None:
    """Recompute sweep_state.yaml from experiments.jsonl + budget.yaml.

    Atomic write. Called by _do_run after each experiment completes.
    The agent reads this file but does NOT write it (post-review fix #3).
    """
    history = _load_history()
    completed = [h for h in history if h.get("status") == "ok"]
    failed = [h for h in history if h.get("status") not in ("ok", None)]
    gpu_hours = sum(h.get("wall_clock_min", 0) or 0 for h in history) / 60.0

    # Rank by AUC-stripped (xattn arm specifically — baselines don't compete
    # with each other in the same way).
    xattn_completed = [
        h for h in completed
        if h.get("arm") == "xattn" and h.get("auc_stripped") is not None
    ]
    ranked = sorted(xattn_completed, key=lambda h: h["auc_stripped"], reverse=True)

    current_best: dict | None = None
    if ranked:
        b = ranked[0]
        current_best = {
            "exp_id": b["exp_id"],
            "arm": b["arm"],
            "auc_stripped": b["auc_stripped"],
            "config_summary": b.get("config_summary"),
        }

    top_3 = [
        {"exp_id": h["exp_id"], "auc_stripped": h["auc_stripped"]}
        for h in ranked[:3]
    ]

    halt_reason = check_halt_conditions()

    state = {
        "schema_version": 1,
        "n_completed": len(completed),
        "n_failed": len(failed),
        "n_xattn_runs": len(xattn_completed),
        "gpu_hours_used": round(gpu_hours, 3),
        "current_best": current_best,
        "top_3": top_3,
        "halted": halt_reason is not None,
        "halt_reason": halt_reason,
        "last_updated": dt.datetime.utcnow().isoformat() + "Z",
    }

    SWEEP_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SWEEP_STATE.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(state, sort_keys=False))
    tmp.rename(SWEEP_STATE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _do_run(config_path: Path) -> int:
    if not config_path.exists():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 1

    cfg = yaml.safe_load(config_path.read_text())
    if not isinstance(cfg, dict):
        print("config must be a YAML mapping", file=sys.stderr)
        return 1

    # 1. Validate schema
    errs = validate_config(cfg)
    if errs:
        print("config validation FAILED:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1

    # 2. Dedup
    config_hash = canonical_hash(cfg)
    prior = already_run(config_hash)
    if prior is not None:
        print(f"DUPLICATE: this config matches prior run {prior} (hash={config_hash})", file=sys.stderr)
        return 2

    # 3. Halt check
    halt_reason = check_halt_conditions()
    if halt_reason is not None:
        print(f"HALT: {halt_reason}", file=sys.stderr)
        return 3

    # 4. Acquire GPU lock
    lock_fd = acquire_lock()
    if lock_fd is None:
        print(f"LOCK CONTENTION: {LOCKFILE} exists (another run in progress)", file=sys.stderr)
        return 4

    run_dir = config_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 5. Launch trainer
        t_start = time.time()
        rc = launch_trainer(cfg["arm"], config_path, run_dir)
        wall_clock_min = (time.time() - t_start) / 60.0

        if rc != 0:
            print(f"TRAINER FAILED: exit code {rc} (see {run_dir}/train.log)", file=sys.stderr)
            # Still parse what we can — captures NaN/OOM markers
            try:
                subprocess.run(
                    [sys.executable, str(REPO_ROOT / "scripts" / "parse_metrics.py"),
                     "--run-dir", str(run_dir)],
                    check=False,
                )
            except Exception:
                pass

            record = {
                "exp_id": cfg["exp_id"],
                "arm": cfg["arm"],
                "config_hash": config_hash,
                "status": "failed",
                "trainer_exit_code": rc,
                "wall_clock_min": wall_clock_min,
                "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            }
            # Read parsed metrics if any
            metrics_json = run_dir / "metrics.json"
            if metrics_json.exists():
                m = json.loads(metrics_json.read_text())
                record["status"] = m.get("status", "failed")
                record["max_gate_magnitude"] = m.get("max_gate_magnitude")
            append_experiment(record)
            update_sweep_state()
            return 5

        # 6. Post-process (parse log, score predictions, bootstrap CI)
        eval_modes = (cfg.get("eval") or {}).get("modes", ["stripped", "opaque", "full"])
        try:
            per_mode_metrics = run_post_processing(run_dir, eval_modes)
        except subprocess.CalledProcessError as e:
            print(f"POST-RUN PARSING FAILED: {e}", file=sys.stderr)
            return 6

        # 7. Append summary to experiments.jsonl
        stripped = per_mode_metrics.get("stripped", {})
        opaque = per_mode_metrics.get("opaque", {})
        full = per_mode_metrics.get("full", {})
        # Read overall training metrics too
        train_metrics_path = run_dir / "metrics.json"
        train_metrics = json.loads(train_metrics_path.read_text()) if train_metrics_path.exists() else {}

        record = {
            "exp_id": cfg["exp_id"],
            "arm": cfg["arm"],
            "config_hash": config_hash,
            "config_summary": _summarize_config(cfg),
            "status": train_metrics.get("status", "ok"),
            "auc_stripped": stripped.get("auc"),
            "auc_opaque": opaque.get("auc"),
            "auc_full": full.get("auc"),
            "r_at_fpr_1pct": (stripped.get("r_at_fpr_0.01") or {}).get("recall"),
            "max_gate_magnitude": train_metrics.get("max_gate_magnitude"),
            "wall_clock_min": wall_clock_min,
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            "leakage_clean": _check_leakage_clean(run_dir),
        }
        append_experiment(record)
        update_sweep_state()
        print(f"OK: appended exp {cfg['exp_id']} to {EXPERIMENTS_JSONL}")
        return 0

    finally:
        release_lock(lock_fd)


def _summarize_config(cfg: dict) -> str:
    arm = cfg["arm"]
    if arm == "xattn":
        x = cfg.get("xattn", {})
        return f"{x.get('insertion_pattern')} / slots={x.get('resampler_slots')} / gate={x.get('gate_init')} / {x.get('encoder')}"
    return arm


def _check_leakage_clean(run_dir: Path) -> bool:
    """Reads leakage_report.json if present. Returns True if no violations."""
    p = run_dir / "leakage_report.json"
    if not p.exists():
        return True  # trainer didn't write one; treat as benign-unknown
    rep = json.loads(p.read_text())
    return bool(rep.get("clean", True))


def _do_mark_failed(run_dir: Path) -> int:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        print(f"no config at {config_path}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(config_path.read_text())
    record = {
        "exp_id": cfg.get("exp_id", run_dir.name),
        "arm": cfg.get("arm", "unknown"),
        "config_hash": canonical_hash(cfg),
        "status": "failed",
        "reason": "marked_failed_by_user",
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
    }
    append_experiment(record)
    update_sweep_state()
    print(f"marked {record['exp_id']} as failed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", type=Path, help="path to config.yaml")
    parser.add_argument("--mark-failed", type=Path, help="mark a run directory as failed")
    parser.add_argument("--halt-check", action="store_true", help="check halt conditions, exit 0 if clear")
    args = parser.parse_args()

    if args.mark_failed is not None:
        return _do_mark_failed(args.mark_failed)

    if args.halt_check:
        reason = check_halt_conditions()
        if reason is None:
            print("clear to launch")
            return 0
        print(f"HALT: {reason}", file=sys.stderr)
        return 3

    if args.config is None:
        parser.error("config path is required (or use --mark-failed / --halt-check)")
    return _do_run(args.config)


if __name__ == "__main__":
    sys.exit(main())
