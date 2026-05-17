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

        # Per-arm base-checkpoint sanity (review 007 finding #3).
        # Reject obviously-wrong arm/base combinations before they
        # blow GPU time on a recursive Stage-0 or an invalidated
        # LoRA-text baseline.
        bc = cfg["training"].get("base_checkpoint")
        if bc is not None:
            bc_str = str(bc)
            is_merged_path = "cpt-light-merged" in bc_str or "cpt_light_merged" in bc_str
            is_raw_qwen = bc_str.startswith("Qwen/") or bc_str.startswith("qwen/")
            if arm in ("cpt_light", "lora_text") and is_merged_path:
                errs.append(
                    f"arm={arm!r} cannot use base_checkpoint={bc_str!r}: "
                    f"this arm trains from raw Qwen3-8B, not the merged "
                    f"CPT-light checkpoint (review 007 finding #3). Omit "
                    f"training.base_checkpoint so the trainer's per-arm "
                    f"default applies, or override with Qwen/Qwen3-8B."
                )
            if arm in ("structured_as_text", "xattn") and is_raw_qwen:
                errs.append(
                    f"arm={arm!r} should use the merged CPT-light "
                    f"checkpoint, not raw Qwen3 ({bc_str!r}). The "
                    f"comparison to cross-attn must isolate the "
                    f"architectural contribution; both arms need the "
                    f"same starting point."
                )

    # No shell-meta in any string value (defensive).
    # `rationale` is intentionally excluded: it's free-form agent
    # commentary that never enters shell context, and the auto-loop
    # agent (Task #38) writes multi-line rationales routinely. Caught
    # at first launcher invocation for the x-attn smoke (Task #35).
    forbidden = ("`", "$(", ";", "&&", "||", ">", "<", "\n")
    SHELL_META_SKIP_KEYS = {"rationale"}
    def _walk(node: Any, where: str, skip: bool = False) -> None:
        if skip:
            return
        if isinstance(node, str):
            for ch in forbidden:
                if ch in node:
                    errs.append(f"{where}: forbidden shell char {ch!r} in string value")
                    return
        elif isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{where}.{k}", skip=(k in SHELL_META_SKIP_KEYS))
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


# --- HN-FPR helpers (review 013 finding #1 + #3) -----------------------

def _hn_worst(hn_point: dict[str, float] | None) -> float | None:
    """Worst-family HN-FPR (max). None if dict is empty or contains only NaN."""
    if not hn_point:
        return None
    vals = [v for v in hn_point.values() if isinstance(v, (int, float))]
    finite = [float(v) for v in vals if v == v]  # NaN-aware
    return max(finite) if finite else None


def _hn_mean(hn_point: dict[str, float] | None) -> float | None:
    """Mean HN-FPR across families (tiebreaker)."""
    if not hn_point:
        return None
    vals = [v for v in hn_point.values() if isinstance(v, (int, float))]
    finite = [float(v) for v in vals if v == v]
    return sum(finite) / len(finite) if finite else None


def _load_hn_ci(run_dir: Path, mode: str = "stripped") -> dict | None:
    """Read the hard_negative_fpr_at_1pct block from ci_report_<mode>.json.
    Returns the full per-family + worst/mean CI bundle, or None if absent
    (e.g., ci_report was produced by an old bootstrap_ci before HN-FPR
    support landed in review 013).
    """
    ci_path = run_dir / f"ci_report_{mode}.json"
    if not ci_path.exists():
        return None
    try:
        data = json.loads(ci_path.read_text())
    except json.JSONDecodeError:
        return None
    return data.get("hard_negative_fpr_at_1pct")


def check_halt_conditions() -> str | None:
    """Returns reason string if halted, None if clear to launch.

    Review 013 finding #4: budget caps and convergence count only
    x-attn arm runs. Baselines (cpt_light, lora_text, structured_as_text,
    event_only) are recorded for Day-3 comparison but don't consume the
    x-attn sweep budget or trigger convergence halt.

    Review 013 finding #1: convergence is now on worst-family HN-FPR
    (the post-Day-1-pivot win condition), not AUC-stripped (saturated
    at 1.0 on every model variant).
    """
    if not BUDGET_YAML.exists():
        return None
    budget = yaml.safe_load(BUDGET_YAML.read_text())
    history = _load_history()

    # Filter to x-attn arm for budget + halt counts (review 013 finding #4)
    xattn_history = [h for h in history if h.get("arm") == "xattn"]
    xattn_valid_all = [h for h in xattn_history if h.get("status") == "ok"]

    # Budget caps -- x-attn only
    if len(xattn_valid_all) >= budget.get("max_experiments", 999):
        return (
            f"max_experiments ({budget['max_experiments']}) reached "
            f"for x-attn arm; baselines excluded from this count"
        )

    # GPU hours -- whole history (cost cap is about total spend, not just x-attn)
    gpu_hours = sum(h.get("wall_clock_min", 0) for h in history) / 60.0
    if gpu_hours >= budget.get("max_gpu_hours", 999):
        return f"max_gpu_hours ({budget['max_gpu_hours']}) reached"

    halt = budget.get("halt", {})

    # NaN cascade -- x-attn only (review 013 finding #4)
    if halt.get("nan_cascade", {}).get("enabled"):
        n = halt["nan_cascade"]["consecutive_threshold"]
        recent = xattn_history[-n:]
        if len(recent) == n and all(h.get("status") == "nan" for h in recent):
            return f"nan_cascade: last {n} x-attn runs were NaN"

    # Zero gates -- x-attn only (always was, kept)
    if halt.get("zero_gate_activation", {}).get("enabled"):
        n = halt["zero_gate_activation"]["consecutive_threshold"]
        mag_thresh = halt["zero_gate_activation"]["magnitude_threshold"]
        recent = xattn_valid_all[-n:]
        if len(recent) == n and all(
            (h.get("max_gate_magnitude") or 0) < mag_thresh for h in recent
        ):
            return f"zero_gate_activation: last {n} x-attn runs had max gate < {mag_thresh}"

    # Convergence -- worst-family HN-FPR over a sliding window (review 013
    # finding #1). Compares the WORST (max) recent worst-family value
    # against the FIRST in the window. If best is not at least `thresh`
    # better than first, we have not improved.
    conv = halt.get("convergence", {})
    if conv.get("enabled"):
        min_runs = conv.get("min_valid_runs_before_halt", 6)
        window = conv.get("window", 4)
        thresh = conv.get("hn_fpr_improvement_threshold",
                          conv.get("auc_lift_threshold", 0.005))
        if len(xattn_valid_all) >= min_runs:
            # Audit 014 H5 / audit 015 confirmed: filter Nones first, then
            # rank against the last `window` VALID HN-FPR records.
            # `all(v is not None ...)` silently bypassed convergence whenever
            # a legacy (pre-review-013) row sat in the window.
            worst_series = [
                h.get("hn_fpr_worst_stripped") for h in xattn_valid_all
            ]
            worst_series = [v for v in worst_series if v is not None]
            if len(worst_series) >= window:
                recent_valid = worst_series[-window:]
                # Lower HN-FPR is better. "Improvement" = first - best.
                best = min(recent_valid)
                first = recent_valid[0]
                if (first - best) < thresh:
                    return (
                        f"convergence: no worst-family HN-FPR "
                        f"improvement >= {thresh} over last {window} "
                        f"x-attn runs (first={first:.4f}, best={best:.4f})"
                    )

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
        # Stale-PID self-heal (audit 014 Blocker 2 / audit 015 confirmed).
        # If the lockfile's PID is no longer alive, the prior owner crashed
        # (SIGKILL / preempt / power loss) without releasing. Unlink and
        # retry once. If the PID is alive, or unreadable, leave the lock.
        try:
            stale_pid = int(LOCKFILE.read_text().strip())
        except (OSError, ValueError):
            return None
        try:
            os.kill(stale_pid, 0)
        except ProcessLookupError:
            try:
                LOCKFILE.unlink()
            except FileNotFoundError:
                pass
            try:
                fd = os.open(str(LOCKFILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.write(fd, f"{os.getpid()}\n".encode())
                print(f"[acquire_lock] cleaned stale lock for dead PID {stale_pid}")
                return fd
            except FileExistsError:
                return None
        except PermissionError:
            return None
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


def _per_experiment_timeout_minutes(cfg: dict) -> int:
    """Read the per-experiment wall-clock cap from budget.yaml. Uses
    `stress_run_max_wall_clock_minutes` when this experiment is marked
    as a stress run, otherwise `max_wall_clock_minutes`. Review 008
    finding #4 — the budget file documented these caps but the
    launcher never enforced them.
    """
    if not BUDGET_YAML.exists():
        return 0
    budget = yaml.safe_load(BUDGET_YAML.read_text())
    per_exp = budget.get("per_experiment", {})
    stress = (cfg.get("training", {}) or {}).get("stress_run", False)
    if stress:
        return int(per_exp.get("stress_run_max_wall_clock_minutes", 150))
    return int(per_exp.get("max_wall_clock_minutes", 90))


def launch_trainer(arm: str, config_path: Path, run_dir: Path,
                   cfg: dict | None = None) -> tuple[int, bool]:
    """Run `accelerate launch <trainer>.py --config <cfg>` and stream to train.log.

    Enforces the per-experiment wall-clock cap from budget.yaml (review
    008 finding #4): wait with timeout, send SIGTERM on overrun, then
    SIGKILL after a 30-second grace period.

    Returns (exit_code, timed_out).
    """
    import signal
    import time

    trainer = ARM_TO_TRAINER[arm]
    accel_cfg = REPO_ROOT / "src" / "train" / "accelerate_configs" / "single_h100.yaml"
    log_path = run_dir / "train.log"

    cmd = [
        "accelerate", "launch",
        "--config_file", str(accel_cfg),
        str(REPO_ROOT / trainer),
        "--config", str(config_path),
    ]

    timeout_min = _per_experiment_timeout_minutes(cfg or {})
    timeout_sec = timeout_min * 60 if timeout_min > 0 else None
    print(f"launching: {' '.join(cmd)} (timeout={timeout_min}min)")

    # Ensure PYTHONPATH includes the repo root so the trainer's
    # `from src.X import Y` resolves regardless of how the launcher
    # itself was invoked. `accelerate launch <script.py>` executes the
    # script directly, so its sys.path starts from the script's parent
    # dir, not cwd. Without this, every trainer ModuleNotFoundError's
    # on `src.train.common` etc. — discovered in Task #36 baselines.
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{existing_pp}" if existing_pp else str(REPO_ROOT)

    timed_out = False
    with log_path.open("w") as logf:
        # start_new_session=True puts the trainer in its own process group
        # so we can SIGTERM/SIGKILL the WHOLE tree on timeout (accelerate
        # spawns DDP workers + dataloader children that wouldn't die with
        # proc.kill() alone). Audit 014 H6 / audit 015 confirmed.
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env=env,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            print(
                f"TIMEOUT after {timeout_min} min — SIGTERM to process group",
                file=sys.stderr,
            )
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("SIGTERM did not stop process group; SIGKILL", file=sys.stderr)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                rc = proc.wait()
            # Append a marker line to the log so post-parse sees the cause.
            with log_path.open("a") as logf2:
                logf2.write(
                    f"\n[run_next_experiment.py] TIMEOUT after "
                    f"{timeout_min} minutes; trainer was killed.\n"
                )

    return rc, timed_out


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

    # Rank by HN-FPR composite (review 013 finding #1).
    # Primary: worst-family HN-FPR (minimize -- a single bad customer
    # segment still matters in fraud/risk).
    # Tiebreaker: mean HN-FPR (minimize).
    # AUC is now a sanity gate (must not regress badly), not the objective.
    xattn_completed = [
        h for h in completed
        if h.get("arm") == "xattn" and h.get("hn_fpr_worst_stripped") is not None
    ]
    # sort ascending on (worst, mean) — both lower-is-better
    ranked = sorted(
        xattn_completed,
        key=lambda h: (
            h["hn_fpr_worst_stripped"],
            h.get("hn_fpr_mean_stripped", float("inf")),
        ),
    )

    current_best: dict | None = None
    if ranked:
        b = ranked[0]
        current_best = {
            "exp_id": b["exp_id"],
            "arm": b["arm"],
            "hn_fpr_worst_stripped": b["hn_fpr_worst_stripped"],
            "hn_fpr_mean_stripped": b.get("hn_fpr_mean_stripped"),
            "auc_stripped_sanity": b.get("auc_stripped"),
            "config_summary": b.get("config_summary"),
        }

    top_3 = [
        {
            "exp_id": h["exp_id"],
            "hn_fpr_worst_stripped": h["hn_fpr_worst_stripped"],
            "hn_fpr_mean_stripped": h.get("hn_fpr_mean_stripped"),
        }
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
        # 5. Launch trainer with per-experiment wall-clock cap (review 008 #4)
        t_start = time.time()
        rc, timed_out = launch_trainer(cfg["arm"], config_path, run_dir, cfg)
        wall_clock_min = (time.time() - t_start) / 60.0

        if rc != 0:
            if timed_out:
                print(
                    f"TRAINER TIMEOUT after {wall_clock_min:.1f} min "
                    f"(see {run_dir}/train.log)",
                    file=sys.stderr,
                )
            else:
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
                "status": "timeout" if timed_out else "failed",
                "trainer_exit_code": rc,
                "wall_clock_min": wall_clock_min,
                "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            }
            # Read parsed metrics if any.
            # parse_metrics.py defaults status="ok" when no NaN/OOM marker is
            # found in the log — fine for successful runs, but here we KNOW
            # the trainer exited non-zero (or timed out). Let the parser only
            # REFINE the cause (nan, oom) — never downgrade the launcher's
            # authoritative "failed"/"timeout" back to "ok". Audit 014
            # Blocker 1 / audit 015 confirmed.
            metrics_json = run_dir / "metrics.json"
            if metrics_json.exists():
                m = json.loads(metrics_json.read_text())
                parsed_status = m.get("status")
                if not timed_out and parsed_status in ("nan", "oom"):
                    record["status"] = parsed_status
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

        # Review 013 finding #3: experiments.jsonl rows now carry the
        # HN-FPR fields the auto-loop ranks/halts on, plus per-family
        # detail + CI bounds for Day-3 audit. AUC stays in the row as a
        # sanity field; it's no longer the comparison metric.
        hn_point = (stripped.get("hard_negative_fpr_at_decision_threshold_1pct") or {})
        hn_ci = _load_hn_ci(run_dir, mode="stripped")  # may be None pre-bootstrap
        record = {
            "exp_id": cfg["exp_id"],
            "arm": cfg["arm"],
            "config_hash": config_hash,
            "config_summary": _summarize_config(cfg),
            "status": train_metrics.get("status", "ok"),
            # Primary comparison metric (review 013 finding #1)
            "hn_fpr_worst_stripped": _hn_worst(hn_point),
            "hn_fpr_mean_stripped":  _hn_mean(hn_point),
            "hn_fpr_per_family_stripped": hn_point,  # full dict for Day-3
            "hn_fpr_ci_stripped": hn_ci,             # worst/mean/per-family CI bounds
            # AUC + R@FPR retained as sanity gates (saturated at 1.0; check
            # for regression, not as a leader board)
            "auc_stripped": stripped.get("auc"),
            "auc_opaque": opaque.get("auc"),
            "auc_full": full.get("auc"),
            "r_at_fpr_1pct": (stripped.get("r_at_fpr_0.01") or {}).get("recall"),
            "r_at_fpr_0.1pct": (stripped.get("r_at_fpr_0.001") or {}).get("recall"),
            # Per-stratum AUCs (Day-3 per-journey + per-actor breakdowns)
            "per_journey_auc_stripped": stripped.get("per_journey_auc"),
            "per_actor_auc_stripped":   stripped.get("per_actor_auc"),
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
