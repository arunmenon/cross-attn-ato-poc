#!/usr/bin/env python
"""Initialize V5 auto-research runtime state from V4 seed runs.

This is a no-training script. It archives the existing runtime state,
rescoring the V4 text-only control and x-attn leader with the current V5
post-processing path, then rewrites experiments.jsonl and sweep_state.yaml
from those two rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_next_experiment import (  # noqa: E402
    EXPERIMENTS_JSONL,
    SWEEP_STATE,
    _check_leakage_clean,
    _hn_mean,
    _hn_worst,
    _load_hn_ci,
    _load_v5_ci,
    _read_clean_eval_record_fields,
    _summarize_config,
    _v5_family_field,
    append_experiment,
    canonical_hash,
    run_post_processing,
    update_sweep_state,
)

SEED_EXP_IDS = ("exp_text_only_v4_001", "exp_xattn_v4_001")


def _archive(path: Path, stamp: str, dry_run: bool) -> Path | None:
    if not path.exists():
        return None
    archive = path.with_name(f"{path.name}.pre_v5_{stamp}")
    if dry_run:
        print(f"would archive {path} -> {archive}")
        return archive
    path.rename(archive)
    print(f"archived {path} -> {archive}")
    return archive


def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    if not isinstance(cfg, dict):
        raise RuntimeError(f"config must be a mapping: {cfg_path}")
    return cfg


def _seed_record(run_dir: Path, cfg: dict, per_mode_metrics: dict) -> dict:
    stripped = per_mode_metrics.get("stripped", {})
    opaque = per_mode_metrics.get("opaque", {})
    full = per_mode_metrics.get("full", {})
    train_metrics_path = run_dir / "metrics.json"
    train_metrics = json.loads(train_metrics_path.read_text()) if train_metrics_path.exists() else {}

    hn_point = stripped.get("hard_negative_fpr_at_decision_threshold_1pct") or {}
    hn_ci = _load_hn_ci(run_dir, mode="stripped")
    v5_adv = stripped.get("v5_adversarial") or {}
    v5_ci = _load_v5_ci(run_dir, mode="stripped")
    clean_eval_fields = _read_clean_eval_record_fields(run_dir)

    wall_clock_min = None
    if train_metrics.get("wall_clock_sec") is not None:
        wall_clock_min = float(train_metrics["wall_clock_sec"]) / 60.0

    return {
        "exp_id": cfg["exp_id"],
        "arm": cfg["arm"],
        "config_hash": canonical_hash(cfg),
        "config_summary": _summarize_config(cfg),
        "status": train_metrics.get("status", "ok"),
        "metric_version": 5,
        "v5_adv_error": v5_adv.get("v5_adv_error"),
        "v5_adv_components_stripped": v5_adv.get("components"),
        "v5_adv_error_ci_stripped": v5_ci,
        "v5_phish_takeover_recall": _v5_family_field(
            v5_adv, "phish_takeover", "recall"
        ),
        "v5_phish_takeover_mfa_phished_recall": _v5_family_field(
            v5_adv, "phish_takeover_mfa_phished", "recall"
        ),
        "v5_hn_recovery_high_amount_fpr": _v5_family_field(
            v5_adv, "hn_recovery_high_amount", "fpr"
        ),
        "hn_fpr_worst_stripped": _hn_worst(hn_point),
        "hn_fpr_mean_stripped": _hn_mean(hn_point),
        "hn_fpr_per_family_stripped": hn_point,
        "hn_fpr_ci_stripped": hn_ci,
        "auc_stripped": stripped.get("auc"),
        "auc_opaque": opaque.get("auc"),
        "auc_full": full.get("auc"),
        "r_at_fpr_1pct": (stripped.get("r_at_fpr_0.01") or {}).get("recall"),
        "r_at_fpr_0.1pct": (stripped.get("r_at_fpr_0.001") or {}).get("recall"),
        "per_journey_auc_stripped": stripped.get("per_journey_auc"),
        "per_actor_auc_stripped": stripped.get("per_actor_auc"),
        "max_gate_magnitude": train_metrics.get("max_gate_magnitude"),
        "wall_clock_min": wall_clock_min or 0.0,
        "timestamp": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "leakage_clean": _check_leakage_clean(run_dir),
        **clean_eval_fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="score but do not archive or write state")
    args = parser.parse_args()

    runs_dir = REPO_ROOT / "src" / "auto_research" / "runs"
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")

    rows: list[dict] = []
    for exp_id in SEED_EXP_IDS:
        run_dir = runs_dir / exp_id
        cfg = _load_config(run_dir)
        eval_modes = (cfg.get("eval") or {}).get("modes", ["stripped", "opaque", "full"])
        print(f"[{exp_id}] scoring modes={eval_modes}")
        per_mode_metrics = run_post_processing(run_dir, eval_modes, cfg=cfg)
        rows.append(_seed_record(run_dir, cfg, per_mode_metrics))
        print(
            f"[{exp_id}] v5_adv_error={rows[-1]['v5_adv_error']} "
            f"hn_fpr_worst={rows[-1]['hn_fpr_worst_stripped']}"
        )

    if args.dry_run:
        for row in rows:
            print(json.dumps(row))
        return 0

    _archive(EXPERIMENTS_JSONL, stamp, dry_run=False)
    _archive(SWEEP_STATE, stamp, dry_run=False)
    EXPERIMENTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_JSONL.write_text("")
    for row in rows:
        append_experiment(row)
    update_sweep_state()
    print(f"wrote V5 state with {len(rows)} seed rows")
    print(f"experiments: {EXPERIMENTS_JSONL}")
    print(f"sweep_state: {SWEEP_STATE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
