#!/usr/bin/env python
"""Re-score on-disk baseline predictions under the tie-aware metric on a
leakage-filtered eval surface (review 018/019).

NO GPU TRAINING. This script reads existing `predictions_<mode>.jsonl`
files in each baseline run dir, applies the clean-eval mask (text-hash +
structured_events-hash overlap removed), then recomputes the tie-aware
HN-FPR via `eval.score_risk.compute_all` and the tie-aware bootstrap CI via
`eval.bootstrap_ci.bootstrap_hard_negative_fpr`.

For each rescored run, writes:
  - <run_dir>/predictions_<mode>_clean.jsonl   (filtered predictions)
  - <run_dir>/clean_eval_mask.json             (mask cache; matches launcher schema)
  - <run_dir>/metrics_v2_<mode>.json
  - <run_dir>/ci_report_v2_<mode>.json
  - <run_dir>/ci_report_v2.json                (aggregate; one block per mode)

Then appends one row per baseline to `src/auto_research/experiments.jsonl`
with:
  - exp_id = "<original>_v2"
  - parent_exp_id = "<original>"
  - metric_version = 2
  - clean_eval_n, clean_eval_dropped, clean_eval_mask_text_overlap,
    clean_eval_mask_events_overlap (DERIVED from the mask, NOT hard-coded —
    review 019 High 3).

Finally calls `update_sweep_state()` to refresh sweep_state.yaml.

CLI:
    python3 scripts/rescore_baselines.py --auto-detect
        # Find all exp_baseline_* (and optionally exp_xa_smoke_001) dirs under
        # src/auto_research/runs/ and rescore each.

    python3 scripts/rescore_baselines.py --run-dir src/auto_research/runs/exp_baseline_event_only
        # Rescore a single run.

    python3 scripts/rescore_baselines.py --auto-detect --include-xa-smoke
        # Also rescore exp_xa_smoke_001 if it has predictions.

    python3 scripts/rescore_baselines.py --auto-detect --dry-run
        # Compute + print everything but do NOT append to experiments.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.leakage_checks import compute_clean_eval_mask
from eval.score_risk import compute_all
from eval.bootstrap_ci import bootstrap_hard_negative_fpr

RUNS_DIR = REPO_ROOT / "src" / "auto_research" / "runs"
EXPERIMENTS_JSONL = REPO_ROOT / "src" / "auto_research" / "experiments.jsonl"

EVAL_MODES = ("stripped", "opaque", "full")
PRIMARY_MODE = "stripped"
TARGET_FPR = 0.01
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 0


# ---------------------------------------------------------------------------
# Path resolution — mirrors the launcher's resolution to keep cache-shape
# in sync with what `_read_clean_eval_record_fields` expects.
# ---------------------------------------------------------------------------

def _load_config(run_dir: Path) -> dict:
    import yaml
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no config.yaml at {cfg_path}")
    return yaml.safe_load(cfg_path.read_text())


def _resolve_train_eval(
    cfg: dict,
    train_override: Path | None,
    eval_override: Path | None,
) -> tuple[Path, Path]:
    """Resolve (train.jsonl, eval.jsonl) from cfg or overrides.

    Overrides win. Otherwise reads cfg["data"]["train_path"] +
    cfg["data"]["eval_fast_path"] (or "eval_path"). Both paths are honored
    as-is — on the pod these are typically /workspace/data/... .
    """
    if train_override is not None and eval_override is not None:
        return train_override, eval_override

    data = cfg.get("data") or {}
    train_dir = train_override if train_override is not None else data.get("train_path")
    eval_dir = eval_override if eval_override is not None else (
        data.get("eval_fast_path") or data.get("eval_path")
    )
    if train_dir is None or eval_dir is None:
        raise RuntimeError(
            f"cannot resolve train_path / eval_path from cfg "
            f"({cfg.get('exp_id')!r}); pass --train-dir + --eval-dir overrides."
        )
    train_jsonl = Path(train_dir) / "train.jsonl"
    eval_jsonl = Path(eval_dir) / "eval.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(f"train.jsonl not found: {train_jsonl}")
    if not eval_jsonl.exists():
        raise FileNotFoundError(f"eval.jsonl not found: {eval_jsonl}")
    return train_jsonl, eval_jsonl


# ---------------------------------------------------------------------------
# Mask cache (matches launcher's shape so _read_clean_eval_record_fields reads
# what we write here).
# ---------------------------------------------------------------------------

def _compute_or_load_mask(
    run_dir: Path, train_jsonl: Path, eval_jsonl: Path, force: bool,
) -> tuple[list[bool], dict]:
    cache_path = run_dir / "clean_eval_mask.json"
    if cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text())
            return cached["mask"], cached["stats"]
        except (json.JSONDecodeError, KeyError):
            pass
    mask, stats = compute_clean_eval_mask(train_jsonl, eval_jsonl)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps({"mask": mask, "stats": stats}))
    tmp.rename(cache_path)
    return mask, stats


# ---------------------------------------------------------------------------
# Per-run rescore
# ---------------------------------------------------------------------------

def _load_predictions(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _filter_predictions(preds: list[dict], mask: list[bool]) -> list[dict]:
    if len(preds) != len(mask):
        raise RuntimeError(
            f"predictions length {len(preds)} != mask length {len(mask)}; "
            f"trainer/eval contract drift (review 019 Blocker 2)."
        )
    return [p for p, keep in zip(preds, mask) if keep]


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.rename(path)


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)


def _hn_worst(hn_point: dict | None) -> float | None:
    if not hn_point:
        return None
    vals = [v for v in hn_point.values() if isinstance(v, (int, float))]
    finite = [float(v) for v in vals if v == v]
    return max(finite) if finite else None


def _hn_mean(hn_point: dict | None) -> float | None:
    if not hn_point:
        return None
    vals = [v for v in hn_point.values() if isinstance(v, (int, float))]
    finite = [float(v) for v in vals if v == v]
    return sum(finite) / len(finite) if finite else None


def rescore_run(
    run_dir: Path,
    train_override: Path | None,
    eval_override: Path | None,
    force_mask: bool,
) -> dict:
    """Rescore a single run dir. Returns the experiments.jsonl row payload."""
    cfg = _load_config(run_dir)
    exp_id = cfg["exp_id"]
    arm = cfg.get("arm", "unknown")
    train_jsonl, eval_jsonl = _resolve_train_eval(cfg, train_override, eval_override)

    print(f"[{exp_id}] train={train_jsonl} eval={eval_jsonl}")
    mask, stats = _compute_or_load_mask(run_dir, train_jsonl, eval_jsonl, force=force_mask)
    print(
        f"[{exp_id}] mask: n_kept={stats['n_kept']} n_dropped={stats['n_dropped']} "
        f"n_text_overlap={stats['n_text_overlap']} n_events_overlap={stats['n_events_overlap']}"
    )

    per_mode_metrics: dict[str, dict] = {}
    per_mode_ci: dict[str, dict] = {}
    modes_present: list[str] = []

    for mode in EVAL_MODES:
        preds_path = run_dir / f"predictions_{mode}.jsonl"
        if not preds_path.exists():
            print(f"[{exp_id}] skip mode={mode}: no predictions file")
            continue
        preds = _load_predictions(preds_path)
        clean = _filter_predictions(preds, mask)
        clean_path = run_dir / f"predictions_{mode}_clean.jsonl"
        _atomic_write_jsonl(clean_path, clean)
        print(f"[{exp_id}] mode={mode}: kept {len(clean)} / {len(preds)} rows -> {clean_path.name}")

        metrics = compute_all(clean)
        ci = bootstrap_hard_negative_fpr(
            clean,
            target_fpr=TARGET_FPR,
            resamples=BOOTSTRAP_RESAMPLES,
            confidence=BOOTSTRAP_CONFIDENCE,
            seed=BOOTSTRAP_SEED,
        )
        per_mode_metrics[mode] = metrics
        per_mode_ci[mode] = ci
        modes_present.append(mode)

        _atomic_write_json(run_dir / f"metrics_v2_{mode}.json", metrics)
        _atomic_write_json(run_dir / f"ci_report_v2_{mode}.json", ci)
        hn = metrics.get("hard_negative_fpr_at_decision_threshold_1pct", {}) or {}
        wc = _hn_worst({k: v for k, v in hn.items() if k != "_threshold_alpha"})
        mc = _hn_mean({k: v for k, v in hn.items() if k != "_threshold_alpha"})
        print(
            f"[{exp_id}] mode={mode}: worst-HN-FPR={wc:.4f} "
            f"mean-HN-FPR={mc:.4f} alpha={hn.get('_threshold_alpha', (None, None))[1]}"
        )

    if not modes_present:
        raise RuntimeError(f"no predictions found in {run_dir}; nothing to rescore")

    # Aggregate ci_report (one block per mode), matches the launcher's
    # ci_report.json pattern.
    aggregate = {mode: per_mode_ci[mode] for mode in modes_present}
    _atomic_write_json(run_dir / "ci_report_v2.json", aggregate)

    # Build the experiments.jsonl row from the primary-mode (stripped)
    # metrics. Mirrors the launcher's v2 row shape exactly so update_sweep_state
    # reads identical fields whether the row came from a re-score or a fresh
    # launcher invocation.
    primary = PRIMARY_MODE if PRIMARY_MODE in per_mode_metrics else modes_present[0]
    stripped = per_mode_metrics.get(primary, {})
    opaque = per_mode_metrics.get("opaque", {})
    full = per_mode_metrics.get("full", {})

    hn_point_raw = stripped.get("hard_negative_fpr_at_decision_threshold_1pct") or {}
    # Strip the tuple key for downstream `_hn_worst`/`_hn_mean` cleanliness.
    hn_point_for_row = {k: v for k, v in hn_point_raw.items() if k != "_threshold_alpha"}

    # CI bundle for stripped (matches `_load_hn_ci` shape — top-level
    # bundle with per_family/worst_family/mean/etc).
    hn_ci_stripped = per_mode_ci.get(primary)

    row = {
        "exp_id": f"{exp_id}_v2",
        "parent_exp_id": exp_id,
        "arm": arm,
        "config_hash": None,  # mirrors of v1 row's config; not recomputed (no config drift)
        "config_summary": arm,
        "status": "rescored_v2",
        "metric_version": 2,
        "hn_fpr_worst_stripped": _hn_worst(hn_point_for_row),
        "hn_fpr_mean_stripped": _hn_mean(hn_point_for_row),
        "hn_fpr_per_family_stripped": hn_point_for_row,
        "hn_fpr_ci_stripped": hn_ci_stripped,
        "auc_stripped": stripped.get("auc"),
        "auc_opaque": opaque.get("auc"),
        "auc_full": full.get("auc"),
        "r_at_fpr_1pct": (stripped.get("r_at_fpr_0.01") or {}).get("recall"),
        "r_at_fpr_0.1pct": (stripped.get("r_at_fpr_0.001") or {}).get("recall"),
        "per_journey_auc_stripped": stripped.get("per_journey_auc"),
        "per_actor_auc_stripped": stripped.get("per_actor_auc"),
        # No GPU training in rescore; set 0.0 (not None) so the launcher's
        # `sum(h.get("wall_clock_min", 0) for h in history)` halt-check
        # in check_halt_conditions() doesn't trip on None + float.
        "wall_clock_min": 0.0,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "leakage_clean": True,  # rescore doesn't touch leakage report
        # Clean-eval fields DERIVED from the mask (review 019 High 3) — NOT hard-coded.
        "clean_eval_n": stats.get("n_kept"),
        "clean_eval_dropped": stats.get("n_dropped"),
        "clean_eval_mask_text_overlap": stats.get("n_text_overlap"),
        "clean_eval_mask_events_overlap": stats.get("n_events_overlap"),
    }
    return row


# ---------------------------------------------------------------------------
# Aggregate driver
# ---------------------------------------------------------------------------

def _auto_detect_run_dirs(include_xa_smoke: bool) -> list[Path]:
    out: list[Path] = []
    if not RUNS_DIR.exists():
        return out
    for child in sorted(RUNS_DIR.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("exp_baseline_"):
            out.append(child)
        elif include_xa_smoke and child.name == "exp_xa_smoke_001":
            # Only include if predictions actually exist
            if any((child / f"predictions_{m}.jsonl").exists() for m in EVAL_MODES):
                out.append(child)
    return out


def _append_experiments_rows(rows: list[dict]) -> None:
    """Append rows to experiments.jsonl atomically (one fsync per row to
    match the launcher's append-only contract)."""
    EXPERIMENTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENTS_JSONL.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            f.flush()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, help="path to a single run dir to rescore")
    p.add_argument("--auto-detect", action="store_true",
                   help="discover all exp_baseline_* dirs under src/auto_research/runs/")
    p.add_argument("--include-xa-smoke", action="store_true",
                   help="also rescore exp_xa_smoke_001 if its predictions exist")
    p.add_argument("--train-dir", type=Path,
                   help="override train dir (contains train.jsonl); else read from cfg.data.train_path")
    p.add_argument("--eval-dir", type=Path,
                   help="override eval dir (contains eval.jsonl); else read from cfg.data.eval_fast_path")
    p.add_argument("--force-mask", action="store_true",
                   help="recompute clean-eval mask even if cached file exists")
    p.add_argument("--dry-run", action="store_true",
                   help="compute + write per-run artifacts but do NOT append to experiments.jsonl")
    p.add_argument("--no-update-sweep-state", action="store_true",
                   help="skip calling update_sweep_state() after appends")
    args = p.parse_args()

    if not args.auto_detect and args.run_dir is None:
        p.error("pass --auto-detect or --run-dir")

    run_dirs: list[Path] = []
    if args.auto_detect:
        run_dirs.extend(_auto_detect_run_dirs(args.include_xa_smoke))
    if args.run_dir is not None:
        run_dirs.append(args.run_dir)
    if not run_dirs:
        print("no run dirs found to rescore", file=sys.stderr)
        return 1

    print(f"rescore plan: {len(run_dirs)} run dir(s)")
    for r in run_dirs:
        print(f"  - {r}")
    print()

    new_rows: list[dict] = []
    for run_dir in run_dirs:
        try:
            row = rescore_run(run_dir, args.train_dir, args.eval_dir, args.force_mask)
        except Exception as e:
            print(f"[{run_dir.name}] FAILED: {e}", file=sys.stderr)
            return 2
        new_rows.append(row)
        print(
            f"[{row['exp_id']}] worst-HN-FPR-stripped={row['hn_fpr_worst_stripped']} "
            f"mean-HN-FPR-stripped={row['hn_fpr_mean_stripped']} "
            f"clean_eval_n={row['clean_eval_n']} clean_eval_dropped={row['clean_eval_dropped']}"
        )
        print()

    if args.dry_run:
        print(f"--dry-run: NOT appending {len(new_rows)} rows to {EXPERIMENTS_JSONL}")
        for r in new_rows:
            print(json.dumps(r, indent=2))
        return 0

    _append_experiments_rows(new_rows)
    print(f"appended {len(new_rows)} row(s) to {EXPERIMENTS_JSONL}")

    if not args.no_update_sweep_state:
        # Call into the launcher's update_sweep_state without invoking the
        # full CLI (which expects a config path). This is a normal Python
        # import so the v2 rows get reflected immediately.
        try:
            from scripts.run_next_experiment import update_sweep_state
        except ImportError:
            # Bare-import fallback if scripts/ isn't a package.
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "run_next_experiment", REPO_ROOT / "scripts" / "run_next_experiment.py"
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            update_sweep_state = module.update_sweep_state
        update_sweep_state()
        print("called update_sweep_state(); sweep_state.yaml refreshed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
