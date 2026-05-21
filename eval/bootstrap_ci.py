"""Bootstrap confidence intervals for AUC and Recall@FPR.

The POC's "win" claims must be CI-defended: no Δ<0.005 AUC delta drives a
Day-4 decision without non-overlapping CIs. This module is called by
`run_next_experiment.py` after metric computation.

Review 018/019: HN-FPR bootstrap is tie-aware. Per resample we recompute
`(threshold, alpha)` via the tie-aware `recall_at_fpr`, then weight tied-at-
threshold rows by alpha in `hard_negative_fpr`.

CLI:
    python eval/bootstrap_ci.py --predictions PRED.jsonl --metrics METRICS.json \\
        --out CI_REPORT.json --resamples 1000 --confidence 0.95
    python -m eval.bootstrap_ci --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .score_risk import LABEL_TO_INT, _load_predictions, recall_at_fpr, v5_adversarial_metrics
from sklearn.metrics import roc_auc_score


def _bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def bootstrap_auc(
    predictions: Sequence[dict],
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for ROC-AUC.

    Stratified by label so that resamples preserve class balance — important
    at the low base rates we have (fraud ~30%).
    """
    rng = np.random.default_rng(seed)
    scores = np.array([p["score"] for p in predictions], dtype=np.float64)
    labels = np.array([LABEL_TO_INT[p["label"]] for p in predictions], dtype=np.int64)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "resamples": 0}

    point = float(roc_auc_score(labels, scores))

    boot_aucs = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([sampled_pos, sampled_neg])
        try:
            boot_aucs[i] = roc_auc_score(labels[idx], scores[idx])
        except ValueError:
            boot_aucs[i] = np.nan

    alpha = (1.0 - confidence) / 2.0
    ci_lo = float(np.nanpercentile(boot_aucs, 100 * alpha))
    ci_hi = float(np.nanpercentile(boot_aucs, 100 * (1.0 - alpha)))

    return {
        "point": point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "resamples": resamples,
        "confidence": confidence,
    }


def bootstrap_recall_at_fpr(
    predictions: Sequence[dict],
    target_fpr: float,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for tie-aware Recall@FPR.

    Per resample we recompute (threshold, alpha, recall) via the new
    `recall_at_fpr`. Returned recall is the tie-aware one
    ((fraud_above + alpha * fraud_tied) / n_fraud).
    """
    rng = np.random.default_rng(seed)
    n = len(predictions)
    point = recall_at_fpr(predictions, target_fpr)["recall"]

    boot_recalls = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        sub = [predictions[j] for j in idx]
        boot_recalls[i] = recall_at_fpr(sub, target_fpr)["recall"]

    alpha = (1.0 - confidence) / 2.0
    ci_lo = float(np.nanpercentile(boot_recalls, 100 * alpha))
    ci_hi = float(np.nanpercentile(boot_recalls, 100 * (1.0 - alpha)))

    return {
        "target_fpr": target_fpr,
        "point": float(point) if not np.isnan(point) else float("nan"),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "resamples": resamples,
        "confidence": confidence,
    }


def bootstrap_hard_negative_fpr(
    predictions: Sequence[dict],
    target_fpr: float = 0.01,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Tie-aware bootstrap CI for hard-negative FPR per family.

    For each resample:
      1. Resample predictions with replacement.
      2. Compute (threshold, alpha) on the resample via tie-aware
         `recall_at_fpr` (review 018/019 — alpha-weighted ties at T).
      3. Per hard-negative family, compute
         (sum(score > T) + alpha * sum(score == T)) / n_in_family.

    Returns:
      {
        "target_fpr": 0.01,
        "metric_version": 2,
        "per_family": {
          "hn_account_recovery": {"point", "ci_lo", "ci_hi", ...},
          "hn_large_purchase":   {...},
          "hn_travel":           {...}
        },
        "worst_family": {"point", "ci_lo", "ci_hi", ...},  # max across families per resample
        "mean":         {"point", "ci_lo", "ci_hi", ...},  # mean across families per resample
        "tie_fraction_mean":   float,   # average tied-at-T fraction across resamples
        "achieved_fpr_mean":   float,   # average achieved_fpr across resamples (≈ target_fpr)
        "threshold_mean":      float,
        "alpha_mean":          float,
        "resamples": 1000,
        "confidence": 0.95,
      }
    """
    from eval.score_risk import hard_negative_fpr

    rng = np.random.default_rng(seed)
    n = len(predictions)

    # Point estimates from the full sample
    point_rec = recall_at_fpr(predictions, target_fpr)
    point_threshold = point_rec["threshold"]
    point_alpha = point_rec.get("alpha", 0.0)
    if isinstance(point_threshold, float) and math.isnan(point_threshold):
        point_per_family: dict[str, float] = {}
    else:
        hn_raw = hard_negative_fpr(predictions, point_threshold, alpha=point_alpha)
        point_per_family = {k: v for k, v in hn_raw.items() if k != "_threshold_alpha"}
    family_order = sorted(point_per_family.keys())

    # Bootstrap: per-family FPRs + worst/mean composites + tie-aware diagnostics
    boot_per_family: dict[str, list[float]] = {k: [] for k in family_order}
    boot_worst: list[float] = []
    boot_mean: list[float] = []
    boot_tie_fraction: list[float] = []
    boot_achieved_fpr: list[float] = []
    boot_threshold: list[float] = []
    boot_alpha: list[float] = []

    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        sub = [predictions[j] for j in idx]
        rec = recall_at_fpr(sub, target_fpr)
        th = rec["threshold"]
        al = rec.get("alpha", 0.0)
        tf = rec.get("tie_fraction", float("nan"))
        af = rec.get("achieved_fpr", float("nan"))
        boot_tie_fraction.append(float(tf))
        boot_achieved_fpr.append(float(af))
        boot_threshold.append(float(th))
        boot_alpha.append(float(al))
        if isinstance(th, float) and math.isnan(th):
            for k in family_order:
                boot_per_family[k].append(float("nan"))
            boot_worst.append(float("nan"))
            boot_mean.append(float("nan"))
            continue
        hn_raw = hard_negative_fpr(sub, th, alpha=al)
        hn = {k: v for k, v in hn_raw.items() if k != "_threshold_alpha"}
        per = [float(hn.get(k, float("nan"))) for k in family_order]
        for k, v in zip(family_order, per):
            boot_per_family[k].append(v)
        per_clean = [v for v in per if not np.isnan(v)]
        boot_worst.append(max(per_clean) if per_clean else float("nan"))
        boot_mean.append(sum(per_clean) / len(per_clean) if per_clean else float("nan"))

    pct_lo = (1.0 - confidence) / 2.0

    def _ci(samples: list[float], point: float) -> dict:
        arr = np.asarray(samples, dtype=np.float64)
        return {
            "point": float(point) if not np.isnan(point) else float("nan"),
            "ci_lo": float(np.nanpercentile(arr, 100 * pct_lo)),
            "ci_hi": float(np.nanpercentile(arr, 100 * (1.0 - pct_lo))),
            "resamples": resamples,
            "confidence": confidence,
        }

    per_family = {
        k: _ci(boot_per_family[k], point_per_family[k])
        for k in family_order
    }
    point_per = [point_per_family[k] for k in family_order if not np.isnan(point_per_family[k])]
    point_worst = max(point_per) if point_per else float("nan")
    point_mean = sum(point_per) / len(point_per) if point_per else float("nan")

    return {
        "target_fpr": target_fpr,
        "metric_version": 2,
        "per_family": per_family,
        "worst_family": _ci(boot_worst, point_worst),
        "mean": _ci(boot_mean, point_mean),
        "tie_fraction_mean": float(np.nanmean(boot_tie_fraction)),
        "achieved_fpr_mean": float(np.nanmean(boot_achieved_fpr)),
        "threshold_mean": float(np.nanmean(boot_threshold)),
        "alpha_mean": float(np.nanmean(boot_alpha)),
        # Point-estimate tie-aware diagnostics from the full sample:
        "threshold_point": float(point_threshold) if not (isinstance(point_threshold, float) and math.isnan(point_threshold)) else float("nan"),
        "alpha_point": float(point_alpha),
        "tie_fraction_point": float(point_rec.get("tie_fraction", float("nan"))),
        "achieved_fpr_point": float(point_rec.get("achieved_fpr", float("nan"))),
        "resamples": resamples,
        "confidence": confidence,
    }


def bootstrap_v5_adv_error(
    predictions: Sequence[dict],
    target_fpr: float = 0.01,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for the V5 adversarial composite.

    Per resample, recomputes the global tie-aware decision threshold at
    `target_fpr`, then recomputes the three V5 adversarial terms:
    phish takeover miss, MFA-phished phish takeover miss, and
    hn_recovery_high_amount FPR. The composite CI is the formal V5
    ranking CI; HN-FPR CI remains available under its existing key.
    """
    rng = np.random.default_rng(seed)
    n = len(predictions)
    point = v5_adversarial_metrics(predictions, target_fpr=target_fpr)

    boot_composite: list[float] = []
    boot_components: dict[str, list[float]] = {
        "phish_takeover_miss": [],
        "phish_takeover_mfa_phished_miss": [],
        "hn_recovery_high_amount_fpr": [],
    }

    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        sub = [predictions[j] for j in idx]
        m = v5_adversarial_metrics(sub, target_fpr=target_fpr)
        boot_composite.append(float(m["v5_adv_error"]))
        for key in boot_components:
            boot_components[key].append(float(m["components"].get(key, float("nan"))))

    pct_lo = (1.0 - confidence) / 2.0

    def _ci(samples: list[float], point_value: float) -> dict:
        arr = np.asarray(samples, dtype=np.float64)
        return {
            "point": float(point_value) if not np.isnan(point_value) else float("nan"),
            "ci_lo": float(np.nanpercentile(arr, 100 * pct_lo)),
            "ci_hi": float(np.nanpercentile(arr, 100 * (1.0 - pct_lo))),
            "resamples": resamples,
            "confidence": confidence,
        }

    component_ci = {
        key: _ci(samples, float(point["components"].get(key, float("nan"))))
        for key, samples in boot_components.items()
    }

    return {
        "target_fpr": float(target_fpr),
        "metric_version": 5,
        **_ci(boot_composite, float(point["v5_adv_error"])),
        "components": component_ci,
        "point_details": point,
    }


def ci_overlap(a: dict, b: dict) -> bool:
    """True iff confidence intervals overlap. Used to decide 'win' claims."""
    return not (a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"])


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _selftest() -> int:
    """Quick CI shape + sanity test on a synthetic fixture."""
    rng = np.random.default_rng(0)
    preds: list[dict] = []
    # Legit
    for s in rng.normal(-1.0, 1.5, size=400):
        preds.append({"score": float(s), "label": "legit", "journey_family": "p2p", "actor_family": "human", "is_hard_negative": False})
    # Fraud
    for s in rng.normal(1.5, 1.5, size=100):
        preds.append({"score": float(s), "label": "fraud", "journey_family": "p2p", "actor_family": "human", "is_hard_negative": False})
    # Hard negatives (two families)
    for s in rng.normal(0.5, 1.5, size=120):
        preds.append({"score": float(s), "label": "legit", "journey_family": "hn_a", "actor_family": "human", "is_hard_negative": True})
    for s in rng.normal(-0.2, 1.5, size=80):
        preds.append({"score": float(s), "label": "legit", "journey_family": "hn_b", "actor_family": "human", "is_hard_negative": True})
    # V5 adversarial families for composite-CI shape.
    for s in rng.normal(2.0, 1.0, size=40):
        preds.append({"score": float(s), "label": "fraud", "journey_family": "phish_takeover", "actor_family": "human", "is_hard_negative": False})
    for s in rng.normal(1.5, 1.0, size=30):
        preds.append({"score": float(s), "label": "fraud", "journey_family": "phish_takeover_mfa_phished", "actor_family": "human", "is_hard_negative": False})
    for s in rng.normal(0.0, 1.0, size=35):
        preds.append({"score": float(s), "label": "legit", "journey_family": "hn_recovery_high_amount", "actor_family": "human", "is_hard_negative": True})

    rep = bootstrap_hard_negative_fpr(preds, target_fpr=0.01, resamples=200, seed=0)

    # Shape checks
    for k in ("target_fpr", "metric_version", "per_family", "worst_family", "mean",
              "tie_fraction_mean", "achieved_fpr_mean", "threshold_mean", "alpha_mean",
              "threshold_point", "alpha_point", "tie_fraction_point", "achieved_fpr_point",
              "resamples", "confidence"):
        assert k in rep, f"missing key: {k}"
    assert rep["metric_version"] == 2
    assert set(rep["per_family"].keys()) == {"hn_a", "hn_b", "hn_recovery_high_amount"}
    for fam, ci in rep["per_family"].items():
        assert ci["ci_lo"] <= ci["point"] <= ci["ci_hi"], f"{fam} CI inverted: {ci}"
    wf = rep["worst_family"]
    assert wf["ci_lo"] <= wf["point"] <= wf["ci_hi"], f"worst CI inverted: {wf}"
    mn = rep["mean"]
    assert mn["ci_lo"] <= mn["point"] <= mn["ci_hi"], f"mean CI inverted: {mn}"

    # achieved_fpr_mean should be very close to target (resamples can wiggle a bit)
    assert abs(rep["achieved_fpr_mean"] - 0.01) < 5e-3, f"achieved_fpr_mean={rep['achieved_fpr_mean']}"

    # Also check bootstrap_recall_at_fpr still returns recall + threshold-like shape
    rec_ci = bootstrap_recall_at_fpr(preds, target_fpr=0.01, resamples=200, seed=0)
    assert "point" in rec_ci and "ci_lo" in rec_ci and "ci_hi" in rec_ci

    v5_ci = bootstrap_v5_adv_error(preds, target_fpr=0.01, resamples=200, seed=0)
    assert v5_ci["metric_version"] == 5
    assert "components" in v5_ci

    print(
        f"selftest pass: worst_point={wf['point']:.4f} "
        f"[{wf['ci_lo']:.4f}, {wf['ci_hi']:.4f}] "
        f"alpha_mean={rep['alpha_mean']:.4f} "
        f"tie_fraction_mean={rep['tie_fraction_mean']:.6f} "
        f"achieved_fpr_mean={rep['achieved_fpr_mean']:.6f}"
    )
    print("selftest OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fpr-targets", default="0.001,0.01,0.05")
    p.add_argument("--selftest", action="store_true", help="run inline selftest and exit")
    args = p.parse_args()

    if args.selftest:
        return _selftest()

    if args.predictions is None or args.out is None:
        p.error("--predictions and --out are required (or pass --selftest)")

    preds = _load_predictions(args.predictions)
    target_fprs = [float(x) for x in args.fpr_targets.split(",")]

    report = {
        "auc": bootstrap_auc(preds, resamples=args.resamples, confidence=args.confidence, seed=args.seed),
    }
    for f in target_fprs:
        report[f"r_at_fpr_{f}"] = bootstrap_recall_at_fpr(
            preds, target_fpr=f, resamples=args.resamples,
            confidence=args.confidence, seed=args.seed,
        )
    # HN-FPR is the Day-3 win condition; always bootstrap at 1% FPR.
    report["hard_negative_fpr_at_1pct"] = bootstrap_hard_negative_fpr(
        preds, target_fpr=0.01, resamples=args.resamples,
        confidence=args.confidence, seed=args.seed,
    )
    report["v5_adv_error"] = bootstrap_v5_adv_error(
        preds, target_fpr=0.01, resamples=args.resamples,
        confidence=args.confidence, seed=args.seed,
    )

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.rename(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
