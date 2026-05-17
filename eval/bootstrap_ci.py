"""Bootstrap confidence intervals for AUC and Recall@FPR.

The POC's "win" claims must be CI-defended: no Δ<0.005 AUC delta drives a
Day-4 decision without non-overlapping CIs. This module is called by
`run_next_experiment.py` after metric computation.

CLI:
    python eval/bootstrap_ci.py --predictions PRED.jsonl --metrics METRICS.json \\
        --out CI_REPORT.json --resamples 1000 --confidence 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .score_risk import LABEL_TO_INT, _load_predictions, recall_at_fpr
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
    """Bootstrap CI for Recall@FPR."""
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
    """Bootstrap CI for hard-negative FPR per family at decision threshold
    corresponding to target_fpr.

    For each resample:
      1. Resample predictions with replacement.
      2. Compute the score threshold that achieves target_fpr on the
         resample (i.e., the threshold consistent with this sample's
         distribution).
      3. Per hard-negative family, compute fraction wrongly above that
         threshold.

    Returns:
      {
        "target_fpr": 0.01,
        "per_family": {
          "hn_account_recovery": {"point", "ci_lo", "ci_hi", ...},
          "hn_large_purchase":   {...},
          "hn_travel":           {...}
        },
        "worst_family": {"point", "ci_lo", "ci_hi", ...},  # max across families per resample
        "mean":         {"point", "ci_lo", "ci_hi", ...},  # mean across families per resample
        "resamples": 1000,
        "confidence": 0.95,
      }

    Review 013 finding #2: HN-FPR is the Day-3 win condition; AUC and
    R@FPR are saturated, so the launcher's leader/halt logic now ranks
    on worst-family HN-FPR (primary) and mean HN-FPR (tiebreaker). Both
    composites get a CI here so Day-3 "non-overlapping CIs" claims are
    supportable.
    """
    from eval.score_risk import hard_negative_fpr, recall_at_fpr

    rng = np.random.default_rng(seed)
    n = len(predictions)

    # Point estimates from the full sample
    point_threshold = recall_at_fpr(predictions, target_fpr)["threshold"]
    point_per_family = (
        hard_negative_fpr(predictions, point_threshold)
        if not np.isnan(point_threshold) else {}
    )
    family_order = sorted(point_per_family.keys())

    # Bootstrap: per-family FPRs + worst/mean composites
    boot_per_family: dict[str, list[float]] = {k: [] for k in family_order}
    boot_worst: list[float] = []
    boot_mean: list[float] = []

    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        sub = [predictions[j] for j in idx]
        th = recall_at_fpr(sub, target_fpr)["threshold"]
        if np.isnan(th):
            for k in family_order:
                boot_per_family[k].append(float("nan"))
            boot_worst.append(float("nan"))
            boot_mean.append(float("nan"))
            continue
        hn = hard_negative_fpr(sub, th)
        per = [float(hn.get(k, float("nan"))) for k in family_order]
        for k, v in zip(family_order, per):
            boot_per_family[k].append(v)
        # Composites — ignore NaN entries (resample may not contain a
        # given family); if all NaN, composite is NaN.
        per_clean = [v for v in per if not np.isnan(v)]
        boot_worst.append(max(per_clean) if per_clean else float("nan"))
        boot_mean.append(sum(per_clean) / len(per_clean) if per_clean else float("nan"))

    alpha = (1.0 - confidence) / 2.0

    def _ci(samples: list[float], point: float) -> dict:
        arr = np.asarray(samples, dtype=np.float64)
        return {
            "point": float(point) if not np.isnan(point) else float("nan"),
            "ci_lo": float(np.nanpercentile(arr, 100 * alpha)),
            "ci_hi": float(np.nanpercentile(arr, 100 * (1.0 - alpha))),
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
        "per_family": per_family,
        "worst_family": _ci(boot_worst, point_worst),
        "mean": _ci(boot_mean, point_mean),
        "resamples": resamples,
        "confidence": confidence,
    }


def ci_overlap(a: dict, b: dict) -> bool:
    """True iff confidence intervals overlap. Used to decide 'win' claims."""
    return not (a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fpr-targets", default="0.001,0.01,0.05")
    args = p.parse_args()

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
    # Review 013 finding #2: HN-FPR is now the Day-3 win condition.
    # Always bootstrap at the 1% FPR threshold (Day-1 pivot consensus).
    report["hard_negative_fpr_at_1pct"] = bootstrap_hard_negative_fpr(
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
