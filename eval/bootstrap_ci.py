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

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.rename(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
