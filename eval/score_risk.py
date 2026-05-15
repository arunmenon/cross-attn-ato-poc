"""Deterministic scoring: AUC, R@FPR, per-journey AUC, hard-negative FPR.

Single source of truth for what AUC means in this POC. The agent does not
get to redefine these — `run_next_experiment.py` calls these directly.

Input format: a list of dicts, one per example, each with:
    {
        "score": float,         # logP(fraud) - logP(legit), from verdict footer
        "label": "fraud" | "legit",
        "journey_family": str,  # for per-journey breakdown
        "actor_family": str,    # for per-actor differential
        "is_hard_negative": bool,
    }

CLI:
    python eval/score_risk.py --predictions PATH.jsonl --out METRICS.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

LABEL_TO_INT = {"fraud": 1, "legit": 0}


def _to_arrays(predictions: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.array([p["score"] for p in predictions], dtype=np.float64)
    labels = np.array([LABEL_TO_INT[p["label"]] for p in predictions], dtype=np.int64)
    return scores, labels


def auc(predictions: Sequence[dict]) -> float:
    """Standard ROC-AUC. Higher score = more likely fraud."""
    scores, labels = _to_arrays(predictions)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")  # degenerate; no positive or no negative
    return float(roc_auc_score(labels, scores))


def recall_at_fpr(predictions: Sequence[dict], target_fpr: float) -> dict:
    """Recall at a target FPR.

    Returns:
        {
            "target_fpr": float,
            "achieved_fpr": float,    # closest <= target on the ROC
            "recall": float,
            "threshold": float,
            "n_positive": int,
            "n_negative": int,
        }
    """
    scores, labels = _to_arrays(predictions)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {
            "target_fpr": target_fpr,
            "achieved_fpr": float("nan"),
            "recall": float("nan"),
            "threshold": float("nan"),
            "n_positive": n_pos,
            "n_negative": n_neg,
        }

    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Find largest fpr <= target.
    mask = fpr <= target_fpr
    if not mask.any():
        # No achievable point below target FPR — return recall at the lowest FPR.
        idx = 0
    else:
        idx = int(np.where(mask)[0].max())

    return {
        "target_fpr": float(target_fpr),
        "achieved_fpr": float(fpr[idx]),
        "recall": float(tpr[idx]),
        "threshold": float(thresholds[idx]) if idx < len(thresholds) else float("nan"),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


def per_journey_auc(predictions: Sequence[dict]) -> dict[str, float]:
    """AUC per `journey_family`. Compares fraud examples of this family vs
    legit examples of all families (so the AUC is interpretable as
    'does the model find THIS family of fraud').
    """
    families = sorted({p["journey_family"] for p in predictions})
    legit_preds = [p for p in predictions if p["label"] == "legit"]
    if not legit_preds:
        return {f: float("nan") for f in families}

    out: dict[str, float] = {}
    for family in families:
        fraud_preds = [p for p in predictions if p["label"] == "fraud" and p["journey_family"] == family]
        if not fraud_preds:
            out[family] = float("nan")
            continue
        subset = fraud_preds + legit_preds
        out[family] = auc(subset)
    return out


def per_actor_auc(predictions: Sequence[dict]) -> dict[str, float]:
    """AUC restricted to examples of a given actor_family (fraud + legit within that actor)."""
    actors = sorted({p["actor_family"] for p in predictions})
    out: dict[str, float] = {}
    for actor in actors:
        subset = [p for p in predictions if p["actor_family"] == actor]
        if not subset:
            out[actor] = float("nan")
            continue
        out[actor] = auc(subset)
    return out


def hard_negative_fpr(predictions: Sequence[dict], threshold: float) -> dict[str, float]:
    """At a given decision threshold, what fraction of hard negatives is misclassified as fraud?

    Reported per-family for the three `hn_*` families.
    """
    out: dict[str, float] = {}
    hn_families = sorted({
        p["journey_family"] for p in predictions
        if p["is_hard_negative"]
    })
    for family in hn_families:
        subset = [p for p in predictions if p["journey_family"] == family]
        if not subset:
            out[family] = float("nan")
            continue
        # All hard negatives are label=legit by construction.
        n_fp = sum(1 for p in subset if p["score"] >= threshold)
        out[family] = n_fp / len(subset)
    return out


def compute_all(predictions: Sequence[dict], target_fprs: Sequence[float] = (0.001, 0.01, 0.05)) -> dict:
    """Top-level metrics bundle. Called by run_next_experiment.py."""
    base_auc = auc(predictions)
    fpr_results = {f"r_at_fpr_{f}": recall_at_fpr(predictions, f) for f in target_fprs}
    # Decision threshold for HN-FPR: the threshold corresponding to FPR=1%.
    threshold_at_1pct = fpr_results["r_at_fpr_0.01"]["threshold"]
    hn = hard_negative_fpr(predictions, threshold_at_1pct) if not np.isnan(threshold_at_1pct) else {}

    return {
        "n": len(predictions),
        "auc": base_auc,
        **fpr_results,
        "per_journey_auc": per_journey_auc(predictions),
        "per_actor_auc": per_actor_auc(predictions),
        "hard_negative_fpr_at_decision_threshold_1pct": hn,
    }


def _load_predictions(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, type=Path, help="jsonl of per-example predictions")
    p.add_argument("--out", required=True, type=Path, help="output metrics.json path")
    p.add_argument("--fpr-targets", default="0.001,0.01,0.05", help="comma-separated target FPRs")
    args = p.parse_args()

    preds = _load_predictions(args.predictions)
    target_fprs = tuple(float(x) for x in args.fpr_targets.split(","))
    metrics = compute_all(preds, target_fprs=target_fprs)

    # Atomic write
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(metrics, indent=2))
    tmp.rename(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
