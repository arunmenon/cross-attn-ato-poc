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
    python -m eval.score_risk --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

LABEL_TO_INT = {"fraud": 1, "legit": 0}


def _to_arrays(predictions: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.array([p["score"] for p in predictions], dtype=np.float64)
    labels = np.array([LABEL_TO_INT[p["label"]] for p in predictions], dtype=np.int64)
    return scores, labels


def auc(predictions: Sequence[dict]) -> float:
    """Standard ROC-AUC. Higher score = more likely fraud."""
    scores, labels = _to_arrays(predictions)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _nan_record(target_fpr: float, n_pos: int, n_neg: int) -> dict:
    return {
        "target_fpr": float(target_fpr),
        "achieved_fpr": float("nan"),
        "threshold": float("nan"),
        "alpha": float("nan"),
        "n_above": 0,
        "n_tied": 0,
        "need": float("nan"),
        "tie_fraction": float("nan"),
        "recall": float("nan"),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
    }


def recall_at_fpr(predictions: Sequence[dict], target_fpr: float) -> dict:
    """Tie-aware exact-target legit-FPR (review 018/019).

    Walks legit scores in descending order; finds threshold T where the
    running count of (legit > T) <= need <= count(legit > T) + count(legit == T).
    Returns alpha in [0, 1] s.t.
        achieved_fpr = (n_above + alpha * n_tied) / n_legit == target_fpr.

    The formula uses FRACTIONAL need = target_fpr * n_legit (NOT rounded).
    Rounding fails the 1e-4 acceptance on small clean-eval n_legit
    (review 019 Blocker 1).

    Returns:
        {
            "target_fpr", "achieved_fpr", "threshold", "alpha",
            "n_above", "n_tied", "need", "tie_fraction",
            "recall", "n_positive", "n_negative"
        }
    """
    scores, labels = _to_arrays(predictions)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return _nan_record(target_fpr, n_pos, n_neg)

    legit_scores = scores[labels == 0]
    fraud_scores = scores[labels == 1]
    n_legit = int(len(legit_scores))

    need = float(target_fpr) * n_legit  # FLOAT — do NOT round (review 019 B1)

    # Sort legit scores descending for clarity; use unique to find candidate
    # thresholds where the cumulative legit count crosses `need`.
    sorted_legit = np.sort(legit_scores)[::-1]
    # cumulative count of legit > T as we walk the unique descending values:
    # at unique value v_k, n_above(v_k) = count(legit > v_k), n_tied(v_k) = count(legit == v_k).
    unique_desc, counts = np.unique(sorted_legit, return_counts=True)
    # np.unique returns ascending; reverse:
    unique_desc = unique_desc[::-1]
    counts = counts[::-1]

    # Find smallest k such that cumulative count after this group >= need.
    # i.e., sum(counts[:k]) + counts[k] >= need; threshold T = unique_desc[k];
    # n_above = sum(counts[:k]); n_tied = counts[k].
    cum = np.cumsum(counts)
    # n_above at index k is cum[k-1] if k>0 else 0
    # cum[k] >= need finds the bucket that first crosses need.
    idx_candidates = np.where(cum >= need)[0]
    if len(idx_candidates) == 0:
        # need exceeds n_legit (target_fpr > 1.0?) — clamp to last bucket
        k = len(counts) - 1
    else:
        k = int(idx_candidates[0])

    threshold = float(unique_desc[k])
    n_above = int(cum[k - 1]) if k > 0 else 0
    n_tied = int(counts[k])

    # Edge case: need exactly satisfied by n_above alone (e.g., need is integer
    # and equals cum[k-1]). In that case, the boundary lands at the gap between
    # bucket k-1 and bucket k. Conventionally, set T to the value at index k-1
    # (the lowest legit value still above), n_above = need_int, n_tied = 0, alpha = 0.
    # But here we keep tie-aware behavior consistent: alpha is the fraction of
    # the tied bucket required to reach `need`.
    if n_tied > 0 and need > n_above:
        alpha = (need - n_above) / n_tied
    else:
        alpha = 0.0
    # Numerical safety: clamp alpha to [0, 1].
    if alpha < 0.0:
        alpha = 0.0
    elif alpha > 1.0:
        alpha = 1.0

    achieved_fpr = (n_above + alpha * n_tied) / n_legit
    tie_fraction = n_tied / n_legit if n_legit > 0 else float("nan")

    # Recall at this threshold (tie-aware): fraction of fraud rows scoring
    # above T, plus alpha-weighted fraction of fraud rows tied AT T.
    # The decision rule is "score >= T + 0 (with prob alpha for ties)"; in
    # expectation, alpha-weighting matches the legit side.
    n_fraud_above = int((fraud_scores > threshold).sum())
    n_fraud_tied = int((fraud_scores == threshold).sum())
    recall = (n_fraud_above + alpha * n_fraud_tied) / n_pos if n_pos > 0 else float("nan")

    return {
        "target_fpr": float(target_fpr),
        "achieved_fpr": float(achieved_fpr),
        "threshold": threshold,
        "alpha": float(alpha),
        "n_above": int(n_above),
        "n_tied": int(n_tied),
        "need": float(need),
        "tie_fraction": float(tie_fraction),
        "recall": float(recall),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


def per_journey_auc(predictions: Sequence[dict]) -> dict[str, float]:
    """AUC per `journey_family`. Compares fraud examples of this family vs
    legit examples of all families.
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


def hard_negative_fpr(
    predictions: Sequence[dict],
    threshold: float,
    alpha: float = 0.0,
) -> dict:
    """Tie-aware HN-FPR per `hn_*` family.

    At a decision threshold T (with tie-weight alpha for legit rows scoring
    EXACTLY at T), what fraction of each hard-negative family is misclassified
    as fraud?

    Tie-aware rule (matches the legit-FPR construction in `recall_at_fpr`):
        count_for_fam = sum(score > T) + alpha * sum(score == T)
        fpr_for_fam   = count_for_fam / n_in_fam

    Returns:
        {
            "<family_name>": float,                 # per-family fraction
            ...,
            "_threshold_alpha": (threshold, alpha), # for downstream verifiability
        }

    Note: all hard negatives are label=legit by construction, so every row in
    the subset is a legit row for FPR computation.
    """
    out: dict[str, float] = {}
    hn_families = sorted({
        p["journey_family"] for p in predictions
        if p.get("is_hard_negative")
    })
    if math.isnan(threshold):
        # Degenerate; return NaN per family
        for family in hn_families:
            out[family] = float("nan")
        out["_threshold_alpha"] = (float("nan"), float("nan"))
        return out

    a = float(alpha) if not (alpha is None or math.isnan(alpha)) else 0.0
    if a < 0.0:
        a = 0.0
    elif a > 1.0:
        a = 1.0

    for family in hn_families:
        subset = [p for p in predictions if p["journey_family"] == family]
        if not subset:
            out[family] = float("nan")
            continue
        n_above = sum(1 for p in subset if p["score"] > threshold)
        n_tied = sum(1 for p in subset if p["score"] == threshold)
        out[family] = (n_above + a * n_tied) / len(subset)

    out["_threshold_alpha"] = (float(threshold), float(a))
    return out


def compute_all(predictions: Sequence[dict], target_fprs: Sequence[float] = (0.001, 0.01, 0.05)) -> dict:
    """Top-level metrics bundle. Called by run_next_experiment.py.

    metric_version: 2 — tie-aware exact-target legit-FPR with alpha
    interpolation (review 018/019). Historical files (metric_version absent
    or == 1) carry the sklearn-largest-FPR-≤-target semantics under the
    same key names.
    """
    base_auc = auc(predictions)
    fpr_results = {f"r_at_fpr_{f}": recall_at_fpr(predictions, f) for f in target_fprs}
    # Decision (threshold, alpha) for HN-FPR: the pair corresponding to FPR=1%.
    one_pct = fpr_results.get("r_at_fpr_0.01") or recall_at_fpr(predictions, 0.01)
    threshold_at_1pct = one_pct.get("threshold", float("nan"))
    alpha_at_1pct = one_pct.get("alpha", 0.0)
    if isinstance(threshold_at_1pct, float) and math.isnan(threshold_at_1pct):
        hn = {}
    else:
        hn = hard_negative_fpr(predictions, threshold_at_1pct, alpha=alpha_at_1pct)

    return {
        "n": len(predictions),
        "metric_version": 2,
        "auc": base_auc,
        **fpr_results,
        "per_journey_auc": per_journey_auc(predictions),
        "per_actor_auc": per_actor_auc(predictions),
        # Canonical key carries tie-aware semantics from this commit forward.
        "hard_negative_fpr_at_decision_threshold_1pct": hn,
        # Explicit alias under the tie_aware name for unambiguous downstream reads.
        "hard_negative_fpr_at_decision_threshold_1pct_tie_aware": hn,
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


# ---------------------------------------------------------------------------
# Selftest fixture (review 019 Blocker 1: fractional `need`, alpha=0.18293...)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    """Synthetic tie-cliff fixture; assert the tie-aware math.

    Build a fixture with n_legit = 100, target_fpr = 0.01 ⇒ need = 1.0
    (integer-need branch); then a fractional-need fixture with n_legit = 2979,
    need = 29.79, mimicking the event_only clean-eval shape from review 018.
    """
    # --- Fixture A: tie-bucket lands exactly at need -------------------------
    # n_legit = 200, need = 2.0. Construct: 0 strictly above T=5.0, 5 tied at
    # T=5.0, rest below. cum after the tied bucket = 5 (>= 2.0), so threshold
    # lands at T=5.0 with n_above = 0, n_tied = 5, alpha = 2/5 = 0.4.
    rng = np.random.default_rng(0)
    legit_tied = [5.0] * 5
    legit_below = list(rng.uniform(-5, 0, size=195))
    fraud_scores = [7.0, 7.0, 7.0, 5.0, -1.0]
    preds = []
    for s in legit_tied + legit_below:
        preds.append({"score": float(s), "label": "legit", "journey_family": "p2p", "actor_family": "human", "is_hard_negative": False})
    for s in fraud_scores:
        preds.append({"score": float(s), "label": "fraud", "journey_family": "p2p", "actor_family": "human", "is_hard_negative": False})

    r = recall_at_fpr(preds, 0.01)
    assert r["n_above"] == 0, f"A: n_above expected 0, got {r['n_above']}"
    assert r["n_tied"] == 5, f"A: n_tied expected 5, got {r['n_tied']}"
    assert abs(r["alpha"] - 0.4) < 1e-12, f"A: alpha expected 0.4, got {r['alpha']}"
    assert abs(r["achieved_fpr"] - 0.01) < 1e-12, f"A: achieved={r['achieved_fpr']}"
    assert abs(r["threshold"] - 5.0) < 1e-12, f"A: threshold expected 5.0, got {r['threshold']}"
    print(f"selftest A pass: n_above={r['n_above']} n_tied={r['n_tied']} alpha={r['alpha']:.6f} achieved={r['achieved_fpr']:.6f}")

    # --- Fixture B: fractional need (review 019 Blocker 1) -------------------
    # n_legit = 2979 ⇒ need = 29.79. Place 4 strictly above T = -9.9375,
    # 141 tied at -9.9375, rest below. Expected alpha = (29.79 - 4) / 141.
    n_legit = 2979
    n_above = 4
    n_tied = 141
    T = -9.9375
    legit_scores = (
        [T + 1.0] * n_above
        + [T] * n_tied
        + list(np.linspace(T - 1.0, T - 5.0, n_legit - n_above - n_tied))
    )
    fraud_scores = [T + 0.5] * 200  # plenty; recall not the focus here
    preds = []
    for s in legit_scores:
        preds.append({"score": float(s), "label": "legit", "journey_family": "p2p", "actor_family": "human", "is_hard_negative": False})
    for s in fraud_scores:
        preds.append({"score": float(s), "label": "fraud", "journey_family": "p2p", "actor_family": "human", "is_hard_negative": False})

    r = recall_at_fpr(preds, 0.01)
    expected_need = 29.79
    expected_alpha = (expected_need - n_above) / n_tied  # 0.182907...
    assert r["n_above"] == n_above, f"B: n_above {r['n_above']} != {n_above}"
    assert r["n_tied"] == n_tied, f"B: n_tied {r['n_tied']} != {n_tied}"
    assert abs(r["need"] - expected_need) < 1e-9, f"B: need {r['need']} != {expected_need}"
    assert abs(r["alpha"] - expected_alpha) < 1e-9, f"B: alpha {r['alpha']} != {expected_alpha}"
    assert abs(r["achieved_fpr"] - 0.01) < 1e-9, f"B: achieved {r['achieved_fpr']} != 0.01"
    assert abs(r["threshold"] - T) < 1e-12, f"B: threshold {r['threshold']} != {T}"
    print(
        f"selftest B pass: need={r['need']:.4f} n_above={r['n_above']} "
        f"n_tied={r['n_tied']} alpha={r['alpha']:.6f} achieved={r['achieved_fpr']:.6f} "
        f"threshold={r['threshold']:.4f}"
    )

    # --- Fixture C: HN-FPR alpha weighting ------------------------------------
    # Add a hard-negative family with rows at and above T to verify the
    # alpha-weighted count.
    hn_preds = list(preds)
    # 3 hn legit strictly above T; 10 hn legit tied at T; 87 below.
    for s in [T + 0.1] * 3 + [T] * 10 + list(np.linspace(T - 0.5, T - 5.0, 87)):
        hn_preds.append({"score": float(s), "label": "legit", "journey_family": "hn_test", "actor_family": "human", "is_hard_negative": True})
    rr = recall_at_fpr(hn_preds, 0.01)
    hn = hard_negative_fpr(hn_preds, rr["threshold"], alpha=rr["alpha"])
    # Per-family fraction: (3 + alpha * 10) / 100
    expected_hn = (3 + rr["alpha"] * 10) / 100
    assert abs(hn["hn_test"] - expected_hn) < 1e-9, f"C: hn={hn['hn_test']} != {expected_hn}"
    print(f"selftest C pass: hn_test={hn['hn_test']:.6f} expected={expected_hn:.6f}")

    print("selftest OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, help="jsonl of per-example predictions")
    p.add_argument("--out", type=Path, help="output metrics.json path")
    p.add_argument("--fpr-targets", default="0.001,0.01,0.05", help="comma-separated target FPRs")
    p.add_argument("--selftest", action="store_true", help="run inline selftest and exit")
    args = p.parse_args()

    if args.selftest:
        return _selftest()

    if args.predictions is None or args.out is None:
        p.error("--predictions and --out are required (or pass --selftest)")

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
