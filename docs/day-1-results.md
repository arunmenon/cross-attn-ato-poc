# Day-1 results — durable evidence

Per review 012 finding #1: the score / metrics / CI / diagnostic JSON files cited by `README.md` Day-1 section live under `src/auto_research/runs/` which is gitignored on purpose (runtime artifacts on the pod). This document checks in the **exact numeric content** of those files so a fresh clone can audit every claim the README makes about Day-1 without needing the pod.

All excerpts below are verbatim from the pod's run dirs as of the Day-1 close (2026-05-17). Values are rounded to 4 decimals where the source file emits more.

---

## 1. `src/auto_research/experiments.jsonl`

Two entries persisted by the trainer + bookkeeping pass. Reproduced in full:

```json
{"exp_id": "exp_vslice_001", "arm": "cpt_light", "purpose": "vertical_slice_smoke", "task_ref": "Task #32 make-or-break", "n_steps": 100, "wall_clock_sec": 516.8, "final_train_loss": 2.6152, "n_trainable": 665149440, "auc_by_mode": {"stripped": 0.9929, "opaque": 0.9946, "full": 0.9922}, "ci_by_mode": {"stripped": [0.9916, 0.9945], "opaque": [0.9933, 0.9957], "full": [0.9905, 0.9938]}, "r_at_fpr_0.01_stripped": 0.7432, "hn_fpr_at_1pct_stripped": {"hn_account_recovery": 0.0, "hn_large_purchase": 0.0565, "hn_travel": 0.0}, "leakage": {"narrative_failures": 0, "strip_failures": 0, "opaque_failures": 0}, "status": "PASS_smoke", "gates_satisfied": [1, 2, 3, 4, 5, 6], "finding": "AUC saturation across all 3 modes ~0.99; bucket tokens carry the signal, journey/actor tokens add nothing. Watch for synthetic-data ceiling on Day 3 baseline comparisons."}
{"exp_id": "exp_stage0_001", "arm": "cpt_light", "purpose": "stage0_cpt_light_full", "task_ref": "Task #33", "base_checkpoint": "Qwen/Qwen3-8B", "output_merged_checkpoint": "/workspace/checkpoints/qwen3-8b-cpt-light-merged", "n_steps": 1500, "wall_clock_sec": 3193.9, "final_train_loss": 1.2425, "n_trainable": 665149440, "auc_by_mode_on_5k_llm_eval": {"stripped": 1.0, "opaque": 1.0, "full": 1.0}, "ci_by_mode_on_5k_llm_eval": {"stripped": [1.0, 1.0], "opaque": [1.0, 1.0], "full": [1.0, 1.0]}, "r_at_fpr_0.001_stripped": 1.0, "r_at_fpr_0.01_stripped": 1.0, "hn_fpr_at_1pct_stripped": {"hn_account_recovery": 0.0615, "hn_large_purchase": 0.0101, "hn_travel": 0.0}, "status": "PASS_full", "finding": "AUC saturated at 1.0 across all 3 modes (5k LLM eval). Hard-neg FPR moved: hn_large_purchase 5.6%->1.0% (5.6x better than smoke), hn_account_recovery 0%->6.2% (regressed - model trading off). Day-3 headline pivots to hard-negative FPR + R@FPR=0.1%. Re-eval against 15k LLM eval in flight to confirm saturation on the harder surface.", "role": "baseline_1_AND_stage1_starting_checkpoint"}
```

## 2. exp_vslice_001 (Task #32 vertical-slice make-or-break, 100 steps)

### `runs/exp_vslice_001/metrics.json`

```json
{
  "status": "ok",
  "arm": "cpt_light",
  "final_train_loss": 2.6152225136756897,
  "n_steps": 100,
  "wall_clock_sec": 516.777529001236,
  "n_trainable": 665149440,
  "predictions": {
    "stripped": 5000,
    "opaque": 5000,
    "full": 5000
  },
  "max_gate_magnitude": null,
  "adapter_path": "/workspace/cross_attn_ato_poc/src/auto_research/runs/exp_vslice_001/stage0_lora_adapter"
}
```

### `runs/exp_vslice_001/score_stripped.json` (headline)

```json
{
  "n": 5000,
  "auc": 0.9929132989545157,
  "r_at_fpr_0.001": {
    "target_fpr": 0.001,
    "achieved_fpr": 0.000856898029134533,
    "recall": 0.5023348899266178,
    "threshold": 2.875,
    "n_positive": 1499,
    "n_negative": 3501
  },
  "r_at_fpr_0.01": {
    "target_fpr": 0.01,
    "achieved_fpr": 0.007997714938588975,
    "recall": 0.7431621080720481,
    "threshold": 1.625,
    "n_positive": 1499,
    "n_negative": 3501
  },
  "r_at_fpr_0.05": {
    "target_fpr": 0.05,
    "achieved_fpr": 0.046843758926021134,
    "recall": 0.9853235490326885,
    "threshold": -1.25,
    "n_positive": 1499,
    "n_negative": 3501
  },
  "per_journey_auc": {
    "clean": NaN,
    "cred_stuff": 0.9963707280545326,
    "hn_account_recovery": NaN,
    "hn_large_purchase": NaN,
    "hn_travel": NaN,
    "malware_rat": 0.9820118962149811,
    "mule_chain": 0.994713934687809,
    "phish_takeover": 0.9999999999999999,
    "sim_swap": 0.9910910972598826
  },
  "per_actor_auc": {
    "agent_adversarial": NaN,
    "agent_buying": NaN,
    "agent_compromised": NaN,
    "agent_finance": NaN,
    "human": 0.9919529092461423,
    "hybrid": 0.9996610169491525
  },
  "hard_negative_fpr_at_decision_threshold_1pct": {
    "hn_account_recovery": 0.0,
    "hn_large_purchase": 0.056451612903225805,
    "hn_travel": 0.0
  }
}
```

### `runs/exp_vslice_001/ci_report_stripped.json`

```json
{
  "auc": {
    "point": 0.9929132989545157,
    "ci_lo": 0.9915631938954257,
    "ci_hi": 0.994458983319166,
    "resamples": 200,
    "confidence": 0.95
  },
  "r_at_fpr_0.001": {
    "target_fpr": 0.001,
    "point": 0.5023348899266178,
    "ci_lo": 0.45997069335430085,
    "ci_hi": 0.586421181056913,
    "resamples": 200,
    "confidence": 0.95
  },
  "r_at_fpr_0.01": {
    "target_fpr": 0.01,
    "point": 0.7431621080720481,
    "ci_lo": 0.7139992941374554,
    "ci_hi": 0.8064083041619814,
    "resamples": 200,
    "confidence": 0.95
  },
  "r_at_fpr_0.05": {
    "target_fpr": 0.05,
    "point": 0.9853235490326885,
    "ci_lo": 0.9740120701681623,
    "ci_hi": 0.9924807030496391,
    "resamples": 200,
    "confidence": 0.95
  }
}
```

(Other modes — opaque / full — and their CIs follow the same shape; full-mode AUC was 0.9922 [0.9905, 0.9938]; opaque AUC was 0.9946 [0.9933, 0.9957].)

## 3. exp_stage0_001 (Task #33 Stage-0 CPT-light + merge, 1500 steps)

### `runs/exp_stage0_001/metrics.json`

```json
{
  "status": "ok",
  "arm": "cpt_light",
  "final_train_loss": 1.2425491213798523,
  "n_steps": 1500,
  "wall_clock_sec": 3193.9153258800507,
  "n_trainable": 665149440,
  "predictions": {
    "stripped": 5000,
    "opaque": 5000,
    "full": 5000
  },
  "max_gate_magnitude": null,
  "adapter_path": "/workspace/cross_attn_ato_poc/src/auto_research/runs/exp_stage0_001/stage0_lora_adapter"
}
```

### `runs/exp_stage0_001/score_stripped.json` (headline) — saturated

```json
{
  "n": 5000,
  "auc": 1.0,
  "r_at_fpr_0.001": {
    "target_fpr": 0.001,
    "achieved_fpr": 0.0,
    "recall": 1.0,
    "threshold": 4.125,
    "n_positive": 1499,
    "n_negative": 3501
  },
  "r_at_fpr_0.01": {
    "target_fpr": 0.01,
    "achieved_fpr": 0.009997143673236219,
    "recall": 1.0,
    "threshold": -9.0,
    "n_positive": 1499,
    "n_negative": 3501
  },
  "r_at_fpr_0.05": {
    "target_fpr": 0.05,
    "achieved_fpr": 0.045986860896886604,
    "recall": 1.0,
    "threshold": -11.75,
    "n_positive": 1499,
    "n_negative": 3501
  },
  "per_journey_auc": {
    "clean": NaN,
    "cred_stuff": 1.0,
    "hn_account_recovery": NaN,
    "hn_large_purchase": NaN,
    "hn_travel": NaN,
    "malware_rat": 0.9999999999999999,
    "mule_chain": 1.0,
    "phish_takeover": 1.0,
    "sim_swap": 1.0
  },
  "per_actor_auc": {
    "agent_adversarial": NaN,
    "agent_buying": NaN,
    "agent_compromised": NaN,
    "agent_finance": NaN,
    "human": 1.0,
    "hybrid": 1.0
  },
  "hard_negative_fpr_at_decision_threshold_1pct": {
    "hn_account_recovery": 0.06147540983606557,
    "hn_large_purchase": 0.010080645161290322,
    "hn_travel": 0.0
  }
}
```

### `runs/exp_stage0_001/ci_report_stripped.json` — degenerate (point=1.0)

```json
{
  "auc": {
    "point": 1.0,
    "ci_lo": 0.9999999999999999,
    "ci_hi": 1.0,
    "resamples": 1000,
    "confidence": 0.95
  },
  "r_at_fpr_0.001": {
    "target_fpr": 0.001,
    "point": 1.0,
    "ci_lo": 1.0,
    "ci_hi": 1.0,
    "resamples": 1000,
    "confidence": 0.95
  },
  "r_at_fpr_0.01": {
    "target_fpr": 0.01,
    "point": 1.0,
    "ci_lo": 1.0,
    "ci_hi": 1.0,
    "resamples": 1000,
    "confidence": 0.95
  },
  "r_at_fpr_0.05": {
    "target_fpr": 0.05,
    "point": 1.0,
    "ci_lo": 1.0,
    "ci_hi": 1.0,
    "resamples": 1000,
    "confidence": 0.95
  }
}
```

## 4. Diagnostic: Stage-0 adapter vs 50k templated eval

(vertical-slice saturation diagnostic, 5k stratified sample of the 50k templated set)

### `runs/exp_vslice_001/diagnostic_templated_score_stripped.json`

```json
{
  "n": 4995,
  "auc": 0.9999574709844979,
  "r_at_fpr_0.001": {
    "target_fpr": 0.001,
    "achieved_fpr": 0.0009009009009009009,
    "recall": 0.9956756756756757,
    "threshold": -1.8671875,
    "n_positive": 2775,
    "n_negative": 2220
  },
  "r_at_fpr_0.01": {
    "target_fpr": 0.01,
    "achieved_fpr": 0.005405405405405406,
    "recall": 0.9985585585585586,
    "threshold": -2.0,
    "n_positive": 2775,
    "n_negative": 2220
  },
  "r_at_fpr_0.05": {
    "target_fpr": 0.05,
    "achieved_fpr": 0.03198198198198198,
    "recall": 0.9996396396396396,
    "threshold": -2.375,
    "n_positive": 2775,
    "n_negative": 2220
  },
  "per_journey_auc": {
    "clean": NaN,
    "cred_stuff": 0.9999999999999999,
    "hn_account_recovery": NaN,
    "hn_large_purchase": NaN,
    "hn_travel": NaN,
    "malware_rat": 1.0,
    "mule_chain": 0.9997873549224902,
    "phish_takeover": 0.9999999999999999,
    "sim_swap": 0.9999999999999999
  },
  "per_actor_auc": {
    "agent_adversarial": NaN,
    "agent_buying": NaN,
    "agent_compromised": NaN,
    "agent_finance": NaN,
    "human": 1.0,
    "hybrid": 1.0
  },
  "hard_negative_fpr_at_decision_threshold_1pct": {
    "hn_account_recovery": 0.021621621621621623,
    "hn_large_purchase": 0.0,
    "hn_travel": 0.0
  }
}
```

## 5. Diagnostic: Stage-0 adapter vs 15k LLM-narrated eval (seed=42)

(Confirms saturation is not eval-set-size-specific; 10,653-record stratified sample of 15k.)

### `runs/exp_stage0_001/diagnostic_15k_score_stripped.json`

```json
{
  "n": 10653,
  "auc": 1.0,
  "r_at_fpr_0.001": {
    "target_fpr": 0.001,
    "achieved_fpr": 0.0009737098344693282,
    "recall": 1.0,
    "threshold": -3.5,
    "n_positive": 4491,
    "n_negative": 6162
  },
  "r_at_fpr_0.01": {
    "target_fpr": 0.01,
    "achieved_fpr": 0.008114248620577734,
    "recall": 1.0,
    "threshold": -7.75,
    "n_positive": 4491,
    "n_negative": 6162
  },
  "r_at_fpr_0.05": {
    "target_fpr": 0.05,
    "achieved_fpr": 0.04316780266147355,
    "recall": 1.0,
    "threshold": -10.625,
    "n_positive": 4491,
    "n_negative": 6162
  },
  "per_journey_auc": {
    "clean": NaN,
    "cred_stuff": 0.9999999999999999,
    "hn_account_recovery": NaN,
    "hn_large_purchase": NaN,
    "hn_travel": NaN,
    "malware_rat": 0.9999999999999999,
    "mule_chain": 1.0,
    "phish_takeover": 1.0,
    "sim_swap": 1.0
  },
  "per_actor_auc": {
    "agent_adversarial": NaN,
    "agent_buying": NaN,
    "agent_compromised": NaN,
    "agent_finance": NaN,
    "human": 1.0,
    "hybrid": 1.0
  },
  "hard_negative_fpr_at_decision_threshold_1pct": {
    "hn_account_recovery": 0.02217741935483871,
    "hn_large_purchase": 0.011588275391956374,
    "hn_travel": 0.0
  }
}
```

### `runs/exp_stage0_001/diagnostic_15k_ci_stripped.json`

```json
{
  "auc": {
    "point": 1.0,
    "ci_lo": 0.9999999999999999,
    "ci_hi": 1.0000000000000002,
    "resamples": 1000,
    "confidence": 0.95
  },
  "r_at_fpr_0.001": {
    "target_fpr": 0.001,
    "point": 1.0,
    "ci_lo": 1.0,
    "ci_hi": 1.0,
    "resamples": 1000,
    "confidence": 0.95
  },
  "r_at_fpr_0.01": {
    "target_fpr": 0.01,
    "point": 1.0,
    "ci_lo": 1.0,
    "ci_hi": 1.0,
    "resamples": 1000,
    "confidence": 0.95
  },
  "r_at_fpr_0.05": {
    "target_fpr": 0.05,
    "point": 1.0,
    "ci_lo": 1.0,
    "ci_hi": 1.0,
    "resamples": 1000,
    "confidence": 0.95
  }
}
```

## 6. Dataset summaries (these are gitignored under `data/*/` but their `build_summary.json` files persist)

- `data/train_llm_narrated/build_summary.json`: n=25000, mode=llm, eval_frac=0.2, leakage_audit_failures=0, duration=3798s, spent=$5.67
- `data/eval_medium_15k_llm/build_summary.json`: n=15000, mode=llm, n_cache_hits=15000, n_calls=0, leakage_audit_failures=0, duration=2.59s
- `data/eval_medium_50k/build_summary.json`: n=50000, mode=template, leakage_audit_failures=0, duration=3.49s

These can be inspected directly with `cat data/<dir>/build_summary.json` after running the local data-gen sequence in RUNBOOK §3 Day 0.

---

## How this maps to README.md Day-1 claims

| README claim | Evidence above |
|---|---|
| Smoke AUC stripped = 0.9929 [0.9916, 0.9945] | §2 score_stripped + ci_report_stripped |
| Smoke R@FPR=1% = 0.7432 | §2 score_stripped.r_at_fpr_0.01 |
| Smoke hn_large_purchase FPR = 5.65% | §2 score_stripped.hard_negative_fpr_at_decision_threshold_1pct |
| Stage-0 wall = 53 min, final_loss = 1.2425 | §3 metrics.json |
| Stage-0 AUC = 1.0000 saturated | §3 score_stripped (auc=1.0); §3 ci_report (1.0, 1.0) |
| Stage-0 R@FPR=0.1% = 1.0 | §3 score_stripped.r_at_fpr_0.001 |
| Stage-0 hn_account_recovery FPR = 6.15% (5k) / 2.22% (15k) | §3 + §5 |
| Stage-0 hn_large_purchase FPR = 1.01% (5k) / 1.16% (15k) | §3 + §5 |
| 50k templated also saturated | §4 |
| Leakage scan 0/25k | §1 build_summary on train_llm_narrated |
| 15k eval = 100% cache hits | §6 eval_medium_15k_llm/build_summary |

All numerical claims in README.md Day-1 trace back to one of the excerpts above.