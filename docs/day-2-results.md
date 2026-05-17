# Day-2 results — durable evidence

This document is the durable, in-repo record of Day-2 (2026-05-17). It parallels `docs/day-1-results.md`. All numbers cited below are reproducible from `src/auto_research/experiments.jsonl` and the on-disk run dirs under `src/auto_research/runs/exp_baseline_*/` and `src/auto_research/runs/exp_xa_smoke_001/`. The "v2" rows referenced throughout were produced by `scripts/rescore_baselines.py` after the baseline-evaluation-correction plan landed; no model weights changed during Day-2's second half — only the evaluation was corrected.

---

## 1. What we set out to measure

Day-1 trained the Stage-0 CPT-light checkpoint and surfaced a known synthetic-data ceiling: AUC and R@FPR=0.1% saturated at 1.0 on every LLM-narrated eval surface (see `docs/day-1-results.md` §3, §5). The README's Day-1 wrap-up explicitly pivoted the Day-2/3 comparison metric to **worst-family hard-negative FPR at 1% legit FPR** (the metric definition lives in `eval/score_risk.py::compute_all`, the per-family fan-out in `hard_negative_fpr`, and bootstrap CIs in `eval/bootstrap_ci.py`). Day-2's first half launched four baselines against that headline metric: `event_only` (a small structured-event classifier), `lora_text` (LoRA adapter on Qwen3-8B narrative input), `structured_as_text` (LoRA adapter on a textualized rendering of the structured stream), and a 100-step `xattn` smoke (`exp_xa_smoke_001`) to keep the architecture warm. The intent was: pick a leader among the baselines, then dispatch Task #37 (full x-attn training) against the leader as the comparison target.

What actually happened on Day-2's second half was that the apparent leaderboard didn't survive scrutiny. A pre-dispatch Codex review (`review/018-day-2-baseline-findings/comments.txt`) confirmed three findings — one in the data, one in the metric implementation, one in the synthetic distribution itself — each of which would have steered Task #37 against an invalid comparison surface. The rest of this document describes the apparent leaderboard, the three findings, the corrected leaderboard, and what the corrections mean for the rest of the POC. The corrections do not require retraining: the model weights and predictions on disk are the same; only the evaluation pipeline changed.

## 2. Apparent leaderboard, before correction

Initial Day-2 launch ranked the three baselines as the launcher's `update_sweep_state` would have, using the original `eval/score_risk.py::recall_at_fpr` (sklearn's "largest achievable FPR ≤ target" threshold rule) on the full 5,000-row eval. The raw values from `src/auto_research/experiments.jsonl` (v1 rows):

| Rank | exp_id                              | Arm                | Worst HN-FPR (stripped, v1) |
|------|-------------------------------------|--------------------|------------------------------|
| 1    | `exp_baseline_event_only`           | event_only         | **0.00820** (`hn_account_recovery`) |
| 2    | `exp_baseline_structured_as_text`   | structured_as_text | 0.04508 (`hn_account_recovery`) |
| 3    | `exp_baseline_lora_text`            | lora_text          | 0.05444 (`hn_large_purchase`) |

The headline reading of this table was "`event_only` crushes the LM baselines by ~5-7x." That is what the launcher's ranking surface produced; it is what the auto-loop would have optimized against. It is, however, an artifact. Codex's verification reproduced the underlying number — `event_only` reaches `hn_account_recovery` FPR = 0.819% at the sklearn-chosen threshold — but did so by demonstrating that the LM baselines were being measured at a materially different operating point on the legit-FPR axis. The next section unpacks what went wrong.

## 3. Three findings during baseline scrutiny

These are documented as findings rather than fixes because each is a structural property of the data, the metric, or the synthetic generator — not a transient bug.

### Finding 1 — train/eval leakage (Blocker)

10.7% of eval rows have identical narrative text to a train row (`review/018-day-2-baseline-findings/comments.txt:150`, reproduced locally by `scripts/diagnose_data_overlap.py --check`). Concentrations are family-specific: `hn_large_purchase` 35.3% (175/496), `hn_account_recovery` 16.4% (80/488), `clean` 13.2% (267/2019), `malware_rat` 3.7% (11/296), all others 0%. Every leaked row also matches a train row on `structured_events`-hash, with one extra row whose structured payload matches train but whose narrative differs in whitespace/LLM-output variance — bringing the total drop count to 534 (`review/018-day-2-baseline-findings/comments.txt:160-162`).

The mechanism is in `data/gen/build_dataset.py::main()`: narration runs before the train/eval split, and `data/gen/narrative_generator.py::_journey_cache_key` (line 252) caches narratives by `(structured_events_hash, model, temp)`. When the generator emits the same structured-event payload twice, the second call hits the cache; the post-narration split can then place identical narrative text on both sides. This is documented in greater depth in `docs/day-2-data-diagnostic.md`, which the script `scripts/diagnose_data_overlap.py` produces. The same diagnostic reports a bucket-event skeleton overlap of 4,661/5,000 eval rows (93.2%) — a separate, deeper property of the synthetic distribution discussed in Finding 3.

The fix-pass corrected this by computing a clean-eval mask (`eval/leakage_checks.py::compute_clean_eval_mask`) that drops the 534 leaked rows, and by adding a pre-narration structured-events-hash stratification gate plus a post-narration text-hash dedup invariant to `data/gen/build_dataset.py` so future regenerations cannot reintroduce the leak.

### Finding 2 — sklearn-cliff metric discretization (Blocker)

The original `eval/score_risk.py::recall_at_fpr` (line 47, sklearn-based) used "largest achievable FPR ≤ target." On models with bimodal score distributions — `event_only` converged to train loss 1e-5 and produces large tied score masses — that rule lands different models at materially different achieved legit FPRs even though all three claim "the 1% operating point." Codex's verification (`review/018-day-2-baseline-findings/comments.txt:184-221`) showed:

- `event_only`: sklearn threshold = -8.484, achieved FPR = **0.114%** (not 1%) → reported worst HN-FPR = 0.820%.
- `lora_text`: sklearn threshold = -3.625, achieved FPR = 0.971% → reported worst HN-FPR = 5.44%.
- `structured_as_text`: sklearn threshold = -9.625, achieved FPR = 0.914% → reported worst HN-FPR = 4.51%.

`event_only` was being graded at roughly a tenth of the legit-FPR budget the LM baselines were given. A naive percentile-of-legit replacement (score >= threshold) has the same cliff in a different form: tied scores at the boundary make `event_only` jump to 4.14% legit FPR and `hn_account_recovery` 29.7% (`comments.txt:195`).

The correct fix is a tie-aware exact-target legit-FPR. The new `recall_at_fpr` in `eval/score_risk.py` walks descending scores until the cumulative legit count hits exactly `target_fpr * n_legit` (kept as a float, not rounded — review 019 Blocker 1), then computes an `alpha` = fraction of tied-at-threshold rows that need to be allocated to hit exactly the target FPR. `hard_negative_fpr` then weights tied rows by `alpha` instead of including or excluding them wholesale. The new function emits `(threshold, alpha, achieved_fpr, n_above, n_tied, tie_fraction)` alongside the point estimate so the operating point is verifiable from the JSON; `eval/bootstrap_ci.py::bootstrap_hard_negative_fpr` recomputes `(threshold, alpha)` per resample. With this metric, the apparent 5-7x advantage of `event_only` collapses to a narrow ordering with overlapping CIs across all three baselines — see §4.

### Finding 3 — synthetic data is label-deterministic in the observed support (High)

This is not a code bug; it is a property of the synthetic generator. Across the entire dataset, `H(label | journey_family) = 0` and `H(label | bucket-event skeleton) = 0` over 2,454 distinct skeletons, with zero mixed-label skeletons (`comments.txt:163-175`; reproduced by `scripts/diagnose_data_overlap.py`). The per-family bucket-combination space is structurally small: `hn_large_purchase` has only 12 distinct skeletons across 2,481 rows (saturation 0.005), `hn_account_recovery` has 31 across 2,442 (saturation 0.013), `sim_swap` has 46 across 1,459 (saturation 0.032). 4,661 of 5,000 eval rows share a bucket-event skeleton with a train row — separate from the 534 narrative-text/events-hash leaks. The per-family table is in `docs/day-2-data-diagnostic.md`.

PLAN.md §3 deliberately puts fraud signal into bucketed derived-feature tokens (privacy-safe design), and PLAN.md §Risks explicitly flagged "synthetic-data saturation (all baselines AUC ceiling)" as a possible finding. What is stronger than that prior framing is that the structured stream is *label-deterministic* in the observed support — the `event_only` classifier is not discovering a subtle behavioral signal, it is learning a compact categorical mapping over a stream where the label is mechanically derivable. The LM baselines and the x-attn variants read the same distribution through a harder interface (text → score), so `event_only`'s performance on this synthetic surface is not a clean proxy for "the LM is unnecessary in the real ATO domain."

This is documented as a POC finding and as a constraint on Day-3 claims. The data pipeline guardrail in `data/gen/build_dataset.py` (Task #6 of the correction plan) closes the narrative-cache reuse mechanism for future regenerations; it does NOT remove skeleton-level overlap, which is a property of the bucket-combination space, not the cache. Removing skeleton overlap would require either enlarging the per-family bucket space or holding out a skeleton-disjoint eval set — both are future-Day-N deliverables, not Day-2 scope.

## 4. The corrected leaderboard

Same predictions, recomputed under (a) the clean eval mask (drops 534 rows; `n_clean = 4466`) and (b) the tie-aware exact-target metric. Raw values from `src/auto_research/experiments.jsonl` (v2 rows; produced by `scripts/rescore_baselines.py`):

| Rank | exp_id                                | Arm                | Worst-family HN-FPR @ 1% legit FPR (point, 95% CI) | Per-family (point) |
|------|---------------------------------------|--------------------|----------------------------------------------------|--------------------|
| 1    | `exp_baseline_structured_as_text_v2`  | structured_as_text | **0.05067** [0.04078, 0.06349] (`hn_account_recovery`) | acct_rec=0.0507, lg_purchase=0.0267, travel=0.0011 |
| 2    | `exp_xa_smoke_001_v2`                 | xattn (100-step smoke) | **0.05366** [0.04277, 0.06536] (`hn_account_recovery`) | acct_rec=0.0537, lg_purchase=0.0246, travel=0.0000 |
| 3    | `exp_baseline_lora_text_v2`           | lora_text          | **0.07014** [0.05635, 0.08468] (`hn_large_purchase`)   | acct_rec=0.0135, lg_purchase=0.0701, travel=0.0035 |
| 4    | `exp_baseline_event_only_v2`          | event_only         | **0.07301** [0.06667, 0.07989] (`hn_account_recovery`) | acct_rec=0.0730, lg_purchase=0.0000, travel=0.0000 |

All four rows carry `metric_version: 2`, `clean_eval_n: 4466`, `clean_eval_dropped: 534`, `clean_eval_mask_text_overlap: 533`, `clean_eval_mask_events_overlap: 534`, and `tie_fraction_point`, `achieved_fpr_point`, `threshold_point`, `alpha_point` so the operating point is fully verifiable. Bootstrap configuration: 1000 resamples, 95% confidence.

Three things to read out of this table:

1. **`structured_as_text` leads, not `event_only`.** This confirms Codex review 018 §4 prediction. The apparent 5-7x advantage of `event_only` from §2 was an artifact of the sklearn-threshold cliff; under the tie-aware metric on the clean eval, `event_only` is *last*, not first. The synthetic data structurally favors event-only models (Finding 3), and yet the corrected metric reveals that on this comparison surface the LM-based variants outperform on worst-family HN-FPR.

2. **The 100-step x-attn smoke is already competitive.** `exp_xa_smoke_001_v2` lands at 5.37%, between `structured_as_text` (5.07%) and `lora_text` (7.01%). This is a 100-step smoke, not a Day-3 result — but it sets a useful prior: cross-attention does not appear to be a regression at the smoke scale. The gap to `structured_as_text` is ~0.003 absolute, with substantially overlapping CIs ([0.04078, 0.06349] vs [0.04277, 0.06536]).

3. **All CIs overlap.** No baseline is separable from its neighbor by 95% bootstrap CI on worst-family HN-FPR:
   - `structured_as_text` [0.04078, 0.06349] overlaps with `xattn_smoke` [0.04277, 0.06536].
   - `xattn_smoke` upper end (0.06536) overlaps with `lora_text` lower end (0.05635).
   - `lora_text` [0.05635, 0.08468] overlaps with `event_only` [0.06667, 0.07989].
   - `structured_as_text` upper end (0.06349) does NOT overlap with `event_only` lower end (0.06667) — the only non-overlapping pair, and even that gap is small.

The narrow gaps + overlapping CIs are themselves a deliverable finding (see §5).

Per-family worst-HN-FPR breakdown with CIs (clean eval, n=4466, stripped, tie-aware exact-target):

| exp_id                  | hn_account_recovery        | hn_large_purchase          | hn_travel                  |
|-------------------------|----------------------------|----------------------------|----------------------------|
| `event_only_v2`         | 0.07301 [0.06667, 0.07989] | 0.00000 [0.00000, 0.00000] | 0.00000 [0.00000, 0.00000] |
| `lora_text_v2`          | 0.01352 [0.00440, 0.02394] | 0.07014 [0.05635, 0.08468] | 0.00353 [0.00000, 0.00831] |
| `structured_as_text_v2` | 0.05067 [0.03852, 0.06349] | 0.02666 [0.01217, 0.04180] | 0.00112 [0.00000, 0.00443] |
| `xa_smoke_001_v2`       | 0.05366 [0.04207, 0.06536] | 0.02460 [0.01074, 0.03960] | 0.00000 [0.00000, 0.00000] |

Two things are visible in the per-family table that the worst-family rollup hides:

- `event_only` collapses to a single hard-negative family. It scores 0.0730 on `hn_account_recovery` and a clean 0.0000 on both `hn_large_purchase` and `hn_travel`. The structured stream cleanly separates `hn_large_purchase`/`hn_travel` from legit, but the event-only-trained classifier's discrimination boundary lands `hn_account_recovery` rows directly at the tie-bucket boundary (point `alpha = 0.183`, `tie_fraction = 0.047` — exactly the review-019 Blocker-1 hallmark). This is the structural reason the sklearn-cliff metric mis-measured `event_only` in §2: its score distribution has a large tied mass right where the 1%-legit-FPR threshold lands, so any non-tie-aware rule lands at the wrong operating point.
- `lora_text` loses on `hn_large_purchase` (0.0701), and that loss is the worst-family for that baseline. Text alone cannot separate legitimate large purchases from `hn_large_purchase` without the bucket-token structured fields, so this is consistent with the design intent in PLAN.md §3 (bucket tokens carry the actual fraud signal). `structured_as_text` is the most balanced row in the table — no family explodes, no family is exactly zero — and that balance is what makes its worst-family the lowest.

Tie-aware operating-point diagnostics (point values from `experiments.jsonl` v2 rows; bootstrap means are within 1e-3 of the points, see `hn_fpr_ci_stripped` for resample averages):

| exp_id                  | threshold | alpha   | tie_fraction | achieved_fpr |
|-------------------------|-----------|---------|--------------|--------------|
| `event_only_v2`         | -9.9375   | 0.18291 | 0.04733      | 0.01000      |
| `lora_text_v2`          | -3.7500   | 0.75800 | 0.00168      | 0.01000      |
| `structured_as_text_v2` | -9.6250   | 0.55800 | 0.00168      | 0.01000      |
| `xa_smoke_001_v2`       | -8.0000   | 0.89500 | 0.00067      | 0.01000      |

Each row achieves the target legit-FPR of exactly 0.01 by allocating `alpha` of the tied-at-threshold rows. `event_only`'s `tie_fraction = 4.7%` is an order of magnitude higher than every other row's (~0.07-0.17%) and is what made it pathologically sensitive to the old sklearn cliff. The other three rows are well-behaved under either metric; their CIs just shift slightly under tie-aware allocation.

For comparison, the FULL-eval (n=5000) tie-aware numbers (also in `review/018-day-2-baseline-findings/comments.txt:231-233`, confirmed by the rescore script):

| exp_id                  | FULL worst HN-FPR (n=5000) | CLEAN worst HN-FPR (n=4466) | Δ      |
|-------------------------|----------------------------|------------------------------|--------|
| `event_only_v2`         | 0.0717                     | 0.0730                       | +0.0013 |
| `lora_text_v2`          | 0.0556                     | 0.0701                       | +0.0145 |
| `structured_as_text_v2` | 0.0497                     | 0.0507                       | +0.0010 |

The leaderboard ordering is the same on the leaky and clean evals; the magnitudes shift because the clean eval removes the family-concentrated leaks reported in §3 Finding 1. `event_only`'s clean > full is small because its concentrated `hn_account_recovery` hits get amplified when the easy-to-classify leaked rows are removed. `structured_as_text`'s shift is similarly small for the same reason. `lora_text`'s clean >> full (0.0701 vs 0.0556) is the most dramatic: the family's apparent edge on FULL was largely measuring memorization of the 175 leaked `hn_large_purchase` rows. This is itself evidence that ranking against the FULL eval would have rewarded narrative-memorization over generalization.

## 5. What this means for the rest of the POC

The POC's Day-3 framing changes shape. The original framing was a single-number win: "does cross-attention beat the strongest baseline on worst-family HN-FPR with non-overlapping CIs?" The corrected Day-2 surface tells us three things that re-scope Day-3:

**The single-number-win framing is too brittle to be the headline.** The strongest baseline (`structured_as_text`) is at 5.07% [0.04078, 0.06349]; the 100-step x-attn smoke is already within that CI. A full-train x-attn would need to land with a worst-family CI whose upper bound is below 0.04078 to claim a non-overlapping win — a worst-family delta of >1% absolute on `hn_account_recovery`. That is achievable but not guaranteed; the smoke trend is in the right direction but the smoke is also high-variance at 100 steps. A "no measurable improvement" outcome is equally informative on this synthetic surface, given Finding 3.

**The deliverable is methodology + findings, not a leaderboard.** Three findings have already landed as Day-2 record material: leakage on synthetic data is mechanical and preventable (Finding 1, fixed in the data pipeline); metric implementation under tied scores is a first-class concern that the launcher must encode explicitly (Finding 2, fixed in `eval/score_risk.py` + `eval/bootstrap_ci.py` + `scripts/run_next_experiment.py`); a privacy-safe bucketed-feature design produces a label-deterministic structured stream in observed support that biases naive interpretations of "event-only wins" (Finding 3, documented as a synthetic-data property). Each of these generalizes to any future POC that uses similar synthetic-data scaffolding, and each is independently citable.

**The sharp x-attn question for Day-3.** "Can a fully-trained cross-attention model (Task #37) beat `structured_as_text` (5.07%) with non-overlapping bootstrap CIs on worst-family HN-FPR @ 1%, computed against the clean eval (n=4466) under the tie-aware exact-target metric, on this synthetic distribution?" The answer in either direction is a deliverable finding. A clean "yes" is the POC's strongest possible outcome; a clean "no" is the second-strongest, because it bounds the value of cross-attention on this particular synthetic surface and motivates the harder eval (skeleton-disjoint hold-out, mixed-label hard negatives) as the next-step requirement for a real-data POC.

Task #38 (auto-loop) and Task #37 (x-attn) proceed from here against the corrected ranking surface: `update_sweep_state` filters to `metric_version >= 2`, the launcher applies the clean-eval mask automatically for every new run (`scripts/run_next_experiment.py::run_post_processing`), and `sweep_state.yaml` reports `schema_version: 2` with a `metric_definition` block describing the tie-aware exact-target semantics. The Day-2 close state is reproducible from this repo without pod access; predictions remain on the pod under each run dir.

---

## Reproducibility

```bash
# Reproduce the data-overlap diagnostic (Finding 1 + Finding 3 numbers).
python3 scripts/diagnose_data_overlap.py \
    --data-dir data/train_llm_narrated \
    --check \
    --write-md docs/day-2-data-diagnostic.md

# Reproduce the clean-eval mask (Finding 1 invariant).
python3 -m eval.leakage_checks --train-eval-overlap data/train_llm_narrated
# Expected: drops exactly 534 rows (533 text + 1 events-only-hash).

# Self-test the tie-aware metric (Finding 2 fix).
python3 -m eval.score_risk --selftest
python3 -m eval.bootstrap_ci --selftest

# Re-derive the v2 leaderboard.
python3 scripts/rescore_baselines.py --auto-detect
# Re-emits exp_baseline_*_v2 + exp_xa_smoke_001_v2 rows; idempotent.
```

The `experiments.jsonl` cited above is at `src/auto_research/experiments.jsonl`; the v2 rows are the last four. The full `hn_fpr_ci_stripped` bundle on each v2 row contains `per_family`, `worst_family`, `mean`, `tie_fraction_mean`, `achieved_fpr_mean`, `threshold_mean`, `alpha_mean`, plus the corresponding `_point` fields. Per-run artifacts (`metrics_v2_<mode>.json`, `ci_report_v2_<mode>.json`, `predictions_<mode>_clean.jsonl`, `clean_eval_mask.json`) live under each run dir on the pod and rsync down on request.

> **Note on `sweep_state.yaml` schema bump.** At the time this writeup landed, the on-pod `sweep_state.yaml` still reads `schema_version: 1`; pipeline-eng's Task #5 schema-bump is committed in repo but the next pod-sync will overwrite the on-pod copy to `schema_version: 2` with the `metric_definition` block. The numerical leaderboard above does not depend on this — `update_sweep_state`'s ranking filter is computed at read-time from `metric_version` per row, not from the YAML header.

## How this maps to README.md Day-2 claims

| README claim | Evidence above |
|---|---|
| Apparent leaderboard had `event_only` winning at 0.820% worst HN-FPR | §2 |
| Tie-aware metric reverses the ranking | §3 Finding 2 + §4 |
| Clean eval drops 534 rows (533 text + 1 events-only) | §3 Finding 1 |
| `structured_as_text` leads at 5.07% [0.04078, 0.06349] | §4 |
| x-attn 100-step smoke at 5.37% [0.04277, 0.06536] | §4 |
| All four baselines have overlapping CIs except `structured_as_text` vs `event_only` | §4 |
| Synthetic data is label-deterministic in observed support (H(label\|skeleton)=0 over 2,454 skeletons) | §3 Finding 3 + `docs/day-2-data-diagnostic.md` |
| Day-3 framing: methodology + findings, not single-number win | §5 |
| Sharp x-attn question for Task #37 | §5 |

All numerical claims in README.md Day-2 trace back to one of the sections above.
