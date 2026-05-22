# Whitepaper Review — Fifth Pass (`cross_attn_ato_poc/whitepaper/v1.1`)

**Reviewer pass · 2026-05-22**
Subject: V4 numeric correction applied. Documents the fix and the remaining unaddressed REVIEW-4 items.
Prior reviews: `REVIEW.md` → `REVIEW-2.md` → `REVIEW-3.md` → `REVIEW-4.md`.

---

## 1. Top-line

This pass **fixes REVIEW-4 §3.1 only** (V4 headline numbers against `ci_report.json`). The other five REVIEW-4 blockers remain open. The paper is not yet publish-ready, but the worst single-document factual gap — V4 hn_recovery_high_amount being reported as ~1.0 when the artifact says ~0.42 — is closed.

REVIEW-3's "publish-ready" verdict was wrong. REVIEW-4 was correct that the paper carries stale numbers; this pass closes the V4 subset of those numbers and leaves the rest documented for follow-up.

---

## 2. What was changed in this pass

All V4 quantitative claims now derive from the current `metric_version: 5` artifacts on disk:

- `src/auto_research/runs/exp_text_only_v4_001/ci_report.json`
- `src/auto_research/runs/exp_xattn_v4_001/ci_report.json`

### 2.1 Ground-truth values used

Pulled directly from `ci_report.json` (`stripped` block, `metric_version: 5`, `n_clean = 5,002`, 95% bootstrap CIs, 1000 resamples):

**`exp_text_only_v4_001`** (`v5_adv_error.point_details.families` for adversarial families; `hard_negative_fpr_at_1pct.per_family` for HN families; `r_at_fpr_0.01` for pooled):

| Field | Point | CI [lo, hi] |
| --- | ---: | --- |
| `v5_adv_error` | 0.7684 | [0.7394, 0.7989] |
| `phish_takeover` recall | 0.1122 | [0.0721, 0.1580] |
| `phish_takeover_mfa_phished` recall | 0.0000 | [0.0000, 0.0119] |
| `hn_recovery_high_amount` FPR | 0.4175 | [0.3418, 0.5121] |
| `hn_account_recovery` FPR | 0.0014 | [0.0000, 0.0053] |
| `hn_large_purchase` FPR | 0.0000 | [0.0000, 0.0000] |
| `hn_travel` FPR | 0.0040 | [0.0000, 0.0101] |
| Pooled `r_at_fpr_0.01` | 0.7391 | [0.7126, 0.7662] |

**`exp_xattn_v4_001`**:

| Field | Point | CI [lo, hi] |
| --- | ---: | --- |
| `v5_adv_error` | 0.1596 | [0.1306, 0.1945] |
| `phish_takeover` recall | 1.0000 | [1.0000, 1.0000] |
| `phish_takeover_mfa_phished` recall | 0.9718 | [0.9310, 1.0000] |
| `hn_recovery_high_amount` FPR | 0.4505 | [0.3685, 0.5624] |
| `hn_account_recovery` FPR | 0.0000 | [0.0000, 0.0000] |
| `hn_large_purchase` FPR | 0.0000 | [0.0000, 0.0000] |
| `hn_travel` FPR | 0.0000 | [0.0000, 0.0000] |
| `max_gate_magnitude` | 0.0221 | (point only) |
| Pooled `r_at_fpr_0.01` | 0.9947 | [0.9888, 0.9993] |

Verification commands:

```bash
python3 -c "import json; d=json.load(open('src/auto_research/runs/exp_xattn_v4_001/ci_report.json')); \
  print(d['stripped']['v5_adv_error']['point_details']['families'])"
python3 -c "import json; d=json.load(open('src/auto_research/runs/exp_text_only_v4_001/ci_report.json')); \
  print(d['stripped']['hard_negative_fpr_at_1pct']['per_family'])"
```

### 2.2 Specific edits applied

**`00-whitepaper-main.md`** (4 sites):

1. **Abstract (line 10):** `text-only 0.0141 / xattn 1.0000 [1.00, 1.00]` → `text-only 0.000 / xattn 0.972 [0.931, 1.000]` on `phish_takeover_mfa_phished` recall.
2. **§1.1 Contribution 4 (line 34):** `(0.33 → 1.00, 0.014 → 1.00)` → `(0.11 → 1.00, 0.00 → 0.97)`.
3. **§4.2 v4 headline table + commentary (lines 187–194):** all six cells updated; added `source:` citation to the two `ci_report.json` paths; rewrote the "hn_recovery is catastrophic for both arms" sentence to reflect the actual ~0.42–0.45 FPR band with overlapping CIs and the +0.033 absolute gap on the point estimate.
4. **§5 Results v4 bullets (lines 231–233):** all three per-family CI strings updated; the third bullet now says "both poor (~0.42–0.45), CIs overlap, xattn +0.033 worse on point estimate" instead of "both catastrophic, CIs overlap".

**`04-cross-attention-experiments.md`** (4 sites in §4 + 1 in §6):

5. **§4 Fraud recall table (lines 148–158):** removed the four "saturated" rows for `cred_stuff`, `sim_swap`, `malware_rat`, `mule_chain` because the current `ci_report.json` does not break out per-family fraud recall for them. Kept the two adversarial-family rows with corrected numbers and Δ values. Added a paragraph below the HN-FPR table reporting the pooled `r_at_fpr_0.01` as the verifiable substitute (`text_only_v4 = 0.7391`, `xattn_v4 = 0.9947`).
6. **§4 HN-FPR table (lines 159–166):** all four HN-family rows updated from `hard_negative_fpr_at_1pct.per_family`. `hn_account_recovery`/`hn_large_purchase`/`hn_travel` rows were already approximately correct (small drift only); `hn_recovery_high_amount` row corrected from 0.9872/1.0000 to 0.4175/0.4505 with the actual CIs.
7. **§4 narrative after HN table (line 168):** "catastrophic for both arms" → "sits at ~0.42–0.45 FPR for both arms with overlapping CIs — the architectural ceiling that no v5 dial subsequently moves (§5.3)".
8. **§4 v4 gates story (line 170):** "+0.99 swing" → "+0.97 swing (0.000 → 0.972)". The `tanh(α) ≈ 0.022 × 3 blocks` derivation is unchanged.
9. **§6 hn_recovery diagnostic (line 246):** `xattn FPR = 1.0000 / text_only FPR = 0.9872 / +0.013 worse` → `xattn 0.4505 [0.369, 0.562] / text_only 0.4175 [0.342, 0.512] / +0.033 worse on point estimate, CIs overlap`. Removed the stale "different denominator than v4" parenthetical — v4 and v5 use the same `metric_version: 5, n_clean = 5,002` surface.

### 2.3 What did NOT need changing on v4

- **Max-gate magnitude (0.0221).** Matches `metrics.json::max_gate_magnitude` for `exp_xattn_v4_001` exactly. No edit.
- **v3 → v4 → v5 max-gate trajectory in 00-main §5 line 240.** All three rows match the corresponding `metrics.json` files.
- **v3 leader and baseline numbers.** Untouched in this pass; not in scope. They are archived in `experiments.jsonl.pre_v5_20260521T035735Z` and are not what REVIEW-4 §3.1 flagged.

---

## 3. Honesty about what REVIEW-3 missed

REVIEW-3 declared "publish-ready" after a cross-document consistency check that did not ground-truth against `ci_report.json`. The failure mode was specific: the paper's V4 tables had drifted away from the artifacts as a *coherent* set (the 0.0141/0.3304/0.9872/1.0/1.0/1.0 numbers all came from a single earlier scoring surface), so internal consistency-greps passed.

The right diagnostic — and the one REVIEW-4 ran — is to read the actual `ci_report.json` for the named runs and reconcile every reported number cell-by-cell. I should have done this in REVIEW-1, REVIEW-2, or REVIEW-3 and did not. REVIEW-4's verdict was correct.

This pass adopts the artifact-first standard for V4 numbers. The same standard needs to be applied to V5 (§4 below).

---

## 4. REVIEW-4 items still open after this pass

The five blocking items below were **not** touched in this pass. They remain blockers for external publication.

### 4.1 [OPEN] V5 Phase-1 table stale values (REVIEW-4 §3.2)

`04-cross-attention-experiments.md:188–196` Phase-1 table contains five known-stale `v5_adv_error` values:

| exp_id | Paper | `experiments.jsonl` |
| --- | ---: | ---: |
| `exp_v5_p1_late_64` | 0.1654 | 0.1549 |
| `exp_v5_p1_slots128` | 0.1591 | 0.1526 |
| `exp_v5_p1_slots32` | 0.1684 | 0.1656 |
| `exp_v5_p1_rank32` | 0.1630 | 0.1617 |
| `exp_v5_p1_slowlr` | 0.3100 | 0.2015 |

`exp_v5_p1_zero_64` CI is reported as `[0.1278, 0.1893]` in two places (00-main §5 line 207, 04 §5.1 line 198); actual `experiments.jsonl` value is `[0.1238, 0.1871]`. The same numbers are visually encoded in `figures/fig4-sweep-results.svg`.

**Fix plan:** Pull `v5_adv_error` and component fields directly from `experiments.jsonl` (or the per-run `ci_report.json` files) for all 12 v5-schema rows. Update the §5.1 table, §5.2 table, and Figure 4 in one pass against a single source of truth. Estimated 20–30 min if scripted, longer if manual.

### 4.2 [OPEN] hn_recovery band overstated as covering "all 11 v5 runs" (REVIEW-4 §3.3)

`00-whitepaper-main.md:219`, `04-cross-attention-experiments.md:246`, and `04 §5.3 line 227` all say the `hn_recovery_high_amount` FPR stayed in `[0.4377, 0.4505]` across all 11 v5 runs. `exp_v5_p1_fastlr` actually has FPR `0.4249`. Codex's recommended rephrasing (restrict the band claim to non-pathological runs and explain the fastlr regress as an operating-point shift after fraud recall collapsed) is the correct fix. Not applied in this pass.

### 4.3 [OPEN] Per-run `leakage_report.json` claimed but absent (REVIEW-4 §3.4)

Verified: `find src/auto_research/runs -name leakage_report.json` returns 0 files across all 28 run directories. `scripts/run_next_experiment.py:1185-1189` treats missing leakage reports as "benign-unknown". The paper claims this file is a per-run launcher artifact in six places (00-main §3.2 line 102, 00-main §6 line 255, 01-data line 259, 02-harness line 37, 03-eval lines 226 and 282, Figure 2 SVG lines 130 and 172). All six claims are unsupported by the local state.

**Fix plan:** Replace each claim with the artifacts that exist — `clean_eval_n`, `clean_eval_dropped`, `clean_eval_mask_text_overlap`, `clean_eval_mask_events_overlap`, `leakage_clean` (rows on `experiments.jsonl`). Or, alternatively, add a real leakage-report writer to the launcher and backfill the missing files. Pick one. Not applied in this pass.

### 4.4 [OPEN] Experiment-count provenance is misleading (REVIEW-4 §3.5)

Current `experiments.jsonl` has 12 rows; the archived `.pre_v5_20260521T035735Z` has 18 rows. The paper's "30 experiments in `experiments.jsonl`" is misleading without explaining the V5 reset. The 22-xattn-plus-8-baselines decomposition also needs a precise definition (smoke/full PASS markers, retry-failed rows, etc.). Codex's suggested provenance paragraph is the right shape. Not applied in this pass.

### 4.5 [OPEN] GPU lockfile path mismatch (REVIEW-4 §4.1)

Code uses `/workspace/.gpu.lock` (period) at `scripts/run_next_experiment.py:66` and `scripts/agent_tick.sh:52,233`. Paper says `/workspace/.gpu_lock` (underscore) at `02-harness §3.4 line 67` and `figures/fig2-auto-research-loop.svg:75`. One-character fix; not applied.

### 4.6 [OPEN] Phase-2 stop is not encoded in `sweep_state.yaml` (REVIEW-4 §4.2)

`sweep_state.yaml:39-40` says `halted: false, halt_reason: null` despite the prose claiming the Phase-2 stop rule fired. The distinction between launcher-recorded halts, instruction-level/agent-queue stops, and `agent_tick.sh` stale-tick auto-stop needs to be drawn in 02-harness. Not applied in this pass.

---

## 5. Suggested fix order for the next pass

In dependency order (each builds on the prior or is independent):

1. **V5 Phase-1 numeric correction (§4.1 above).** Largest single block of stale data. Best done by writing a small `tools/make_v5_table.py` that reads `experiments.jsonl` and emits a markdown table; then drop the table into 04 §5.1 and regenerate Figure 4 from the same script. This also fixes the §5.2 Phase-1-winner-CI mismatch.
2. **hn_recovery band rephrase (§4.2).** Surgical; ~3 lines across 2 files. Best done after §4.1 so the fastlr correction sits next to the corrected fastlr number.
3. **leakage_report.json claims (§4.3).** Decide writer-vs-rewrite policy first; then apply across 6 sites + 1 SVG.
4. **Provenance paragraph (§4.4).** Add as a new subsection in 02-harness §9 or §11; reference from 00-main §6.
5. **GPU lock path (§4.5).** One-character fix in 2 files.
6. **Phase-2 halt clarification (§4.6).** Add a paragraph in 02-harness §6 distinguishing the three halt mechanisms.

Estimated total remaining effort: ~1 hour, dominated by §4.1 (V5 numeric correction + Figure 4 regen).

---

## 6. Cross-document consistency check after this pass

Re-running the v4-specific consistency check against `ci_report.json`:

| Claim | 00-main | 04-experiments | `ci_report.json` |
| --- | --- | --- | --- |
| `text_only_v4` mfa_phished recall | 0.000 ✓ | 0.0000 ✓ | 0.0000 ✓ |
| `xattn_v4` mfa_phished recall | 0.972 ✓ | 0.9718 ✓ | 0.9718 ✓ |
| `text_only_v4` phish_takeover recall | 0.11 ✓ | 0.1122 ✓ | 0.1122 ✓ |
| `xattn_v4` phish_takeover recall | 1.00 ✓ | 1.0000 ✓ | 1.0000 ✓ |
| `text_only_v4` hn_recovery FPR | 0.4175 ✓ | 0.4175 ✓ | 0.4175 ✓ |
| `xattn_v4` hn_recovery FPR | 0.4505 ✓ | 0.4505 ✓ | 0.4505 ✓ |
| Pooled `r_at_fpr_0.01` text_only | (n/a) | 0.7391 ✓ | 0.7391 ✓ |
| Pooled `r_at_fpr_0.01` xattn | (n/a) | 0.9947 ✓ | 0.9947 ✓ |
| max_gate_magnitude xattn | 0.0221 ✓ | 0.0221 ✓ | 0.0221 (metrics.json) ✓ |

V4 numbers across the two affected documents now agree with each other and with the artifacts.

---

## 7. Verdict after this pass

**Not yet publish-ready.** One of six REVIEW-4 blockers closed (the largest factual gap). Five remain.

The qualitative arc — v3 false null → v4 CI-separated win on adversarial fraud → v5 dial-robust but data-shaped ceiling — survives all the numeric corrections. The headline interpretation does not change. What changes is that the numbers backing the interpretation now come from the artifacts on disk instead of from a stale earlier scoring run.

Next pass should clear §4.1 (V5 numeric correction + Figure 4) and §4.3 (leakage_report.json claims), which together account for the bulk of remaining factual risk. The other three items (§4.2, §4.4, §4.5, §4.6) are smaller and can ride on the same correction pass.
