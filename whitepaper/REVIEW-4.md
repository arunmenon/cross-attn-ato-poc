# Whitepaper Review -- Fourth Pass (`cross_attn_ato_poc/whitepaper/v1.1`)

**Reviewer pass · 2026-05-22**
Subject: same five `.md` files + four SVG figures, checked against the current run ledgers and launcher code.
Prior reviews: `REVIEW.md` -> `REVIEW-2.md` -> `REVIEW-3.md`.

---

## 1. Top-line verdict

**Not publish-ready yet.** REVIEW-3 says the whitepaper is publish-ready, but a fresh artifact-backed pass found several factual mismatches that are larger than wording polish.

The paper's structure is still strong: v3 false null -> v4 data pivot -> v5 robustness/ceiling is the right narrative, and the harness/eval/data companions are the right decomposition. The issue is that some numbers and artifact claims are stale relative to the current source of truth:

- `src/auto_research/experiments.jsonl` now contains 12 V5-schema rows.
- `src/auto_research/experiments.jsonl.pre_v5_20260521T035735Z` contains the archived earlier rows.
- Current V5 seed rows report V4/V5 metrics that do not match the whitepaper's V4 tables.
- No `runs/exp_*/leakage_report.json` files exist locally, despite multiple docs and Figure 2 claiming they are per-run artifacts.
- The docs and Figure 2 still use the old GPU lock path `/workspace/.gpu_lock`, while the actual launcher and tick script use `/workspace/.gpu.lock`.

The result is fixable. I would call this **~85-90% there**, not externally ready. The blocking work is a numeric/provenance cleanup pass, not a conceptual rewrite.

---

## 2. Verification sources used in this pass

Claims below were checked against:

- `src/auto_research/experiments.jsonl` -- current V5 state, 12 rows.
- `src/auto_research/experiments.jsonl.pre_v5_20260521T035735Z` -- archived pre-V5 state, 18 rows.
- `src/auto_research/runs/exp_*/ci_report.json` -- bootstrap CI source for V4/V5 rows.
- `src/auto_research/runs/exp_*/metrics.json` -- per-run metric artifacts.
- `src/auto_research/sweep_state.yaml` -- current derived V5 state.
- `scripts/run_next_experiment.py` and `scripts/agent_tick.sh` -- lockfile, leakage, halt, and state behavior.
- `whitepaper/figures/*.svg` -- figure text.

Current V5 ledger summary:

| File | Rows | Notes |
| --- | ---: | --- |
| `experiments.jsonl` | 12 | 11 x-attn + 1 text-only, all `metric_version: 5` |
| `experiments.jsonl.pre_v5_20260521T035735Z` | 18 | archived earlier rows; 15 `ok`, 1 failed, 2 PASS rows |
| Combined | 30 | 21 x-attn rows by `arm`, 9 non-xattn rows by `arm` |

---

## 3. Blocking findings

### 3.1 CRITICAL -- V4 headline tables do not match current `metric_version: 5` artifacts

**Files:** `00-whitepaper-main.md`, `04-cross-attention-experiments.md`

Current text:

- `00-whitepaper-main.md:187-194` reports:
  - `text_only_v4`: mfa recall `0.0141`, phish recall `0.3304`, HN FPR `0.9872`
  - `xattn_v4`: mfa recall `1.0000`, phish recall `1.0000`, HN FPR `1.0000`
- `00-whitepaper-main.md:231-233` repeats the same V4 values.
- `04-cross-attention-experiments.md:148-166` labels the V4 table as `metric_version: 5`, but uses the same stale values.
- `04-cross-attention-experiments.md:246` says the V4 HN result is `xattn FPR = 1.0000`, `text_only FPR = 0.9872`.

Current artifact values from `src/auto_research/experiments.jsonl` and `runs/exp_xattn_v4_001/ci_report.json`:

| Arm | phish_takeover recall | phish_takeover_mfa_phished recall | hn_recovery_high_amount FPR |
| --- | ---: | ---: | ---: |
| `text_only_v4` | `0.1122` [0.0721, 0.1580] | `0.0000` [0.0000, 0.0119] | `0.4175` [0.3418, 0.5121] |
| `xattn_v4` | `1.0000` [1.0000, 1.0000] | `0.9718` [0.9310, 1.0000] | `0.4505` [0.3685, 0.5624] |

The qualitative fraud-family conclusion still holds: x-attn is CI-separated from text-only on the adversarial fraud families. But the quantitative story is materially different:

- `phish_takeover` text-only recall is `0.1122`, not `0.3304`.
- `phish_takeover_mfa_phished` x-attn recall is `0.9718`, not exactly `1.0000`.
- `hn_recovery_high_amount` is not near-1.0 under the current V5 scoring surface; it is around `0.42-0.45`.

**Fix:** Decide whether the paper is reporting old V4 pre-V5-scoring numbers or the current `metric_version: 5` rescore. If it is the current paper, update all V4 tables and prose to the artifact values above. If the old V4 values are retained for historical reasons, label them explicitly as an older scoring surface and do not call that table `metric_version: 5`.

### 3.2 CRITICAL -- V5 Phase-1 table and Figure 4 still contain stale values

**Files:** `04-cross-attention-experiments.md`, `00-whitepaper-main.md`, `figures/fig4-sweep-results.svg`

The Phase-1 table at `04-cross-attention-experiments.md:188-196` has multiple stale values. Figure 4 repeats them visually at `fig4-sweep-results.svg:66-118`.

Correct values from current `experiments.jsonl`:

| exp_id | Current `v5_adv_error` [CI] | Current mfa miss | Current HN FPR | Current max_gate |
| --- | --- | ---: | ---: | ---: |
| `exp_xattn_v4_001` | `0.1596` [0.1306, 0.1945] | 0.0282 | 0.4505 | 0.0221 |
| `exp_v5_p1_every4_64` | `0.1506` [0.1235, 0.1886] | 0.0141 | 0.4377 | 0.0171 |
| `exp_v5_p1_late_64` | `0.1549` [0.1279, 0.1927] | 0.0141 | 0.4505 | 0.0179 |
| `exp_v5_p1_zero_64` | `0.1506` [0.1238, 0.1871] | 0.0141 | 0.4377 | 0.0128 |
| `exp_v5_p1_slots128` | `0.1526` [0.1278, 0.1893] | 0.0201 | 0.4377 | 0.0212 |
| `exp_v5_p1_slots32` | `0.1656` [0.1410, 0.2017] | 0.0463 | 0.4505 | 0.0182 |
| `exp_v5_p1_rank32` | `0.1617` [0.1347, 0.1979] | 0.0347 | 0.4505 | 0.0089 |
| `exp_v5_p1_slowlr` | `0.2015` [0.1670, 0.2435] | 0.1539 | 0.4505 | 0.0165 |
| `exp_v5_p1_fastlr` | `0.7516` [0.7208, 0.7828] | 0.9998 | 0.4249 | 0.0058 |

Examples of stale claims:

- `04-cross-attention-experiments.md:190`: `late_64 = 0.1654`; actual `0.1549`.
- `04-cross-attention-experiments.md:192`: `slots128 = 0.1591`; actual `0.1526`.
- `04-cross-attention-experiments.md:193`: `slots32 = 0.1684`; actual `0.1656`.
- `04-cross-attention-experiments.md:194`: `rank32 = 0.1630`; actual `0.1617`.
- `04-cross-attention-experiments.md:195`: `slowlr = 0.3100`; actual `0.2015`.
- `00-whitepaper-main.md:207` and `04-cross-attention-experiments.md:198` give the Phase-1 winner CI as `[0.1278, 0.1893]`; actual `exp_v5_p1_zero_64` CI is `[0.1238, 0.1871]`.
- `fig4-sweep-results.svg:72`, `:91`, `:100`, `:109`, `:118` visually encode the stale values.

**Fix:** Regenerate Figure 4 directly from `experiments.jsonl`, or manually update both the table and SVG from the current ledger. This should be treated as one source-of-truth correction; otherwise the figure and prose will keep drifting.

### 3.3 HIGH -- The `hn_recovery_high_amount` ceiling band is stated too strongly

**Files:** `00-whitepaper-main.md`, `04-cross-attention-experiments.md`

Current text:

- `00-whitepaper-main.md:219`: "Across all 11 v5 runs, the `hn_recovery_high_amount` FPR stayed in the band [0.4377, 0.4505]".
- `04-cross-attention-experiments.md:246`: "across 11 configurations, FPR is stuck at 0.4377-0.4505".

Current ledger:

- Most non-pathological x-attn rows are indeed in `[0.4377, 0.4505]`.
- `exp_v5_p1_fastlr` has `hn_recovery_high_amount_fpr = 0.4249`.
- The current `text_only_v4` seed has `0.4175`, though it is not an x-attn V5 run.

The fastlr row is pathological because fraud recall collapsed; its lower HN FPR is not a useful win. But the phrase "across all 11 v5 runs" is still false as written.

**Fix:** Rephrase to something like:

> "Across the non-pathological V5 x-attn runs, `hn_recovery_high_amount` FPR stayed in [0.4377, 0.4505]. The fast-LR failure lowered HN FPR to 0.4249 only because the operating point shifted after fraud recall collapsed, so it is not evidence that the HN ceiling moved."

That preserves the correct interpretation without making a false range claim.

### 3.4 HIGH -- Per-run `leakage_report.json` is claimed but absent

**Files:** `00-whitepaper-main.md`, `01-data-curation-and-distribution.md`, `02-agentic-experiment-harness.md`, `03-eval-strategy.md`, `figures/fig2-auto-research-loop.svg`

Current claims:

- `00-whitepaper-main.md:102` says structured per-run JSON artifacts include `leakage_report.json`.
- `00-whitepaper-main.md:255` says `runs/exp_*/leakage_report.json` had flagged family-concentrated overlap.
- `01-data-curation-and-distribution.md:259` says every run writes `runs/exp_NNN/leakage_report.json`.
- `02-agentic-experiment-harness.md:37` lists `leakage_report` under launcher-owned structured artifacts.
- `03-eval-strategy.md:226` repeats "Every run writes...".
- `03-eval-strategy.md:282` says diagnostics assert counts match recorded `leakage_report.json` fields.
- `figures/fig2-auto-research-loop.svg:130` and `:172` include `leakage_report`.

Actual local state:

- `rg --files src/auto_research/runs | rg 'leakage_report\.json$'` returns no files.
- 26 run directories have `metrics.json` but no `leakage_report.json`.
- `scripts/run_next_experiment.py:1185-1189` explicitly treats a missing leakage report as `benign-unknown`:

```python
def _check_leakage_clean(run_dir: Path) -> bool:
    """Reads leakage_report.json if present. Returns True if no violations."""
    p = run_dir / "leakage_report.json"
    if not p.exists():
        return True  # trainer didn't write one; treat as benign-unknown
```

The rows do contain clean-eval fields (`leakage_clean`, `clean_eval_n`, overlap counts). That is the artifact actually present.

**Fix:** Replace the per-run `leakage_report.json` claims with the artifacts that exist:

- `clean_eval_n`
- `clean_eval_dropped`
- `clean_eval_mask_text_overlap`
- `clean_eval_mask_events_overlap`
- `leakage_clean`

If you want to keep the stronger claim, first add a real `leakage_report.json` writer and backfill reports.

### 3.5 HIGH -- Experiment-count provenance is misleading after V5 state reset

**Files:** `00-whitepaper-main.md`, `02-agentic-experiment-harness.md`, `REVIEW-3.md`

Current text:

- `00-whitepaper-main.md:225` says reproduction artifacts are committed in `src/auto_research/runs/`.
- `02-agentic-experiment-harness.md:255` says "30 experiments recorded in `src/auto_research/experiments.jsonl`, comprising 22 cross-attention runs and 8 baseline runs".
- `REVIEW-3.md:41` says the "30 experiments" count matches `experiments.jsonl`.

Current artifact layout:

- Current `src/auto_research/experiments.jsonl` has 12 rows, not 30.
- Archived `src/auto_research/experiments.jsonl.pre_v5_20260521T035735Z` has 18 rows.
- Combined rows = 30, but by `arm` the count is `21 xattn` and `9 non-xattn` rows, not obviously "22 xattn + 8 baselines".
- Some rows are status markers (`PASS_smoke`, `PASS_full`) or pre/post-rescore duplicate rows, so "30 experiments" needs a precise definition.

**Fix:** Add an "artifact provenance" paragraph:

> "The current V5 ledger is `experiments.jsonl` (12 rows). The pre-V5 ledger is archived as `experiments.jsonl.pre_v5_20260521T035735Z` (18 rows). The paper's 30-row count refers to the union of these ledgers; rows include baselines, smoke/full pass records, failed runs, and V5 seed rows."

Then adjust the 22/8 decomposition to match whatever counting convention you actually want.

---

## 4. Medium findings

### 4.1 GPU lockfile path is wrong in docs and Figure 2

**Files:** `02-agentic-experiment-harness.md`, `figures/fig2-auto-research-loop.svg`

Current doc/figure path:

- `02-agentic-experiment-harness.md:67`: `/workspace/.gpu_lock`
- `figures/fig2-auto-research-loop.svg:75`: `/workspace/.gpu_lock`

Actual code:

- `scripts/agent_tick.sh:27` documents default `GPU_LOCK_FILE` as `/workspace/.gpu.lock`.
- `scripts/agent_tick.sh:52` sets `GPU_LOCK_FILE="${GPU_LOCK_FILE:-/workspace/.gpu.lock}"`.
- `scripts/run_next_experiment.py:66` sets `LOCKFILE = Path("/workspace/.gpu.lock")`.

**Fix:** Replace `/workspace/.gpu_lock` with `/workspace/.gpu.lock` in text and Figure 2.

### 4.2 `sweep_state.yaml` does not encode the Phase-2 stop that the prose treats as a state halt

**Files:** `00-whitepaper-main.md`, `02-agentic-experiment-harness.md`, `sweep_state.yaml`

The prose says the Phase-2 stop rule fired. That is operationally true at the agent/cron level, but current `sweep_state.yaml:39-40` still says:

```yaml
halted: false
halt_reason: null
```

This is not necessarily a bug if the stop was instruction-level and the stale-tick detector handled pod stop. But the harness documentation should distinguish:

- launcher-recorded halts in `sweep_state.yaml`
- instruction-level/agent queue stops
- `agent_tick.sh` stale-tick auto-stop behavior

Otherwise readers will expect `sweep_state.yaml` to show the Phase-2 halt reason, and it does not.

### 4.3 Figure 4 should not be hand-maintained

Figure 4 is a central result figure and currently drifts from the ledger. The safest fix is not just to patch the visible text: add a small script or documented command that regenerates `fig4-sweep-results.svg` from `experiments.jsonl`.

Minimum acceptable fix: update the SVG manually and add a caption note that values are from the current `experiments.jsonl` as of `last_updated: 2026-05-21T12:26:29Z`.

Better fix: create `whitepaper/scripts/make_fig4.py` or a repo script that writes the SVG from the ledger.

---

## 5. What REVIEW-3 got right

These areas still look solid and do not need another rewrite:

1. **Architecture block counts and layer-12 offset.** `00-whitepaper-main.md:143`, `04-cross-attention-experiments.md:46-48`, and Figure 1 now agree with code: every_4 = 6 inserts, every_8 = 3, late_only = 4, starting at layer 12 or later.

2. **Measured parameter inventory.** `04-cross-attention-experiments.md:75-91` is strong. The measured parameter counts and per-block arithmetic are exactly the right standard for this paper.

3. **Narrator model/cost cleanup.** `00-whitepaper-main.md:68`, `01-data-curation-and-distribution.md:281`, and `03-eval-strategy.md:39` now avoid the old `gpt-5.4-nano` issue and use the current `$2.03` cost.

4. **ft_transformer framing.** The current wording correctly frames the result as inductive-bias mismatch rather than a raw capacity comparison.

5. **Limitations section.** The synthetic-only, single-LM, gate-magnitude, family-specific ceiling, compute-budget, and single-engineer limitations are clear and worth preserving.

---

## 6. Recommended fix order

In priority order:

1. **Fix V4 metric tables and prose** in `00-main` and `04`. Decide old-V4-vs-current-V5 scoring, then make labels and numbers unambiguous.
2. **Regenerate/update Figure 4 and the V5 Phase-1 table** from current `experiments.jsonl`.
3. **Rewrite all `leakage_report.json` artifact claims** to match actual clean-eval fields, or add/backfill real reports.
4. **Add artifact provenance for current vs archived ledgers** and correct the 30-run decomposition.
5. **Fix GPU lock path** from `.gpu_lock` to `.gpu.lock` in docs and Figure 2.
6. **Clarify instruction-level Phase-2 stop vs `sweep_state.yaml.halted`**.

After these, REVIEW-3's "publish-ready" conclusion will be much more defensible.

---

## 7. Estimated effort

- V4/V5 numeric correction and Figure 4 update: 30-45 min if manual, less if scripted.
- Leakage-report/provenance wording: 15-20 min.
- GPU lock path + Phase-2 halt wording: 5-10 min.
- Final consistency grep: 5 min.

Total: roughly 1 hour for a careful cleanup pass.

---

## 8. Bottom line

The whitepaper has the right story and enough artifact depth to be credible, but the current v1.1 draft is carrying stale result numbers and a few nonexistent artifact claims. I would not send it externally until the V4/V5 tables, Figure 4, leakage-report wording, and ledger provenance are corrected.

**Fourth-pass status: revise before publication.**
