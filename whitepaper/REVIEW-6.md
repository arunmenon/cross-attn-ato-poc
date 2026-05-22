# Whitepaper Review — Sixth Pass (`cross_attn_ato_poc/whitepaper/v1.1`)

**Reviewer pass · 2026-05-22**
Subject: V5 Phase-1 numeric corrections + hn_recovery band rephrase + Figure 4 SVG update.
Prior reviews: `REVIEW.md` → `REVIEW-2.md` → `REVIEW-3.md` → `REVIEW-4.md` → `REVIEW-5.md`.

---

## 1. Top-line

This pass closes **REVIEW-4 §3.2** (V5 Phase-1 stale values) and **REVIEW-4 §3.3** (hn_recovery band overstated) — including Figure 4's stale bar labels and the slowlr bar geometry. The four remaining REVIEW-4 items (`leakage_report.json`, 30-run provenance, `.gpu_lock` → `.gpu.lock`, `sweep_state.yaml` halt clarification) are all wording/path corrections that don't need ledger re-reads; they're documented in §4 below for the next pass.

All V5 quantitative claims are now grounded against the per-run `ci_report.json` files in `src/auto_research/runs/exp_*/` (the same artifact-first standard adopted for V4 in REVIEW-5).

---

## 2. What was changed in this pass

### 2.1 Ground-truth used

Pulled directly from `runs/exp_*/ci_report.json::stripped.v5_adv_error` for all 11 v5 runs + the v4 seed + the text_only seed (12 rows total in `experiments.jsonl`).

Authoritative `v5_adv_error` point + CI for each run:

| exp_id | point | CI [lo, hi] | mfa miss | hn FPR | max_gate |
| --- | ---: | --- | ---: | ---: | ---: |
| `exp_xattn_v4_001` (seed) | 0.1596 | [0.1306, 0.1945] | 0.0282 | 0.4505 | 0.0221 |
| `exp_v5_p1_every4_64` | 0.1506 | [0.1235, 0.1886] | 0.0141 | 0.4377 | 0.0171 |
| `exp_v5_p1_late_64` | 0.1549 | [0.1279, 0.1927] | 0.0141 | 0.4505 | 0.0179 |
| `exp_v5_p1_zero_64` ★ | 0.1506 | [0.1238, 0.1871] | 0.0141 | 0.4377 | 0.0128 |
| `exp_v5_p1_slots128` | 0.1526 | [0.1278, 0.1893] | 0.0201 | 0.4377 | 0.0212 |
| `exp_v5_p1_slots32` | 0.1656 | [0.1410, 0.2017] | 0.0463 | 0.4505 | 0.0182 |
| `exp_v5_p1_rank32` | 0.1617 | [0.1347, 0.1979] | 0.0347 | 0.4505 | 0.0089 |
| `exp_v5_p1_slowlr` | 0.2015 | [0.1670, 0.2435] | 0.1539 | 0.4505 | 0.0165 |
| `exp_v5_p1_fastlr` | 0.7516 | [0.7208, 0.7828] | 0.9998 | 0.4249 | 0.0058 |
| `exp_v5_p2_pooled_mlp` | 0.1549 | [0.1282, 0.1938] | 0.0141 | 0.4505 | 0.0146 |
| `exp_v5_p2_ft_transformer` | 0.1736 | [0.1448, 0.2136] | 0.0704 | 0.4505 | 0.0157 |

Verification command:

```bash
python3 -c "
import json
from pathlib import Path
for run_dir in sorted(Path('src/auto_research/runs').glob('exp_*')):
    cip = run_dir / 'ci_report.json'
    if not cip.exists(): continue
    d = json.load(cip.open())
    v5 = d.get('stripped', {}).get('v5_adv_error', {})
    if 'point' in v5:
        print(run_dir.name, round(v5['point'],4), '[', round(v5['ci_lo'],4), round(v5['ci_hi'],4), ']')
"
```

### 2.2 Notable surprise during this pass

The Phase-1 winner CI that the v1.1 draft attributed to `exp_v5_p1_zero_64` (`[0.1278, 0.1893]`) is *actually* the CI for `exp_v5_p1_slots128`. The two runs both have point `0.1506` and `0.1526` respectively, and their CIs were mixed up in the original table. Three downstream sites in 00-main repeated the same swapped CI for the winner. All four are now corrected to the actual zero_64 CI `[0.1238, 0.1871]`.

This is the kind of error that survives a careful internal-consistency review (REVIEW-3) because the CIs *look* consistent across documents — they were consistent with each other, just not with the artifacts. Same failure mode as REVIEW-5 §3 documented for V4.

### 2.3 Specific edits applied

**`04-cross-attention-experiments.md`** (4 sites):

1. **§5.1 Phase-1 table (lines 186–196):** replaced all "(approx)" point estimates with the actual `v5_adv_error.point` and added full CIs for every row. Added `source: runs/exp_*/ci_report.json` citation. Specific row corrections:
   - `late_64`: 0.1654 (approx) → 0.1549 [0.1279, 0.1927]; mfa miss ~0.045 → 0.0141
   - `zero_64`: CI 0.1278/0.1893 → 0.1238/0.1871; max_gate 0.0127 → 0.0128
   - `slots128`: 0.1591 (approx) → 0.1526 [0.1278, 0.1893]; mfa miss ~0.019 → 0.0201
   - `slots32`: 0.1684 (approx) → 0.1656 [0.1410, 0.2017]; mfa miss ~0.029 → 0.0463
   - `rank32`: 0.1630 (approx) → 0.1617 [0.1347, 0.1979]; mfa miss ~0.022 → 0.0347; max_gate 0.0088 → 0.0089
   - `slowlr`: 0.3100 (approx) → 0.2015 [0.1670, 0.2435]; mfa miss 0.846 (which was actually recall) → 0.1539 (the actual miss)
   - `fastlr`: added CI [0.7208, 0.7828]; mfa miss 0.83 (was actually phish_takeover miss) → 0.9998 (the actual mfa_phished miss)
   - Removed the broken `(regress)` placeholder in the hn_recovery FPR column for fastlr; replaced with the actual value 0.4249.
2. **§5.1 winner prose (line 198):** `0.1506 [CI 0.1278, 0.1893]` → `0.1506 [CI 0.1238, 0.1871]`; `winner_hi = 0.1893` → `0.1871`.
3. **§5.2 Phase-2 table (lines 213–216):** zero_64 row CI `0.1278, 0.1893` → `0.1238, 0.1871`; ft_transformer point `0.1737` → `0.1736` (rounding fix).
4. **§5.3 lead-in + dial-variation LR row (lines 223 + 231):** lead-in changed from "Across all 11 v5 runs" to "Across the non-pathological v5 x-attn runs", with an added sentence documenting the `fastlr 0.4249` excursion as an operating-point artifact (fraud recall collapsed → tied score mass at a different threshold) rather than a real architectural movement. LR row in the dial-variation table changed from "(catastrophic regress under fastlr)" to `[0.4249, 0.4505]` with the same operating-point caveat.

**`00-whitepaper-main.md`** (5 sites):

5. **§4.3 Finding 1 (line 207):** Phase-1 winner CI `[0.1278, 0.1893]` → `[0.1238, 0.1871]`.
6. **§4.3 Finding 2 Phase-2 table (lines 213–215):** zero_64 row CI fix + ft_transformer rounding fix (matches the 04 §5.2 edit).
7. **§4.3 Finding 3 (line 219):** "Across all 11 v5 runs ... no dial moves it" → "Across the non-pathological v5 x-attn runs ... no dial moves the underlying family", with the fastlr operating-point note inserted.
8. **§5 Results v5 paragraph (line 235):** winner CI `[0.1278, 0.1893]` → `[0.1238, 0.1871]`; "FPR ∈ [0.4377, 0.4505] across all 11 runs" → "in [0.4377, 0.4505] across all non-pathological runs (fastlr drops to 0.4249 only as an operating-point artifact after fraud recall collapsed)".
9. **§5 Gates story (line 241):** v5 max-gate range 0.0059–0.0220 → 0.0058–0.0221; lowest gate 0.0059 → 0.0058; Phase-1 winner gate 0.0127 → 0.0128 (rounding fixes against `metrics.json::max_gate_magnitude`).
10. **§6 Discussion (line 259):** "Across 11 v5 configurations, no architectural change moves the FPR below 0.4377" → "Across the non-pathological v5 configurations, no architectural or training-dial change moves the FPR below 0.4377; the one excursion (`fastlr` at 0.4249) is an operating-point artifact after fraud recall collapsed".

**`figures/fig4-sweep-results.svg`** (5 bar edits):

11. **Bar 3 (late_only):** label `0.165` → `0.155`; bar `y=286, h=74` → `y=290, h=70`; error-bar lines moved from y=270/300 to y=273/302.
12. **Bar 5 (slots=128):** label `0.159` → `0.153`; bar `y=289, h=71` → `y=291, h=69`; error-bar lines moved from y=272/301 to y=275/302.
13. **Bar 6 (slots=32):** label `0.168` → `0.166`; bar `y=284, h=76` → `y=285, h=75`; error-bar lines moved from y=268/298 to y=269/297.
14. **Bar 7 (lora_r=32):** label `0.163` → `0.162`; bar geometry unchanged (point 0.1617 rounds to same pixel); error-bar top moved 1 px (y=270 → y=271).
15. **Bar 8 (slowlr):** label `0.310` → `0.202`; bar `y=221, h=139` → `y=269, h=91` (this is the visually largest correction — the bar height drops by ~48 px because point dropped from 0.310 to 0.2015); error-bar lines moved from y=205/235 to y=250/285; label y position 198 → 244.

The two correct bars (Bar 4 zero_64 winner at 0.151, Bar 11 fastlr at 0.752) are unchanged; Bars 1, 2, 9, 10 round to the same 3-dp label and were left untouched.

**fig4 PNG companion:** the corresponding `figures/fig4-sweep-results.png` was *not* regenerated. The markdown documents reference the SVG by path, not the PNG; the PNG is a legacy snapshot for slide use. If the PNG is needed downstream, regenerate with `soffice --convert-to png` or a comparable tool.

### 2.4 Phase-1 finding bullets — no change needed

§5.1 line 208 ("LR is critical") describes `phish_takeover` recall collapse to 0.17 (matches fastlr's actual 0.1699) and `phish_takeover_mfa_phished` recall slowlr 0.986 → 0.846 (matches winner 0.9859 → slowlr 0.8461). Those prose claims are correct against the artifacts. The only mismatch was that the §5.1 *table*'s "mfa_phished miss" column had been populated with recall values for slowlr and with phish_takeover miss for fastlr — both fixed in edit 1 above.

---

## 3. Cross-document consistency check after this pass

V5 numbers across 00-main, 04-experiments, and Figure 4 now agree with each other and with `runs/exp_*/ci_report.json`:

| Claim | 00-main | 04-experiments | Figure 4 | ci_report.json |
| --- | --- | --- | --- | --- |
| Phase-1 winner `v5_adv_error` point | 0.1506 ✓ | 0.1506 ✓ | 0.151 ✓ | 0.1506 |
| Phase-1 winner CI | [0.1238, 0.1871] ✓ | [0.1238, 0.1871] ✓ | (n/a) | [0.1238, 0.1871] |
| `late_64` point | (n/a) | 0.1549 ✓ | 0.155 ✓ | 0.1549 |
| `slots128` point | (n/a) | 0.1526 ✓ | 0.153 ✓ | 0.1526 |
| `slots32` point | (n/a) | 0.1656 ✓ | 0.166 ✓ | 0.1656 |
| `rank32` point | (n/a) | 0.1617 ✓ | 0.162 ✓ | 0.1617 |
| `slowlr` point | (n/a) | 0.2015 ✓ | 0.202 ✓ | 0.2015 |
| `fastlr` point | (n/a) | 0.7516 ✓ | 0.752 ✓ | 0.7516 |
| `pooled_mlp` point | 0.1549 ✓ | 0.1549 ✓ | 0.155 ✓ | 0.1549 |
| `ft_transformer` point | 0.1736 ✓ | 0.1736 ✓ | 0.174 ✓ | 0.1736 |
| hn_recovery band claim qualifier | "non-pathological" ✓ | "non-pathological" ✓ | (n/a) | [0.4377, 0.4505] for non-pathological; 0.4249 for fastlr |
| fastlr hn FPR | (mentioned 0.4249) ✓ | 0.4249 ✓ | (n/a) | 0.4249 |
| v5 max_gate range | 0.0058–0.0221 ✓ | 0.0128 (winner) ✓ | (n/a) | matches `metrics.json::max_gate_magnitude` |

Zero V5 cross-doc contradictions remain after this pass.

---

## 4. REVIEW-4 items still open after this pass

Four items remain. None require ledger re-reads; all are wording/path corrections.

### 4.1 [OPEN] Per-run `leakage_report.json` claimed but absent (REVIEW-4 §3.4)

Verified again: `find src/auto_research/runs -name leakage_report.json` returns 0 files. Affected sites:
- `00-whitepaper-main.md:102`, `:255`
- `01-data-curation-and-distribution.md:259`
- `02-agentic-experiment-harness.md:37`
- `03-eval-strategy.md:226`, `:282`
- `figures/fig2-auto-research-loop.svg:130`, `:172`

**Fix plan:** Choose between (a) replacing claims with the actual clean-eval fields (`clean_eval_n`, `clean_eval_dropped`, `clean_eval_mask_text_overlap`, `clean_eval_mask_events_overlap`, `leakage_clean`) recorded on each `experiments.jsonl` row, or (b) adding a real `leakage_report.json` writer to the launcher and backfilling reports for completed runs. (a) is the smaller change; (b) is the more truthful long-term fix.

### 4.2 [OPEN] 30-run provenance after V5 state reset (REVIEW-4 §3.5)

Verified again: current `experiments.jsonl` has 12 rows; archived `experiments.jsonl.pre_v5_20260521T035735Z` has 18 rows. Union = 30. Affected sites:
- `00-whitepaper-main.md:225` (reproduction artifacts)
- `02-agentic-experiment-harness.md:255` (the "30 experiments, 22 xattn + 8 baselines" decomposition)

**Fix plan:** Add a short "artifact provenance" paragraph to 02-harness §9 explaining the V5 schema reset, the archive convention, and the precise definition of the 22/8 split (which rows count as xattn vs baseline; how PASS/smoke/failed rows are counted).

### 4.3 [OPEN] GPU lockfile path (REVIEW-4 §4.1)

Code uses `/workspace/.gpu.lock` (period). Docs use `/workspace/.gpu_lock` (underscore). Affected sites:
- `02-agentic-experiment-harness.md:67`
- `figures/fig2-auto-research-loop.svg:75`

**Fix plan:** Two single-character edits.

### 4.4 [OPEN] `sweep_state.yaml: halted: false` vs prose "Phase-2 stop fired" (REVIEW-4 §4.2)

Verified: `sweep_state.yaml` shows `halted: false, halt_reason: null` despite the Phase-2 stop being described in prose as a halt. The distinction is real (launcher-recorded halts vs agent-queue/instruction-level stops vs `agent_tick.sh` stale-tick auto-stop), but the docs don't draw it.

**Fix plan:** Add a paragraph to `02-harness §6` distinguishing the three halt mechanisms and noting that the Phase-2 stop is instruction-level (handled by the agent reading `AGENT_INSTRUCTIONS.md` and concluding no further runs are worth proposing), not launcher-recorded.

---

## 5. Suggested grouping for the next pass

All four remaining items are wording-only and can be batched into one pass:

1. Replace `leakage_report.json` claims with `clean_eval_*` fields (6 prose sites + 2 SVG nodes).
2. Add an "artifact provenance" paragraph to 02-harness §9; reference from 00-main §6.
3. Fix `.gpu_lock` → `.gpu.lock` in 2 places (1 prose + 1 SVG).
4. Add the 3-way halt-distinction paragraph in 02-harness §6.

Estimated total: 30–45 min, no ledger re-reads required.

---

## 6. Verdict after this pass

**Closer to publish-ready.** The largest numeric correction (V4 in REVIEW-5) and the second-largest (V5 Phase-1 + Figure 4 in this pass) are now applied. The remaining items are all in the "wording polish + path correction" tier — none affect the substantive findings.

After REVIEW-7 closes the four remaining items, the paper should be externally sendable.

What did *not* change qualitatively across REVIEW-5 → REVIEW-6:
- The v3 → v4 → v5 narrative arc.
- Q1 PASS A (xattn CI-separated from text_only on adversarial fraud families).
- The architectural ceiling on `hn_recovery_high_amount`.
- The methodology framing (agentic harness + bootstrap CI + tie-aware metric).
- The limitations section.

What changed:
- Every V4 and V5 quantitative claim now derives from the on-disk artifacts.
- The hn_recovery band claim is properly qualified.
- Figure 4 reflects the artifact values.

The paper is meaningfully more credible than the REVIEW-3 "publish-ready" snapshot, and the credibility increase is concentrated in the numbers an external reviewer would check first.
