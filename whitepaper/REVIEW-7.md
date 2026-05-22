# Whitepaper Review — Seventh Pass (`cross_attn_ato_poc/whitepaper/v1.1`)

**Reviewer pass · 2026-05-22**
Subject: Closes the four remaining REVIEW-4 items — leakage_report wording, run-count provenance, GPU lock path, and the three-halt-mechanism distinction. No ledger re-reads needed; this is the wording/path cleanup pass.
Prior reviews: `REVIEW.md` → `REVIEW-2.md` → `REVIEW-3.md` → `REVIEW-4.md` → `REVIEW-5.md` → `REVIEW-6.md`.

---

## 1. Top-line

All six REVIEW-4 blockers are now closed:

| REVIEW-4 item | Closed in |
| --- | --- |
| §3.1 V4 headline tables vs `ci_report.json` | REVIEW-5 |
| §3.2 V5 Phase-1 stale values + Figure 4 | REVIEW-6 |
| §3.3 hn_recovery band overstated | REVIEW-6 |
| §3.4 `leakage_report.json` claimed but absent | **this pass** |
| §3.5 30-run provenance after V5 reset | **this pass** |
| §4.1 GPU lockfile path mismatch | **this pass** |
| §4.2 `sweep_state.yaml` halt clarification | **this pass** |

The paper is now externally sendable in my judgment, modulo any review-cycle nits a third reader might surface. The substantive arc, the architecture description, the numbers, the methodology framing, the limitations, and the references are all in agreement with the on-disk artifacts.

---

## 2. What was changed in this pass

### 2.1 `leakage_report.json` → inline `clean_eval_*` fields on `experiments.jsonl` (REVIEW-4 §3.4)

The pre-existing claim was that every run wrote `runs/exp_NNN/leakage_report.json` summarizing the leakage checks. Verified state: `find src/auto_research/runs -name leakage_report.json` returns 0 files across 28 run directories; `scripts/run_next_experiment.py:1185-1189` treats missing leakage reports as "benign-unknown" by design. The actual mechanism is that each run records its leakage state inline on its `experiments.jsonl` row via five fields:

- `leakage_clean` (bool)
- `clean_eval_n` (post-mask eval set size, 5,002 in v4/v5)
- `clean_eval_dropped` (rows excluded by the mask, 0 in v4/v5)
- `clean_eval_mask_text_overlap`
- `clean_eval_mask_events_overlap`

Six prose sites + 2 SVG nodes were updated to describe the inline-on-jsonl mechanism instead of the non-existent per-run JSON:

1. **`00-whitepaper-main.md:102`** (§3.2 ownership invariant) — removed `leakage_report.json` from the launcher-only artifacts list; added an inline sentence naming the five fields.
2. **`00-whitepaper-main.md:255`** (§6 diagnostic anecdote) — "the leakage report (`runs/exp_*/leakage_report.json`)" → "the inline leakage fields on each `experiments.jsonl` row (`clean_eval_mask_text_overlap`, `clean_eval_mask_events_overlap`, `clean_eval_dropped`)".
3. **`01-data-curation-and-distribution.md:257-259`** (§5.4) — section retitled "Per-experiment leakage record"; body rewritten to describe the five inline fields.
4. **`02-agentic-experiment-harness.md:37`** (§2 ownership table) — removed `leakage_report` from the structured per-run JSON artifacts cell; added an explanatory clause naming the five inline fields.
5. **`03-eval-strategy.md:224-226`** (§6.3) — section retitled "Per-experiment leakage record"; body rewritten to match (1)/(3).
6. **`03-eval-strategy.md:282`** (§9 selftests) — "leakage counts match the recorded `leakage_report.json` fields" → "leakage counts match the recorded `clean_eval_mask_text_overlap` and `clean_eval_mask_events_overlap` fields on each `experiments.jsonl` row".
7. **`figures/fig2-auto-research-loop.svg:130`** (per-run artifacts panel) — `leakage_report.json` line replaced with italicized `(leakage state inline on jsonl)`.
8. **`figures/fig2-auto-research-loop.svg:172`** (launcher-owns box) — `gate_trajectory · leakage_report` → `gate_trajectory.json`.

After this pass, `grep -l leakage_report` returns zero non-REVIEW files.

### 2.2 30-run provenance (REVIEW-4 §3.5)

Verified state:
- Current `experiments.jsonl`: 12 rows (11 xattn + 1 text_only, all `metric_version: 5`, all `status: ok`).
- Archived `experiments.jsonl.pre_v5_20260521T035735Z`: 18 rows (10 xattn + 2 cpt_light + 2 lora_text + 2 structured_as_text + 2 event_only; mostly `metric_version: 2`; statuses include 15 `ok`, 1 `failed`, 1 `PASS_smoke`, 1 `PASS_full`).
- Union: **30 rows = 21 cross-attention + 9 baselines**.

The pre-existing claim was "30 experiments recorded in `src/auto_research/experiments.jsonl`, comprising 22 cross-attention runs and 8 baseline runs". Both the location ("in `experiments.jsonl`") and the decomposition (22+8) were inaccurate.

**Edit (02-harness §9, line 255):** rewrote the bullet to:

> "30 experiments total across the union of the current `experiments.jsonl` (12 rows, all `metric_version: 5`) and the archived `experiments.jsonl.pre_v5_20260521T035735Z` (18 rows, mostly `metric_version: 2`; archived at the v5 schema reset on 2026-05-21). The two ledgers were split rather than merged because the v3-era rows used incompatible field shapes (no `v5_adv_*` columns, no `clean_eval_*` columns); the archive preserves them verbatim for audit. By `arm`: 21 cross-attention runs (10 archived + 11 current v5-schema) and 9 baseline runs (2 `cpt_light`, 2 `lora_text`, 2 `structured_as_text`, 2 `event_only`, 1 `text_only` v4 seed). One archived row carries a `failed` status (`exp_xa_round1_005`, hung in Round-1, retried as `exp_xa_grid_014`); two carry `PASS_smoke` / `PASS_full` markers."

The Abstract's "30 cross-attention and baseline configurations" and §1.1 Contribution 1's "30 experiments" are unchanged — they're at the right level of compression for the headline and don't conflict with the now-detailed §9 decomposition.

### 2.3 GPU lockfile path (REVIEW-4 §4.1)

Code uses `/workspace/.gpu.lock` (period). Two single-character edits:

9. **`02-agentic-experiment-harness.md:67`** (§3.4 Step 4) — `/workspace/.gpu_lock` → `/workspace/.gpu.lock`.
10. **`figures/fig2-auto-research-loop.svg:75`** — same fix in the loop-step legend.

After this pass, `grep -nE "\.gpu_lock"` (with underscore, no period) returns zero non-REVIEW results.

### 2.4 Three-halt-mechanism distinction (REVIEW-4 §4.2)

Added a new subsection **`02-agentic-experiment-harness.md §6.3`** between the existing §6.2 ("max_gate_magnitude halt-floor tuning") and the section break. The new subsection explains why `sweep_state.yaml` shows `halted: false` despite the prose describing the Phase-2 stop as having fired, and distinguishes three halt mechanisms:

1. **Launcher-recorded halts** — rule-based; writes `sweep_state.yaml::halted: true`. Examples: v3 convergence halt, NaN cascade, GPU-hours cap.
2. **Instruction-level / agent-queue stops** — judgment-based; no state-file write. Example: v5's Phase-2 stop after Phase-2 alternatives failed to beat the winner by ≥0.005.
3. **`agent_tick.sh` stale-tick auto-stop** — liveness-based; writes a halt flag and calls `runpodctl podStop`. Safety net for runaway pods.

The closing sentence gives the reader a mental model and tells them which file to consult for each kind of halt:

> "Reading `sweep_state.yaml` alone will tell you whether (1) fired; you have to read `AGENT_INSTRUCTIONS.md` and the recent `notes.md` to see whether (2) fired; and the tick log (`/workspace/agent_tick.log`) is the source of truth for (3)."

The existing "Phase-2 stop rule fired" references in 00-main §4.3, 00-main §7, 04 §5.2, 04 §8.4, and 02 §8 are now backed by §6.3 and don't require editing.

---

## 3. Cross-document consistency check after this pass

| Claim | Pre-fix state | Post-fix state |
| --- | --- | --- |
| `leakage_report.json` is a per-run artifact | claimed in 6 prose sites + 2 SVG | replaced in all 8 with inline-on-jsonl description |
| "30 experiments in `experiments.jsonl`" | misleading (12 current + 18 archived) | now describes union explicitly with archive provenance |
| "22 xattn + 8 baselines" | inaccurate (actual 21+9) | corrected to "21 xattn + 9 baselines" with the family decomposition spelled out |
| GPU lockfile path | `.gpu_lock` (wrong) in 2 sites | `.gpu.lock` (matches code) in both |
| Phase-2 stop and `sweep_state.yaml: halted: false` | reader-visible contradiction | resolved by §6.3 distinguishing 3 halt mechanisms |

Zero remaining REVIEW-4 items.

---

## 4. Cumulative diff across REVIEW-4 → REVIEW-7

Documents touched:
- `00-whitepaper-main.md` — 4 sites for V4 numbers (REVIEW-5), 4 sites for V5 numbers + hn_recovery band (REVIEW-6), 2 sites for leakage wording (REVIEW-7). 10 distinct edits total.
- `04-cross-attention-experiments.md` — 5 sites for V4 numbers, 4 sites for V5 numbers + hn_recovery band. 9 edits total. No edits this pass.
- `01-data-curation-and-distribution.md` — 1 site for leakage wording (REVIEW-7).
- `02-agentic-experiment-harness.md` — 1 site for ownership table leakage wording, 1 site for gpu_lock path, 1 site for 30-run provenance, 1 new subsection (§6.3) for 3-halt distinction. 4 edits this pass.
- `03-eval-strategy.md` — 2 sites for leakage wording (REVIEW-7).
- `figures/fig2-auto-research-loop.svg` — 2 sites for leakage wording, 1 site for gpu_lock path. 3 edits this pass.
- `figures/fig4-sweep-results.svg` — 5 bars updated (REVIEW-6). No edits this pass.

Total edit count across all three passes: **34 surgical edits** spanning all five `.md` files and both affected SVGs.

---

## 5. What did *not* change qualitatively across the REVIEW-4 → REVIEW-7 arc

The substantive findings of the paper are unchanged. Specifically:

- **The v3 → v4 → v5 narrative arc.** v3 null → root-cause audit → v4 data pivot → CI-separated win → v5 robustness sweep + ceiling. Same story.
- **Q1 PASS A.** Cross-attention CI-separated from text-only on adversarial fraud families (`phish_takeover` recall 0.11 → 1.00, `phish_takeover_mfa_phished` recall 0.00 → 0.97). Same finding; just-corrected numbers.
- **Architectural ceiling on `hn_recovery_high_amount`.** ~0.42–0.45 FPR across non-pathological v5 configurations; no architectural dial moves it. Same finding; band claim now properly qualified for the fastlr edge case.
- **Methodology framing.** Agentic harness + bootstrap CI + tie-aware metric. Unchanged.
- **Limitations.** Synthetic-only, single LM, gates never reached the Flamingo "open" target, single adversarial-legit family, compute budget unused, single-engineer POC. Unchanged.
- **References, generalizing-the-architecture section, generalizing-the-harness section.** Unchanged.

What changed:
- Every V4 and V5 quantitative claim now derives from `runs/exp_*/ci_report.json` or `experiments.jsonl` on disk.
- The hn_recovery band claim is properly qualified (non-pathological runs; fastlr excursion noted).
- The leakage-state mechanism is described accurately (inline on jsonl, not as a separate per-run file).
- The 30-run count has a provenance paragraph that explains the current-vs-archived ledger split.
- The GPU lockfile path matches the code.
- The `sweep_state.yaml: halted: false` vs Phase-2-stop apparent contradiction is resolved by §6.3.

---

## 6. Verdict

**Publish-ready, modulo a final pre-send proofread.**

REVIEW-3 declared "publish-ready" prematurely. REVIEW-4 correctly identified the artifact-grounding gap. REVIEW-5 (V4), REVIEW-6 (V5 + Figure 4), and REVIEW-7 (this pass: wording + path corrections) close that gap.

The remaining risk before external send is the standard pre-send proofread risk:
- Re-render the figures to confirm the SVG edits look correct visually (especially Bar 8 slowlr's much shorter bar in Figure 4).
- One human read-through end-to-end to catch any prose that drifted out of sync with the corrections.
- Final spell-check / typography pass.

None of those require artifact re-reads. The paper is technically correct against the on-disk state as of this pass; it just needs one final reader sweep before it leaves the building.

### Suggested verification before send

```bash
# Confirm leakage_report.json is no longer claimed anywhere outside REVIEW files
grep -rL leakage_report whitepaper/ --include='*.md' --include='*.svg' | grep -v REVIEW

# Confirm gpu_lock (with underscore) is gone outside REVIEW files
grep -rEn "\\.gpu_lock" whitepaper/ --include='*.md' --include='*.svg' | grep -v REVIEW

# Confirm V5 numbers in 04 §5.1 table match experiments.jsonl
python3 -c "
import json
from pathlib import Path
for run in Path('src/auto_research/runs').glob('exp_v5_*'):
    cip = run / 'ci_report.json'
    if not cip.exists(): continue
    d = json.load(cip.open())
    v5 = d.get('stripped', {}).get('v5_adv_error', {})
    if 'point' in v5:
        print(run.name, round(v5['point'],4), '[', round(v5['ci_lo'],4), round(v5['ci_hi'],4), ']')
"
```

All three checks should return clean.
