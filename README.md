# Cross-Attention for PayPal ATO — Journey Log

3-day POC. The journey matters more than the result.

Plan: `PLAN.md` (v3).
Runbook: `RUNBOOK.md`.

> **Status — 2026-05-18: Expanded Sweep in progress; the Day-4 "Pivot, then stop"
> recommendation below is superseded.** After the Day-3 synthesis closed
> with 8 valid x-attn runs and the `zero_gate_activation` halt, the user
> authorized a four-phase Expanded Sweep (LR/warmup → seq4096 stress →
> grid completion → conditional rank capacity) to probe whether
> training-dial perturbations unlock gate learning before declaring
> architecture invariance on this surface. `configs/budget.yaml` lifted
> `max_experiments` 12 → 999 and disabled both `zero_gate_activation`
> and `convergence` halts for the duration; `nan_cascade` and the GPU-
> hours cap remain the real stops. Source of truth:
> `.claude/tasks/xattn-expanded-sweep-plan.md`. First arm
> (`exp_xa_lr_009`: every_8 / slots=64 / gate=small_0.01, lr=3e-4,
> warmup=100) is training on the pod as of this writing. The Final
> Synthesis and Day-4 recommendation below will be re-written by the
> auto-loop agent when the expansion closes (early-exit success,
> Phase-3 completion, `nan_cascade`, or GPU-hours cap — whichever fires
> first).

---

## Day 0 — Pre-flight (this directory, before GPU)

*Filled in by the human at end of Day 0.*

---

## Day 1 — Foundation: data + CPT-light baseline

*Filled in by the human at end of Day 1.*

### Hr 0-2: Tokens + fencer + bucketer
*Pending.*

### Hr 2-6: Vertical slice
*Pending. Make-or-break block.*

### Hr 6-10: Scale + Stage-0 CPT-light + merge
*Pending.*

### Hr 10-14: CPT 3-mode eval + writeup
*Pending.*

### Day-1 friction log
*Bullet list, what surprised us, what broke.*

### Day-1 metrics snapshot
*qwen3-8b-cpt-light-merged eval AUC across three modes, per-journey breakdown, leakage audit.*

---

## Day 2 — Architecture + first sweep batch

Closed by the auto-research agent on 2026-05-17 at convergence-halt
(`halt_reason: "convergence: no worst-family HN-FPR improvement >= 0.005
over last 4 x-attn runs (first=0.0524, best=0.0524)"`). 6 valid x-attn runs
in `experiments.jsonl` (1 smoke + 5 round-1 cells + 1 failed cell #5).
Per-run interpretation lives in `runs/exp_xa_*/notes.md`; durable Day-2
evidence (leakage / metric corrections) is in `docs/day-2-results.md`.

### Architecture surgery friction

- **Blackwell-arch image pinning** (`review/010-blackwell-compat-patch`):
  H100 image only worked after bumping `bitsandbytes` to 0.45+ and pinning
  CUDA 12.4. Default RunPod image's bnb 0.43 silently fell back to a
  paged-optimizer-incompatible path on Hopper; preflight script in
  `scripts/preflight_xattn.py` now hard-fails on that mismatch before
  `accelerate launch` ever loads weights.
- **Stage-0 LoRA merge as a hard prerequisite** (review 005, finalized in
  Path A Batch 4): cross-attention training requires the CPT-light LoRA
  *merged into base* (`/workspace/checkpoints/qwen3-8b-cpt-light-merged`),
  not a live PEFT adapter — otherwise `lora_r_on_q=16` ends up stacking on
  the Stage-0 adapter and gates train against a moving target. The merge
  script is deterministic; the friction was discovering this the hard way
  via early gate-magnitude drift on a non-merged checkpoint.
- **Narrator throughput** (commits `a39cc5f`, `4fddbf8`, `f5c4ce7`): OpenAI
  gpt-5.4-nano became the default narrator only after fixing two
  provider-specific quirks — the gpt-5.4 family requires
  `max_completion_tokens` (not `max_tokens`), and serial calls couldn't
  saturate the 200 USD budget in any reasonable wall-clock; ThreadPoolExecutor
  concurrency landed in `a39cc5f`. Narrator throughput is the binding
  constraint on dataset regeneration, not training.
- **Gate-init floor tuning** (`src/auto_research/configs/budget.yaml`
  lines 22-29): the original `zero_gate_activation.magnitude_threshold:
  0.05` was too aggressive — `gate_init=small_0.01` initializes gates at
  exactly 0.01, and 1500 steps of training only lifted max-gate-magnitude
  to ~0.0106-0.0112. The threshold was lowered to 0.005 (still catches a
  true collapse to zero) after exp_xa_smoke_001 and exp_xa_round1_001
  both landed in the 0.010-0.011 band — gates are learning to use
  cross-attention sparsely, not staying dead.
- **Baseline-eval metric correction mid-Day-2** (reviews 018/019/020,
  `docs/day-2-results.md`): the sklearn-cliff `recall_at_fpr` rule landed
  `event_only` at achieved-FPR=0.114% while the LM baselines hit 0.91-0.97%,
  fabricating a 5-7x advantage that vanished under the tie-aware
  exact-target metric on the clean eval. Required a v2 rescore of all
  four pre-correction rows in `experiments.jsonl`; sweep ranking is now
  filtered to `metric_version >= 2` at read-time. Cost: most of Day-2's
  second half. Caught before Task #37 (full x-attn training against the
  wrong leader) launched.
- **Train/eval narrative leakage on synthetic data** (also docs/day-2):
  10.7% of eval rows shared narrative text with train, concentrated in
  hn_large_purchase (35.3%) and hn_account_recovery (16.4%). Mechanism
  is narration caching by `(structured_events_hash, model, temp)` before
  the split. Clean-eval mask in `eval/leakage_checks.py` drops the 534
  affected rows; pre-narration structured-events-hash stratification +
  post-narration text-hash dedup invariant now in `data/gen/build_dataset.py`
  prevent future regenerations from reintroducing it.

### Baseline metrics (95% CIs on 5k fast eval, stripped mode, metric_version 2 on clean eval n=4466)

Primary: worst-family HN-FPR @ 1% legit FPR (lower is better; tie-aware
exact-target, per `docs/day-2-results.md` §3 Finding 2). AUC shown as a
sanity column only — it saturates at 1.0 on every variant (`PLAN.md` §Risks
flagged this; Day-1 confirmed it).

| Baseline | HN-FPR-worst [CI] | HN-FPR-mean [CI] | AUC (sanity) | Notes |
|---|---|---|---|---|
| CPT-light-merged (`exp_stage0_001`) | not directly comparable | not directly comparable | 1.00 | v1-only row; pre-correction metric on full 5k eval. See `experiments.jsonl` for v1 per-family numbers. |
| LoRA-text (`exp_baseline_lora_text_v2`) | 0.0701 [0.0564, 0.0847] (hn_large_purchase) | 0.0291 [0.0268, 0.0316] | 1.00 | Loses on hn_large_purchase — text alone can't separate legit large purchases from the hard-negative family. |
| structured-as-text (`exp_baseline_structured_as_text_v2`) | **0.0507 [0.0408, 0.0635]** (hn_account_recovery) | 0.0262 [0.0242, 0.0283] | 1.00 | The load-bearing baseline. Most balanced — no family explodes, none is exactly zero. |
| event-only classifier (`exp_baseline_event_only_v2`) | 0.0730 [0.0667, 0.0799] (hn_account_recovery) | 0.0243 [0.0222, 0.0266] | 1.00 | Zero on hn_large_purchase and hn_travel; concentrated failure on hn_account_recovery. tie_fraction=4.7% — pathological under non-tie-aware metrics. |

### Sweep round-1 results

Spread across the 3×2 `insertion_pattern × resampler_slots` grid at
`gate_init=small_0.01`, `encoder=small_transformer`. Cell #5 (every_8 /
slots=128) was marked failed by user during a Blackwell-image hiccup; not
retried because cells #4 and #6 already demonstrated the slots dial was
neutral. Per-run interpretation in `runs/exp_xa_round1_*/notes.md`.

| exp_id | config (pattern / slots) | HN-FPR-worst [CI] | HN-FPR-mean | gate_max | final_loss |
|---|---|---|---|---|---|
| exp_xa_round1_001 | every_4 / 64 | 0.0572 [0.0455, 0.0691] | 0.0258 | 0.0106 | ~1.2 |
| **exp_xa_round1_002** | **every_8 / 64** | **0.0524 [0.0420, 0.0647]** | **0.0262** | **0.0112** | **1.150** |
| exp_xa_round1_003 | late_only / 64 | 0.0586 [0.0460, 0.0683] | 0.0256 | 0.0109 | 1.308 |
| exp_xa_round1_004 | every_4 / 128 | 0.0608 [0.0481, 0.0724] | 0.0254 | 0.0112 | 1.367 |
| exp_xa_round1_005 | every_8 / 128 | — | — | — | failed (marked) |
| exp_xa_round1_006 | late_only / 128 | 0.0604 [0.0472, 0.0709] | 0.0255 | 0.0109 | 1.368 |

Round-1 leader: `exp_xa_round1_002` (every_8 / slots=64) at worst-family
HN-FPR-stripped **0.0524 [0.0420, 0.0647]**. The 100-step smoke
(`exp_xa_smoke_001_v2`, 0.0537) is within the leader's CI — additional
training did not produce a meaningful lift on the worst-family bottleneck.

**Gates story.** Every round-1 run (and the smoke) cleared the 0.005
halt floor at step 1500, with `max_gate_magnitude` clustered tightly in
0.0106-0.0112. No run came close to a zero-gate collapse; no run got
gates to commit hard either. This is consistent with cross-attention
learning to use the side-stream sparsely — a small but non-trivial
fraction of tokens, not "the gate is dead" and not "the gate is wide
open." Whether the sparsity is structural (model found a small useful
signal) or a symptom of the synthetic data being too easy on the LM side
(no pressure to lean on the resampler) is not separable from this
surface alone — that's a Day-3+ medium-eval question.

**Architectural dial reads:**

- **insertion_pattern**: slots=64 row (cells 1-3) lands every_8 ≤
  every_4 ≤ late_only on worst-family HN-FPR, but CIs overlap heavily
  and mean HN-FPR is within 0.001 across all three. Insertion density
  past 3 layers (every_8) is neutral on this surface.
- **resampler_slots**: every direct slots=64 vs slots=128 pair (every_4:
  0.057 vs 0.061, late_only: 0.059 vs 0.060) is tied within CIs and
  trends *mildly worse* at slots=128. Most likely: the clean-eval
  surface is too easy / AUC-saturated to discriminate between 64- and
  128-slot capacity.
- **Bottleneck family**: `hn_account_recovery` is worst for *every*
  x-attn run at ~0.052-0.061. hn_large_purchase mid (~0.016-0.026),
  hn_travel zeroed. The failure mode is the same family regardless of
  architecture. The data-shaped saturation hypothesis (`docs/day-2-results.md`
  §3 Finding 3) is strengthening: worst-family HN-FPR is bottlenecked
  upstream by data signal in `hn_account_recovery`, not by the
  insertion_pattern × slots dial.

### Current leader vs each baseline (worst-family HN-FPR-stripped; tiebreak mean)

Leader: `exp_xa_round1_002`, worst 0.0524 [0.0420, 0.0647], mean 0.0262.

- **vs CPT-light-merged (`exp_stage0_001`)**: not directly comparable —
  v1-only row, no v2 rescore against the clean eval surface. Day-1
  recorded CPT-light hn_account_recovery v1 = 0.0615 on the leaky 5k
  LLM eval. Treat as qualitatively in the same band as round-1 x-attn;
  no decisive claim possible.
- **vs LoRA-text v2** (0.0701 [0.0564, 0.0847]): leader nominally
  better by **-0.018 absolute** on worst-family, but CIs overlap
  (lora_lo=0.0564 vs xattn_hi=0.0647). Marginal separation, not decisive.
- **vs structured-as-text v2** (0.0507 [0.0408, 0.0635]) — **the
  load-bearing comparison**: leader is **+0.0017 absolute *worse*** on
  worst-family; CIs heavily overlap ([0.0420, 0.0647] vs [0.0408, 0.0635]).
  Cross-attention does not beat the structured-as-text concat baseline
  on this surface within 95% bootstrap CIs.
- **vs event-only v2** (0.0730 [0.0667, 0.0799]): leader better by
  **-0.021 absolute**; CIs barely separated (event_lo=0.0667 vs
  xattn_hi=0.0647 — marginal non-overlap). The LM-based variants do
  outperform pure event-only on worst-family, consistent with
  `docs/day-2-results.md` §4. This is one of only two CI-separated pairs
  in the table — the other being structured-as-text vs event-only.

### Open questions for Day 3

1. **Does the round-1 leader survive medium eval (50k)?** Worst-family
   point estimates on the 5k surface are bunched in 0.052-0.061 with
   ±0.012 CIs that swallow the differences. A 10x larger eval should
   tighten CIs enough to either confirm the structured-as-text tie or
   reveal a separation — answers the load-bearing question.
2. **Is `hn_account_recovery` truly a data-shape ceiling?** Every
   architecture variant + every baseline (except `lora_text`, which
   fails on hn_large_purchase instead) lands worst-family on
   hn_account_recovery in the 0.05-0.07 band. If a 50k eval keeps the
   same family as bottleneck at the same magnitude, the ceiling is
   structural to the synthetic generator (Finding 3) and no
   architectural dial will move it.
3. **Should Round-2 (gate_init=zero) run?** Budget remaining: 12 - 6 =
   6 x-attn slots; GPU hours used 6.166/18.000. The convergence halt
   fired because worst-family didn't move ≥0.005 across last 4 runs.
   Round-2 perturbations might still illuminate the gate-init
   sensitivity question even if they don't move the leaderboard. Punt
   the call to the user — auto-loop has stopped per halt-condition
   policy.
4. **Stress run (`stress_run: true`, steps=3000, seq_len=4096)** — not
   launched because convergence-halt fired before Round-2. Would test
   whether longer training + longer context lifts worst-family below
   the structured-as-text bar. Worth one slot if user extends the budget.

---

## Day 3 — Round-2 sweep + analysis + synthesis

Closed by the auto-research agent on 2026-05-18 at zero-gate-cascade halt
(`halt_reason: "zero_gate_activation: last 2 x-attn runs had max gate <
0.005"`). 8 valid x-attn runs total (1 smoke + 5 round-1 cells + 2 round-2
perturbations + 1 failed round-1 cell #5). Per-run interpretation in
`runs/exp_xa_round2_*/notes.md`.

The convergence-halt that closed Day-2 on 2026-05-17 was disabled in
`src/auto_research/configs/budget.yaml` (commit comment lines 32-44) before
Day-3 so Round-2 (gate_init=zero perturbations) could run. Convergence halt
keys off "no worst-family HN-FPR improvement ≥0.005 over last 4 x-attn
runs" — but with Round-1's leader (round1_002) sitting in the window's
first slot, it was impossible for any later sibling to beat it by 0.005;
the halt fired before any zero-init perturbation could probe whether
Round-1's gate magnitudes were init-bias carried. zero_gate and NaN
cascades remained as the real stop conditions, and one of them
(zero_gate) fired cleanly on the second Round-2 run.

### Sweep round-2 results

Round-2 was the two top-N siblings of Round-1 with `gate_init=zero` (the
unexplored half of the gate_init dial; everything else fixed).

| exp_id | config (pattern / slots / gate) | HN-FPR-worst [CI] | HN-FPR-mean | gate_max | final_loss | wall (min) |
|---|---|---|---|---|---|---|
| exp_xa_round2_007 | every_8 / 64 / zero | 0.0608 [0.0475, 0.0716] | 0.0254 | **0.00385** | 1.317 | 46.6 |
| exp_xa_round2_008 | every_4 / 64 / zero | 0.0594 [0.0470, 0.0708] | 0.0256 | **0.00412** | 1.414 | 47.6 |

Matched-architecture pairs (the comparison Round-2 was designed for):

| pattern / slots | gate=small_0.01 (R1)             | gate=zero (R2)                   | CI overlap? |
|---|---|---|---|
| every_8 / 64 (top-1) | 0.0524 [0.0420, 0.0647], max-gate 0.0112 | 0.0608 [0.0475, 0.0716], max-gate 0.0038 | yes, heavily |
| every_4 / 64 (top-2) | 0.0572 [0.0455, 0.0691], max-gate 0.0106 | 0.0594 [0.0470, 0.0708], max-gate 0.0041 | yes, heavily |

Both zero-init runs landed `max_gate_magnitude < 0.005` at step 1500 →
two consecutive xattn runs below the halt threshold → launcher tripped
`zero_gate_activation`. Sweep stopped after `n_xattn_runs: 8`,
`gpu_hours_used: 7.735/18.000`. Round-3 stress run (`stress_run: true`,
steps=3000, seq_len=4096) was not proposed per halt policy.

**Gates-from-zero finding.** With `gate_init=zero`, gates moved from
0.0 to ~0.004 in 1500 steps — below the (already-lowered) 0.005 "open"
threshold. With `gate_init=small_0.01`, gates moved from 0.01 to ~0.011
in the same training budget. Movement-from-init is negligible in both
regimes; gates ride whatever bias they start with. This rules out the
generous reading of Round-1 ("gates learned to use cross-attn sparsely")
in favor of the strict reading: Round-1's max-gate magnitudes were
init-bias-carried, not learned, on this dataset / lr / step budget. A
denser insertion (every_4, 2× the gates getting gradient signal) did not
help — round2_008 max_gate is essentially identical to round2_007's.

**HN-FPR-from-zero finding.** Despite ~3× difference in max-gate
magnitude between the matched small_0.01 and zero pairs (0.011 vs 0.004),
worst-family HN-FPR is statistically tied within 95% bootstrap CIs in
both pairings. The base CPT-light-merged LM is doing essentially all of
the discrimination on this 5k clean-eval surface; cross-attn provides at
most marginal lift that the 5k eval cannot detect. This holds across
both architectures tested in Round-2.

**Failure-mode invariance.** `hn_account_recovery` remains the worst
family for both Round-2 runs (0.06 band), `hn_large_purchase` mid
(~0.016-0.017), `hn_travel` zeroed. This now holds across **all 8 valid
x-attn runs** (smoke + 5 round-1 + 2 round-2) regardless of insertion
pattern, slots, or gate init. Failure mode is invariant to every dial
exercised — strengthening the Day-2 "ceiling is upstream of architecture"
hypothesis (`docs/day-2-results.md` §3 Finding 3).

### Top-3 medium eval (50k)

*Not run.* Halt fired before the medium-eval slot in the budget was
allocated. The medium-eval surface (`data/eval_medium_50k/` is present in
the repo but no `eval_medium.jsonl`-based runs were launched against it)
remains the only credible way to separate the structured-as-text vs
cross-attn tie observed on 5k. Recommended as the **first Day-4 action**
if the user extends the budget.

### (Optional) Top-1 large eval (100-200k)

*Not run.* Out of scope without medium-eval results first.

---

## Final synthesis

### The v3 question answered

> After controlling for token leakage, narrative leakage, structured-stream parity, an event-only classifier baseline, and reported with bootstrap CIs across three eval modes — did cross-attn add **classification** lift, or is its value confined to **explanation/grounding**?

**No detectable classification lift on the 5k clean-eval surface.** The
Round-1 leader (`exp_xa_round1_002`, every_8 / slots=64 / gate=small_0.01)
landed worst-family HN-FPR-stripped 0.0524 [CI 0.0420, 0.0647]. The
load-bearing baseline `structured-as-text v2` is at 0.0507 [CI 0.0408,
0.0635] — leader is +0.0017 absolute **worse** on the point estimate,
CIs heavily overlap. Cross-attn does not separate from concatenating the
structured stream into the prompt within 95% bootstrap CIs.

The Day-3 Round-2 perturbations (`exp_xa_round2_007`, `exp_xa_round2_008`)
collapsed the alternative reading. Switching `gate_init` from `small_0.01`
to `zero` left `max_gate_magnitude` near zero at step 1500 (~0.0038-0.0041
vs ~0.011-0.012) yet produced **statistically tied** HN-FPR on both
matched architectures — so the ~3× max-gate movement seen in Round-1 was
init-bias carried, not learned cross-attn signal. The base CPT-light-merged
LM is doing essentially all of the classification work on this surface;
cross-attn-via-gated-residual is contributing at most marginal lift the
5k eval cannot detect.

Whether cross-attn's value is confined to explanation/grounding cannot
be answered from this surface — HN-FPR / AUC measure classification, not
grounding-quality. The 5k surface is consistent with: cross-attn is
neutral-to-mildly-useful for classification on this synthetic ATO task,
and any usefulness it has is upstream of the discrimination metric this
POC is wired to measure.

### Integration friction catalog

The journey artifact — every place an engineer would burn time, in
roughly the order we hit them:

1. **Blackwell / bitsandbytes incompatibility** (review 010,
   `RUNBOOK.md` §Blackwell): default RunPod image with bnb 0.43 silently
   fell back to a non-paged optimizer on Hopper, dropping ~2× throughput
   without an error message. Fix: pin bnb≥0.45 + CUDA 12.4 + the
   preflight hard-fail in `scripts/preflight_xattn.py`.
2. **Stage-0 LoRA-merge-before-x-attn precondition** (review 005, Path A
   Batch 4): cross-attn training against a live PEFT-wrapped Stage-0
   produced a gate-magnitude drift that looked like a learning-rate bug.
   Root cause was `lora_r_on_q=16` stacking on a Stage-0 adapter that
   was itself still updating. Merge-then-x-attn is the only stable
   ordering. Hard-coded in trainer defaults.
3. **paged_adamw_8bit / Accelerate / DataParallel interaction**
   (`scripts/preflight_xattn.py` and `src/train/accelerate_configs/`):
   accelerate launched with multiple processes silently fell back from
   paged-8bit to fp32 optimizer states; preflight now hard-asserts
   single-process + paged-8bit before the trainer touches weights.
4. **Synthetic-narrator throughput** (`a39cc5f`, `4fddbf8`): OpenAI
   gpt-5.4-family quirks — `max_completion_tokens` not `max_tokens`, and
   serial calls couldn't saturate the 200 USD narrator budget. Concurrent
   `ThreadPoolExecutor` was the only way to make end-of-Day-1 budget.
5. **Narrative leakage through narrator caching**
   (`docs/day-2-results.md`): the narrator cached by
   `(structured_events_hash, model, temp)` before the train/eval split,
   so distinct journeys with the same structured-events footprint shared
   text across the split. 10.7% of eval text-overlapped with train,
   concentrated 35% in `hn_large_purchase` and 16% in `hn_account_recovery`.
   Caught by `eval/leakage_checks.py`; clean-eval mask drops the 534
   affected rows and `data/gen/build_dataset.py` now stratifies on
   pre-narration structured-events-hash.
6. **AUC saturation** (`PLAN.md` §Risks → confirmed Day-1, formalized
   review 013 finding #1): AUC saturates at 1.0 on every model variant
   on this synthetic surface. Ranking and halt logic both keyed off
   AUC-stripped initially; both had to migrate to worst-family HN-FPR
   at FPR=1% mid-Day-2. The `metric_version: 2` rescore of 4 pre-fix
   experiments rows is in `experiments.jsonl`.
7. **`recall_at_fpr` sklearn cliff** (reviews 018/019/020): the naïve
   "first row at sklearn's `_binary_clf_curve` threshold ≥ target_fpr"
   rule landed `event_only` at achieved-FPR=0.114% while LM baselines
   hit 0.91-0.97%, fabricating a 5-7× event-only advantage that
   vanished under the tie-aware exact-target metric. Cost: most of
   Day-2's second half. New metric in `eval/score_risk.py`,
   `eval/bootstrap_ci.py`; cleared via the v2 rescore.
8. **Convergence-halt premature firing** (`budget.yaml` lines 32-44):
   the convergence halt's window-based "improvement ≥0.005 across last
   4 x-attn runs" fired before Round-2 perturbations could probe gate-
   init sensitivity, because the Round-1 leader sat in slot 1 of the
   window. Disabling convergence-halt (keeping NaN + zero_gate as the
   real stop conditions) was the right surgical fix; zero_gate fired
   cleanly on Round-2 anyway and produced the gate-bias finding.
9. **`max_gate_magnitude` halt-floor tuning** (`budget.yaml` lines 22-29):
   original 0.05 threshold was too aggressive — `gate_init=small_0.01`
   initializes at exactly 0.01, lifts to ~0.011 in 1500 steps; the
   threshold was lowered to 0.005 to still catch a true zero-collapse.
   Round-2 zero-init then landed at ~0.004 → halt fired correctly, as
   designed.
10. **Launcher / agent ownership split** (review 013 finding #7,
    `src/auto_research/AGENT_INSTRUCTIONS.md`): `experiments.jsonl` and
    `sweep_state.yaml` are launcher-owned, agent reads only. Early
    versions had the agent appending to `experiments.jsonl` directly,
    producing format drift between agent rows and launcher rows. Now a
    clean split.

### Gates story

Did the cross-attn gates actually open? **No, not meaningfully.** Across
all 8 valid x-attn runs:

- `gate_init=small_0.01` × 6 runs (smoke + 5 round-1 cells): max-gate
  magnitude at step 1500 ranged 0.0106-0.0112, i.e. **0.0006-0.0012
  above the 0.01 initialization**. Effective learned movement: ≤10%
  of init magnitude.
- `gate_init=zero` × 2 runs (round-2 perturbations): max-gate magnitude
  at step 1500 was 0.00385 and 0.00412, both below the 0.005 "open"
  threshold. Two consecutive sub-threshold runs tripped the
  `zero_gate_activation` halt.

Gates rode their init bias. Whatever lift cross-attn provided on this
task came from the bias-on-init dot-product through the resampler, not
from any meaningful gradient-driven gate opening in 1500 steps. With the
gate near-zero in round-2 and HN-FPR statistically unchanged, the bias-
driven contribution is also bounded above by "no detectable lift on the
5k eval."

### Per-baseline deltas (with CIs)

All numbers stripped-mode, metric_version=2, clean-eval n=4466.
Leader: `exp_xa_round1_002` (every_8 / slots=64 / gate=small_0.01).

- **vs CPT-light-merged** (`exp_stage0_001`): not directly comparable —
  v1-only row, no v2 rescore. Qualitatively in the same 0.05-0.07 band
  on the leaky pre-clean eval. No claim possible.
- **vs LoRA-text v2** (worst 0.0701 [0.0564, 0.0847] on
  `hn_large_purchase`): leader -0.0177 absolute on worst-family,
  CIs **overlap** (`lora_lo=0.0564 < xattn_hi=0.0647`). Different worst
  family (LoRA-text fails on hn_large_purchase; x-attn fails on
  hn_account_recovery), so the comparison is more apples-to-oranges
  than the table suggests.
- **vs structured-as-text v2** (worst 0.0507 [0.0408, 0.0635] on
  `hn_account_recovery`) — **load-bearing**: leader +0.0017 absolute
  **worse**, CIs heavily overlap. **No separation.** Same worst family.
  Same magnitude. This is the headline result.
- **vs event-only classifier v2** (worst 0.0730 [0.0667, 0.0799] on
  `hn_account_recovery`): leader -0.0206 absolute, CIs **marginally
  non-overlapping** (`event_lo=0.0667` vs `xattn_hi=0.0647`,
  separation = 0.0020). The LM-based variants do outperform pure
  event-only on worst-family. **The LM matters.** But x-attn-on-top-of-LM
  does not separate from feeding the same structured stream as text.

### Per-journey breakdown

Across all 8 x-attn runs (and all 3 LM baselines): `clean`, `cred_stuff`,
`malware_rat`, `mule_chain`, `phish_takeover`, `sim_swap` are all
saturated at AUC=1.0. Discrimination is decided entirely on the three
`hn_*` families. Per-journey AUC is uninformative on this surface —
the only signal is in per-family HN-FPR.

### Per-actor differential

Human vs agent-driven AUCs both 1.0 (saturated). No differential
extractable from the 5k clean-eval surface. The original v3 question
("does cross-attn do better on agent-driven journeys than human?") is
not answerable on this surface; would require a recall-at-low-FPR
analysis on the 50k medium eval, which was not run.

### Hard-negative FPR

The headline numbers. `hn_account_recovery` is the bottleneck for
**every variant we tested** (x-attn × 8, structured-as-text, event-only
classifier, CPT-light qualitative) at 0.05-0.07. `hn_large_purchase`
mid (~0.015-0.026) for x-attn / structured-as-text and zero for
event-only. `hn_travel` zeroed for every variant. This shape held
**invariant** under: insertion_pattern ∈ {every_4, every_8, late_only},
resampler_slots ∈ {64, 128}, gate_init ∈ {zero, small_0.01}. Conclusion:
the worst-family ceiling is upstream of the architectural dials this
sweep exercised — most likely in the synthetic generator's
`hn_account_recovery` template (`docs/day-2-results.md` §3 Finding 3).

### Day-4 recommendation: extend / pivot / stop

**Pivot, then stop.** More architectural sweep on this surface won't
separate cross-attn from structured-as-text within CIs — every dial
tested is invariant to the bottleneck family. The two unlocked next
moves are (1) **medium eval (50k)** with the Round-1 leader + structured-
as-text head-to-head, which can tighten CIs ~3× and either confirm the
tie or expose a separation; (2) **data-side pivot** into the
`hn_account_recovery` generator, since the family ceiling is what's
capping every variant. Run (1) first; if (1) still ties, stop spending
on x-attn architecture and move the budget to (2). The cross-attn
gated-residual mechanism is not a classification-lift lever on this
task at this scale.

### Concrete next-steps for a real PayPal-internal POC

Tied back to `.claude/tasks/cross-attn-ato/next-steps-checklist.md`:

- **Held**: leakage-safety scaffolding (text-hash + structured-events-hash
  dedup), three-mode eval, bootstrap CIs, baselines-as-arms (cpt_light /
  lora_text / structured_as_text / event_only), launcher/agent ownership
  split. These transferred directly to the auto-research workflow with
  no rework.
- **Broke**: AUC-as-primary-metric (saturated; migrated to worst-family
  HN-FPR mid-Day-2); convergence-halt-as-primary-stop (fires on first
  leader-in-window; zero_gate is the only safe stop on this dataset);
  narrator caching (had to add pre-narration stratification); sklearn
  recall_at_fpr (replaced with tie-aware exact-target metric).
- **New for an internal POC**: a *real* medium-eval (50k synthetic +
  10k held-out replay of a real anonymized window) needs to land
  *before* x-attn is touched — the 5k surface saturated every variant
  except on the three HN families, and the HN families are where the
  generator most differs from real data. Without that, the
  architectural sweep is exercising the generator's ceiling, not the
  model's.
