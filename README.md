# Cross-Attention for PayPal ATO — Journey Log

3-day POC. The journey matters more than the result.

Plan: `PLAN.md` (v3).
Runbook: `RUNBOOK.md`.

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

*Filled in by the agent at end of Day 3.*

### Sweep round-2 results
*Round-2 + stress run.*

### Top-3 medium eval (50k)
*R@FPR=0.1% with CIs, per-journey, agent-vs-human differential.*

### (Optional) Top-1 large eval (100-200k)
*Only if Day-3 Hr 11 was reached on schedule.*

---

## Final synthesis

### The v3 question answered

> After controlling for token leakage, narrative leakage, structured-stream parity, an event-only classifier baseline, and reported with bootstrap CIs across three eval modes — did cross-attn add **classification** lift, or is its value confined to **explanation/grounding**?

*Answer here.*

### Integration friction catalog
*The journey artifact. Every place an engineer would burn time.*

### Gates story
*Did the cross-attn gates actually open? When? On which configs?*

### Per-baseline deltas (with CIs)
- vs CPT-light:
- vs LoRA-text:
- vs structured-as-text (load-bearing):
- vs event-only classifier (load-bearing — does the LM matter at all?):

### Per-journey breakdown
- clean / cred_stuff / sim_swap / phish_takeover / malware_rat / mule_chain / hn_travel / hn_large_purchase / hn_account_recovery

### Per-actor differential
- human vs agent-driven journeys

### Hard-negative FPR
- hn_travel, hn_large_purchase, hn_account_recovery

### Day-4 recommendation: extend / pivot / stop
*2-3 lines of rationale.*

### Concrete next-steps for a real PayPal-internal POC
*Tied back to `.claude/tasks/cross-attn-ato/next-steps-checklist.md` — which assumptions held, which broke.*
