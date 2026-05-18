# Expanded X-Attention Sweep Plan (post-Day-3, revised 2026-05-18)

## Summary

Continue x-attn exploration without redoing any successful experiment.
Reordered after Day-3 to put **training-dial dials FIRST** because the
central scientific question has shifted from *"which insertion pattern
wins?"* to *"can we make the cross-attention gates actually learn to
use the side stream?"*

The Round-2 zero-init runs (round2_007, round2_008) showed gates barely
moved from 0.0 to ~0.004 in 1500 steps — below even the lowered 0.005
"open" threshold. Round-1's "open" gates (~0.011) were init-bias-carried,
not learned. LR / warmup / longer training answer the gate-learning
question more directly than finishing leftover grid cells.

Current successful x-attn leader:

```text
exp_xa_round1_002
every_8 / gate=small_0.01 / slots=64
lr=1e-4 / warmup=500 / steps=1500 / seq_len=2048 / lora_r=16
HN-FPR-worst = 0.0524
```

Comparison bar (the LM baseline x-attn must beat with non-overlapping CIs to justify the architecture):

```text
structured_as_text_v2
HN-FPR-worst = 0.0507 [0.0408, 0.0635]
```

Day-3 finding worth retaining: across all 8 valid x-attn runs to date,
`hn_account_recovery` is the worst family. The failure mode is invariant
to insertion_pattern, slots, and gate_init. Training-dial sweep + stress
are the strongest remaining levers to probe whether the architecture
can move the worst family at all.

---

## Phase 0 — Agent control update (PREREQUISITE, do before any experiment)

This plan only takes effect if the cron-driven agent reads it. The
agent reads `src/auto_research/AGENT_INSTRUCTIONS.md` at every
iteration, NOT this file directly.

**Required**: AGENT_INSTRUCTIONS.md must include an "Expanded Sweep
directive" section that:

1. Supersedes the original Round 1/2/3 heuristic with the phase queue below.
2. Specifies the **dedup tuple** (see "Duplicate Guard" below) verbatim.
3. Specifies the **no-redo rule**: skip any tuple with an existing `status: ok` row.
4. Specifies the **retry-failed-once rule**: failed tuples may be retried ONCE with a new `exp_id`; not twice.
5. Specifies the **early-exit-on-success rule**: if a run produces `max_gate_magnitude ≥ 0.05` AND beats current_best with non-overlapping 95% CIs, stop the mechanical queue and pivot to local perturbations.

Phase 0 is complete when the AGENT_INSTRUCTIONS.md file contains the
expanded-sweep directive AND the next cron tick logs the agent reading
it (visible in agent_tick.log as the agent's reasoning references
"Phase 1" or "exp_xa_lr_*" naming).

---

## Phase 1 — LR / warmup sweep around current best (highest scientific VOI)

Base config = `every_8 / small_0.01 / 64 / encoder=small_transformer / lora_r=16`.
Only LR + warmup move. **Run all three in this order; do not skip:**

```text
exp_xa_lr_009    lr=3e-4   warmup=100   steps=1500   seq_len=2048
exp_xa_lr_010    lr=3e-4   warmup=500   steps=1500   seq_len=2048
exp_xa_lr_011    lr=3e-5   warmup=500   steps=1500   seq_len=2048
```

**Hypothesis being tested:** Round-1 + Round-2 used lr=1e-4 / warmup=500.
Gates moved < 0.005 from init in both regimes. Either (a) the lr is too low
for gates to escape init-bias OR (b) warmup is too long, gating the gates
from any meaningful gradient signal early in training. lr=3e-4 + warmup=100
tests both at once (most aggressive). lr=3e-4 + warmup=500 isolates lr.
lr=3e-5 tests the opposite extreme (could gates open with finer-grained
updates).

**Expected behaviors:**
- If `exp_xa_lr_009` (3e-4 + warmup=100) produces `max_gate >= 0.05` AND beats current_best CI-strictly → **early-exit fires**, the queue stops, agent pivots to local perturbations of THAT cell.
- If all three land with max_gate < 0.05 → strong evidence that gates do not learn on this surface regardless of LR/warmup. The Day-3 finding ("ceiling is upstream of architecture") is reinforced.

Wall-clock per run: ~50 min. Phase 1 total: ~2.5 h.

---

## Phase 2 — One stress run on current-best config

```text
exp_xa_stress_012   stress_run=true   steps=3000   seq_len=4096
                    base = best config after Phase 1 (likely exp_xa_round1_002 if Phase 1 didn't beat it)
```

If Phase 1 produced a new leader, use ITS config as the base for the
stress run. Otherwise use exp_xa_round1_002's config.

**Hypothesis being tested:** PLAN.md's original "Round 3" stress run was
never executed. Longer training + longer context might let gates accumulate
enough gradient signal to actually open, or might surface a worst-family
shift that the 1500-step / 2048-seq runs hid.

**Resource caps:** `stress_run_max_wall_clock_minutes: 150` in budget.yaml
governs this. If it times out, mark failed via the launcher's `--mark-failed`
path; do not retry within this plan.

Wall-clock: up to 150 min. Phase 2 total: ~2.5 h (assuming run completes).

---

## Phase 3 — Finish non-duplicate original-grid cells

Remaining cells from the original 3 × 2 × 2 grid (insertion × gate_init × slots)
that have NOT completed successfully:

```text
exp_xa_grid_013    late_only / zero       / 64
exp_xa_grid_014    every_8   / small_0.01 / 128   (retry of round1_005 — that one HUNG, status=failed)
exp_xa_grid_015    every_4   / zero       / 128
exp_xa_grid_016    every_8   / zero       / 128
exp_xa_grid_017    late_only / zero       / 128
```

Same architecture as their Round-1/2 siblings; only the
insertion_pattern × gate_init × slots tuple changes.

**Per-tuple status check (must match before queuing):**
- `late_only / zero / 64` — no prior row → run as `exp_xa_grid_013`.
- `every_8 / small_0.01 / 128` — round1_005 = `status: failed`. Retry-once allowed → run as `exp_xa_grid_014`.
- `every_4 / zero / 128` — no prior row → run as `exp_xa_grid_015`.
- `every_8 / zero / 128` — no prior row → run as `exp_xa_grid_016`.
- `late_only / zero / 128` — no prior row → run as `exp_xa_grid_017`.

Wall-clock: ~50 min each. Phase 3 total: ~4.2 h.

---

## Phase 4 — Rank-capacity sweep around best config (CONDITIONAL on budget)

```text
exp_xa_rank_018    lora_r_on_q=32
exp_xa_rank_019    lora_r_on_q=64
```

**Conditional gate:** Only enter Phase 4 if `max_gpu_hours - gpu_hours_used >= 1.5`.
With 18-hour cap and ~7.7h used at Day-3 close, Phase 1 (~2.5h) + Phase 2 (~2.5h) + Phase 3 (~4.2h) ≈ **16.9h total**, leaving ~1.1h headroom. **Phase 4 likely will NOT fit** — that's by design, the budget arithmetic enforces stop-when-done.

If by chance Phases 1-3 hit early-exit or partial-budget, Phase 4 cells may run.

Use the best completed config (LR, warmup, steps, seq_len, insertion, slots, gate_init from current leader at time of Phase 4 dispatch) as the base; vary only `lora_r_on_q`.

Wall-clock per run: ~50 min. Phase 4 total (if it runs): ~1.7 h.

---

## Budget arithmetic

| Phase | Experiments | Per-run | Subtotal |
|---|---|---|---|
| 1 | 3 | ~50 min | ~2.5 h |
| 2 | 1 | up to 150 min | up to 2.5 h |
| 3 | 5 | ~50 min | ~4.2 h |
| 4 | 2 (conditional) | ~50 min | ~1.7 h |
| **Total (Phases 1-3)** | **9** | | **~9.2 h** |
| **Total (Phases 1-4)** | **11** | | **~10.9 h** |

Remaining GPU budget at Day-3 close = 18 − 7.735 ≈ **10.3 h**. Phase 4
likely won't fit; Phases 1-3 fit cleanly.

---

## Auto-Loop Control (budget.yaml)

```yaml
max_experiments: 999       # effectively disabled; GPU-hours + early-exit are the real stops
max_gpu_hours: 18

halt:
  nan_cascade:
    enabled: true          # safety; real signal
    consecutive_threshold: 2

  zero_gate_activation:
    enabled: false         # Day-3 disabled — gates legitimately don't move past 0.005
                           # on this surface; halt was firing on a real signal that the
                           # expanded sweep specifically wants to probe further (does
                           # LR/warmup unlock gate learning?)

  convergence:
    enabled: false         # Day-3 disabled — see budget.yaml comment lines 32-44
```

After editing budget.yaml:

```bash
python -c "from scripts.run_next_experiment import update_sweep_state; update_sweep_state()"
python scripts/run_next_experiment.py --halt-check
```

Expected output:

```text
clear to launch
```

---

## Duplicate Guard (must be checked BEFORE writing each config)

For each candidate queue item, compare to ALL prior `status: ok` rows in
`experiments.jsonl` by this exact tuple:

```text
(insertion_pattern, gate_init, resampler_slots, encoder, lora_r_on_q,
 lr, warmup_steps, steps, seq_len)
```

Rules:

- **No-redo:** if any `status: ok` row matches the tuple exactly, SKIP that queue item and move to the next.
- **Retry-failed-once:** if a `status: failed | nan | timeout` row matches the tuple, the agent MAY retry it ONCE with a new exp_id. Tracking: search experiments.jsonl for ALL rows with the same tuple; if there's already a successful row OR two-or-more failed rows for the tuple, skip.
- **Phase 3's `exp_xa_grid_014` is the explicit retry-once-allowed cell** (round1_005 is the failed prior; one retry permitted).

The launcher's `canonical_hash()` (line 208 of `scripts/run_next_experiment.py`)
catches exact-config-hash duplicates and returns exit code 2 ("DUPLICATE").
This is the safety net; the agent should not rely on it — write configs
that pass the tuple check up front.

---

## Early-exit-on-success condition

If any single run records BOTH:

1. `max_gate_magnitude >= 0.05` (the original "gates open" threshold per
   PLAN.md, which the current sweep has never satisfied), AND
2. `hn_fpr_worst_stripped` beats `current_best.hn_fpr_worst_stripped`
   with **non-overlapping** 95% CIs. Concretely: `new_row.hn_fpr_ci_stripped.worst_family.ci_hi < current_best_row.hn_fpr_ci_stripped.worst_family.ci_lo`.

→ **STOP the mechanical queue.** Skip remaining phases.

→ **Pivot:** the next experiment proposal becomes a local perturbation of
the early-exit cell. Vary one dial at a time:
- `insertion_pattern` ±1 step in the {every_4, every_8, late_only} list
- `resampler_slots` toggle between 64 and 128
- `gate_init` toggle between small_0.01 and zero
- `lora_r_on_q` from current to {16, 32, 64}

Run local perturbations until budget exhausts or another early-exit fires
or convergence is judged by the agent (no improvement over 3 perturbations).

The early-exit row must record this in its `notes.md`: state which trigger
fired (gate or HN-FPR) and what local perturbations are next on the queue.

---

## Test Plan (before first cron tick of expanded sweep)

```bash
# 1. Halt check on pod after budget.yaml + sweep_state refresh:
python scripts/run_next_experiment.py --halt-check
# Expected: clear to launch

# 2. Dry-run validation for first config (agent writes it; we verify):
python scripts/run_next_experiment.py --dry-run-validate-clean-eval \
  src/auto_research/runs/exp_xa_lr_009/config.yaml
# Expected: clean_eval_n=4466, clean_eval_dropped=534

# 3. After first run lands, verify row fields:
#    status == ok AND metric_version == 2 AND clean_eval_n == 4466 AND
#    clean_eval_dropped == 534 AND hn_fpr_worst_stripped finite AND
#    hn_fpr_ci_stripped present
```

---

## Final Decision Rule

Stop and write final synthesis when ANY of:

1. `max_gpu_hours >= 18`
2. All Phases 1-3 queue items complete-or-failed (Phase 4 optional)
3. Early-exit rule fires AND its local-perturbation queue exhausts or 3 perturbations show no further improvement

Final writeup (appended to README.md "Final synthesis" section or written
fresh if the existing section is stale) must answer:

```text
1. Did any non-duplicate x-attn config beat structured_as_text_v2 with
   non-overlapping CIs?
2. Did LR/warmup/rank/longer training change the gate story
   (i.e., did max_gate ever reach 0.05+)?
3. Was the Day-3 "ceiling is upstream of architecture" finding
   confirmed or refuted by the expanded sweep?
4. Is the next architectural step (Day-4+) more x-attn variants
   (FT-Transformer encoder, learnable gate_init), harder synthetic
   data (more bucket-combination diversity, mixed-label skeletons),
   or pivot to structured-as-text concat as the chosen production
   architecture?
```
