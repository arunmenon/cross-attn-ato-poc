# Experiment Log — Cross-Attention ATO POC

Running, append-only history of every experiment in this POC. Each entry
captures the hypothesis, the dials moved, the result, and what we learned.
Source of truth for the numbers: `src/auto_research/experiments.jsonl`
(metric_version=2 rows after review 020); this file is the human-readable
companion.

Headline metric across this POC: **worst-family hard-negative FPR at
target legit-FPR = 1%, stripped eval mode** (`hn_fpr_worst_stripped`).
Lower is better; ties broken by `hn_fpr_mean_stripped`. AUC is saturated at
1.0 on every model variant and is a sanity gate only, not the objective.

Last update: 2026-05-18 (post-grid_017 launch, ~5 GPU-hr remaining of 18).

---

## Why a "log" not a "report"

The auto-research loop appends to `experiments.jsonl` on every run. This
markdown shadows it for readers who want context: what was the dial moved,
what was the hypothesis under test, why was the next config chosen?
Updated as new arms land. The README at the repo root holds the
synthesis; this file holds the working history.

---

## Quick index

| Phase | Date | Arms | Status | Headline finding |
|---|---|---|---|---|
| 0. Pre-experimental | 2026-05-15 → 16 | vslice + stage-0 | done | data + Stage-0 CPT-light land; merge succeeds |
| 1. Baselines (v1) | 2026-05-16 | 3 baselines | done | first pass — later invalidated by clean-eval bug (review 018/019) |
| 2. First x-attn smoke | 2026-05-16 | smoke_001 | done | x-attn architecture trains in bf16 without NaN |
| 3. Round-1 grid | 2026-05-17 | 6 (5 ok, 1 hang) | done | architecture dials inert within CIs |
| 4. Round-2 gate=zero | 2026-05-17 | 2 | done | gates from zero stay near zero (~0.004 max) |
| 5. Baseline rescore (v2) | 2026-05-17 | 4 baselines | done | clean-eval correction → leader unchanged but baselines worse |
| 6. Day-3 synthesis | 2026-05-17 | (writeup) | done | x-attn provides no detectable lift; halt fires |
| 7. Expanded Sweep — Phase 1 (LR/warmup) | 2026-05-18 | 3 | done | LR/warmup inert; gate dynamics LR-invariant in 3e-5 → 3e-4 |
| 8. Expanded Sweep — Phase 2 (stress) | 2026-05-18 | 1 | failed | killed by 180m session boundary; F8 v1 → v2 patch |
| 9. Expanded Sweep — Phase 3 (grid completion) | 2026-05-18 | 5 so far | running | gate=zero × architecture matrix nearly complete; all CI-tied |
| 10. Expanded Sweep — Phase 4 (rank capacity) | — | 0 | conditional | only if Phases 1-3 don't surface signal |

---

## Phase 0 — Pre-experimental (Day 1)

### `exp_vslice_001` — vertical-slice smoke

- **Date**: 2026-05-15
- **Hypothesis**: confirm the data + Stage-0 CPT-light + eval pipeline
  works end-to-end on 100 examples before scaling.
- **Config**: tiny dataset (100 narratives), CPT-light LoRA, three-mode
  eval scaffold, leakage scans.
- **Result**: `PASS_smoke` — Stage-0 CPT-light LoRA trained, merged via
  `scripts/merge_stage0_lora.py`, three-mode eval produced
  predictions/metrics/CI per the scaffold.
- **What we learned**: the scaffold works. PII fencing, journey/actor token
  registry, eval-mode dropout, leakage scans, bootstrap CI all wired
  correctly. Cleared us to scale to the 1.5k LLM-narrated set.

### `exp_stage0_001` — Stage-0 CPT-light on the full 1.5k set

- **Date**: 2026-05-16
- **Hypothesis**: CPT-light over the LLM-narrated training set gives a
  text-only baseline strong enough that any x-attn variant has to clearly
  beat it (and not just match it) to count as a win.
- **Config**: 1.5k LLM-narrated narratives, embedding + LoRA → merge into
  `qwen3-8b-cpt-light-merged`. ~3.5 hr H100.
- **Result**: `PASS_full` — merged checkpoint saved at
  `/workspace/checkpoints/qwen3-8b-cpt-light-merged`. Subsequent x-attn
  Stage-1 always loads this and freezes it.
- **What we learned**: CPT-light produces a usable frozen base. The same
  checkpoint is reused across every downstream baseline (`lora_text`,
  `structured_as_text`) and every x-attn arm — apples-to-apples comparison
  is preserved.

---

## Phase 1 — Baselines v1 (Day 2 AM)

Three baselines trained at the start of Day 2 to anchor the comparison.

| Baseline | Idea | hn_worst (v1) |
|---|---|---|
| `lora_text` | LoRA r=16 on text-only narratives (no events, no x-attn) | 0.0544 [0.039, 0.062] |
| `structured_as_text` | compact-serialized event stream prepended to narrative; CPT-light on the concat (no x-attn) | **0.0451** [0.031, 0.056] |
| `event_only` | small_transformer + classification head, no LM at all | 0.0082 [0.002, 0.017] |

**Caveat — these v1 numbers are deprecated.** Review 018/019 surfaced that
the v1 evaluation didn't drop train/eval text-hash and structured-events-
hash overlap from the eval set; the resulting "clean eval" surface
inflated the apparent gap between event-only and the LM baselines. Rescored
v2 numbers are below in Phase 5.

**What we learned (at the time)**: the structured stream alone (event_only
0.0082) was already cleaning up almost every fraud → x-attn would need to
match or beat it on the **stripped** eval to count, and `structured_as_text
0.0451` was the harder bar because it sees the same signal x-attn does
through its conditioning chain.

---

## Phase 2 — First x-attn smoke run (Day 2 PM)

### `exp_xa_smoke_001` — first cross-attn experiment

- **Date**: 2026-05-16
- **Hypothesis**: a frozen-base + cross-attn + LoRA-on-Q wrapper can train
  in bf16 without NaN cascades, and gate magnitudes start small and grow
  (or stay) as the LM learns to use the side stream.
- **Config**: `every_4 / slots=64 / gate=small_0.01 / small_transformer /
  lora_r_on_q=16`, lr=1e-4, warmup=10, **steps=100, seq_len=2048**.
- **Result**: `ok`. max_gate_magnitude **0.0104** at step 100. No NaN. Loss
  curve smooth.
- **What we learned**: the architecture trains. 1500-step Round-1 runs are
  safe to launch. The agent's permission DSL (`AGENT_INSTRUCTIONS.md`) was
  also exercised end-to-end for the first time here.

---

## Phase 3 — Round-1 grid (Day 2 evening)

Sweep `insertion_pattern × resampler_slots` at fixed `gate_init=small_0.01`
to find the leading architecture cell.

| exp_id | pattern | slots | max_gate | hn_worst [CI] | status |
|---|---|---|---|---|---|
| `round1_001` | every_4 | 64 | 0.0106 | 0.0572 [0.046, 0.069] | ok |
| **`round1_002`** ⭐ | every_8 | 64 | 0.0112 | **0.0524** [0.042, 0.065] | **leader** |
| `round1_003` | late_only | 64 | 0.0109 | 0.0586 [0.046, 0.068] | ok |
| `round1_004` | every_4 | 128 | 0.0112 | 0.0608 [0.048, 0.072] | ok |
| `round1_005` | every_8 | 128 | — | — | **failed (hang)** |
| `round1_006` | late_only | 128 | 0.0109 | 0.0604 [0.047, 0.071] | ok |

All five valid cells: hn_worst in **0.0524 - 0.0608**, all CIs heavily
overlap. Slots=128 cells are tightly clustered (0.0604-0.0608) and slightly
worse than slots=64. round1_005 hung mid-training — F2 self-heal cleaned
the stale GPU lock and the agent moved on (the cell was later retried as
`grid_014`).

**What we learned**: the architectural dial space is narrow. round1_002
emerged as leader on point-estimate (0.0524) but its CI overlaps every
other Round-1 cell — within-Round-1 ranking is not statistically defensible.

---

## Phase 4 — Round-2 gate=zero perturbation (Day 3 AM)

Two top-N siblings of Round-1 with `gate_init=zero` (the unexplored half
of the gate_init dial). Everything else fixed.

| exp_id | pattern | slots | gate_init | max_gate | hn_worst [CI] |
|---|---|---|---|---|---|
| `round2_007` | every_8 | 64 | zero | **0.0038** | 0.0608 [0.048, 0.072] |
| `round2_008` | every_4 | 64 | zero | **0.0041** | 0.0594 [0.047, 0.071] |

**Matched-pair comparisons:**

| Matched cell | gate=small_0.01 | gate=zero | CI overlap? |
|---|---|---|---|
| every_8 / 64 | 0.0524 (round1_002) | 0.0608 (round2_007) | yes, heavy |
| every_4 / 64 | 0.0572 (round1_001) | 0.0594 (round2_008) | yes, heavy |

Both zero-init max_gates landed below the (already-lowered) 0.005 "open"
threshold → two consecutive xattn runs tripped `zero_gate_activation` → the
sweep halted on `n_xattn_runs: 8`, `gpu_hours_used: 7.735/18`.

**What we learned (the headline)**: with `gate_init=zero`, gates moved
~0.004 in 1500 steps; with `gate_init=small_0.01`, gates moved ~0.011. The
movement *from init* is essentially the same in both regimes — **gates ride
whatever bias they start with**. The strict reading wins: Round-1's "open"
gates were init-bias-carried, not learned. The HN-FPR-worst is statistically
tied across all 4 matched pairs.

---

## Phase 5 — Baseline rescore v2 (review 018/019/020)

Audit chain surfaced that the v1 eval set contained train/eval overlap by
text-hash AND by structured-events-hash. After the launcher was patched to
drop the overlap (`clean_eval_n=4466` after dropping 534 overlapping rows),
all four baselines were rescored on the clean eval surface as
`metric_version=2`.

| Baseline | v1 hn_worst | **v2 hn_worst (authoritative)** |
|---|---|---|
| `event_only` | 0.0082 [0.002, 0.017] | **0.0730** [0.067, 0.080] |
| `lora_text` | 0.0544 [0.039, 0.062] | **0.0701** [0.056, 0.085] |
| `structured_as_text` | 0.0451 [0.031, 0.056] | **0.0507** [0.041, 0.064] |

The rescore moved every baseline upward (worse). The leader's relative
position didn't change — round1_002 at 0.0524 is still tied with
`structured_as_text_v2` at 0.0507 (CIs overlap). But the gap between
x-attn and the baselines shrank.

**What we learned**: a chunk of the apparent x-attn advantage was reading
"easy" memorized examples from the train set. The clean-eval surface is
the real comparison. Note that x-attn arms were already on metric_version=2
(the bug was in the baseline scoring path), so this affected baselines
only.

---

## Phase 6 — Day-3 synthesis (Round-2 halt fires)

With Round-2 closed and the v2 baseline rescore in, the auto-loop wrote
the Final Synthesis section of `README.md`:

- Cross-attn provides no detectable classification lift on this surface
  (round1_002 leader at 0.0524 [CI 0.042, 0.065] ties with
  `structured_as_text_v2` at 0.0507 [0.041, 0.064]).
- Per-family invariance: `hn_account_recovery` is the worst family across
  all 8 x-attn runs, all 4 baselines, every dial setting.
- Failure mode is upstream of the model — the synthetic generator's
  `hn_account_recovery` family is the ceiling.
- Recommendation: **pivot, then stop** — 50k medium eval before any more
  architecture sweep, then move budget to the data side.

**The user authorized an Expanded Sweep** on 2026-05-18 to harden the
finding by perturbing dials Round 1+2 hadn't touched: training-dial
perturbations (Phase 1), seq-length stress (Phase 2), and grid completion
at gate=zero (Phase 3), with Phase 4 rank capacity conditional.

---

## Phase 7 — Expanded Sweep, Phase 1: LR/warmup (Day 4 AM)

Hypothesis: are Round-1 gates so low because the LR/warmup schedule isn't
giving them enough gradient signal? Test by varying LR (3× and 1/3×) and
warmup (1/5×) around the leader architecture (every_8 / 64 / small_0.01).

| exp_id | lr | warmup | max_gate | hn_worst [CI] |
|---|---|---|---|---|
| `lr_009` | **3e-4** | **100** | 0.0107 | 0.0559 [0.042, 0.066] |
| `lr_010` | **3e-4** | 500 | 0.0118 | 0.0608 [0.049, 0.071] |
| `lr_011` | **3e-5** | 500 | **0.0134** | 0.0584 [0.047, 0.070] |

(Reference: leader `round1_002` at lr=1e-4 / warmup=500 → max_gate 0.0112,
hn_worst 0.0524.)

**What we learned**:

1. **Gates are LR-invariant in the 3e-5 → 3e-4 range** (10× LR span). max_gate
   sits in a 0.0107-0.0134 band (~25% spread), nowhere near 0.05.
2. **Counterintuitive**: the **lowest** LR (3e-5) produced the **highest**
   max_gate (0.0134). The dial isn't LR — gate dynamics reflect signal
   availability (how much the model wants the side stream), not gradient
   step size.
3. **HN-FPR**: all three CIs overlap the leader. No statistical separation.
4. **Combined dial (lr=3e-4 AND warmup=100, `lr_009`)** did not surface
   anything that the isolated perturbations didn't.

Phase 1 is closed with a clean negative.

---

## Phase 8 — Expanded Sweep, Phase 2: seq-length stress (failed)

### `exp_xa_stress_012` — seq_len=4096, steps=3000

- **Date**: 2026-05-18, started 06:55Z
- **Hypothesis**: doubling sequence length and training duration gives
  gates more long-context signal to learn from. The only Phase-1 dial not
  exercised.
- **Config**: same architecture as leader (every_8 / 64 / small_0.01),
  **seq_len=4096**, **steps=3000**.
- **Result**: **failed** at 07:30Z when the 04:30 cron-tick claude session
  hit its 180m timeout. The launcher was a bash-tool child of claude;
  claude's SIGTERM cascaded to the launcher, which broke the trainer's
  stdout pipe with SIGPIPE.
- **Root cause**: F6 (`start_new_session=True` for the trainer subprocess)
  protected the trainer's session — but not the launcher's. The launcher's
  death is what propagated and killed everything.

### F8 patch (v1 → v2)

The first patch attempt added `os.setsid()` + stdout redirect to
`launcher.log`. Empirical verification on `grid_014` showed `setsid()`
silently failed (returned EPERM because claude's Bash tool puts each
command into its own process group → launcher was already a PG leader).

**F8 v2** swapped session detachment for explicit signal handling:

- `signal.signal(SIGHUP, SIG_IGN)` — survives session-leader death
- `signal.signal(SIGPIPE, SIG_IGN)` — survives broken stdout pipe
- stdout/stderr → `launcher.log` (defense-in-depth + auditable per-run log)

Verified active on `grid_015`, `grid_016`, `grid_017` (launcher.log first
line: `survival mode active … state=shared-session-but-SIGHUP-ignored`).

**What we learned**: long-running experiments cross cron-tick session
boundaries. The original design ("agent_tick.sh wraps claude in `timeout
180m`") implicitly assumed each experiment fits in one window. Once that
broke, the trainer-only F6 isolation wasn't enough.

stress_012 will be retried later if budget allows; for now it's an
"upstream cause: F8" failure, not a research finding.

---

## Phase 9 — Expanded Sweep, Phase 3: grid completion (Day 4 mid-day)

Hypothesis: complete the `gate_init × insertion_pattern × resampler_slots`
matrix so every architectural variant is measured at gate=zero (the only
gate_init the Round-1 grid hadn't tested).

| exp_id | pattern | slots | gate_init | max_gate | hn_worst [CI] |
|---|---|---|---|---|---|
| `grid_013` | late_only | 64 | zero | **0.0028** | 0.0608 [0.049, 0.072] |
| `grid_014` | every_8 | 128 | small_0.01 | 0.0109 | 0.0608 [0.048, 0.072] |
| `grid_015` | every_4 | 128 | zero | 0.0034 | 0.0598 [0.048, 0.071] |
| `grid_016` | every_8 | 128 | zero | 0.0034 | 0.0608 [0.049, 0.072] |
| `grid_017` | (training) | — | — | — | — |

Grid_014 was the round1_005 cell that originally hung in Round 1 (`every_8 /
128 / small_0.01`); retried here cleanly under the F8 v2 patch's
predecessor (still pre-F8 at launch but inside one session window).

### Gate=zero quintet (gate=zero × {every_4, every_8, late_only} × {64, 128})

| Pattern | Slots | max_gate | hn_worst | CI |
|---|---|---|---|---|
| every_4 | 64 | 0.0041 | 0.0594 | [0.047, 0.071] |
| every_4 | 128 | 0.0034 | 0.0598 | [0.048, 0.071] |
| every_8 | 64 | 0.0038 | 0.0608 | [0.048, 0.072] |
| every_8 | 128 | 0.0034 | 0.0608 | [0.049, 0.072] |
| late_only | 64 | 0.0028 | 0.0608 | [0.049, 0.072] |
| late_only | 128 | (pending: grid_017?) | — | — |

**What we learned (running)**:

- max_gate band across the quintet: **0.0028 - 0.0041** (32% spread, all
  ≪ 0.05). Gate dynamics under gate=zero are architecture-invariant.
- hn_worst band: **0.0594 - 0.0608** (Δ = 0.0014, all CIs collapse onto
  each other and onto the leader).
- **Counterintuitive (again)**: doubling slots 64→128 under gate=zero
  *lowers* max_gate in both matched pairs (every_4: 0.0041→0.0034, every_8:
  0.0038→0.0034). More slots = fewer per-slot gradient signals = lower
  per-slot gate magnitudes. Architectural arithmetic, not a research
  finding.
- **The pathway is inert**. Three orthogonal dial families (architecture,
  init, training schedule) have now been swept; every CI overlaps the
  leader.

---

## Phase 10 — Expanded Sweep, Phase 4: rank capacity (conditional)

Per directive, Phase 4 runs only if Phase 1-3 don't surface gate learning.
The plan is `lora_r_on_q ∈ {8, 32}` on the leader architecture — probes
whether the LoRA-on-Q parameter budget is the bottleneck. As of this
writeup the prior is strongly against finding anything new, but the
expansion runs to completeness unless the GPU-hours cap closes the sweep
first.

---

## Failure roster (for completeness)

| exp_id | failure mode | resolution |
|---|---|---|
| `round1_005` | trainer hung mid-training | F2 stale-PID lock self-heal cleaned the lock; agent moved on; cell later validated cleanly as `grid_014` |
| `stress_012` | killed by 180m claude session boundary | F8 v2 patch (SIGHUP/SIGPIPE IGN) deployed; retry possible if budget allows |

---

## Running budget snapshot

| | |
|---|---|
| Last updated | 2026-05-18 ~11:15Z |
| `n_xattn_runs` (valid) | 15 (16 launched, 1 failed boundary-kill) |
| `current_best` | `round1_002` @ 0.0524 [0.042, 0.065], **unchanged through 8 expanded arms** |
| `gpu_hours_used` | 13.17 / 18.00 → ~4.8 hr remaining |
| `halted` | false |
| Bar to beat (CI-strict for early-exit) | `ci_hi < 0.0420` AND `max_gate >= 0.05` |

The bar has not been hit by any arm. Expansion continues until early-exit
fires, GPU-hours cap closes, or the Phase-1-through-4 queue exhausts.

---

## What this log will look like when it closes

When the sweep terminates (early-exit / budget / queue done), the agent
will:

1. Append the closing rows here with their hn_worst/CI/max_gate.
2. Rewrite the README's Final Synthesis section to reflect the full sweep,
   not just the Day-3 8-arm cut.
3. Remove the status banner at the top of README.

Until then, this log is the source of truth for "what's been tried" and
"what we learned" beyond the raw `experiments.jsonl` rows.
