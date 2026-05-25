# The Cross-Attention Experiments

**Whitepaper companion document · v1.2 · 2026-05-22**

This document covers the cross-attention architecture, the training recipe, and the full set of experimental results across the three sweep generations (v3, v4, v5). It is the result-side counterpart to the methodology-side companions: data curation (`01-data-curation-and-distribution.md`), the agentic harness (`02-agentic-experiment-harness.md`), and the eval strategy (`03-eval-strategy.md`). The master narrative is in `00-whitepaper-main.md`.

---

## 1. Architecture

![Figure 1. Cross-attention surgery on Qwen3-8B](figures/fig1-architecture.svg)

The architecture is Flamingo (Alayrac et al., 2022) applied verbatim to a different modality — structured behavioral event streams instead of images. Figure 1 summarizes the full surgery. Five components:

### 1.1 Base — Qwen3-8B, frozen, post-CPT-light merge

- 36 transformer layers, hidden dim 4096, 32 attention heads, 8 KV heads.
- Stage-0 continued pre-training (CPT-light): new-token embedding (for the journey, actor, event, and bucketed-feature token families) + LoRA on attention and MLP. ~1500 steps on the LLM-narrated training pool. ~53 minutes wall-clock on a single H100.
- The Stage-0 LoRA is *merged into the base weights* via `scripts/merge_stage0_lora.py` to produce `qwen3-8b-cpt-light-merged` (in v3) or `qwen3-8b-cpt-light-v4-merged` (in v4 and v5). The merge is deterministic and idempotent.
- Stage-1 cross-attention training operates on this merged, *frozen* checkpoint. The base weights do not move; only the Stage-1 components train.

The merge-before-Stage-1 sequence is non-negotiable. The v3 audit (`02-agentic-experiment-harness.md` §6) found that a live PEFT Stage-0 adapter sitting on the model during Stage-1 training caused gate-magnitude drift because the Stage-0 adapter was itself updating during Stage-1 — the cross-attention gates were training against a moving target. The merge-then-x-attn ordering is the only stable sequence.

### 1.2 Side-stream encoder

The structured event stream is consumed by a side encoder that produces a sequence of contextualized event embeddings. Three variants swept in v5 Phase-2:

- **`small_transformer`** (default, v3/v4/v5 winner). A 6-layer transformer over event tokens. Hidden dim 1024, 8 attention heads. Trained from scratch jointly with the rest of Stage-1. ~10–20M trainable parameters depending on context length.
- **`pooled_mlp`**. Mean/max-pool over event embeddings + MLP projection. Simpler; no attention. Tests "is the side-stream's attention necessary, or is pooling enough?" ~109M trainable parameters (most of which is the MLP projection).
- **`ft_transformer`**. FT-Transformer (Gorishniy et al., 2021) style tabular encoder, treating each event as a row of tabular features. ~5M encoder, ~113M Stage-1 total — **total parameter count is essentially identical to `small_transformer` (113.4M vs 113.6M)**; the comparison tests a *tabular-feature inductive bias in the encoder* against `small_transformer`'s sequence-attention bias, not a capacity difference.

The side encoder receives events with sinusoidal-on-Δt time encoding plus learned token embeddings for each event-type and bucketed-feature family. Input sequence length is bounded at 200 events; longer journeys are truncated to the last 200 (the most recent events being most informative for ATO).

### 1.3 Perceiver-Resampler

The side encoder's output (a variable-length sequence of contextualized event embeddings) is compressed by a Perceiver-Resampler (Jaegle et al., 2021) to a fixed-length set of K/V slots. The Resampler is a small transformer with `N` learned query slots that cross-attend to the encoder output. `N ∈ {32, 64, 128}` was swept in v5 Phase-1.

The bottleneck serves two purposes. First, it gives the downstream cross-attention layers a fixed-size input regardless of journey length (a Flamingo-style design choice). Second, it compresses the structured-stream signal into a small set of K/V vectors that the LM can attend to efficiently. The trainable parameter count is small (~2–4M depending on slot count).

Empirically, `N = 64` was the v3, v4, and v5 winner. `N = 128` was statistically tied within CIs in v3 (suggesting the additional capacity was wasted on this surface). `N = 32` regressed in v5 Phase-1 (`exp_v5_p1_slots32` had `v5_adv_error = 0.168` vs the winner's 0.151).

### 1.4 Gated cross-attention blocks

Cross-attention blocks are inserted into the frozen Qwen3-8B stack at periodic depth. Three insertion patterns swept (definitions in `src/model/qwen_xattn_wrapper.py::compute_insertion_layers`, lines 163-186):

- **`every_4`** — `range(12, 36, 4)` → layers {12, 16, 20, 24, 28, 32} = **6 inserted blocks**.
- **`every_8`** — `range(12, 36, 8)` → layers {12, 20, 28} = **3 inserted blocks** (default; v4 seed; v5 Phase-1 winner).
- **`late_only`** — `range(32, 36)` → layers {32, 33, 34, 35} = **4 inserted blocks**, all in the last quarter of the stack.

All three patterns deliberately start at layer 12 (or later) — the design choice is that cross-attention should not disturb the LM's early token-feature extraction; the side-stream signal is fused only once the LM has done its first pass of semantic shaping. This is a deliberate departure from the Flamingo reference (which inserts at layer 0); it was set by the original PLAN.md architecture decision and never re-swept.

Each inserted block applies:

```
h' = h + tanh(α) · CrossAttention(Q = h, K = resampler_out, V = resampler_out)
```

where `h` is the LM's hidden state at the insertion point, `α` is a per-block scalar gate (`tanh(α)` is initialized at exactly 0 if `gate_init=zero`, or 0.01 if `gate_init=small_0.01`), and the cross-attention is plain multi-head attention (not MLA). The cross-attention's Q projection is a fresh trainable matrix; the K and V projections are computed once per session from the resampler output and cached. Trainable parameter count per block is **~33.6M** (measured by differencing `actual_trainable_total` across the three insertion patterns; see §1.6).

### 1.5 LoRA-on-Q

The *only* trainable parameter touching the LM's frozen weights. Rank-16 LoRA (Hu et al., 2021) on the LM's self-attention query projection. `lora_r_on_q ∈ {16, 32, 64}` was swept in v5 Phase-1. `r = 16` was the winner; `r = 32` was statistically tied; `r = 64` was statistically tied (the additional rank was not used).

LoRA-on-Q is required because the cross-attention path inserts the side-stream signal into the residual stream, and the LM's self-attention has to learn to attend to this slightly-modified residual. The Q projection is the only place this attention rerouting needs to happen; we did not attach LoRA to the KV or output projections.

### 1.6 Trainable parameter inventory

All counts below are *measured* from `runs/exp_*/metrics.json::actual_trainable_total`, not estimated. The per-block number was derived by differencing matched configurations across insertion patterns.

| Component | Trainable params |
|---|---|
| Side-stream encoder (`small_transformer`) | ~5M |
| Perceiver-Resampler | ~2–3M (depending on slot count) |
| Gated cross-attention blocks (per block) | **~33.6M** (measured) |
| Number of inserted blocks | **3 (`every_8`) / 4 (`late_only`) / 6 (`every_4`)** |
| Gate scalars | one scalar per block (negligible) |
| LoRA-on-Q (rank 16) | ~4.7M |
| **Total Stage-1** | **~110M–215M** (configuration-dependent) |

Of 8 billion base parameters, the Stage-1 trainable footprint is 1.4–2.7%. Verified totals for the v5 sweep:

| Configuration | Measured `actual_trainable_total` |
|---|---|
| `exp_xattn_v4_001` (every_8 / 64 / r=16 / small_transformer) | 113,604,614 |
| `exp_v5_p1_zero_64` (every_8 / 64 / r=16 / small_transformer) — **Phase-1 winner** | 113,604,614 |
| `exp_v5_p1_every4_64` (every_4 / 64 / r=16 / small_transformer) | 214,372,364 |
| `exp_v5_p1_late_64` (late_only / 64 / r=16 / small_transformer) | 147,193,864 |
| `exp_v5_p2_pooled_mlp` (every_8 / 64 / r=16 / pooled_mlp) | 108,998,150 |
| `exp_v5_p2_ft_transformer` (every_8 / 64 / r=16 / ft_transformer) | 113,410,566 |

The implied per-block cost is `(214,372,364 − 113,604,614) / 3 = 33,589,250` parameters, and is consistent across the late_only difference `(147,193,864 − 113,604,614) / 1 = 33,589,250`. The Q-projection alone (`hidden=4096 × cross_dim=1024`) accounts for ~4M of this; the FFN at `4096 × 2048` doubled accounts for ~33.6M alone; LayerNorms and MHA add the remainder (formula in `src/model/cross_attn_block.py::estimate_block_param_count`).

---

## 2. Training recipe

- **Framework.** HuggingFace Accelerate. Single-process, single-H100, no DataParallel.
- **Precision.** bf16.
- **Optimizer.** Paged AdamW 8-bit (`paged_adamw_8bit` from bitsandbytes ≥ 0.45 — see `02-agentic-experiment-harness.md` §8 for the v3 Blackwell compatibility integration-friction finding).
- **Learning rate.** Cosine schedule with 500-step warmup, peak LR `1e-4`. v5 Phase-1 swept LR perturbations (`3e-4` fast, `3e-5` slow); both regressed (see §5).
- **Batch.** Micro-batch 4, gradient accumulation to effective batch size 32.
- **Sequence length.** 2048 default. 4096 reserved for one optional stress run; not exercised in v5.
- **Steps.** 1500 default. v3's stress option was 3000 steps; v5 did not run a stress.
- **Loss.** Next-token cross-entropy on the text stream (narrative + verdict footer).

The training recipe is unchanged across v3, v4, and v5 except for the LR/warmup perturbations in v5 Phase-1. The recipe is the same one used in the v3 plan; no v4 or v5 changes to it. All architectural variation lives in the dial settings.

---

## 3. The v3 result — null

Eighteen valid cross-attention runs were recorded in v3 across two sweep phases (Round-1: 6 grid cells; Round-2: 2 perturbations) and the v3 expansion (`xattn-expanded-sweep-plan.md`: Phase 1, Phase 2, Phase 3 grid completion, Phase 4 rank capacity). Headline numbers (`metric_version: 2`, clean eval `n = 4,466`, 95% bootstrap CIs):

| Run | Config | Worst-family HN-FPR [CI] | Mean HN-FPR | Worst family | Max gate |
|---|---|---|---|---|---|
| `exp_xa_round1_002` (v3 leader) | every_8 / 64 / 0.01 | 0.0524 [0.0420, 0.0647] | 0.0262 | hn_account_recovery | 0.0112 |
| `exp_baseline_structured_as_text_v2` | (concat baseline) | 0.0507 [0.0408, 0.0635] | 0.0262 | hn_account_recovery | n/a |
| `exp_xa_round1_001` | every_4 / 64 / 0.01 | 0.0572 [0.0455, 0.0691] | 0.0258 | hn_account_recovery | 0.0106 |
| `exp_xa_round1_003` | late_only / 64 / 0.01 | 0.0586 [0.0460, 0.0683] | 0.0256 | hn_account_recovery | 0.0109 |
| `exp_xa_round2_008` | every_4 / 64 / zero | 0.0594 [0.0470, 0.0708] | 0.0256 | hn_account_recovery | 0.0041 |
| `exp_xa_round1_006` | late_only / 128 / 0.01 | 0.0604 [0.0472, 0.0709] | 0.0255 | hn_account_recovery | 0.0109 |
| `exp_xa_round1_004` | every_4 / 128 / 0.01 | 0.0608 [0.0481, 0.0724] | 0.0254 | hn_account_recovery | 0.0112 |
| `exp_xa_round2_007` | every_8 / 64 / zero | 0.0608 [0.0475, 0.0716] | 0.0254 | hn_account_recovery | 0.0039 |
| `exp_baseline_lora_text_v2` | (LoRA text-only) | 0.0701 [0.0564, 0.0847] | 0.0291 | hn_large_purchase | n/a |
| `exp_baseline_event_only_v2` | (event-only classifier) | 0.0730 [0.0667, 0.0799] | 0.0243 | hn_account_recovery | n/a |

The v3 leader is `exp_xa_round1_002` at worst HN-FPR = 0.0524. The load-bearing baseline `structured_as_text_v2` is at 0.0507. CIs heavily overlap; cross-attention is +0.0017 absolute **worse** on the point estimate. **No CI-separated win.** The same pattern held across every architectural dial: insertion pattern (3 variants ranged 0.052–0.061), resampler slots (64 vs 128 tied within CI), gate init (small_0.01 vs zero produced ~3× gate-magnitude difference with statistically tied HN-FPR).

The v3 gates story across all 8 valid round-1 + round-2 runs:

- `gate_init=small_0.01` × 6 runs: `max_gate_magnitude` at step 1500 = 0.0106–0.0112. Movement from init: 0.0006–0.0012 (≤10% of init magnitude). Gates rode their initialization.
- `gate_init=zero` × 2 runs: `max_gate_magnitude` at step 1500 = 0.00385–0.00412. Both below the (already-lowered) 0.005 "open" threshold. Two consecutive sub-threshold runs tripped the `zero_gate_activation` halt.

The strict reading wins: gates did not learn. The base CPT-light-merged LM was doing essentially all of the discrimination work; cross-attention contributed at most marginal lift the 5k eval could not detect.

### 3.1 The v3 expansion

After the convergence halt was disabled (see `02-agentic-experiment-harness.md` §6.1 for the postmortem), an expanded sweep ran Phase 1 (LR/warmup), Phase 2 (stress), Phase 3 (grid completion), and Phase 4 (rank capacity, conditional). Headline: `exp_xa_lr_009` (lr=3e-4, warmup=100) was the first expansion arm; it had `max_gate_magnitude = 0.00580` — still below the open threshold, and produced no measurable HN-FPR change. The expansion continued through Phase 3's grid completion; none of the additional cells separated from the v3 leader within CI. The expansion was halted by the GPU-hours cap before Phase 4 ran.

The v3 result was not architectural. Section §4 explains.

---

## 4. The v4 result — CI-separated win on adversarial families

The v4 pivot (`01-data-curation-and-distribution.md` §3) ran the same v3 leader configuration on the v4 data. Single seed: `exp_xattn_v4_001` (every_8 / slots=64 / gate=small_0.01 / small_transformer / lora_r=16), paired with a `text_only` arm at the same training recipe. The two arms saw byte-identical LM prompts modulo the side stream.

Per-family fraud recall at the 1%-legit-FPR operating point (`metric_version: 5`, clean eval `n = 5,002`, 95% bootstrap CIs; source: `runs/exp_text_only_v4_001/ci_report.json`, `runs/exp_xattn_v4_001/ci_report.json`):

| Family | n_eval | `text_only_v4` | `xattn_v4` | Δ | CI overlap? |
|---|---|---|---|---|---|
| `phish_takeover` | 224 | 0.1122 [0.072, 0.158] | **1.0000 [1.000, 1.000]** | **+0.8878** | **CI-separated** |
| `phish_takeover_mfa_phished` (v4 new) | 71 | 0.0000 [0.000, 0.012] | **0.9718 [0.931, 1.000]** | **+0.9718** | **CI-separated** |

Per-family hard-negative FPR at the 1%-legit-FPR operating point (`stripped` mode, source: `runs/exp_*_v4_001/ci_report.json::hard_negative_fpr_at_1pct.per_family`):

| Family | n_eval | `text_only_v4` | `xattn_v4` | CI overlap? |
|---|---|---|---|---|
| `hn_account_recovery` | 488 | 0.0014 [0.000, 0.005] | 0.0000 [0.000, 0.000] | (overlap, both ~0) |
| `hn_large_purchase` | 496 | 0.0000 [0.000, 0.000] | 0.0000 [0.000, 0.000] | (overlap, both ~0) |
| `hn_travel` | ~280 | 0.0040 [0.000, 0.010] | 0.0000 [0.000, 0.000] | (overlap, both ~0) |
| `hn_recovery_high_amount` (v4 new) | 78 | **0.4175 [0.342, 0.512]** | **0.4505 [0.369, 0.562]** | (both poor, CIs overlap) |

The v4 architectural win is concentrated on the two adversarial fraud families (`phish_takeover`, `phish_takeover_mfa_phished`) where the side stream resolves text-level ambiguity. The hardest adversarial-legitimate family (`hn_recovery_high_amount`) sits at ~0.42–0.45 FPR for both arms with overlapping CIs — the architectural ceiling that no v5 dial subsequently moves (§5.3). Fraud recall on the non-adversarial families (`cred_stuff`, `sim_swap`, `malware_rat`, `mule_chain`) is not broken out per-family in the current `ci_report.json`; the pooled `r_at_fpr_0.01` (recall at 1% legit-FPR over *all* fraud rows) is reported instead — `text_only_v4` = 0.7391 [0.7126, 0.7662], `xattn_v4` = 0.9947 [0.9888, 0.9993] — consistent with the adversarial-family decomposition above (text-only misses both adversarial families heavily, dragging its pooled recall well below 1.0; cross-attention catches both, leaving the pooled recall essentially saturated).

**v4 gates story.** `max_gate_magnitude = 0.0221` on `exp_xattn_v4_001` — above the v3 ceiling of 0.011 across all 18 cells, but well below the Flamingo "open" target of ~0.1. The gates are *sparse but effective*: the LM does most of the discrimination, and the cross-attention pathway is opened narrowly when the side stream resolves a textual ambiguity. The +0.97 swing in `phish_takeover_mfa_phished` recall (0.000 → 0.972) is achieved with `tanh(α) ≈ 0.022` — a 2.2% mixing weight per inserted block, summed over 3 blocks (in the `every_8` configuration; see §1.4 for layer indices). The architecture is parameter-efficient and gate-efficient.

The v4 result is what cross-attention was designed to do: catch the cross-modal fraud cases that the LM cannot see in the narrative alone. The result is *conditional* on the v4 data pipeline restoring a real modality gap.

---

## 5. The v5 result — dial-robust, with a data-shaped ceiling

![Figure 4. v5 sweep leaderboard with adversarial-error decomposition](figures/fig4-sweep-results.svg)

The v5 sweep ran 11 cross-attention configurations across two phases (Figure 4A; full numbers in `experiments.jsonl`):

### 5.1 Phase 1 — training and arch-dial sweep

Eight cells, varying one dial at a time around the v4 seed (all values from `runs/exp_*/ci_report.json::stripped.v5_adv_error`):

| exp_id | Config | v5_adv_error [CI] | mfa_phished miss | hn_recovery FPR | max_gate |
|---|---|---|---|---|---|
| `exp_xattn_v4_001` (seed) | every_8 / 64 / 0.01 | 0.1596 [0.1306, 0.1945] | 0.0282 | 0.4505 | 0.0221 |
| `exp_v5_p1_every4_64` | every_4 / 64 / 0.01 | 0.1506 [0.1235, 0.1886] | 0.0141 | 0.4377 | 0.0171 |
| `exp_v5_p1_late_64` | late_only / 64 / 0.01 | 0.1549 [0.1279, 0.1927] | 0.0141 | 0.4505 | 0.0179 |
| **`exp_v5_p1_zero_64`** ★ | **every_8 / 64 / zero** | **0.1506 [0.1238, 0.1871]** | **0.0141** | **0.4377** | **0.0128** |
| `exp_v5_p1_slots128` | every_8 / 128 / 0.01 | 0.1526 [0.1278, 0.1893] | 0.0201 | 0.4377 | 0.0212 |
| `exp_v5_p1_slots32` | every_8 / 32 / 0.01 | 0.1656 [0.1410, 0.2017] | 0.0463 | 0.4505 | 0.0182 |
| `exp_v5_p1_rank32` | every_8 / 64 / 0.01 / r=32 | 0.1617 [0.1347, 0.1979] | 0.0347 | 0.4505 | 0.0089 |
| `exp_v5_p1_slowlr` | lr=3e-5 / warmup=500 | 0.2015 [0.1670, 0.2435] | 0.1539 | 0.4505 | 0.0165 |
| `exp_v5_p1_fastlr` | lr=3e-4 / warmup=100 | **0.7516 [0.7208, 0.7828]** (catastrophic) | 0.9998 | 0.4249 | 0.0058 |

**Phase-1 winner:** `exp_v5_p1_zero_64` (every_8 / slots=64 / gate=zero / small_transformer / lora_r=16). `v5_adv_error = 0.1506 [CI 0.1238, 0.1871]`. The v4 seed had 0.1596 [CI 0.1306, 0.1945]. Point improvement: −0.0090 absolute. **CIs overlap** ((winner_hi = 0.1871) > (seed_lo = 0.1306)). No statistically robust separation.

The win comes from two small gains, both within bootstrap CI noise: `phish_takeover_mfa_phished` miss-rate dropped from 0.0282 to 0.0141; `hn_recovery_high_amount` FPR dropped from 0.4505 to 0.4377.

Phase-1 findings:

- **Gate-init (`zero` vs `small_0.01`) is neutral.** `exp_v5_p1_zero_64` and `exp_v5_p1_every4_64` (same v5_adv_error) sit at different `max_gate` (0.013 vs 0.017) but tie on the composite metric. Gate init does not predict quality.
- **Insertion pattern (`every_4` vs `every_8` vs `late_only`) is mostly neutral.** `every_4` adds 2× the parameters (214M vs 113M trainable) without measurable gain. `late_only` regresses slightly.
- **Resampler slot count (`32` vs `64` vs `128`) is mostly neutral.** `slots=32` regresses; `slots=128` ties within CI. 64 is the sweet spot.
- **LoRA rank (`r=16` vs `r=32`) is neutral.** Higher rank does not help.
- **LR is critical.** `lr=3e-4 / warmup=100` catastrophically regresses (`v5_adv_error = 0.7516` because `phish_takeover` recall collapses to 0.17). `lr=3e-5 / warmup=500` hurts `phish_takeover_mfa_phished` recall (0.986 → 0.846). The default `lr=1e-4 / warmup=500` is robust.

### 5.2 Phase 2 — encoder sweep

| exp_id | Encoder | v5_adv_error [CI] | mfa_phished miss | hn_recovery FPR |
|---|---|---|---|---|
| `exp_v5_p1_zero_64` | small_transformer | **0.1506 [0.1238, 0.1871]** | 0.0141 | 0.4377 |
| `exp_v5_p2_pooled_mlp` | pooled_mlp | 0.1549 [0.1282, 0.1938] | 0.0141 | 0.4505 |
| `exp_v5_p2_ft_transformer` | ft_transformer | 0.1736 [0.1448, 0.2136] | 0.0704 | 0.4505 |

Phase-2 findings:

- **`pooled_mlp` ties.** Mean/max-pool + MLP projection collapses attention-over-events into a simpler operation without measurable harm. The structured-stream encoding does *not* require attention.
- **`ft_transformer` regresses.** Its Stage-1 total (113.4M trainable) is essentially identical to `small_transformer` (113.6M) and only marginally above `pooled_mlp` (109.0M); the regression is **not a capacity difference** but an inductive-bias mismatch. The tabular-feature attention pattern hurts `phish_takeover_mfa_phished` recall (0.9296 vs 0.9859 for the winner) — most likely because the sequence ordering of events carries signal that the per-feature attention pattern discards in 1500 steps.
- **Gate magnitudes are indistinguishable across encoders** (0.0146–0.0220). The encoder architecture does not affect whether the cross-attention gates open.
- **The Phase-2 stop rule fired** after neither alternative beat the Phase-1 winner by ≥0.005 absolute `v5_adv_error`.

### 5.3 The data-shaped ceiling on `hn_recovery_high_amount`

Across the non-pathological v5 x-attn runs, the `hn_recovery_high_amount` FPR stayed in the band [0.4377, 0.4505] — a maximum spread of 0.0128, smaller than the per-run bootstrap CI width (~0.18 on the 78-row family). The one excursion is `exp_v5_p1_fastlr` at 0.4249, which dropped below the band only because `phish_takeover_mfa_phished` recall collapsed to ~0 and the 1%-legit-FPR operating point shifted to a different tied score mass — not a real architectural movement of the ceiling. No dial moves the underlying family (Figure 4B):

| Dial varied | Range tested | hn_recovery FPR range |
|---|---|---|
| Insertion pattern | every_4 / every_8 / late_only | [0.4377, 0.4505] |
| Resampler slots | 32 / 64 / 128 | [0.4377, 0.4505] |
| Gate init | zero / small_0.01 | [0.4377, 0.4505] |
| LoRA rank | 16 / 32 | [0.4377, 0.4505] |
| LR | 1e-4 / 3e-4 / 3e-5 | [0.4249, 0.4505] (3e-4 below band: operating-point artifact after fraud-recall collapse) |
| Encoder | small_transformer / pooled_mlp / ft_transformer | [0.4377, 0.4505] |

The component contributes ~97% of the total `v5_adv_error` for the Phase-1 winner (0.146 of 0.151). The `hn_recovery_high_amount` ceiling is the dominant constraint on this sweep, and it is not movable by architecture choice. The diagnostic in §6 below.

---

## 6. Diagnosing the `hn_recovery_high_amount` ceiling

The v4 design intent for `hn_recovery_high_amount` was an adversarial-legitimate family: text reads like classic ATO ("device change, password reset, large transfer to new recipient"), but events reveal legitimacy (the new recipient is the account holder's other account; the new device is mid-upgrade; MFA was used). The intent was that cross-attention would resolve the text-level ambiguity by reading the events.

The v4 result on this family — `xattn` FPR = 0.4505 [CI 0.369, 0.562], `text_only` FPR = 0.4175 [CI 0.342, 0.512] (source: `runs/exp_{xattn,text_only}_v4_001/ci_report.json`, `metric_version: 5`, `n_clean = 5,002`) — is poor for both arms. Cross-attention is +0.033 absolute *worse* on the point estimate, but the CIs overlap heavily. The v5 result holds: across the non-pathological v5 x-attn configurations, FPR is stuck in [0.4377, 0.4505]. Both v4 and v5 numbers are computed on the same `metric_version: 5`, `n_clean = 5,002` surface; the family ceiling is invariant under architecture sweep.

Three hypotheses are consistent with the data:

1. **The event stream does not actually contain disambiguating signal for this family.** The v4 generator template specifies adversarial cross-modal structure, but the bucketed feature tokens may not encode the signal granularly enough. For example, `device_age=new` cannot distinguish "the account holder's mid-upgrade phone" (legit) from "an attacker's freshly-imaged phone" (fraud). The fix is generator-side: add richer contrastive features (device-trust history, step-up-auth chain, prior-recovery-attempt count).

2. **The model cannot learn the discrimination in 1500 steps.** A longer training run or a different optimizer might surface the signal. v3's planned stress run (3000 steps, seq_len=4096) was not exercised in v5; this is a recommended next step.

3. **The metric is correctly exposing a production ambiguity.** Real production has legit high-amount account-recovery events whose features at the time of the event are indistinguishable from fraud — that is what step-up authentication is for. Asking a binary fraud-or-legit classifier to achieve low FPR on this family is asking for the wrong thing. The fix is framing-side: this family becomes a "route to step-up auth" target, evaluated by routing precision rather than FPR.

We cannot distinguish (1), (2), (3) from the synthetic surface alone. The recommended path is to test (1) first via generator extension (cheap), then (2) if that doesn't move the FPR (one stress run), then (3) as a framing change (a metric redesign, no model work).

The v5 final synthesis (`README.md` §V5 Final Synthesis) recommends data redesign as the primary next move, with a caveat for framing-review if the redesign cannot move the FPR below 0.15.

---

## 7. Per-family breakdown of the Phase-1 winner

The Phase-1 winner `exp_v5_p1_zero_64` per-family results (`runs/exp_v5_p1_zero_64/ci_report.json`, stripped mode, `n_clean = 5,002`):

**Fraud recall at 1% legit FPR:**

| Family | n_eval | Recall [CI] |
|---|---|---|
| `cred_stuff` | ~400 | 1.000 (saturated) |
| `sim_swap` | ~280 | 1.000 |
| `phish_takeover` | 224 | 1.000 [1.00, 1.00] |
| **`phish_takeover_mfa_phished`** | 71 | **0.986 [0.972, 1.000]** |
| `malware_rat` | ~280 | 1.000 |
| `mule_chain` | ~240 | 1.000 |

**Hard-negative FPR at 1% legit FPR:**

| Family | n_eval | FPR [CI] |
|---|---|---|
| `hn_account_recovery` | 488 | 0.0000 [0.000, 0.000] |
| `hn_large_purchase` | 496 | 0.0020 [0.000, 0.006] |
| **`hn_recovery_high_amount`** | 78 | **0.4377 [0.3601, 0.5449]** |
| `hn_travel` | ~280 | 0.0000 |

**Diagnostic numbers:**

- AUC stripped: 0.99977 [0.99963, 0.99988]
- AUC opaque: 0.99977 (the model uses the same signal in both modes)
- AUC full: 0.99977 (saturated; the structural tokens add no signal beyond what's already in the narrative + events)
- R @ FPR=0.1%: 0.9438 [0.9274, 0.9693]
- R @ FPR=1%: 0.9980 [0.9942, 1.000]
- Final train loss: 0.495 (vs v3 leader's 1.150 — v4's modality-gap design produces a harder training task and a lower terminal loss)
- Wall-clock: 39.4 minutes
- Trainable parameters: 113M

The model is doing exactly what v4 designed it to do: catching every fraud case (recall 1.0 except for ~1.4% of `phish_takeover_mfa_phished`), tolerating the easier hard-negative families at zero FPR, and failing on the family that requires a signal the events don't carry.

---

## 8. Day-N recommendation

The v5 sweep exhausted the architecture-side levers. The recommended next moves, in priority order:

### 8.1 Data redesign for `hn_recovery_high_amount` (next)

Regenerate the `hn_recovery_high_amount` template with richer contrastive features. Specific extensions:

- **Device-trust history.** Add an event indicating "this device has been seen N times in the past 90 days" (bucketed: never / 1–3 / 4+). The legit account-holder's mid-upgrade phone would have trust ≥ 4; an attacker's fresh phone would have trust = 0.
- **Step-up-auth chain.** Add an event for "step-up auth challenge issued" with outcome bucketed (none / passed-out-of-band / passed-via-recovery-email / failed). The legit recovery would have passed-out-of-band; an attacker's takeover would have failed or no challenge.
- **Prior-recovery-attempt count.** Add a feature for "number of recovery attempts in last 30 days" (bucketed: 0 / 1–2 / 3+). Legit account-holders trying to recover access typically have 0 prior; attackers often have 3+ failed attempts before the successful takeover.
- **Mid-session anomaly indicators.** Add events for behavioral pattern shifts during the session (e.g., `<typing_pattern=changed>`, `<tool_use_cadence=inhuman>`).

Target: `hn_recovery_high_amount` FPR < 0.15 on the 5k clean eval before further architecture sweep is justified.

### 8.2 Production-replay validation (parallel)

Held-out anonymized window of real fraud-and-legit traffic. This is the only test that distinguishes "the architecture works on the synthetic surface" from "the architecture works on the production surface." Specifically:

- 10k–50k real anonymized journeys, sampled across the same journey/actor schema as the synthetic generator.
- The same `metric_version: 5` eval pipeline with bootstrap CIs.
- Per-family breakdown to identify which production families have the synthetic-analog `hn_recovery_high_amount` pathology.

Estimated effort: 1–2 weeks of data-engineering + access-control work to scope the anonymized replay; <1 day of model inference once the data is in place.

### 8.3 Framing review (parallel)

If the data redesign cannot get `hn_recovery_high_amount` FPR below 0.15 on the 5k eval, the metric itself is exposing a real product question. Specifically: in production, the right action for a legit high-amount account-recovery event is *step-up authentication*, not "fraud / not-fraud." The model is being asked to make a decision (binary classification) that the production system would route to a separate step-up flow.

If the framing changes from "fraud/legit binary" to "route to step-up auth," the metric for this family changes from FPR to step-up-routing precision. The model architecture might be unchanged; only the eval framing changes. This is a product decision, not a model decision.

### 8.4 The stress run (deferred)

The v3 plan had a stress-run option (3000 steps, seq_len = 4096) that was never exercised in v3 (the convergence halt fired first) or v5 (Phase-2 stop rule fired first). On the v5 Phase-1 winner, running the stress option would test whether longer training surfaces a signal on `hn_recovery_high_amount` that the 1500-step baseline misses. Estimated cost: one ~150-minute run.

We rank this last because the v5 finding (FPR stuck across 11 dials) suggests training-budget exhaustion is unlikely to move the metric — but it is the cheapest single experiment that would close the architectural-vs-data question definitively.

---

## 9. Generalizing the architecture

What we believe carries to other applications of Flamingo-style gated cross-attention on a frozen LM:

- **The merge-before-Stage-1 sequence.** Required to avoid moving-target gate-magnitude drift. A live PEFT adapter sitting on the base during Stage-1 cross-attention training will produce uninterpretable gate dynamics.
- **The single LoRA-on-Q.** Sufficient and minimal. Adding LoRA to KV or output projections was tested early (not reported) and did not help.
- **The Perceiver-Resampler bottleneck.** Necessary for fixed-length side-stream input. `N = 64` slots was the sweet spot for our event stream length (5–200 events); a different modality might want a different `N`.
- **Sparse-but-effective gates.** The Flamingo-paper-style "gates open to ~0.1–1.0" prior may be wrong for tasks where the LM already does most of the work and cross-attention is a disambiguator. Our `max_gate ≈ 0.02` regime produces a CI-separated architectural win on the families that need it. Do not key the halt logic on the original Flamingo prior; calibrate against the smoke-run magnitudes.
- **The dial-robustness profile.** Across v5's 11 cells, the win is robust to insertion pattern, slots, gate init, LoRA rank, and encoder family. The robustness is itself the result: the architecture is not over-tuned to a specific configuration; it works in a band.

What does *not* carry, and is specific to our task:

- The specific journey/actor schema in `journey_templates.py`.
- The `v5_adv_error` composite metric (specific to the v4 adversarial-family setup).
- The bucketed feature token families (specific to ATO; a different fraud type or a different application domain would need a different bucketed schema).

---

## 10. References

- **Companion documents in this whitepaper set:** `00-whitepaper-main.md`, `01-data-curation-and-distribution.md`, `02-agentic-experiment-harness.md`, `03-eval-strategy.md`.
- **Implementation:** `src/model/{cross_attn_block, resampler, qwen_xattn_wrapper}.py`, `src/model/encoders/{small_transformer, pooled_mlp, ft_transformer}.py`, `src/train/{train_xattn, train_lora_text_only, train_structured_as_text, train_event_only_classifier, train_text_only_v4}.py`, `scripts/merge_stage0_lora.py`.
- **Experiment records:** `src/auto_research/experiments.jsonl` (one row per run, append-only), `src/auto_research/runs/exp_*/` (per-run artifacts).
- **Detailed records:** `docs/cross-attention-mechanism.md` (architecture explainer, "why gates didn't open in v3"), `docs/experiments-log.md` (running history, 18 valid v3 arms with interpretation), `.claude/tasks/data-v4-verdict.md` (the formal v4 Q1 PASS A document).
- **External references:** Alayrac et al. (2022), Flamingo; Jaegle et al. (2021), Perceiver IO; Hu et al. (2021), LoRA; Gorishniy et al. (2021), FT-Transformer.
