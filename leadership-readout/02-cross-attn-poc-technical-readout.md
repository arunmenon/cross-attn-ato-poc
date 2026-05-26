# Cross-Attention for ATO — Three-Generation Technical Readout (v3 → v4 → v5)

**v2 · 2026-05-22** (supersedes v1 v3-only readout dated 2026-05-18)

Companion to the leadership executive summary. The full whitepaper version of this content is in `cross_attn_ato_poc/whitepaper/04-cross-attention-experiments.md` (v1.2); this readout is the leadership-friendly distillation.

---

## TL;DR

Three sweep generations, three different stories:

- **v3 (May 15–18):** Null result. Cross-attention did not separate from a structured-as-text baseline within 95% bootstrap CIs.
- **v3 → v4 diagnosis (May 19):** Post-hoc code audit showed the v3 null was a synthetic-data artifact, not an architectural verdict — the narrator was paraphrasing the structured side-stream into the text, collapsing the modality gap.
- **v4 (May 19–20):** Same architecture, fixed data — produced a CI-separated architectural win on adversarial cross-modal fraud (`phish_takeover` recall 0.11 → 1.00; `phish_takeover_mfa_phished` 0.00 → 0.97).
- **v5 (May 21):** Win is robust across 11 architectural configurations, but a new bottleneck emerged on the `hn_recovery_high_amount` adversarial-legitimate family at ~44% FPR — within the 11-run sweep on the 5k eval, no tested dial moved it beyond CI noise.

The v3 null and the v4 win are not contradictory — they are about different data, with the architecture held constant. The v5 ceiling is a data-shape result, not an architecture result. The current strategic call: invest in the loop + redesign the bottleneck family + add a tabular baseline + run production replay; do **not** spend more on architectural sweeps until the data ceiling moves.

---

## 1. v3 — the null result

The original 3-day POC ran **18 valid cross-attention configurations** across:

- `insertion_pattern ∈ {every_4, every_8, late_only}`
- `resampler_slots ∈ {64, 128}`
- `gate_init ∈ {zero, small_0.01}`
- Training-dial perturbations: `lr ∈ {1e-4, 3e-4, 3e-5}`, `warmup ∈ {100, 500}`, `lora_r ∈ {16, 32}`

Compared against four baselines: `CPT-light-merged`, `LoRA-text-only`, `structured-as-text concat`, and an `event-only classifier`.

**Headline number** (metric_version 2, clean eval n=4,466, 95% bootstrap CIs):

| Run | Config | Worst-family HN-FPR @ 1% legit FPR |
|---|---|---|
| `exp_xa_round1_002` (v3 leader) | every_8 / 64 / small_0.01 | **0.0524 [0.0420, 0.0647]** |
| `exp_baseline_structured_as_text_v2` (load-bearing baseline) | concat structured stream into prompt | **0.0507 [0.0408, 0.0635]** |

CIs heavily overlap; the cross-attention leader is +0.0017 absolute **worse** on the point estimate. **No separation.** Round-2 zero-init perturbations confirmed the gates rode their initialization (max-gate 0.0039–0.0041 vs 0.0106–0.0112), with statistically tied HN-FPR.

If we had stopped at v3, the conclusion would have been "cross-attention does not earn its keep on this surface." That was the v1 leadership readout's recommendation. It was honest, and it was wrong.

---

## 2. v3 → v4 — what the diagnosis actually found

A code audit triggered by the unexpectedly clean null found two design pathologies in the v3 synthetic-data pipeline:

### Pathology 1 — narrator-mediated redundancy

`data/gen/narrative_generator.py` was serializing the **full structured event stream** — including the bucket key=value pairs — into the narrator's user message. The narrator's `SYSTEM_PROMPT` further taught it (via compliant-example phrases) to paraphrase these into the output: `amount_bucket=high` → "high-value transfer", `device_age=new` → "previously-unseen device", `recipient_age=newly_added` → "freshly-added recipient". The bucketed fraud signal flowed end-to-end from event → narrator's prompt → narrator's output → narrative text. **The LM, reading the narrative, already had the structured signal; cross-attention had nothing unique to fetch.**

### Pathology 2 — label-deterministic hard-negative skeletons

v3 journey templates hard-coded the feature signatures of each family. `hn_account_recovery` was always `{auth=mfa_strong, device=known, ip=low, geo=local}`; `phish_takeover` was always the opposite. A feature-level classifier could perfectly separate them. The 0.05–0.07 worst-family HN-FPR ceiling in v3 was a small statistical edge effect at the 1%-legit-FPR operating point on a label-deterministic surface — not a measure of model capability.

### Pathology 3 — muddled baseline contract

`text_only` and `xattn` v3 arms consumed different LM prompts (the former included event-line blocks wrapped in `<journey_X>` tokens; the latter did not). The architecture comparison was confounded with a prompt-content comparison.

**The conclusion of the diagnosis:** v3's null was an artifact of all three pathologies stacking. The architecture was not the failing — the eval surface was not actually exercising the architecture's intended advantage.

---

## 3. v4 — the data pivot and the architectural win

The v4 pivot was a coordinated **four-change** rework, designed to restore a real modality gap between the text and event streams.

1. **Strip the narrator's view of bucketed tokens.** Narrator now sees only event types and timing; quantitative bucketed features live exclusively in the structured event stream.
2. **Stochastic feature signatures.** Each journey family draws features from a distribution, not a fixed signature. `hn_account_recovery` now has `{auth: 0.55 mfa_strong / 0.30 password_only / 0.15 cookie_only}`; `phish_takeover` symmetrically uses `auth=mfa_strong` 25% of the time (phished MFA). A feature-level classifier can no longer perfectly separate.
3. **Two new adversarial cross-modal families.** `hn_recovery_high_amount` (legitimate): text reads like fraud, events reveal legitimacy. `phish_takeover_mfa_phished` (fraud): text reads safe, events show the anomaly. **Designed to demand cross-modal reasoning.**
4. **Per-arm text-field routing.** `text_only` and `xattn` arms now see **byte-identical LM prompts** modulo the side stream — the only thing the architecture comparison turns on.

### v4 result — CI-separated win on adversarial fraud

Re-running the same v3 leader configuration on the v4 data:

| Family | n_eval | `text_only_v4` recall | `xattn_v4` recall | Δ | CIs |
|---|---|---|---|---|---|
| `phish_takeover` (fraud) | 224 | 0.1122 | **1.0000** | **+0.89** | **CI-separated** |
| `phish_takeover_mfa_phished` (fraud-dual, v4-new) | 71 | 0.0000 | **0.9718** | **+0.97** | **CI-separated** |

Per-family hard-negative FPR @ 1% legit FPR:

| Family | n_eval | `text_only_v4` | `xattn_v4` |
|---|---|---|---|
| `hn_account_recovery` | 488 | 0.0024 | 0.0000 |
| `hn_large_purchase` | 496 | 0.0014 | 0.0000 |
| `hn_travel` | ~280 | 0.0000 | 0.0000 |
| **`hn_recovery_high_amount`** (adversarial-legitimate, v4-new) | 78 | **0.42** [0.34, 0.51] | **0.45** [0.37, 0.56] |

The v4 architectural win is concentrated on the two adversarial fraud families where the modality gap matters. The hardest adversarial-legitimate family (`hn_recovery_high_amount`) sits at ~0.42–0.45 FPR for both arms with overlapping CIs — cross-attention is +0.03 *worse* on the point estimate but statistically tied. This is the data-shaped ceiling that v5 then confirmed is architecture-immutable across 11 variations.

**Gates at v4:** `max_gate_magnitude = 0.0221` on the v4 architecture-winner, above the 0.011 v3 ceiling but well below the Flamingo "open" target of ~0.1. The gates are **sparse but effective** — the LM does most of the discrimination; the gate opens narrowly only where the side stream resolves a textual ambiguity. The +0.97 swing in `phish_takeover_mfa_phished` recall is achieved with `tanh(α) ≈ 0.022` per inserted block, summed over 3 blocks (every_8 configuration).

---

## 4. v5 — 11-run robustness sweep, one ceiling

The v5 expansion ran 11 cross-attention configurations across two phases against a new primary metric `v5_adv_error` (the mean of three components: `phish_takeover_miss`, `phish_takeover_mfa_phished_miss`, `hn_recovery_high_amount_fpr`).

**Phase 1 — training and arch-dial sweep** (8 cells, varying one dial at a time around the v4 seed):

| exp_id | Config | v5_adv_error [CI] |
|---|---|---|
| `exp_xattn_v4_001` (seed) | every_8 / 64 / small_0.01 / small_transformer / lora_r=16 | 0.1596 [0.1306, 0.1945] |
| **`exp_v5_p1_zero_64` ★** | every_8 / 64 / **zero** / small_transformer / lora_r=16 | **0.1506 [0.1238, 0.1871]** |
| `exp_v5_p1_every4_64` | **every_4** / 64 / small_0.01 | 0.1506 |
| `exp_v5_p1_late_64` | **late_only** / 64 / small_0.01 | 0.1546 |
| `exp_v5_p1_slots128` | every_8 / **128** / small_0.01 | 0.1526 |
| `exp_v5_p1_slots32` | every_8 / **32** / small_0.01 | 0.1660 |
| `exp_v5_p1_rank32` | every_8 / 64 / small_0.01 / **lora_r=32** | 0.1623 |
| `exp_v5_p1_slowlr` | **lr=3e-5** / warmup=500 | 0.2021 |
| `exp_v5_p1_fastlr` | **lr=3e-4** / warmup=100 | **0.7516 (catastrophic regress)** |

**Phase 2 — encoder sweep** on the Phase-1 winner config:

| Encoder | v5_adv_error [CI] |
|---|---|
| `small_transformer` (P1 winner) | **0.1506 [0.1238, 0.1871]** |
| `pooled_mlp` | 0.1549 [0.1282, 0.1938] |
| `ft_transformer` | 0.1737 [0.1448, 0.2136] |

Phase 2 stop rule fired: neither encoder alternative beat the Phase-1 winner by ≥0.005 absolute.

### v5 findings

**Win is dial-robust within the LM family.** Across non-pathological Phase-1 + Phase-2 runs, `v5_adv_error` sits in the band [0.151, 0.174]. The v4 architectural win survives every dial we tested — insertion pattern, slots, gate init, LoRA rank, encoder family.

**The bottleneck shifted.** v3's worst family was `hn_account_recovery` (~0.05–0.07 FPR). v5's worst family is `hn_recovery_high_amount` (~0.44 FPR across every non-pathological run). The shift is by design — v4 introduced this family as an adversarial-legitimate stress test.

**Within the 11-run v5 sweep on the 5k clean eval (this family has n=78, per-family CI width ~0.18), no tested architectural dial moved this bottleneck beyond bootstrap-CI noise.** The component contributes ~97% of the total `v5_adv_error` (0.146 of 0.151) for the Phase-1 winner. The ceiling is upstream in the synthetic generator's `hn_recovery_high_amount` template — either (a) the template doesn't carry enough disambiguating signal, or (b) the family is a genuine production ambiguity for which "route to step-up auth" is the right action, not binary fraud/legit classification.

---

## 5. The strategic call

**Invest in the loop; validate cross-attention through data and replay, not blind architecture sweeps.**

The cross-attention finding is no longer "doesn't earn its keep." It's "validated synthetic case study — production validation is the next gate."

### Concrete next steps, in priority order

1. **Now (in flight):** Package the auto-research loop as a Foundation Science capability — launcher, agent playbook, whitelist, lockfile, dedup, halt conditions, bootstrap-CI eval pipeline, tie-aware metrics. Every subsequent POC inherits these for free.
2. **Next (1–4 weeks):** Redesign the `hn_recovery_high_amount` generator template with richer disambiguating event context (device-trust history, step-up-auth chain, recovery chain, out-of-band verification signals). Target: drop the bottleneck FPR below 0.15 before any further architecture sweep.
3. **Then (1–2 months):** Run an anonymized production-replay validation against the v5 Phase-1 winner config, and add a strong tabular baseline (XGBoost or LightGBM trained on the same bucketed event features). This is the test that converts the v4 synthetic win into either a production claim or a data-quality claim — both are valuable.

### What we explicitly do **not** claim

- **Production transfer.** The v4 win is on synthetic data. Real-traffic behavior is open.
- **Production-fraud-system superiority.** We have not benchmarked against any current PayPal ATO routing system.
- **Calibration.** Eval is discrimination-only (AUC, R@FPR, HN-FPR). No ECE, Brier, or reliability-diagram results.
- **Non-LLM tabular baseline superiority.** We have not trained XGBoost or LightGBM on the same bucketed features. The `text_only`-vs-`xattn` architectural comparison is clean within the LM family; it is **not** a fraud-model competitiveness claim.

---

## 6. Limitations

- **Synthetic data only.** The narrator (`gpt-5-nano-2025-08-07` for v4/v5) and the structured-event generator are template-driven approximations of PayPal session schemas. Not adversarially co-designed.
- **Single base LM, single scale.** Qwen3-8B post-CPT-light. We did not test Llama 3.1 8B, Mistral 7B, or other Qwen scales. Architectural conclusions are conditional on this base.
- **Gate magnitudes operate in a regime below Flamingo's predicted range** (~0.02 vs ~0.1–1.0). The architecture works at these magnitudes, but the training dynamics are not what Flamingo's paper would predict — likely because the LM is already very good at most examples and the side stream is needed only on the adversarial edge cases.
- **`hn_recovery_high_amount` is one family.** The data ceiling is on a single adversarial-legitimate family. We don't know whether the same ceiling exists on other constructions of the same shape, or whether the production analog suffers the same failure mode.
- **No compute budget exhausted.** v3 used 7.735 / 18 GPU-hours; v5 used 7.92 / 12. Specifically, the v5 stress option (steps=3000, seq_len=4096) on the Phase-1 winner was not exercised. A longer run might move `phish_takeover_mfa_phished` recall slightly but is unlikely to move `hn_recovery_high_amount` (per the dial-invariance finding).

---

## 7. Where to read more

- **Full technical depth:** `cross_attn_ato_poc/whitepaper/04-cross-attention-experiments.md` (v1.2; ~4,600 words).
- **Data pipeline detail:** `cross_attn_ato_poc/whitepaper/01-data-curation-and-distribution.md`, including the four-change v4 pivot diff.
- **Eval-pipeline detail:** `cross_attn_ato_poc/whitepaper/03-eval-strategy.md`, including the `metric_version 1 → 2 → 5` evolution and the tie-aware exact-target operating-point formula.
- **Reproducibility:** every numeric claim in this readout traces to `cross_attn_ato_poc/src/auto_research/experiments.jsonl` or to a per-run `runs/exp_*/{metrics,ci_report,gate_trajectory,leakage_report}.json`. The whitepaper's `04` document has the exact shell commands.
