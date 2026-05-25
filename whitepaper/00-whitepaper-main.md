# A Guardrailed Agentic Research Loop for Cross-Modal Fraud Modeling

### A Case Study in Gated Cross-Attention for Synthetic ATO Detection

**Whitepaper · v1.2 · 2026-05-22**
Arun Menon — Foundation Science (PayPal)

---

## Executive summary

A one-page take for non-research readers; the rest of the paper supports each line.

- **What we built.** An agentic research loop with hard-edge guardrails — an LLM agent proposes the next experiment, a deterministic Python launcher validates, locks, runs, parses, computes bootstrap CIs, and writes one immutable row to history. Paired with a leakage-safe synthetic-data pipeline and a tie-aware exact-target operating-point eval, this is the **reusable artifact**.
- **What we tested.** Flamingo-style gated cross-attention on a frozen Qwen3-8B language model, with a side-stream encoder over a structured event timeline gated into the LM stack via a Perceiver-Resampler and per-layer scalar gates. Account-takeover (ATO) on a synthetic dataset modeled on PayPal session schemas — the **worked example**.
- **What happened.** v3 returned a null result; a code audit traced it to a synthetic-data leak where the narrator paraphrased the event signal into the text. v4 fixed the data pipeline; the same cross-attention configuration produced a confidence-interval-separated win on adversarial cross-modal fraud (`phish_takeover` recall 0.11 → 1.00; `phish_takeover_mfa_phished` recall 0.00 → 0.97). v5 expanded the sweep to 11 cells; the win is dial-robust, and within that sweep no tested architecture dial moved a separate data-shaped ceiling on the hardest adversarial-legitimate family beyond CI noise — bounded by the family's small per-eval sample (n=78), which the next 50k-eval test would tighten ~3×.
- **What we learned.** The loop is portable: harness + leakage-safe data + bootstrap-CI eval generalize beyond this architecture and task. The architecture works **when given a problem it can solve**; on this synthetic surface, it solves adversarial cross-modal fraud and hits a data ceiling on one adversarial-legitimate family.
- **What's next.** Real-data replay on an anonymized production window (§8.1 roadmap), bottleneck-family data redesign, and a non-LLM tabular baseline (XGBoost on bucketed events) for fraud-audience defensibility. Production transfer is **not claimed** in this paper.

**Bottom line.** *The cross-attention finding is the worked example; the loop is the reusable artifact.*

### How to read this paper

The paper is organized for three audience types and you can take any of them as the entry point:

- **Readers interested in the reusable research system.** Start with §3.2 (the harness in 10 steps) and Figure 2 (the dataflow). The companion `02-agentic-experiment-harness.md` is the full design document. Skim §4 only for context on what the harness was used to discover.
- **Readers interested in the model result.** Start with §4 (the v3 → v4 → v5 arc) and §5 (the leaderboard), then look at Figure 4 (sweep results with the adversarial-error decomposition). The companion `04-cross-attention-experiments.md` has the full per-cell numbers and the gates story.
- **Readers evaluating transfer risk to production.** Start with §1.2 (claims at a glance — the table that marks production transfer, non-LLM tabular baselines, and calibration as "not claimed"), then §7 (limitations) and §8.1 (roadmap). The architectural result and the methodology contribution are both bounded by the synthetic-data caveat; the table is the most efficient way to see what *is* and *is not* in evidence.

---

## Abstract

We report on a three-generation study (v3 → v4 → v5) of Flamingo-style gated cross-attention applied to **synthetic** account-takeover (ATO) detection on a single PayPal-internal H100 GPU. All results in this paper are on synthetic data modeled on PayPal session schemas; production transfer is explicitly out of scope and not claimed (§7, §8.1). A frozen Qwen3-8B language model (post-light continued pre-training) reads an analyst-style narrative while a side-stream encoder over a structured event timeline is gated into the LM stack via a small Perceiver-Resampler and per-layer scalar gates. The work has two contributions. First, we present a Karpathy-style **agentic experiment harness** in which an LLM agent proposes the next experiment and a deterministic Python launcher validates, locks, runs, parses, computes bootstrap confidence intervals, and writes one immutable row to history — with a strict single-writer-per-file ownership invariant. The harness ran 30 cross-attention and baseline configurations end-to-end across the three sweep generations with zero format-drift, zero concurrency races, and zero manual reconciliation. Second, we use the harness to surface, diagnose, and correct two design pathologies that hid the architecture's signal: (a) a synthetic-data pipeline in which the narrator paraphrased the structured side-stream into the text (collapsing the modality gap that cross-attention was designed to exploit) and (b) deterministic feature signatures in hard-negative templates that made the eval surface label-deterministic in observed support. Once both pathologies were repaired in the v4 data pivot, the same cross-attention configuration that ranked as null in v3 produced a confidence-interval-separated lift on adversarial cross-modal fraud families: text-only achieved 0.000 recall on `phish_takeover_mfa_phished` against cross-attention's 0.972 (95% bootstrap CI [0.931, 1.000]). The v5 expansion (11 runs across training-dial and encoder sweeps) confirmed the win was robust to gate initialization, insertion density, resampler capacity, LoRA rank, and side-stream encoder family, and surfaced a new data-shaped ceiling on the `hn_recovery_high_amount` adversarial-legitimate family at ~44% false-positive rate: within the 11-run v5 sweep on the 5k clean eval (this family has n=78, per-family CI width ~0.18), **no tested architecture dial moved the bottleneck beyond bootstrap-CI noise**. A 50k LLM-narrated eval would tighten this claim ~3× and is recommended as the next test. We argue the methodology — agentic loop + tie-aware bootstrap-CI evaluation + iteratively repaired synthetic data — generalizes beyond this specific architecture and task. The cross-attention finding is the worked example; the loop is the reusable artifact.

**Keywords.** Cross-attention, Flamingo-style adapter, frozen language model, account-takeover detection, agentic experiment harness, auto-research loop, bootstrap confidence intervals, tie-aware operating-point metrics, synthetic-data evaluation, structured-stream encoding.

---

## 1. Introduction

A line of recent work — Flamingo (Alayrac et al., 2022) and its descendants — frames non-textual modalities as side documents that a frozen language model attends to through inserted gated cross-attention layers. This has worked for images and video; the obvious question for risk and fraud teams is whether it works for structured behavioral event streams. The application domain is account-takeover (ATO) detection at PayPal: each session has two natural views — an analyst-style narrative of what happened, and a chronological log of structured events (logins, transactions, device changes, password resets, with bucketed features such as `amount_bucket=high` and `geo_distance=international`). Self-attention reads the narrative well; the event stream is where the fraud signal historically lives. Cross-attention is the obvious bridge.

This whitepaper documents three sweep generations of that hypothesis, the agentic experiment harness that drove every run, and the data and eval design that made the results credible. The headline outcome is twofold: a methodology contribution (the harness and the bootstrap-CI-defended evaluation pipeline that comes with it) and a result contribution (a CI-separated architectural win on adversarial cross-modal fraud, conditional on a data pipeline that does not paraphrase the side stream into the text — and a finding that, within the 11-run v5 sweep on the 5k eval, no tested architecture dial moved a separate data-shaped ceiling on the hardest adversarial-legitimate family beyond bootstrap-CI noise). Figure 2 (below) shows the loop that drove every run; it is the artifact this paper claims is most reusable beyond the specific cross-attention case study.

![Figure 2. Auto-research loop dataflow](figures/fig2-auto-research-loop.svg)

The full arc is summarized in §2 (related work), §3 (method — including a deeper walk-through of the harness in §3.2), §4 (the v3 → v4 → v5 experimental arc), §5 (results), §6 (discussion), §7 (limitations), §8 (conclusion). Four companion documents go deeper on each pillar: `01-data-curation-and-distribution.md`, `02-agentic-experiment-harness.md`, `03-eval-strategy.md`, and `04-cross-attention-experiments.md`. Four figures accompany the set: an architecture diagram (Figure 1), the loop's dataflow (Figure 2, above), data distribution and eval-mode design (Figure 3), and the v5 sweep leaderboard with bootstrap-CI decomposition (Figure 4).

### 1.1 Contributions, plainly

> **The cross-attention finding is the worked example; the loop is the reusable artifact.**

The contributions of this work, ranked by what we believe will most generalize beyond the specific architecture and task:

1. **A Karpathy-style agentic experiment harness with hard-edge guardrails.** An LLM agent proposes the next experiment as a YAML config; a deterministic Python launcher does everything else. Single-writer-per-file ownership eliminates format drift; bootstrap CIs on every reported metric (1000 resamples) are computed inline; tie-aware exact-target operating-point metrics are the default; halt conditions are explicit and configurable per phase. The harness ran 30 experiments across three sweep generations with zero out-of-band intervention. Section §3.2 and the companion document `02-agentic-experiment-harness.md` give the full design.

2. **A leakage-safe synthetic-data pipeline for cross-modal fraud detection.** Three token families (PII-fencing, bucketed feature, journey/actor structural), three eval modes (stripped, opaque, full), and three eval-set sizes (5k LLM-narrated / 15k LLM-narrated diagnostic / 50k templated medium-eval) with pre-narration structured-events-hash stratification and a post-narration text-hash dedup invariant — collectively close every leakage vector we identified across the three sweep generations. Section §3.1 and `01-data-curation-and-distribution.md`.

3. **A reproducible mid-POC evaluation correction.** The v3 first-cut leaderboard had `event_only` apparently outperforming the LM baselines by 5–7× on worst-family hard-negative false-positive rate. Codex code review caught the artifact: sklearn's `recall_at_fpr` uses a "largest achievable FPR ≤ target" boundary rule, which under tied score masses lands different models at materially different achieved legit-FPRs. We replaced it with a tie-aware exact-target metric (`metric_version: 2`), then evolved that into a multi-component adversarial-error decomposition (`metric_version: 5`) once the v4 data made adversarial fraud families a first-class concern. Section §3.3 and `03-eval-strategy.md`.

4. **A negative-then-positive result on Flamingo-style gated cross-attention for ATO.** v3 produced a null result that was honest but uninformative (the architecture cannot win when the structured signal is already paraphrased into the text). v4 fixed the data pipeline and produced a clean CI-separated win on two adversarial cross-modal fraud families (`phish_takeover` recall 0.11 → 1.00, `phish_takeover_mfa_phished` recall 0.00 → 0.97). v5 confirmed the win is dial-robust across 11 architecture and training configurations but exposed a new data ceiling on one adversarial-legitimate family. Section §4 and `04-cross-attention-experiments.md`.

### 1.2 Claims at a glance

Every quantitative claim in this paper, classified by strength. Read this table before any specific number elsewhere in the paper.

| Claim | Evidence | Strength | Where in paper |
| --- | --- | --- | --- |
| Harness prevents format drift, concurrency races, and out-of-band intervention | 30 runs across three sweep generations, immutable history, zero manual reconciliation | **Strong** | §3.2; `02-harness §9` |
| Two mid-POC metric corrections rolled forward without retraining | `metric_version: 1 → 2 → 5` via `scripts/rescore_baselines.py` and the v5 trainer-side scoring | **Strong** | §3.3; `03-eval §3` |
| v4 cross-attention beats text-only on adversarial cross-modal fraud | CI-separated on `phish_takeover` (recall 0.11 → 1.00) and `phish_takeover_mfa_phished` (0.00 → 0.97); source `runs/exp_{text_only,xattn}_v4_001/ci_report.json` | **Strong within synthetic eval** | §4.2; `04 §4` |
| v5 architectural dials do not move the `hn_recovery_high_amount` ceiling | 11-run sweep, FPR stuck in [0.4377, 0.4505] for all non-pathological runs | **Medium** — bottleneck family n=78 in 5k eval; per-family CI width ~0.18; 50k LLM-narrated eval (available, not yet scored) would tighten ~3× | §4.3; `04 §5.3` |
| Production transfer (held-out anonymized window) | Not tested | **Not claimed** | §7; §8.1 roadmap |
| Beats a non-LLM tabular baseline (XGBoost / LightGBM on bucketed events) | Not tested | **Not claimed** | §7; `04 §8` |
| Calibration (ECE, Brier, reliability) | Not measured; eval is discrimination-only | **Not claimed** | §7; §8.1 roadmap |

### 1.3 What this paper is not

**Every result in this paper is on synthetic data.** We do not generalize beyond the synthetic distribution; production transfer is explicitly out of scope. We do not claim to beat any production fraud system, nor against any non-LLM tabular baseline (XGBoost / LightGBM on bucketed features) — neither was tested. We do not propose a new cross-attention variant; the architecture is Flamingo's, applied verbatim. We do not claim the agentic harness is novel in principle (Karpathy publicly sketched the agent-proposes/script-enforces pattern); we claim that an end-to-end implementation, with the specific ownership-invariant + bootstrap-CI + tie-aware-metric instantiation we describe, was decisive in producing a reproducible mid-POC correction and an honest negative-then-positive result that a less-disciplined harness would have either missed or muddled.

---

## 2. Related Work

**Frozen LMs with cross-attention to non-text modalities.** Flamingo (Alayrac et al., 2022) introduced the gated cross-attention adapter on a frozen LM as a vision-language recipe; its design choices — a frozen base, a Perceiver-Resampler bottleneck, scalar `tanh(α)` gates initialized at zero, inserted at periodic depth in the LM stack — were adopted verbatim in our cross-attention surgery (Figure 1). Concurrent and follow-on work (BLIP-2, IDEFICS, GIT-2, OpenFlamingo) explored the same backbone for image-text and video-text; we are not aware of published work applying the same shape to structured behavioral event streams for fraud detection. The closest neighbors are tabular-foundation-model proposals (FT-Transformer; TabPFN) — we use FT-Transformer as one of three Phase-2 v5 encoder variants and find it neutral-to-negative on this task; see §5.

**Adapter-based fine-tuning on frozen LMs.** Our trainable Stage-1 footprint comprises (i) a small side-stream encoder, (ii) the Perceiver-Resampler, (iii) the inserted gated cross-attention blocks, and (iv) LoRA-on-Q with rank ∈ {16, 32, 64} on the LM's self-attention query projection. LoRA (Hu et al., 2021) is canonical; we attach it only to Q to avoid stacking adapters on a CPT-light adapter that has itself been merged into the base (one of our v3-discovered integration frictions; see `02-agentic-experiment-harness.md`).

**Agentic and autonomous experiment loops.** Karpathy popularized the "agent proposes, deterministic script enforces" pattern in talks and online writing; AutoML systems (Auto-WEKA, AutoKeras, AutoGluon) have automated hyperparameter search for two decades. Recent LLM-driven autonomous-research systems (Sakana AI Scientist; researcher-coding-agent demonstrations) sit on the propose side without the deterministic guardrail. Our harness leans toward the AutoML side on safety (single-writer-per-file, dedup tuple, configurable halt conditions) and toward the agent side on flexibility (the agent decides which dial is worth perturbing given the history). The bootstrap-CI + tie-aware-metric component is independent of the agentic layer and would compose with either.

**Operating-point-controlled evaluation under tied score masses.** sklearn's `recall_at_fpr` uses the "largest achievable FPR ≤ target" boundary rule, which under bimodal classifier outputs with large tied masses (as in our v3 `event_only` baseline) reports metrics computed at materially different operating points. Tie-aware exact-target operating-point computation has been described in the calibration and AUC-decomposition literature (DeLong et al., 1988, in the AUC variance context); we apply it inline in the bootstrap pipeline. Section §3.3 and `03-eval-strategy.md` give the formula.

**Synthetic-data evaluation pitfalls.** A separate body of work (Schwartz et al., 2022; Caton & Haas, 2024) documents that high-fidelity synthetic data can saturate downstream metrics if the generator's structured signal leaks into the surface form available to the model. Our v3 narrative-leakage finding is an instance of this pathology; we describe the diagnostic (overlap audit + per-family hash analysis) in `01-data-curation-and-distribution.md`. Schemata-stratified train/eval splits (Krause et al., 2020) are a partial answer; we add a post-narration text-hash dedup invariant on top.

---

## 3. Method

### 3.1 Data

![Figure 3. Data distribution & eval-mode mix](figures/fig3-data-distribution.svg)

The synthetic ATO dataset has three token families (Figure 3D):

1. **PII-fencing tokens** (`<acct_id>`, `<email>`, `<device_id>`, `<phone>`, `<ip>`, `<recipient>`, `<merchant>`, `<browser>`). Opaque placeholders carrying no signal. Always visible at eval. Hygiene, not features.
2. **Bucketed feature tokens** (`<amount_bucket=high>`, `<geo_distance=international>`, `<ip_risk=high>`, `<device_age=new>`, `<merchant_risk=elevated>`, `<txn_velocity=bursty>`, `<recipient_age=newly_added>`, `<session_dwell=extended>`, `<auth_strength=mfa_strong>`). Privacy-respecting derived features. **This is where the fraud signal lives.** Always visible at eval.
3. **Journey / actor structural tokens** (`<journey_phish_takeover>`, `<actor_agent_compromised>`, etc.). Mark the journey type and the acting party. **Stripped or opacified at eval** to prevent the model from cheating off the label.

Each training example is a paired observation: a structured event stream (consumed by the side-stream encoder) and an analyst-style narrative + verdict footer (consumed by the LM). The narrator (OpenAI `gpt-5-nano` family — v3 used the then-current alias, and v4 and v5 pin the dated snapshot `gpt-5-nano-2025-08-07` for reproducibility) was prompted to ban explicit class names ("fraud", "ATO", "phishing"); a post-generation regex scan in `eval/leakage_checks.py::narrative_leakage_scan` enforced the ban. Volume: 25,000 LLM-narrated training pairs and a 5,002-row stratified held-out eval after the v4 leakage controls drop zero rows (vs. v3, where 534 of 5,000 rows had to be excluded by `compute_clean_eval_mask` due to narrator-cache leakage; see §4.1).

**The v4 four-change pivot** (introduced in §4.2; full rationale in `01-data-curation-and-distribution.md`) closed the two data pathologies the v3 sweep exposed. (i) The narrator's view was stripped of bucketed feature tokens — the narrator now sees only event types and timing, not the quantitative buckets that carry the fraud signal. (ii) Hard-negative templates were re-implemented to sample feature signatures from distributions rather than fixed signatures, breaking the label-deterministic feature-level discrimination that v3's `hn_account_recovery` family inadvertently created. (iii) Two new adversarial cross-modal families were added — `hn_recovery_high_amount` (legitimate but text-reads-fraud) and `phish_takeover_mfa_phished` (fraud but text-reads-safe) — designed to demand that the model attend to the event stream to disambiguate. (iv) Trainer dataloading was refactored so `text_only`, `structured_as_text`, and `xattn` arms receive byte-identical LM prompts modulo the presence/absence of the side stream — the only thing the architecture comparison turns on.

Distribution details, leakage controls, dataset-card schema, and the full v3-vs-v4 generator diff are in `01-data-curation-and-distribution.md` and Figure 3.

### 3.2 Agentic experiment harness

Figure 2 (rendered in §1 above) shows the full dataflow. The decomposition is:

```
cron (every 30 min)
  → agent_tick.sh  (GPU-lock pre-check, 180-min timeout wrapper)
    → Claude Code CLI  (one tick)
      ├── reads:    AGENT_INSTRUCTIONS.md, sweep_state.yaml,
      │             experiments.jsonl (last 5), runs/exp_*/notes.md
      ├── proposes: runs/exp_NNN/config.yaml
      ├── invokes:  python scripts/run_next_experiment.py <config>
      │              ├── 1. whitelist validate
      │              ├── 2. canonical_hash dedup vs history
      │              ├── 3. halt-check on sweep_state.yaml
      │              ├── 4. acquire GPU lockfile
      │              ├── 5. accelerate launch train_xattn.py
      │              ├── 6. parse_metrics (stdout + W&B)
      │              ├── 7. bootstrap_ci (1000 resamples, tie-aware)
      │              ├── 8. atomic write all outputs
      │              ├── 9. append one row to experiments.jsonl
      │              └── 10. refresh sweep_state.yaml
      ├── writes:   runs/exp_NNN/notes.md (interpretation)
      └── (later)   README.md (Day-N + Final journey log)
```

The ownership invariant is hard: the launcher is the *only* writer of `experiments.jsonl`, `sweep_state.yaml`, and the structured per-run JSON artifacts (`metrics.json`, `ci_report.json`, `gate_trajectory.json`). Leakage state is recorded inline on each `experiments.jsonl` row (`leakage_clean`, `clean_eval_n`, `clean_eval_dropped`, `clean_eval_mask_text_overlap`, `clean_eval_mask_events_overlap`) rather than as a separate per-run file. The agent is the *only* writer of `config.yaml`, `notes.md`, and the README journey log. Source-code edits (`src/`, `scripts/`, `eval/`, `data/gen/`) require an explicit `git add -A && git commit -m "snapshot before <change>"` from the agent before any modification, so the launcher's behavior is always revertible to a known state.

Halt conditions are configurable per sweep phase. v3 ran with NaN-cascade, zero-gate-activation, convergence, and budget caps all active; the convergence halt fired prematurely on Day-2 (the leader sat in window-slot 1, making it mathematically impossible for a sibling to beat it by ≥0.005 within the window), and we disabled it for v4 and v5. NaN-cascade + zero-gate-activation + budget caps remained the real stops. v5 added an `early-exit-on-success` rule that pivots to local perturbations if any single run records `max_gate_magnitude ≥ 0.05` AND beats the current best with non-overlapping bootstrap CIs.

Full launcher logic, agent playbook, halt-condition design, dedup-tuple rationale, and the harness's three-generation evolution are documented in `02-agentic-experiment-harness.md`.

### 3.3 Evaluation

Three eval modes are applied symmetrically during training (via per-example dropout: 50% full / 25% opaque / 25% stripped, Figure 3C) and at evaluation:

- **`stripped`** — all journey/actor structural tokens removed. **Headline metric.**
- **`opaque`** — journey/actor tokens replaced with neutral random IDs. Secondary.
- **`full`** — all tokens visible. Debug only; never reported as a win condition.

Three eval-set sizes form a ladder: 5k LLM-narrated (per-experiment), 15k LLM-narrated (diagnostic, used to confirm saturation is not eval-size-specific), and 50k templated medium-eval (built but unused in v3 due to the convergence halt; the v4 and v5 runs evaluated on the 5k surface).

**Primary scoring** is `score = logP(' fraud' | prefix) − logP(' legit' | prefix)` where the prefix is the narrative + `<risk_verdict>\nlabel:` portion of the example. AUC over this score against ground truth is the sanity-only column (it saturated at 1.0 on every variant by v3 Day-1, so it cannot rank). The headline metric pivoted three times across the sweep:

- **`metric_version: 1`** (v3 Day-1, pre-correction): worst-family hard-negative FPR @ 1% legit-FPR, sklearn's `recall_at_fpr` rule. Replaced after the sklearn-cliff correction.
- **`metric_version: 2`** (v3 Day-2, post-correction): the same worst-family HN-FPR, but with a **tie-aware exact-target legit-FPR** computation that walks descending scores until cumulative legit count hits exactly `target_fpr × n_legit` (kept as a float, not rounded), then weights tied-at-threshold rows by `alpha = (target_count − n_above) / n_tied`. The metric emits `(threshold, alpha, achieved_fpr, n_above, n_tied, tie_fraction)` per row, so every CI bound is verifiable from JSON. Bootstrap CIs recompute `(threshold, alpha)` per resample.
- **`metric_version: 5`** (v5, current): the **v5_adv_error** decomposition. v4's two new adversarial families needed first-class metric treatment; v5 combines them into a single scalar that the loop optimizes:

```
v5_adv_error = (1/3) × (phish_takeover_miss
                       + phish_takeover_mfa_phished_miss
                       + hn_recovery_high_amount_fpr)
```

Each component is a 1%-legit-FPR-operating-point error rate computed with the same tie-aware exact-target machinery as `metric_version: 2`. Bootstrap CIs are propagated by resampling the underlying score table and recomputing the three components per resample, then re-averaging. Per-component CIs are also reported for diagnostic decomposition (Figure 4B).

Full metric derivations, CI computations, and the train/eval leakage-control regime (text-hash and structured-events-hash dedup, narrative-leakage scan, clean-eval mask) are in `03-eval-strategy.md`.

### 3.4 Architecture

![Figure 1. Cross-attention surgery on Qwen3-8B](figures/fig1-architecture.svg)

We follow Flamingo (Alayrac et al., 2022) with the minimal adaptations required to swap images for structured event streams. Figure 1 illustrates.

- **Base.** Qwen3-8B (36 layers, hidden 4096, 32 attention heads, 8 KV heads). The base is first put through a Stage-0 continued pre-training pass (`CPT-light`: new-token embedding + LoRA on attention and MLP, ~1500 steps on the LLM-narrated training pool). The Stage-0 LoRA is then **merged into the base weights** (`scripts/merge_stage0_lora.py`) to produce `qwen3-8b-cpt-light-merged` (and after the v4 pivot, `qwen3-8b-cpt-light-v4-merged`). Stage-1 cross-attention training operates on this merged, frozen checkpoint.
- **Side-stream encoder.** Three variants swept in v5 Phase-2: `small_transformer` (6-layer transformer over event tokens; default), `pooled_mlp` (mean/max-pool + MLP projection), `ft_transformer` (FT-Transformer style tabular encoder). Trained from scratch jointly with the rest of Stage-1.
- **Perceiver-Resampler.** Sinusoidal-on-Δt time encoding on input; N learned query slots cross-attend to encoder output, producing N × hidden K/V slots for downstream cross-attention. N swept ∈ {32, 64, 128}.
- **Gated cross-attention blocks.** Inserted at `every_4`, `every_8`, or `late_only` depth in the LM stack — 6, 3, or 4 inserted blocks respectively in a 36-layer model. Insertions start at layer 12 by design (so cross-attention does not disturb early token-feature extraction; see `src/model/qwen_xattn_wrapper.py:163-186`). Per-block scalar `tanh(α)` gate, initialized at `α ∈ {0, 0.01}`.
- **LoRA-on-Q.** Rank-16 (also 32, 64 swept in v5 Phase-1) on the LM's self-attention query projection. This is the *only* trainable parameter touching the LM's frozen weights.

Stage-1 trainable parameter count: ~110M–220M depending on insertion density and slot count (1.5–2.7% of the 8B base). Training: 1500 steps, bf16, paged-adamw-8bit, cosine LR schedule with 500-step warmup, peak LR 1e-4. Sequence length 2048 (4096 reserved for one optional stress run; not exercised in the v5 final sweep).

Full architecture detail, the alternative architectures we considered and rejected, and the Flamingo-style design-choice rationale are in `04-cross-attention-experiments.md`.

---

## 4. Experimental Arc

The three sweep generations were not pre-planned; each followed from the findings of the prior generation. The arc is the story.

### 4.1 v3 — the false null

The original 3-day POC ran 18 valid cross-attention configurations across `insertion_pattern ∈ {every_4, every_8, late_only}`, `resampler_slots ∈ {64, 128}`, `gate_init ∈ {zero, small_0.01}`, plus training-dial perturbations (LR ∈ {1e-4, 3e-4, 3e-5}, warmup ∈ {100, 500}, LoRA-r ∈ {16, 32}, and one stress run at seq_len=4096). All comparisons against four baselines: CPT-light-merged, LoRA-text-only, structured-as-text concat, and an event-only classifier.

**v3 headline.** The Round-1 leader `exp_xa_round1_002` (every_8 / slots=64 / gate=small_0.01) achieved worst-family HN-FPR @ 1% legit-FPR = **0.0524 [CI 0.0420, 0.0647]** on the clean 5k eval. The load-bearing baseline `structured_as_text_v2` achieved **0.0507 [CI 0.0408, 0.0635]**. CIs heavily overlapped; cross-attention did not separate from concatenating the structured stream into the prompt text. Round-2 zero-init perturbations confirmed the gates rode their initialization (max-gate magnitudes 0.00385–0.00412 vs. 0.0106–0.0112 for small-init), with statistically tied HN-FPR across the matched pairs — the architectural lift was, on the v3 surface, indistinguishable from noise.

**Post-hoc diagnosis.** During the v4 pivot planning, a code audit revealed two root causes of the null result:

1. **Narrator-mediated redundancy.** `data/gen/narrative_generator.py::_serialize_events_for_prompt` embedded the full structured event stream — including the bucket key=value pairs — in the narrator's user message. The narrator's `SYSTEM_PROMPT` further taught it (via compliant-example phrases) to paraphrase these into the output: `amount_bucket=high` → "high-value transfer", `device_age=new` → "previously-unseen device", `recipient_age=newly_added` → "freshly-added recipient". The result: the bucketed signal flowed end-to-end from event → narrator's prompt → narrator's output → narrative text. The LM, reading the narrative through self-attention, already had the structured signal; cross-attention had no unique information to fetch.

2. **Label-deterministic hard-negative skeletons.** `data/gen/journey_templates.py` hard-coded the feature signatures of fraud-vs-hard-negative families. `hn_account_recovery` (legit) always had `{auth=mfa_strong, device=known, ip=low, geo=local, recipient=known}`; `phish_takeover` (fraud) always had the opposite. A feature-level classifier could perfectly separate them; the worst-family HN-FPR ceiling at 0.05–0.07 was not a measure of model capability but of a small statistical edge effect at the 1% legit-FPR operating point on a label-deterministic surface.

3. **Pathology 3 — muddled baseline contract.** `text_only` and `xattn` saw different LM prompts (the former included event-line blocks wrapped in `<journey_X>` tokens; the latter did not). The architecture comparison was confounded with a prompt-content comparison.

The null result was honest but uninformative. The eval surface did not admit an architectural win.

### 4.2 v4 — the data pivot

The v4 pivot was a coordinated four-change rework of the data and trainer contract, designed to restore a real modality gap between text and event streams (Figure 3B; `01-data-curation-and-distribution.md` has the full diff).

1. **Strip the narrator's view of bucketed tokens.** The narrator now sees only event types and timing, plus qualitative actor descriptors. All nine bucket key=value pairs are removed from the narrator's prompt. Compliant-example phrases in `SYSTEM_PROMPT` are rewritten to avoid quantifying ("a transfer" instead of "a high-value transfer"). The narrator is now describing behavioral *shape*, not features. The features live exclusively in the structured event stream consumed by the side encoder.

2. **Stochastic feature signatures.** Each journey family draws features from distributions, not fixed signatures. `hn_account_recovery` now has `{auth_strength: 0.55 mfa_strong / 0.30 password_only / 0.15 cookie_only}` and similar distributions on every other feature. Symmetrically, `phish_takeover` (fraud) sometimes uses `auth=mfa_strong` (attacker phished MFA). A feature-level classifier can no longer perfectly separate.

3. **Adversarial cross-modal hard-negative families.** Two new families:

   - **`hn_recovery_high_amount` (legitimate).** Text reads like classic ATO: "device change, password reset, large transfer to new recipient." Events reveal legitimacy: the new recipient is the account holder's other account; the new device is the holder's mid-upgrade phone; MFA was used. **Text alone misses it; events catch it.**
   - **`phish_takeover_mfa_phished` (fraud).** The dual. Text reads safe: "user logged in, transferred funds to known recipient." Events reveal the anomaly: `recipient_age=newly_added` (just added 20 min ago), subtle device anomaly, MFA token reuse pattern. **Text alone misses it; events catch it.**

4. **Per-arm text-field routing.** The dataset is stored in canonical form with explicit `narrative`, `events_text`, `structured_events`, wrapper tokens, and verdict footer fields. Each trainer constructs its arm-specific input — `text_only` sees wrapper + narrative + verdict; `structured_as_text` sees wrapper + events_text + narrative + verdict; `xattn` sees wrapper + narrative + verdict and consumes events only through the side encoder. **`text_only` and `xattn` now see byte-identical LM prompts.**

**v4 headline.** Re-running the v3 leader configuration on the v4 data (`metric_version: 5`, `n_clean = 5,002`; source: `runs/exp_text_only_v4_001/ci_report.json`, `runs/exp_xattn_v4_001/ci_report.json`):

| Arm | phish_takeover_mfa_phished recall | phish_takeover recall | hn_recovery_high_amount FPR |
|---|---|---|---|
| `text_only_v4` | 0.0000 [CI 0.000, 0.012] | 0.1122 [CI 0.072, 0.158] | 0.4175 [CI 0.342, 0.512] |
| `xattn_v4`     | **0.9718 [CI 0.931, 1.000]** | **1.0000 [CI 1.000, 1.000]** | 0.4505 [CI 0.369, 0.562] |

The two adversarial fraud families are CI-separated. Cross-attention catches them; text alone effectively does not (recall 0.00 vs 0.97 on the MFA-phished family; recall 0.11 vs 1.00 on the base phish-takeover family). The `hn_recovery_high_amount` family is the architectural ceiling: both arms sit at ~0.42–0.45 FPR with overlapping CIs, and cross-attention is +0.033 absolute *worse* on the point estimate — the adversarial-legitimate signal is hard for *both* modalities, foreshadowing the v5 ceiling that no tested architectural dial moves beyond CI noise in §4.3. Max-gate magnitude rose to 0.0221 on the v4 architecture-winner (above the 0.011 v3 ceiling but below the 0.05 "open" target) — a small, sparse, but evidently effective gate opening.

### 4.3 v5 — scaling and the data-shaped ceiling

![Figure 4. v5 sweep leaderboard with adversarial-error decomposition](figures/fig4-sweep-results.svg)

The v5 expansion ran 11 cross-attention configurations across two phases against the new `v5_adv_error` primary metric (§3.3) on the v4 dataset using the `qwen3-8b-cpt-light-v4-merged` base:

- **Phase 1 — training and arch-dial sweep.** Insertion pattern (every_4, every_8, late_only), gate_init (zero, small_0.01), resampler slots (32, 64, 128), LoRA-r (16, 32), LR (1e-4 default plus 3e-4 fast / 3e-5 slow perturbations), warmup (500 default).
- **Phase 2 — encoder sweep.** Three encoder variants on the Phase-1 winner config: `small_transformer`, `pooled_mlp`, `ft_transformer`. Stop rule: halt if neither alternative beats the Phase-1 winner by ≥0.005 absolute `v5_adv_error`.

Full results are in Figure 4A and tabulated in `04-cross-attention-experiments.md`. Three findings:

**Finding 1: the Phase-1 winner.** `exp_v5_p1_zero_64` (every_8 / slots=64 / gate=zero / `small_transformer` / lora_r=16) achieved `v5_adv_error = 0.1506 [CI 0.1238, 0.1871]`. The v4 seed `exp_xattn_v4_001` had `0.1596 [CI 0.1306, 0.1945]`. Point improvement: −0.0090 absolute. CIs overlap; no statistically robust separation. The point improvement comes entirely from a `phish_takeover_mfa_phished` miss-rate drop from 0.0282 to 0.0141 and an `hn_recovery_high_amount` FPR drop from 0.4505 to 0.4377 — both within bootstrap CI noise.

**Finding 2: encoder choice is neutral-to-negative.** Phase-2 ran two alternatives:

| Encoder | v5_adv_error [CI] | phish_takeover_mfa_phished miss | hn_recovery FPR |
|---|---|---|---|
| `small_transformer` (P1 winner) | 0.1506 [0.1238, 0.1871] | 0.0141 | 0.4377 |
| `pooled_mlp` | 0.1549 [0.1282, 0.1938] | 0.0141 | 0.4505 |
| `ft_transformer` | 0.1736 [0.1448, 0.2136] | 0.0704 | 0.4505 |

`pooled_mlp` ties within CI — a learned attention-over-events encoder confers no measurable advantage over mean/max-pooling + MLP on this task. `ft_transformer` actively regressed on `phish_takeover_mfa_phished` recall (0.9296 vs 0.9859). The Phase-2 stop rule fired after both alternatives.

**Finding 3: the data-shaped ceiling on `hn_recovery_high_amount`.** Within the 11-run v5 sweep on the 5k clean eval (`hn_recovery_high_amount` n=78, per-family CI width ~0.18), no architectural dial moved the bottleneck beyond CI noise. Across the non-pathological v5 x-attn runs, the FPR stayed in the band [0.4377, 0.4505] — a maximum spread of 0.0128, well below the run-to-run bootstrap CI width. The one excursion is `exp_v5_p1_fastlr` at 0.4249, which dropped below the band only because the 1%-legit-FPR operating point shifted after `phish_takeover_mfa_phished` recall collapsed to ~0 — not a real architectural movement of the ceiling. The component contributes ~97% of the total `v5_adv_error` for the Phase-1 winner (0.146 of 0.151; Figure 4B). The data generator's `hn_recovery_high_amount` template appears to produce examples that are indistinguishable from fraud at the signal level visible to the model. **Caveat on strength of claim:** the per-family CI is wide because only 78 rows land in this family in the 5k clean eval; a 50k LLM-narrated eval (already built at `data/eval_medium_50k_llm/`, not yet scored on the v5 winner) would tighten the per-family CI ~3× and is the recommended next test before declaring the ceiling structural. Either the generator needs richer contrastive features (device-trust history, step-up-auth chain, out-of-band verification), or the framing should change — the production analog of `hn_recovery_high_amount` is plausibly a "route to step-up auth" case rather than a binary fraud/legit classification target, and the model is being asked an unanswerable question.

---

## 5. Results

The synthetic, mid-eval-corrected, bootstrap-CI-defended numbers from §4 are reproduced in summary form here. The full per-run leaderboard is in Figure 4 and `04-cross-attention-experiments.md`. Reproduction artifacts (`experiments.jsonl`, `ci_report.json`, `gate_trajectory.json`) are committed in `src/auto_research/runs/`.

**v3 (metric_version: 2, n_clean = 4,466).** Cross-attention leader vs structured-as-text baseline on worst-family HN-FPR @ 1% legit FPR: 0.0524 [CI 0.0420, 0.0647] vs 0.0507 [CI 0.0408, 0.0635]. CIs overlap; **no win**.

**v4 (per-family fraud-recall + HN-FPR, n = 5,002).** On the two new adversarial cross-modal families:

- `phish_takeover_mfa_phished` fraud recall: text_only 0.0000 [CI 0.000, 0.012] vs xattn 0.9718 [CI 0.931, 1.000] — **CI-separated win**.
- `phish_takeover` fraud recall: text_only 0.1122 [CI 0.072, 0.158] vs xattn 1.0000 [CI 1.000, 1.000] — **CI-separated win**.
- `hn_recovery_high_amount` FPR: text_only 0.4175 [CI 0.342, 0.512] vs xattn 0.4505 [CI 0.369, 0.562] — **both poor (~0.42–0.45), CIs overlap, xattn +0.033 worse on point estimate**.

**v5 (metric_version: 5, n = 5,002, 11 cross-attention runs).** Phase-1 winner `v5_adv_error = 0.1506 [CI 0.1238, 0.1871]` vs v4 seed 0.1596 [CI 0.1306, 0.1945] — point improvement, CIs overlap. The win is dial-robust: of nine non-pathological runs (excluding `fastlr` regress and `slowlr`), `v5_adv_error` lies in the band [0.151, 0.174]. The bottleneck family `hn_recovery_high_amount` has FPR in [0.4377, 0.4505] across all non-pathological runs (fastlr drops to 0.4249 only as an operating-point artifact after fraud recall collapsed) and contributes 97% of the `v5_adv_error` for the winner. **All v4/v5 numbers above should be read with the §7 limitations in mind** — synthetic data only, single LM family/scale, gate magnitudes below the Flamingo "open" target, and per-family CI width ~0.18 on the 5k-eval bottleneck family.

**Gates story.** Max-gate magnitudes across the three generations:

- v3: 0.0106–0.0112 (small-init runs); 0.0039–0.0041 (zero-init runs). Gates rode their initialization.
- v4: 0.0221 (seed). Small but measurable opening; sufficient for the CI-separated win.
- v5: 0.0058–0.0221 across 11 runs. Lowest gate (0.0058, `fastlr` regress) corresponded to the worst result; the Phase-1 winner had 0.0128. Sparse-but-effective gating: the LM does most of the discrimination; cross-attention serves as a disambiguator on the adversarial families where text alone fails.

**Harness throughput.** Across all three generations:

- 30 cross-attention and baseline runs recorded in `experiments.jsonl`.
- 18 GPU-hour budget cap in v3 (used 7.735); 12 GPU-hour cap in v5 (used 7.92).
- Zero out-of-band intervention. Zero format-drift incidents. Two mid-POC metric corrections rolled forward via `scripts/rescore_baselines.py` without rerunning training. Convergence-halt premature-firing on v3 Day-2 caught by the halt-condition log; convergence-halt disabled for v4 and v5.

---

## 6. Discussion

**What the loop did, and what it could not do.** The agentic harness made the result reproducible, the CIs verifiable, and the metric correction roll-forward feasible. It did not substitute for research judgment about which question to ask. Two findings during the POC required a human to step in and reframe: the AUC saturation pivot at v3 Day-1 evening (headline metric switched from AUC-stripped to worst-family HN-FPR @ 1% legit FPR), and the sklearn-cliff metric correction at v3 Day-2 second-half (`metric_version: 1 → 2`). The agent caught the symptoms in both cases (AUC=1.0 logged in every row; `event_only` reporting an implausibly large advantage); the human caught the reframe. We argue this is the right division of labor for autonomous-research systems with hard guardrails: the harness automates the slow, repetitive, error-prone work (validation, dedup, lockfile, CI, atomic writes, leakage audit); the human handles the qualitative judgment about what to measure and why.

**What the v3 null result is good for.** The v3 result, in isolation, says "cross-attention does not separate from structured-as-text on this synthetic surface." That sentence is technically true. The honest sentence is "the synthetic data, as we built it, paraphrases the structured signal into the text — collapsing the modality gap that cross-attention is designed to exploit." Both sentences come from the same numbers. The first is what a less-disciplined process would have published; the second is what the code audit found. The agentic harness did not write the second sentence — a person did. But the harness made the diagnostic cheap: every cell's bootstrap CI was already on disk, the gate trajectories were already logged, and the inline leakage fields on each `experiments.jsonl` row (`clean_eval_mask_text_overlap`, `clean_eval_mask_events_overlap`, `clean_eval_dropped`) had already flagged the family-concentrated overlap. The diagnostic took an afternoon, not a week.

**What the v4 win is, and is not.** v4's `phish_takeover_mfa_phished` 0.0000 → 0.9718 recall jump is a CI-separated architectural win on a family that, by construction, requires the model to attend to the event stream. It is not evidence that cross-attention beats production fraud systems; it is evidence that the architecture works *when given a problem it can solve*. The right test for production transfer is a held-out replay of a real anonymized window of fraud-and-legit traffic — explicitly out of scope for this synthetic POC. We discuss this in §7.

**What the v5 data ceiling means.** `hn_recovery_high_amount` is the v4 adversarial-legitimate family: text reads exactly like fraud, events reveal legitimacy. Across the non-pathological v5 configurations, no architectural or training-dial change moves the FPR below 0.4377; the one excursion (`fastlr` at 0.4249) is an operating-point artifact after fraud recall collapsed, not a real movement of the ceiling. Three interpretations are consistent with the data:

1. The generator template is too aggressive — even the events do not carry enough disambiguating signal. The fix is data-side: enrich the template with device-trust history, step-up-auth chain, out-of-band verification signals. The model would then have something to learn.

2. The v5 metric is correctly exposing a production ambiguity — there are real high-amount account-recovery events for which a fraud classifier *cannot* be expected to achieve low FPR without additional context. The fix is to reframe the metric: this family becomes a "route to step-up auth" target, evaluated by step-up-routing precision rather than fraud/legit FPR.

3. Both. The synthetic generator's template is an approximation of a real production ambiguity, and the model's failure to disambiguate is a faithful reflection of the real failure mode.

We cannot distinguish (1), (2), (3) from the synthetic surface alone. The v5 final synthesis recommends data redesign as the next move; we agree, but caveat it with the framing-review note from (2): if the data redesign cannot get this family below FPR=0.15, the question itself is wrong, and the production analog is a routing problem, not a classification one.

**Reproducibility.** The full experimental record is in the repository: `src/auto_research/experiments.jsonl` (one row per run, append-only), per-run artifacts under `src/auto_research/runs/exp_*/`, the harness in `scripts/run_next_experiment.py` and `src/auto_research/AGENT_INSTRUCTIONS.md`, the evaluation pipeline in `eval/score_risk.py` and `eval/bootstrap_ci.py`. The corrected leaderboard (metric_version=2) is rederivable via `scripts/rescore_baselines.py --auto-detect`. The data overlap diagnostic is rederivable via `scripts/diagnose_data_overlap.py --check`. Pinned environment (CUDA 12.4, bitsandbytes ≥0.45) and the Blackwell-architecture preflight hard-fail are documented in `RUNBOOK.md`.

---

## 7. Limitations

**Synthetic data only.** Every result in this paper is computed on a synthetic ATO dataset. The narrator (`gpt-5-nano-2025-08-07`) is a foundation model approximating an analyst; the structured-event generator is a template-driven program approximating PayPal's production behavioral telemetry. The two cohorts (narrator and generator) are not adversarially trained against each other; they are coupled only by the journey/actor schema. Production transfer is not validated.

**Single LM family, single scale.** The base is Qwen3-8B post-CPT-light. We did not test other 7–10B-parameter LMs (Llama 3.1 8B, Mistral 7B) nor other scales (Qwen3-32B, Qwen3-3B). The architecture-vs-data conclusions are conditional on this base; a stronger or weaker base might shift the bottleneck.

**Gate magnitudes never reached the "open" target.** The Flamingo paper's reported max-gate magnitudes are O(0.1–1.0); ours sit in O(0.005–0.02) even on the v4 architectural win. The architecture works — the v4 CI-separated win is evidence of that — but the gates are operating in a regime not predicted by Flamingo's training dynamics. We do not have a clean explanation; the most likely reading is that the LM is already very good at the task on most examples, and the gate opens narrowly only where the side stream resolves a textual ambiguity. A controlled study with a deliberately under-capable base LM might force gates open further; we did not run it.

**`hn_recovery_high_amount` is one family.** The data ceiling discovered in v5 is on one adversarial-legitimate family. We do not know whether (i) the same ceiling exists on other adversarial-legitimate constructions, (ii) ceilings exist symmetrically on adversarial-fraud constructions we did not build, or (iii) the production analog suffers the same failure mode. All three are open.

**Compute budget.** v3 used 7.735 GPU-hours of 18 budgeted; v5 used 7.92 of 12. We did not exhaust either budget. Specifically, the v5 expansion did *not* run the stress-run option (steps=3000, seq_len=4096) because the Phase-2 stop rule fired first. A longer stress run on the Phase-1 winner config might have moved the `phish_takeover_mfa_phished` miss rate or, less plausibly, the `hn_recovery_high_amount` FPR; we do not have that evidence.

**Single-engineer POC.** The work was carried out by one engineer (the author) over a calendar span of roughly two weeks of wall-clock with extensive use of the agentic harness for overnight unattended runs. The methodology benefits and risks of agentic loops in larger team settings (coordination cost, shared-state contention, multi-agent variance) are not addressed.

**No non-LLM tabular baseline.** The architectural comparison in this paper is between `text_only` and `xattn` arms that consume byte-identical LM prompts modulo the side stream — a clean causal comparison for "does cross-attention add lift over the same LM reading the same text without the side stream?" It is *not* a comparison against the strongest non-LLM model available for this task. A gradient-boosted decision-tree model — XGBoost or LightGBM trained directly on the bucketed event features — was not built. **The absence of this baseline does not invalidate the cross-attention-vs-text-only causal comparison; it does prevent any claim of fraud-model competitiveness.** Before any reader interprets the v4 architectural win as "cross-attention beats fraud baselines," a tabular baseline trained on the same bucketed features should be run. We expect it to land in the same band as or above structured-as-text on the easy fraud families and to fail on `phish_takeover_mfa_phished` (text reads safe; tabular has no narrative signal) and on `hn_recovery_high_amount` (the structural-resemblance problem is upstream of all three model families). Until that experiment runs, the paper claims an architectural finding within the LM family, not a competitive-modeling finding.

---

## 8. Conclusion

We presented a three-generation cross-attention POC for synthetic ATO detection driven by an agentic experiment harness with hard-edge guardrails. The harness — agent proposes, deterministic launcher enforces, single-writer-per-file ownership, bootstrap CIs on every metric, tie-aware exact-target operating points — ran 30 experiments end-to-end with zero out-of-band intervention and made two mid-POC metric corrections roll forward cleanly. The cross-attention finding evolved across the three generations: a null result in v3 (later traced to a synthetic-data pipeline that paraphrased the structured signal into the text), a confidence-interval-separated architectural win in v4 (after the data pivot restored a real modality gap), and a data-shaped ceiling in v5 on one adversarial-legitimate family that no architectural dial moved beyond CI noise within the 11-run sweep. The methodological contribution — harness + leakage-safe synthetic data + bootstrap-CI eval — is, we argue, more durable than the architectural result. The cross-attention finding is the worked example; the loop is the reusable artifact. Next steps, in priority order: (1) data redesign targeting `hn_recovery_high_amount` to test whether the v5 ceiling is structural to the template or genuine, (2) production-replay validation on a held-out anonymized window (§8.1 roadmap below), (3) packaging the harness as a Foundation Science capability so subsequent POCs inherit the corrections — Blackwell + bitsandbytes preflight, leakage-controlled data pipeline, tie-aware metric, atomic-write history — for free.

### 8.1 Real-data validation roadmap

Production transfer is **not claimed** by this paper. The 5k LLM-narrated and 50k templated eval surfaces are both synthetic, and the bottleneck family `hn_recovery_high_amount` is constructed by template, not observed in real traffic. To move any claim in this paper from "Strong within synthetic eval" to "Strong on production data," the following six steps are the concrete next experiment:

1. **Anonymized held-out window.** A real fraud-and-legit traffic slice of 10k–50k sessions, scoped through PayPal's data-engineering and privacy-clearance process. Schema-mapped to the synthetic journey/actor schema so the v5 Phase-1 winner can run inference without re-training; mismatch rows are dropped or flagged.
2. **Temporal split, not random.** Train cutoff and eval cutoff must respect production label-delay realities. Label-delay handling: include only sessions whose fraud label has matured (typically 30–90 days post-session); rolling-window eval to surface temporal drift.
3. **Calibration metrics.** Add Expected Calibration Error (ECE), Brier score, and a reliability diagram alongside the existing AUC + R@FPR + per-family decomposition. The current eval is discrimination-only; for a production-risk-scoring use case, calibration matters as much as discrimination.
4. **Comparison against current production risk features.** Whatever the baseline-of-record is for ATO routing on the chosen production window — feature-store-driven rule, gradient-boosted model, or hybrid — should be the floor the LM-plus-cross-attention arm is compared against. The right reportable number is **lift over the baseline at fixed legit-FPR**, not raw recall.
5. **Routing-precision evaluation for the `hn_recovery_high_amount` analog.** The synthetic family was constructed as an adversarial-legitimate case ("text reads like fraud, events reveal legitimacy"). The production analog is the case where step-up auth is the right action, not "fraud / not-fraud." The metric for this slice changes from FPR to step-up-routing precision. This is a framing change, not a model change.
6. **Statistical-significance protocol.** Pre-register: which CI to compute (bootstrap, n=1000 resamples, same machinery as the synthetic eval); what counts as a win (CI-separated lift on the headline metric AND no regression on any per-family); what counts as a regression (point-estimate drop > 0.005 absolute OR any CI-separated decrease).

The harness is designed to run this validation with minor adapter work — a new trainer entry-point that loads the production schema, a new eval-mode for the calibration metrics, and a new dedup-tuple field for temporal split. Estimated effort: 1–2 weeks of data-engineering + privacy-clearance work to scope the anonymized replay; less than one day of model inference once the data is in place.

What this paper explicitly **does not** commit to: a specific deliverable date, a specific production system to compare against, or a claim that the synthetic-surface ceiling on `hn_recovery_high_amount` will transfer to real traffic. Those are downstream decisions for the team that owns the production-replay scoping.

---

## References

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., Ring, R., Rutherford, E., Cabi, S., Han, T., Gong, Z., Samangouei, S., Monteiro, M., Menick, J., Borgeaud, S., Brock, A., Nematzadeh, A., Sharifzadeh, S., Binkowski, M., Barreira, R., Vinyals, O., Zisserman, A., & Simonyan, K. (2022). Flamingo: a Visual Language Model for Few-Shot Learning. *NeurIPS 2022*.

Caton, S., & Haas, C. (2024). Fairness in Machine Learning: A Survey. *ACM Computing Surveys, 56*(7).

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach. *Biometrics, 44*(3), 837–845.

Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. *NeurIPS 2021*.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.

Jaegle, A., Borgeaud, S., Alayrac, J.-B., Doersch, C., Ionescu, C., Ding, D., Koppula, S., Zoran, D., Brock, A., Shelhamer, E., Hénaff, O., Botvinick, M., Zisserman, A., Vinyals, O., & Carreira, J. (2021). Perceiver IO: A General Architecture for Structured Inputs & Outputs. *arXiv:2107.14795*.

Karpathy, A. (2017–2024). Public talks and online writings on the "agent proposes, deterministic script enforces" pattern for LLM-driven research loops, including the "Software 2.0" essay and the "Let's reproduce GPT-2" series. Cited as personal communication / public talks; no single canonical reference.

Krause, J., Stark, M., Deng, J., & Fei-Fei, L. (2020). 3D Object Representations for Fine-Grained Categorization. *4th IEEE Workshop on 3D Representation and Recognition (3dRR-13), at ICCV-13.*

Schwartz, R., Vassilev, A., Greene, K., Perine, L., Burt, A., & Hall, P. (2022). Towards a Standard for Identifying and Managing Bias in Artificial Intelligence. *NIST Special Publication 1270*.

Internal references (PayPal Foundation Science, this repository):

- `cross_attn_ato_poc/PLAN.md` — v3 plan (the POC scaffold).
- `cross_attn_ato_poc/README.md` — three-generation journey log, current at 2026-05-21.
- `cross_attn_ato_poc/docs/{day-1-results,day-2-results,experiments-log,auto-research-loop,cross-attention-mechanism}.md` — durable per-day records.
- `cross_attn_ato_poc/.claude/tasks/{data-v4-pivot-plan,data-v4-verdict,xattn-expanded-sweep-plan,agent-native-journey-families-plan}.md` — v4/v5 planning artifacts.
- `cross_attn_ato_poc/whitepaper/{01,02,03,04}-*.md` — companion whitepaper deep-dives.

---

## Appendix A — companion documents and figures

This whitepaper is the master narrative. Four companion documents are designed to stand alone for readers who want a single-pillar deep dive:

- **`01-data-curation-and-distribution.md`** — synthetic-data generator design (three token families, journey/actor schema, bucketed features, narrator policy with banned-phrase scan), leakage controls (text-hash and structured-events-hash dedup, pre-narration stratification), eval-mode dropout, the v3→v4 four-change pivot rationale, and the v4 generator's data distribution audited against the schema.
- **`02-agentic-experiment-harness.md`** — full launcher walkthrough (whitelist → dedup → lock → launch → parse → CI → atomic → append → refresh), ownership invariants, halt-condition design and the v3 convergence-halt postmortem, dedup tuple, expanded-sweep directive (v5's early-exit-on-success rule), and cron + agent-tick orchestration.
- **`03-eval-strategy.md`** — three eval modes (stripped, opaque, full), three eval-set sizes (5k / 15k / 50k), metric definitions for `metric_version` 1, 2, and 5, the tie-aware exact-target operating-point computation with worked example, bootstrap CI derivation, the v3 sklearn-cliff finding, narrative-leakage scan and clean-eval mask.
- **`04-cross-attention-experiments.md`** — architecture detail (Qwen3-8B + side encoder + Perceiver-Resampler + gated cross-attention + LoRA-on-Q), the alternative architectures considered, training recipe, the full v3+v4+v5 leaderboard with bootstrap CIs and gate magnitudes, ablation reads per dial, the gates story across three generations, and the data-ceiling diagnostic.

Figures:

- **Figure 1** (`figures/fig1-architecture.svg`) — cross-attention surgery on Qwen3-8B.
- **Figure 2** (`figures/fig2-auto-research-loop.svg`) — auto-research loop dataflow.
- **Figure 3** (`figures/fig3-data-distribution.svg`) — class balance, journey × actor distribution, eval-mode dropout, three token families.
- **Figure 4** (`figures/fig4-sweep-results.svg`) — v5 sweep leaderboard with v5_adv_error decomposition.

All four figures are SVG; all four companion documents are CommonMark Markdown. They are designed to be readable both inline in the repository and as components of an external arXiv-style submission.
