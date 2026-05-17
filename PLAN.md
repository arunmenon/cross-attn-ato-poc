# Plan: Cross-Attention for PayPal ATO — 3-Day Journey POC (v3, post-review-2)

## Context

The prior research study (5-pair adversarial dialogue) produced a research brief, PoC spec, and next-steps checklist. The user now wants a **3-day hands-on POC on a single RunPod H100**, where **the journey matters more than the result**: prove the pipeline bones, surface integration friction, get trustworthy directional signal, decide whether to extend.

**v3 changelog vs v2** (post second review). v2 fixed eval-leakage, low-FPR noise, objective/metric mismatch, Day-1 overreach, full-CPT memory risk, missing apples-to-apples baseline, brittle orchestration, RunPod persistence. v3 closes the remaining false-signal risks the second review flagged:

1. **Cheap-large eval** now uses **templated narrative + verdict footer**, not verdict-only — keeps eval distribution close to training.
2. **Training-time token dropout** (random eval-mode application during training) — model cannot over-rely on `<journey_*>` / `<actor_*>` tokens.
3. **Narrative leakage policy** — narrator prompt bans class names; `leakage_checks.py` detects banned phrases; post-generation scrub.
4. **Bucketed derived feature tokens** — fence raw PII (`<acct_id>`, `<email>`, `<device_id>`, `<phone>`, `<ip>`) but preserve fraud signal via `<amount_bucket=high>`, `<geo_distance=international>`, `<ip_risk=high>`, etc.
5. **Event-only classifier baseline** — 4th baseline (`small_transformer` + classification head, no LM). If it wins, x-attn becomes an explanation/grounding tool, not a classifier.
6. **Eval ladder** — 5k every experiment → 50k top-3 + baselines → 100-200k only top-1 vs structured-as-text if time.
7. **Compact structured-as-text serialization** — defined explicitly.
8. **Stage-0 LoRA merged into base** before Stage-1; fresh Stage-1 LoRA-on-Q. No adapter confusion.
9. **Minor**: git checkpoint before agent code edits; config whitelist in `run_next_experiment.py`; atomic checkpoint writes; bootstrap CI on AUC comparisons; convergence halt only after ≥6 valid x-attn runs.

The Karpathy-style auto-research loop is *agent proposes, deterministic script enforces*. The auto-loop **plumbing** is itself a deliverable that carries to Day 4+ unchanged.

**What we are trading away to fit 3 days** (honest list):

- 20-30k LLM-narrated pairs. Plus 50-200k cheap **templated-narrative** examples for low-FPR eval.
- Side-stream encoder: `small_transformer` only. FT-Transformer + CNN+LSTM deferred to Day 4+.
- Resampler slots: {64, 128}. {32, 256} deferred.
- Insertion patterns: `every_4`, `every_8`, `late_only`. `every_2` deferred.
- Gate init: {zero, small_0.01}. `learnable` deferred.
- Stage-0 CPT: **embedding + LoRA, then merge**.
- Sequence length: **2048**. 4096 for one final stress run.
- Sweep budget: **10-12 clean experiments**.
- Stage-2 SFT: cut.

Existing context to honor:

- The user has an HF-Accelerate CPT/DAPT training harness on Qwen models in a separate account (not reachable). We mirror the *concept* (token-fencing for PII + journey-type structure), not the code.
- Custom tokens serve **three** purposes now: (a) PII hygiene (raw-identifier fencing); (b) **bucketed derived-feature** signal preservation; (c) marking ATO journey types for structural attention.
- Threat surface includes **agentic actors** (AI agents acting on behalf of users — buying assistants, finance assistants, compromised agents, adversarial agents).
- **Hard negatives** matter: legitimate behaviors that look like fraud.

---

## Goals & non-goals

### Goals

1. End-to-end synthetic ATO pipeline rich enough to exercise multi-journey, multi-actor, hard-negative-aware cross-attention training, with **bucketed features preserving fraud signal**.
2. Flamingo-style gated cross-attention on a frozen-after-CPT-merge Qwen3-8B with a fixed side-stream encoder.
3. **Four apples-to-apples baselines**: CPT-light, LoRA-text, structured-as-text concat, **event-only classifier**.
4. **Leakage-safe evaluation**: three modes (stripped / opaque / full), three eval-set sizes (5k / 50k / 100-200k), narrative-leakage detector, AUC bootstrap CIs.
5. Karpathy-style auto-research loop with deterministic guardrails: agent proposes, script enforces (lockfile, validate, whitelist, dedup, parse, atomic write).
6. Honest journey log: daily writeups covering decisions, failures, integration friction, root causes.
7. End with concrete next-steps for a real PayPal-internal POC.

### Non-goals

- Beat any production fraud system.
- Generalize beyond the synthetic distribution.
- Build a serving stack.
- Replicate the user's exact existing CPT codebase.
- Decide whether cross-attention is the right architecture for production.

---

## Synthetic data design

### Three token families

| Family | Purpose | Visible at fraud-classification eval? |
|---|---|---|
| **PII-fencing tokens** | Hygiene — replace raw identifiers | Yes (no info leak) |
| **Bucketed feature tokens** | Preserve fraud signal in privacy-safe form | Yes (this is the signal) |
| **Journey / actor structural tokens** | Mark journey type & actor type | **Stripped or opacified** depending on eval mode |

### Journey-type structural tokens (stripped at eval)

| Token pair | Description |
|---|---|
| `<journey_clean>` … `</journey_clean>` | Normal user behavior |
| `<journey_cred_stuff>` … `</journey_cred_stuff>` | High-velocity logins from rotating IPs |
| `<journey_sim_swap>` … `</journey_sim_swap>` | Device change → password reset → large txn |
| `<journey_phish_takeover>` … `</journey_phish_takeover>` | Phished creds → quick monetization |
| `<journey_malware_rat>` … `</journey_malware_rat>` | Legit device, anomalous behavior |
| `<journey_mule_chain>` … `</journey_mule_chain>` | Receives then forwards funds rapidly |
| `<journey_hn_travel>` … `</journey_hn_travel>` | **Hard negative**: legitimate travel |
| `<journey_hn_large_purchase>` … `</journey_hn_large_purchase>` | **Hard negative**: legitimate large purchase |
| `<journey_hn_account_recovery>` … `</journey_hn_account_recovery>` | **Hard negative**: legitimate password reset |

### Actor-type structural tokens (stripped at eval, same three-mode regime)

| Token | Description |
|---|---|
| `<actor_human>` | Direct human user |
| `<actor_agent_buying>` | Legitimate AI shopping/buying assistant |
| `<actor_agent_finance>` | Legitimate financial assistant |
| `<actor_agent_compromised>` | AI agent compromised by attacker |
| `<actor_agent_adversarial>` | Malicious AI agent attempting ATO |
| `<actor_hybrid>` | Human + agent collaboration |

Agent-actor classes get distinct event-timing distributions (API-like cadence, programmatic step patterns, tool-use traces).

### PII-fencing tokens (raw identifiers — fence completely, no signal leak)

`<acct_id>`, `<email>`, `<phone>`, `<device_id>`, `<ip>`, `<recipient>`, `<merchant>`, `<browser>`.

These appear as opaque placeholders. They are **not** the fraud signal; they are hygiene.

### Bucketed derived-feature tokens (v3 — the fraud signal lives here)

These are the safe, derived, privacy-respecting features that carry actual fraud-detection signal. Both text-side and structured-side training paths see them.

| Family | Buckets |
|---|---|
| `<amount_bucket=…>` | `low` (<$50), `medium` ($50-500), `high` ($500-5k), `extreme` (>$5k) |
| `<geo_distance=…>` | `local` (<50km from baseline), `domestic_far`, `international` |
| `<ip_risk=…>` | `low`, `medium`, `high` (VPN/Tor/datacenter ASN) |
| `<device_age=…>` | `known` (>30d), `new` (<7d), `rare` (seen <3 times) |
| `<merchant_risk=…>` | `normal`, `elevated` |
| `<txn_velocity=…>` | `normal`, `bursty` (>N in 1h), `extreme` (>N in 5min) |
| `<recipient_age=…>` | `known` (>30d in account graph), `newly_added` (<24h) |
| `<session_dwell=…>` | `short`, `normal`, `extended` |
| `<auth_strength=…>` | `mfa_strong`, `password_only`, `cookie_only` |

These tokens are **always visible** to all baselines (text-side) and to the structured-side encoder. They are the signal. The journey/actor tokens are *labels for the signal pattern* — they're the thing we strip at eval.

### Event tokens (structural inside a journey)

`<event_login>`, `<event_txn>`, `<event_pw_reset>`, `<event_device_add>`, `<event_recipient_add>`, `<event_chat_to_support>`, `<event_tool_call>`. Always visible.

### Verdict footer (carried from v2)

Every training example ends with:

```text
…narrative body…

<risk_verdict>
label: fraud
journey_family: sim_swap
confidence: high
evidence: device_change, pw_reset, large_txn, new_recipient
</risk_verdict>
```

`label ∈ {fraud, legit}`. `journey_family ∈ {clean, cred_stuff, sim_swap, phish_takeover, malware_rat, mule_chain, hn_travel, hn_large_purchase, hn_account_recovery}`. Primary classification score: `logP(' fraud' | prefix) - logP(' legit' | prefix)`.

### Narrative leakage policy (v3 — closes the LLM-narrator answer-leakage hole)

The LLM narrator (Claude/GPT-4-class) generates the narrative body. Its prompt **bans** the following phrases (case-insensitive, including stems):

- `fraud`, `fraudulent`, `fraudster`
- `legit`, `legitimate`, `genuine` *(except `genuine identifier`)*
- `account takeover`, `ATO`
- `hard negative`, `hard-negative`
- Journey-family names as such: `SIM swap`, `phishing`, `mule`, `malware`, `credential stuffing`, `RAT`, `takeover` — **unless** they are operational evidence the narrator can plausibly cite in a real analyst log (e.g., "device-change event observed" is fine; "this is a SIM-swap" is not).
- `legitimate travel`, `legitimate purchase`, `legitimate recovery` — banned outright.

The narrator prompt has a worked example of compliant vs non-compliant narrative.

**Post-generation enforcement**: `eval/leakage_checks.py` includes a `narrative_leakage_scan(text)` function that runs a regex over the banned phrase list and flags any narrative that breaks the policy. Day-1 vertical-slice block runs this scan on all 1-2k narratives; non-compliant ones are regenerated or hard-removed. The full 20-30k generation pass uses the same scan as a quality gate.

### Paired streams

Each training example is a pair:

- **Structured stream** (consumed by side-stream encoder → x-attn K/V): JSON-event list with timestamps, event type, **bucketed derived features**, fenced PII. Length 5-200 events.
- **Text stream** (consumed by Qwen3 self-attn, with bucketed-feature tokens + journey/actor structural tokens during training): 100-300 word analyst-style narrative + verdict footer. PII fenced. Generated by a strong LLM (Claude/GPT-4-class, single pass, cached by structured-stream hash).

Volume targets:

- **LLM-narrated pairs**: 20-30k (training set)
- **Cheap templated-narrative pairs**: 50-200k (for low-FPR eval — see eval section)
- **5k stratified held-out eval** carved from the LLM-narrated pool

Class balance: ~30% fraud, ~30% hard negatives, ~40% clean.

### Generation pipeline

- `data/gen/journey_templates.py`
- `data/gen/agent_actor_mixer.py`
- `data/gen/feature_bucketer.py` *(v3 — new)* — derives bucketed feature tokens from raw event values
- `data/gen/pii_fencer.py`
- `data/gen/narrative_generator.py` *(v3 — narrator prompt enforces narrative-leakage policy)*
- `data/gen/cheap_template_generator.py` *(v3 — produces templated NARRATIVE + verdict footer, not verdict-only; see Patch 1)*
- `data/gen/build_dataset.py`
- `data/cards/dataset_card.md` — records journey/actor distribution, balance, template families, **field visibility per baseline**, **leakage-scan summary**

Generation cost rough: 25k narratives × ~$0.005/narrative ≈ $125. Cheap templated-narrative pairs are free.

---

## Architecture (cross-attn surgery on Qwen3-8B)

- **Base**: Qwen3-8B (36 layers, hidden 4096, 32 attn heads, 8 KV heads).
- **Frozen**: after Stage-0 CPT-light is merged into the base (v3 change), the resulting `qwen3-8b-cpt-light-merged` is frozen for Stage-1.
- **Cross-attn layers**: gated cross-attention dense blocks, per sweep dial (`every_4` / `every_8` / `late_only`).
- **Resampler**: Perceiver-Resampler with sinusoidal-on-Δt time encoding. Slots ∈ {64, 128}.
- **x-attn variant**: plain multi-head attention (not MLA). K/V cache-once per session.
- **Side-stream encoder**: `small_transformer` (6-layer transformer over event tokens). FT-Tx + CNN+LSTM deferred to Day 4+.
- **Gate init**: tanh(α) with α ∈ {0, 0.01}.
- **LoRA on Q (Stage-1, fresh)**: r=16 on self-attention query projection. Distinct from Stage-0 LoRA (which is merged into base before Stage-1 begins).
- **Trainable params (Stage-1)**: ~200-400M.

### File layout

Plan / planning artifacts:
```
.claude/tasks/cross-attn-ato-poc/
  PLAN.md                                    # this file (planning artifact only)
```

Code, configs, runtime state — at the project root, mirrors the `sft_autoresearch/` convention:
```
cross_attn_ato_poc/
  RUNBOOK.md
  README.md                                  # journey log
  data/
    gen/
      journey_templates.py
      agent_actor_mixer.py
      feature_bucketer.py                    # v3 NEW
      pii_fencer.py
      narrative_generator.py                 # v3 — enforces leakage policy
      cheap_template_generator.py            # v3 — templated NARRATIVE + verdict
      build_dataset.py
    cards/
      dataset_card.md
  src/
    model/
      cross_attn_block.py
      resampler.py
      qwen_xattn_wrapper.py
      encoders/
        small_transformer.py
        ft_transformer.py                    # placeholder, Day 4+
        cnn_lstm.py                          # placeholder, Day 4+
    tokenizer/
      custom_tokens.py
      fencer.py
    train/
      train_cpt_light.py                     # Stage-0 (LoRA, later merged)
      train_xattn.py                         # main x-attn
      train_lora_text_only.py                # baseline
      train_structured_as_text.py            # baseline
      train_event_only_classifier.py         # v3 NEW baseline
      mixers/
        eval_mode_dropout.py                 # v3 NEW — 50/25/25 mix
      accelerate_configs/
        single_h100.yaml
  scripts/
    preflight_check.py
    run_next_experiment.py                   # config whitelist, lockfile, atomic
    parse_metrics.py
    merge_stage0_lora.py                     # v3 NEW
    backup_to_external.sh
  eval/
    score_risk.py                            # AUC + R@FPR
    leakage_checks.py                        # incl. narrative-leakage scan (v3)
    eval_modes.py                            # strip/opaque/full
    bootstrap_ci.py                          # v3 NEW
  src/auto_research/
    AGENT_INSTRUCTIONS.md
    experiments.jsonl
    sweep_state.yaml
    experiment_template.yaml
    configs/
      sweep_space.yaml
      budget.yaml
    runs/
      exp_NNN/
        config.yaml
        train.log
        metrics.json
        gate_trajectory.json
        leakage_report.json
        ci_report.json                       # v3 NEW
```

---

## Training pipeline

- **Framework**: HuggingFace Accelerate.
- **Precision**: bf16.
- **Optimizer**: paged_adamw_8bit.
- **LR schedule**: cosine + 500-step warmup; peak LR 1e-4 (x-attn), 5e-5 (CPT-light).
- **Batch**: micro-batch 4, grad-accum to effective 32.
- **Seq length**: 2048 default. 4096 for one final stress run only.
- **Loss**: next-token CE on text stream (narrative + verdict footer).

### Eval-mode dropout during training (v3 — Patch 2)

Every training batch applies eval-mode dropout per example:

| Mode | Probability | Behavior |
|---|---|---|
| `full` | 0.50 | All tokens visible (journey, actor, bucketed-feature, PII-fenced) |
| `opaque` | 0.25 | Journey/actor tokens replaced with neutral IDs |
| `stripped` | 0.25 | Journey/actor tokens removed entirely |

This ensures the model sees all three eval distributions during training. **Without this**, stripped-mode eval would measure OOD behavior. Implemented in `src/train/mixers/eval_mode_dropout.py`, applied in dataloader collate.

### Stages and adapter lifecycle (v3 — Patch 8 explicit)

- **Stage 0 — Compressed CPT.** Train Qwen3-8B with frozen base + new-token embeddings + LoRA on attention+MLP. ~1-2 epochs over the LLM-narrated text-side narratives, ~3-5 hours H100, seq 2048. Eval-mode dropout already active.
  - **Adapter lifecycle**: After Stage 0 completes, run `scripts/merge_stage0_lora.py` to merge the LoRA weights *into the base*. Output: `qwen3-8b-cpt-light-merged` — a single set of weights, no adapter attached. This is the head-to-head baseline #1 and the starting point for Stage 1.
  - **Fallback ladder if Stage 0 underperforms** (decided Day 1, recorded in journey log):
    1. embedding + LoRA-merged (default)
    2. embedding + last-4-layers FT
    3. full FT only if memory smoke test passes with ≥10 GB headroom
- **Stage 1 — Cross-attention adaptation.** Start from `qwen3-8b-cpt-light-merged` (frozen). Add: side-stream encoder + Perceiver-Resampler + gated x-attn layers + **fresh** LoRA-on-Q r=16 (no relation to Stage-0 LoRA). Train jointly on paired data. ~1500-3000 steps per experiment.
- **Stage 2 SFT — cut.**

### Two distinct x-attn components (clarification carried)

1. **Side-stream encoder** — converts structured event stream (with bucketed-feature tokens) to embeddings. Trained from scratch jointly with x-attn layers.
2. **Gated cross-attention layers + Perceiver-Resampler** — inserted into Qwen3 stack. Resampler compresses encoder output to fixed K/V budget; x-attn layers let LM attend to those slots.

---

## Baselines (four — the head-to-head)

All baselines trained once at Day-1/Day-2 boundary. Cross-attn sweep arms compete against all four.

1. **CPT-light** = `qwen3-8b-cpt-light-merged` (Stage-0 output post-merge). Text-only, no structured-side, no x-attn.
2. **LoRA-text-only** (`train_lora_text_only.py`) — LoRA r=16 on Qwen3-8B, narrative + verdict-footer data. Tells us whether CPT is pulling weight vs LoRA alone.
3. **Structured-as-text concat** (`train_structured_as_text.py`) — **apples-to-apples baseline**. Compact serialization (defined below) of the structured event stream prepended to the narrative, train CPT-light on the concatenated input. No cross-attention.
4. **Event-only classifier** *(v3 — Patch 5)* — `train_event_only_classifier.py`. Take the same `small_transformer` side encoder + a fraud/legit classification head, train on structured events with binary CE. **No LM at all.** If this wins, the finding becomes "structured stream carries the synthetic ATO signal alone; x-attn earns its keep as an explanation/grounding tool, not as a classifier" — that's a real result.

### Compact structured-as-text serialization (v3 — Patch 7)

One line per event, key fields only. Token budget ≤ 800 (leaves >1200 for narrative + verdict at seq 2048).

```text
<events>
t=0  login          actor=<actor_*>  geo_distance=local      ip_risk=low     auth_strength=password_only
t=2  device_add     actor=<actor_*>  device_age=new
t=4  pw_reset       actor=<actor_*>  auth_strength=password_only
t=7  txn            actor=<actor_*>  amount_bucket=high      merchant_risk=normal   recipient_age=newly_added
t=9  txn            actor=<actor_*>  amount_bucket=high      txn_velocity=bursty
</events>
<narrative>
…
</narrative>
<risk_verdict>
label: fraud
…
</risk_verdict>
```

If an event stream exceeds 800 tokens, events are truncated to the **last** 30 events (the most recent is most informative for ATO). Truncation flagged in metadata.

A verbose-format variant is documented but not implemented in v3.

The no-conditioning floor baseline from v1 is removed.

---

## Evaluation (v3 — leakage-safe + distribution-matched + statistically defensible)

### Three eval sets (ladder)

| Set | Size | Source | Used for |
|---|---|---|---|
| **Fast** | 5k | Stratified slice of LLM-narrated pool | AUC, loss, R@FPR=1% — **every experiment** |
| **Medium** | 50k | Templated narrative + verdict footer | R@FPR=0.1% — **top-3 + 4 baselines** at Day 3 |
| **Large** | 100-200k | Templated narrative + verdict footer | Only **top-1 vs structured-as-text**, only if Day 3 time remains |

**Patch 1 fix**: medium and large eval sets use **templated narratives**, not verdict-footer-only. Format example:

```text
<events>…compact serialization…</events>

<session_summary>
The account shows a login sequence followed by a password reset and a high-value transaction
to a newly added recipient. The session includes a device change and a compressed event cadence.
</session_summary>

<risk_verdict>
label:
```

The session_summary is generated by a deterministic template (parameterized per journey + actor type). Cheap (no LLM calls) but distribution-shape matches training.

### Three eval modes (carried from v2)

| Mode | What's stripped | Use |
|---|---|---|
| **`stripped`** (headline) | All `<journey_*>` and `<actor_*>` tokens removed | Primary headline across all experiments |
| **`opaque`** | Journey/actor tokens replaced with neutral IDs | Secondary — structure-without-name signal |
| **`full`** | All tokens visible | Debug only. **Never reported as a win condition.** |

`eval/leakage_checks.py` audits each eval set: (a) confirms `stripped` mode actually removed every journey/actor token (including synonyms, numerals, partials); (b) runs the `narrative_leakage_scan` from the generation policy to flag any banned-phrase leak that slipped through.

### Primary scoring

`score = logP(' fraud' | prefix_up_to_label) - logP(' legit' | prefix_up_to_label)`.

Prefix is the narrative + `<risk_verdict>\nlabel:` portion. AUC over this score against ground-truth label.

### Bootstrap confidence intervals (v3 — Minor 4)

`eval/bootstrap_ci.py`: for any reported metric on the medium or large eval set, compute 95% bootstrap CI (1000 resamples). **No Day-4 decision based on Δ<0.005 AUC without non-overlapping CIs.** Per-experiment CI summary written to `runs/exp_NNN/ci_report.json`.

**Tie-aware exact-target HN-FPR (post Day-2 baseline correction; `metric_version: 2`).** The headline "HN-FPR @ 1% FPR" metric is the tie-aware worst per-family hard-negative FPR computed at the threshold where the achieved legit-FPR equals the target exactly, not the sklearn "largest achievable FPR ≤ target" cliff. Concretely: walk legit-score-descending until the cumulative count first reaches `need = target_fpr * n_legit` (kept as a float, not rounded — review 019 Blocker 1); set `T` to the score at that boundary; let `n_above = count(legit > T)`, `n_tied = count(legit == T)`, `alpha = (need - n_above) / n_tied`. The per-family HN-FPR weights tied-at-threshold rows by `alpha` so the operating point is exact regardless of score-distribution cliffs. Bootstrap CIs recompute `(threshold, alpha)` per resample and report `tie_fraction`, `achieved_fpr`, `threshold`, `alpha` alongside the point estimate, so every CI bound is verifiable from the JSON. The metric is computed against a leakage-filtered eval (`compute_clean_eval_mask` drops eval rows whose `text_hash` or `structured_events_hash` appears in train); rows are excluded from the comparison surface, not from on-disk predictions, so v1 rows remain auditable. `update_sweep_state` filters `current_best`/`top_3` to `metric_version >= 2`. See `docs/day-2-results.md` for the corrected leaderboard and the three findings (leakage, sklearn cliff, label-deterministic synthetic data) that motivated this revision.

### Reported metrics per experiment

- AUC (stripped) — headline.
- AUC (opaque) — secondary.
- AUC (full) — debug.
- R@FPR=1% on 5k fast.
- Per-journey-type AUC breakdown (stripped).
- Hard-negative FPR (the three `hn_*`).
- Agent-actor vs human-actor differential AUC.
- Train loss curves + gate-activation magnitude trajectory.
- **Top-3 + baselines (Day 3)**: R@FPR=0.1% on 50k medium eval, CI bootstrapped, leakage report attached.
- **Top-1 vs structured-as-text only**: 100-200k large eval, R@FPR=0.1% with tight CIs (if Day 3 time allows).

---

## Auto-research loop (Karpathy-style — agent proposes, deterministic script enforces)

Split as in v2, with v3 hardening:

| Layer | Responsibility | Who |
|---|---|---|
| Propose | Read state + history, output next `config.yaml` + rationale | Claude Code / Codex agent |
| **Validate (config whitelist)** | Reject unknown keys, no shell injection, sanity ranges | `run_next_experiment.py` *(Minor 2)* |
| Lock | GPU lockfile | `run_next_experiment.py` |
| Launch | `accelerate launch …` | `run_next_experiment.py` |
| Parse | Stream stdout + W&B → `metrics.json`, `gate_trajectory.json` | `parse_metrics.py` |
| **Atomic write** | All checkpoints written to temp path → rename | `run_next_experiment.py` *(Minor 3)* |
| Append | One line to `experiments.jsonl`, update `sweep_state.yaml` | `run_next_experiment.py` |
| **Bootstrap CI** | Compute CI on every reported metric | `bootstrap_ci.py` |
| Backup | Sync top-3 checkpoints + jsonl to S3/R2/HF | `backup_to_external.sh` (cron) |
| Summarize | One-paragraph natural-language summary | Agent |
| **Git checkpoint** | `git commit` before any agent code edit | Agent, per `AGENT_INSTRUCTIONS.md` *(Minor 1)* |
| Decide next | Choose next config or propose halt | Agent |
| Daily writeup | Day 2 and Day 3 README sections | Agent |

`AGENT_INSTRUCTIONS.md` explicitly instructs the agent to never edit code without first running `git add -A && git commit -m "snapshot before <change>"`.

### Sweep space (carried from v2)

```yaml
# config/sweep_space.yaml
insertion_pattern: [every_4, every_8, late_only]      # 3
gate_init:         [zero, small_0.01]                  # 2
resampler_slots:   [64, 128]                           # 2
encoder:           [small_transformer]                 # 1
# Fixed: synth_data_ratio=0.5
# Full grid: 3 × 2 × 2 = 12 cells; budget = 10-12 experiments → near-exhaustive
```

### Proposer heuristic (encoded in `AGENT_INSTRUCTIONS.md`)

- **First 6-8** (Day 2 PM): cover `insertion_pattern × resampler_slots` (6 cells) at `gate_init=small_0.01`.
- **Middle 2-4** (Day 3 AM): perturb top-2 along `gate_init`; longer-training stress.
- **Final 1-2** (Day 3 midday): seq 4096 on top-1 if VRAM allows.
- **Halt** (enforced by `run_next_experiment.py`):
  - NaN cascade on 2 consecutive runs, OR
  - Zero-gate-activation on 2 consecutive runs, OR
  - No AUC-stripped improvement of ≥0.005 over last 4 runs **AND** ≥6 valid x-attn runs have completed (*Minor 5*).

### How the loop is invoked

Three options, decided at execution time:

1. **Cron-driven re-invocation** (default): `scripts/agent_tick.sh` runs every N minutes via crontab. Wrapper pipes a fixed prompt — "Read `AGENT_INSTRUCTIONS.md` and continue" — into `claude` or `codex` CLI. Lockfile prevents concurrent runs.
2. **One long Claude Code session**: simplest, session-context-fragile over 3 days.
3. **Slash command** (e.g., `/loop`) only if verified present on the GPU box.

Default: option (1). Documented in `RUNBOOK.md`.

### Budget control

```yaml
# configs/budget.yaml
max_experiments: 12
max_gpu_hours: 18
stop_on_nan_cascade: true
stop_on_zero_gates: true
stop_on_convergence: true        # only after ≥6 valid x-attn runs (Minor 5)
min_valid_runs_before_halt: 6
```

---

## Day-by-day timeline (3 days + Day 0 pre-flight)

### Day 0 — Pre-flight (2-3 hours, ideally before renting H100)

| Block | Focus | Deliverables |
|---|---|---|
| Hr 0-2 | Env + substrate | Pin Docker image; `RUNBOOK.md` skeleton; `scripts/{preflight_check,run_next_experiment,parse_metrics,merge_stage0_lora,backup_to_external}.{py,sh}`; `eval/{score_risk,leakage_checks,eval_modes,bootstrap_ci}.py`; `src/train/mixers/eval_mode_dropout.py`; `src/auto_research/AGENT_INSTRUCTIONS.md`; `configs/{sweep_space,budget}.yaml`; `experiment_template.yaml` |
| Hr 2-3 | First RunPod boot | Network volume attached at pod creation, mounted at `/workspace`; `preflight_check.py` confirms GPU/VRAM/model-download/tokenizer-roundtrip/write-access; W&B online/offline configured |

### Day 1 — Vertical slice first, scale second

| Block | Focus | Deliverables |
|---|---|---|
| **Hr 0-2** | Tokens + fencer + bucketer | `src/tokenizer/custom_tokens.py` (journey, actor, event, PII, bucketed-feature); `src/tokenizer/fencer.py`; `data/gen/feature_bucketer.py`; round-trip smoke test; embedding-avg init verified; `data/cards/dataset_card.md` skeleton |
| **Hr 2-6** | **Vertical slice** | 3 journeys (`clean`, `sim_swap`, `hn_travel`) × 2 actors (`human`, `agent_compromised`); 1-2k LLM-narrated pairs **with narrative leakage scan green**; 5k templated-narrative cheap pairs; `train_cpt_light.py` 100 steps green with eval-mode dropout active; `eval/score_risk.py` produces AUC on 200-example mini-eval; `eval/leakage_checks.py` + `narrative_leakage_scan` confirm strip/opaque/policy compliance; `experiments.jsonl` append + bootstrap CI work. **Make-or-break.** If past Hr 8, descope and stay on vertical slice. |
| Hr 6-10 | Scale + CPT-light → merge | All 9 journeys × 6 actors; 20-30k LLM-narrated pairs (leakage-scanned); 50-100k templated-narrative; carve 5k stratified fast eval + 50k templated medium eval; `train_cpt_light.py` ~3-4 hr → LoRA checkpoint; `merge_stage0_lora.py` → `qwen3-8b-cpt-light-merged` |
| Hr 10-14 | CPT 3-mode eval + Day-1 writeup | CPT-light-merged evaluated under stripped/opaque/full on 5k fast; per-journey AUC + leakage audit + bootstrap CI recorded. Human writes README Day-1 section. Day-2 starting state confirmed. |

### Day 2 — Architecture + first sweep batch

| Block | Focus | Deliverables |
|---|---|---|
| Hr 0-4 | X-attn architecture | `cross_attn_block.py`, `resampler.py`, `qwen_xattn_wrapper.py`, `small_transformer` encoder, LoRA-on-Q wiring (Stage-1, fresh) |
| Hr 4-6 | Three more baselines | `train_lora_text_only.py`, `train_structured_as_text.py`, `train_event_only_classifier.py` *(v3)* — all three baselines complete; each evaluated under three eval modes; metrics + CI recorded |
| Hr 6-9 | First x-attn smoke | Load `qwen3-8b-cpt-light-merged`, freeze, attach x-attn; conservative defaults (`every_4`, `gate_init=0.01`, `slots=64`); 1500 steps, no NaN; gate trajectory logged; first x-attn experiment in `experiments.jsonl` via `run_next_experiment.py` |
| Hr 9-14 | Auto-loop first 4-6 sweeps | Agent reads `AGENT_INSTRUCTIONS.md`, proposes configs, launcher enforces; sweep covers `insertion_pattern × resampler_slots` (6 cells) at `gate_init=small_0.01`; summaries appended |
| Hr 14-16 | Day-2 writeup | Agent writes Day-2 README: integration friction, gates story, current leader vs **all four** baselines |

### Day 3 — Round-2 sweep + analysis + synthesis

| Block | Focus | Deliverables |
|---|---|---|
| Hr 0-8 | Round-2 + stress | 4-6 more experiments: top-2 perturbed on `gate_init`; one stress run at seq 4096 on top-1 if VRAM allows. Total Day 2+3: **10-12** experiments. Halt enforced by `run_next_experiment.py` against `budget.yaml`. |
| **Hr 8-11** | **Top-3 medium eval** | Top-3 x-attn configs + 4 baselines evaluated on **50k medium eval** under stripped mode for R@FPR=0.1%; per-journey-type AUC; agent-vs-human differential; hard-negative FPR; bootstrap CI on every reported number; leakage audit recorded |
| Hr 11-12 | **Top-1 large eval (if time)** | Top-1 x-attn vs structured-as-text on **100-200k large eval** for R@FPR=0.1% with tight CIs. **Skip if Hr 11 is reached late** — synthesis time is sacred |
| Hr 12-16 | **Final synthesis (sacred)** | Agent writes Day-3 README + final synthesis: full 3-day log, integration-friction catalog, gates story, **deltas across all three eval modes with CIs** for x-attn vs each of 4 baselines, per-journey/per-actor breakdowns, hard-negative FPR, **explicit Day-4 extend/pivot/stop recommendation** with rationale + concrete next-steps for real PayPal-internal POC |

**Slip cascade**: synthesis (Day 3 Hr 12-16) is sacred. If Day 2 slips, sweep budget shrinks. If Day 1 slips, vertical slice is floor. Large eval (Hr 11-12) is the first thing dropped if Hr 8-11 medium eval overruns.

---

## Critical files & their reuse

| File / path | Purpose |
|---|---|
| `RUNBOOK.md` | RunPod setup, env vars, storage layout, start/stop/resume, recovery, git-checkpoint policy |
| `data/cards/dataset_card.md` | Distribution, balance, template families, field visibility per baseline, leakage-scan summary |
| `data/gen/feature_bucketer.py` *(v3)* | Bucketed derived-feature token derivation |
| `data/gen/cheap_template_generator.py` *(v3)* | Templated-narrative + verdict generator (no LLM) |
| `eval/score_risk.py` | Deterministic AUC + R@FPR |
| `eval/leakage_checks.py` | Token-leakage + narrative-leakage detectors |
| `eval/eval_modes.py` | strip / opaque / full transforms |
| `eval/bootstrap_ci.py` *(v3)* | 95% CI on every metric |
| `src/train/mixers/eval_mode_dropout.py` *(v3)* | 50/25/25 full/opaque/stripped mix during training |
| `scripts/preflight_check.py` | GPU/VRAM/model/tokenizer/write-access |
| `scripts/run_next_experiment.py` | Lockfile + whitelist + dedup + launch + parse + atomic + append |
| `scripts/merge_stage0_lora.py` *(v3)* | Merge Stage-0 LoRA into base before Stage-1 |
| `scripts/parse_metrics.py` | stdout/W&B → metrics.json |
| `scripts/backup_to_external.sh` | Periodic external sync of top-3 + jsonl |
| `src/tokenizer/custom_tokens.py` | All token families |
| `src/tokenizer/fencer.py` | PII fencing |
| `src/model/{cross_attn_block, resampler, qwen_xattn_wrapper}.py` | x-attn architecture |
| `src/model/encoders/small_transformer.py` | Day-1-3 side encoder + classification-head variant for event-only baseline |
| `src/train/{train_cpt_light, train_lora_text_only, train_structured_as_text, train_event_only_classifier, train_xattn}.py` | Stage-0 + four baselines + main x-attn |
| `src/auto_research/AGENT_INSTRUCTIONS.md` | Agent playbook (incl. git-checkpoint policy) |
| `src/auto_research/{experiments.jsonl, sweep_state.yaml}` | State |
| `README.md` (journey log) | Daily writeups + final synthesis |

Reuse from prior research bundle: architectural defaults in `.claude/tasks/cross-attn-ato/poc-spec.md` (Pair 2 consensus) are inputs.

---

## RunPod persistence

- **Network volume attached at pod creation**, mounted at `/workspace`.
- All artifacts under `/workspace`: repo, datasets, checkpoints, W&B offline logs, experiment DB.
- **Atomic checkpoint writes** (v3): write to `path.tmp` then `os.rename(path.tmp, path)` so `backup_to_external.sh` never syncs a half-written file.
- `scripts/backup_to_external.sh` cron'd every ~30 min: syncs top-3 checkpoints + `experiments.jsonl` + `README.md` + `runs/*/metrics.json` + `runs/*/ci_report.json` to S3/R2/HF-private.
- Save checkpoints selectively: CPT-light-merged (always), 4 baselines (always), top-3 sweep arms (always), current-best on every update. Do **not** save every failed sweep checkpoint.
- Pod-restart recovery: `RUNBOOK.md` documents resume sequence (mount volume, re-run `preflight_check.py`, agent reads `experiments.jsonl` to know where it stopped).

---

## Risks & mitigations (v3 — superseding earlier versions)

| Risk | Mitigation |
|---|---|
| Eval leakage from journey/actor tokens | Three eval modes enforced by `eval/eval_modes.py`; **training-time eval-mode dropout (50/25/25)** so stripped-eval isn't OOD; headline metric = stripped; per-experiment `leakage_report.json` |
| **LLM narrator leaks class names into narrative body** | **Narrative-leakage policy** in narrator prompt; `narrative_leakage_scan()` in `leakage_checks.py`; non-compliant narratives regenerated; vertical-slice gates on leakage scan passing |
| **PII fencing erases the fraud signal** | **Bucketed derived-feature tokens** (`<amount_bucket=high>`, `<geo_distance=international>`, etc.) carry the signal in privacy-safe form; raw identifiers stay fenced |
| **Cheap-large eval distribution mismatch** | Templated **narrative** + verdict footer, not verdict-only; deterministic per-journey template generator |
| R@FPR=0.1% noise on small holdout | Eval ladder: 5k fast (sweep selection) → 50k medium (top-3) → 100-200k large (top-1 only if time); bootstrap CI on every reported metric |
| Objective/metric mismatch | Verdict footer in every training example; AUC scored via `label` log-prob — same surface trained on |
| Day-1 overreach | Vertical slice (3 journeys × 2 actors × 1-2k) by Hr 6 is make-or-break; scaling conditional on green slice + green narrative-leakage scan |
| Full CPT memory blowup | Default embedding + LoRA + merge; fall-back ladder documented; 500-step memory smoke gates full FT |
| **Adapter confusion across stages** | Stage-0 LoRA explicitly **merged into base** via `merge_stage0_lora.py`; Stage-1 starts fresh LoRA-on-Q on the merged checkpoint |
| **Event-only classifier wins** | This *is* a valid POC finding; the writeup must report it honestly (and reframe x-attn as explanation/grounding, not classifier) |
| **Premature convergence halt** | Halt requires ≥6 valid x-attn runs *and* no ≥0.005 lift in last 4 |
| Cross-attn beats CPT but not structured-as-text concat | The load-bearing comparison. Reported with bootstrap CIs; non-overlapping CIs required for a "win" claim |
| Agent-only orchestration brittleness | Deterministic launcher owns lockfile/whitelist/dedup/parse/atomic-write; agent only proposes + summarizes + git-checkpoints code edits |
| `/loop` not available | Cron-driven `agent_tick.sh` is default |
| RunPod pod terminated mid-run | Network volume + external backups + atomic writes |
| LLM narrative generation cost overrun | Cap at 25k narratives; templated-narrative path serves all eval-set needs without LLM calls |
| Synthetic-data saturation (all baselines AUC ceiling) | Increase hard-negative ratio; subtler adversarial-actor signatures; documented as a finding |
| **Agent modifies code unsafely** | `AGENT_INSTRUCTIONS.md` mandates `git add -A && git commit` before any code edit; revert path documented |
| **Δ AUC <0.005 drives a Day-4 decision** | Bootstrap CI required; non-overlapping CIs required for a "win" claim |

---

## Verification (how we know we're done)

End-of-POC checklist:

1. `README.md` exists with 3 daily writeups + final synthesis + explicit **Day-4 extend/pivot/stop recommendation**.
2. `experiments.jsonl` has ≥8 valid entries (10-12 target).
3. All **four** baselines (CPT-light-merged, LoRA-text, structured-as-text, event-only classifier) trained, evaluated under three eval modes, recorded with bootstrap CIs.
4. Top-3 x-attn configs evaluated on the **50k medium eval** for R@FPR=0.1% with CIs.
5. (If time) Top-1 vs structured-as-text on **100-200k large eval** with tight CIs.
6. Per-experiment `leakage_report.json` present including **narrative-leakage scan** result; no headline metric from `full`-mode eval.
7. Gate-activation magnitude logged for every x-attn experiment.
8. Per-journey-type AUC breakdown for top-3 + baselines (stripped mode).
9. Agent-actor vs human-actor differential reported.
10. Final synthesis answers all of: integration friction, gates story, x-attn-vs-{CPT, LoRA, structured-as-text, event-only} deltas with CIs, per-journey breakdown, per-actor differential, hard-negative FPR, **whether cross-attn is providing classification lift or only explanation/grounding** (the v3 question), Day-4 recommendation.
11. Next-steps section ties findings back to `.claude/tasks/cross-attn-ato/next-steps-checklist.md` — which assumptions held, which broke.

### Smoke tests — three layers

**Layer A — Local scaffold smoke (no GPU, no data, no model download; runnable today on any laptop):**

```bash
# Pure-Python eval/leakage assertions. Verifies eval_modes,
# leakage_checks.narrative_leakage_scan, and that the randomized opaque
# mapping works. Uses python3 (no `python` alias on macOS by default).
python3 - <<'PY'
import sys, random
sys.path.insert(0, '.')
from eval import eval_modes
from eval.leakage_checks import narrative_leakage_scan
text = '<journey_sim_swap><actor_human><event_login> <amount_bucket=high></journey_sim_swap>'
assert 'journey_sim_swap' not in eval_modes.apply(text, 'stripped')[0]
assert '<amount_bucket=high>' in eval_modes.apply(text, 'stripped')[0]
assert 'journey_type_' in eval_modes.apply(text, 'opaque', rng=random.Random(0))[0]
a = eval_modes.apply(text, 'opaque', rng=random.Random(0))[0]
b = eval_modes.apply(text, 'opaque', rng=random.Random(1))[0]
assert a != b, 'opaque must be randomized per call'
assert not narrative_leakage_scan('This is fraudulent SIM-swap')['clean']
assert narrative_leakage_scan('Device change followed by password reset')['clean']
print('layer A (local scaffold) smoke OK')
PY
```

**Layer B — Pod preflight (requires H100 + RunPod-like env):**

```bash
# Validates CUDA, VRAM>=70GB, /workspace writable, model download,
# tokenizer roundtrip, bitsandbytes version, W&B configured.
python3 scripts/preflight_check.py
```

This is not runnable locally; it lives in the pod boot sequence (Day 0 / Hr 2-3). Use it as the gate between booting the pod and any Day-1 work.

**Layer C — Day-1 end-to-end (requires Day-1 modules: data generators, tokenizer registry, trainers):**

```bash
python3 data/gen/build_dataset.py --n 100 --out data/samples/smoke
python3 -m src.tokenizer.custom_tokens --check
python3 -m eval.leakage_checks --dataset data/samples/smoke --modes stripped,opaque,full --narrative-scan
accelerate launch src/train/train_xattn.py \
  --config src/auto_research/runs/exp_smoke/config.yaml
# bootstrap_ci.py is invoked automatically by run_next_experiment.py; the
# stand-alone call below is for ad-hoc debugging only.
python3 -m eval.bootstrap_ci \
  --predictions src/auto_research/runs/exp_smoke/predictions_stripped.jsonl \
  --out src/auto_research/runs/exp_smoke/ci_report_stripped.json
```

If Layer A passes, the scaffold itself is sound. Layer B is the pod gate. Layer C unblocks once Task #31 (tokens + fencer + bucketer) and Task #32 (vertical-slice data + trainers) land.

---

## Out of scope (for clarity)

- Multi-GPU / FSDP / multi-node training.
- Real PayPal data — synthetic only.
- Production serving / quantization / vLLM integration.
- A/B testing against a real fraud system.
- Beating the literature.
- MoE base models (Kimi, DeepSeek).
- Tabular-foundation-model parallel track.
- Stage-2 SFT.
- FT-Transformer / CNN+LSTM side encoders (Day 4+).
- `every_2` insertion pattern, `learnable` gate init, resampler slots {32, 256}.
- Verbose structured-as-text format (compact only).
