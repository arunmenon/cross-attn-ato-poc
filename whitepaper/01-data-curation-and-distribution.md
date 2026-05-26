# Data Curation and Distribution

**Whitepaper companion document · v1.2 · 2026-05-22**

This document covers the synthetic data pipeline that powered all three sweep generations (v3 → v4 → v5). It is one of four companion deep-dives behind the master whitepaper (`00-whitepaper-main.md`). The other three are the agentic harness (`02-agentic-experiment-harness.md`), the eval strategy (`03-eval-strategy.md`), and the cross-attention experiments (`04-cross-attention-experiments.md`).

The data is the substrate every other choice rests on. The v3 → v4 pivot was a data pivot, not an architecture pivot — and the v5 result was constrained by a data ceiling, not an architecture ceiling. This document explains how the data was built, what went wrong in v3, what was fixed in v4, and what the data looks like as of v5.

---

## 1. Design goals

The synthetic ATO dataset has to support four downstream operations: (1) train a frozen-LM-plus-cross-attention model under a Flamingo-style adapter shape; (2) evaluate that model against four baselines (`CPT-light-merged`, `LoRA-text-only`, `structured-as-text concat`, `event-only classifier`) under apples-to-apples conditions; (3) survive a leakage audit on the train/eval split; (4) admit per-family error decomposition with bootstrap confidence intervals.

The four operations imply four design choices:

1. **Paired streams per example.** Every training row has a structured event timeline (consumed by the side-stream encoder) and an analyst-style narrative + verdict footer (consumed by the LM). The two streams encode the same underlying session at different abstractions. This is the modality gap that cross-attention is designed to bridge — and the gap that v3 inadvertently collapsed.

2. **Three token families with distinct visibility regimes.** PII-fencing tokens (hygiene), bucketed feature tokens (the fraud signal), and journey/actor structural tokens (label-adjacent metadata that has to be hidden at eval). Each family has a different policy.

3. **A leakage-control regime built into generation, not bolted on at eval.** The v3 narrator-caching bug taught us that any post-hoc leakage audit can only catch what its scanner is configured for; we now stratify on `structured_events_hash` *before* narration and enforce a `text_hash` dedup invariant *after* narration. The clean-eval mask in `eval/leakage_checks.py` is the last line of defense, not the first.

4. **Class balance that supports per-family CI on hard negatives.** Roughly 40% fraud / 30% hard-negative / 30% legit (Figure 3A). Hard negatives are subdivided into 4–5 families (depending on generation) with class-stratified train/eval splits to make per-family bootstrap CIs meaningful.

The v4 pivot satisfied all four constraints; the v3 implementation satisfied only the first and third in spirit. Section §3 below walks the diff.

---

## 2. Three token families

The token families and their visibility regimes are summarized in Figure 3D and detailed here.

### 2.1 PII-fencing tokens

Eight opaque placeholders replace raw personally-identifying information at generation time:

```text
<acct_id>   <email>     <phone>     <device_id>
<ip>        <recipient> <merchant>  <browser>
```

These tokens carry **no signal**. The model cannot learn that `<acct_id>=42` correlates with fraud; the integer was scrubbed before tokenization. The point is hygiene — to make the synthetic data shippable to non-cleared engineers, to prevent any one example from being memorizable, and to maintain the discipline that derives features from raw identifiers rather than reading them directly. PII-fencing tokens are always visible at eval; they do not change between stripped, opaque, and full modes.

Implementation: `src/tokenizer/fencer.py` deterministically replaces raw identifiers in the event stream and narrative before generation. The narrator never sees the unfenced version.

### 2.2 Bucketed feature tokens — **the fraud signal**

Nine bucketed derived-feature families carry the actual fraud signal:

| Family | Buckets |
|---|---|
| `<amount_bucket=…>` | `low` (<$50), `medium` ($50–500), `high` ($500–5k), `extreme` (>$5k) |
| `<geo_distance=…>` | `local` (<50 km), `domestic_far`, `international` |
| `<ip_risk=…>` | `low`, `medium`, `high` (VPN / Tor / datacenter ASN) |
| `<device_age=…>` | `known` (>30 d), `new` (<7 d), `rare` (seen <3 times) |
| `<merchant_risk=…>` | `normal`, `elevated` |
| `<txn_velocity=…>` | `normal`, `bursty` (>N in 1 h), `extreme` (>N in 5 min) |
| `<recipient_age=…>` | `known` (>30 d in account graph), `newly_added` (<24 h) |
| `<session_dwell=…>` | `short`, `normal`, `extended` |
| `<auth_strength=…>` | `mfa_strong`, `password_only`, `cookie_only` |

These tokens are derived from raw event values by `data/gen/feature_bucketer.py`. The buckets are coarse enough to be privacy-respecting (you cannot reconstruct the exact transaction amount from `amount_bucket=high`) and granular enough to support fraud-vs-legit discrimination.

**Visibility regime — and the v3 → v4 pivot.** In v3, bucketed feature tokens appeared in *both* the structured event stream *and* the narrator's input. The narrator's `SYSTEM_PROMPT` had compliant-example phrases that taught it to paraphrase these into the narrative — `amount_bucket=high` → "a high-value transfer", `recipient_age=newly_added` → "a freshly-added recipient". The result was that the fraud signal flowed end-to-end from event → narrator's prompt → narrator's output → narrative text. The LM, reading the narrative through self-attention, already had the structured signal; cross-attention had nothing unique to fetch.

In v4 (§3), the narrator's view was stripped of bucketed tokens. The narrator now writes behavioral *shape* ("the user logged in, changed credentials, initiated a transfer") without quantifying ("a $4,500 transfer" → "a transfer"; "to a freshly-added recipient" → "to a recipient"). Bucketed feature tokens live only in the structured event stream.

### 2.3 Journey / actor structural tokens

Two structural-token families bracket each example with metadata about *what kind of session this is*:

**Journey families.** v3 had seven (`clean`, `cred_stuff`, `sim_swap`, `phish_takeover`, `malware_rat`, `mule_chain`, `hn_travel`, `hn_large_purchase`, `hn_account_recovery`). v4 added two more — `phish_takeover_mfa_phished` (a fraud-dual: text reads safe, events show the anomaly) and `hn_recovery_high_amount` (an adversarial-legitimate: text reads like fraud, events reveal legitimacy). v5 reuses the v4 schema verbatim.

**Actor types.** Six (`actor_human`, `actor_agent_buying`, `actor_agent_finance`, `actor_agent_compromised`, `actor_agent_adversarial`, `actor_hybrid`). Agent-actors get distinct event-timing distributions (programmatic step patterns, tool-use traces, API-like cadence) so the model has a meaningful signal to learn on agent-vs-human discrimination.

**Visibility regime.** Journey and actor tokens are *eval-mode-controlled*:

- **`stripped`** (the headline eval mode) — all `<journey_*>` and `<actor_*>` tokens removed. The model has to infer journey type from the narrative + bucketed features + event stream.
- **`opaque`** — journey/actor tokens replaced with neutral random IDs (`<journey_type_a3f>` / `<actor_type_b71>`). The structure is preserved but the label is hidden.
- **`full`** — all tokens visible. Debug only; never reported as a win condition. Used to confirm the model can solve the task when it sees the label, as a saturation sanity-check.

During training, **eval-mode dropout** (Figure 3C, `src/train/mixers/eval_mode_dropout.py`) applies one of the three modes per example with probabilities 50% / 25% / 25%. The intent is to make stripped-mode eval an in-distribution evaluation rather than an out-of-distribution one — every model sees all three eval-mode views during training.

---

## 3. The v4 pivot — restoring the modality gap

The v3 null result on cross-attention was, in retrospect, a data-design failure rather than an architectural one. Two pathologies in the v3 data pipeline collapsed the modality gap that cross-attention is designed to bridge. Both were diagnosed during the v4 pivot planning. The v4 implementation is a coordinated four-change fix.

### 3.1 Pathology 1 — narrator-mediated redundancy

In v3, `data/gen/narrative_generator.py::_serialize_events_for_prompt` constructed the narrator's user message by serializing the full structured event stream verbatim:

```text
[t=0]   login        actor=<actor_*>  geo_distance=local      ip_risk=low     auth_strength=password_only
[t=2]   device_add   actor=<actor_*>  device_age=new
[t=4]   pw_reset     actor=<actor_*>  auth_strength=password_only
[t=7]   txn          actor=<actor_*>  amount_bucket=high      recipient_age=newly_added
```

The bucket key=value pairs were in plain view. The narrator's `SYSTEM_PROMPT` (lines 179–188 of `narrative_generator.py` in v3) further taught it to paraphrase:

> "A *high-value* transfer to a *freshly-added recipient* on a *previously-unseen device* should be described in plain English. Avoid the words 'fraud' or 'high risk' — describe what happened."

The compliant-example phrases were rich. Across 25,000 generated narratives, the v3 narrator faithfully paraphrased `amount_bucket=high` into "high-value", "large", "substantial"; `recipient_age=newly_added` into "freshly-added", "just-added", "new"; `device_age=new` into "previously-unseen", "new", "first-time". The bucketed fraud signal flowed end-to-end from event → narrator's prompt → narrator's output → narrative text.

**Consequence.** The LM, reading the narrative through self-attention, already had the structured signal. The structured-as-text baseline (which concatenates a serialized event stream into the prompt) and the cross-attention arm (which feeds the same event stream through a side encoder) were both receiving the same information through different surface paths. There was no modality gap for cross-attention to exploit.

**Diagnostic evidence.** A post-hoc analysis (`scripts/diagnose_data_overlap.py`) showed 10.7% of v3 eval rows shared narrative text with a train row, concentrated 35% in `hn_large_purchase` and 16% in `hn_account_recovery`. The mechanism was narrator caching by `(structured_events_hash, model, temperature)` — distinct examples with the same structured-events footprint shared a cached narrative — but the deeper finding was that the narrator was producing nearly-deterministic paraphrases of the same bucket combinations across distinct journeys.

### 3.2 Pathology 2 — label-deterministic hard-negative skeletons

In v3, `data/gen/journey_templates.py` (lines 367–398) hard-coded the feature signatures of each journey family:

```python
hn_account_recovery = JourneyTemplate(
    auth_strength="mfa_strong",      # always
    device_age="known",              # always
    ip_risk="low",                   # always
    geo_distance="local",            # always
    recipient_age="known",           # always
    ...
)

phish_takeover = JourneyTemplate(
    auth_strength="password_only",   # always
    device_age="rare",               # always
    ip_risk="high",                  # always
    geo_distance="international",    # always
    recipient_age="newly_added",     # always
    ...
)
```

The two families were *perfectly separable* at the feature level. A feature-level classifier with full visibility into the bucketed feature tokens could achieve 100% accuracy on the binary `hn_account_recovery` vs `phish_takeover` discrimination by learning a single decision boundary — and the synthetic generator made that boundary degenerate (the feature distributions did not overlap at all).

`scripts/diagnose_data_overlap.py` quantified this: `H(label | journey_family) = 0` and `H(label | bucket-event skeleton) = 0` across 2,454 distinct skeletons, with zero mixed-label skeletons. The per-family bucket-combination space was structurally small (`hn_large_purchase`: 12 skeletons across 2,481 rows; `hn_account_recovery`: 31 skeletons across 2,442 rows). 4,661 of 5,000 v3 eval rows shared a bucket-event skeleton with a train row — independent of the 534 narrative-text leaks.

**Consequence.** The 0.05–0.07 worst-family HN-FPR ceiling in v3 was *not* a measure of the model's capability. It was a small statistical edge effect at the 1%-legit-FPR operating point on a label-deterministic surface. Different models landed at slightly different points on the boundary because of tied-score handling and bootstrap noise; the architecture could not actually move the metric.

### 3.3 Pathology 3 — muddled baseline contract (the trainer-input mismatch)

A secondary issue: v3 trainers consumed differently-shaped inputs across arms. The `text_only` arm included event-line blocks wrapped in `<journey_X>` tokens in its LM prompt; the `xattn` arm did not (it fed events only through the side encoder). The "architecture comparison" was confounded with a "prompt-content comparison." Even if the data pathologies had been absent, the v3 result would have been hard to interpret: did `xattn` not separate from `text_only` because cross-attention adds nothing, or because the two arms saw different prompts?

### 3.4 The four-change fix

The v4 pivot addresses all three pathologies in one coordinated change. Each change has its own implementation footprint:

**Change 1 — Strip the narrator's view of bucketed tokens.** `data/gen/narrative_generator.py::_serialize_events_for_prompt` now produces a behavioral-skeleton-only serialization for the narrator: event types and timing, plus qualitative actor descriptors (`agent_buying`, `human`), but no bucketed feature tokens. The narrator's input becomes:

```text
[t=0]   login        actor=human
[t=2]   device_add   actor=human
[t=4]   pw_reset     actor=human
[t=7]   txn          actor=human
[t=9]   txn          actor=human
```

The `SYSTEM_PROMPT` compliant-example phrases were rewritten to avoid quantifying. "A high-value transfer" became "a transfer." "On a previously-unseen device" became "from this device." "To a freshly-added recipient" became "to a recipient." The narrator now writes behavioral *shape*; the features live only in the structured event stream.

A paraphrase scanner (`eval/leakage_checks.py::narrative_leakage_scan`) was extended to detect 40+ patterns across the amount, device, IP, recipient, velocity, auth, and session-dwell families. Non-compliant narratives are flagged and regenerated. The narrator cache key now includes a `NARRATOR_PROMPT_VERSION = 2` field to prevent v3-cached narratives from leaking into v4 generation.

**Change 2 — Stochastic feature signatures.** `journey_templates.py` was rewritten so each family draws features from a *distribution*, not a fixed signature:

```python
hn_account_recovery = JourneyTemplate(
    auth_strength={"mfa_strong": 0.55, "password_only": 0.30, "cookie_only": 0.15},
    device_age={"known": 0.60, "new": 0.30, "rare": 0.10},
    ip_risk={"low": 0.70, "medium": 0.25, "high": 0.05},
    ...
)

phish_takeover = JourneyTemplate(
    auth_strength={"password_only": 0.55, "mfa_strong": 0.25, "cookie_only": 0.20},
    device_age={"rare": 0.60, "new": 0.30, "known": 0.10},
    ...
)
```

The fraud and legit feature distributions now overlap. A feature-level classifier can no longer perfectly separate; the model has to find subtler signal. `H(label | bucket-event skeleton)` is now > 0 on a per-skeleton basis, and the v3 ceiling effect dissolves.

**Change 3 — Adversarial cross-modal hard-negative families.** Two new journey families were added, designed to *demand* that the model attend to the event stream:

- **`hn_recovery_high_amount` (legitimate).** Text reads like classic ATO: "the user logged in from a new device, reset the password, then made a large transfer to a newly added recipient." Events reveal legitimacy: the new device is the account holder's own (matched on hardware fingerprint pattern); the new recipient is the account holder's other account; MFA was used in the password reset. **Text alone misses it; events catch it.** This family contains 500 training rows (human actor) and ~100 hybrid/finance-agent rows, plus 78 in the 5k held-out eval.
- **`phish_takeover_mfa_phished` (fraud).** The dual. Text reads safe: "the user logged in, transferred funds to a known recipient." Events reveal the anomaly: the "known" recipient was added 20 minutes prior (`recipient_age=newly_added`); the device has a subtle anomaly (`device_age=new` despite matching one hardware feature); MFA was used but in a token-reuse pattern that the side encoder can flag. **Text alone misses it; events catch it.** 400 training rows for human actor, plus ~400 for compromised/adversarial agents, plus 71 in the 5k held-out eval.

These two families are *the* test of cross-attention on this data. They are constructed so the LM cannot solve them from the narrative alone, and the model is forced to either learn cross-attention (in the `xattn` arm) or fail (in the `text_only` arm). The v4 result on the adversarial fraud families (`phish_takeover` recall: text_only 0.1122 vs xattn 1.0000; `phish_takeover_mfa_phished` recall: text_only 0.0000 vs xattn 0.9718; CIs separated) is what the architecture pivots from being a null result to being an architectural win.

**Change 4 — Per-arm text-field routing.** The dataset is now stored in canonical form. Each example has explicit fields:

```json
{
  "narrative": "...",                    // the LM-readable analyst-style prose
  "events_text": "<events>...</events>", // serialized structured stream (for SAS arm)
  "structured_events": [...],            // list-of-dicts (for the side encoder)
  "wrapper_tokens": ["<journey_X>", "</journey_X>"],
  "verdict_footer": "<risk_verdict>label: fraud\n...</risk_verdict>",
  "metadata": {...}
}
```

Each trainer constructs its arm-specific input by composing canonical fields:

- **`text_only`:** `wrapper_tokens + narrative + verdict_footer` (no events at all)
- **`structured_as_text`:** `wrapper_tokens + events_text + narrative + verdict_footer`
- **`xattn`:** `wrapper_tokens + narrative + verdict_footer`, with `structured_events` fed only into the side encoder

**The critical invariant:** `text_only` and `xattn` now see byte-identical LM prompts. The architecture comparison turns *only* on the presence or absence of the side stream. Any difference in scores between the two arms is causally attributable to cross-attention.

---

## 4. Data distribution as of v4 / v5

![Figure 3. Data distribution & eval-mode mix](figures/fig3-data-distribution.svg)

Figure 3 summarizes the four key views: class balance (panel A), the journey × actor heatmap (panel B), the eval-mode dropout mix during training (panel C), and the three token families with their visibility regimes (panel D). Detailed numbers below.

### 4.1 Class balance

The training pool is 25,000 LLM-narrated paired examples. Class balance:

- **Fraud — 40% (10,000 rows).** Distributed across `cred_stuff`, `sim_swap`, `phish_takeover`, `phish_takeover_mfa_phished` (v4 new), `malware_rat`, `mule_chain`.
- **Hard negatives — 30% (7,500 rows).** Distributed across `hn_travel`, `hn_large_purchase`, `hn_account_recovery`, `hn_recovery_high_amount` (v4 new).
- **Legit — 30% (7,500 rows).** The `clean` journey family.

The 5k held-out eval (n = 5,002 after v4 leakage controls) is class-stratified to preserve these ratios. Two 50k medium-eval surfaces exist alongside it: a templated medium-eval at `data/eval_medium_50k/` (free to produce — no LLM calls — preserves the same class balance with broader coverage) and an LLM-narrated medium-eval at `data/eval_medium_50k_llm/` (built but not yet scored on the v5 winner; the recommended next test for tightening per-family confidence intervals).

### 4.2 Journey × actor distribution

Figure 3B shows the per-cell counts on the training pool. Empty cells (`agent_buying` committing fraud, `agent_finance` doing `cred_stuff`, etc.) are intentionally zero — they represent journey × actor combinations that are implausible in the design space. The new v4 families (`phish_takeover_mfa_phished` and `hn_recovery_high_amount`, shown with outlined cells in the figure) are concentrated on `human` and `agent_compromised` actors, where the adversarial dynamic is most plausible.

### 4.3 Eval-mode dropout

Training applies one of the three eval modes per example with probabilities 50% (full) / 25% (opaque) / 25% (stripped). This makes stripped-mode eval an in-distribution evaluation rather than an OOD one. The full-mode 50% weight ensures the model also learns to use the journey/actor tokens when they are present (for opacity-controlled and full-mode eval); the 25%/25% stripped/opaque weights ensure the model can also solve the task when the labels are hidden. Implementation: `src/train/mixers/eval_mode_dropout.py`, applied per example in the dataloader collate function.

---

## 5. Leakage controls

The v3 narrator-caching bug taught us that any post-hoc leakage audit can only catch what its scanner is configured for. The v4 leakage-control regime is layered: generation-time prevention, post-generation detection, and eval-time clean-mask defense.

### 5.1 Generation-time prevention

- **Pre-narration `structured_events_hash` stratification.** Examples are bucketed by `structured_events_hash` before narration; identical-skeleton rows are kept on the same side of the train/eval split. This prevents the narrator-cache mechanism from leaking text across the split.
- **Cache key includes `NARRATOR_PROMPT_VERSION`.** The narrator cache key is `(structured_events_hash, model, temperature, NARRATOR_PROMPT_VERSION)`. v3 narratives cannot leak into v4 generation; v4 narratives cannot leak into a future v5 (or v6) regen if the prompt changes.
- **Narrator paraphrase ban + regex enforcement.** The narrator's `SYSTEM_PROMPT` bans explicit class names (`fraud`, `legit`, `ATO`, `phishing`, etc.) and quantifying paraphrases of the bucketed feature tokens (40+ patterns covering amount, device, IP, recipient, velocity, auth, session families). `eval/leakage_checks.py::narrative_leakage_scan` enforces the ban via regex; non-compliant narratives are flagged and regenerated.

### 5.2 Post-generation detection

- **Per-example `text_hash` dedup invariant.** After generation, the build pipeline computes `text_hash = hash(narrative)` for every row and asserts no two rows in the dataset share both a `structured_events_hash` and a `text_hash`. Violations would indicate a deeper bug (e.g., the narrator emitting a fixed boilerplate for some inputs) that the structured-events stratification missed.
- **Per-family overlap audit.** `scripts/diagnose_data_overlap.py --check` produces a per-family report of train/eval text-hash and structured-events-hash overlap. In v4, this report shows zero family-concentrated overlap (vs. v3 where `hn_large_purchase` had 35% and `hn_account_recovery` had 16%).

### 5.3 Eval-time clean-mask defense

- **`compute_clean_eval_mask`** (`eval/leakage_checks.py`) drops any eval row whose `text_hash` or `structured_events_hash` appears in the training set. The mask is applied automatically by the launcher (`scripts/run_next_experiment.py::run_post_processing`) for every new run. In v3, this mask dropped 534 of 5,000 rows (10.7%); in v4 it drops 0 of 5,002 rows. The clean-eval mask is a defense in depth: if generation-time prevention misses something, the eval surface is still trusted.

### 5.4 Per-experiment leakage record

Every run records its leakage state inline on the `experiments.jsonl` row via five fields: `leakage_clean` (bool — the launcher's assertion that the run was evaluated on a clean surface), `clean_eval_n` (post-mask eval set size, 5,002 in v4/v5), `clean_eval_dropped` (count of rows excluded by the mask, 0 in v4/v5), `clean_eval_mask_text_overlap` (rows dropped by the text-hash overlap check), and `clean_eval_mask_events_overlap` (rows dropped by the structured-events-hash overlap check). Any `leakage_clean: false` row would have stopped the run; none did in v4 or v5.

---

## 6. Reproducibility

The data pipeline is deterministic given a seed and the generator versions. The key knobs:

```bash
# Reproduce the v4 training pool (25k LLM-narrated pairs).
python3 data/gen/build_dataset.py --n 25000 --mode llm --out data/train_llm_narrated_v4

# Reproduce the 5k held-out eval.
python3 data/gen/build_dataset.py --n 5000 --mode llm --eval_frac 0.2 --out data/eval_fast_5k_v4

# Run the overlap diagnostic on the result.
python3 scripts/diagnose_data_overlap.py --data-dir data/train_llm_narrated_v4 --check

# Run the leakage check (also runs automatically as part of run_next_experiment).
python3 -m eval.leakage_checks --train-eval-overlap data/train_llm_narrated_v4
```

The narrator cost for the 25k pool was **$2.03** (`gpt-5-nano-2025-08-07`, 26,319 API calls with 332 cache hits, `ThreadPoolExecutor` concurrency saturating the rate limit; see `data/train_llm_narrated_v4/build_summary.json::llm_cost`). The 50k templated medium-eval is free (no LLM calls). Build times: **~72 minutes** for the 25k LLM-narrated pool, ~2 seconds for the 50k templated medium-eval, ~30 seconds for the 5k stratified held-out eval after the LLM pool is built.

---

## 7. What we did not do

**Real-data validation.** No PayPal-internal production data was used. Every result in the whitepaper is conditional on the synthetic distribution. The most important next step for any production-transfer claim is to run the v4 leader configuration against a held-out anonymized window of real fraud-and-legit traffic.

**Adversarial training of narrator and generator.** The narrator and the generator were not co-designed adversarially. The narrator could in principle learn to leak structured signal through subtle stylistic patterns that the paraphrase scanner does not catch (sentence-length distributions, lexical choice frequencies, etc.). We did not test for such second-order leakage.

**Counterfactual hard negatives.** The v4 `hn_recovery_high_amount` family was constructed by manually specifying "text reads like fraud, events reveal legitimacy." A more rigorous approach would be to generate counterfactual pairs: take a fraud journey, perturb it minimally until the event-level features tip it to legit, and report the model's behavior on the paired (fraud, legit) examples. This would require generator extensions that v4 does not have.

**Distribution shift over time.** The dataset is static. A production analog would have to model concept drift in actor patterns (new agent types entering the ecosystem), fraud-method shifts (new takeover modalities), and legitimate-behavior changes (new device hardware, new browser fingerprints). The synthetic generator's `journey_templates.py` is point-in-time as of 2026-05-21.

---

## 8. References

- **Companion documents in this whitepaper set:** `00-whitepaper-main.md`, `02-agentic-experiment-harness.md`, `03-eval-strategy.md`, `04-cross-attention-experiments.md`.
- **Implementation:** `data/gen/{journey_templates,agent_actor_mixer,feature_bucketer,pii_fencer,narrative_generator,cheap_template_generator,build_dataset}.py`, `data/cards/dataset_card.md`.
- **Diagnostics:** `scripts/diagnose_data_overlap.py`, `eval/leakage_checks.py`.
- **Detailed v4 pivot plan:** `.claude/tasks/data-v4-pivot-plan.md` and `.claude/tasks/data-v4-verdict.md` in the repository.
- **Day-2 data diagnostic record:** `docs/day-2-data-diagnostic.md` and `docs/day-2-results.md`.
