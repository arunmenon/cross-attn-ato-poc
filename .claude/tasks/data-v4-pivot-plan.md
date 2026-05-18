# Plan: Data v4 — Restore the Modality Gap That Cross-Attention Needs

## Context

The 3-day POC (v3) is closed. After 18 valid cross-attention arms across
4 dial families (architecture, init, training-schedule, rank capacity),
the headline finding is unambiguous: **the leader cell (`round1_002`,
hn_worst 0.0524 [0.042, 0.065]) is statistically tied with the
`structured_as_text` baseline (0.0507 [0.041, 0.064]) — the gates
stayed near init across every perturbation we tried**. Cross-attention
provides no detectable classification lift on this synthetic surface.

The user's diagnosis — confirmed by code inspection — is that we built
the synthetic data with the structured event signal already embedded
into the LLM-narrated text via two distinct mechanisms:

1. **Narrator pre-conditioning** (`data/gen/narrative_generator.py`).
   The narrator's user-message embeds the structured event stream
   verbatim (including bucket key=value pairs like
   `amount_bucket=medium`, `device_age=new`, `recipient_age=newly_added`).
   The SYSTEM_PROMPT's compliant-example phrases (`narrative_generator.py:179-188`)
   literally teach the LLM to paraphrase these as "high-value transfer,"
   "previously-unseen device," "freshly-added recipient." Result: the
   bucketed signal flows from event → narrator's prompt → narrator's
   output → text stream of every training example. The LM reading the
   text already has the structured signal via self-attention; cross-attn
   has nothing to add.

2. **Label-deterministic hard-negative skeletons** (`data/gen/journey_templates.py`).
   Every `hn_account_recovery` sample is generated with a fixed feature
   signature: `{auth_strength=mfa_strong, device_age=known, ip_risk=low,
   geo_distance=local}`. Every `phish_takeover` has the opposite. They
   are not ambiguous — they are *perfectly separable* at the
   event-feature level. The 0.052-0.061 worst-family FPR ceiling is
   not generator ambiguity; it's a statistical edge effect (a small
   number of fraud samples bleed into the hn feature region under the
   FPR=1% threshold). No architecture can fix this either — the
   features are already as separated as the design allows.

Both findings together explain the null result. **The sweep was set up
to fail the moment v3's data design was finalized.** The architecture
worked; the data didn't give it a problem to solve.

The intended outcome of this plan is to **rebuild the dataset (v4) so
that the text and event streams encode genuinely complementary
information** — text describes session behavior qualitatively, events
carry the quantitative feature ground truth, and hard negatives become
genuinely ambiguous at the single-stream level. Then re-run the leader
cross-attention cell on v4 data and observe whether the gates open.

This plan is for **Day-4+ work**, not the closed 3-day POC.

---

## Scope at a glance (the simplification rule)

**Required (the only must-run pair):**

1. `text_only_v4` — same shared base, narrative-only prompt, no events
2. `xattn_v4` — same shared base, identical prompt, events routed via side stream

**Optional (run only if the must-run motivates further investigation):**

3. `structured_as_text_v4` — events serialized into the prompt (answers Q2: routing vs serialization)
4. `event_only_v4` — small_transformer + classifier head, no LM (characterizes events-alone ceiling)

The minimum valid question this experiment answers:

> Does adding the structured event side stream improve over the same
> model reading clean narrative only?

`text_only_v4` and `xattn_v4` must share **everything except event
access**: same v4 dataset, same Stage-0 merged checkpoint, same clean
narrative prompt, same training steps, same metric, same eval set.
Only difference: presence/absence of the structured-event side stream.

| Outcome | What it means |
|---|---|
| xattn beats text_only outside CI | side-stream events helped |
| xattn tied with text_only | side-stream events did not help |
| xattn worse than text_only | side-stream architecture hurt or optimization failed |

That is the entire experimental design. Everything else in this plan
is the supporting work to make those two arms actually apples-to-apples.

---

## Objective

Restore a real modality gap between the text and event streams, so that:

1. **Cross-attention has unique signal to fetch** — the text alone does
   not contain bucketed feature values; events are the only source of
   per-event feature precision.
2. **Text-only baselines (`cpt_light`, `lora_text`) become genuinely
   limited** — they can read the behavioral narrative but cannot recover
   the specific bucket values that distinguish edge cases.
3. **Hard negatives become genuinely ambiguous in a single stream** —
   `hn_account_recovery` requires *both* the narrative (which describes
   the recovery flow) AND the events (which carry the recovery-specific
   feature signature) to disambiguate from fraud.

Pass condition for v4 (before any conclusion is drawn): on the same
leader cell, **gates open past 0.05 max-gate-magnitude** OR `hn_worst`
moves outside the v3 leader's CI of [0.042, 0.065]. Either signal is
sufficient evidence that the modality gap matters.

---

## Two confirmed root causes (with code pointers)

### Root cause 1: Soft narrator-mediated redundancy

`data/gen/narrative_generator.py:314-394` (`_serialize_events_for_prompt`):

```python
def _serialize_events_for_prompt(journey: Journey) -> str:
    ...
    for ev in journey.events:
        bits = [f"t={ev['t']}s", f"event={ev['event']}"]
        for key in ("amount_bucket", "geo_distance", "ip_risk", "device_age",
                    "merchant_risk", "txn_velocity", "recipient_age",
                    "session_dwell", "auth_strength", ...):
            if key in ev:
                bits.append(f"{key}={ev[key]}")
        lines.append("  " + " ".join(bits))
    return "\n".join(lines)
```

This serialization is fed to the LLM narrator as its user message. The
narrator sees every bucketed feature value. The SYSTEM_PROMPT then
trains it (via compliant-example phrases at `narrative_generator.py:179-188`)
to translate them into natural English. **The narrator's output is
already a paraphrase of the structured signal.** There is no
information in the events that isn't already in the text.

### Root cause 2: Label-deterministic hard-negative skeletons

`data/gen/journey_templates.py` (around lines 367-398 for hn_account_recovery,
211-239 for phish_takeover):

| Journey | Feature signature |
|---|---|
| `hn_account_recovery` (legit) | `auth=mfa_strong, device=known, ip=low, geo=local, rcpt=known` |
| `phish_takeover` (fraud) | `auth=password_only, device=rare, ip=high, geo=international, rcpt=newly_added` |
| `hn_travel` (legit) | `auth=mfa_strong, device=known, ip=low, geo=international` |
| `hn_large_purchase` (legit) | `auth=mfa_strong, device=known, ip=low, geo=local` |

The hn families share `{auth=mfa_strong, device=known, ip=low}` as a
deterministic signature. Fraud families share the opposite. A model
that learns these signatures gets perfect classification at the
feature level — and the remaining FPR comes from a small slice of
fraud cases that the generator accidentally placed in the hn feature
region. No architecture can recover that slice because the events
themselves don't distinguish it.

---

## Why dual-view (text + events) is still the right shape

Before designing the pivot, let's confirm dual-view is worth keeping.

**The medical-risk analog**: clinical notes (qualitative) + labs/vitals
(quantitative) are genuinely complementary. The note says "patient
reported fatigue and shortness of breath"; the labs say "hemoglobin
8.2 g/dL, BNP 1500 pg/mL." Neither stream is the other's paraphrase.
Cross-attention between a frozen clinical LM and a structured-lab
encoder has real work to do.

**Apply to fraud**: a real analyst note doesn't say
`<amount_bucket=high>` — it says "the customer transferred a large sum
to a recently-added contact." A real event log doesn't have a narrative
— it has timestamped rows with precise feature values. The two streams
encode genuinely different views. Cross-attention should fetch the
quantitative ground truth when the LM's qualitative reasoning hits a
decision boundary.

**Conclusion: keep dual-view; fix the synthetic data so the streams
actually become complementary.**

---

## A third muddle: the baseline/checkpoint contract was never apples-to-apples

A code audit during this planning session surfaced a **second**, separate
problem alongside the redundancy bug: the v3 baselines started from
different base checkpoints AND saw differently-laden prompts, so the
"head-to-head" was never a clean architecture comparison even leaving
data redundancy aside.

### What v3 actually compared

| Arm | Base checkpoint | Text-field input |
|---|---|---|
| `cpt_light` | raw Qwen3-8B → CPT-light LoRA merged | narrative + event lines + journey/actor tokens + verdict footer |
| `lora_text` | **raw Qwen3-8B** (no CPT-light) | same as above |
| `structured_as_text` | qwen3-8b-cpt-light-merged | **events block** prepended **AND** events in the narrative wrapper text **AND** journey/actor tokens |
| `xattn` | qwen3-8b-cpt-light-merged | narrative + event lines + journey/actor tokens (the event lines are *in the text field already* — they're not pulled out for the cross-attn arm) |
| `event_only` | n/a (small_transformer + classifier head) | n/a |

The actual text field for every LM arm (from `build_dataset.py::serialize_journey`)
is wrapped as `<journey_X>...<event_login>t=0 <ip_risk=high>...<narrative>...</narrative>...</journey_X>`.
**The event-line block sits inside the LM's prompt for every arm**, not
just `structured_as_text`. So the "structured_as_text vs xattn" comparison
was actually:

- `structured_as_text`: events in text (twice — once in the prepended
  block, once in the wrapper) — no side stream
- `xattn`: events in text (once, in the wrapper) — *plus* a side stream
  that carries a third copy of the same information

That isn't a clean test of "does cross-attention work" — it's a test of
"does giving the LM a redundant side stream of information it already
has help." The answer was unsurprisingly no.

### What v4 must look like

For the central architectural question — *does routing events through a
side-stream encoder + cross-attn beat serializing them into the LM's
context window?* — to be answered honestly, every Stage-1 arm must:

1. **Start from the same base checkpoint**: `qwen3-8b-cpt-light-v4-merged`.
2. **See the same narrative text** (the v4 narrator with no bucketed
   tokens — Change 1).
3. **Differ only in how event information reaches the LM** — and that
   difference must be the single experimental variable.

The clean v4 prompt contract:

| Arm | Base checkpoint | Text-field input | Side stream |
|---|---|---|---|
| `text_only` (replaces `lora_text`) | qwen3-8b-cpt-light-v4-merged | journey/actor tokens + clean narrative + verdict footer. **No event lines.** | none |
| `structured_as_text` | qwen3-8b-cpt-light-v4-merged | journey/actor tokens + **`<events>` block** + clean narrative + verdict footer. | none |
| `xattn` | qwen3-8b-cpt-light-v4-merged | journey/actor tokens + clean narrative + verdict footer. **No event lines.** Identical to `text_only`. | structured event dicts (the only place events live) |
| `event_only` | n/a | n/a | structured event dicts → classifier head |
| `cpt_light_v4` (reference only, not a Stage-1 arm) | qwen3-8b-cpt-light-v4-merged | same as `text_only` | none — reports the Stage-0-only baseline |

### Two distinct questions, two distinct comparisons

The v4 setup separates two questions that v3 conflated:

| Question | Comparison | Status in v4 |
|---|---|---|
| **Q1: Does side-stream event information help over text alone?** | `text_only_v4` vs `xattn_v4` | **MUST RUN** — primary question |
| **Q2: Is cross-attention better than putting events in the prompt?** | `structured_as_text_v4` vs `xattn_v4` | **Optional** — only if Q1 is positive and budget remains |

**Running `xattn_v4` alone produces an uninterpretable FPR number.**
We can't tell if a good result is from cross-attn helping, the v4
data being easier, Stage-0 getting better, or the narrative already
solving the task on its own. The minimum scientifically useful run
set is therefore **`(text_only_v4, xattn_v4)`** — never just xattn.

The v3 `lora_text` baseline (raw Qwen + LoRA, no CPT-light) is
**dropped in v4** — it isn't apples-to-apples with any of the
Stage-1 arms. `cpt_light_v4` is reported as a reference number but
doesn't need its own training step (it IS the merged base checkpoint).

---

## The pivot — four coordinated changes

### Change 1: Strip the narrator's view of bucketed tokens (the load-bearing change)

Modify `data/gen/narrative_generator.py:_serialize_events_for_prompt`
to feed the narrator a **behavioral skeleton only** — event types,
timing, and qualitative actor descriptors — without bucket key=value
pairs:

```python
# v4 (proposed)
for ev in journey.events:
    bits = [f"t={ev['t']}s", f"event={ev['event']}"]
    # NEW: omit all bucket keys; only pass actor + event metadata.
    lines.append("  " + " ".join(bits))
```

The narrator now writes "the account holder logged in, then changed
credentials, then initiated a transfer" — without quantifying the
transfer amount, the device age, or the recipient relationship age.
Those facts live ONLY in the structured event stream.

Update SYSTEM_PROMPT compliant-example phrases to remove value-laden
adjectives: "high-value transfer" → "an outbound transfer";
"previously-unseen device" → "a device"; "freshly-added recipient" →
"a recipient." The narrator can mention an event type happened; it
cannot quantify it.

### Change 2: Stochastic feature signatures (kill label-determinism)

Refactor `journey_templates.py` so each journey family draws features
from a **distribution**, not a fixed signature. For each hn family,
some samples have fraud-like features; for each fraud family, some
samples have legitimate-looking features. Specifically for
`hn_account_recovery`:

```python
# v4: hn_account_recovery samples vary across the feature space
auth_choices = {
    "mfa_strong": 0.55,       # most legit recoveries used MFA
    "password_only": 0.30,    # some legit recoveries don't have MFA configured
    "cookie_only": 0.15,      # some are via persisted sessions
}
device_age_choices = {
    "known": 0.60,
    "new": 0.30,              # legitimate users do recover from new devices
    "rare": 0.10,
}
# similarly for ip_risk, geo_distance, recipient_age
```

Make fraud families overlap symmetrically: some `phish_takeover` cases
have `auth=mfa_strong` (attacker phished MFA), some have
`device=known` (attacker used compromised primary device). The
intent: a feature-level classifier cannot perfectly separate hn from
fraud — it has to find subtler signal.

### Change 3: Adversarial cross-modal hard negatives

Generate a new sub-family `hn_recovery_high_amount`: text reads as
classic ATO ("device change, password reset, large transfer to a new
recipient"), but events contain the disambiguating context (recipient
is actually the account holder's other account; device is "new" only
because of a recent OS upgrade; MFA was used). The text-only model
sees the fraudish flow; the event stream contains the legitimacy
signal.

Symmetrically: generate `phish_takeover_mfa_phished` where text reads
as boring ("login, transfer to known recipient") but events show
`recipient_age=newly_added` and a subtle device fingerprint anomaly.
Text alone misses it; events catch it.

Class balance: aim for ~10% of fraud and ~10% of hn to be these
adversarial subtypes — enough that they materially affect the
worst-family FPR.

### Change 4: Per-arm text-field routing (fixes the baseline contract)

Refactor `data/gen/build_dataset.py::serialize_journey` so each example
is stored in a **canonical form** (separate fields for narrative,
event-text-serialization, structured-events, journey/actor wrapper
tokens, verdict footer) rather than a pre-baked monolithic `text`
string. Each trainer then constructs the input text it actually wants:

```python
# Canonical stored form (v4)
{
    "wrapper_open":  "<journey_phish_takeover>\n<actor_human>\n",
    "events_text":   "<events>\nt=0 event=login ip_risk=high ...\n...\n</events>\n",
    "narrative":     "<narrative>\nThe account holder logged in, changed credentials, and made an outbound transfer.\n</narrative>\n",
    "verdict_footer":"<risk_verdict>\nlabel: fraud\n...\n</risk_verdict>\n",
    "wrapper_close": "</journey_phish_takeover>\n",
    "structured_events": [ {...}, {...} ],
    "label": "fraud",
}
```

Each trainer composes its prompt:

```python
# text_only / cpt_light_v4 / xattn:
text = wrapper_open + narrative + verdict_footer + wrapper_close
# structured_as_text:
text = wrapper_open + events_text + narrative + verdict_footer + wrapper_close
# event_only:
# uses structured_events only; no text construction
```

Update each trainer (`src/train/train_lora_text_only.py` →
`train_text_only.py`, `train_structured_as_text.py`, `train_xattn.py`)
to consume the canonical form and construct its specific text. The
side-stream branch of `train_xattn.py` reads `structured_events`
unchanged.

**This is what makes the central comparison clean**: the only
difference between `xattn` and `structured_as_text` in v4 is the
presence/absence of the `events_text` block in the LM's prompt and
the presence/absence of the side-stream encoder path. Everything
else — base checkpoint, narrative, wrapper, footer, label — is
identical.

---

## Minimum viable v4 — what we regenerate

This is the smallest data change that tests the modality gap.

### Must-run minimum (answers Q1 — does the side stream help?)

| Artifact | Action | Cost |
|---|---|---|
| `data/gen/narrative_generator.py` | Modify `_serialize_events_for_prompt` + `SYSTEM_PROMPT` examples per Change 1 | code only |
| `data/gen/journey_templates.py` | Refactor hn family generators per Change 2; add `hn_recovery_high_amount` and `phish_takeover_mfa_phished` per Change 3 | code only |
| `data/gen/build_dataset.py` | Refactor `serialize_journey` to canonical fields per Change 4; add new journey weights | code only |
| `src/train/train_text_only.py` (rename from `train_lora_text_only.py`) | Compose text from canonical form: wrapper + narrative + verdict. No event lines. | code only |
| `src/train/train_xattn.py` | Compose text identical to `train_text_only.py`; events flow ONLY through side stream. | code only |
| `data/train_llm_narrated_v4/` | Regenerate 20-30k pairs at ~$0.005/narrative | ~$100-150 USD |
| `data/eval_fast_5k_v4/` | Regenerate stratified 5k slice | (carved from above) |
| `data/eval_medium_50k_v4/` | Regenerate 50k templated-narrative eval | $0 (template path) |
| `qwen3-8b-cpt-light-v4-merged` | **Re-train** Stage-0 CPT-light on v4 data; re-merge. Single shared base. | ~3.5 H100-hours |
| `text_only_v4` baseline | Train Stage-1 LoRA on text-only from the shared base | ~1.5 H100-hours |
| Leader cross-attn cell (`every_8 / 64 / small_0.01 / r=16`) | Train on v4 from the shared base | ~1 H100-hour |
| Bootstrap CIs, metrics, predictions, gate_trajectory | Auto-produced by launcher | $0 |

**Must-run total: ~$150 USD + ~6-7 GPU-hours.** Roughly half a GPU day.

### Optional extensions (answer Q2 + characterize the surface)

Run only if Q1 came back positive (i.e., `xattn_v4` beats `text_only_v4`
outside CIs) AND budget remains. None of these are needed to answer
the primary architectural question.

| Artifact | Action | Cost | Purpose |
|---|---|---|---|
| `src/train/train_structured_as_text.py` | Compose text: wrapper + `<events>` block + narrative + verdict | code only | |
| `structured_as_text_v4` baseline | Train Stage-1 LoRA from the shared base | ~1.5 H100-hours | Answers Q2: is cross-attn better than prompt serialization? |
| `event_only_v4` baseline | Train small_transformer + classifier head | ~1 H100-hour | Characterizes the surface: how much fraud signal lives in events alone? |

**Optional total: ~2.5 additional GPU-hours.** Still fits the same day.

(`lora_text_v4` is dropped — it required a separate raw-Qwen base and
isn't apples-to-apples with the rest.)

---

## Stages of validation

### Phase 0 (low-risk, do first): Medium-eval confirmation on v3 data

Before regenerating anything, run the v3 leader cell + v3
`structured_as_text` baseline on the existing 50k templated medium
eval (`data/eval_medium_50k_llm/` already exists). Bootstrap CI on
50k will be ~3× tighter than on 5k. Outcomes:

- **Likely**: confirms the v3 tie at tight CI → strengthens the
  motivation for v4 with no ambiguity.
- **Unlikely**: 50k surfaces a small effect 5k missed → revisit
  before regenerating.

Cost: ~30 min (no training; eval-only pass).

### Phase 1: Implement v4 generators

Modify `narrative_generator.py` + `journey_templates.py` per Changes
1-3. Run `data/gen/build_dataset.py --n 100 --out data/samples/v4_smoke`.
Manually inspect 10 generated examples to confirm:

- Narratives no longer contain `<amount_bucket=high>`-style strings or
  obvious paraphrases ("medium-value", "previously-unseen", "freshly-
  added").
- Hn samples show varied feature signatures (not all `mfa_strong`).
- The new adversarial subtypes look plausible.

Update `eval/leakage_checks.py` to flag *new* leakage patterns (a
narrator that says "medium-value transaction" is partially leaking).

### Phase 2: Regenerate v4 dataset

Run `build_dataset.py` for the full 20-30k LLM-narrated set + carve a
5k stratified fast eval + a 50k templated medium eval. Inspect the
generated data card for class balance + leakage scan results.

### Phase 3: Re-train Stage-0 CPT-light + merge

`scripts/merge_stage0_lora.py` produces `qwen3-8b-cpt-light-merged-v4`.

### Phase 4 — Must-run: train `text_only_v4` baseline

Train Stage-1 LoRA from `qwen3-8b-cpt-light-v4-merged` on the canonical
narrative-only text input. Same hyperparameters as the v3 lora_text
baseline (LoRA r=16, lr=1e-4, 1500 steps) so we're holding training
configuration constant.

**Decision gate 1 — did the data pivot work?** Compare `text_only_v4`
hn_worst to v3's `cpt_light_v3`:

- If `text_only_v4` is **noticeably worse** than v3's `cpt_light` (>
  0.005 absolute, ideally outside CIs), the text-only model lost its
  feature-token shortcut. Change 1 worked. **Proceed to Phase 5.**
- If `text_only_v4` is the same as v3 `cpt_light`, the narrator is
  still leaking. Go back to Phase 1 with a stricter narrator prompt
  (block more paraphrase phrases).

### Phase 5 — Must-run: train the leader cross-attn cell on v4

Same dial as `round1_002`: `every_8 / slots=64 / gate=small_0.01 /
small_transformer / lora_r=16`, 1500 steps, seq 2048. Same shared
base. Same canonical narrative-only text input as `text_only_v4`.
Side stream gets `structured_events`.

**The load-bearing primary comparison (Q1)**: `xattn_v4` vs
`text_only_v4`. Both have the same base, the same narrative, the
same wrapper, the same verdict footer. The *only* difference is
whether the LM has access to the structured event stream via cross-
attention.

- **Pass A (Q1 positive — side-stream signal helps)**: `xattn_v4`
  beats `text_only_v4` outside CIs. The side stream carried signal
  the narrative alone didn't have, and the cross-attention pathway
  surfaced it. This is the architectural win.
- **Pass B (gates opened)**: `max_gate >= 0.05` even if hn_worst is
  CI-tied. The LM *did* find the side stream useful even if it didn't
  translate to a metric improvement on this eval — a softer but still
  meaningful signal. Worth investigating further (probably with the
  optional Q2 comparison).
- **Fail**: gates stay near init AND `xattn_v4` is CI-tied with
  `text_only_v4`. Either the v4 data still doesn't have a real
  modality gap, or cross-attn genuinely doesn't help on this surface.

### Phase 6 — Optional: answer Q2 (is cross-attn better than prompt serialization?)

**Skip if Phase 5 was a Fail.** If `xattn_v4` couldn't beat
`text_only_v4`, comparing it to `structured_as_text_v4` is moot.

Run only if Phase 5 was Pass A or Pass B AND budget remains. Train
`structured_as_text_v4` (Stage-1 LoRA from the same shared base, but
with the `<events>` block prepended to the prompt). Then compare:

- **`xattn_v4` vs `structured_as_text_v4`**: same base, same narrative,
  same access to event information — only difference is routing
  (side-stream vs prompt-serialized).
- **`xattn_v4` better than `structured_as_text_v4`**: cross-attn's
  side-stream routing is better than the serialization workaround.
  Architectural win at the routing level.
- **`xattn_v4` tied with `structured_as_text_v4`**: events carry
  signal, but the routing doesn't matter — putting events in the
  prompt is just as good (and a lot simpler) than building the
  cross-attention surgery.
- **`structured_as_text_v4` beats `xattn_v4`**: the serialization
  beats cross-attn on this surface. Honest negative result for the
  architecture.

### Phase 7 — Optional: characterize with `event_only_v4`

Run only if Phases 5 and 6 surfaced unexpected results and we want
to understand "how much of the fraud signal lives in events alone."
`event_only_v4` (small_transformer + classifier head, no LM) gives
the ceiling on event-only signal. Useful for the writeup, not for
the architectural decision.

---

## Critical files to modify (and reuse)

### Modify
| File | Change |
|---|---|
| `data/gen/narrative_generator.py` | `_serialize_events_for_prompt` (Change 1); update SYSTEM_PROMPT exemplars |
| `data/gen/journey_templates.py` | Refactor hn + fraud families to stochastic feature draws (Change 2); add 2 adversarial subtypes (Change 3) |
| `data/gen/build_dataset.py` | Refactor `serialize_journey` to emit canonical fields (Change 4); add new journey weights for adversarial subtypes |
| `src/train/train_lora_text_only.py` → **rename to** `train_text_only.py` | Compose text from canonical form: wrapper + narrative + verdict. Starts from `qwen3-8b-cpt-light-v4-merged` (no longer raw Qwen). |
| `src/train/train_structured_as_text.py` | Compose text: wrapper + `<events>` block + narrative + verdict. Same shared base. |
| `src/train/train_xattn.py` | Compose text identical to `train_text_only.py`. Events flow only via side stream. |
| `eval/leakage_checks.py` | Add narrator-leakage-paraphrase scan (catches "high-value", "previously-unseen", "freshly-added") |
| `data/cards/dataset_card.md` | v4 section documenting the new distribution + design rationale + the explicit baseline/checkpoint contract |

### Reuse without change
| File | Why |
|---|---|
| `data/gen/feature_bucketer.py` | Bucket definitions stay; only their *visibility per stream* changes |
| `data/gen/cheap_template_generator.py` | Template path already doesn't include bucketed tokens — no change needed |
| `src/model/*.py` | The cross-attn architecture is fine; this is a data fix, not a model fix |
| `scripts/run_next_experiment.py` | F8 v2 launcher protections stay |
| `eval/bootstrap_ci.py`, `eval/eval_modes.py` | No metric changes; same headline metric, same modes |
| `src/auto_research/AGENT_INSTRUCTIONS.md` | The auto-loop can re-run unchanged |

### New
| File | Purpose |
|---|---|
| `docs/data-v4-design.md` | Design rationale: why v4 differs from v3 (the redundancy bug, the baseline-contract muddle), what we expect to learn, what falsifies the pivot |
| `tools/print_baseline_contract.py` | Verification utility: iterates over each Stage-1 trainer's data-loader and prints the constructed text for 3 random examples. Used to confirm the apples-to-apples contract before training (see Verification §3). |
| `data/eval_medium_50k_v4/` | Carved at Phase 2 |

---

## Verification — how we'll know it worked

End-state checklist (must-run minimum):

1. **Narrative scan**: a regex over v4 narratives finds zero verbatim
   `<bucket_X=Y>` tokens AND fewer than 5% containing the
   value-paraphrase regex (`(high|medium|low)-value`, `previously-
   unseen`, `freshly-added`, `mfa`, `unfamiliar`).
2. **Hn feature stochasticity**: `hn_account_recovery` samples show
   `mfa_strong` proportion in [0.45, 0.65] (not 1.0 as in v3); fraud
   samples show non-zero `mfa_strong` proportion. Computed from a
   manifest pass over the 20-30k v4 set.
3. **Baseline contract verified**: `tools/print_baseline_contract.py`
   (new) prints the text constructed by `text_only_v4` and `xattn_v4`
   trainers on 3 random examples. The two outputs must be
   **byte-identical**. Both arms must report
   `qwen3-8b-cpt-light-v4-merged` as their base checkpoint.
4. **Decision Gate 1 (data pivot worked)**: `text_only_v4` hn_worst
   is > 0.005 worse than v3's `cpt_light` — the text-only model lost
   its feature shortcut.
5. **Primary architectural test (Q1)**: `xattn_v4` vs `text_only_v4` —
   one of:
   - Pass A: `xattn_v4` beats `text_only_v4` outside CIs (side stream
     carries signal AND cross-attn surfaces it).
   - Pass B: `max_gate >= 0.05` (LM found the side stream useful even
     if hn_worst is tied — softer win, motivates Q2).
   - Fail: tied and gates closed (architectural pathway doesn't help
     on this surface; honest negative result).
6. **Updated README + experiments-log**: v4 phase appended; the v3
   verdict reframed as "v3 found null because v3 confounded data and
   architecture; v4 isolates the architectural variable."

Optional checklist additions (if Phase 6 runs):

7. **Secondary architectural test (Q2)**: `xattn_v4` vs
   `structured_as_text_v4` — answers whether side-stream routing
   beats prompt serialization when both have access to events.
8. **Surface characterization (Phase 7)**: `event_only_v4` reports
   the events-alone ceiling.

Smoke tests:

```bash
# Layer A — local scaffold smoke (no GPU, no API spend):
python3 data/gen/build_dataset.py --n 50 --out data/samples/v4_smoke --narrator-mode template
python3 eval/leakage_checks.py --dataset data/samples/v4_smoke --paraphrase-scan

# Layer B — pod, narrator regen (small dose first):
python3 data/gen/build_dataset.py --n 500 --out data/samples/v4_llm_smoke \
  --narrator-mode llm --usd-budget 5

# Layer C — full pipeline (the actual work):
python3 data/gen/build_dataset.py --n 25000 --out data/train_llm_narrated_v4 \
  --narrator-mode llm --usd-budget 150
accelerate launch src/train/train_cpt_light.py \
  --data data/train_llm_narrated_v4 --out checkpoints/qwen3-8b-cpt-light-v4
# ... then merge, baselines, leader cell, etc.
```

---

## Cost & risk

**Cost**:
- **Must-run minimum** (data regen + Stage-0 + `text_only_v4` + `xattn_v4`):
  ~$150 USD narrator API spend + **~6-7 GPU-hours** on one H100.
- **Optional extensions** (`structured_as_text_v4` for Q2,
  `event_only_v4` for surface characterization): **+~2.5 GPU-hours**.
- Total if everything runs: ~$150 + ~9-10 GPU-hours. Still fits one
  GPU day.

The must-run minimum is what answers the primary question
(*does the side stream help?*). Everything else is optional and only
worth running if the must-run produces a result that motivates further
investigation.

**Risks**:

| Risk | Mitigation |
|---|---|
| Narrator (Change 1) still leaks structured signal via context understanding (e.g., it sees `event=txn` followed by `event=device_add` and infers "high-value risk"). | The paraphrase-leakage scan in `eval/leakage_checks.py` catches the obvious patterns. If it misses, the Phase 4 baseline-degradation check is the safety net — if `lora_text_v4` doesn't degrade, the leakage is too strong and we iterate. |
| Removing feature tokens from the narrator makes narratives unnaturally vague, hurting Stage-0 CPT-light. | Acceptable — the narrator should still describe the SEQUENCE of events; only the QUANTITATIVE buckets are removed. If CPT-light convergence collapses, we add back a few high-level qualifiers (e.g., "transfer" without value, "device change" without age) that don't carry bucket information. |
| Hard-negative stochasticity (Change 2) accidentally makes the task too easy (some hn samples now look very fraud-y to a naive classifier) or too hard. | Phase 4's baseline check catches both — `event_only_v4` should still be strong (events carry signal) but not perfect; `lora_text_v4` should be weaker than v3 but not random. If the spread is wrong, tune the choice probabilities in Change 2. |
| Adversarial subtypes (Change 3) need careful narrative crafting — if the narrator can't write them without label leakage, we have a generation-quality problem. | Manual review of 50 generated samples at Phase 1 smoke. If quality is poor, we can implement adversarial subtypes via deterministic templates first and only use LLM narration on the conventional families. |
| Total cost overshoots ($150 → $300+) | Hard budget cap in `narrative_generator.CostTracker` is already implemented; the build_dataset.py call has a `--usd-budget` parameter. We can also stage: regenerate 5k first, validate the pipeline, then commit to the remaining 20k. |
| Re-trained baselines don't show degradation → modality gap wasn't restored | Iterate Phase 1 with a stronger paraphrase suppressor in the narrator prompt (e.g., add the bucket-paraphrase regex to the SYSTEM_PROMPT's banned phrases). |
| Gates open on v4 — great! — but only one cell. Could be noise. | Phase 6's mini-sweep checks for consistency across at least 5 cells before declaring "cross-attn now works." Same CI-strict rule as before. |

**Unknown**: whether the v4 redesign actually changes the answer.
That's exactly the question this plan tests. The 3-day POC's verdict
was "cross-attention provides no lift on this surface." v4 restores
the surface cross-attention was designed for. If gates still don't
open on v4, that's a much stronger statement about cross-attention's
practical value — and a much cleaner result to publish than v3's
"the synthetic data had a redundancy bug."

---

## Out of scope for this plan

- Switching the side-stream encoder (small_transformer → FT-Tx or
  CNN+LSTM). The encoder stays; we want to isolate the data effect.
- Different base model (Qwen3-8B stays).
- Multi-GPU / FSDP / scaling.
- Real PayPal data. Still synthetic.
- F8 v3 (`setsid -w` wrapper). The v3 sweep is closed; F8 v2 is
  adequate for the much smaller v4 sweep that fits inside fewer
  cron sessions.
