# Review 021: v4 Phase 1 + Phase 2 — data pivot and trainer wiring

**Date prepared:** 2026-05-19
**Repo root:** /Users/arunmenon/projects/Foundation-Science/cross_attn_ato_poc
**Diff base:** `35db0eb~1` (i.e., everything up to but not including the first v4 commit)
**Diff head:** `330e150`
**Net change:** 8 files modified (incl. 1 rename), 1143 insertions, 249 deletions
**No GPU spent. No API spent. No data regenerated.** All work is code + scaffolding.

This file is **maintainer-supplied context** for the next reviewer pass.
The reviewer should still follow `review/CODEX_PROMPT.md` and produce
`comments.txt` per the standard schema. This document just lays out what
to focus on so the review hits the load-bearing claims rather than
trivia.

---

## Why this work exists

The v3 sweep (closed 2026-05-18) ran 18 valid cross-attention arms across
4 dial families and found no detectable lift over the `structured_as_text`
baseline. Post-hoc code audit during planning surfaced **two distinct
root causes** for that null result:

1. **Narrator-mediated redundancy.** The v3 narrator received the full
   structured event stream (with bucket key=value pairs like
   `amount_bucket=high`) in its user-message and the SYSTEM_PROMPT's
   compliant-example phrases ("high-value transfer", "previously-unseen
   device", "freshly-added recipient") effectively taught it to
   paraphrase those buckets into the narrative. The text-side of every
   LM arm was already reading the structured signal; cross-attention
   had nothing unique to fetch.

2. **Label-deterministic hard-negative skeletons.** Every
   `hn_account_recovery` sample had a fixed feature signature
   `{mfa_strong, device_age=known, ip_risk=low, geo_distance=local}`;
   every `phish_takeover` had the opposite. A feature-level classifier
   could perfectly separate them; the 0.052-0.061 worst-family FPR
   ceiling came from a small number of fraud samples bleeding into the
   hn feature region under the FPR=1% threshold, NOT from genuine
   ambiguity.

A third issue surfaced mid-planning (the **baseline/checkpoint
muddle**): v3 arms started from different base checkpoints (raw Qwen vs
cpt_light_merged), AND the text field in every LM arm included event
lines wrapped in `<journey_X>` tokens, so even "structured_as_text vs
xattn" was not a clean architecture comparison.

The v4 plan (`.claude/tasks/data-v4-pivot-plan.md`) addresses all three
with four coordinated changes plus trainer wiring. **The goal is not to
re-run the sweep; it's to make the experimental setup honest** so that
*if* we ever run v4, the result is interpretable.

---

## The commits, in chronological order

```
35db0eb  v4 Change 1: strip bucketed tokens from narrator's view
65a7515  v4 Change 1 hardening: cache version + paraphrase scanner + cheap-template rewrite
88e33e1  v4 leakage scanner: inserted-modifier slot + CLI self-test
6c1d5ff  v4 Changes 2+3: stochastic feature signatures + adversarial subtypes
dd690fa  v4 Change 4: canonical-form refactor + tightened prompt contract
669d56e  v4: rename train_lora_text_only.py → train_text_only.py + docstring
330e150  v4 Phase 2: wire trainers to the canonical-form contract
```

`65a7515` and `88e33e1` are direct responses to two prior code reviews
the user did mid-Phase-1 (cache versioning blocker + inserted-modifier
scanner gap). Those were already round-tripped.

---

## What was actually changed, file by file

### `data/gen/narrative_generator.py` (+119 lines)

- `_serialize_events_for_prompt` now passes ONLY `event` type and `t`
  (plus structural flags `direction`, `prepares_event`) to the LLM
  narrator. The 9 bucket-feature keys (`amount_bucket`, `geo_distance`,
  `ip_risk`, `device_age`, `merchant_risk`, `txn_velocity`,
  `recipient_age`, `session_dwell`, `auth_strength`) are stripped.
- SYSTEM_PROMPT compliant exemplars rewritten to remove
  value-laden adjectives. New rule 5 explicitly bans 40+ paraphrase
  patterns across amount, device, IP/network, recipient, velocity,
  auth, and session-dwell families. Non-compliant examples section
  added so the LLM sees both sides of the contract.
- `NARRATOR_PROMPT_VERSION = 2` introduced; `_journey_cache_key` now
  namespaces by this version. v3 (implicit version 1) and v4 keys
  hash to different values, so v4 generation cannot silently inherit
  v3 cached narratives that contain banned paraphrases.

### `data/gen/cheap_template_generator.py` (+137 lines / -89 deletions)

- Rewrote all 9 v3 journey-family templates to remove every value-laden
  adjective. Each variant audited against the new paraphrase scanner.
- Added 2 new template entries for the v4 adversarial subtypes
  (`hn_recovery_high_amount`, `phish_takeover_mfa_phished`). They share
  the prose pattern of their adversarial counterparts (sim_swap and
  clean respectively) so the narrative does NOT disambiguate; only the
  events do.

### `data/gen/journey_templates.py` (+433 lines / -89 deletions)

- New module-level `_FEATURE_DIST` table mapping
  `(family, phase, feature) → {value: weight}`. Per-feature
  distributions are biased toward the family's characteristic
  signature but with deliberate overlap (e.g., `hn_account_recovery`
  is 30% password_only, 10% high ip_risk; `phish_takeover` is 25%
  mfa_strong = "phished MFA", 25% known device = "compromised
  primary device").
- New helpers `_sample_value`, `_login_features`, `_txn_features`
  consume the distribution table.
- All 9 v3 generators refactored to use the helpers. Hard-coded
  feature values (e.g., `auth="password_only"`, `device_age_value="rare"`)
  are gone.
- Two NEW generators added: `gen_hn_recovery_high_amount` (legit-label
  with sim_swap-style event sequence) and `gen_phish_takeover_mfa_phished`
  (fraud-label with clean-style event sequence). The latter
  deterministically forces at least one transfer to a
  `newly_added` recipient so single-stream models can't dismiss it
  as routine.
- `JOURNEY_GENERATORS` dispatch grows 9 → 11.

### `data/gen/build_dataset.py` (+345 lines / -47 deletions)

- `JOURNEY_WEIGHTS` rebalanced: ~1.5% each for the 2 new adversarial
  families, drawn proportionally from their conventional counterparts.
  Overall fraud / hn / clean split stays ~30/30/40.
- `ACTOR_BY_JOURNEY`, `EVIDENCE_BY_FAMILY`, `CONFIDENCE_BY_FAMILY`
  extended with entries for the 2 new families.
- **`serialize_journey` replaced with canonical-form composition.**
  Old v3 emitted a monolithic text with `<journey_X>` wrappers,
  event lines IN the text, and a verdict footer carrying
  `journey_family`, `confidence`, `evidence`. New v4 emits clean
  per-arm compositions:
  - `compose_text_only(record)` →
    `<case>\n<narrative>...</narrative>\n\n<risk_verdict>\nlabel: X\n</risk_verdict>\n</case>`
  - `compose_structured_as_text(record)` → same but with an `<events>` block
    between `<case>` and `<narrative>`.
- `journey_to_record` now emits canonical fields (`narrative`,
  `structured_events`, `label`) plus a back-compat `text` field set
  to `compose_text_only(record)`.
- `assert_byte_identical_invariant(record)` — idempotence assertion
  on compose_text_only.
- `verify_v4_text_contract(dataset, sample_n, arm_name)` — trainer
  startup smoke that confirms dataset's `text` field equals
  `compose_text_only(row)` for sample rows. Gracefully skips v3-format
  datasets (no canonical fields).

### `eval/leakage_checks.py` (+225 lines)

- New `_BUCKET_PARAPHRASE_STEMS` regex set covering 7 bucket families
  (amount, device, IP/network, recipient, velocity, auth, session-dwell).
  Each multi-word pattern uses `_GAP = r"\s+(?:\w+\s+){0,2}"` to allow
  0-2 neutral modifiers between adjective and target noun. Catches
  variants like "large outbound transfer", "previously unseen mobile
  device", "high-risk residential network".
- New `paraphrase_leakage_scan(text)` function.
- `narrative_leakage_scan(text, include_paraphrase=True)` now runs both
  the v3 class/actor scan AND the v4 paraphrase scan by default.
  `include_paraphrase=False` retains v3 backward-compat for legacy
  audit callers.
- New `--self-test` CLI entrypoint: 40 fixtures (4 v3-class, 12 v4-direct,
  14 v4-with-inserted-modifier, 10 v4-compliant exemplars). Used as
  a regression gate before any LLM v4 spend.

### `src/train/train_text_only.py` (renamed from `train_lora_text_only.py`, +54 lines)

- File renamed to reflect v4 contract (no longer a "lora" baseline as
  distinct from cpt_light — in v4 it IS the text-only arm).
- Docstring rewritten: explicit v4 contract description, references
  `compose_text_only`, names the byte-identical invariant against
  `train_xattn.py`.
- Default `base_checkpoint` updated from `Qwen/Qwen3-8B` (raw) to
  `/workspace/checkpoints/qwen3-8b-cpt-light-merged` (shared base).
  Per-config override still honored.
- Startup `verify_v4_text_contract(train_ds, sample_n=3,
  arm_name="text_only")` call added.

### `src/train/train_xattn.py` (+9 lines)

- Startup `verify_v4_text_contract(train_ds, sample_n=3,
  arm_name="xattn")` call added.
- No other changes — the trainer already defaulted to the shared CPT-
  light base and read `text` from the dataset (which now equals
  `compose_text_only(row)` in v4 data).

### `src/train/train_structured_as_text.py` (+70 lines)

- Collator rewritten to dispatch on data version:
  - **v4 path** (row has `narrative` + `label` canonical fields):
    `new_ex["text"] = compose_structured_as_text(new_ex)`. Single
    source of truth.
  - **v3 fallback**: `new_ex["text"] = _serialize_events_compact(events)
    + "\n" + new_ex["text"]` (same as v3).
- `_serialize_events_compact` retained as legacy helper with a
  pointer comment to the v4 path.

---

## Things the reviewer should focus on (the load-bearing claims)

### Claim 1: The narrator can no longer paraphrase bucket tokens

Verify by reading:
- `data/gen/narrative_generator.py:_serialize_events_for_prompt`
  (around line 314): does it actually omit all 9 bucket families?
- `narrative_generator.SYSTEM_PROMPT` rule 5 (the banned-adjective
  list) — is the list complete? Are there obvious paraphrases the
  LLM could use that aren't covered?
- `eval/leakage_checks._BUCKET_PARAPHRASE_STEMS` — does the regex
  set match the SYSTEM_PROMPT's banned list 1-to-1? If the prompt
  bans something the scanner doesn't catch, that's a leak.

Specifically: is `_GAP = r"\s+(?:\w+\s+){0,2}"` the right strictness?
Could an LLM dodge it by using a 3-word modifier insertion ("large
outbound international wire transfer")?

### Claim 2: v3 cached narratives cannot leak into v4 generation

Verify by reading:
- `data/gen/narrative_generator.py:NARRATOR_PROMPT_VERSION` =
  current value, comment block.
- `_journey_cache_key` — when does it include `prompt_version`?
- Confirm: any future prompt change must bump
  `NARRATOR_PROMPT_VERSION` to a fresh integer. Is this called out
  loudly enough in the code?

### Claim 3: Hard negatives are no longer label-deterministic

Verify by reading:
- `data/gen/journey_templates.py:_FEATURE_DIST` — read every entry
  for `hn_account_recovery` vs `phish_takeover`. Do they overlap
  meaningfully? E.g., does `hn_account_recovery` have ≥20%
  `password_only`? Does `phish_takeover` have ≥20% `mfa_strong`?
- The smoke output at `data/samples/v4_smoke/data.jsonl`: 100 rows,
  feature distributions per family.

Specifically: is the overlap large enough that an event-only
classifier can't ace the task? Or too large that the task becomes
unlearnable? (No empirical confirmation yet — we haven't trained
event_only_v4.)

### Claim 4: text_only_v4 and xattn_v4 see byte-identical LM text

Verify by reading:
- `data/gen/build_dataset.py:compose_text_only` — is it free of
  arm-specific branching? (Confirm: takes only `narrative` + `label`,
  no `arm` parameter.)
- `data/gen/build_dataset.py:journey_to_record` — does it populate
  `text` field with `compose_text_only(record)`?
- `src/train/train_text_only.py` and `src/train/train_xattn.py` —
  do they both consume the `text` field via the SAME path?
- `verify_v4_text_contract` — does it actually verify per-row
  `text == compose_text_only(row)`?

Smoke result: passes 5/5 rows on the v4 smoke set for both arms.

Specifically: is there ANY way the `text` field could diverge between
text_only and xattn in production? E.g., a collator that mutates the
text in only one arm?

### Claim 5: structured_as_text_v4 prompt is the proper v4 contract

Verify by reading:
- `data/gen/build_dataset.py:compose_structured_as_text` — does it
  produce `<case>\n<events>...</events>\n<narrative>...</narrative>\n\n<risk_verdict>\nlabel: X\n</risk_verdict>\n</case>`?
- `src/train/train_structured_as_text.py` collator — does it actually
  CALL `compose_structured_as_text` on the canonical row, or does it
  use the v3 fallback under any condition v4 data shouldn't hit?

Specifically: the dispatch in the collator is `if "narrative" in new_ex
and "label" in new_ex`. Are there any code paths where v4 data might
arrive WITHOUT these fields (e.g., a partial preprocessing pass)?
Should the dispatch be stricter?

### Claim 6: The 4 self-tests + smoke test are sufficient regression coverage

Each of these passes today; the reviewer should confirm they're
non-trivial:

1. `python3 -m data.gen.journey_templates --self-test` (11 families)
2. `python3 -m data.gen.cheap_template_generator --self-test`
   (66 family × actor combos, each scanned for both v3 and v4 leakage)
3. `python3 -m data.gen.narrative_generator --self-test`
   (cache replay, concurrent batch path)
4. `python3 -m eval.leakage_checks --self-test` (40 fixtures)
5. End-to-end smoke: `python3 -m data.gen.build_dataset --n 100
   --out data/samples/v4_smoke --mode template`
   → produces 100 records, 0 leakage failures, all v4 contract
   invariants verified per the inline check script.

Specifically: are there obvious regressions NOT covered? E.g., does
anything verify that `compose_structured_as_text` matches the format
documented in the v4 plan? (The structured_as_text smoke composition
was printed for one row but not formally asserted as part of a
self-test.)

### Claim 7: v3 datasets still work with v4 trainers

Verify by tracing the dispatch logic:
- `verify_v4_text_contract` skips gracefully on rows lacking
  canonical fields (v3 data).
- `train_structured_as_text` collator falls back to v3 prepend logic
  when canonical fields are absent.

Specifically: is the v3 fallback semantically EQUIVALENT to the v3
trainer's original behavior? Or did the refactor change any
edge-case handling?

---

## Known limitations / open questions

These are things the maintainer is aware of and the reviewer should
not flag as new — but should validate as accurate:

1. **No event_only_v4 trainer changes.** `train_event_only_classifier.py`
   is untouched. v4 event_only would consume `structured_events`
   directly; no text composition is needed. The trainer needs no
   modification, but this hasn't been explicitly verified by smoke.

2. **No retrain of Stage-0 CPT-light on v4 data.** The base model
   `qwen3-8b-cpt-light-merged` currently sits on disk from the v3
   training. v4 would need a fresh Stage-0 retrain on v4 narratives
   (~3.5 GPU-hr). The trainer references the v4 path
   `qwen3-8b-cpt-light-merged` but the checkpoint at that path is
   v3-trained. Running v4 trainers against it would technically work
   but the base model wouldn't have seen v4 narratives. Worth
   flagging as a "before v4 training: bump the path, retrain, point
   the trainers at the new path" pre-condition.

3. **Adversarial subtype distributions are best-guess.** The 1.5%
   class weight and the feature-distribution choices for
   `hn_recovery_high_amount` and `phish_takeover_mfa_phished` have
   no empirical basis. They're the maintainer's prior on "how often
   would an adversarial cross-modal case appear in real data."
   Sensitivity to these choices is unknown.

4. **No GPU training has been run on v4 data.** All verification is
   data-pipeline level. The trainers' compile-ability has not been
   verified end-to-end with v4 data + tokenizer + model. The startup
   `verify_v4_text_contract` call would catch contract violations
   but won't catch tokenizer mismatches, OOM, or other training-time
   issues.

5. **PLAN.md and docs/batch-4-trainers.md still reference the old
   `train_lora_text_only.py` name.** Intentionally not updated — those
   are historical documents of v3 work. The v4 plan
   (`.claude/tasks/data-v4-pivot-plan.md`) is the authoritative
   reference and uses the new name.

6. **The `_GAP` regex tolerates 0-2 modifiers.** An LLM could
   theoretically dodge by inserting 3+ modifiers. The maintainer's
   judgment was that 3+ inserted modifiers in a 2-4 sentence
   narrative is so unusual it would be obvious. Worth a sanity-check
   challenge from the reviewer.

---

## Suggested reviewer protocol

The standard `review/CODEX_PROMPT.md` workflow applies: read the diff
window, write `comments.txt` per the schema in `review/README.md`. This
document just primes the focus.

If you want to run the smoke checks yourself:

```bash
cd /Users/arunmenon/projects/Foundation-Science/cross_attn_ato_poc

# 1. Pure-Python self-tests (no GPU, no API)
python3 -m data.gen.journey_templates --self-test
python3 -m data.gen.cheap_template_generator --self-test
python3 -m data.gen.narrative_generator --self-test
python3 -m eval.leakage_checks --self-test

# 2. End-to-end smoke (writes ~200KB to data/samples/v4_smoke_review/):
python3 -m data.gen.build_dataset --n 100 \
    --out data/samples/v4_smoke_review --mode template --seed 42

# 3. Read the produced rows and verify the contract manually:
head -1 data/samples/v4_smoke_review/data.jsonl | python3 -m json.tool

# 4. (Optional) Trace the byte-identical invariant explicitly:
python3 -c "
import json, sys; sys.path.insert(0, '.')
from data.gen.build_dataset import compose_text_only, compose_structured_as_text
row = json.loads(open('data/samples/v4_smoke_review/data.jsonl').readline())
print('text field matches compose_text_only:', row['text'] == compose_text_only(row))
print('structured_as_text has <events> block:', '<events>' in compose_structured_as_text(row))
print('text_only has NO <events> block:', '<events>' not in compose_text_only(row))
"
```

---

## Maintainer's self-assessment

In one paragraph: the data pipeline and trainer wiring are internally
consistent and pass all self-tests, but the v4 question ("does cross-
attention help when the modality gap is real?") has NOT been answered
empirically by this work — Phase 1+2 is preparation, not evidence.
The riskiest claim is **Claim 1** (narrator can't paraphrase) because
an LLM is not a regex-matcher and could find dodges we haven't
enumerated. The cleanest claim is **Claim 4** (text_only/xattn byte-
identical) because it's enforced by both arms calling the same pure
function on the same input row. The user explicitly said "not sure if
we'll run v4" — so the trainer-side Phase 2 work is somewhat
speculative; defending its cost is part of the rebuttal expected here.
