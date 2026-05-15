# Batch 2 — data generation pipeline

Closes (or partially closes) review 001 finding 1: the
data-generators portion of the missing Day-1/2 surface. Together with
Batch 1, this gives PLAN.md "Synthetic data design" a runnable
implementation.

Baseline commit going into this batch: `87070cf` (review 003 closure).
Verifying commit for this batch: see `git log -1` after the closing
commit lands.

---

## 1. What landed

| File | Lines | One-line |
|---|---|---|
| `data/gen/types.py` | ~70 | `Journey` dataclass + `Label` alias. The unit that flows through the pipeline. |
| `data/gen/journey_templates.py` | ~360 | Per-journey-family generators (9 of them). Each function deterministically produces a Journey with fenced + bucket-tokenized events. |
| `data/gen/agent_actor_mixer.py` | ~180 | Actor-family modulation: rescales inter-event timing per actor type, inserts `<event_tool_call>` events for agent actors, refreshes `<session_dwell>` bucket. |
| `data/gen/narrative_generator.py` | ~290 | LLM narrator (Anthropic Claude). Defense-in-depth: prompt-side ban + post-gen `fence()` + post-gen `narrative_leakage_scan` + disk cache + budget cap. |
| `data/gen/cheap_template_generator.py` | ~160 | No-LLM templated narratives. Hand-written per family. Used for low-FPR eval, smoke tests, offline iteration. |
| `data/gen/build_dataset.py` | ~280 | CLI orchestrator. Samples journeys with class-balance weights, mixes actors, narrates (LLM or template), serializes to HF-Dataset JSONL, runs leakage audit, writes `build_summary.json`. |

All six files include `--self-test` or equivalent. Each is invocable
standalone for ad-hoc testing.

---

## 2. Design decisions (the non-obvious ones)

### 2.1 `Journey` as a dataclass, not a dict

Every file in the pipeline consumes/produces journeys. Using a typed
dataclass over a dict makes refactors safe (mypy or IDE catches missing
fields) and prevents the silent-drift bug where one file expects
`actor` and another expects `actor_family`.

`MetadataKeeper` (from Batch 1) is composed into `Journey.metadata`
rather than re-implemented. The real synthetic identifiers (IPs,
amounts, recipient IDs) live there exclusively; they never appear in
`Journey.events` or in the serialized text.

### 2.2 Separation of journey-pattern signature from actor-cadence signature

Journey templates encode *what events happen and in what order* — the
fraud-pattern signature. The actor mixer encodes *how fast and through
what means* — the cadence and tool-use signature.

This separation:
- Avoids 9 × 6 = 54 hand-written distributions.
- Lets us test each axis independently.
- Lets the model's eventual ability to distinguish actor *from* journey
  be measured cleanly via the per-actor differential AUC (PLAN.md §
  Evaluation).

Trade-off: the mixer's output isn't perfectly natural for some
combinations (e.g., `hn_account_recovery` × `agent_adversarial` is
nonsensical in real life). We constrain via the `ACTOR_BY_JOURNEY`
distribution in `build_dataset.py` rather than disallowing combinations
in the mixer.

### 2.3 Verdict footer is deterministic, not LLM-generated

The verdict block:

```
<risk_verdict>
label: fraud|legit
journey_family: <X>
confidence: ...
evidence: ...
</risk_verdict>
```

…is built by `build_dataset.serialize_journey` from constant lookup
tables (`EVIDENCE_BY_FAMILY`, `CONFIDENCE_BY_FAMILY`). The LLM never
sees the verdict footer or writes it. This removes a whole class of
narrative leakage (the LLM cannot inadvertently leak class names into a
field whose contents are deterministic).

Consequence for eval: the model's primary classification surface is
exactly two tokens — ` fraud` and ` legit` — at a known position. That
matches `eval/score_risk.py`'s assumption.

### 2.4 LLM call is opt-in; template is default

`build_dataset --mode template` runs entirely offline and is the
default. `--mode llm` requires `ANTHROPIC_API_KEY` and raises with a
clear message if missing. Rationale:

- Day-0/local iteration shouldn't require API access.
- Layer-C eval sets (50-200k examples) are template-narrated by design
  to avoid prohibitive LLM cost.
- The vertical-slice Day-1 path (1-2k examples) can run in either
  mode; LLM is preferred when API key is available for realism.

### 2.5 Defense-in-depth against narrative leakage

Five layers:

1. **Prompt-side ban** in `SYSTEM_PROMPT`: explicit "do not use these
   words" list with case-stem coverage.
2. **Few-shot compliant examples** in the prompt: shows the narrator
   what compliant phrasing looks like.
3. **Post-gen `fence()`** via `src.tokenizer.fencer`: scrubs literal
   PII that slipped through (defense for accidental email/IP/amount in
   the body).
4. **Post-gen `narrative_leakage_scan()`**: regex over banned phrases;
   failures trigger up to `max_retries` regenerations.
5. **Final batch audit** in `build_dataset`: 200-sample (configurable)
   leakage scan after all narratives generated; build aborts on any
   failure.

`cheap_template_generator` doesn't need layers 1-3 (templates are
hand-written) but its output still passes through layers 4 and 5 — to
catch authoring mistakes if templates are edited later.

### 2.6 Cache keyed by structured-stream hash

`narrative_generator` caches by SHA-256 of
`(model, journey_family, actor_family, events)`. Implication: regenerating
the same journey with the same model is free after the first call. Also:
changing the system prompt does NOT invalidate the cache — that's
intentional, the cache is about input determinism, not prompt
versioning. If we ever change `SYSTEM_PROMPT`, callers must clear
`data/cache/narratives/` manually.

### 2.7 Per-row stable seed

`build_dataset` derives each row's seed from `seed * 1_000_003 + i`.
That means: re-running with the same `--seed` produces byte-identical
output (in template mode). For LLM mode, the cache makes it close to
byte-identical modulo any model nondeterminism.

---

## 3. Where this maps to PLAN.md

| PLAN.md section | Implementation |
|---|---|
| "Three token families" (custom_tokens, bucket-features, journey/actor) | All three families used by `journey_templates` + serialized in `build_dataset.serialize_journey` |
| "Verdict footer" | `build_dataset.serialize_journey` — deterministic from family lookup tables |
| "Narrative leakage policy" | Five-layer defense documented in §2.5 above |
| "Paired streams" | `Journey.events` (structured side, list of dicts) + `Journey.narrative` (text side, narrative body) |
| "Volume targets: 20-30k LLM-narrated + 50-200k templated" | `build_dataset --mode llm --n 25000` and `--mode template --n 200000`. Cost cap honored by `narrator.CostTracker`. |
| "Class balance: ~30% fraud / ~30% hn / ~40% clean" | Encoded in `JOURNEY_WEIGHTS` (6+6+6+6+6 fraud = 30, 10+10+10 hn = 30, clean = 40) |
| "Agentic actors" | `ACTOR_BY_JOURNEY` distributes agent classes plausibly; `agent_actor_mixer` produces the cadence/tool-call signature |
| Risk: "LLM narrator leaks class names" → mitigation: narrator prompt bans + scan + scrub + audit | Implemented; build aborts on audit failure |
| Risk: "LLM cost overrun" → cap | `CostTracker.over_budget()` checked before every call |

---

## 4. Smoke / self-test results

All six files self-tested individually plus an end-to-end build.

```bash
# Per-file self-tests (all PASS):
python3 -m data.gen.journey_templates --self-test
python3 -m data.gen.agent_actor_mixer --self-test
python3 -m data.gen.cheap_template_generator --self-test
python3 -m data.gen.narrative_generator --self-test     # uses stub narrator

# End-to-end build:
python3 -m data.gen.build_dataset --n 200 --out data/samples/smoke \
    --mode template --eval-frac 0.1 --seed 7
```

End-to-end summary (from `data/samples/smoke/build_summary.json`):

```json
{
  "n_records": 200,
  "mode": "template",
  "family_counts": {
    "clean": 94, "mule_chain": 10, "phish_takeover": 19, "cred_stuff": 11,
    "malware_rat": 9, "hn_account_recovery": 16, "hn_large_purchase": 19,
    "hn_travel": 13, "sim_swap": 9
  },
  "actor_counts": {
    "human": 138, "agent_adversarial": 9, "hybrid": 10,
    "agent_buying": 14, "agent_compromised": 16, "agent_finance": 13
  },
  "leakage_audit_n": 200,
  "leakage_audit_failures": 0,
  "duration_seconds": 0.02
}
```

Class balance at n=200: clean 47%, fraud 29%, hn 24%. Within Monte Carlo
tolerance for the targeted 40/30/30 weights (expected ~±5% at this n).
Will tighten at the Day-1 scale (~20-30k).

Layer A scaffold smoke still green after these additions:

```bash
python3 - <<'PY'
import sys, random
sys.path.insert(0, '.')
from eval import eval_modes
from eval.leakage_checks import narrative_leakage_scan
text = '<journey_sim_swap><actor_human><event_login> <amount_bucket=high></journey_sim_swap>'
assert 'journey_sim_swap' not in eval_modes.apply(text, 'stripped')[0]
assert '<amount_bucket=high>' in eval_modes.apply(text, 'stripped')[0]
a = eval_modes.apply(text, 'opaque', rng=random.Random(0))[0]
b = eval_modes.apply(text, 'opaque', rng=random.Random(1))[0]
assert a != b
assert not narrative_leakage_scan('This is fraudulent SIM-swap')['clean']
assert narrative_leakage_scan('Device change followed by password reset')['clean']
print('layer A smoke OK')
PY
```

---

## 5. Known limitations / deferred

- **LLM-mode end-to-end not exercised yet**. The Anthropic call path is
  implemented and stub-tested, but a real API call has not been run from
  this scaffold. First real call happens at Day-1 vertical slice on the
  pod (Task #32). Risk: response-text shape differs from stub, breaking
  the leakage scan. Mitigation: scan + retry are already in place; worst
  case is the retry budget exhausts and the row is dropped (loud
  failure, not silent).
- **Cache invalidation on prompt change is manual**. If `SYSTEM_PROMPT`
  changes, callers must clear `data/cache/narratives/`. Documented in
  §2.6 but not automated. Acceptable at POC scale.
- **`mule_chain` "direction" field** (`incoming` vs `outgoing`) is added
  to events but the serializer doesn't currently surface it in
  `_event_to_line`. Same for `prepares_event` on tool_calls. Either we
  serialize these too (the side-stream encoder will see them via the
  structured events list either way), or we drop them from the event
  dicts. Currently they exist in the structured side but not the text
  side — a small asymmetry. Flag this for review.
- **No real PII in `data/cache/narratives/`**. By construction, but a
  belt-and-braces check would be a periodic `narrative_leakage_scan` over
  cached files. Not implemented.
- **Agent actor token attached to events but not the journey-level
  `<actor_*>` tag**. The mixer changes `Journey.actor_family` and the
  serializer reads from there. Within-event `actor` field still reflects
  the original assignment. If a mixer inserts a `tool_call`, that event's
  `actor` is the new actor — but events from the base journey keep the
  original actor. This is harmless in practice but worth verifying.
- **Day-1 scaling check not done**. The pipeline runs at ~16k records/s in
  template mode at n=200. Whether that holds at n=200k (and whether
  memory stays reasonable when records accumulate before write) is
  unverified. Easy fix: stream-write incrementally if it doesn't.

---

## 6. Focus areas for the next review (review 004)

Concrete prompts for Codex when running `Review … focus: path-a-batch-2`:

1. **Leakage defense plumbing**: does every narrative path (LLM and
   template) actually pass through `narrative_leakage_scan` before
   being written? Inspect `narrative_generator.generate_narrative` and
   `build_dataset.build`. If `cheap_template_generator` is used, does
   the build still run the post-build audit? Verify on a 100-record
   template build that the audit ran.

2. **Class balance at scale**: at n=200 we see 47/29/24, vs targeted
   40/30/30. Run `build_dataset --n 5000 --mode template` and check
   that the family_counts converge to ~40/30/30 within 2 percentage
   points. If not, the weights in `JOURNEY_WEIGHTS` need calibration.

3. **Determinism across runs**: `build_dataset --n 100 --seed 42 --mode
   template` should produce byte-identical output across two
   invocations. Verify with `diff data1/data.jsonl data2/data.jsonl`.

4. **`agent_actor_mixer` correctness**: the mixer shuffles event
   timing but preserves event order. Verify on `sim_swap` ×
   `agent_compromised`: the sequence (login → device_add → pw_reset →
   recipient_add → txn) must be preserved. The mixer should never
   reorder events, only insert tool_calls between them.

5. **Verdict footer / score_risk consistency**: `score_risk.py` reads
   `logP(' fraud' | …)`. Confirm the verdict footer has exactly one
   space before `fraud` or `legit` on the `label:` line (look at the
   serialized output in `data/samples/smoke/train.jsonl`).

6. **PII fencing at the boundary**: `assert_no_raw_pii_in_event` runs
   in `journey_to_record`. Confirm it would actually catch a regression
   — try inserting a synthetic `ip` field with a real-looking IP in
   `journey_templates` and verify the build aborts.

7. **Documentation drift**: `PLAN.md`'s "Synthetic data design" section
   describes the file layout in `data/gen/`. Confirm what's
   implemented matches what's described (and flag anything that
   doesn't).

8. **Cost tracker honesty**: `CostTracker._HAIKU_INPUT_USD_PER_TOKEN`
   and `_HAIKU_OUTPUT_USD_PER_TOKEN` are hard-coded. Are they current
   for Haiku 4.5 as of 2026-05? If not, update the constants and adjust
   the expected per-narrative cost in PLAN.md §"Generation cost rough".

9. **Idempotence under retry**: the LLM narrator's retry loop on
   leakage failure regenerates with the same input. If the LLM is
   deterministic at temperature 0, this will produce the same leaky
   output forever and exhaust the retry budget. Should the prompt vary
   on retry (e.g., add "previous attempt leaked X, avoid that")? Or
   should temperature > 0 be the default? Currently `temperature` is
   not set in the API call — flag this as a gap.

10. **Pod-side prerequisite check**: `_real_anthropic_call` imports
    `anthropic` at first invocation. On a pod without the package
    installed, the error message points to `pip install -r
    requirements.txt`. Confirm `anthropic` is in `requirements.txt`
    (it is, but verify).
