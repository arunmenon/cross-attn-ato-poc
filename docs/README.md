# docs/

Per-batch (or per-milestone) documentation. Each file documents what
landed in a specific commit range, why specific design decisions were
made, and what to focus a review on.

These docs are written by the Maintainer at the end of a batch *before*
asking Codex for a review. Codex reads the relevant `docs/batch-N-*.md`
alongside `PLAN.md` and the git diff to give a grounded review.

## Index

### Running references (read these for current state)

| File | Covers |
|---|---|
| `docs/experiments-log.md` | **Running history of every experiment**, with hypothesis, dials moved, result, and what was learned. Append-only; updated as new arms land. |
| `docs/auto-research-loop.md` | **Plain-language walkthrough of the auto-research loop**: cron → agent_tick → claude → launcher → trainer. Explains the agent-proposes-launcher-enforces design, the shared state surface, halt conditions, and the F-numbered fix lineage. |
| `docs/cross-attention-mechanism.md` | **Architecture, hypothesis, and alternatives** for the cross-attention pathway. The Flamingo-style gated x-attn block, the Perceiver-Resampler, the side-stream encoder choices (small_transformer vs FT-Tx vs CNN+LSTM), the gate-init math, the parameter budget, and what the sweep so far says about the hypothesis. |

### Per-day results

| File | Covers |
|---|---|
| `docs/day-1-results.md` | Day-1 vertical slice + Stage-0 CPT-light + scale + 3-mode eval |
| `docs/day-2-data-diagnostic.md` | Day-2 data-side investigation (per-family generator audit) |
| `docs/day-2-results.md` | Day-2 architecture surgery + four baselines + first sweep batch |

### Implementation batches (historical, frozen at end of each batch)

| File | Covers | Closes review at |
|---|---|---|
| `docs/batch-1-tokenizer-foundations.md` | Path A Batch 1 (custom_tokens, fencer, feature_bucketer, pii_fencer) | *(retroactive: review 001 finding 1 partial-close)* |
| `docs/batch-2-data-generators.md` | Path A Batch 2 (types, journey_templates, agent_actor_mixer, narrative_generator, cheap_template_generator, build_dataset) | *(to be set by review 004)* |
| `docs/batch-3-architecture.md` | Path A Batch 3 — model surgery (cross_attn_block, resampler, qwen_xattn_wrapper, small_transformer) | *(planned)* |
| `docs/batch-4-trainers.md` | Path A Batch 4 — five trainers (cpt_light, lora_text, structured_as_text, event_only, xattn) | *(planned)* |

## Format

Each `batch-N-*.md` should answer, in this order:

1. **What landed** — bullet list of files + one-line each.
2. **Design decisions** — non-obvious choices, with rationale.
3. **Where this maps to PLAN.md** — section references.
4. **Smoke / self-test results** — what was verified locally, and the
   exact commands.
5. **Known limitations** — things explicitly deferred or trade-offs
   accepted.
6. **Focus areas for review** — concrete prompts for the next reviewer
   ("does X handle Y correctly?"). These become Codex's checklist.
