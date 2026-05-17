# Cross-Attention for PayPal ATO — Journey Log

3-day POC. The journey matters more than the result.

Plan: `PLAN.md` (v3).
Runbook: `RUNBOOK.md`.

---

## Day 0 — Pre-flight (this directory, before GPU)

*Filled in by the human at end of Day 0.*

---

## Day 1 — Foundation: data + CPT-light baseline

**Status: COMPLETE.** Baseline #1 (`qwen3-8b-cpt-light-merged`) trained, merged, and evaluated. Pipeline plumbing proven end-to-end. **Key finding: synthetic data saturates AUC; Day 2-3 must pivot to hard-negative FPR as the comparison metric.**

**Durable evidence:** every numeric claim in this section is backed by JSON excerpts in [`docs/day-1-results.md`](docs/day-1-results.md). The trainer outputs (`src/auto_research/runs/*/`) are gitignored on purpose (runtime artifacts on the pod), so `docs/day-1-results.md` checks in the verbatim score/CI/metrics content needed to audit this writeup from a fresh clone (review 012 finding #1).

### Hr 0-2: Tokens + fencer + bucketer

Code was already shipped in Path A batches 1-2 (commits before Day 0). Task #31 marked complete pre-pod. Pod-side install of the 65-token registry, PII fencer, and bucketed-feature derivation all worked first try after the bnb 0.45+ / transformers 4.51+ / tokenizers 0.21+ pin alignment (review 010 + Day-0 boot fixes).

### Hr 2-6: Vertical slice (Task #32 — make-or-break)

**Result: PASS, all 6 gates green.** 100-step CPT-light smoke on the full 20k train set + 5k LLM-narrated eval × 3 modes + bootstrap CI + leakage scan. Wall time 8.6 min on Blackwell. Final loss 5.65 → 2.62.

**Five integration bugs surfaced during the slice — exactly what the vertical slice exists for.** Every one was a small fix, but each would have wasted ~hours of GPU if discovered later. Captured for future-me:

| # | Bug | Fix commit | Where it would have bitten next |
|---|---|---|---|
| 1 | `transformers==4.46.3` doesn't recognize `qwen3` model type (Qwen3 support landed in 4.51) | `4bf30f3` | Stage-0 CPT-light first model load — would have wasted the H100 boot time |
| 2 | `bitsandbytes==0.45.5` ships no `libbitsandbytes_cuda130.so` for the Blackwell torch 2.12 / CUDA 13 image; silent CPU fallback for the paged optimizer | `4bf30f3` (range bumped to `<0.50`) | 2× memory + 10× slow optimizer step → would have looked like a model bug |
| 3 | `Dataset.from_json` rejects heterogeneous list-of-dict (event types have different keys); silent `DatasetGenerationError` | `3218f56` (serialize `structured_events` as JSON string on load) | First dataset load in any trainer |
| 4 | `load_paired_dataset` had no branch for "eval.jsonl only" directory layout (the `eval_fast_5k` symlink) | `2043d78` (third branch added) | Any trainer that reads `eval_fast_path` |
| 5 | PyYAML 6.x parses `5e-5` / `1e-4` as strings (no decimal point); bitsandbytes optimizer init then crashed with `'<=' not supported between float and str` | `43e3c6c` (coerce scientific-notation in `load_config`) | Every YAML-driven trainer launch |

The vertical slice paid for itself in surfaced bugs alone. Total slice cost: 4 attempts before all gates green; each rerun was minutes because Qwen3 weights and event cache survived the failures.

**Headline numbers (5k LLM-narrated eval, stripped mode):**
- AUC = 0.9929, 95% CI [0.9916, 0.9945]
- R@FPR=1% = 0.7432
- Hard-neg FPR @ 1%: hn_large_purchase = 5.6% (the model's weak spot at 100 steps); others 0%
- Leakage scan: **0/25,000 narrative leakage failures.**

### Hr 6-10: Scale + Stage-0 CPT-light + merge (Task #33)

**Result: PASS.** 1500 steps on the full 20k train, effective batch 32, paged_adamw_8bit on Blackwell SM_120. Wall time **53 min** on RTX PRO 6000 Blackwell (faster than the PLAN.md H100 time budget; *not* a controlled H100 vs Blackwell comparison — different GPU class, different host, single run). Final loss 1.2425. The 53-min figure is `metrics.json:wall_clock_sec` and includes training + adapter save; the merge step (~3 min) ran separately via `scripts/merge_stage0_lora.py`.

Adapter merged via `scripts/merge_stage0_lora.py` into `/workspace/checkpoints/qwen3-8b-cpt-light-merged/` (16 GB across 4 safetensor shards). This serves as both:
1. **Baseline #1** for the Day-3 comparison
2. **Starting checkpoint** for Stage-1 cross-attention

**Stage-1 architecture pre-flight (`scripts/preflight_xattn.py`) ran green:** wrapper construction + LoRA-on-Q attach + dummy forward pass on the merged checkpoint completed in ~2 min, 214M trainable (matches PLAN.md's "~200-400M Stage-1" target). The pre-flight caught one real bug on first run (bf16 dtype cast — fixed in the script itself). Now wired into RUNBOOK Day 2 Step 0 (review 011 finding #5).

### Hr 10-14: CPT-light 3-mode eval + writeup

**Result: saturation discovered.** PLAN.md anticipated this in §Risks ("Synthetic-data saturation — all baselines AUC ceiling") and recommended the mitigations now applied. The narrower assumption that *did* fail: PLAN.md said templated narratives "distribution-shape match training" so the 50k templated medium eval could serve as a cheaper substitute for an LLM-narrated medium eval. The diagnostic below falsifies that specific premise (templated is materially *easier* than LLM-narrated, by a non-overlapping CI margin).

Stage-0 evaluated on the 5k LLM-narrated fast eval × 3 modes (stripped / opaque / full):

| Mode | AUC | 95% CI | R@FPR=1% | R@FPR=0.1% |
|---|---|---|---|---|
| stripped (headline) | **1.0000** | [1.0, 1.0] | 1.0 | 1.0 |
| opaque | **1.0000** | [1.0, 1.0] | 1.0 | 1.0 |
| full | **1.0000** | [1.0, 1.0] | 1.0 | 1.0 |

Every fraud case scored above every legit case in 5,000 records. Bootstrap CI degenerate.

**Two diagnostics confirmed saturation is fundamental, not eval-set-specific:**

1. **vs 50k templated eval** (vertical-slice diagnostic, 100-step adapter): AUC 1.0000 across all 3 modes; templated narratives are deterministic enough that even the under-trained smoke adapter ceilinged. CIs do not overlap with the LLM-narrated 5k eval CIs at the smoke checkpoint — i.e., templated is statistically easier than LLM-narrated. This is the specific PLAN.md sub-premise that failed (cf. the section intro above).

2. **vs 15k LLM-narrated eval** (seed=42, different journeys than train; salvaged from a rate-limited 50k gen attempt — see Day-1 friction log): Stage-0 AUC **still 1.0000** across all 3 modes. Bigger eval, different journeys, same model — no change. Saturation is not a sample-size artifact.

**Hard-negative FPR is the only metric still moving** (stripped mode @ 1% FPR):

| Family | smoke (100 steps) | Stage-0 (1500 steps, 5k LLM) | Stage-0 (1500 steps, 15k LLM) |
|---|---|---|---|
| hn_account_recovery | 0.0% | 6.15% | **2.22%** |
| hn_large_purchase | 5.65% | **1.01%** | 1.16% |
| hn_travel | 0.0% | 0.0% | 0.0% |

`hn_account_recovery` and `hn_large_purchase` retain 1-6 percentage points of room and are **sensitive to model state** (smoke and Stage-0 produce different trade-off curves). These are the Day-3 comparison surfaces.

### Day-1 friction log

**Slowest cost: OpenAI rate limits, not GPU.** The 25k LLM-narrated train set generated cleanly in 63 min at concurrency=8 (`a39cc5f` patch). The follow-on 50k LLM eval (different seed, different temp) hit OpenAI rate limits hard — concurrency=8 stalled at 5,000/50,000 in 2.5h; concurrency=4 restart did 7,500 cache-replay + then dropped to 14/min for 7h. Salvaged the 15k records that completed and pivoted to "eval is 15k LLM-narrated, not 50k". Lesson: budget for OpenAI account-level rate-limit fatigue separately from the per-call cost projection.

**Cache-key collision on narrator_temp (review 011 finding #3).** The eval gen was supposed to use `--narrator-temp 0.5` for a less-narrator-style-correlated eval surface. The cache key didn't include temperature, so cache hits served back the train-time temp-0.3 narratives. The final salvaged `data/eval_medium_15k_llm/build_summary.json` records **n_cache_hits=15000 / n_calls=0**, confirming every record came from the train-time temp-0.3 cache. Net result: the 15k eval is temp-0.3 (same narrator style as train), not temp-0.5 as planned. The fix landed in commit `db723c1`; future temp ≠ 0.3 gens produce fresh narratives.

**Five integration bugs in the vertical slice (table above).** Most ate one re-run each — small in isolation, dangerous if they had landed during Day-2 sweep.

**Three external-system Day-0 frictions** (not unique to this POC, but worth recording):
- RunPod H100/H200 Secure Cloud showed "Unavailable" in our region; pivoted to RTX PRO 6000 Blackwell on Community Cloud ($1.69/hr).
- Network volume creation forced a region match with the GPU host; we ended up using a Volume disk on Community Cloud instead — backup_to_external.sh covers the preemption risk.
- macOS-bundled rsync is 2.6.9 from 2006 and doesn't support `--info=progress2`. Used `--progress` instead.

### Day-1 metrics snapshot

```
exp_vslice_001  (100 steps, smoke):
  arm=cpt_light  wall=516s  final_train_loss=2.62
  AUC stripped=0.9929 [0.9916, 0.9945]
  R@FPR=1% stripped = 0.7432
  hn_large_purchase FPR @ 1% = 5.65%
  status = PASS_smoke

exp_stage0_001  (1500 steps, full Stage-0):
  arm=cpt_light  wall=3194s  final_train_loss=1.24
  AUC stripped=1.0000 [1.0, 1.0]   (saturated)
  R@FPR=1% stripped = 1.0          (saturated)
  hn_account_recovery FPR @ 1% = 6.15% (5k eval) / 2.22% (15k eval)
  hn_large_purchase FPR @ 1% = 1.01% (5k eval) / 1.16% (15k eval)
  leakage_failures = 0 / 1000 audit
  status = PASS_full
  output_merged_checkpoint = /workspace/checkpoints/qwen3-8b-cpt-light-merged
```

Both entries persisted in `src/auto_research/experiments.jsonl`.

### Day-2 starting state (verified)

- ✅ `qwen3-8b-cpt-light-merged` exists, loads via AutoModelForCausalLM
- ✅ `scripts/preflight_xattn.py` runs green against it: wrapper builds, LoRA-on-Q attaches, dummy forward returns finite loss
- ✅ All event-side trainers wired with `parse_structured_events()` (review 011 finding #1)
- ✅ 11 reviews closed; Day-1 integration fixes recorded in `4bf30f3` (Day-0 pin bumps), `3218f56` (events JSON-string), `2043d78` (eval-only dir), `43e3c6c` (scientific-notation YAML), `4fddbf8` (`max_completion_tokens`), plus `110f0d1` (x-attn preflight) and `db723c1` (review-011 closure)
- ✅ Cache: 30,780+ narratives on the laptop, full copy on `/workspace/data/cache/`
- ✅ Eval surfaces: 5k LLM-narrated (`eval_fast_5k`) + 15k LLM-narrated (`eval_medium_15k_llm`) + 50k templated (`eval_medium_50k` — at ceiling, demoted to "regression test only")

### Headline pivot for Day 2-3

> **AUC and R@FPR=0.1% are saturated at 1.0 on every LLM-narrated eval surface for Stage-0. Use them as sanity checks ("did anything regress?"), not as comparison metrics. The Day-3 cross-attn vs baselines win condition is hard-negative FPR @ 1% on `hn_account_recovery` and `hn_large_purchase`. PLAN.md's Δ AUC <0.005 + non-overlapping CIs guard is now Δ hn_FPR + non-overlapping CIs. The synthetic-data-saturation risk PLAN.md flagged in §Risks did materialize.**

---

## Day 2 — Baselines + evaluation correction

**Status: COMPLETE.** Three baselines (`event_only`, `lora_text`, `structured_as_text`) plus a 100-step `xattn` smoke landed; a Codex pre-Task-#37 review surfaced three findings (leakage, sklearn-cliff metric, label-deterministic synthetic data) that invalidated the apparent leaderboard. A re-score-only correction (no GPU training) produced the corrected `*_v2` rows in `src/auto_research/experiments.jsonl`. Full narrative + per-finding evidence: **`docs/day-2-results.md`**; per-family data-overlap diagnostic: **`docs/day-2-data-diagnostic.md`**.

Corrected leaderboard (worst-family HN-FPR @ 1% legit FPR, stripped mode, tie-aware exact-target metric, clean eval n=4466 after dropping 534 leaked rows; 1000-resample 95% CI):

| Rank | Arm | Worst HN-FPR (point, 95% CI) |
|------|-----|------------------------------|
| 1 | `structured_as_text` | **0.05067** [0.04078, 0.06349] |
| 2 | `xattn` (100-step smoke) | **0.05366** [0.04277, 0.06536] |
| 3 | `lora_text` | **0.07014** [0.05635, 0.08468] |
| 4 | `event_only` | **0.07301** [0.06667, 0.07989] |

All baseline CIs overlap except `structured_as_text` ↔ `event_only` (gap 0.0032). Day-3's sharp x-attn question is: can a fully-trained Task #37 cross-attention model beat `structured_as_text` (5.07%) with **non-overlapping** CIs on this synthetic distribution? Either answer is a deliverable finding. See `docs/day-2-results.md` §5.

---

## Day 3 — Round-2 sweep + analysis + synthesis

*Filled in by the agent at end of Day 3.*

### Sweep round-2 results
*Round-2 + stress run.*

### Top-3 medium eval (50k)
*R@FPR=0.1% with CIs, per-journey, agent-vs-human differential.*

### (Optional) Top-1 large eval (100-200k)
*Only if Day-3 Hr 11 was reached on schedule.*

---

## Final synthesis

### The v3 question answered

> After controlling for token leakage, narrative leakage, structured-stream parity, an event-only classifier baseline, and reported with bootstrap CIs across three eval modes — did cross-attn add **classification** lift, or is its value confined to **explanation/grounding**?

*Answer here.*

### Integration friction catalog
*The journey artifact. Every place an engineer would burn time.*

### Gates story
*Did the cross-attn gates actually open? When? On which configs?*

### Per-baseline deltas (with CIs)
- vs CPT-light:
- vs LoRA-text:
- vs structured-as-text (load-bearing):
- vs event-only classifier (load-bearing — does the LM matter at all?):

### Per-journey breakdown
- clean / cred_stuff / sim_swap / phish_takeover / malware_rat / mule_chain / hn_travel / hn_large_purchase / hn_account_recovery

### Per-actor differential
- human vs agent-driven journeys

### Hard-negative FPR
- hn_travel, hn_large_purchase, hn_account_recovery

### Day-4 recommendation: extend / pivot / stop
*2-3 lines of rationale.*

### Concrete next-steps for a real PayPal-internal POC
*Tied back to `.claude/tasks/cross-attn-ato/next-steps-checklist.md` — which assumptions held, which broke.*
