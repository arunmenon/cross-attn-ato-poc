# Whitepaper Review — `cross_attn_ato_poc/whitepaper/v1`

**Reviewer pass · 2026-05-22**
Subject: `00-whitepaper-main.md`, `01-data-curation-and-distribution.md`, `02-agentic-experiment-harness.md`, `03-eval-strategy.md`, `04-cross-attention-experiments.md`

---

## 1. Overall verdict

**Solid foundation, ~85% there.** The methodology contribution ("agent proposes, deterministic launcher enforces") is well-articulated. The eval-rigor story (tie-aware operating-point metrics, 1000-resample bootstrap CIs, leakage controls) is real. The companion-document decomposition works — each one stands alone.

Two classes of fixes needed before this goes external:

1. **Several factual claims about the architecture do not match the code.** Reverse-engineered from `src/model/qwen_xattn_wrapper.py:163-186` and `metrics.json` per-cell trainable-param counts. See §3 below.
2. **Stale narrator-model name and narrator-cost number from v3-era drafting.** The conversation history is explicit that `gpt-5.4-nano` was a hallucinated model name we corrected to `gpt-5-nano` (dated snapshot `gpt-5-nano-2025-08-07` used for v4 regen). See §2 below.

The arc framing (v3 null → v4 win → v5 ceiling) and the §1.2 "what this paper is not" disclosure are exemplary and should be preserved. Critical fixes are localized — find-replace + a few table rewrites.

---

## 2. Must-fix factual errors

### 2.1 ⚠️ Wrong narrator model name throughout

**Files:** `00-whitepaper-main.md` (§3.1, §7), `01-data-curation-and-distribution.md` (§1, §3.4, §6).

Cites `gpt-5.4-nano` ~6 times. This model name is hallucinated; OpenAI's API has `gpt-5-nano` (alias) and `gpt-5-nano-2025-08-07` (dated snapshot). Our v4 narrator regen explicitly used the dated snapshot for reproducibility (commit `1fb772d`'s `_PRICING` entry).

**Fix:** find-replace `gpt-5.4-nano` → `gpt-5-nano-2025-08-07` everywhere.

### 2.2 ⚠️ Wrong v4 narrator cost

**File:** `01-data-curation-and-distribution.md` §6.

Claim: *"The narrator cost for the 25k pool was ~$5.67"*.

Actual from `data/train_llm_narrated_v4/build_summary.json`:
```json
"llm_cost": {
    "spent_usd": 2.0299,
    "n_calls": 26319,
    "n_cache_hits": 332,
    "by_model": {"gpt-5-nano-2025-08-07": 2.0299}
}
```

**Fix:** $5.67 → $2.03 (or "$2.03 with 0.4% template-fallback rate on banned-phrase retries"; build duration was 72 min, not 63 — also worth correcting if accuracy matters).

### 2.3 ⚠️ Wrong insertion-pattern block counts (verified against code)

**Files:** `00-whitepaper-main.md` §3.4 AND `04-cross-attention-experiments.md` §1.4.

Real values from `src/model/qwen_xattn_wrapper.py:163-186`:

| Pattern | Actual code | Layer indices | Real block count |
| --- | --- | --- | ---: |
| `every_4` | `range(12, 36, 4)` | 12, 16, 20, 24, 28, 32 | **6** |
| `every_8` | `range(12, 36, 8)` | 12, 20, 28 | **3** |
| `late_only` | `range(32, 36)` | 32, 33, 34, 35 | **4** |

Critically: insertions **start at layer 12, not 0** — code comment: *"so x-attn doesn't disturb early token-feature extraction."*

Whitepaper claims vs reality:

| Where | Claim | Reality |
| --- | --- | --- |
| `00-main` §3.4 | "9, 4, or 3 inserted blocks" | 6, 3, 4 |
| `04` §1.4 | every_4 = 9, every_8 = 5, late_only = 3 | 6, 3, 4 |
| `04` §1.4 | every_4 layers `{0,4,8,…,32}` | `{12,16,20,24,28,32}` |
| `04` §1.4 | every_8 layers `{0,8,16,24,32}` | `{12,20,28}` |
| `04` §1.4 | late_only layers `{24,28,32}` | `{32,33,34,35}` |

**Fix:** Replace the insertion-pattern table in both documents with the corrected values. Also add a sentence noting the deliberate layer-12 offset as a design choice.

### 2.4 ⚠️ Wrong Stage-1 per-block param estimate

**File:** `04-cross-attention-experiments.md` §1.6 "Trainable parameter inventory".

Computed from `src/model/cross_attn_block.py::estimate_block_param_count` with `hidden=4096, cross_dim=1024, dim_ff=2048`:

```
q_proj_in:   4096 × 1024  = 4.2M
kv_proj_in:  4096 × 1024  = 4.2M
MHA@cross_dim:           ~ 4.2M
out_proj:    1024 × 4096  = 4.2M
FFN:         (4096×2048)×2 ≈ 16.8M
─────────────────────────────────
per-block:                ~33.6M  (single value, not a 30-60M range)
```

Verified by actual `metrics.json::actual_trainable_total` for every V5 cell. For every_8 (3 blocks) + slots=64 + r=16:
- 3 × 33.6M (xattn blocks) + ~5M (encoder) + ~3M (resampler) + ~4.7M (LoRA-on-Q) ≈ **113.6M ✓**

This matches `exp_xattn_v4_001 = 113,604,614` exactly.

For every_4 (6 blocks): 6 × 33.6M + ~12M baseline ≈ **214M ✓** (vs `exp_v5_p1_every4_64 = 214,372,364`).

For late_only (4 blocks): 4 × 33.6M + ~12M ≈ **147M ✓** (vs `exp_v5_p1_late_64 = 147,193,864`).

**Fix the table in `04` §1.6:**

| Component | Whitepaper claim | Reality |
| --- | --- | --- |
| Side-stream encoder (small_transformer) | 10–20M | **~5M** |
| Perceiver-Resampler | 2–4M | ~2–3M (correct) |
| Gated cross-attention blocks (per block) | 30–60M | **~33.6M** (single value) |
| Number of inserted blocks | 3 / 5 / 9 | **4 / 3 / 6** (late_only / every_8 / every_4) |
| LoRA-on-Q (rank 16) | ~5M | 4.7M (close enough) |
| Total Stage-1 | 110M–220M | Correct |

### 2.5 ⚠️ Misleading "ft_transformer has higher capacity" framing

**File:** `04-cross-attention-experiments.md` §1.2 and §5.2.

Claim: *"`ft_transformer`. Higher capacity (~113M trainable parameters); tests whether a tabular-foundation-model-style encoder beats the lightweight transformer."*

Real `metrics.json::actual_trainable_total`:

| Encoder | Total Stage-1 (slots=64, r=16, every_8) |
| --- | ---: |
| `small_transformer` (zero_64) | 113,604,614 |
| `pooled_mlp` | 108,998,150 (−4.6M vs baseline) |
| `ft_transformer` | 113,410,566 (≈ same as baseline) |

**ft_transformer is NOT higher-capacity** in total trainable params — it's roughly identical to `small_transformer`. The encoder-specific portion differs by < ~5M; the Stage-1 total is dominated by the 3 cross-attn blocks (~100M).

**Fix:** Reframe to: *"`ft_transformer` (~5M encoder, ~113M Stage-1 total); tests whether a tabular-feature inductive bias in the encoder beats `small_transformer`'s sequence-attention bias. Total Stage-1 parameter count is close to baseline; capacity claims should be encoder-specific."*

### 2.6 ⚠️ Date stamp

All five documents are dated `2026-05-21`. Either bump to `v1.1 · 2026-05-22` to reflect the review cycle, or accept as original-draft date.

---

## 3. Cross-document consistency issues

| Issue | Files | Notes |
| --- | --- | --- |
| Every_8 block count (4 vs 5) | `00-main §3.4` vs `04 §1.4` | Both wrong; reality is 3 |
| Stage-1 per-block param range | `04 §1.6` self-inconsistent vs measured 33.6M | 30-60M range is too loose; actual is tighter |
| Narrator model name | `00-main` + `01-data` | Find-replace as in §2.1 |
| Narrator cost | `01-data §6` | $5.67 → $2.03 |
| `04` §4.1 calls Pathology 3 a "**Bonus**" | `00-main §4.1` | Rename "Pathology 3 (baseline contract)" — drop "bonus" framing for cleaner numbering |

Also flagged but not blocking:
- `00-main` §1.1 contributions list has 4 items, but item 3 (the mid-POC eval correction) reads as a sub-point of item 1 (the harness). Consider merging or reordering.
- Multiple documents use `gate_init=small_0.01` (consistent), but some places say `gate_init = small_0.01` (with spaces). Cosmetic.

---

## 4. References + citations

### 4.1 Karpathy citation missing from References

`00-main §2` references *"Karpathy popularized the 'agent proposes, deterministic script enforces' pattern"* but no formal entry in References. `02-harness §12` cites it more concretely (talks, online writing).

**Fix:** Add a Karpathy citation. If linking to a specific source, "Software 2.0" (Medium 2017) and the more recent "Let's reproduce GPT-2" talk are both candidates. Or accept the cite as "personal communication / public talks" with no formal reference.

### 4.2 Gorishniy et al. (FT-Transformer) cited in `04` but not in `00-main` References

`04-experiments §1.2` references "FT-Transformer (Gorishniy et al., 2021)" but the master References block omits it.

**Fix:** Add to `00-main` References:
> Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. *NeurIPS 2021*.

### 4.3 Jaegle et al. (Perceiver IO) cited in `04` but not in `00-main` References

`04-experiments §1.3` references "Perceiver-Resampler (Jaegle et al., 2021)". Same issue.

**Fix:** Add to `00-main` References:
> Jaegle, A., Borgeaud, S., Alayrac, J.-B., Doersch, C., Ionescu, C., Ding, D., Koppula, S., Zoran, D., Brock, A., Shelhamer, E., Hénaff, O., Botvinick, M., Zisserman, A., Vinyals, O., & Carreira, J. (2021). Perceiver IO: A General Architecture for Structured Inputs & Outputs. *arXiv:2107.14795*.

---

## 5. Editorial / polish

### 5.1 Filesystem hygiene in `whitepaper/`

```
whitepaper/.DS_Store                                 ← gitignore
whitepaper/figures/.~lock.fig1-architecture.png#    ← LibreOffice lockfile; delete
whitepaper/figures/.~lock.fig2-auto-research-loop.png# ← same
whitepaper/figures/.~lock.fig3-data-distribution.png# ← same
whitepaper/figures/.~lock.fig4-sweep-results.png#   ← same
whitepaper/figures/lu122qrt5.tmp                     ← diagram-editor temp, delete (118 KB)
whitepaper/figures/lu163qrz9.tmp                     ← same (70 KB)
whitepaper/figures/lu40qr9v.tmp                      ← same (114 KB)
whitepaper/figures/lu81qrlz.tmp                      ← same (147 KB)
```

**Fix:** `rm` the lock and tmp files. Add `whitepaper/.DS_Store` and `whitepaper/figures/.~lock.*` + `whitepaper/figures/*.tmp` to `.gitignore`.

### 5.2 Title

Current: *"Cross-Attention for Account-Takeover Detection on a Frozen LLM, Driven by an Agentic Experiment Harness"*

**Suggestion (preserves your strongest narrative angle):**
*"Cross-Attention for Account-Takeover Detection: A Three-Generation Study Driven by an Agentic Experiment Harness"*

The "three-generation" framing is the most distinctive thing about this paper; surfacing it in the title primes the reader for the v3 → v4 → v5 arc.

### 5.3 Table formatting

Mix of markdown pipe tables (good for external rendering) and code-block tables (good for monospaced data but fragile in markdown renderers). Recommend standardizing on markdown pipe tables throughout.

### 5.4 Reproducibility command verification (`03` §9)

```bash
python3 -m eval.bootstrap_ci \
  --predictions src/auto_research/runs/exp_v5_p1_zero_64/predictions_stripped.jsonl \
  --metric_version 5 \
  --resamples 1000
```

**Check:** does `eval.bootstrap_ci` actually accept `--metric_version 5`? The selftest invocation uses different args (`--selftest`). Worth a quick `python3 -m eval.bootstrap_ci --help` to confirm the CLI matches.

---

## 6. What's working well (preserve)

1. **The v3 → v4 → v5 arc framing.** The negative-then-positive structure is more credible than a "we won on first try" story. Don't soften the v3 null.

2. **§1.2 "What this paper is not."** Exemplary scope-setting. Should be the model for every methodology paper from this group.

3. **The sklearn-cliff metric correction** (`03 §3.2`). Reproducible numbers (`event_only` at threshold -8.484 → achieved FPR 0.114%) make it credible. This is a publishable methodological finding in its own right.

4. **The three-hypothesis decomposition for the `hn_recovery_high_amount` ceiling** (`04 §6`). Honest "we can't distinguish (1), (2), (3) from the synthetic surface" framing avoids overclaim and invites future work.

5. **Ownership invariant** (`02 §2`). Transferable engineering pattern. Other teams will want to copy this.

6. **Sparse-but-effective gates finding** (`04 §9`). The "gates don't need to open to 0.1-1.0 like Flamingo says; 0.02 is enough" claim is novel and supported by the architecture-level CI-separated win.

7. **Concrete operational claims** (`02 §9`). "30 experiments, zero format-drift, zero concurrency races, two mid-POC metric corrections rolled forward" is verifiable from `experiments.jsonl`. Strong evidence for the methodology contribution.

---

## 7. Recommended fix order

In priority of "would catch a reviewer's eye":

1. **§2.3** — insertion-pattern block counts (`00-main §3.4` + `04 §1.4`). Reviewer with code access would spot this in 10 minutes; fix before any external send.
2. **§2.4** — Stage-1 trainable parameter inventory in `04 §1.6`. Replace 30-60M-per-block range with measured 33.6M; correct the block-count column.
3. **§2.1** — find-replace `gpt-5.4-nano` → `gpt-5-nano-2025-08-07` everywhere.
4. **§2.5** — ft_transformer "higher capacity" reframe in `04 §1.2`.
5. **§2.2** — narrator cost $5.67 → $2.03 in `01 §6`.
6. **§4** — add Karpathy / Gorishniy / Jaegle to References.
7. **§5.1** — clean up filesystem hygiene in `whitepaper/`.
8. **§5.2** — title tweak (optional).
9. **§3** — cross-document consistency pass (mostly absorbed by the above fixes).

Items 1-3 are the must-fix-before-external-send. Items 4-6 are credibility polish. Items 7-9 are operational cleanup.

---

## 8. Estimated effort

- §2 (factual fixes): ~30 min of careful find-replace and table editing
- §3 (consistency): rolls in with §2
- §4 (references): ~5 min
- §5 (polish): ~10 min cleanup + 5 min title decision
- **Total:** ~45-60 min from draft to publish-ready

---

## 9. Files to edit

| File | Sections needing edits |
| --- | --- |
| `00-whitepaper-main.md` | §3.1 (narrator model), §3.4 (insertion blocks), §4.1 (pathology numbering), References (add 2-3 entries) |
| `01-data-curation-and-distribution.md` | §1 + §3.4 + §6 (narrator model), §6 (narrator cost) |
| `02-agentic-experiment-harness.md` | No factual fixes; maybe add Karpathy ref |
| `03-eval-strategy.md` | §9 (verify CLI), no factual fixes |
| `04-cross-attention-experiments.md` | §1.2 (encoder param framing), §1.4 (insertion blocks), §1.6 (trainable inventory table) |
| `.gitignore` (repo root) | Add `whitepaper/.DS_Store`, `whitepaper/figures/.~lock.*`, `whitepaper/figures/*.tmp` |
| `whitepaper/figures/*.tmp`, `whitepaper/figures/.~lock.*` | Delete |

---

## 10. Verification artifacts referenced

Numbers cited in this review came from:

- `src/model/qwen_xattn_wrapper.py:163-186` (insertion-pattern logic)
- `src/model/cross_attn_block.py:59-95` (per-block param formula)
- `src/auto_research/runs/exp_*/metrics.json::actual_trainable_total` (measured trainable counts)
- `data/train_llm_narrated_v4/build_summary.json` (narrator cost, model name)
- `data/gen/narrative_generator.py::_PRICING` (model pricing table, `gpt-5-nano-2025-08-07` entry committed in `1fb772d`)
- `src/auto_research/experiments.jsonl` (cross-checking per-run statistics)

Any future reviewer can re-verify by reading these files directly.
