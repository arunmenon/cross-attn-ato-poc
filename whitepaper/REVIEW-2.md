# Whitepaper Review — Second Pass (`cross_attn_ato_poc/whitepaper/v1.1`)

**Reviewer pass · 2026-05-22**
Subject: same five files, post-fix.
Prior review: `REVIEW.md` (~v1.0 baseline)

---

## 1. Top-line verdict

**Significant improvement.** Most REVIEW.md must-fix items are cleanly addressed. The §1.6 parameter-inventory rewrite in `04-cross-attention-experiments.md` is particularly strong — measured numbers + the verification math `(214,372,364 − 113,604,614) / 3 = 33,589,250` proves the per-block cost is consistent across patterns. The layer-12 insertion-offset is now first-class architectural content, not a footnote. The title tweak landed cleanly.

**~95% there.** Three small residual issues + one critical inconsistency that *crept in* during the fix pass. ~10 minutes of cleanup separates this from publish-ready.

---

## 2. REVIEW.md items — disposition

| # | Issue | Status | Notes |
| --- | --- | --- | --- |
| §2.1 | `gpt-5.4-nano` → `gpt-5-nano-2025-08-07` | ✅ **Fixed** | Find-replace landed in 00-main §3.1, §7, 01-data §1, §3.4, §6, 03-eval §2. **Minor wrinkle:** see §3.2 below |
| §2.2 | Narrator cost $5.67 → $2.03 | ✅ **Fixed** | 01-data §6 now reads "$2.03, 26,319 API calls with 332 cache hits". 03-eval §2 also updated |
| §2.3 | Insertion-pattern block counts | ✅ **Fixed (mostly)** | 00-main §3.4 + 04 §1.4 now say 6/3/4. Layer-12 offset is documented. **But:** see §3.1 critical leftover |
| §2.4 | Stage-1 per-block param estimate | ✅ **Fixed (excellent)** | 04 §1.6 now uses measured 33.6M with verification math; trainable inventory table is clean |
| §2.5 | ft_transformer "higher capacity" reframe | ✅ **Fixed** | 04 §1.2 + §5.2 now correctly attribute the regression to inductive-bias mismatch, not capacity |
| §2.6 | Date stamp | ✅ **Fixed** | All five docs at "v1.1 · 2026-05-22" |
| §3 | Cross-doc consistency | ✅ Mostly fixed | One critical leftover; see §3.1 |
| §4 | Missing References (Karpathy, Gorishniy, Jaegle) | ✅ **Fixed** | All three added to 00-main References block |
| §5.1 | Filesystem hygiene | ✅ **Fixed** | `.gitignore` now includes `whitepaper/figures/.~lock.*` + `whitepaper/figures/*.tmp` |
| §5.2 | Title | ✅ **Fixed** | Now uses the "Three-Generation Study" framing |
| §5.4 | Repro command (03 §9) | ✅ **Fixed** | `--metric_version` removed; correct flags + clarifying comment added |

**Score:** 11/11 of the REVIEW.md issues meaningfully addressed. Quality of fixes is high — most include additional context/explanation rather than just patching the surface claim.

---

## 3. New residual issues found in this pass

### 3.1 ⚠️ CRITICAL — leftover "5 blocks" reference in `04 §4`

**File:** `04-cross-attention-experiments.md` §4, line 170.

Current text (in the v4 gates story):

> "The +0.99 swing in `phish_takeover_mfa_phished` recall is achieved with `tanh(α) ≈ 0.022` — a 2.2% mixing weight per inserted block, **summed over 5 blocks (in the `every_8` configuration)**."

This is wrong — `every_8` has **3 blocks**, not 5. The pre-fix version of §1.4 claimed "every_8 = 5 inserted blocks {0, 8, 16, 24, 32}", which we corrected to "every_8 = 3 blocks {12, 20, 28}". The §4 narrative still references the old number.

**Fix:** s/`summed over 5 blocks`/`summed over 3 blocks`/

This is the most important leftover. It directly contradicts the new (correct) §1.4 architecture table within the same document.

### 3.2 v3 narrator model claim is now slightly inaccurate (historical)

**File:** `00-whitepaper-main.md` §3.1, line 68.

Current text:

> "The narrator (OpenAI `gpt-5-nano-2025-08-07` in v3, the same in v4 and v5) was prompted to ban explicit class names..."

The find-replace was over-aggressive. Historical fact: **v3 was built before we pinned the dated snapshot** — the v3 narrator was configured with a model-name string that did not pin a specific OpenAI snapshot. Only v4's regen explicitly used `gpt-5-nano-2025-08-07` (committed in `1fb772d` via the `_PRICING` table addition).

**Strictly speaking**, the v3 narrator hit whatever OpenAI was serving as the current `gpt-5-nano` alias at the time, which may or may not have been the same underlying snapshot. We pinned in v4 specifically to remove that ambiguity for reproducibility.

**Suggested fix** (preserves the spirit of the original claim while being accurate):

> "The narrator (OpenAI `gpt-5-nano` family — v3 used the then-current alias; v4 and v5 pin the dated snapshot `gpt-5-nano-2025-08-07` for reproducibility) was prompted to ban explicit class names..."

Minor wrinkle, low-priority.

### 3.3 `02-agentic-experiment-harness.md` §7 undersells the cron behavior

**File:** `02-agentic-experiment-harness.md` §7, line 224.

Current text:

> "The CLI receives the loop prompt on stdin, runs one iteration of work (which may launch zero, one, or two experiments depending on speed), and exits."

Empirically (per the v5 sweep observed during the auto-loop session), a single 180-min tick can run **three or more back-to-back experiments**. The 04:30 → 07:30 UTC tick of the V5 sweep ran `exp_v5_p1_every4_64` → `exp_v5_p1_late_64` → `exp_v5_p1_zero_64` in sequence — 3 experiments within one CLI invocation. The cron logs from 05:00, 05:30, 06:00, 06:30 all reported "GPU lock held by live PID=..." and skipped.

This is actually a *positive* operational claim — it shows the back-to-back-loop behavior works correctly. The current "one or two experiments" wording undersells it.

**Suggested fix:**

> "The CLI receives the loop prompt on stdin and loops within one invocation — proposing, launching, evaluating, writing notes, and proposing the next — until either a halt condition is met or the 180-minute outer timeout fires. In v5, a single CLI tick ran three back-to-back experiments within its 180-minute budget; intervening cron ticks observed the held GPU lock and skipped cleanly."

This phrasing also makes the cron+lock+CLI choreography clearer.

### 3.4 Figure 1 architecture diagram may not show the layer-12 offset

**File:** `whitepaper/figures/fig1-architecture.svg` (not opened in this review).

00-main §3.4 and 04 §1.4 both now state that cross-attention insertions deliberately start at layer 12 (not layer 0 as the Flamingo reference does). If Figure 1 was drawn before this offset was made explicit, it may still depict insertions starting at layer 0 — which would now contradict the text.

**Action:** open `fig1-architecture.svg` and verify the layer indices. If the figure shows insertions at layers 0, 4, 8, ..., update it to show 12, 16, 20, ..., 32 (every_4 example) or 12, 20, 28 (every_8 example). Not blocking if the diagram is generic enough that "periodic depth" is shown without specific indices.

### 3.5 Cross-document Karpathy reference alignment

**File:** `02-agentic-experiment-harness.md` §12, line 301.

The "External reference" footer still names Karpathy's "'Let's reproduce GPT-2' and 'I'm running an LLM agent overnight'" — but 00-main's References block (line 309) now consolidates these under a single "Karpathy (2017–2024)" entry that says "Cited as personal communication / public talks; no single canonical reference."

The two entries are not in contradiction, but they describe the same source with different specificity. Either:
- (a) Tighten 02-harness §12 to "External reference: Karpathy (2017–2024), per the master References block in 00-main."
- (b) Or accept the duplication as an artifact of the companion-docs-stand-alone design.

Trivial. (a) is cleaner.

---

## 4. Newly-noticed positives (preserve)

These are improvements over REVIEW.md baseline that are worth calling out:

1. **04 §1.6 verification math.** The line *"The implied per-block cost is `(214,372,364 − 113,604,614) / 3 = 33,589,250` parameters, and is consistent across the late_only difference `(147,193,864 − 113,604,614) / 1 = 33,589,250`"* converts the parameter inventory from "estimate range" to "measured + arithmetically verifiable." This is the single biggest credibility improvement in the second pass.

2. **04 §1.4 layer-12 design rationale.** *"All three patterns deliberately start at layer 12 (or later) — the design choice is that cross-attention should not disturb the LM's early token-feature extraction; the side-stream signal is fused only once the LM has done its first pass of semantic shaping. This is a deliberate departure from the Flamingo reference (which inserts at layer 0); it was set by the original PLAN.md architecture decision and never re-swept."* — turning a code-implementation detail into a stated design choice with a reproducibility footnote is exactly right.

3. **04 §5.2 inductive-bias framing of the ft_transformer regression.** *"the regression is **not a capacity difference** but an inductive-bias mismatch. The tabular-feature attention pattern hurts `phish_takeover_mfa_phished` recall (0.9296 vs 0.9859 for the winner) — most likely because the sequence ordering of events carries signal that the per-feature attention pattern discards in 1500 steps."* — strong substantive interpretation, not just a number correction.

4. **03 §9 reproducibility command honesty.** The clarifying comment *"`metric_version` is read from the predictions file's `metric_version` field; the CLI does not accept a `--metric_version` flag"* shows the doc was actually tested against the CLI, not assumed.

5. **Karpathy citation graceful resolution.** Acknowledging there's no single canonical reference and citing "personal communication / public talks" is the honest move.

---

## 5. Recommended fix order for this second pass

1. **§3.1** — "summed over 5 blocks" → "summed over 3 blocks" in `04 §4` line 170. ~30 seconds. **Critical.**
2. **§3.3** — Rewrite the §7 CLI-tick paragraph in `02-harness` to reflect actual multi-experiment-per-tick behavior. ~2 min.
3. **§3.2** — Soften the v3 narrator-model claim in `00-main §3.1`. ~1 min.
4. **§3.4** — Open `fig1-architecture.svg` and verify the insertion-pattern depiction. Update if it shows layer 0; leave if generic. ~3 min.
5. **§3.5** — Optional trivial alignment of the Karpathy ref between `02-harness §12` and `00-main` References. ~30 seconds.

Total: ~7 minutes for the must-fix (#1) plus the medium-priority items.

---

## 6. Files to edit this pass

| File | Sections needing edits this pass | Severity |
| --- | --- | --- |
| `04-cross-attention-experiments.md` | §4 line 170 ("summed over 5 blocks") | **Critical** |
| `00-whitepaper-main.md` | §3.1 line 68 (v3 narrator model phrasing) | Minor |
| `02-agentic-experiment-harness.md` | §7 line 224 (CLI tick behavior) + §12 line 301 (Karpathy ref alignment, optional) | Medium / Trivial |
| `whitepaper/figures/fig1-architecture.svg` | Verify layer-12 offset is depicted (or accepted as generic) | Audit |

No file from this pass needs more than ~2 minutes of editing.

---

## 7. What still does NOT need editing

These passed both REVIEW.md and REVIEW-2.md scrutiny and don't need further action:

- All References block changes (00-main lines 297-313). Clean.
- The 01-data §3 narrator/template-pivot narrative. Reads honestly and matches the v4 implementation.
- 03-eval's §3 metric-evolution story (`metric_version: 1 → 2 → 5`). Reproducible and well-worked.
- 02-harness §6 v3 convergence-halt postmortem. Strong.
- The §1.2 "What this paper is not" disclosure. Exemplary.
- 04 §6 three-hypothesis decomposition for the `hn_recovery_high_amount` ceiling.
- The full v5 leaderboard tables in 04 §5.

---

## 8. Estimated effort

- §3.1 (critical 5-blocks fix): ~30 sec
- §3.2 + §3.3 (medium-priority text rewrites): ~3 min combined
- §3.4 (figure audit): ~3 min
- §3.5 (optional ref alignment): ~30 sec
- **Total to publish-ready:** ~5-7 minutes.

After this second pass the documents should be externally-shareable without follow-up corrections.

---

## 9. Verification artifacts referenced (same as REVIEW.md §10)

All claims about the codebase in this review were verified against:

- `src/model/qwen_xattn_wrapper.py:163-186` (insertion-pattern logic)
- `src/model/cross_attn_block.py:59-95` (per-block param formula)
- `src/auto_research/runs/exp_*/metrics.json::actual_trainable_total` (measured trainable counts)
- `data/train_llm_narrated_v4/build_summary.json` (narrator cost, model name)
- The conversation history's observed v5 cron behavior (04:30 → 07:30 UTC sweep, 3 experiments in one tick)

The whitepaper's claims now match these sources for everything except the §3.1 critical leftover.

---

## 10. Bottom-line message to the author

The v1.0 → v1.1 pass closed the bulk of the substantive issues. The remaining work is ~7 minutes — one critical text fix and ~3 minor polish items. The architectural-section rewrite (especially `04 §1.4` and `§1.6`) is materially stronger than v1.0 and should anchor any future "how to write a credible POC whitepaper" template from this group.

**One more pass and this is publish-ready.**
