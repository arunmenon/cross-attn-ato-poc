# Whitepaper Review — Eighth Pass (`cross_attn_ato_poc/whitepaper/v1.2`)

**Reviewer pass · 2026-05-22**
Subject: Phase A + B repositioning per the senior-reviewer external-readiness feedback. Whitepaper version bumped v1.1 → v1.2.
Prior reviews: `REVIEW.md` → `REVIEW-2.md` → `REVIEW-3.md` → `REVIEW-4.md` → `REVIEW-5.md` → `REVIEW-6.md` → `REVIEW-7.md`.
Pre-repositioning checkpoint SHA: **`6ebf829`** (`git reset --hard 6ebf829` to revert).

---

## 1. Top-line

REVIEW-7 closed the artifact-grounding gap (V4/V5 numbers, leakage-record wording, lock paths, run-count provenance, halt-mechanism distinction). The senior-reviewer external-readiness pass then surfaced a separate class of issue: the paper's most defensible contribution is the **agentic research harness + leakage-safe + tie-aware-eval methodology**, but the title, abstract, and section ordering led with the cross-attention architecture as if it were the main result.

This pass repositions the paper around the reviewer's recommended lead: *"the cross-attention finding is the worked example; the loop is the reusable artifact."* The substantive findings do not change; the framing, the lead message, and the supporting evidence labels move.

Scope chosen by user: **Phase 0 (git checkpoint) + Phases A + B (internal repositioning).** Phases C (figure cleanup), D (50k LLM eval), and E (XGBoost baseline) deferred to a future pass per `.claude/plans/i-want-to-do-compressed-bee.md`.

---

## 2. Phase 0 — Git checkpoint (revert safety net)

Prior to any rewrite, the entire `whitepaper/` directory was untracked in git despite 8 review passes + V4/V5 numeric corrections + Figure 4 SVG edits + Codex's fourth-pass cleanup. A checkpoint commit was pushed before Phase A began so the repositioning has a known-good revert point.

- **Local commit:** `6ebf829` — *"checkpoint: whitepaper v1.1 + 7-pass review trail (pre-repositioning)"*
- **Pushed to:** `origin/main` (`https://github.com/arunmenon/cross-attn-ato-poc.git`)
- **Files committed:** 21 (5 `.md` docs, 7 `REVIEW*.md` files, 4 SVG + 4 PNG figures, `.gitignore` with the figures-lockfile exclusion additions). 4046 insertions.
- **Verification:** `git ls-remote origin main` returned `6ebf82905206b83a7212745327b7e4f6fadf5c31` matching local HEAD.
- **Revert command (if needed):** `git reset --hard 9fb0947` (the SHA before this checkpoint).

The pre-commit hook rejected an initial attempt due to a `Co-Authored-By` trailer (repo convention bans Claude attribution); the commit succeeded on the second try without the trailer. No hook bypass (`--no-verify`) was used.

Once this lands, all subsequent Phase A/B edits are recoverable.

---

## 3. Phase A — Framing & claims tightening (5 sub-edits)

### 3.1 A.1 — Soften abstract claims to synthetic-only

**`00-main` Abstract (line 10):**
Added "**synthetic** account-takeover (ATO) detection" qualifier to the opening sentence, plus an explicit follow-up: *"All results in this paper are on synthetic data modeled on PayPal session schemas; production transfer is explicitly out of scope and not claimed (§7, §8.1)."*

### 3.2 A.2 — Qualify the v5 ceiling claim with CI-width context

**`00-main` §4.3 Finding 3 (line 219):**
Rewrote the lead sentence to: *"Within the 11-run v5 sweep on the 5k clean eval (`hn_recovery_high_amount` n=78, per-family CI width ~0.18), no architectural dial moved the bottleneck beyond CI noise."*
Added a "Caveat on strength of claim" sentence at the end of the paragraph noting that the per-family CI is wide because only 78 rows land in this family in the 5k eval, and that a 50k LLM-narrated eval (already built at `data/eval_medium_50k_llm/`, not yet scored on the v5 winner) would tighten the per-family CI ~3×.

### 3.3 A.3 — §1.2 (now §1.3) disclosure ordering

**`00-main` §1.3 (was §1.2):**
- Moved "synthetic-only" disclosure from second sentence to first; bolded "Every result in this paper is on synthetic data."
- Added explicit "no non-LLM tabular baseline (XGBoost / LightGBM on bucketed features) was tested" disclosure.

### 3.4 A.4 — Promote "loop is the reusable artifact" to §1.1 lead

**`00-main` §1.1:**
Inserted as a blockquote prefix above the existing contributions list:
> *"The cross-attention finding is the worked example; the loop is the reusable artifact."*

The same sentence still appears in §8 Conclusion and now in the new Executive Summary, intentionally — it's the spine claim.

### 3.5 A.5 — §5 Results cross-reference to §7 Limitations

**`00-main` §5 v5 paragraph (line 237):**
Appended: *"All v4/v5 numbers above should be read with the §7 limitations in mind — synthetic data only, single LM family/scale, gate magnitudes below the Flamingo 'open' target, and per-family CI width ~0.18 on the 5k-eval bottleneck family."*

---

## 4. Phase B — Structural repositioning (5 sub-edits)

### 4.1 B.1 — New title + version bump

**`00-main` line 1:**
- **Old:** *"Cross-Attention for Account-Takeover Detection: A Three-Generation Study Driven by an Agentic Experiment Harness"*
- **New:** *"A Guardrailed Agentic Research Loop for Cross-Modal Fraud Modeling"* (H1) with subtitle *"A Case Study in Gated Cross-Attention for Synthetic ATO Detection"* (H3)
- **Version:** bumped `v1.1 · 2026-05-22` → `v1.2 · 2026-05-22`

### 4.2 B.2 — §0 Executive summary (one page)

**`00-main` inserted between header and Abstract:**
Five-bullet structure as specified in plan:
1. What we built (harness + leakage-safe data + bootstrap-CI tie-aware eval).
2. What we tested (gated x-attn on frozen Qwen3-8B for synthetic ATO).
3. What happened (v3 null → v4 CI-separated win → v5 ceiling).
4. What we learned (loop is portable; architecture works when given a problem it can solve).
5. What's next (real-data replay + bottleneck redesign + tabular baseline).
Plus a "Bottom line" callout with the lead sentence.

### 4.3 B.3 — §1.2 Claims at a glance

**`00-main` new §1.2:**
Seven-row table (extended from the plan's five-row template) covering: harness durability, mid-POC metric corrections, v4 x-attn win, v5 ceiling, production transfer, tabular-baseline beat, calibration. Each row has: Claim · Evidence · Strength · Where in paper. Strength labels: **Strong**, **Strong within synthetic eval**, **Medium**, **Not claimed**.

Numbers in the table are artifact-grounded against `runs/exp_{text_only,xattn}_v4_001/ci_report.json` and `runs/exp_v5_p1_zero_64/ci_report.json` (verified post-edit). The existing §1.2 was renumbered to §1.3.

### 4.4 B.4 — §8.1 Real-data validation roadmap

**`00-main` new §8.1 (end of §8 Conclusion):**
Six-step concrete roadmap as specified in plan: anonymized window + temporal split + calibration + production-baseline comparison + step-up-routing precision for hn_recovery analog + statistical-significance protocol. Closing sentence makes explicit that the paper does NOT commit to deliverable dates or claim that the synthetic-surface ceiling will transfer to real traffic.

### 4.5 B.5 — Promote Figure 2 (harness dataflow) to §1

**`00-main`:**
- §1 ¶2: added "Figure 2 (below) shows the loop that drove every run; it is the artifact this paper claims is most reusable beyond the specific cross-attention case study." + inserted the `![Figure 2. ...]` render directly after the paragraph.
- §3.2: removed the duplicate Figure 2 render; replaced "Figure 2 shows the full dataflow." with "Figure 2 (rendered in §1 above) shows the full dataflow." — render-once, reference-elsewhere.

Now the reader sees the harness loop before reading about the cross-attention architecture, matching the lead-with-harness positioning.

---

## 5. What did NOT change

- All quantitative findings (V4 recall numbers, V5 ceiling band, gate magnitudes, run counts).
- The companion documents `01-data`, `02-harness`, `03-eval`, `04-experiments` were not edited in this pass (no claim drift; their content was already artifact-grounded post-REVIEW-7).
- The References block.
- The §7 Limitations content (only cross-referenced from §5 in A.5).
- The figures themselves (no SVG edits; that's Phase C, deferred).

---

## 6. Cross-document consistency check after this pass

| Claim | Abstract | Executive summary §0 | §1.1 lead | §1.2 table | §5 Results | §7 Limitations | §8 / §8.1 | ci_report.json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Synthetic-only | "synthetic ATO" + "not claimed" ✓ | "modeled on PayPal" ✓ | (n/a) | "Not claimed" rows ✓ | §5 cross-ref ✓ | "Synthetic data only" ✓ | §8.1 roadmap ✓ | (n/a) |
| Loop is reusable artifact | closing sentence ✓ | "Bottom line" callout ✓ | blockquote lead ✓ | (n/a) | (n/a) | (n/a) | conclusion sentence ✓ | (n/a) |
| `phish_takeover` recall 0.11 → 1.00 | (not in abstract) | (paraphrase) ✓ | (n/a) | row ✓ | bullet ✓ | (n/a) | (n/a) | matches |
| `phish_takeover_mfa_phished` recall 0.00 → 0.97 | "text-only 0.000 vs 0.972" ✓ | (paraphrase) ✓ | (n/a) | row ✓ | bullet ✓ | (n/a) | (n/a) | matches |
| v5 ceiling at ~0.44 FPR | "44% FPR" ✓ | "data ceiling" ✓ | (n/a) | row + caveat ✓ | §5 bullet ✓ | (referenced) | §8.1 step 5 ✓ | matches |
| Production transfer not claimed | (implied) | "Production transfer is not claimed" ✓ | (n/a) | "Not claimed" ✓ | (n/a) | §7.1 ✓ | §8.1 explicit ✓ | (n/a) |
| Tabular baseline not tested | (n/a) | "What's next" ✓ | (n/a) | "Not claimed" ✓ | (n/a) | (n/a) | (n/a) | (n/a) |

Zero contradictions across documents. Two appearances of the lead sentence (executive summary + §1.1 + §8) are intentional — it's a spine claim.

Post-edit greps confirmed:
- `grep -nEi "we present a (new|novel)? cross.attention"`: 0 hits
- `grep -nEi "our (new|novel)? architecture"`: 0 hits
- Section structure (`grep -n "^##\|^###"`): clean, no orphaned subsections.

---

## 7. Verdict

The paper is now positioned as a **methodology paper grounded in a worked case study**, per the senior-reviewer recommendation. The harness is the lead artifact; the cross-attention result is the case study that validates it.

Remaining open items from the senior-reviewer pass (all in "Deferred future work" in the plan, none blocking the internal-distribution scope):

- **Phase C** (figure cleanup): Fig 1 split into simple + technical, Fig 3 font/name simplification, PNG re-render. Trigger: paper goes external or to a slide deck.
- **Phase D** (50k LLM-narrated eval): score v5 Phase-1 winner on the larger eval to tighten the `hn_recovery_high_amount` CI ~3× and upgrade the claims-table strength. Trigger: external send or claims-table strength upgrade needed.
- **Phase E** (XGBoost tabular baseline): adds non-LLM floor for fraud-audience defensibility. Trigger: paper goes to fraud-detection reviewers.

For internal Foundation Science distribution, the paper is **ready**. The pre-send proofread + figure re-render that Codex flagged at REVIEW-7 still applies; PNG companions of the figures were not re-rendered in this pass (none of the SVG edits in this pass were visual — they were markdown content changes only).

---

## 8. Suggested follow-up commit

The repositioned state should be committed and pushed alongside the Phase 0 checkpoint, leaving both SHAs in remote history:

```bash
git add whitepaper/00-whitepaper-main.md whitepaper/REVIEW-8.md
git commit -m "$(cat <<'EOF'
whitepaper v1.2: repositioning around harness-as-lead contribution

Phase A + B repositioning per senior-reviewer external-readiness feedback.
The cross-attention result becomes the worked example; the agentic
research loop + leakage-safe data + tie-aware eval becomes the lead
contribution.

Phase A (framing tightening, 5 sub-edits):
- Abstract softened to synthetic-only with explicit production-transfer-
  not-claimed disclosure.
- v5 ceiling claim qualified with CI-width context (n=78, CI width ~0.18)
  and 50k-eval recommendation.
- §1.3 (was §1.2) disclosure reordered; synthetic-only first; tabular
  baseline explicitly disclaimed.
- "Loop is the reusable artifact" promoted to §1.1 lead blockquote.
- §5 Results cross-references §7 Limitations inline.

Phase B (structural repositioning, 5 sub-edits):
- New title: "A Guardrailed Agentic Research Loop for Cross-Modal Fraud
  Modeling: A Case Study in Gated Cross-Attention for Synthetic ATO
  Detection". Version bumped v1.1 -> v1.2.
- New §0 Executive Summary (one page, 5 bullets + bottom-line callout).
- New §1.2 Claims at a glance (7-row table: claim/evidence/strength/
  where-in-paper; numbers verified against runs/exp_*/ci_report.json).
- New §8.1 Real-data validation roadmap (6-step concrete protocol).
- Figure 2 (harness dataflow) promoted from §3.2 to §1 so the reader
  sees the loop before the architecture.

REVIEW-8.md documents the changes and lists Phase C/D/E as deferred
future work.
EOF
)"
git push origin main
```

The checkpoint SHA `6ebf829` and this follow-up SHA together give a clean revert path if any of the repositioning needs to be unwound.
