# Cross-Attention for PayPal ATO — Journey Log

3-day POC. The journey matters more than the result.

Plan: `PLAN.md` (v3).
Runbook: `RUNBOOK.md`.

---

## Day 0 — Pre-flight (this directory, before GPU)

*Filled in by the human at end of Day 0.*

---

## Day 1 — Foundation: data + CPT-light baseline

*Filled in by the human at end of Day 1.*

### Hr 0-2: Tokens + fencer + bucketer
*Pending.*

### Hr 2-6: Vertical slice
*Pending. Make-or-break block.*

### Hr 6-10: Scale + Stage-0 CPT-light + merge
*Pending.*

### Hr 10-14: CPT 3-mode eval + writeup
*Pending.*

### Day-1 friction log
*Bullet list, what surprised us, what broke.*

### Day-1 metrics snapshot
*qwen3-8b-cpt-light-merged eval AUC across three modes, per-journey breakdown, leakage audit.*

---

## Day 2 — Architecture + first sweep batch

*Filled in by the agent at end of Day 2.*

### Architecture surgery friction
*Where did `qwen_xattn_wrapper.py` integration cost time?*

### Baselines summary
*CPT-light, LoRA-text, structured-as-text, event-only classifier — metrics with CIs.*

### Sweep round-1 results
*First 4-6 experiments, leader, gates story.*

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
