# Cross-Attention for ATO + Auto-Research Loop — Executive Summary

**Audience:** Prakhar + AI Leadership Team
**Date:** 2026-05-22 · v2 (v4/v5-aware; supersedes v1 dated 2026-05-18)
**POC owner:** Arun Menon (Foundation Science)
**Status:** Three sweep generations (v3 → v4 → v5) complete. Current synthesis lives in `cross_attn_ato_poc/whitepaper/` at v1.2.

---

## The bet, in one paragraph

Take a recently-published architecture (Flamingo-style gated cross-attention) and ask whether it works for PayPal Account Takeover (ATO) detection — a frozen Qwen3-8B reads an analyst-style narrative while a side-stream encoder over a structured event timeline is fused via gated cross-attention. Drive every experiment with a Karpathy-style **auto-research loop**: an LLM agent proposes the next config, a deterministic launcher validates, locks, runs, parses, computes bootstrap CIs, and writes one immutable row to history. Treat the loop itself as a first-class deliverable, alongside any result it produces.

## The story in three acts

This is no longer a "we ran 8 experiments and found nothing" briefing. The work has gone through three sweep generations, each one re-framing the prior:

1. **v3 — honest null.** 18 cross-attention configurations, no separation from a structured-as-text baseline within 95% bootstrap CIs. The headline metric (worst-family HN-FPR @ 1% legit FPR) sat at 0.052 ± noise for every architectural dial tested.
2. **v3 → v4 diagnosis.** A post-hoc code audit, triggered by the unexpectedly clean null, traced the result to two design pathologies in the synthetic data pipeline: (a) the narrator was paraphrasing the structured event signal into the narrative, collapsing the modality gap cross-attention was designed to exploit; (b) hard-negative templates had deterministic feature signatures, making the eval surface label-deterministic in observed support. **The null was an artifact of the data, not the architecture.**
3. **v4 — CI-separated architectural win.** A four-change data pivot (strip narrator's view of bucketed features; stochastic feature signatures; two new adversarial cross-modal families; byte-identical LM prompts across arms) restored a real modality gap. Same architecture, same training recipe, new data: **`phish_takeover` fraud recall jumped from 0.11 (text-only) to 1.00 (cross-attention); `phish_takeover_mfa_phished` from 0.00 to 0.97 — both CI-separated.**
4. **v5 — robust win, one ceiling.** 11 cross-attention configurations across training-dial and encoder sweeps. The v4 win is dial-robust: gate initialization, insertion density, resampler capacity, LoRA rank, and side-stream encoder family all confirm the architectural lift. A new bottleneck surfaced: the adversarial-legitimate family `hn_recovery_high_amount` sits at ~44% FPR across every configuration. **Within the 11-run v5 sweep on the 5k eval (this family has n=78, per-family CI width ~0.18), no tested architectural dial moved this bottleneck beyond bootstrap-CI noise** — the ceiling is upstream in the synthetic generator, not in the model.

## What the AI Leadership Team should take away

1. **The auto-research loop is the durable asset.** 30 experiments across three sweep generations with zero format drift, zero concurrency races, zero manual reconciliation at synthesis time. Two mid-POC metric corrections (`metric_version 1 → 2 → 5`) rolled forward without retraining. The loop caught the metric corrections and the leakage diagnostic at low cost — the same kind of audit on a less-disciplined pipeline would have either missed the v3 data pathology or made the v4 pivot expensive to validate.

2. **Cross-attention is a validated synthetic case study — not yet a production claim.** The v4 result is real architectural lift, CI-separated on the families that demand cross-modal reasoning. But it lives on a synthetic eval surface; production transfer has not been validated. The right next step is data redesign + production-replay + a non-LLM tabular baseline (XGBoost / LightGBM on bucketed events). Until those run, the paper claims an architectural finding within the LM family, not a competitive-modeling claim.

3. **The integration-friction catalog is reusable.** Ten engineering lessons from v3 (Blackwell + bitsandbytes silently dropping to FP32 optimizer; Stage-0 LoRA-merge precondition; narrator-cache leakage; AUC saturation; sklearn-cliff metric bug; convergence-halt premature firing) — each independently transferable to any future POC. The framing is *"engineering lessons from v3 that made v4/v5 credible."*

## The strategic recommendation

**Invest in the loop; validate cross-attention through data and replay, not blind architecture sweeps.**

The cross-attention finding is no longer a clean negative result. v3 was a data-confounded null, v4 produced a CI-separated synthetic win, v5 exposed a data-shaped ceiling. The next dollar should go into:

- **Now (in flight):** Package the loop as a Foundation Science capability — launcher, agent playbook, whitelist, lockfile, dedup, halt conditions, bootstrap-CI eval, tie-aware metrics. Every subsequent POC inherits these for free.
- **Next (1–4 weeks):** Redesign the `hn_recovery_high_amount` generator template with richer disambiguating event context (device-trust history, step-up-auth chain, recovery chain, out-of-band verification). Target: drop the bottleneck FPR below 0.15 before any further architecture sweep.
- **Then (1–2 months):** Run anonymized production replay; add an XGBoost / LightGBM baseline trained on the same bucketed features. This is the test that converts the v4 synthetic win into either a production claim or a data-quality claim — both are valuable.

What we explicitly **do not** claim, today: production transfer, production-fraud-system superiority, calibration, or non-LLM baseline superiority. These are deliberate gaps in the evidence and are listed in the whitepaper's §1.2 "Claims at a glance" table.

## What is in the rest of this package

This summary is the headline. Four companion documents go deeper:

- **`01-auto-research-loop-deep-dive.md`** — methodology asset; the loop's design, ownership invariants, halt logic, three-generation evolution.
- **`02-cross-attn-poc-technical-readout.md`** — three-generation technical readout: v3 result + v3 diagnosis + v4 pivot + v4 result + v5 ceiling + limitations.
- **`03-integration-friction-catalog.md`** — engineering lessons from v3 that made v4/v5 credible.
- **`leadership-readout.pptx`** — 14-slide deck following the v3 → diagnose → v4 → v5 arc, embedding the whitepaper figures.

The full technical depth lives in the whitepaper at `cross_attn_ato_poc/whitepaper/` (5 markdown docs + 6 figures + reviewer chain, v1.2).

## Bottom line

> **We built a guardrailed research loop that prevented false conclusions.** It first found a null result, then helped diagnose why the null was uninformative, then validated cross-attention under a corrected synthetic setup, and finally exposed the next bottleneck as data and task framing rather than architecture.
>
> **The cross-attention finding is the worked example; the loop is the reusable artifact.**
