# Auto-Research Loop — Deep Dive

**A reusable methodology asset from the Cross-Attention ATO POC**

This document describes the auto-research loop that drove the 3-day cross-attention POC end-to-end. It is intentionally written so the next research question — a different architecture, a different domain, a different team — can adopt the same plumbing without re-deriving it from the POC repo. The cross-attention findings live in the companion readout; this is the methodology.

---

## The core principle: agent proposes, deterministic script enforces

The loop is a Karpathy-style split. An LLM agent (Claude Code or Codex CLI) reads the sweep state, the experiment history, and a written playbook (`AGENT_INSTRUCTIONS.md`), then proposes the next experiment as a YAML config. A deterministic Python launcher (`scripts/run_next_experiment.py`) takes that config and does everything else: validates it against a whitelist, dedups it against history, acquires the GPU lock, runs `accelerate launch`, parses metrics from stdout and W&B, computes bootstrap confidence intervals, writes atomic outputs, and appends one immutable row to `experiments.jsonl`. The agent's only structured write is to `config.yaml` and a one-paragraph `notes.md` per run. Everything else the agent does is natural-language summarization for the journey log.

The reason this split matters is not aesthetic; it is operational. Early versions of this loop had the agent appending to `experiments.jsonl` directly, and produced format drift between agent-written rows and trainer-written rows within two days. Moving to a clean owner-per-file model — launcher owns history and state files, agent owns proposal and prose — ended the format drift on day one and never produced a single concurrency or data-loss issue across **30 experiments and three sweep generations (v3 + v4 + v5)**.

## Ownership map

Every file in the system has exactly one owner, and the owner is the only thing allowed to write to it:

| Artifact | Owner | What's in it |
|---|---|---|
| `experiments.jsonl` | Launcher | One JSON row per completed run. Append-only. Immutable once written. |
| `sweep_state.yaml` | Launcher | Derived state: `current_best`, `top_3`, `gpu_hours_used`, `halted`, `halt_reason`. Recomputed from `experiments.jsonl` + `budget.yaml` on every run. |
| `runs/exp_NNN/metrics.json`, `ci_report.json`, `gate_trajectory.json`, `leakage_report.json` | Launcher | Structured per-run artifacts. |
| `runs/exp_NNN/config.yaml` | Agent | The proposed experiment — what dials are set, why. One file per planned run. |
| `runs/exp_NNN/notes.md` | Agent | One-paragraph natural-language summary written after the launcher completes. Interpretation, not numbers. |
| `README.md` Day-2 / Day-3 / Final sections | Agent | The journey log — what happened, what it means. |
| Source code under `src/`, `scripts/`, `eval/`, `data/gen/` | Either, but only after `git add -A && git commit -m "snapshot before <change>"` | Reverts are non-negotiable. |

This split is enforced by the agent's `AGENT_INSTRUCTIONS.md` playbook ("What you must NOT do" section) and verified by `git diff` after any agent-initiated edit.

## The launcher's responsibilities, in order

When the agent writes a new `config.yaml` and invokes `python scripts/run_next_experiment.py src/auto_research/runs/exp_NNN/config.yaml`, the launcher does the following, top to bottom:

1. **Config whitelist validation.** Reject any key not in `experiment_template.yaml`. This blocks the agent from introducing unsupported dials, accidentally setting `shell: rm -rf /`, or perturbing a config in a way the trainer can't handle. Returns a clear error message and exits non-zero.
2. **Dedup against history.** Compute `canonical_hash(config)` and scan `experiments.jsonl` for any prior row with the same hash. If found and the prior row succeeded, exit with code 2 (DUPLICATE). The agent reads this and skips the duplicate. The agent is supposed to perform a tuple-based dedup check before writing the config; the launcher's hash check is the safety net.
3. **Halt check.** Read `sweep_state.yaml`. If `halted: true`, exit immediately — the agent must do daily writeup, not propose new work. Halt conditions are configurable in `budget.yaml`: NaN cascade on N consecutive runs, zero-gate-activation on N consecutive runs, no-improvement-over-window after a minimum number of valid runs, and any of {max_experiments, max_gpu_hours} caps.
4. **Acquire GPU lockfile.** Single-writer guard at `/workspace/.gpu_lock`. Prevents concurrent agent invocations (which can happen with cron-driven re-invocation) from racing each other onto the GPU.
5. **Launch.** `accelerate launch src/train/train_xattn.py --config <path>` with a pinned environment. The trainer streams stdout to a per-run log file.
6. **Parse.** `parse_metrics.py` consumes stdout + W&B local files and emits `metrics.json` (final loss, wall-clock, gate trajectory pointer) and `gate_trajectory.json` (per-step gate magnitudes for the gates story in the journey log).
7. **Compute CIs.** `bootstrap_ci.py` runs 1000-resample bootstrap on every reported metric (AUC, R@FPR, worst-family HN-FPR, mean HN-FPR). Each per-resample run recomputes the tie-aware operating point so the CI bounds are verifiable from the JSON.
8. **Atomic write.** Every output file is written to `<path>.tmp` and `os.rename`'d into place. This guarantees `backup_to_external.sh` (cron'd every 30 minutes) never syncs a half-written file to S3.
9. **Append to history.** One JSON line appended to `experiments.jsonl`. The line carries `metric_version` (**phase-specific**: v3 baselines and Round-1/2 use `2`, the tie-aware exact-target metric; v4 and v5 runs use `5`, the adversarial-error decomposition), `clean_eval_n`, `clean_eval_dropped`, and the full CI bundle.
10. **Refresh state.** `update_sweep_state` re-derives `sweep_state.yaml` from `experiments.jsonl + budget.yaml`. Ranking is filtered to the current generation's `metric_version` (v3 ranking used `>= 2`; v5 ranking uses `>= 5`); older rows remain in history for audit but are excluded from `current_best` and `top_3`.

The agent runs once after each completed launcher invocation. It reads the new row, writes `notes.md`, and decides whether to propose the next experiment or write the daily journal section. If a cron is driving the loop, the agent is re-invoked every N minutes by `scripts/agent_tick.sh`. If a single Claude Code session is driving the loop, the human reinvokes the agent prompt manually. Both modes have shipped during the POC; cron is the default for unattended overnight runs.

## The proposer heuristic

The agent's playbook prescribes a three-round structure that biases exploration early and exploitation late, with halt-by-design:

**Round 1 — spread.** First 6 experiments cover the `insertion_pattern × resampler_slots` grid at a fixed gate init. The agent doesn't propose an experiment "smarter" than the next grid cell in this phase; the point is to fill the cells, not to optimize. Order: every_4 / 64, every_8 / 64, late_only / 64, every_4 / 128, every_8 / 128, late_only / 128. If any of the first two fail (NaN, zero gates), the agent pauses and reports rather than blindly continuing.

**Round 2 — perturb.** After Round 1, identify the top-2 configs by worst-family HN-FPR (lower is better; tiebreak on mean HN-FPR). For each, propose a single-dial perturbation: top-1 with `gate_init=zero`, top-2 with `gate_init=zero`. The point is to probe sensitivity, not to grid-fill. If a Round-2 result beats its Round-1 sibling with non-overlapping bootstrap CIs, it becomes the new leader.

**Round 3 — stress.** Take the top-1. Propose one stress run (`stress_run: true`, steps=3000, seq_len=4096) to test whether longer training and longer context lift the leader past the strongest baseline.

This heuristic is encoded in `AGENT_INSTRUCTIONS.md` so the agent can re-derive it from scratch on every cron tick without prior session context. The actual decisions the agent makes are constrained by the heuristic but not micromanaged: which Round-2 perturbation to dispatch first, how to phrase the per-run interpretation, whether to flag a surprising result for the human — all judgment calls the agent owns.

## Halt conditions — designed to stop, not to keep going

The hardest single design call in the loop is "when do we stop?" A loop that doesn't know how to stop will spend an 18-hour GPU budget on a 1-hour-of-signal task. We encoded four halt conditions, all in `budget.yaml`, all enforced by the launcher:

1. **NaN cascade.** Two consecutive runs with NaN final loss. Catches divergent training before it eats the budget.
2. **Zero-gate-activation.** Two consecutive x-attn runs with `max_gate_magnitude` below a threshold (currently `0.005`, originally `0.05`; the threshold itself was tuned mid-POC after we observed `small_0.01` gate-init initializing at exactly 0.01 and lifting only to ~0.011). When this fires, the finding is "gates failed to open on this config" — that itself is a research result, not a bug.
3. **Convergence.** No improvement on worst-family HN-FPR by at least `min_delta` over the last `window_size` valid x-attn runs, *and* at least `min_valid_runs_before_halt` runs have completed. The "and" matters: a naive convergence check fires before any real signal, costing the sweep its exploration phase.
4. **Budget caps.** Either `max_experiments` or `max_gpu_hours` exceeded. Hard stop.

Three of these (NaN, zero-gate, convergence) fired during the POC. The convergence halt fired *prematurely* on v3 Day-2 because the Round-1 leader sat in slot 1 of the rolling window, making it mathematically impossible for a Round-2 sibling to beat it by 0.005. That bug was a finding in its own right (see the integration-friction catalog, item 8). The fix was to make convergence-halt configurable and toggle it off for v4 and v5, leaving NaN and zero-gate as the real stops. v5 added an `early-exit-on-success` rule on top of that: if any single run records `max_gate ≥ 0.05` AND beats the current best with non-overlapping CIs, the mechanical phase queue stops and the agent pivots to local perturbations. This is the kind of subtlety you only discover by running the loop on a real research question — the cost of discovering it was one halt + one debug pass + one halt-config rework, spread across three sweep generations.

## The dedup tuple — preventing redundant work without preventing exploration

The launcher's `canonical_hash` is a safety net, not the primary dedup mechanism. The primary mechanism is a **tuple-based dedup check** the agent runs before writing each `config.yaml`. The current tuple is:

```text
(insertion_pattern, gate_init, resampler_slots, encoder, lora_r_on_q,
 lr, warmup_steps, steps, seq_len)
```

Rules:
- **No-redo**: if any `status: ok` row matches the tuple exactly, skip and advance to the next queue item.
- **Retry-failed-once**: if a `status: failed | nan | timeout` row matches the tuple, the agent may retry it once with a new exp_id. If there are already two failed rows for the tuple, or any successful row exists, skip.
- **Explicit retry-allowed slots**: the playbook can name specific cells as "retry-once-allowed" (e.g., a cell that failed for environmental reasons rather than algorithmic ones).

This tuple is wider than the launcher's hash on purpose: the agent operates on the cognitive units of the experiment (which dials are different, semantically), not on YAML-serialized bytes. A `lr: 3e-4` and `lr: 0.0003` would hash differently but tuple-match; the agent skips the redundant proposal before writing the file, saving config-writing work.

## The expanded-sweep directive — how the loop adapts to new questions mid-flight

After Day-3's halt — a clean `zero_gate_activation` on the second Round-2 run, with gates barely lifting from zero in 1500 steps — the natural next question shifted from *"which insertion pattern wins?"* to *"can we make the gates actually learn?"* Rather than spinning up a new loop from scratch, we appended an "Expanded Sweep directive" section to `AGENT_INSTRUCTIONS.md` that supersedes the original Round 1/2/3 heuristic with a new phase queue:

- **Phase 1.** LR / warmup sweep around the current-best config (3 runs). Tests whether the gate-stagnation result is robust to optimization settings, or whether lr=1e-4 was too low for gates to escape init-bias in 1500 steps.
- **Phase 2.** One stress run at steps=3000, seq_len=4096 on the best config from Phase 1.
- **Phase 3.** Finish the original 3×2×2 grid cells that didn't run during Day-2/3, mostly because the Round-2 halt fired before they could be dispatched.
- **Phase 4.** Rank-capacity sweep (`lora_r_on_q ∈ {32, 64}`) — conditional on GPU budget remaining ≥ 1.5h, which under current budget arithmetic likely won't fit.

Three loop-level changes were made to support the expansion: `max_experiments` lifted from 12 to 999, `zero_gate_activation` and `convergence` halts disabled, GPU-hours cap and `nan_cascade` kept as the real stops. An **early-exit-on-success rule** was added: if any single run records `max_gate_magnitude >= 0.05` *and* beats current_best with non-overlapping CIs, the mechanical phase queue stops and the agent pivots to local perturbations of that cell. This caps the cost of exploration — if the loop discovers a real result, it stops grid-marching and starts local-searching.

The expansion ran on the agent's existing playbook with no code changes to the launcher.

## What the loop got us, in numbers

- **30 experiments across three sweep generations** (v3 + v4 + v5): 22 cross-attention runs + 8 baseline runs. v3 used 7.735 GPU-hours of an 18-hour cap; v5 used 7.92 of 12 hours. Both halted by the loop's stop conditions before exhausting budget.
- **4 baselines** trained, evaluated under three eval modes (stripped / opaque / full), bootstrap-CI'd, and rescored under the corrected metric — all in the same pipeline that produced the x-attn runs.
- **Two findings caught before they wasted training time.** The sklearn-cliff metric bug (5-7× phantom advantage for the event-only baseline) was caught by the Codex review pass before a full x-attn dispatched against the wrong leader. The narrative leakage was caught by `leakage_checks.py` running on every eval — leakage state is recorded inline on each `experiments.jsonl` row (via the `leakage_clean` flag and `clean_eval_dropped` count), with the detail also written to `runs/exp_*/leakage_report.json`. Either source is sufficient to audit any run; the in-row flag is what the launcher's halt logic keys off.
- **Three metric corrections rolled forward without retraining.** `metric_version 1 → 2` (v3 Day-2): the tie-aware exact-target operating-point fix, applied to existing predictions via `scripts/rescore_baselines.py`. `metric_version 2 → 5` (v5): the `v5_adv_error` adversarial-error decomposition (mean of `phish_takeover_miss`, `phish_takeover_mfa_phished_miss`, `hn_recovery_high_amount_fpr`), computed inline in the v5 trainer.
- **Zero data-engineering at the end.** The final synthesis was driven directly from `experiments.jsonl`. No spreadsheets, no manual joins, no late-night reconciliation. The reproducibility sections of `whitepaper/03-eval-strategy.md` §9 and `whitepaper/04-cross-attention-experiments.md` give the exact shell commands to re-derive every claim from the repo.

## What the loop did *not* get us, and why that matters

The loop does not substitute for research judgment about what question to ask. **Three** findings across the three sweep generations required a human to step in and reframe the question. (1) AUC saturation on v3 Day-1 — the headline metric pivoted from AUC to worst-family HN-FPR. (2) sklearn-cliff metric correction on v3 Day-2 — the entire eval pipeline migrated to `metric_version: 2`. (3) The v3 null-result diagnosis between v3 and v4 — the loop produced an honest null, but it was a human code audit that traced the null to a synthetic-data leakage pathology (narrator paraphrasing the event signal into the narrative) rather than an architectural failure, motivating the v4 data pivot. In all three cases the agent caught the *symptoms* (AUC=1.0 logged in every row; `event_only` showing an implausibly large advantage; gates riding init bias); the human caught the *reframe*. The loop accelerates the iteration speed of "ask question, run experiment, read result"; it does not redefine the question being asked. Treat it as a force multiplier, not a substitute.

## What is reusable, beyond this POC

The following components are intentionally written to be domain-independent and should drop into the next research question with light modification:

- **`scripts/run_next_experiment.py`** — the launcher. Domain-coupling is in the trainer it invokes and the metric module it imports; the orchestration (validate, dedup, lock, launch, parse, CI, atomic, append, refresh) is generic.
- **`src/auto_research/AGENT_INSTRUCTIONS.md`** — the agent playbook template. The proposer heuristic is research-question-specific; the ownership map, the dedup tuple framework, and the halt-condition framework are not.
- **`eval/bootstrap_ci.py`** — generic 1000-resample bootstrap. Wraps any scalar metric function and emits per-resample diagnostics alongside the point estimate.
- **`eval/leakage_checks.py`** — train/eval leakage detector. The structured-events-hash + text-hash dedup pattern generalizes to any paired-stream dataset.
- **`scripts/preflight_check.py`** — GPU, VRAM, model-download, tokenizer-roundtrip, writable-volume gate. Catches environmental issues before they cost training time. The Blackwell-image patch in this script (item 1 in the integration-friction catalog) saved ~2× throughput before anyone noticed it was missing.

The recommendation in the executive summary — *invest in the loop; validate cross-attention through data, not blind sweeps* — is grounded in this list. The architecture-specific code in the POC (cross-attention block, resampler, qwen wrapper) is single-use. The loop is multi-use.

## Open questions for the leadership team

1. **Where should the loop live next?** The natural candidates are a data-side question (rebuild the synthetic ATO generator to remove skeleton-overlap, then re-run the cross-attention sweep on the cleaner surface) or a different-architecture-different-domain question (e.g., a tabular foundation model sweep, or a structured-event encoder sweep without the LM). The auto-loop plumbing transfers either way.
2. **Should we publish the methodology?** The Karpathy-style agent-proposes/script-enforces split is interesting enough on its own merits — and the metric-correction-mid-flight story is interesting enough as a case study — that a short writeup for the AI Engineering Blog or an internal capability deck would land. The data is already in the repo.
3. **How do we package this as a Foundation Science capability?** Today the loop is a folder structure and a launcher script. Making it a paved-road internal tool — `fsci-research-loop --task <yaml>` — is one or two weeks of engineering, and would make every POC the team runs faster from day one.

---

## Appendix — file map

For anyone auditing this against the repo:

```text
cross_attn_ato_poc/
  PLAN.md                                    # v3 plan
  RUNBOOK.md                                 # RunPod setup, env vars, recovery
  README.md                                  # journey log
  docs/
    day-1-results.md                         # full numeric backing for Day-1 claims
    day-2-results.md                         # metric_version=2 leaderboard, three findings
    day-2-data-diagnostic.md                 # skeleton-overlap diagnostic
  src/auto_research/
    AGENT_INSTRUCTIONS.md                    # the agent playbook
    experiments.jsonl                        # immutable history (launcher only)
    sweep_state.yaml                         # derived state (launcher only)
    experiment_template.yaml                 # config whitelist
    configs/
      sweep_space.yaml
      budget.yaml                            # halt conditions, GPU caps
    runs/
      exp_xa_round1_002/                     # the leader
        config.yaml
        metrics.json
        ci_report.json
        gate_trajectory.json
        leakage_report.json
        notes.md
      ...
  scripts/
    run_next_experiment.py                   # the launcher
    parse_metrics.py
    bootstrap_ci.py
    preflight_check.py
    rescore_baselines.py                     # produced the metric_version=2 rows
    diagnose_data_overlap.py                 # produced day-2-data-diagnostic.md
    agent_tick.sh                            # cron entrypoint
  eval/
    score_risk.py                            # tie-aware exact-target HN-FPR
    leakage_checks.py                        # text-hash + events-hash dedup
    eval_modes.py
    bootstrap_ci.py
  .claude/tasks/xattn-expanded-sweep-plan.md # the in-flight expansion plan
```
