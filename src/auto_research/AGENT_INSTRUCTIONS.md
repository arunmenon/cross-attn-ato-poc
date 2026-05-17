# AGENT_INSTRUCTIONS — Cross-Attention ATO POC

You are the auto-research agent for a 3-day POC running on a single H100. Your job: read state, propose the next experiment, hand it to the deterministic launcher, summarize results, decide what's next. **You do not own GPU lifecycle, validation, or metric parsing — `scripts/run_next_experiment.py` does.**

Read this file in full at every loop iteration.

---

## Mission

Find the best cross-attention configuration on a synthetic ATO dataset given a 12-cell sweep space and a 10-12 experiment budget, then write a leakage-safe, CI-defended end-of-POC synthesis comparing cross-attn against four baselines. The journey log matters as much as the result.

---

## State you read

| File | Purpose |
|---|---|
| `PLAN.md` | The plan. Re-read sections "Auto-research loop", "Baselines", "Evaluation" if unsure. |
| `src/auto_research/configs/sweep_space.yaml` | The dial. Treat as read-only. |
| `src/auto_research/configs/budget.yaml` | Budget + halt conditions. Treat as read-only. |
| `src/auto_research/experiment_template.yaml` | Config schema. Every new config must conform. |
| `src/auto_research/sweep_state.yaml` | Mutable state — **written only by the launcher**. You read it for budget / current_best / halt status. |
| `src/auto_research/experiments.jsonl` | Append-only history — **written only by the launcher**. You read it for prior results. |
| `src/auto_research/runs/exp_NNN/` | Per-experiment artifacts (config, log, metrics, gates, CI, leakage). |
| `README.md` | Journey log. You write the Day-2 and Day-3 sections. |

---

## Ownership (clean split)

| Artifact | Owner |
|---|---|
| `experiments.jsonl` | **Launcher only** — structured records, do not touch |
| `sweep_state.yaml` | **Launcher only** — derived from experiments.jsonl + budget.yaml on every run |
| `runs/exp_NNN/metrics.json`, `ci_report.json`, `gate_trajectory.json`, `leakage_report.json` | **Launcher only** |
| `runs/exp_NNN/config.yaml` | **Agent** — one per proposed experiment |
| `runs/exp_NNN/notes.md` | **Agent** — one per completed experiment, natural-language summary (template below) |
| `README.md` Day-2 and Day-3 sections | **Agent** |
| Source code under `src/model/`, `src/train/`, `scripts/`, `eval/` | Modifiable only after `git add -A && git commit -m "snapshot before <change>"` |

---

## Per-iteration loop

1. **Read state.** Read this file, `sweep_state.yaml`, last 5 lines of `experiments.jsonl`, the four baseline entries. If `sweep_state.yaml` or `experiments.jsonl` is missing (gitignored runtime state — fresh clone or just-reset), treat it as zero completed runs and proceed to step 2 normally. The first launcher invocation will create both files. If you're uncertain whether the missing-file case is intentional, run `python scripts/init_auto_research_state.py` to create empty state files (idempotent; does not overwrite non-empty state).
2. **Check halt status.** `sweep_state.yaml` exposes `halted: true/false` and `halt_reason: <string or null>`. If halted: skip to "Daily writeup" below.
3. **Propose next config.** Apply the proposer heuristic (next section). Write to `src/auto_research/runs/exp_NNN/config.yaml` where NNN is the next free slot.
4. **Launch.** Run:
   ```bash
   python scripts/run_next_experiment.py src/auto_research/runs/exp_NNN/config.yaml
   ```
   The launcher validates, dedups against history, acquires GPU lock, runs accelerate, parses metrics, writes per-experiment results, appends to `experiments.jsonl`, and refreshes `sweep_state.yaml`.
5. **On completion**, read:
   - `src/auto_research/runs/exp_NNN/metrics.json`
   - `src/auto_research/runs/exp_NNN/ci_report.json` (aggregate; sections per eval mode)
   - `src/auto_research/runs/exp_NNN/gate_trajectory.json`
   - `src/auto_research/runs/exp_NNN/leakage_report.json` (if trainer wrote one)
6. **Write notes.** Write a one-paragraph natural-language summary to `src/auto_research/runs/exp_NNN/notes.md` (template below). This is your only write of run-level content; you do NOT touch `experiments.jsonl` or `sweep_state.yaml`.
7. **Decide.** Re-read `sweep_state.yaml`. If `halted` → daily writeup. Otherwise → return to step 1.

---

## Proposer heuristic

Three rounds:

### Round 1 — Spread (first 6-8 experiments, Day 2 PM)

Cover the `insertion_pattern × resampler_slots` cells (3 × 2 = 6). Fix `gate_init=small_0.01` and `encoder=small_transformer`. Order:

| Order | insertion_pattern | resampler_slots |
|---|---|---|
| 1 | every_4 | 64 |
| 2 | every_8 | 64 |
| 3 | late_only | 64 |
| 4 | every_4 | 128 |
| 5 | every_8 | 128 |
| 6 | late_only | 128 |

If any of 1-2 fail (NaN or zero gate), pause and report — do not blindly continue. The launcher's halt logic enforces this.

### Round 2 — Perturb top-2 along gate_init (2-4 experiments, Day 3 AM)

After Round 1, identify the top-2 configs by worst-family hard-negative FPR (`hn_fpr_worst_stripped` in `experiments.jsonl`, lower is better; tiebreak on `hn_fpr_mean_stripped`). AUC-stripped is a sanity gate, not the ranking metric — it saturates at 1.0 on every variant (review 013 finding #1). All rows from this commit forward carry `metric_version: 2` with tie-aware exact-target HN-FPR computed against a leakage-filtered eval (text-hash + structured_events-hash overlap removed). Older rows are visible in history but excluded from `current_best`. The launcher applies the clean-eval mask automatically; agent configs do not need to set anything special. For each top-2 config, propose a perturbation:
- Top-1: same config but `gate_init=zero` (probe initialization sensitivity).
- Top-2: same config but `gate_init=zero`.
- If either round-2 result beats its round-1 sibling by non-overlapping CI, that becomes a new top-1 candidate.

### Round 3 — Stress (1-2 experiments, Day 3 midday)

Take the top-1. Propose ONE stress run with `stress_run: true` (steps=3000, seq_len=4096). If VRAM allows. If OOM on the smoke check inside `run_next_experiment.py`, skip and stop. Total Day 2+3 experiments: 10-12.

---

## Halt conditions (must respect)

| Condition | Trigger | Action |
|---|---|---|
| NaN cascade | 2 consecutive experiments with NaN final loss | Halt new launches. Write Day-2 or Day-3 writeup. |
| Zero gates | 2 consecutive x-attn runs with max gate magnitude < 0.05 at step 1500 | Halt. Report as a finding: "gates failed to open on configs X, Y". |
| Convergence | ≥6 valid x-attn runs AND no worst-family HN-FPR improvement ≥ 0.005 (absolute) over the last 4 valid runs | Halt. Top-1 is the winner; proceed to top-3 medium eval. |
| Budget exhausted | `max_experiments` or `max_gpu_hours` reached | Halt. |

When halted, **do not propose new experiments**. Write the daily section of the README, then stop.

---

## Notes template (one markdown file per run at `runs/exp_NNN/notes.md`)

After the launcher completes, write a one-paragraph natural-language summary to `runs/exp_NNN/notes.md`. The launcher has already recorded the structured fields (AUC, CI, gates, wall-clock) in `experiments.jsonl` — your job is the *interpretation*.

```markdown
# exp_NNN notes

Config: every_4 / slots=64 / gate=zero / small_transformer

Trained cleanly. Gates opened to ~0.21 by step 600 and held. Worst-family
HN-FPR-stripped 0.018 with CI [0.012, 0.026] (worst family: hn_travel) beats
CPT-light (0.041, CI [0.033, 0.050]) by non-overlapping CIs. Ties
structured-as-text concat (0.021, CI [0.015, 0.029]) within overlapping CIs —
no decisive lift there yet. Mean HN-FPR-stripped 0.011 (tiebreak metric, same
ordering). AUC-stripped 1.000 (saturated, sanity-only). Per-journey: strong on
sim_swap and mule_chain, weaker on phish_takeover.

Next: try every_4/slots=64/gate=zero with longer training to see if SAS gap
widens on worst-family HN-FPR.
```

Keep it specific: gates story, baseline deltas (with CI overlap status), per-journey signal, what to try next. The Day-2 and Day-3 README writeups will join your notes with the launcher's structured records.

---

## Daily writeup templates

### Day-2 README section

```markdown
## Day 2 — Architecture + first sweep batch

### Architecture surgery friction
- <bullet>: <one sentence>
- <bullet>: <one sentence>

### Baseline metrics (with 95% CIs on 5k fast eval, stripped mode)
Primary: worst-family HN-FPR @ 1% (lower is better). AUC is shown as a sanity column only — it saturates at 1.0 on every variant.

| Baseline | HN-FPR-worst [CI] | HN-FPR-mean [CI] | AUC (sanity) | Notes |
|---|---|---|---|---|
| CPT-light-merged | X.XXX [a, b] | X.XXX [a, b] | X.XX | … |
| LoRA-text-only   | X.XXX [a, b] | X.XXX [a, b] | X.XX | … |
| structured-as-text | X.XXX [a, b] | X.XXX [a, b] | X.XX | … |
| event-only classifier | X.XXX [a, b] | X.XXX [a, b] | X.XX | … |

### Sweep round-1 results
| exp_id | config | HN-FPR-worst [CI] | HN-FPR-mean | gate_max | notes |
|---|---|---|---|---|---|
| …      | …      | …                 | …           | …        | …     |

### Current leader vs each baseline (worst-family HN-FPR; tiebreak mean)
- vs CPT-light: <delta with CI-overlap status>
- vs LoRA-text: <delta>
- vs structured-as-text: <delta>  ← the load-bearing one
- vs event-only: <delta>  ← does the LM matter at all?

### Open questions for Day 3
- <list>
```

### Day-3 README section + final synthesis

Mirror the Day-2 structure for the round-2 sweep and medium-eval results. Then add the final synthesis at the end (see template in `README.md` skeleton).

The final synthesis **must** answer:

> After controlling for token leakage, narrative leakage, structured-stream parity, an event-only classifier baseline, and reported with bootstrap CIs across three eval modes — did cross-attn add **classification** lift, or is its value confined to **explanation/grounding**?

With a concrete Day-4 extend/pivot/stop recommendation in 2-3 lines.

---

## Git-checkpoint policy

Before editing any file under `src/`, `scripts/`, `eval/`, or `data/gen/`:

```bash
git add -A
git commit -m "snapshot before <one-line description of change>"
```

You may freely write to `src/auto_research/runs/exp_NNN/config.yaml`, `src/auto_research/runs/exp_NNN/notes.md`, and `README.md` without git-checkpoints — these are state files, not source code. **Do not touch `experiments.jsonl` or `sweep_state.yaml`** — the launcher owns them.

If a code change you make causes a training failure, the user can `git revert <hash>` to recover the previous good state. Without the checkpoint, that's lost.

---

## Error handling

| Symptom | Action |
|---|---|
| `run_next_experiment.py` exits non-zero with "config validation failed" | Fix the config and retry. Do NOT modify `run_next_experiment.py`. |
| Lockfile exists but no process holds it | Surface this to the user via README — do not delete the lockfile yourself. |
| `metrics.json` missing after launcher returned 0 | Re-run the launcher with `--mark-failed runs/exp_NNN` so the record and `sweep_state.yaml` are updated correctly. Do not edit `experiments.jsonl` by hand. Do not retry the same config. |
| Gate magnitude is NaN | Mark `status: failed`. Counts toward `nan_cascade`. |
| Halt condition met but you're tempted to "try one more" | Do not. Write the daily section and stop. The user can decide to extend. |

---

## What you must NOT do

- Edit `run_next_experiment.py`, `parse_metrics.py`, `score_risk.py`, `leakage_checks.py`, `bootstrap_ci.py`, `merge_stage0_lora.py`, `eval_modes.py`, `eval_mode_dropout.py` — these are deterministic and must remain so.
- Edit `sweep_space.yaml` or `budget.yaml` mid-run.
- Bypass `run_next_experiment.py` and call `accelerate launch` directly.
- Manually edit `metrics.json` or `ci_report.json`.
- Delete or modify completed `runs/exp_NNN/` directories.
- Skip git-checkpoint before code edits.
- Report a "win" without non-overlapping CIs.

---

## What you should do generously

- Write specific, evidence-cited summaries.
- Flag surprising results in the journey log — the journey is the deliverable.
- Note integration friction as you encounter it (HF subclass quirks, gate-init weirdness, data-collate edge cases).
- Be honest about negative results. If x-attn ties structured-as-text within CIs, that *is* the finding.
