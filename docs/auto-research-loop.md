# Auto-Research Loop — How It Works

Plain-language walkthrough of the Karpathy-style auto-research loop that
ran most of the experiments in this POC. The system was built so that a
researcher (you) goes to bed, wakes up to a finished sweep, and the
log/synthesis sections are written. The design's one rule: **the agent
proposes, the deterministic launcher enforces**.

This doc is for readers who want to understand the system end-to-end
before reading the code. Companion docs:

- `docs/experiments-log.md` — the running history of every experiment.
- `RUNBOOK.md` — operational setup (RunPod boot, cron entry, recovery).
- `src/auto_research/AGENT_INSTRUCTIONS.md` — the prompt the agent reads
  every cron tick.
- `scripts/run_next_experiment.py` — the launcher.
- `scripts/agent_tick.sh` — the cron wrapper.

---

## The 30-second pitch

```
  ┌────────┐ every 30 min  ┌──────────────────┐   ┌──────────────┐   ┌─────────┐
  │  cron  ├──────────────▶│ agent_tick.sh    ├──▶│ claude CLI   ├──▶│ trainer │
  └────────┘               │ (lock pre-check) │   │ (proposes    │   │ (1500   │
                           │ (180m timeout)   │   │  config,     │   │  steps) │
                           └──────────────────┘   │  invokes…)   │   └─────────┘
                                                  └──────┬───────┘
                                                         │
                                                         ▼
                                                ┌─────────────────────┐
                                                │ run_next_experiment │
                                                │ (validate, lock,    │
                                                │  launch, parse,     │
                                                │  CI, append row,    │
                                                │  release lock)      │
                                                └─────────────────────┘
                                                         │
                              ┌──────────────────────────┼──────────────────────────┐
                              ▼                          ▼                          ▼
                      experiments.jsonl         sweep_state.yaml          runs/exp_NNN/*
                      (1 row per run)           (current_best,            (config, metrics,
                                                 budget, halt)             predictions, CI,
                                                                            launcher.log,
                                                                            train.log)
```

Every 30 minutes cron wakes a thin wrapper, which wakes claude, which reads
state, picks the next experiment, calls the launcher. The launcher does
all the dangerous stuff (validation, lockfile, atomic writes, halt checks)
and the agent just decides what to try next.

---

## Why this split?

A naive "give the model a shell, see what happens" loop has three risks:

1. **The agent will skip safety checks** when pressed for output. Dedup,
   halt conditions, GPU lockfile, atomic write of the row — any of these
   can be missed.
2. **The agent will lose its place across sessions.** Conversations are
   ephemeral. Files are not.
3. **The agent will optimize for the wrong metric** if the prompt is
   ambiguous about ranking.

The split fixes all three:

- The launcher (`run_next_experiment.py`) owns every safety check. It
  refuses configs with unknown keys, refuses duplicate hashes, refuses to
  launch if a halt condition fires, and is the only writer of
  `experiments.jsonl` / `sweep_state.yaml`. The agent CANNOT bypass it.
- State lives in files. The agent reads `sweep_state.yaml`, the last 5
  rows of `experiments.jsonl`, and the per-run `notes.md`/`metrics.json`
  files. Every wake-up starts from disk, not from memory.
- The ranking criterion is encoded in the launcher
  (`hn_fpr_worst_stripped`, lower is better, ties broken by
  `hn_fpr_mean_stripped`), and the agent's prompt explicitly references it.

This is the Karpathy "agent proposes, deterministic script enforces"
pattern. It's also why the loop can run unattended overnight: there's a
narrow surface where a misbehaving agent can do damage, and the
deterministic guardrails close every other path.

---

## The components, in plain terms

### `cron`

A standard Linux cron entry:

```
*/30 * * * * /workspace/cross_attn_ato_poc/scripts/agent_tick.sh \
  >> /workspace/agent_tick.log 2>&1
```

Every 30 minutes, wake the agent_tick wrapper. Cron's role is just timing.
It doesn't know anything about experiments.

### `scripts/agent_tick.sh` — the cron wrapper

This is a 200-line bash script. Its job is to:

1. **`cd` into the repo** and source `/workspace/.env` (HF_HOME,
   ANTHROPIC_API_KEY, etc.).
2. **Pre-check the GPU lockfile.** If an experiment is already running,
   exit IMMEDIATELY without invoking the CLI. The lockfile contains the
   live launcher's PID; the script does `kill -0 PID` to confirm. If the
   PID is dead (stale lock), remove the lockfile and proceed.
3. **Halt check (informational).** Run `python run_next_experiment.py
   --halt-check` so the cron log shows the launcher's halt assessment.
   This is informational only — the agent itself reads `sweep_state.yaml`
   on every tick and handles halted state per its prompt.
4. **Invoke the CLI with a fixed prompt** wrapped in `timeout 180m`. The
   prompt tells the agent to read `AGENT_INSTRUCTIONS.md` and follow it.
5. **Log the exit code** so cron logs show clean completions, timeouts,
   and other failure modes.

### `claude` (or `codex`) — the agent

The CLI receives the loop prompt on stdin, runs ONE iteration of work,
and exits. "One iteration" can include multiple back-to-back experiments
if the launcher returns quickly — but the prompt's structure encourages
the agent to do one experiment per wake-up and exit cleanly.

Why `claude` and not direct Python? Because the value-add the agent
provides is **picking the next dial to perturb**, given the current
state. That's a judgment call that depends on what's been tried, what
worked, what failed, and what was learned. A Python script could enumerate
the grid, but it can't read `notes.md` and reason about why a particular
cell is worth retrying.

The agent reads (in order):

1. `src/auto_research/AGENT_INSTRUCTIONS.md` — its prompt / playbook.
2. `src/auto_research/sweep_state.yaml` — budget remaining, current best,
   halted status.
3. The last 5 entries of `src/auto_research/experiments.jsonl`.
4. Optionally: prior `runs/exp_NNN/notes.md` for context.

Then it writes:

1. `src/auto_research/runs/exp_NNN/config.yaml` — the proposed config.
2. Invokes the launcher via shell: `python scripts/run_next_experiment.py
   src/auto_research/runs/exp_NNN/config.yaml`.
3. Waits for the launcher to return (47-150 min).
4. Reads the row that was just appended, plus the per-run
   `ci_report.json` and `metrics.json`.
5. Writes `runs/exp_NNN/notes.md` explaining its reasoning for next time.
6. Exits.

The agent NEVER edits `experiments.jsonl` or `sweep_state.yaml`. Those are
owned by the launcher.

### `scripts/run_next_experiment.py` — the launcher

This is the workhorse. Single-file Python script (~1200 lines). When
invoked with a config path:

1. **Validate** the config against a whitelist of known keys. Reject
   unknown keys, out-of-range values, shell-metacharacter injection.
2. **Dedup** by canonical config hash. Reject if the same dial tuple has
   already been tried at status=ok.
3. **Halt check.** If any halt condition fires (NaN cascade, zero-gate
   activation, convergence, GPU-hours cap), refuse to launch.
4. **Acquire GPU lockfile** via `open(O_CREAT | O_EXCL)`. Refuse if
   another launcher holds it.
5. **F8 v2 defenses** — install `SIGHUP` + `SIGPIPE` IGN handlers,
   attempt `os.setsid()` (typically fails EPERM under claude's Bash tool,
   that's fine), dup stdout/stderr to `<run_dir>/launcher.log`. These
   keep the launcher alive when claude's session times out at 180m.
6. **Launch** the trainer via `accelerate launch …` with
   `start_new_session=True` (F6 — trainer in its own process group, so it
   survives launcher death if that ever happens despite F8).
7. **Stream stdout** into the run dir's `train.log`. Block on the trainer
   for up to `max_wall_clock_minutes` (90 for normal arms, 150 for stress
   runs).
8. **Post-process**: compute clean-eval mask (drop train/eval text-hash
   and structured-events-hash overlap), score predictions in three modes
   (stripped / opaque / full), compute per-family HN-FPR, run 1000-resample
   bootstrap CI, write `predictions_*.jsonl`, `metrics_*.json`,
   `ci_report_*.json`, `gate_trajectory.json`.
9. **Atomically append** the row to `experiments.jsonl` (write to .tmp,
   rename). Update `sweep_state.yaml`.
10. **Release lockfile**.

The launcher returns 0 on success, non-zero on failure modes (validation
error, dedup, halt, lock contention, trainer crash, post-processing fail).
The agent reads the return code and adapts.

### The shared state surface

| File | Owner | Purpose | Updated |
|---|---|---|---|
| `src/auto_research/experiments.jsonl` | launcher (append-only) | one row per run with every metric | per run |
| `src/auto_research/sweep_state.yaml` | launcher | current_best, budget, halted | per run |
| `src/auto_research/runs/exp_NNN/config.yaml` | agent | the proposed dials | once per run |
| `src/auto_research/runs/exp_NNN/notes.md` | agent | reasoning / context | once per run |
| `src/auto_research/runs/exp_NNN/launcher.log` | launcher | F8 v2 detached log | per run |
| `src/auto_research/runs/exp_NNN/train.log` | trainer (via launcher) | training step output | per run |
| `src/auto_research/runs/exp_NNN/metrics_*.json` | launcher | per-mode metrics | per run |
| `src/auto_research/runs/exp_NNN/ci_report_*.json` | launcher | 95% bootstrap CIs | per run |
| `src/auto_research/runs/exp_NNN/predictions_*.jsonl` | launcher | per-example scores | per run |
| `src/auto_research/runs/exp_NNN/clean_eval_mask.json` | launcher | overlap-dropped row mask | per run |
| `src/auto_research/runs/exp_NNN/gate_trajectory.json` | launcher | per-step max/mean gate magnitudes | per run |
| `README.md` (Day-N sections, Final Synthesis) | agent | human-readable writeup | once per halt |

The agent writes the right column when its policy says to. Everything in
the middle column is launcher-owned. The deterministic guardrails are
enforced by who-writes-what.

---

## The hard parts (and how they were solved)

### 1. The agent must not lose its place across sessions

**Solution**: state lives entirely in files. Every cron tick, the agent
starts from a blank conversation and reads `sweep_state.yaml`. There is
no persistent agent memory.

### 2. The agent must not corrupt the row schema

**Solution**: the agent CANNOT write `experiments.jsonl`. The launcher is
the sole writer, and the launcher's row schema is fixed in code.

### 3. Two experiments must never run concurrently on the same GPU

**Solution**: GPU lockfile at `/workspace/.gpu.lock`, opened with
`O_CREAT | O_EXCL`. The launcher acquires; the launcher releases. The
cron wrapper pre-checks so it doesn't waste API quota waking claude when
the GPU is busy. Stale lockfiles (owner PID dead) are self-healed.

### 4. The trainer can hang

**Solution**: the launcher wraps the trainer in `subprocess.run(...,
timeout=...)`. If the timeout fires (90 min default, 150 min for stress),
the launcher SIGTERMs the trainer, marks the row failed, and releases
the lock. F2: any stale lock left by a previous hung trainer is cleaned
on the next launcher invocation.

### 5. The agent must converge or stop, not loop forever

**Solution**: the launcher enforces three halt conditions:

- `nan_cascade`: two consecutive xattn runs ended with NaN loss.
- `zero_gate_activation`: two consecutive xattn runs ended with max_gate
  < 0.005 at step 1500.
- `convergence`: no improvement of ≥0.005 absolute HN-FPR over the last 4
  valid xattn runs (after ≥6 valid xattn runs have completed).

When a halt fires, the launcher refuses new launches. The agent reads
`sweep_state.halted: true` and writes the daily README section instead.

In the post-Day-3 Expanded Sweep, the user authorized disabling
`zero_gate_activation` and `convergence` (both were correctly firing but
were the very signal the expansion wanted to probe further). `nan_cascade`
and the GPU-hours cap remain the real stops.

### 6. Experiments can outlive a single 180-minute claude session

**Solution**: this took two iterations.

- **F6** (first attempt): launch the trainer with
  `start_new_session=True` so it survives the launcher's death. Insufficient:
  when claude dies, the launcher dies, the trainer's stdout pipe breaks,
  SIGPIPE kills the trainer.
- **F8 v1** (second attempt): launcher does `os.setsid()` to leave
  claude's session. Insufficient: claude's Bash tool puts each command
  into its own process group, so the launcher is already a PG leader
  and `setsid()` returns EPERM. Silently fails.
- **F8 v2** (current): launcher installs `SIGHUP` and `SIGPIPE` as
  `SIG_IGN`, redirects stdout/stderr to `<run_dir>/launcher.log`. The
  signal handlers are the actual defense; the redirect prevents the
  SIGPIPE path entirely. When claude dies at 180m, the launcher catches
  SIGHUP and continues, the trainer continues uninterrupted, the row
  lands normally. The next cron tick starts a fresh claude.

Verification: every `launcher.log` in v2-era run dirs starts with
`[launcher] survival mode active … state=shared-session-but-SIGHUP-ignored`.

### 7. The 180-minute timeout itself is a real cap (not unlimited claude)

**Solution**: each cron-tick claude session has a hard 180m wall-clock
ceiling. The agent's prompt encourages doing ONE iteration per wake-up
and exiting cleanly. In practice, the agent can fit 2-3 47-min experiments
back-to-back inside one 180m window before exiting. F8 v2 ensures the
final experiment of a window can straddle the boundary safely.

---

## The journey of one experiment, end-to-end

Let's trace `exp_xa_grid_017` (the run at the time of this writing).

1. **10:18Z**: prior experiment `grid_016` lands. Launcher releases lock,
   appends row to `experiments.jsonl`, updates `sweep_state.yaml`.
2. **10:18Z**: agent (still inside the 09:30 claude session) reads the
   new state. Sees `gate=zero × insertion=late_only × slots=128` still
   uncovered. Decides to propose that cell.
3. **11:05Z**: agent writes
   `src/auto_research/runs/exp_xa_grid_017/config.yaml`.
4. **11:08Z**: agent invokes `python scripts/run_next_experiment.py
   src/auto_research/runs/exp_xa_grid_017/config.yaml`.
5. **11:08Z**: launcher validates (✓), dedups (✓ no prior tuple match),
   halt-checks (✓), acquires the GPU lockfile, installs F8 v2 signal
   handlers + log redirect, launches accelerate-launch.
6. **11:08-11:54Z**: trainer trains for 1500 steps at seq_len=2048.
   `gate_trajectory.json` records per-step max/mean gate magnitudes.
7. **11:30Z**: 09:30 claude session's `timeout 180m` would fire here in a
   pre-F8 world. With F8 v2, claude dies, the launcher catches SIGHUP
   and continues. (At the time of writing, this is the first real
   F8 v2 boundary test — pending observation.)
8. **11:30Z**: fresh cron tick fires. Sees lockfile owned by alive PID
   85884. Exits cleanly without invoking a new claude.
9. **~11:55Z**: trainer exits. Launcher post-processes: clean-eval mask,
   three-mode scoring, per-family HN-FPR, bootstrap CIs, atomic row
   append, sweep_state update.
10. **~11:55Z**: launcher releases lock and exits. Row visible.
11. **12:00Z (next cron tick)**: fresh claude wakes up, reads new state,
    sees grid_017 landed, writes `notes.md` for grid_017, decides what
    to do next.

The whole experiment never depended on a single human being awake.

---

## What the loop is *not* good at

- **Picking a fundamentally new metric.** If `hn_fpr_worst_stripped`
  turns out to be the wrong objective, the agent can't switch on its
  own. The launcher's ranking is encoded in code.
- **Recovering from a permanent failure.** If the GPU is wedged or the
  W&B remote is down, the loop will mark experiments failed but won't
  fix the underlying issue.
- **Designing the next architectural change.** Within the sweep_space,
  the agent perturbs. Across architectures (e.g., switching x-attn for
  a different conditioning mechanism), a human picks the next direction.

What it IS good at: **exhaustively probing a dial space while a human
sleeps, with all results checked in, all CIs computed, all halts enforced**.

---

## F-numbered fix lineage (for the historian)

| Fix | What broke | What was added |
|---|---|---|
| F1 | initial loop never written down | RUNBOOK §6, AGENT_INSTRUCTIONS.md |
| F2 | trainer hung leaves stale GPU lock → loop wedges | self-heal: launcher checks lock-owner PID liveness, removes if dead |
| F3 | settings.json `permissions.allow` had wrong path grammar (absolute paths needed `//`, repo-root needed `/`) | corrected DSL grammar |
| F4 | Codex CLI dispatch needed different env vars than claude | `AGENT_CLI=codex` branch in agent_tick.sh |
| F5 | bnb / bitsandbytes / Blackwell compat | bnb sanity script in preflight, adamw_8bit fallback |
| F6 | trainer subprocess died with launcher on cron timeout | `start_new_session=True` for accelerate-launch spawn |
| F7 | claude CLI could wedge in pre-launcher state | `timeout 180m` outer cap on the CLI invocation |
| F8 v1 | F6 wasn't enough; launcher death broke trainer's pipe | `os.setsid()` + stdout redirect (silently failed EPERM) |
| F8 v2 | F8 v1 setsid silently failed | SIGHUP/SIGPIPE → SIG_IGN explicitly |

Each fix was discovered by a specific failure mode in production. The
loop is robust now because every wedge case has been hit and patched.

---

## Reading order

If you want to extend or debug this loop:

1. This document (you're here).
2. `RUNBOOK.md` — setup and operational sequence.
3. `src/auto_research/AGENT_INSTRUCTIONS.md` — what the agent reads.
4. `scripts/agent_tick.sh` — the cron wrapper.
5. `scripts/run_next_experiment.py` — the launcher.
6. `src/auto_research/configs/budget.yaml` and `sweep_space.yaml` —
   the dial space and halt conditions.
7. `docs/experiments-log.md` — what this loop actually produced.
