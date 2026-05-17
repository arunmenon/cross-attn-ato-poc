#!/usr/bin/env bash
# Cron-driven auto-research agent tick.
#
# Default loop mechanism per PLAN.md "How the loop is invoked" /
# RUNBOOK.md §6. Each tick:
#   1. cd to repo root (derived from this script's location).
#   2. Source /workspace/.env if present (HF_HOME, WANDB_DIR, etc. —
#      the trainer that the agent spawns reads these).
#   3. Activate the venv.
#   4. Pre-check the GPU lockfile — if an experiment is already
#      running, exit IMMEDIATELY before invoking the CLI. (Review 013
#      follow-on: previously the script went straight to the CLI and
#      burned a claude quota every 30 min during a 90-min experiment,
#      only to have the launcher reject on lock contention after the
#      agent had already read state and proposed a config.)
#   5. Check the launcher's halt status; exit if halted.
#   6. Pipe the standard loop prompt into the selected CLI (claude or
#      codex). The CLI runs ONE cycle of work and exits.
#
# Cron entry (set up Day 0 / Hr 2-3):
#   */30 * * * * /workspace/cross_attn_ato_poc/scripts/agent_tick.sh >> /workspace/agent_tick.log 2>&1
#
# Configurable env vars:
#   AGENT_CLI         claude | codex  (default: claude)
#   AGENT_VENV        path to venv to activate (default: /workspace/.venv)
#   AGENT_ENV_FILE    path to .env to source first (default: /workspace/.env)
#   GPU_LOCK_FILE     path to launcher's GPU lockfile (default: /workspace/.gpu.lock)
#   DRY_RUN           1 prints the command without launching the CLI
#
# Concurrency safety: this script holds NO tick-level lock; the GPU
# lock is owned by scripts/run_next_experiment.py for the duration of
# an experiment. Two overlapping ticks were always "safe" in that they
# don't crash, but the pre-check above means we ALSO don't waste
# Claude/OpenAI quota waking the agent during an active experiment.
#
# Review 008 finding #2 — this script previously did not exist.
# Review 013 follow-on (post-014 explanation) — added the GPU-lock
# pre-check, .env sourcing, and synced the LOOP_PROMPT with the
# post-pivot RUNBOOK §6 text (HN-FPR ranking, launcher owns state).

set -euo pipefail

# ---------------------------------------------------------------------------
# Derive paths (script-location-relative; layout-independent)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="${AGENT_VENV:-/workspace/.venv}"
CLI="${AGENT_CLI:-claude}"
ENV_FILE="${AGENT_ENV_FILE:-/workspace/.env}"
GPU_LOCK_FILE="${GPU_LOCK_FILE:-/workspace/.gpu.lock}"

DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Loop prompt (RUNBOOK §6 verbatim)
# ---------------------------------------------------------------------------

LOOP_PROMPT='You are the auto-research agent for the cross-attention ATO POC.

Read src/auto_research/AGENT_INSTRUCTIONS.md and follow it.

Read src/auto_research/sweep_state.yaml to see budget remaining and current best.
Read the last 5 entries of src/auto_research/experiments.jsonl for history.

If a halt condition is met, stop launching new experiments and write the
Day-2 or Day-3 README section as appropriate.

Otherwise, propose the next experiment by writing
src/auto_research/runs/exp_NNN/config.yaml, then run:

    python scripts/run_next_experiment.py src/auto_research/runs/exp_NNN/config.yaml

When it completes, the launcher has ALREADY appended a record to
src/auto_research/experiments.jsonl and updated src/auto_research/sweep_state.yaml.
Do NOT write either file from the agent side — the launcher owns them
(review 013 finding #7; matches src/auto_research/AGENT_INSTRUCTIONS.md).
Your job is to READ those files, plus the run'\''s metrics.json and
ci_report.json, then decide what config to propose next. You may write
src/auto_research/runs/exp_NNN/notes.md if you want to record your
reasoning for future reads.

Ranking + halt: the launcher ranks experiments by worst-family
hard-negative FPR (lower is better; tiebreaker is mean HN-FPR), not by
AUC (AUC is saturated at 1.0 on every model variant — review 013
finding #1). The auto-loop halts when worst-family HN-FPR has not
improved by >= 0.005 absolute over the last 4 valid x-attn runs (after
at least 6 valid x-attn runs have completed). Baselines (cpt_light,
lora_text, structured_as_text, event_only) are recorded in
experiments.jsonl for Day-3 comparison but do not count toward the
x-attn sweep budget or convergence count.

Before editing any source code file, run:
    git add -A && git commit -m "snapshot before <change description>"
'

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

cd "$REPO_ROOT"

# Source .env BEFORE the venv so trainer-facing env vars (HF_HOME,
# WANDB_DIR, TRANSFORMERS_CACHE, TOKENIZERS_PARALLELISM, etc.) are set
# in the environment that claude inherits when it spawns the trainer.
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

if [[ ! -d "$VENV" ]]; then
    echo "[agent_tick] venv not found at $VENV; aborting" >&2
    exit 2
fi
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "[agent_tick] $VENV missing bin/activate; aborting" >&2
    exit 2
fi

# Activate venv (for python deps + CLI path)
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# GPU-lock pre-check: if an experiment is already running, exit IMMEDIATELY
# before invoking the CLI. Without this, every 30-minute tick during a
# 90-minute experiment wakes the agent up, makes it read state, and
# proposes a config that the launcher then rejects with lock contention
# — burning Claude/OpenAI quota for nothing. The launcher still owns
# the lock as the authoritative concurrency gate; this is just a polite
# pre-check so we don't bother the CLI when the GPU is busy.
if [[ -f "$GPU_LOCK_FILE" ]]; then
    LOCK_PID="$(cat "$GPU_LOCK_FILE" 2>/dev/null || echo unknown)"
    echo "[agent_tick] $(date -u +%Y-%m-%dT%H:%M:%SZ) GPU lock held (PID=$LOCK_PID); skipping this tick"
    exit 0
fi

# Halt check: if the launcher says the budget is exhausted or
# convergence has been reached, exit promptly without invoking the CLI.
# This script's exit code is informational for cron logs.
if ! python scripts/run_next_experiment.py --halt-check; then
    echo "[agent_tick] launcher reports halted; skipping this tick"
    exit 0
fi

# ---------------------------------------------------------------------------
# Launch the CLI (or print in dry-run mode)
# ---------------------------------------------------------------------------

if ! command -v "$CLI" >/dev/null 2>&1; then
    echo "[agent_tick] $CLI not on PATH after venv activation; aborting" >&2
    exit 2
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[agent_tick] DRY_RUN=1; would have run:"
    echo "  cd $REPO_ROOT"
    echo "  source $VENV/bin/activate"
    echo "  $CLI <<< <loop prompt (omitted, see RUNBOOK §6)>"
    exit 0
fi

# Pipe the loop prompt into the selected CLI. The CLI is expected to
# perform ONE iteration of agent work and exit.
echo "[agent_tick] $(date -u +%Y-%m-%dT%H:%M:%SZ) launching $CLI"
printf '%s' "$LOOP_PROMPT" | "$CLI"
echo "[agent_tick] $(date -u +%Y-%m-%dT%H:%M:%SZ) tick complete"
