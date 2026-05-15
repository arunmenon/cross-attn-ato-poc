#!/usr/bin/env bash
# Cron-driven auto-research agent tick.
#
# Default loop mechanism per PLAN.md "How the loop is invoked" /
# RUNBOOK.md §6. Each tick:
#   1. cd to repo root (derived from this script's location).
#   2. Activate the venv.
#   3. Check the launcher's halt status; exit if halted.
#   4. Pipe the standard loop prompt into the selected CLI (claude or
#      codex). The CLI runs ONE cycle of work and exits.
#
# Cron entry (set up Day 0 / Hr 2-3):
#   */30 * * * * /workspace/cross_attn_ato_poc/scripts/agent_tick.sh >> /workspace/agent_tick.log 2>&1
#
# Configurable env vars:
#   AGENT_CLI         claude | codex  (default: claude)
#   AGENT_VENV        path to venv to activate (default: /workspace/.venv)
#   DRY_RUN           1 prints the command without launching the CLI
#
# Concurrency safety: scripts/run_next_experiment.py holds the GPU lock,
# so two overlapping ticks are safe — the second will fail to acquire
# the lock and return promptly. We do NOT need a separate tick-level
# lock.
#
# Review 008 finding #2 — this script previously did not exist.

set -euo pipefail

# ---------------------------------------------------------------------------
# Derive paths (script-location-relative; layout-independent)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="${AGENT_VENV:-/workspace/.venv}"
CLI="${AGENT_CLI:-claude}"

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

When it completes, read the run'\''s metrics.json and ci_report.json, and
write a one-paragraph summary to src/auto_research/runs/exp_NNN/notes.md.
Then update src/auto_research/sweep_state.yaml is owned by the launcher;
do NOT edit it yourself.

Before editing any source code file, run:
    git add -A && git commit -m "snapshot before <change description>"
'

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

cd "$REPO_ROOT"

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
