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

Otherwise, propose the next V5 queue experiment by writing
src/auto_research/runs/<exp_id>/config.yaml, then run:

    python scripts/run_next_experiment.py src/auto_research/runs/<exp_id>/config.yaml

When it completes, the launcher has ALREADY appended a record to
src/auto_research/experiments.jsonl and updated src/auto_research/sweep_state.yaml.
Do NOT write either file from the agent side — the launcher owns them
(review 013 finding #7; matches src/auto_research/AGENT_INSTRUCTIONS.md).
Your job is to READ those files, plus the run'\''s metrics.json and
ci_report.json, then decide what config to propose next. You may write
src/auto_research/runs/<exp_id>/notes.md if you want to record your
reasoning for future reads.

Ranking + halt: V5 ranks experiments by v5_adv_error (lower is better):
mean(1-recall(phish_takeover), 1-recall(phish_takeover_mfa_phished),
fpr(hn_recovery_high_amount)). Worst-family HN-FPR is the secondary
tiebreak metric; AUC is sanity-only. Baseline seed rows are recorded in
experiments.jsonl for comparison but do not count toward the x-attn
sweep budget or convergence count.

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
    if [[ "$LOCK_PID" =~ ^[0-9]+$ ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "[agent_tick] $(date -u +%Y-%m-%dT%H:%M:%SZ) GPU lock held by live PID=$LOCK_PID; skipping this tick"
        exit 0
    fi
    # Stale lock (owner dead, or PID unreadable): self-heal so the
    # auto-loop doesn't wedge until the user notices. The launcher's
    # acquire_lock() has the same check; this is the polite outer guard
    # so we don't even invoke the CLI when the lock is already dead.
    # Audit 014 Blocker 2 / audit 015 confirmed.
    echo "[agent_tick] $(date -u +%Y-%m-%dT%H:%M:%SZ) stale GPU lock (PID=$LOCK_PID); removing and proceeding"
    rm -f "$GPU_LOCK_FILE"
fi

# Halt check (informational only): run --halt-check so cron logs see
# the launcher's halt assessment, but do NOT short-circuit. The agent
# itself reads sweep_state.yaml at every iteration and handles halted
# state per AGENT_INSTRUCTIONS step 2 ("If halted: skip to Daily
# writeup"). Short-circuiting here would prevent the writeup from
# firing through cron — the original 2026-05-17 design wasted a manual
# CLI invocation to write the Day-2 README section. Keep the script
# thin and let the agent decide.
python scripts/run_next_experiment.py --halt-check || true

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
# perform ONE iteration of agent work and exit — but per
# AGENT_INSTRUCTIONS step 5, that single iteration includes blocking
# on `run_next_experiment.py` for the entire 90- or 150-minute
# launcher run, so a normal "iteration" can be ~150+ min long.
echo "[agent_tick] $(date -u +%Y-%m-%dT%H:%M:%SZ) launching $CLI"
# Wall-clock cap for the whole tick (review 016 finding 1 fix). The CLI
# STAYS ATTACHED through the launcher's experiment run per AGENT_INSTRUCTIONS
# step 5 ("When it completes, read..."). A normal x-attn experiment caps at
# 90 min (budget.yaml max_wall_clock_minutes); a stress run caps at 150 min;
# post-processing + notes adds a few minutes. The outer cap must be ABOVE
# that — 180m gives ~25 min of slack above stress runs, but still bounds a
# CLI that wedges in pre-launcher state (network stall, MCP server wedge).
# Default `timeout` sends SIGTERM to the immediate child only, so a wedged
# launcher orphans cleanly to init and finishes (accelerate trainer is in
# its own session via start_new_session=True in the launcher per F6).
set +e
printf '%s' "$LOOP_PROMPT" | timeout 180m "$CLI"
TICK_RC=$?
set -e
if [[ $TICK_RC -eq 124 ]]; then
    echo "[agent_tick] $(date -u +%Y-%m-%dT%H:%M:%SZ) CLI TIMED OUT after 180m (rc=124)"
fi
echo "[agent_tick] $(date -u +%Y-%m-%dT%H:%M:%SZ) tick complete (rc=$TICK_RC)"

# ---------------------------------------------------------------------------
# Auto-stop on sweep halt (cost-control safety net)
# ---------------------------------------------------------------------------
# Three independent halt triggers; ANY of them stops the pod:
#
#   1. Launcher-set: sweep_state.yaml has `halted: true`
#      Covers budget cap, NaN cascade, zero-gate, convergence — the
#      conditions the launcher itself records.
#
#   2. Launcher-budget mirror: count V5 x-attn rows + total wall-clock
#      from experiments.jsonl ourselves. Catches the case where the
#      launcher's halt check ran but the file write was incomplete, OR
#      where an agent restart skipped updating sweep_state.yaml.
#
#   3. Stale-tick detector: if `n_xattn_runs` in sweep_state.yaml has
#      not advanced across 2 consecutive ticks AND the GPU lockfile is
#      absent (no training in flight), the sweep has stopped making
#      progress. Catches the INSTRUCTION-LEVEL halts that the launcher
#      does NOT record: V5 Phase 2 stop rule, queue exhaustion, and
#      "early-exit-on-success-then-local-perturbations-exhausted".
#
# Required env (sourced from /workspace/.env earlier in this script):
#   RUNPOD_API_KEY  — must have write/admin scope; a read-only key will
#                     fail the mutation and leave the pod up.
#   RUNPOD_POD_ID   — static pod-id from the RunPod UI. Set in /workspace/.env
#                     for the current pod; the hardcoded fallback below is
#                     a defensive default that becomes wrong if the pod
#                     rotates and /workspace/.env isn't updated.
#
# Defensive: never crashes the tick. Logs failures so the operator can
# investigate via agent_tick.log. The next tick re-evaluates state and
# tries again if still halted.
RUNPOD_POD_ID="${RUNPOD_POD_ID:-b0b6dnykxttbgv}"
SWEEP_STATE="$REPO_ROOT/src/auto_research/sweep_state.yaml"
EXPERIMENTS_JSONL="$REPO_ROOT/src/auto_research/experiments.jsonl"
BUDGET_YAML="$REPO_ROOT/src/auto_research/configs/budget.yaml"
TICK_STATE_FILE=/workspace/.agent_tick_state.json
GPU_LOCK_FILE="${GPU_LOCK_FILE:-/workspace/.gpu.lock}"

SHOULD_STOP=0
HALT_REASON=""

# Trigger 1: launcher-set halted flag.
if [[ -f "$SWEEP_STATE" ]] && grep -qE '^halted:[[:space:]]*true' "$SWEEP_STATE"; then
    HALT_REASON="launcher halt: $(grep -E '^halt_reason:' "$SWEEP_STATE" | sed 's/^halt_reason:[[:space:]]*//')"
    SHOULD_STOP=1
fi

# Trigger 2: budget mirror + Trigger 3: stale-tick detector.
# Both done via one Python helper for cleaner state I/O.
if [[ $SHOULD_STOP -eq 0 ]]; then
    STOP_CHECK=$(SWEEP_STATE="$SWEEP_STATE" EXPERIMENTS_JSONL="$EXPERIMENTS_JSONL" \
                 BUDGET_YAML="$BUDGET_YAML" TICK_STATE_FILE="$TICK_STATE_FILE" \
                 GPU_LOCK_FILE="$GPU_LOCK_FILE" \
                 python3 - <<'PYEOF'
import json
import os
import sys
from pathlib import Path

sweep_state = Path(os.environ["SWEEP_STATE"])
experiments = Path(os.environ["EXPERIMENTS_JSONL"])
budget_yaml = Path(os.environ["BUDGET_YAML"])
tick_state_file = Path(os.environ["TICK_STATE_FILE"])
gpu_lock_file = Path(os.environ["GPU_LOCK_FILE"])

# Budget caps (mirror launcher logic, V5 x-attn rows only).
max_exp = 999
max_hr = 999
try:
    import yaml
    if budget_yaml.exists():
        b = yaml.safe_load(budget_yaml.read_text()) or {}
        max_exp = int(b.get("max_experiments", 999))
        max_hr = float(b.get("max_gpu_hours", 999))
except Exception:
    pass

n_xattn_v5 = 0
gpu_min = 0.0
if experiments.exists():
    for line in experiments.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if (r.get("arm") == "xattn"
            and r.get("metric_version", 1) >= 5
            and r.get("status") == "ok"):
            n_xattn_v5 += 1
        gpu_min += float(r.get("wall_clock_min") or 0)
gpu_hr = gpu_min / 60.0

if n_xattn_v5 >= max_exp:
    print(f"STOP|budget mirror: V5 x-attn count {n_xattn_v5} >= max_experiments {max_exp}")
    sys.exit(0)
if gpu_hr >= max_hr:
    print(f"STOP|budget mirror: gpu_hours_used {gpu_hr:.2f} >= max_gpu_hours {max_hr}")
    sys.exit(0)

# Stale-tick detector. Compare n_xattn_runs in sweep_state.yaml against
# the value we recorded after the previous tick. Increment a stale
# counter when it hasn't moved AND the GPU lockfile is absent (so we
# know no training is mid-flight, which is the legitimate "long wall
# clock" case).
current_n = None
if sweep_state.exists():
    for line in sweep_state.read_text().splitlines():
        if line.startswith("n_xattn_runs:"):
            try:
                current_n = int(line.split(":", 1)[1].strip())
            except Exception:
                pass
            break

if current_n is None:
    print("OK|n_xattn_runs not parseable; skipping stale detector")
    sys.exit(0)

prior = {"last_n_xattn_runs": -1, "stale_count": 0}
if tick_state_file.exists():
    try:
        prior = json.loads(tick_state_file.read_text())
    except Exception:
        pass
last_n = prior.get("last_n_xattn_runs", -1)
stale = prior.get("stale_count", 0)

gpu_locked = gpu_lock_file.exists()

if current_n == last_n and not gpu_locked:
    stale += 1
else:
    stale = 0

# Persist for next tick.
tick_state_file.write_text(json.dumps(
    {"last_n_xattn_runs": current_n, "stale_count": stale}
))

# 2 consecutive ticks (= 60 min) of no new runs + GPU idle => done.
if stale >= 2:
    print(f"STOP|stale detector: n_xattn_runs={current_n} unchanged across {stale} ticks, GPU idle")
else:
    print(f"OK|n_xattn_v5={n_xattn_v5}/{max_exp} gpu_hr={gpu_hr:.2f}/{max_hr} stale={stale} (current_n={current_n}, last_n={last_n}, gpu_locked={gpu_locked})")
PYEOF
)
    echo "[agent_tick] halt-check: $STOP_CHECK"
    if [[ "${STOP_CHECK:0:5}" == "STOP|" ]]; then
        HALT_REASON="${STOP_CHECK:5}"
        SHOULD_STOP=1
    fi
fi

# Fire the stop if any trigger fired.
if [[ $SHOULD_STOP -eq 1 ]]; then
    echo "[agent_tick] $(date -u +%Y-%m-%dT%H:%M:%SZ) AUTO-STOP triggered: ${HALT_REASON}"
    if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
        echo "[agent_tick] auto-stop SKIPPED: RUNPOD_API_KEY not in env (add to /workspace/.env to enable)"
    elif ! command -v curl >/dev/null 2>&1; then
        echo "[agent_tick] auto-stop SKIPPED: curl not on PATH"
    else
        echo "[agent_tick] auto-stop: calling podStop on $RUNPOD_POD_ID..."
        set +e
        STOP_RESP=$(curl -sS -w '\n__HTTP__:%{http_code}' \
            --max-time 30 \
            -X POST https://api.runpod.io/graphql \
            -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}" 2>&1)
        STOP_RC=$?
        set -e
        echo "[agent_tick] auto-stop curl rc=$STOP_RC"
        echo "[agent_tick] auto-stop response:"
        echo "$STOP_RESP" | sed 's/^/  /'
        if echo "$STOP_RESP" | grep -q '"desiredStatus"'; then
            echo "[agent_tick] auto-stop ACCEPTED by API; container should terminate shortly"
        else
            echo "[agent_tick] auto-stop did NOT receive desiredStatus — pod may still be running (check key scope / pod id)"
        fi
    fi
fi
