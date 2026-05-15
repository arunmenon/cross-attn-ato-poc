#!/usr/bin/env bash
# Periodic backup of critical artifacts to external storage.
#
# Cron entry (Day-0 / Hr 2-3, run on the RunPod box):
#   */30 * * * * /workspace/repo/cross_attn_ato_poc/scripts/backup_to_external.sh >> /workspace/backup.log 2>&1
#
# Configure target via env var BACKUP_TARGET, one of:
#   s3://<bucket>/cross-attn-ato-poc/
#   r2://<bucket>/cross-attn-ato-poc/   (requires rclone alias 'r2')
#   hf://datasets/<user>/cross-attn-ato-poc-artifacts/
#
# What gets backed up: experiments.jsonl, sweep_state.yaml, runs/*/metrics.json,
# runs/*/ci_report.json, runs/*/leakage_report.json, README.md, top-3 checkpoints.
# What does NOT: failed-run checkpoints, HF cache, W&B offline logs.

set -euo pipefail

# Derive REPO_ROOT from script location so the script is portable across
# clone layouts (review 008 finding #1). The hard-coded
# `/workspace/repo/cross_attn_ato_poc` default was wrong for this repo
# (this IS the repo root; the clone path is /workspace/cross_attn_ato_poc).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TARGET="${BACKUP_TARGET:-}"
LOCK="/workspace/.backup.lock"

if [[ -z "$TARGET" ]]; then
    echo "BACKUP_TARGET not set; skipping" >&2
    exit 0
fi

# Concurrency lock — don't overlap if a previous run is still going.
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "backup already in progress (lock $LOCK exists); skipping" >&2
    exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

log "backup start; target=$TARGET"

ARTIFACTS=(
    "$REPO_ROOT/src/auto_research/experiments.jsonl"
    "$REPO_ROOT/src/auto_research/sweep_state.yaml"
    "$REPO_ROOT/README.md"
)

# Per-run metric/CI/leakage files (lightweight)
RUNS_DIR="$REPO_ROOT/src/auto_research/runs"
if [[ -d "$RUNS_DIR" ]]; then
    while IFS= read -r f; do
        ARTIFACTS+=("$f")
    done < <(find "$RUNS_DIR" -maxdepth 2 -type f \
              \( -name 'metrics.json' -o -name 'ci_report.json' -o -name 'leakage_report.json' -o -name 'config.yaml' \))
fi

# Sync small artifacts
for src in "${ARTIFACTS[@]}"; do
    if [[ ! -f "$src" ]]; then
        continue
    fi
    rel="${src#$REPO_ROOT/}"
    dst="${TARGET}${rel}"
    case "$TARGET" in
        s3://*) aws s3 cp "$src" "$dst" --no-progress >/dev/null ;;
        r2://*) rclone copy "$src" "$dst" --quiet ;;
        hf://*)
            # HF backup is slower; collect into a staging dir and push once at the end
            STAGING="${STAGING:-/tmp/hf-stage-$$}"
            mkdir -p "$(dirname "$STAGING/$rel")"
            cp "$src" "$STAGING/$rel"
            ;;
        *)
            log "unknown target scheme: $TARGET"
            exit 2
            ;;
    esac
done

if [[ "${STAGING:-}" != "" ]]; then
    log "pushing staged files to HF: $TARGET"
    huggingface-cli upload "${TARGET#hf://}" "$STAGING" . --quiet
    rm -rf "$STAGING"
fi

# Top-3 x-attn checkpoints (rotating). Review 008 finding #5 fixed two
# bugs at once:
#   (a) the old YAML parser printed each top_3 entry as a Python dict
#       (`{'exp_id': 'exp_001', 'auc_stripped': 0.8}`), so the shell
#       for-loop received garbage tokens like `{'exp_id':` and never
#       resolved a real exp_id.
#   (b) checkpoints were looked up under /workspace/checkpoints/xattn_*
#       but train_xattn.py actually saves xattn_state.pt under the run
#       directory at src/auto_research/runs/exp_NNN/. Nothing was
#       getting backed up.
TOP3_FILE="$REPO_ROOT/src/auto_research/sweep_state.yaml"
if [[ -f "$TOP3_FILE" ]] && command -v python3 >/dev/null; then
    TOP3=$(python3 - <<PYEOF || true
import yaml
state = yaml.safe_load(open("$TOP3_FILE")) or {}
for entry in state.get("top_3", []) or []:
    # Each entry is now {exp_id: ..., auc_stripped: ...}; we want JUST exp_id.
    if isinstance(entry, dict) and "exp_id" in entry:
        print(entry["exp_id"])
    elif isinstance(entry, str):
        print(entry)
PYEOF
)
    for exp_id in $TOP3; do
        # Per-run directory where train_xattn.py writes xattn_state.pt
        # and where the launcher records metrics.json + ci_report.json
        # + gate_trajectory.json. Backing this up gives us everything
        # needed to resurrect a top-3 run on a new pod.
        run_dir="$REPO_ROOT/src/auto_research/runs/$exp_id"
        if [[ -d "$run_dir" ]]; then
            log "syncing top-3 run dir $exp_id"
            case "$TARGET" in
                s3://*) aws s3 sync "$run_dir" "${TARGET}runs/$exp_id/" --no-progress >/dev/null ;;
                r2://*) rclone sync "$run_dir" "${TARGET}runs/$exp_id/" --quiet ;;
                hf://*) huggingface-cli upload "${TARGET#hf://}" "$run_dir" "runs/$exp_id" --quiet ;;
            esac
        else
            log "top-3 run dir $exp_id not found at $run_dir; skipping"
        fi
    done
fi

log "backup done"
