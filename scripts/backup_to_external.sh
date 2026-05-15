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

REPO_ROOT="${REPO_ROOT:-/workspace/repo/cross_attn_ato_poc}"
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

# Top-3 checkpoints (rotating; identified by sweep_state.yaml's current_best
# and the agent's marked top_3 list).
TOP3_FILE="$REPO_ROOT/src/auto_research/sweep_state.yaml"
if [[ -f "$TOP3_FILE" ]] && command -v python >/dev/null; then
    TOP3=$(python -c "
import yaml, sys
state = yaml.safe_load(open('$TOP3_FILE'))
for exp_id in state.get('top_3', []):
    print(exp_id)
" || true)
    for exp_id in $TOP3; do
        ckpt_dir="/workspace/checkpoints/xattn_$exp_id"
        if [[ -d "$ckpt_dir" ]]; then
            log "syncing checkpoint $exp_id"
            case "$TARGET" in
                s3://*) aws s3 sync "$ckpt_dir" "${TARGET}checkpoints/xattn_$exp_id/" --no-progress >/dev/null ;;
                r2://*) rclone sync "$ckpt_dir" "${TARGET}checkpoints/xattn_$exp_id/" --quiet ;;
                hf://*) huggingface-cli upload "${TARGET#hf://}" "$ckpt_dir" "checkpoints/xattn_$exp_id" --quiet ;;
            esac
        fi
    done
fi

log "backup done"
