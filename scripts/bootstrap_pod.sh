#!/usr/bin/env bash
# Bootstrap a fresh RunPod container for the cross_attn_ato_poc work.
#
# Assumes:
#   - /workspace is mounted from the persistent network volume that
#     already has /workspace/.venv, /workspace/cross_attn_ato_poc,
#     /workspace/data, /workspace/checkpoints (i.e. a pod restart, not
#     a from-scratch network volume).
#   - The container is a Runpod Pytorch image (any version; we use the
#     venv, not the container's torch).
#
# What this script does (idempotent — safe to re-run):
#   1. Install apt packages: cron, rsync, tmux.
#   2. Install Node.js 20 via NodeSource (needed by Claude Code CLI).
#   3. Install Claude Code CLI globally.
#   4. Install /etc/cron.d/auto_research_loop (every 30 min agent tick).
#   5. Start the cron daemon.
#   6. Print a checklist of remaining manual steps (Claude OAuth, env vars).
#
# What this script does NOT do:
#   - Generate a Claude OAuth token. That requires an interactive
#     browser flow (`claude setup-token`). Run it manually after this
#     script and export CLAUDE_CODE_OAUTH_TOKEN.
#   - Touch /workspace contents. The whole point of the network volume
#     is that data, code, venv, and checkpoints survive container loss.
#
# Usage:
#   curl -fsSL https://... or scp this file to the pod, then:
#       bash /workspace/cross_attn_ato_poc/scripts/bootstrap_pod.sh
#
# Verification after running:
#   - `cron` running: pgrep -af cron
#   - Claude CLI on PATH: claude --version
#   - Cron entry: cat /etc/cron.d/auto_research_loop
#   - Token set: echo $CLAUDE_CODE_OAUTH_TOKEN | head -c 12  (should
#     print "sk-ant-oat0" if exported from /workspace/.env)

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/cross_attn_ato_poc}"
VENV_DIR="${VENV_DIR:-/workspace/.venv}"

echo "[bootstrap] starting at $(date -u +%FT%TZ)"

# ----------------------------------------------------------------------
# Sanity: confirm the network volume is mounted with the expected layout.
# ----------------------------------------------------------------------
echo "[bootstrap] verifying /workspace layout..."
for p in "$REPO_DIR" "$VENV_DIR" /workspace/data /workspace/checkpoints; do
    if [[ ! -e "$p" ]]; then
        echo "[bootstrap] ERROR: $p missing — is the network volume mounted?" >&2
        exit 2
    fi
done
echo "[bootstrap] /workspace looks right (repo + venv + data + checkpoints present)"

# ----------------------------------------------------------------------
# 1. apt packages
# ----------------------------------------------------------------------
echo "[bootstrap] installing apt packages (cron, rsync, tmux)..."
apt-get update -qq
apt-get install -y -qq cron rsync tmux

# ----------------------------------------------------------------------
# 2. Node.js 20 via NodeSource (Claude Code CLI requires >=18)
# ----------------------------------------------------------------------
NODE_NEED_INSTALL=1
if command -v node >/dev/null 2>&1; then
    NODE_MAJOR=$(node --version | sed -E 's/^v([0-9]+).*/\1/')
    if [[ "$NODE_MAJOR" -ge 18 ]]; then
        echo "[bootstrap] Node $(node --version) already installed; skipping NodeSource"
        NODE_NEED_INSTALL=0
    else
        echo "[bootstrap] Node $(node --version) too old for Claude Code; reinstalling"
        apt-get purge -y -qq nodejs libnode-dev libnode72 npm 2>/dev/null || true
        apt-get autoremove -y -qq 2>/dev/null || true
    fi
fi
if [[ "$NODE_NEED_INSTALL" -eq 1 ]]; then
    # Manual NodeSource setup — no curl|bash pipe (some auto-classifiers
    # block that and security-conscious operators reject it too).
    curl -fsSL -o /tmp/nodesource_keyring.gpg https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key
    gpg --dearmor -o /usr/share/keyrings/nodesource.gpg /tmp/nodesource_keyring.gpg 2>&1 ||
        gpg --no-default-keyring --keyring /usr/share/keyrings/nodesource.gpg --import /tmp/nodesource_keyring.gpg
    echo 'deb [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main' \
        > /etc/apt/sources.list.d/nodesource.list
    apt-get update -qq
    apt-get install -y -qq nodejs
    echo "[bootstrap] Node $(node --version) installed"
fi

# ----------------------------------------------------------------------
# 3. Claude Code CLI
# ----------------------------------------------------------------------
if command -v claude >/dev/null 2>&1; then
    echo "[bootstrap] claude $(claude --version 2>&1 | head -1) already installed"
else
    echo "[bootstrap] installing Claude Code CLI..."
    npm install -g @anthropic-ai/claude-code
    echo "[bootstrap] $(claude --version 2>&1 | head -1) installed"
fi

# ----------------------------------------------------------------------
# 4. Cron entry — every 30 min agent tick
# ----------------------------------------------------------------------
CRON_FILE=/etc/cron.d/auto_research_loop
if [[ -f "$CRON_FILE" ]]; then
    echo "[bootstrap] cron entry already at $CRON_FILE"
else
    echo "[bootstrap] installing $CRON_FILE"
    cat > "$CRON_FILE" <<EOF
# Auto-research loop tick — every 30 minutes.
# Exits rc=2 with 'aborting' until CLAUDE_CODE_OAUTH_TOKEN is set in
# the env (sourced from /workspace/.env by agent_tick.sh).
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
*/30 * * * * root $REPO_DIR/scripts/agent_tick.sh >> /workspace/agent_tick.log 2>&1
EOF
fi

# ----------------------------------------------------------------------
# 5. Start cron daemon
# ----------------------------------------------------------------------
if ! pgrep -x cron >/dev/null; then
    echo "[bootstrap] starting cron daemon"
    /usr/sbin/cron
else
    echo "[bootstrap] cron daemon already running (PID $(pgrep -x cron))"
fi

# ----------------------------------------------------------------------
# 6. Remaining manual steps
# ----------------------------------------------------------------------
echo
echo "================ BOOTSTRAP COMPLETE ================"
echo
echo "Remaining manual steps before the auto-loop runs:"
echo
echo "  a) Verify OpenAI key is set in /workspace/.env:"
echo "       grep -c OPENAI_API_KEY /workspace/.env"
echo
echo "  b) Generate / persist Claude OAuth token (if not already in"
echo "     /workspace/.env from a prior session):"
echo "       tmux new-session -d -s claude_login 'claude setup-token; sleep 600'"
echo "       tmux capture-pane -t claude_login -p -S -100   # find the URL"
echo "     Open the URL in your browser, complete OAuth, copy the"
echo "     verification code, then:"
echo "       tmux attach -t claude_login   # paste code, press Enter"
echo "       # Detach with Ctrl-B then D"
echo "     Append the resulting token to /workspace/.env:"
echo "       echo 'export CLAUDE_CODE_OAUTH_TOKEN=<your-token>' >> /workspace/.env"
echo
echo "  c) (Optional) Reset sweep_state.yaml for a new sweep — the"
echo "     existing file may still show v4 state. Either edit"
echo "     $REPO_DIR/src/auto_research/sweep_state.yaml or delete"
echo "     the run history before the agent picks up where it left off."
echo
echo "  d) Verify a tick works manually:"
echo "       DRY_RUN=1 $REPO_DIR/scripts/agent_tick.sh"
echo "     (DRY_RUN=1 prints what it would do without actually invoking"
echo "      the trainer; should exit 0 if everything is wired)"
echo
echo "First scheduled tick: the next :00 or :30 minute boundary."
echo "Log: /workspace/agent_tick.log"
