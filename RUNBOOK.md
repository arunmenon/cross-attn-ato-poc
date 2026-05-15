# RUNBOOK — Cross-Attention ATO POC

Operational reference for running this POC on RunPod. Read in full before booting the pod.

---

## 1. RunPod pod setup

### 1.1 Network volume (mandatory)

Create a RunPod **Network Volume** *before* launching the pod (volumes only attach at pod creation):

- Size: ≥ 200 GB
- Region: match GPU region
- Name: `cross-attn-ato-poc`

Everything that must survive a pod restart goes on the network volume. Container disk is treated as scratch.

### 1.2 Pod template

- GPU: 1× H100 80GB SXM (or PCIe if SXM unavailable)
- Base image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or equivalent — pin the version, do not use `latest`)
- Volume mount: network volume → `/workspace`
- Ports: 8888 (Jupyter, optional), 22 (SSH)
- Env vars (set in pod template):
  ```
  HF_HOME=/workspace/.hf
  WANDB_DIR=/workspace/.wandb
  TRANSFORMERS_CACHE=/workspace/.hf/transformers
  TOKENIZERS_PARALLELISM=false
  ```

### 1.3 First-boot bootstrap

```bash
cd /workspace
git clone <your-repo-url> repo
cd repo/cross_attn_ato_poc
python -m venv /workspace/.venv
source /workspace/.venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
python scripts/preflight_check.py
```

`preflight_check.py` exits non-zero on any environment problem. Do **not** proceed past failed preflight.

---

## 2. Storage layout (everything under `/workspace`)

```
/workspace/
├── .venv/                      Python virtualenv
├── .hf/                        HF cache (models, datasets)
├── .wandb/                     W&B offline logs
├── repo/                       This repository (git clone)
├── data/                       Generated synthetic datasets
│   ├── train_llm_narrated/     20-30k LLM-narrated pairs
│   ├── eval_fast_5k/           5k stratified eval
│   ├── eval_medium_50k/        50k templated medium eval
│   └── eval_large/             100-200k templated large eval (optional)
├── checkpoints/
│   ├── qwen3-8b-cpt-light-lora/   Stage-0 LoRA (pre-merge)
│   ├── qwen3-8b-cpt-light-merged/ Stage-0 post-merge — baseline #1, Stage-1 start
│   ├── baseline_lora_text/
│   ├── baseline_structured_as_text/
│   ├── baseline_event_only/
│   └── xattn_exp_NNN/          Top-3 only; failed sweep checkpoints not kept
└── experiments/                Synced from repo/src/auto_research/runs/ via backup
```

---

## 3. Day-by-day operational sequence

### Day 0 (locally, before pod)

Already complete if this file exists.

### Day 1

1. Boot pod, run §1.3 bootstrap.
2. Activate `/workspace/.venv`.
3. Run Hr 0-2 tokens/fencer/bucketer task (Task #31).
4. Run vertical-slice task (Task #32). **Do not proceed past Hr 8 if this fails.**
5. Run scale + Stage-0 CPT-light task (Task #33). Takes 3-4 hours.
6. Run `scripts/merge_stage0_lora.py` to produce `qwen3-8b-cpt-light-merged`.
7. Run CPT eval task (Task #34). Write Day-1 README section by hand.

### Day 2

1. X-attn architecture surgery (Task #35).
2. Three more baselines: LoRA-text, structured-as-text, event-only classifier (Task #36).
3. First x-attn smoke run (Task #37).
4. Start agent loop. Either:
   - **Cron-driven (default)**: `crontab -e` and add `*/30 * * * * /workspace/repo/cross_attn_ato_poc/scripts/agent_tick.sh`
   - **Single long session**: `cd /workspace/repo/cross_attn_ato_poc && claude` (or `codex`), then paste the loop prompt from §6 below.
5. Agent runs first 4-6 experiments (Task #38). Writes Day-2 README section.

### Day 3

1. Auto-loop completes round-2 sweep (Task #40).
2. Top-3 medium eval (Task #41, Hr 8-11).
3. Top-1 large eval *only if Hr 11 is reached on schedule* (Task #41, Hr 11-12).
4. Final synthesis (Task #42). **Sacred — never compressed.**

---

## 4. Recovery from pod restart / termination

If the pod restarts or is terminated:

1. Verify network volume still exists (RunPod console).
2. Boot a new pod with the same volume attached.
3. Run `scripts/preflight_check.py` to verify env.
4. Inspect `src/auto_research/experiments.jsonl` — last entry tells you what experiment was in flight.
5. Check `src/auto_research/runs/exp_NNN/` for any partial outputs.
6. If a run was in flight: delete the partial `exp_NNN/` directory and let the agent re-propose.
7. If the loop was idle: re-start it (§3 Day-2 step 4).

**Atomic-write guarantee**: `run_next_experiment.py` writes all artifacts to `.tmp` paths and renames at the end, so partial writes are never visible. A missing `metrics.json` means the run did not complete; treat it as if it never started.

---

## 5. Killing a stuck run

```bash
# Find the pid
cat /workspace/.gpu.lock
# (the lockfile contains the pid of the current accelerate launch)

# Kill it
kill -TERM $(cat /workspace/.gpu.lock)
# Wait 10 seconds. If still alive:
kill -KILL $(cat /workspace/.gpu.lock)

# Manual lock release (only after confirming process is dead)
rm /workspace/.gpu.lock

# Mark the experiment as failed in experiments.jsonl
python scripts/run_next_experiment.py --mark-failed runs/exp_NNN
```

---

## 6. Agent loop prompt

For Day 2 onwards. Use this verbatim when starting a Claude Code or Codex session for the loop:

```
You are the auto-research agent for the cross-attention ATO POC.

Read src/auto_research/AGENT_INSTRUCTIONS.md and follow it.

Read src/auto_research/sweep_state.yaml to see budget remaining and current best.
Read the last 5 entries of src/auto_research/experiments.jsonl for history.

If a halt condition is met, stop launching new experiments and write the
Day-2 or Day-3 README section as appropriate.

Otherwise, propose the next experiment by writing
src/auto_research/runs/exp_NNN/config.yaml, then run:

    python scripts/run_next_experiment.py src/auto_research/runs/exp_NNN/config.yaml

When it completes, read the run's metrics.json and ci_report.json, append a
one-paragraph summary to experiments.jsonl, update sweep_state.yaml, and
decide the next action.

Before editing any source code file, run:
    git add -A && git commit -m "snapshot before <change description>"
```

---

## 7. Git-checkpoint policy

The agent must `git commit` before any code edit. This is enforced by convention (the prompt above) but not by hook. The user can revert any unwanted change with:

```bash
cd /workspace/repo
git log --oneline -n 20
git revert <hash>          # or
git reset --hard <hash>    # destructive — only with intent
```

---

## 8. External backup

`scripts/backup_to_external.sh` rsyncs critical artifacts to external storage every 30 min (via cron, set up Day 0 / Hr 2-3).

Target options (pick one; configure via env var `BACKUP_TARGET`):

- `s3://<bucket>/cross-attn-ato-poc/`  (AWS CLI)
- `r2://<bucket>/cross-attn-ato-poc/`  (rclone)
- `hf://datasets/<user>/cross-attn-ato-poc-artifacts/`  (HF private dataset)

What gets backed up:

- `src/auto_research/experiments.jsonl`
- `src/auto_research/sweep_state.yaml`
- `src/auto_research/runs/*/metrics.json`
- `src/auto_research/runs/*/ci_report.json`
- `src/auto_research/runs/*/leakage_report.json`
- `README.md`
- `checkpoints/qwen3-8b-cpt-light-merged/` (once)
- `checkpoints/baseline_*/` (once each)
- Top-3 `checkpoints/xattn_exp_NNN/` (rotating)

What is **not** backed up: failed sweep checkpoints, HF cache, W&B offline logs.

---

## 9. Cost guardrails

- LLM narration budget: **$200 hard cap** for the 25k LLM-narrated pairs (`narrative_generator.py` tracks running cost; aborts if exceeded).
- H100 hours: ~24 hours of GPU usage across 3 days. Reserved pricing ≈ $50-80; on-demand ≈ $100-160.
- Total POC cost target: < $400 including LLM narration.

If RunPod billing is approaching $300 with the POC incomplete, pause and escalate before continuing.

---

## 10. Known gotchas

- **`HF_HOME=/workspace/.hf`** must be set *before* importing `transformers` or it caches to container disk and loses on restart.
- **`TOKENIZERS_PARALLELISM=false`** — without it, Accelerate prints warnings on every batch.
- **`paged_adamw_8bit`** requires `bitsandbytes>=0.43`. The `preflight_check.py` validates the version.
- **Adding new tokens to Qwen3-8B** requires `model.resize_token_embeddings(len(tokenizer))` *after* tokenizer.add_special_tokens. Forgetting this produces silent NaN.
- **Cross-attention KV cache** is not natively supported by the base Qwen3 `forward()`; our `qwen_xattn_wrapper.py` patches it. If you see `KeyError: 'cross_kv'` during generation, the wrapper isn't being used.
- **W&B offline mode**: set `WANDB_MODE=offline` before training. Sync later with `wandb sync /workspace/.wandb/offline-run-*`.
