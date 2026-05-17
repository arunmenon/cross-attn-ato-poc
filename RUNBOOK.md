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

- GPU: 1× H100 80GB SXM (or PCIe if SXM unavailable). On Community Cloud
  H200 SXM (141 GB) and RTX PRO 6000 Blackwell (96 GB) are common
  substitutes — see Blackwell note below.
- Base image — **architecture-dependent**:
  - Hopper (H100, H200): `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
  - Blackwell (B200, B300, RTX PRO 6000 96GB): `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (a.k.a. "Pytorch 2.8.0" tile)
  - Pin the exact tag — do not use `:latest`. Mixing Hopper image + Blackwell GPU silently degrades to non-optimal kernels for `paged_adamw_8bit` and FlashAttention.
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

The git repository's root IS `cross_attn_ato_poc` (verified via review
008). Clone it directly into `/workspace/cross_attn_ato_poc` — there is
NO additional nesting. All commands below assume this layout.

```bash
cd /workspace
git clone <your-repo-url> cross_attn_ato_poc
cd /workspace/cross_attn_ato_poc

python -m venv /workspace/.venv
source /workspace/.venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

# Initialize the auto-research state files (gitignored by design; the
# agent loop expects them to exist before its first iteration).
python scripts/init_auto_research_state.py

# Preflight: GPU/VRAM/persistence-env/tokenizer/W&B (fails closed if
# HF_HOME et al. are not under /workspace — review 008 finding #3).
python scripts/preflight_check.py

# Install the agent CLI for the Task #38 auto-research loop (§6 below).
# RunPod base image has no node/npm, so install Node 20 LTS first.
# Default: claude. Alternative: codex (OpenAI) — see "Alternative
# agent" note in §6.
apt-get update -qq && apt-get install -y -qq curl ca-certificates
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y -qq nodejs
npm install -g @anthropic-ai/claude-code
claude --version    # expect 2.1.x or newer

# Auth: cron-driven loop needs ANTHROPIC_API_KEY in /workspace/.env
# (no TTY in cron). Generate at https://console.anthropic.com/settings/keys
# and add to /workspace/.env:
#   export ANTHROPIC_API_KEY=sk-ant-...
# Alternative for non-cron interactive use: `claude login` once on
# the pod via SSH (OAuth browser flow).
```

`preflight_check.py` exits non-zero on any environment problem. Do **not** proceed past failed preflight.

---

## 2. Storage layout (everything under `/workspace`)

```
/workspace/
├── .venv/                      Python virtualenv
├── .hf/                        HF cache (models, datasets)
├── .wandb/                     W&B offline logs
├── cross_attn_ato_poc/         This repository (git clone — repo-as-root layout)
├── data/                       Generated synthetic datasets
│   ├── train_llm_narrated/     LLM-narrated pool: train.jsonl + eval.jsonl
│   │                           (build_dataset --eval-frac 0.2 → ~20k train + 5k stratified eval)
│   ├── eval_fast_5k/           Symlink to train_llm_narrated/eval.jsonl
│   ├── eval_medium_50k/        50k templated medium eval (standalone gen, single data.jsonl)
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

Maximize work done on the laptop so the H100 clock starts on training, not setup. Steps that need a GPU or transformers/torch are marked **pod-only**; the rest run from any Python 3.11 venv on the laptop.

**0.1 Pure-Python scaffold smoke (no torch needed)**

```bash
cd /path/to/cross_attn_ato_poc
python3 -m venv .venv-local && source .venv-local/bin/activate
pip install -r requirements-local.txt    # narrator SDKs + faker/tqdm/pyyaml/numpy/scipy/sklearn — NO transformers/accelerate/bitsandbytes
python3 -m data.gen.narrative_generator --self-test
# Expected (last three lines):
#   provider/model resolution OK (review 009 finding #1)
#   Anthropic-default dispatch OK (review 009 finding #1)
#   narrative_generator self-test OK (using stub narrator)
```

`requirements-local.txt` is the laptop-side subset; `requirements.txt` is the pod-side superset (adds transformers / accelerate / peft / bitsandbytes / wandb / safetensors). Do **not** install `requirements.txt` locally unless you also intend to run the full trainer — none of the pre-pod steps need it.

The self-test uses a stub narrator (no API key, no network) and verifies (a) the (provider, model) resolution agrees with LLM_PROVIDER overrides, (b) cache-hit + budget-cap paths, and (c) that `LLM_PROVIDER=anthropic` actually routes the haiku model id, not the gpt default. It does not exercise real pricing-table math against a live API; pricing is asserted on a tiny stub call only.

**0.2 Tokenizer sanity — two parts**

```bash
# Part A (no torch, no model download — runs anywhere):
python3 -m src.tokenizer.fencer                 # bare invocation = self-test
python3 -m src.tokenizer.custom_tokens --check  # validates token registry only

# Part B (torch + transformers + Qwen3 download — pod-only):
python3 scripts/preflight_check.py              # tokenizer download + 6-token decode roundtrip
```

Part A asserts the PII fencer is idempotent and the journey/actor/event/PII/bucketed-feature token registry is internally consistent. Neither call loads torch or downloads model weights.

Part B confirms `AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")` succeeds and that a six-token sample round-trips through encode/decode without losing special tokens. It is **narrower** than a full tokenizer-contract check — it does NOT call `src.tokenizer.custom_tokens.install(...)`, it does NOT install all 65 custom tokens, it does NOT load the base model, and it does NOT call `model.resize_token_embeddings(...)` (review 009 finding #3). The full tokenizer/embedding contract is enforced at trainer startup: `src/train/common.py:get_label_token_ids()` fails fast if " fraud" or " legit" is empty, multi-token, or aliased, and the trainers call `model.resize_token_embeddings(len(tok))` after `custom_tokens.install(tok)`. A pod that passes preflight can still fail at first trainer launch if the embedding-resize path regresses, so treat the trainer smoke (Task #32) as the real gate.

**0.3 LLM-narrated training set (~$10 with gpt-5.4-nano default)**

```bash
export OPENAI_API_KEY=...
# Optional: smoke first with --n 100 --mode template (free, no API)
python3 -m data.gen.build_dataset --n 100 --out data/samples/smoke --mode template

# Real run: 25k LLM-narrated pairs, with stratified 5k eval carve.
python3 -m data.gen.build_dataset \
    --n 25000 \
    --out data/train_llm_narrated \
    --mode llm \
    --eval-frac 0.2 \
    --usd-budget 12.0           # nano @ 25k ≈ $9.81 uncached; 12.0 leaves slack
```

`build_dataset.py` with `--eval-frac > 0` writes **`train_llm_narrated/train.jsonl`** plus **`train_llm_narrated/eval.jsonl`** (the stratified 5k carve) and a `build_summary.json` next to them. With `--eval-frac 0` it instead writes a single `data.jsonl`; `src/train/common.py` accepts both layouts. It does **not** create `data/eval_fast_5k/` as a separate directory — wire that up explicitly:

```bash
mkdir -p data/eval_fast_5k
ln -sf ../train_llm_narrated/eval.jsonl data/eval_fast_5k/eval.jsonl
```

**Switching narrator provider**: gpt-5.4-nano is the default. To use Anthropic haiku instead:

```bash
export ANTHROPIC_API_KEY=...
python3 -m data.gen.build_dataset --n 25000 --out data/train_llm_narrated \
    --mode llm --eval-frac 0.2 --usd-budget 40.0 \
    --llm-provider anthropic    # equivalent to LLM_PROVIDER=anthropic
```

`--llm-model` overrides the per-provider default if you need a specific model id (e.g., `--llm-model gpt-5.4-mini` for the larger OpenAI model). `--llm-provider` and `--llm-model` must agree — passing `--llm-provider anthropic --llm-model gpt-5.4-nano` is rejected loud at startup (review 009 finding #1).

**Cost-cap behavior**: `--usd-budget` is checked **after** each API call returns. A single in-flight call can push the running total past the cap by one charge (worst case ~$0.001 at nano). Set the cap with that one-call slack in mind; do not set it equal to the wall.

**0.4 Templated medium eval (free, no API)**

```bash
python3 -m data.gen.build_dataset --n 50000 --out data/eval_medium_50k --mode template
```

**0.5 Upload to pod-side storage**

```bash
# rsync over SSH (dependency-light; the RunPod pod has ssh by default).
rsync -avh --progress data/ user@<pod>:/workspace/data/
```

Alternative upload paths (S3, R2, HF private dataset) require extra client tooling on **both** ends — see §8 caveats. `rsync` is the only option that needs nothing beyond ssh.

### Day 1

1. Boot pod, run §1.3 bootstrap.
2. Activate `/workspace/.venv`.
3. Run Hr 0-2 tokens/fencer/bucketer task (Task #31).
4. Run vertical-slice task (Task #32). **Do not proceed past Hr 8 if this fails.**
5. Run scale + Stage-0 CPT-light task (Task #33). Takes 3-4 hours.
6. Run `scripts/merge_stage0_lora.py` to produce `qwen3-8b-cpt-light-merged`.
7. Run CPT eval task (Task #34). Write Day-1 README section by hand.

### Day 2

0. **Stage-1 architecture pre-flight (before Task #35).** Run after the Stage-0 merge produces `qwen3-8b-cpt-light-merged`, BEFORE any Stage-1 trainer launches. Catches dtype / vocab-row / wrapper-construction / LoRA-on-Q / event-tensor integration bugs in ~2 min instead of at hour 4 of a real training run (review 011 finding #5).
   ```bash
   python3 scripts/preflight_xattn.py \
     --merged-checkpoint /workspace/checkpoints/qwen3-8b-cpt-light-merged
   # Expected last line: "[xa-pre] DONE — Stage-1 architecture is ready for trainer launch"
   # Exit code 4 if CUDA missing; 2 if forward raised; 3 if loss NaN.
   ```
   If it fails: do NOT proceed to Task #35. Loop on the wrapper / checkpoint until green.

1. X-attn architecture surgery (Task #35).
2. Three more baselines: LoRA-text, structured-as-text, event-only classifier (Task #36).
3. First x-attn smoke run (Task #37).
4. Start agent loop. Either:
   - **Cron-driven (default)**: `crontab -e` and add `*/30 * * * * /workspace/cross_attn_ato_poc/scripts/agent_tick.sh`
   - **Single long session**: `cd /workspace/cross_attn_ato_poc && claude` (or `codex`), then paste the loop prompt from §6 below.
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
3. `cd /workspace/cross_attn_ato_poc && source /workspace/.venv/bin/activate`.
4. Run `python scripts/preflight_check.py` to verify env.
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

- `s3://<bucket>/cross-attn-ato-poc/`  (requires `awscli` + IAM creds in env)
- `r2://<bucket>/cross-attn-ato-poc/`  (requires `rclone` configured for R2)
- `hf://datasets/<user>/cross-attn-ato-poc-artifacts/`  (requires `huggingface_hub` + write token)
- `rsync://user@host:/path/`           (requires only `ssh` + `rsync` — dependency-light)

Only the rsync target works with a stock RunPod base image out of the box. The other three need a one-time client install + credential provisioning in §1.3 bootstrap if used.

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

- LLM narration budget: **$200 hard cap** for the 25k LLM-narrated pairs (`narrative_generator.py` tracks running cost; aborts when exceeded). Default narrator is `gpt-5.4-nano` (~$9.81 uncached, ~$8.24 with prompt caching at 25k narratives), so the cap has ~20x headroom over the projected spend.
- **Cost-cap overshoot is bounded by one API call.** The CostTracker checks the running total *after* each completed call (sunk-cost realism — we cannot un-charge a returned response). Worst case at gpt-5.4-nano: one ~$0.001 overshoot above the cap. Set `--usd-budget` slightly above the wall you actually care about.
- H100 hours: ~24 hours of GPU usage across 3 days. Reserved pricing ≈ $50-80; on-demand ≈ $100-160.
- Total POC cost target: < $400 including LLM narration.

If RunPod billing is approaching $300 with the POC incomplete, pause and escalate before continuing.

---

## 10. Known gotchas

- **`HF_HOME=/workspace/.hf`** must be set *before* importing `transformers` or it caches to container disk and loses on restart.
- **`TOKENIZERS_PARALLELISM=false`** — without it, Accelerate prints warnings on every batch.
- **`paged_adamw_8bit`** requires `bitsandbytes>=0.45.0` (Blackwell paged-kernel floor — review 010). `preflight_check.py:check_bitsandbytes` validates this, and `requirements.txt` pins `>=0.45.0,<0.50` so the three contracts agree (review 011 finding #2 raised the upper bound). Below 0.45 on Blackwell GPUs the paged path silently falls back to non-paged, ~2x'ing optimizer memory. The `<0.50` ceiling exists because the RunPod Pytorch 2.8.0 image pulls torch 2.12's CUDA 13 runtime; bnb 0.45.x ships no `libbitsandbytes_cuda130.so`, but 0.49.x does.
- **Adding new tokens to Qwen3-8B** requires `model.resize_token_embeddings(len(tokenizer))` *after* tokenizer.add_special_tokens. Forgetting this produces silent NaN.
- **Cross-attention KV cache** is not natively supported by the base Qwen3 `forward()`; our `qwen_xattn_wrapper.py` patches it. If you see `KeyError: 'cross_kv'` during generation, the wrapper isn't being used.
- **W&B offline mode**: set `WANDB_MODE=offline` before training. Sync later with `wandb sync /workspace/.wandb/offline-run-*`.
- **Blackwell (RTX PRO 6000 / B200 / B300) — pod-side checks before the trainer**: confirm the kernel paths are healthy before Task #32 (vertical slice) kicks off; quick to verify, painful to discover at hour 4 of the make-or-break window.
  ```bash
  # GPU + CUDA version sanity (expect SM_100 or SM_120 + CUDA 12.8+)
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
  # bitsandbytes Blackwell kernel check — proves the paged-optimizer
  # kernel actually executes on the GPU, not just that the class imports.
  # A bad bnb build raises "no kernel image is available" at .step(), not
  # at __init__, so we must run one step before declaring victory.
  python3 - <<'PY'
  import torch
  print("torch.version.cuda:", torch.version.cuda)
  print("compute_capability :", torch.cuda.get_device_capability(0))
  assert torch.cuda.is_available(), "CUDA not visible to torch"
  import bitsandbytes as bnb
  import bitsandbytes.optim as bnb_optim
  print("bitsandbytes:", bnb.__version__)
  p = torch.nn.Parameter(torch.randn(8, device="cuda", dtype=torch.bfloat16))
  p.grad = torch.randn_like(p)
  opt = bnb_optim.PagedAdamW8bit([p], lr=1e-4)
  opt.step()
  torch.cuda.synchronize()
  print("PagedAdamW8bit.step() OK on", torch.cuda.get_device_name(0))
  PY
  ```
  If bnb fails to import or PagedAdamW8bit's `.step()` raises `RuntimeError: CUDA error` / "no kernel image is available" on Blackwell:
  1. `pip install --upgrade "bitsandbytes>=0.45.0,<0.50"` — install a build that includes `libbitsandbytes_cuda130.so` (0.49.x ships it; 0.45.x doesn't). Mirrors the `requirements.txt` pin updated in review 011.
  2. If that still fails: `pip install bitsandbytes --upgrade --pre` to get nightly wheels with broader Blackwell coverage.
  3. Last resort: set `optimizer: adamw_8bit` in the trainer config (instead of the default `paged_adamw_8bit`). `build_optimizer()` in `src/train/common.py` accepts that value and routes to `bnb.optim.AdamW8bit` — non-paged but still 8-bit, so memory stays close to the paged path. Avoid `optimizer: adamw` unless 8-bit kernels are also broken — that path uses full-fp32 optimizer state (~4x memory).
- **Community Cloud preemption**: Community hosts can reclaim the GPU. We treat this as expected, not exceptional: `backup_to_external.sh` rsyncs the top-3 + jsonl + summaries every 30 min, atomic checkpoint writes mean partial files are never visible, and `experiments.jsonl` lets the auto-research loop resume from wherever it stopped. Worst case is ~30 min of work lost. If you're preempted, just re-rent any GPU in the same region, run §1.3 bootstrap, and the loop picks back up.
