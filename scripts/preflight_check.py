#!/usr/bin/env python
"""Pre-flight environment validation.

Runs before the first training step. Exits non-zero with a specific code
on any environment problem. Do NOT proceed past failed preflight.

Exit codes:
    0   all checks passed
    10  CUDA not available
    11  insufficient VRAM
    12  /workspace not writable
    13  model download failed
    14  tokenizer roundtrip failed
    15  bitsandbytes version too old
    16  required package missing
    17  W&B not configured
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path


def _fail(code: int, msg: str) -> None:
    print(f"PREFLIGHT FAIL ({code}): {msg}", file=sys.stderr)
    sys.exit(code)


def _ok(msg: str) -> None:
    print(f"PREFLIGHT OK: {msg}")


def check_cuda() -> None:
    try:
        import torch
    except ImportError:
        _fail(16, "torch not installed")
    if not torch.cuda.is_available():
        _fail(10, "CUDA not available")
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if n > 0 else "none"
    _ok(f"CUDA available — {n} GPU(s), device 0: {name}")


def check_vram(min_gb: int = 70) -> None:
    import torch
    props = torch.cuda.get_device_properties(0)
    gb = props.total_memory / (1024 ** 3)
    if gb < min_gb:
        _fail(11, f"insufficient VRAM: {gb:.1f}GB < {min_gb}GB")
    _ok(f"VRAM = {gb:.1f}GB")


def check_workspace_writable(path: str = "/workspace") -> None:
    p = Path(path)
    if not p.exists():
        _fail(12, f"{path} does not exist (network volume not mounted?)")
    try:
        with tempfile.NamedTemporaryFile(dir=p, delete=True) as f:
            f.write(b"preflight\n")
            f.flush()
    except OSError as e:
        _fail(12, f"{path} not writable: {e}")
    free_gb = shutil.disk_usage(p).free / (1024 ** 3)
    if free_gb < 50:
        _fail(12, f"{path} has only {free_gb:.1f}GB free; need >= 50GB")
    _ok(f"{path} writable, {free_gb:.1f}GB free")


def check_model_download(model_id: str = "Qwen/Qwen3-8B") -> None:
    try:
        from transformers import AutoConfig
    except ImportError:
        _fail(16, "transformers not installed")
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        _fail(13, f"could not fetch config for {model_id}: {e}")
    n_layers = getattr(cfg, "num_hidden_layers", None)
    h = getattr(cfg, "hidden_size", None)
    _ok(f"{model_id} config fetched — layers={n_layers} hidden={h}")


def check_tokenizer_roundtrip(model_id: str = "Qwen/Qwen3-8B") -> None:
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        _fail(14, f"tokenizer download failed: {e}")

    new_tokens = ["<journey_sim_swap>", "</journey_sim_swap>", "<actor_human>",
                  "<amount_bucket=high>", "<event_login>", "<acct_id>"]
    added = tok.add_tokens(new_tokens, special_tokens=True)
    if added == 0:
        # tokens already in vocab is OK
        pass

    sample = "Session begins. <journey_sim_swap><actor_human><event_login> from <acct_id> with <amount_bucket=high>. </journey_sim_swap>"
    ids = tok.encode(sample, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=False)
    if "<journey_sim_swap>" not in decoded or "<acct_id>" not in decoded:
        _fail(14, f"tokenizer roundtrip lost custom tokens; decoded={decoded[:200]!r}")
    _ok(f"tokenizer roundtrip OK ({added} new tokens added)")


def check_bitsandbytes(min_version: str = "0.43.0") -> None:
    try:
        import bitsandbytes as bnb  # noqa: F401
    except ImportError:
        _fail(16, "bitsandbytes not installed")
    import bitsandbytes
    version = getattr(bitsandbytes, "__version__", "0.0.0")
    def _parse(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())
    if _parse(version) < _parse(min_version):
        _fail(15, f"bitsandbytes {version} < {min_version}")
    _ok(f"bitsandbytes = {version}")


def check_required_packages() -> None:
    required = ["transformers", "accelerate", "peft", "datasets", "sklearn", "numpy", "scipy", "yaml"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        _fail(16, f"missing packages: {missing}")
    _ok(f"all required packages importable ({len(required)} checked)")


def check_wandb_or_offline() -> None:
    mode = os.environ.get("WANDB_MODE")
    if mode == "offline":
        _ok("W&B configured as offline")
        return
    try:
        import wandb  # noqa: F401
    except ImportError:
        _fail(16, "wandb not installed")
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        _fail(17, "WANDB_API_KEY not set and WANDB_MODE != 'offline'")
    _ok("W&B online mode configured (API key present)")


def check_persistent_env() -> None:
    """Verify all caches + state dirs point under /workspace (the
    network-volume mount). Without these, the first transformers /
    HF / W&B import will land artifacts on the ephemeral container
    disk and a pod restart will lose them. RunPod-specific footgun;
    review 008 finding #3.

    MUST be called BEFORE any `from transformers import ...` or
    `import wandb`, because those imports read the env vars at import
    time.
    """
    required = {
        "HF_HOME":            "/workspace",
        "TRANSFORMERS_CACHE": "/workspace",
        "WANDB_DIR":          "/workspace",
    }
    for var, prefix in required.items():
        val = os.environ.get(var)
        if not val:
            _fail(
                18,
                f"persistence-critical env var {var!r} is unset. Set it "
                f"in the pod template (e.g., {var}={prefix}/{var.lower().replace('_', '')}) "
                f"or export in the bootstrap shell. RUNBOOK.md §1.2."
            )
        if not val.startswith(prefix):
            _fail(
                18,
                f"persistence-critical env var {var!r}={val!r} is not "
                f"under {prefix}. Artifacts would land on ephemeral "
                f"container disk and be lost on pod restart."
            )

    # TOKENIZERS_PARALLELISM must be explicitly 'false' to avoid the
    # noisy warning + occasional deadlock under Accelerate's
    # multi-worker dataloader.
    tp = os.environ.get("TOKENIZERS_PARALLELISM")
    if tp != "false":
        _fail(
            18,
            f"TOKENIZERS_PARALLELISM={tp!r} (expected 'false'). Set "
            f"in the pod template to avoid warnings + deadlocks under "
            f"Accelerate's dataloader. RUNBOOK.md §1.2."
        )

    _ok(
        f"persistent-env OK: HF_HOME={os.environ['HF_HOME']!r}, "
        f"TRANSFORMERS_CACHE={os.environ['TRANSFORMERS_CACHE']!r}, "
        f"WANDB_DIR={os.environ['WANDB_DIR']!r}, "
        f"TOKENIZERS_PARALLELISM='false'"
    )


def main() -> int:
    print("=== preflight_check.py ===")
    # Order matters: env-var checks BEFORE the transformers/wandb
    # imports they govern.
    check_persistent_env()
    check_cuda()
    check_vram()
    check_workspace_writable()
    check_required_packages()
    check_bitsandbytes()
    check_model_download()
    check_tokenizer_roundtrip()
    check_wandb_or_offline()
    print("=== all checks passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
