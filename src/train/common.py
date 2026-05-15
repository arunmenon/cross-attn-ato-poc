"""Shared training utilities for the cross-attn ATO POC.

Used by all five trainers in src/train/:
  - train_cpt_light
  - train_lora_text_only
  - train_structured_as_text
  - train_event_only_classifier
  - train_xattn

Public surface:
  - load_config(path) -> dict
  - prepare_tokenizer(model_id) -> tokenizer with custom tokens installed
  - load_paired_dataset(data_dir) -> HF DatasetDict
  - get_label_token_ids(tokenizer) -> (fraud_id, legit_id)
  - find_label_score_position(input_ids, tokenizer) -> list[int]
  - build_optimizer(model, lr, ...) -> torch.optim.Optimizer (paged_adamw_8bit)
  - build_lr_scheduler(opt, warmup, total) -> scheduler
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a trainer config (YAML). Schema is the one in
    src/auto_research/experiment_template.yaml; scripts/run_next_experiment.py
    validates before invoking the trainer, so trainers can assume the
    config is well-formed."""
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a YAML mapping; got {type(cfg).__name__}")
    return cfg


# ---------------------------------------------------------------------------
# Tokenizer with custom tokens installed
# ---------------------------------------------------------------------------

def prepare_tokenizer(model_id: str):
    """Load the HF tokenizer for `model_id`, install all custom tokens
    from src.tokenizer.custom_tokens, and ensure pad_token is set.

    Returns: (tokenizer, n_new_tokens_added).

    Caller MUST then call:
        model.resize_token_embeddings(len(tokenizer))
    on the underlying model before any training step. Forgetting this
    produces silent NaN.
    """
    from transformers import AutoTokenizer
    from src.tokenizer.custom_tokens import install

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n_added = install(tok)
    return tok, n_added


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_paired_dataset(data_dir: str | Path):
    """Load a directory produced by `data/gen/build_dataset.py`.

    Layout (per build_dataset.py):
      <data_dir>/data.jsonl         — single-file split (eval_frac=0)
      OR
      <data_dir>/train.jsonl        — train split
      <data_dir>/eval.jsonl         — eval split

    Returns: a dict of HF Datasets keyed by "train" / "eval" (if both
    splits exist) or just {"train": ds} for the single-file case.
    """
    from datasets import Dataset

    d = Path(data_dir)
    if not d.exists():
        raise FileNotFoundError(f"dataset directory not found: {d}")

    splits: dict[str, "Dataset"] = {}
    train_path = d / "train.jsonl"
    eval_path = d / "eval.jsonl"
    single_path = d / "data.jsonl"

    if train_path.exists():
        splits["train"] = Dataset.from_json(str(train_path))
        if eval_path.exists():
            splits["eval"] = Dataset.from_json(str(eval_path))
        else:
            splits["eval"] = None  # type: ignore[assignment]
    elif single_path.exists():
        splits["train"] = Dataset.from_json(str(single_path))
        splits["eval"] = None  # type: ignore[assignment]
    else:
        raise FileNotFoundError(
            f"no train.jsonl/eval.jsonl/data.jsonl in {d}"
        )

    return splits


# ---------------------------------------------------------------------------
# Label token IDs (fraud vs legit) for the score surface
# ---------------------------------------------------------------------------

# These strings have a leading space — the tokenizer typically encodes
# " fraud" and " legit" as a single token each. We score the model's
# next-token distribution at the position right after "<risk_verdict>\nlabel:".
LABEL_FRAUD_STR = " fraud"
LABEL_LEGIT_STR = " legit"


def get_label_token_ids(tokenizer) -> tuple[int, int]:
    """Returns (fraud_id, legit_id) — the single token of " fraud" and
    " legit" respectively.

    Review 007 finding #4: fail fast on ALL three failure modes that
    would silently corrupt scoring:

      a. Empty encoding (tokenizer config broken).
      b. Multi-token encoding (single-token scoring would only see the
         first token, missing discriminative information).
      c. Same first-token ID for both labels (score would be 0 for
         every example → AUC = 0.5).
    """
    fraud_ids = tokenizer.encode(LABEL_FRAUD_STR, add_special_tokens=False)
    legit_ids = tokenizer.encode(LABEL_LEGIT_STR, add_special_tokens=False)
    if not fraud_ids or not legit_ids:
        raise ValueError(
            f"could not tokenize label strings: "
            f"fraud_ids={fraud_ids} legit_ids={legit_ids}"
        )
    if len(fraud_ids) != 1 or len(legit_ids) != 1:
        raise NotImplementedError(
            f"label strings tokenize to multi-token sequences: "
            f"fraud_ids={fraud_ids} legit_ids={legit_ids}. "
            f"The current scoring code is single-token only; "
            f"implement multi-token sequence-log-prob scoring before "
            f"running, or change LABEL_FRAUD_STR / LABEL_LEGIT_STR to "
            f"strings the tokenizer produces as a single token each."
        )
    if fraud_ids[0] == legit_ids[0]:
        raise ValueError(
            f"' fraud' and ' legit' tokenize to the same first ID "
            f"({fraud_ids[0]}); score would be identically zero for "
            f"every example. Check tokenizer configuration."
        )
    return fraud_ids[0], legit_ids[0]


# ---------------------------------------------------------------------------
# Label-position finding within input_ids
# ---------------------------------------------------------------------------

# The verdict-footer prefix the trainer scores at:
LABEL_MARKER_STR = "<risk_verdict>\nlabel:"


def find_label_score_position(text: str) -> int:
    """Return the character offset right AFTER "<risk_verdict>\\nlabel:"
    in `text`. The model is scored on the next-token distribution at the
    token aligned to that offset.

    Returns -1 if the marker is not found.
    """
    idx = text.find(LABEL_MARKER_STR)
    if idx < 0:
        return -1
    return idx + len(LABEL_MARKER_STR)


def find_label_token_position(input_ids: list[int], tokenizer) -> int:
    """Locate the token index in `input_ids` corresponding to the
    position right after "<risk_verdict>\\nlabel:".

    Strategy: decode incrementally, find the substring offset, then
    map back to the token index.

    Returns -1 if not found.
    """
    text = tokenizer.decode(input_ids, skip_special_tokens=False)
    char_offset = find_label_score_position(text)
    if char_offset < 0:
        return -1

    # Walk forward through token decodings until we accumulate >= char_offset
    accumulated = ""
    for i, tid in enumerate(input_ids):
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        accumulated += piece
        if len(accumulated) >= char_offset:
            return i
    return -1


# ---------------------------------------------------------------------------
# Optimizer + LR scheduler factories
# ---------------------------------------------------------------------------

def build_optimizer(model, *, lr: float, weight_decay: float = 0.0,
                    optimizer_name: str = "paged_adamw_8bit"):
    """Build the optimizer per PLAN.md Training pipeline. Default is
    paged_adamw_8bit which cuts AdamW optimizer state from 12 bytes/param
    to ~5 bytes/param via bitsandbytes' 8-bit quantization + paged
    state."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    if n_trainable == 0:
        raise RuntimeError("model has zero trainable parameters")

    if optimizer_name == "paged_adamw_8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(
            trainable, lr=lr, weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        import torch
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name!r}")
    return opt, n_trainable


def build_lr_scheduler(optimizer, *, warmup_steps: int, total_steps: int):
    """Cosine schedule with linear warmup."""
    from transformers import get_cosine_schedule_with_warmup
    return get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


# ---------------------------------------------------------------------------
# JSONL writer for predictions
# ---------------------------------------------------------------------------

def write_predictions_jsonl(records: list[dict], path: str | Path) -> None:
    """Atomic write of prediction records. Each record is a dict with
    keys consumed by eval/score_risk.py:
      score, label, journey_family, actor_family, is_hard_negative.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tmp.rename(p)


# ---------------------------------------------------------------------------
# Self-test (torch-free portions only locally)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # Config load: synthesize a tiny YAML and parse
    import tempfile, yaml
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"arm": "xattn", "training": {"steps": 100}}, f)
        cfg_path = f.name
    cfg = load_config(cfg_path)
    assert cfg["arm"] == "xattn"
    assert cfg["training"]["steps"] == 100
    print("load_config OK")

    # find_label_score_position
    sample = "...<risk_verdict>\nlabel: fraud\njourney_family: clean\n</risk_verdict>..."
    pos = find_label_score_position(sample)
    assert pos == len("...<risk_verdict>\nlabel:")
    assert sample[pos:pos+6] == " fraud"
    print("find_label_score_position OK")

    # write_predictions_jsonl
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "preds.jsonl"
        records = [
            {"score": 0.5, "label": "fraud", "journey_family": "sim_swap",
             "actor_family": "human", "is_hard_negative": False},
            {"score": -0.3, "label": "legit", "journey_family": "clean",
             "actor_family": "human", "is_hard_negative": False},
        ]
        write_predictions_jsonl(records, out)
        assert out.exists()
        with out.open() as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["label"] == "fraud"
    print("write_predictions_jsonl OK")

    # torch/transformers/bitsandbytes-dependent paths
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("torch/transformers not installed; tokenizer / dataset / "
              "optimizer self-tests skipped (run on the pod)")
        return

    # Tokenizer install would download Qwen3-8B; we don't do that locally.
    # The pod-side preflight (scripts/preflight_check.py) covers this.
    print("torch + transformers importable; full tokenizer/optimizer tests run on the pod")


if __name__ == "__main__":
    _self_test()
