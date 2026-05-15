#!/usr/bin/env python
"""Merge Stage-0 LoRA adapter into the base Qwen3-8B and save as a single checkpoint.

After Stage-0 CPT-light completes, run this to produce `qwen3-8b-cpt-light-merged`.
This becomes:
  1. Baseline #1 (the CPT-only baseline)
  2. The starting point for Stage-1 cross-attention training (which adds fresh LoRA-on-Q)

Why merge: per the v3 plan, mixing Stage-0 LoRA with Stage-1 LoRA produces
adapter confusion. Merging Stage-0 means Stage-1 starts from clean weights.

CLI:
    python scripts/merge_stage0_lora.py \\
        --base Qwen/Qwen3-8B \\
        --lora /workspace/checkpoints/qwen3-8b-cpt-light-lora \\
        --out /workspace/checkpoints/qwen3-8b-cpt-light-merged \\
        --tokenizer /workspace/checkpoints/qwen3-8b-cpt-light-lora
"""

from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="HF model ID or path of the base model")
    p.add_argument("--lora", required=True, type=Path, help="path to Stage-0 LoRA checkpoint")
    p.add_argument("--out", required=True, type=Path, help="output directory for merged checkpoint")
    p.add_argument("--tokenizer", type=Path, default=None,
                   help="tokenizer to save with merged model (defaults to --lora path)")
    args = p.parse_args()

    if args.out.exists():
        print(f"output {args.out} already exists; refusing to overwrite", file=sys.stderr)
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"loading base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )

    # If the base vocab was expanded during CPT-light, resize before loading the adapter.
    tok_path = args.tokenizer or args.lora
    print(f"loading tokenizer: {tok_path}")
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if len(tok) != base.config.vocab_size:
        print(f"resizing embeddings: {base.config.vocab_size} -> {len(tok)}")
        base.resize_token_embeddings(len(tok))

    print(f"loading LoRA adapter from: {args.lora}")
    peft_model = PeftModel.from_pretrained(base, args.lora)

    print("merging LoRA into base...")
    merged = peft_model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=False)
    print(f"saving merged checkpoint to: {args.out}")
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)

    # Drop a marker so we know this is a merged checkpoint, not raw Qwen.
    (args.out / "MERGED_FROM.txt").write_text(
        f"base: {args.base}\nlora: {args.lora}\ntokenizer: {tok_path}\n"
    )

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
