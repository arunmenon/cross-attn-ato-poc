"""LoRA-on-Qwen3-8B text-only baseline (#2 of 4).

Per PLAN.md "Baselines" §2: LoRA r=16 on raw Qwen3-8B (no CPT), trained
on the SAME narrative + verdict-footer text the other LM trainers see.
Tells us whether CPT-light is pulling its weight vs LoRA alone.

CLI:
    accelerate launch src/train/train_lora_text_only.py \\
        --config src/auto_research/runs/exp_NNN/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    from src.train.common import (
        load_config, load_paired_dataset, prepare_tokenizer,
        build_optimizer, build_lr_scheduler,
    )

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    accelerator = Accelerator(mixed_precision=train_cfg.get("precision", "bf16"))
    device = accelerator.device

    # Base model + tokenizer
    model_id = train_cfg.get("base_checkpoint", "Qwen/Qwen3-8B")
    tokenizer, n_new = prepare_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    old_vocab_size = model.config.vocab_size
    if n_new:
        model.resize_token_embeddings(len(tokenizer))
        from src.tokenizer.custom_tokens import init_new_embeddings
        init_new_embeddings(model, tokenizer, old_vocab_size=old_vocab_size)
    # Freeze base, apply LoRA r=16 on q_proj only. Custom-token
    # embeddings need to be trainable so this baseline can READ the
    # journey/actor tokens (review 007 finding #1 — frozen mean-init
    # rows would make this baseline blind to the structural tokens).
    for p in model.parameters():
        p.requires_grad = False
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj"],
        modules_to_save=["embed_tokens"],
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    # Eval-mode dropout collator
    from src.train.mixers.eval_mode_dropout import EvalModeDropoutCollator
    collator = EvalModeDropoutCollator(
        tokenizer=tokenizer,
        text_field="text",
        max_length=train_cfg.get("seq_len", 2048),
        seed=cfg.get("training", {}).get("seed", 0),
    )

    # Data
    splits = load_paired_dataset(data_cfg["train_path"])
    train_ds = splits["train"]
    eval_ds_path = data_cfg.get("eval_fast_path")
    if eval_ds_path:
        eval_splits = load_paired_dataset(eval_ds_path)
        eval_ds = eval_splits.get("eval") or eval_splits["train"]
    else:
        eval_ds = splits.get("eval") or train_ds

    batch_size = train_cfg.get("micro_batch", 4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator,
    )

    # Optimizer
    optimizer, n_trainable = build_optimizer(
        model, lr=train_cfg.get("lr", 1e-4),
        optimizer_name=train_cfg.get("optimizer", "paged_adamw_8bit"),
    )
    total_steps = train_cfg.get("steps", 1500)
    scheduler = build_lr_scheduler(
        optimizer, warmup_steps=train_cfg.get("warmup_steps", 500),
        total_steps=total_steps,
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler,
    )

    print(f"lora_text_only: trainable params = {n_trainable:,}")
    print(f"train: {len(train_ds)} examples; eval: {len(eval_ds)} examples")
    print(f"total_steps={total_steps} seq_len={train_cfg.get('seq_len', 2048)}")

    # Training loop
    model.train()
    step = 0
    t_start = time.time()
    losses: list[float] = []
    grad_accum = train_cfg.get("grad_accum", 8)

    train_iter = iter(train_loader)
    while step < total_steps:
        accum_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss / grad_accum
            accelerator.backward(loss)
            accum_loss += loss.detach().item()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.append(accum_loss)
        if step % 100 == 0:
            print(f"step {step:>5} loss {accum_loss:.4f}")
        step += 1

    # Final eval (three-mode)
    run_dir = args.config.parent
    model.eval()
    from src.train.eval_runner import run_three_mode_eval
    eval_summary = run_three_mode_eval(
        model, eval_ds, tokenizer, run_dir,
        modes=cfg.get("eval", {}).get("modes", ["stripped", "opaque", "full"]),
        batch_size=batch_size,
        max_length=train_cfg.get("seq_len", 2048),
    )

    metrics = {
        "status": "ok", "arm": "lora_text",
        "final_train_loss": losses[-1] if losses else None,
        "n_steps": step, "wall_clock_sec": time.time() - t_start,
        "n_trainable": n_trainable,
        "predictions": eval_summary,
        "max_gate_magnitude": None,
    }
    metrics_path = run_dir / "metrics.json"
    tmp = metrics_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics, indent=2))
    tmp.rename(metrics_path)

    # Save LoRA adapter
    model.save_pretrained(run_dir / "lora_adapter")
    print(f"wrote {metrics_path} + lora_adapter/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
