"""Stage-0 CPT-light: vocab expansion + LoRA on attention+MLP.

Per PLAN.md "Stages and adapter lifecycle":
  - Take base Qwen3-8B.
  - Add custom tokens to the tokenizer + resize embeddings.
  - Apply LoRA to attention (q_proj + k_proj + v_proj + o_proj) AND
    to MLP (gate_proj + up_proj + down_proj). This is the
    "embedding + LoRA" variant — broader than train_lora_text_only's
    attention-Q-only LoRA so Stage-0 actually adapts the LM, not just
    a narrow Q lens.
  - Train on text-only narratives (no structured stream).
  - Save the LoRA adapter. scripts/merge_stage0_lora.py then merges
    into the base to produce `qwen3-8b-cpt-light-merged`, which is
    BOTH baseline #1 and the starting checkpoint for Stage-1 x-attn.

CLI:
    accelerate launch src/train/train_cpt_light.py \\
        --config src/auto_research/runs/exp_NNN/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# Stage-0 LoRA targets attention + MLP (broader than Stage-1's Q-only).
CPT_LIGHT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


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

    # Freeze base; LoRA + embeddings (via modules_to_save) are trainable.
    for p in model.parameters():
        p.requires_grad = False

    # Review 007 finding #1: the embedding table MUST be trainable in
    # Stage-0. Without this, the new custom-token rows stay at their
    # mean-init values forever and the model cannot distinguish
    # <journey_*>, <actor_*>, <event_*>, PII, or bucket tokens at the
    # embedding layer. `modules_to_save=["embed_tokens"]` tells PEFT to
    # (a) unfreeze the full input-embedding module and (b) save it as
    # part of the adapter so `merge_stage0_lora.py` consumes it.
    lora_config = LoraConfig(
        r=cfg.get("xattn", {}).get("lora_r_on_q", 16),  # reuse same r value
        lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=CPT_LIGHT_LORA_TARGETS,
        modules_to_save=["embed_tokens"],
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    # Sanity check (review 007 finding #1 — fail fast if PEFT's
    # modules_to_save name didn't match the actual Qwen3 embedding
    # module). Without this assertion, a Stage-0 run could silently
    # train with frozen new-token rows and the bug would propagate
    # into the merged base.
    input_emb = model.get_input_embeddings()
    if not input_emb.weight.requires_grad:
        raise RuntimeError(
            "Stage-0 expects embed_tokens to be trainable but "
            "input_embeddings.weight.requires_grad is False. Check that "
            "PEFT's modules_to_save list contains the right name for "
            "this base's embedding module (Qwen3 uses 'embed_tokens')."
        )

    # Eval-mode dropout collator
    from src.train.mixers.eval_mode_dropout import EvalModeDropoutCollator
    collator = EvalModeDropoutCollator(
        tokenizer=tokenizer, text_field="text",
        max_length=train_cfg.get("seq_len", 2048),
        seed=cfg.get("training", {}).get("seed", 0),
    )

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
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator,
    )

    optimizer, n_trainable = build_optimizer(
        model, lr=train_cfg.get("lr", 5e-5),  # CPT-light uses lower LR (PLAN.md)
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

    print(f"cpt_light: trainable params = {n_trainable:,}")
    print(f"train: {len(train_ds)} examples; eval: {len(eval_ds)} examples")
    print(f"total_steps={total_steps} lr={train_cfg.get('lr', 5e-5)}")
    print(f"LoRA targets: {CPT_LIGHT_LORA_TARGETS}")

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

    # Save LoRA adapter (consumed by scripts/merge_stage0_lora.py)
    run_dir = args.config.parent
    adapter_dir = run_dir / "stage0_lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Final three-mode eval on the CPT-light adapter (this checkpoint is
    # baseline #1)
    model.eval()
    from src.train.eval_runner import run_three_mode_eval
    eval_summary = run_three_mode_eval(
        model, eval_ds, tokenizer, run_dir,
        modes=cfg.get("eval", {}).get("modes", ["stripped", "opaque", "full"]),
        batch_size=batch_size,
        max_length=train_cfg.get("seq_len", 2048),
    )

    metrics = {
        "status": "ok", "arm": "cpt_light",
        "final_train_loss": losses[-1] if losses else None,
        "n_steps": step, "wall_clock_sec": time.time() - t_start,
        "n_trainable": n_trainable,
        "predictions": eval_summary,
        "max_gate_magnitude": None,
        "adapter_path": str(adapter_dir),
    }
    metrics_path = run_dir / "metrics.json"
    tmp = metrics_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics, indent=2))
    tmp.rename(metrics_path)
    print(f"wrote {metrics_path} + {adapter_dir}/")
    print(f"next: run scripts/merge_stage0_lora.py --lora {adapter_dir} "
          f"--out /workspace/checkpoints/qwen3-8b-cpt-light-merged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
