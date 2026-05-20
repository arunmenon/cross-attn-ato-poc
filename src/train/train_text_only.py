"""text_only_v4 — LoRA-on-Qwen text-only baseline (Stage-1 from shared CPT-light base).

Renamed from train_lora_text_only.py for v4. The v3 lora_text baseline
started from RAW Qwen3-8B (no CPT-light merge), which made it not
apples-to-apples with structured_as_text / xattn (both of which start
from qwen3-8b-cpt-light-merged). v4 drops that ambiguity:

  - In v4, this trainer starts from `qwen3-8b-cpt-light-v4-merged`
    (the shared base used by structured_as_text_v4 and xattn_v4),
    NOT raw Qwen.
  - The text-input shape is the v4 text_only composition from
    `data/gen/build_dataset.py::compose_text_only`:
        <case>
        <narrative>...narrative body...</narrative>

        <risk_verdict>
        label: {fraud|legit}
        </risk_verdict>
        </case>
  - NO journey/actor wrapper tokens, no event lines, minimal verdict.
  - This trainer's input is BYTE-IDENTICAL to what train_xattn.py
    sees on the LM-text branch. The architectural difference between
    text_only_v4 and xattn_v4 is whether the side-stream encoder is
    consumed — nothing else.

The v4 question this arm answers (paired with xattn_v4):
    Does adding the structured event side stream improve over the
    same model reading clean narrative only?

CLI:
    accelerate launch src/train/train_text_only.py \\
        --config src/auto_research/runs/exp_NNN/config.yaml

Per .claude/tasks/data-v4-pivot-plan.md baseline/checkpoint contract.
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

    # Base model + tokenizer.
    # v4: default to the shared CPT-light merged base — same starting
    # point as train_structured_as_text.py and train_xattn.py. This
    # is what makes text_only_v4 / structured_as_text_v4 / xattn_v4
    # apples-to-apples. (Per-config `base_checkpoint` override still
    # honored; defaults to the v4 path.)
    model_id = train_cfg.get(
        "base_checkpoint", "/workspace/checkpoints/qwen3-8b-cpt-light-merged",
    )
    tokenizer, n_new = prepare_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    old_vocab_size = model.config.vocab_size
    if n_new:
        model.resize_token_embeddings(len(tokenizer))
        from src.tokenizer.custom_tokens import init_new_embeddings
        init_new_embeddings(model, tokenizer, old_vocab_size=old_vocab_size)
    # Freeze base, apply LoRA r=16 on q_proj only. Embeddings stay
    # FROZEN here — Stage-0 v4 already trained the custom-token rows
    # and merged them into the base, so this baseline must consume
    # them as-is. Training embed_tokens here would break the apples-
    # to-apples contract with train_xattn.py (which freezes the LM
    # body including embeddings); the metric difference between the
    # two arms must isolate the cross-attn pathway, not "text_only
    # got to retrain the dictionary too."
    #
    # Historical note: pre-v4 this trainer set `modules_to_save=
    # ["embed_tokens"]` because it ran on raw Qwen and the custom-
    # token rows were still at mean-init values. v4 moved that
    # learning into Stage-0; Stage-1 freezes.
    for p in model.parameters():
        p.requires_grad = False
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj"],
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

    # v4 startup smoke: confirm dataset rows match compose_text_only(row).
    # train_text_only and train_xattn both call this against the SAME
    # dataset, so if both pass, they're seeing byte-identical text.
    from data.gen.build_dataset import verify_v4_text_contract
    # Review 021 finding #4 fix: strict=True is the v4 default (hard
    # fail on v3-format data), sample_n=32 spreads checks across the
    # dataset rather than just the first 3 rows. Legacy v3 reruns
    # should pass strict=False explicitly via config if needed.
    verify_v4_text_contract(train_ds, sample_n=32, arm_name="text_only", strict=True)

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

    # v4 (review 021 finding #1): record the arm name from the config,
    # not a hardcoded string. The renamed trainer is now invoked by
    # BOTH `arm: text_only` (v4 canonical) and `arm: lora_text` (v3
    # legacy alias). The recorded arm preserves which contract the
    # config asked for so the leaderboard groups runs correctly.
    arm_from_config = cfg.get("arm", "text_only")
    if arm_from_config not in {"text_only", "lora_text"}:
        # Defensive: someone invoked this trainer with an unexpected arm.
        # Default to text_only since that's the v4 contract, but warn.
        print(f"[train_text_only] WARN: unexpected arm {arm_from_config!r}; "
              f"recording as text_only")
        arm_from_config = "text_only"
    metrics = {
        "status": "ok", "arm": arm_from_config,
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
