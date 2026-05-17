"""Main cross-attention trainer — Stage-1 x-attn adaptation.

Per PLAN.md "Stages and adapter lifecycle":
  - Start from `qwen3-8b-cpt-light-merged` (Stage-0 LoRA merged into
    base; the merged checkpoint is itself baseline #1 produced by
    train_cpt_light + scripts/merge_stage0_lora).
  - Construct QwenXAttnWrapper around the merged base.
  - Attach Stage-1 LoRA-on-Q (r=16, fresh — distinct from Stage-0).
  - Train side-stream encoder + resampler + x-attn blocks + LoRA-on-Q
    jointly on PAIRED (structured_events, narrative) data with
    next-token CE on the text side.

The trainer logs the gate-activation trajectory at every step into
gate_trajectory.json (consumed by run_next_experiment.py's
zero_gate_activation halt check).

CLI:
    accelerate launch src/train/train_xattn.py \\
        --config src/auto_research/runs/exp_NNN/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _build_structured_collate(vocab, max_events: int = 200):
    """Returns a callable that takes a list of dataset examples and
    produces tensor dicts for the wrapper's `event_*` kwargs.
    """
    def _collate(ex_list):
        import torch
        from src.model.encoders.small_transformer import tokenize_events
        # Review 011 finding #1: load_paired_dataset now serializes
        # structured_events as a JSON string. Parse it back per-row.
        from src.train.common import parse_structured_events

        type_ids = []
        bucket_ids = []
        delta_t = []
        mask = []
        for ex in ex_list:
            events = parse_structured_events(ex)
            toks = tokenize_events(events, vocab, max_events=max_events)
            type_ids.append(toks["event_type_ids"])
            bucket_ids.append(toks["bucket_ids"])
            delta_t.append(toks["delta_t"])
            mask.append(toks["attention_mask"])
        return {
            "event_type_ids": torch.tensor(type_ids, dtype=torch.long),
            "bucket_ids": torch.tensor(bucket_ids, dtype=torch.long),
            "delta_t": torch.tensor(delta_t, dtype=torch.float32),
            "event_mask": torch.tensor(mask, dtype=torch.long),
        }
    return _collate


def _build_paired_collator(text_collator, structured_collator):
    """Wraps the eval-mode-dropout text collator to ALSO emit the
    structured-side tensors per batch.
    """
    def _collator(batch):
        text_out = text_collator(batch)
        struct_out = structured_collator(batch)
        text_out.update(struct_out)
        return text_out
    return _collator


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
    xattn_cfg = cfg.get("xattn", {})

    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    from src.model.qwen_xattn_wrapper import QwenXAttnWrapper, estimate_wrapper_trainable_params
    from src.model.encoders.small_transformer import EventVocab

    accelerator = Accelerator(mixed_precision=train_cfg.get("precision", "bf16"))
    device = accelerator.device

    # Load the MERGED CPT-light base.
    base_id = train_cfg.get(
        "base_checkpoint", "/workspace/checkpoints/qwen3-8b-cpt-light-merged",
    )
    tokenizer, n_new = prepare_tokenizer(base_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    old_vocab_size = base.config.vocab_size
    if n_new:
        base.resize_token_embeddings(len(tokenizer))
        from src.tokenizer.custom_tokens import init_new_embeddings
        init_new_embeddings(base, tokenizer, old_vocab_size=old_vocab_size)

    # Build the wrapper.
    vocab = EventVocab()
    wrapper = QwenXAttnWrapper(
        base,
        insertion_pattern=xattn_cfg.get("insertion_pattern", "every_4"),
        n_slots=xattn_cfg.get("resampler_slots", 64),
        gate_init=xattn_cfg.get("gate_init", "small_0.01"),
        encoder_hidden_dim=256,
        encoder_n_layers=6, encoder_n_heads=4,
        resampler_n_layers=2, resampler_n_heads=8,
        vocab=vocab,
    )
    # Stage-1 fresh LoRA-on-Q.
    wrapper.attach_lora_on_q(r=xattn_cfg.get("lora_r_on_q", 16))
    wrapper = wrapper.to(device).to(torch.bfloat16)

    # Sanity log: how many trainable params did we actually allocate?
    actual_summary = wrapper.trainable_param_summary()
    estimated = estimate_wrapper_trainable_params(
        insertion_pattern=xattn_cfg.get("insertion_pattern", "every_4"),
        n_hidden_layers=base.config.num_hidden_layers,
        hidden_size=base.config.hidden_size,
        n_slots=xattn_cfg.get("resampler_slots", 64),
        lora_r_on_q=xattn_cfg.get("lora_r_on_q", 16),
    )
    print(f"trainable params (actual)   : {actual_summary['total_trainable']:,}")
    print(f"trainable params (estimated): {estimated['total']:,}")

    # Collators
    from src.train.mixers.eval_mode_dropout import EvalModeDropoutCollator
    text_collator = EvalModeDropoutCollator(
        tokenizer=tokenizer, text_field="text",
        max_length=train_cfg.get("seq_len", 2048),
        seed=train_cfg.get("seed", 0),
    )
    structured_collator = _build_structured_collate(vocab)
    paired_collator = _build_paired_collator(text_collator, structured_collator)

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
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=paired_collator,
    )

    optimizer, n_trainable = build_optimizer(
        wrapper, lr=train_cfg.get("lr", 1e-4),
        optimizer_name=train_cfg.get("optimizer", "paged_adamw_8bit"),
    )
    total_steps = train_cfg.get("steps", 1500)
    scheduler = build_lr_scheduler(
        optimizer, warmup_steps=train_cfg.get("warmup_steps", 500),
        total_steps=total_steps,
    )

    wrapper, optimizer, train_loader, scheduler = accelerator.prepare(
        wrapper, optimizer, train_loader, scheduler,
    )

    run_dir = args.config.parent

    # Gate trajectory logging
    gate_log: list[dict] = []
    gate_log_path = run_dir / "gate_trajectory.json"

    # Training loop
    wrapper.train()
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

            out = wrapper(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                event_type_ids=batch["event_type_ids"],
                bucket_ids=batch["bucket_ids"],
                delta_t=batch["delta_t"],
                event_mask=batch["event_mask"],
            )
            loss = out.loss / grad_accum
            accelerator.backward(loss)
            accum_loss += loss.detach().item()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.append(accum_loss)

        # Log gate trajectory every step (for the halt check).
        # gate_diagnostics returns list[(layer_idx, |α_attn|, |α_ffn|)].
        diag = wrapper.gate_diagnostics() if hasattr(wrapper, "gate_diagnostics") else \
               accelerator.unwrap_model(wrapper).gate_diagnostics()
        gate_log.append({
            "step": step,
            "loss": accum_loss,
            "gates": [(int(li), float(a), float(f)) for li, a, f in diag],
        })

        if step % 100 == 0:
            max_gate = max((a for _, a, _ in diag), default=0.0)
            print(f"step {step:>5} loss {accum_loss:.4f}  max_gate {max_gate:.4f}")

        step += 1

    # Persist gate trajectory
    tmp = gate_log_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(gate_log, indent=2))
    tmp.rename(gate_log_path)

    max_gate_magnitude = max(
        (a for entry in gate_log for _, a, _ in entry["gates"]),
        default=0.0,
    )

    # Final three-mode eval.
    wrapper.eval()
    unwrapped = accelerator.unwrap_model(wrapper)
    structured_fn = structured_collator

    def _eval_structured_input_fn(ex_list):
        return structured_fn(ex_list)

    from src.train.eval_runner import run_three_mode_eval
    eval_summary = run_three_mode_eval(
        unwrapped, eval_ds, tokenizer, run_dir,
        modes=cfg.get("eval", {}).get("modes", ["stripped", "opaque", "full"]),
        batch_size=batch_size,
        max_length=train_cfg.get("seq_len", 2048),
        structured_input_fn=_eval_structured_input_fn,
    )

    metrics = {
        "status": "ok", "arm": "xattn",
        "final_train_loss": losses[-1] if losses else None,
        "n_steps": step, "wall_clock_sec": time.time() - t_start,
        "n_trainable": n_trainable,
        "predictions": eval_summary,
        "max_gate_magnitude": max_gate_magnitude,
        "estimated_trainable_total": estimated["total"],
        "actual_trainable_total": actual_summary["total_trainable"],
    }
    metrics_path = run_dir / "metrics.json"
    tmp = metrics_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics, indent=2))
    tmp.rename(metrics_path)

    # Save: trainable state dict only (x-attn machinery + LoRA-on-Q).
    # The frozen base is not saved here — checkpoint consumers load the
    # base from base_checkpoint and reapply the wrapper.
    sd = {k: v for k, v in unwrapped.state_dict().items() if not k.startswith("base.base_model")}
    torch.save(sd, run_dir / "xattn_state.pt")
    print(f"wrote {metrics_path} + xattn_state.pt + gate_trajectory.json")
    print(f"max gate magnitude across training: {max_gate_magnitude:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
