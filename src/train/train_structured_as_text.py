"""Structured-as-text concat baseline (#3 of 4).

Per PLAN.md "Baselines" §3 — the load-bearing apples-to-apples
comparator. Serialize the structured event stream into a compact text
block (one line per event, key fields only, ≤800 tokens) and PREPEND
to the narrative. Train CPT-light-style (same LoRA target set, same LR)
on the concatenated input.

If cross-attn cannot beat this baseline with non-overlapping CI, the
architecture is not justified: the same structured information was
available to the LM through the text channel all along.

Format (per PLAN.md "Baselines §3"):

    <events>
    t=0  login      actor=<actor_*>  geo_distance=local  ip_risk=low  …
    t=4  device_add actor=<actor_*>  device_age=new
    t=7  txn        actor=<actor_*>  amount_bucket=high  …
    </events>
    <narrative>
    …LLM- or template-generated narrative…
    </narrative>
    <risk_verdict>
    label: fraud|legit
    …
    </risk_verdict>

The original `text` field in the dataset already contains everything
EXCEPT the leading `<events>…</events>` block. We splice it in
on-the-fly inside this trainer's collator.

CLI:
    accelerate launch src/train/train_structured_as_text.py \\
        --config src/auto_research/runs/exp_NNN/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# Reuse the CPT-light LoRA target set so the comparison is apples-to-apples
# at the adapter level — same trainable surface, different inputs.
from src.train.train_cpt_light import CPT_LIGHT_LORA_TARGETS


def _serialize_events_compact(events) -> str:
    """Compact one-line-per-event serialization. Same shape as the
    `_event_to_line` helper in data/gen/build_dataset.py, but here we
    re-derive it from the dataset's stored event dicts.

    Keep this in sync with data/gen/build_dataset.py::_event_to_line —
    a divergence would silently make this baseline's input
    distribution differ from what the prompt expects.
    """
    lines = []
    for ev in events:
        parts = [f"t={ev.get('t', 0)}", f"event={ev.get('event', '?')}"]
        actor = ev.get("actor")
        if actor:
            parts.append(f"actor=<actor_{actor}>")
        for key in ("amount_bucket", "geo_distance", "ip_risk", "device_age",
                    "merchant_risk", "txn_velocity", "recipient_age",
                    "session_dwell", "auth_strength"):
            if key in ev:
                parts.append(ev[key])
        for key in ("ip", "device_id", "recipient", "merchant"):
            if key in ev:
                parts.append(ev[key])
        lines.append(" ".join(parts))
    return "<events>\n" + "\n".join(lines) + "\n</events>"


def _structured_as_text_collator_factory(eval_mode_collator, max_events: int = 200):
    """Wraps EvalModeDropoutCollator to prepend the compact event
    serialization to each example's text BEFORE the eval-mode dropout
    transform is applied.
    """
    def _collator(batch):
        # Mutate the text field in-place per example.
        new_batch = []
        for ex in batch:
            events = ex.get("structured_events", [])[:max_events]
            preamble = _serialize_events_compact(events) + "\n"
            new_ex = dict(ex)
            new_ex["text"] = preamble + ex["text"]
            new_batch.append(new_ex)
        return eval_mode_collator(new_batch)
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

    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    accelerator = Accelerator(mixed_precision=train_cfg.get("precision", "bf16"))
    device = accelerator.device

    # Use the merged Stage-0 CPT-light checkpoint as the base (NOT raw
    # Qwen3-8B). This baseline should have the same starting point as
    # the cross-attn arm so the comparison isolates the architectural
    # contribution, not the CPT-light contribution.
    model_id = train_cfg.get("base_checkpoint",
                              "/workspace/checkpoints/qwen3-8b-cpt-light-merged")
    tokenizer, n_new = prepare_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    old_vocab_size = model.config.vocab_size
    if n_new:
        # Should be 0 if the merged checkpoint already has the custom
        # tokens, but defensively resize.
        model.resize_token_embeddings(len(tokenizer))
        from src.tokenizer.custom_tokens import init_new_embeddings
        init_new_embeddings(model, tokenizer, old_vocab_size=old_vocab_size)

    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=CPT_LIGHT_LORA_TARGETS,
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    # Collator: eval-mode dropout WRAPPED to prepend structured-as-text.
    from src.train.mixers.eval_mode_dropout import EvalModeDropoutCollator
    inner = EvalModeDropoutCollator(
        tokenizer=tokenizer, text_field="text",
        max_length=train_cfg.get("seq_len", 2048),
        seed=cfg.get("training", {}).get("seed", 0),
    )
    collator = _structured_as_text_collator_factory(inner)

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
        model, lr=train_cfg.get("lr", 5e-5),
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

    print(f"structured_as_text: trainable params = {n_trainable:,}")
    print(f"train: {len(train_ds)} examples; eval: {len(eval_ds)} examples")
    print(f"total_steps={total_steps}")

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

    # Final three-mode eval. The eval text needs to ALSO have events
    # prepended; we synthesize a wrapper dataset that does that.
    run_dir = args.config.parent
    model.eval()

    # For the eval pass we need text with structured events prepended.
    # The eval_runner expects dataset[i]["text"] to be the full prefix
    # ending at "<risk_verdict>\nlabel:". We pre-transform eval_ds.
    eval_records = []
    for i in range(len(eval_ds)):
        ex = eval_ds[i]
        events = ex.get("structured_events", [])
        preamble = _serialize_events_compact(events) + "\n"
        new_ex = dict(ex)
        new_ex["text"] = preamble + ex["text"]
        eval_records.append(new_ex)
    from datasets import Dataset
    eval_ds_pre = Dataset.from_list(eval_records)

    from src.train.eval_runner import run_three_mode_eval
    eval_summary = run_three_mode_eval(
        model, eval_ds_pre, tokenizer, run_dir,
        modes=cfg.get("eval", {}).get("modes", ["stripped", "opaque", "full"]),
        batch_size=batch_size,
        max_length=train_cfg.get("seq_len", 2048),
    )

    metrics = {
        "status": "ok", "arm": "structured_as_text",
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
    model.save_pretrained(run_dir / "lora_adapter")
    print(f"wrote {metrics_path} + lora_adapter/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
