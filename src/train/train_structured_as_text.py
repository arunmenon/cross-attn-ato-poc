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
    """One-line-per-event serialization wrapped in <events>...</events>.

    LEGACY v3 path. Kept for backward compatibility with v3 datasets
    that lack the canonical `narrative` + `label` fields. v4 datasets
    use `compose_structured_as_text(record)` from build_dataset.py
    instead — see `_structured_as_text_collator_factory` below for
    the dispatch logic.

    Imports `event_to_line` directly from `data/gen/build_dataset.py` so
    the per-event format is byte-identical to the corpus's event lines
    (review 007 finding #5).
    """
    from data.gen.build_dataset import event_to_line
    lines = [event_to_line(ev) for ev in events]
    return "<events>\n" + "\n".join(lines) + "\n</events>"


def _structured_as_text_collator_factory(eval_mode_collator, max_events: int = 200):
    """Compose each row's text via the structured-as-text recipe BEFORE
    the eval-mode dropout transform is applied.

    Two code paths:

    1. **v4** (canonical-form data — has `narrative` + `label` fields):
       call `compose_structured_as_text(row)` from build_dataset.py,
       which produces:
            <case>
            <events>...</events>
            <narrative>...</narrative>
            <risk_verdict>label: {fraud|legit}</risk_verdict>
            </case>
       This is the single source of truth for the v4 structured_as_text
       prompt; xattn's text branch reads `compose_text_only(row)`
       against the same row, differing only by the absence of the
       `<events>` block. That's the architectural comparison.

    2. **v3** (legacy data — `text` is the v3 monolithic format):
       prepend `_serialize_events_compact(events)` to the existing
       `text` field. Same behavior as the original v3 trainer; kept
       so we can re-run the v3 baseline if needed for direct
       comparison.

    The dispatch is per-row, so a mixed dataset (rare) won't break —
    each row uses the right path based on whether it has canonical
    fields.
    """
    def _collator(batch):
        from src.train.common import parse_structured_events
        from data.gen.build_dataset import compose_structured_as_text

        new_batch = []
        for ex in batch:
            new_ex = dict(ex)
            events = parse_structured_events(new_ex)[:max_events]
            # Stash the parsed events back so compose_structured_as_text
            # sees the truncated list (v3 path expects events as a list,
            # v4's compose_structured_as_text reads new_ex["structured_events"]).
            new_ex["structured_events"] = events
            if "narrative" in new_ex and "label" in new_ex:
                # v4 path: single source of truth for the prompt format.
                new_ex["text"] = compose_structured_as_text(new_ex)
            else:
                # v3 fallback: prepend events block to existing v3 text.
                new_ex["text"] = (
                    _serialize_events_compact(events) + "\n" + new_ex["text"]
                )
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

    # Defensive embed_tokens unfreeze (review 007 finding #1). The
    # merged base SHOULD already include the custom-token embeddings
    # trained during Stage-0, so n_new will typically be 0 above and
    # this `modules_to_save` clause is a no-op. But if for any reason
    # the merged checkpoint is loaded with an older tokenizer that
    # lacks some custom tokens, defaulting to frozen-mean-init would
    # silently invalidate this baseline.
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=CPT_LIGHT_LORA_TARGETS,
        modules_to_save=["embed_tokens"],
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
    #
    # Review 021 finding #2 fix: this MUST use the same dispatch as the
    # training collator (v4 compose_structured_as_text vs v3 prepend),
    # otherwise the model trains on one prompt shape and is evaluated on
    # another — invalidating the structured_as_text_v4 measurement.
    # Review 011 finding #1: structured_events is a JSON string after
    # load_paired_dataset; parse before serialization.
    from src.train.common import parse_structured_events
    from data.gen.build_dataset import compose_structured_as_text

    eval_records = []
    for i in range(len(eval_ds)):
        ex = eval_ds[i]
        events = parse_structured_events(ex)
        new_ex = dict(ex)
        new_ex["structured_events"] = events  # ensure list, not JSON str
        if "narrative" in new_ex and "label" in new_ex:
            # v4 path — matches the training collator's v4 branch
            new_ex["text"] = compose_structured_as_text(new_ex)
        else:
            # v3 fallback — matches the training collator's v3 branch
            new_ex["text"] = _serialize_events_compact(events) + "\n" + ex["text"]
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
