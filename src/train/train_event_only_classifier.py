"""Event-only classifier baseline (#4 of 4).

The load-bearing comparator from PLAN.md "Baselines" §4. Asks: does the
LM matter at all for classification, or is the structured event stream
sufficient?

Architecture: `SmallTransformerEncoder` (the same one used by
QwenXAttnWrapper) + a 2-class linear head. No LM. No eval-mode dropout
(the eval modes only affect text-side tokens which this baseline does
not see).

If this baseline wins the bake-off, the Day-3 synthesis says
"cross-attn's value is explanation/grounding, not classification" —
that's a real finding per PLAN.md.

CLI:
    accelerate launch src/train/train_event_only_classifier.py \\
        --config src/auto_research/runs/exp_NNN/config.yaml

Outputs (per run dir):
  metrics.json         — training metrics (loss curve, final eval, etc.)
  predictions_<mode>.jsonl  — score-per-example for stripped/opaque/full
                              (all three are identical for this baseline
                              by construction; we emit all three so
                              downstream scoring is shape-compatible)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def collate_classifier_batch(examples, vocab, max_events=200):
    """Convert a list of dataset rows to the tensors the classifier eats.
    Pure helper — torch is imported lazily so the file is import-safe
    on laptops without torch.
    """
    import torch
    from src.model.encoders.small_transformer import tokenize_events

    type_ids_batch = []
    bucket_ids_batch = []
    delta_t_batch = []
    mask_batch = []
    labels = []
    for ex in examples:
        toks = tokenize_events(ex["structured_events"], vocab, max_events=max_events)
        type_ids_batch.append(toks["event_type_ids"])
        bucket_ids_batch.append(toks["bucket_ids"])
        delta_t_batch.append(toks["delta_t"])
        mask_batch.append(toks["attention_mask"])
        labels.append(0 if ex["label"] == "fraud" else 1)

    return {
        "event_type_ids": torch.tensor(type_ids_batch, dtype=torch.long),
        "bucket_ids": torch.tensor(bucket_ids_batch, dtype=torch.long),
        "delta_t": torch.tensor(delta_t_batch, dtype=torch.float32),
        "event_mask": torch.tensor(mask_batch, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _build_classifier(hidden_dim=256, n_layers=6, n_heads=4, vocab_size=None):
    """Lazy-import factory for the classifier nn.Module."""
    import torch
    import torch.nn as nn
    from src.model.encoders.small_transformer import SmallTransformerEncoder, EventVocab

    vocab = EventVocab()

    class _EventOnlyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = SmallTransformerEncoder(
                hidden_dim=hidden_dim, n_heads=n_heads, n_layers=n_layers,
                vocab=vocab,
            )
            # Pool + classification head. 2-class output: [fraud, legit].
            self.pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.head = nn.Linear(hidden_dim, 2)

        def forward(self, event_type_ids, bucket_ids, delta_t, event_mask, labels=None):
            del delta_t  # the classifier doesn't use time directly
            enc_out = self.encoder(event_type_ids, bucket_ids, event_mask)  # (B, N, H)
            mask = event_mask.unsqueeze(-1).to(enc_out.dtype)
            summed = (enc_out * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = self.head(self.pool(summed / counts))                  # (B, 2)
            return pooled  # logits

    return _EventOnlyClassifier(), vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    from src.train.common import load_config, load_paired_dataset, build_optimizer, build_lr_scheduler

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})

    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    # Dataset
    splits = load_paired_dataset(data_cfg["train_path"])
    train_ds = splits["train"]
    eval_ds = splits.get("eval") or train_ds  # fall back to train if no eval split

    # Model
    model, vocab = _build_classifier(hidden_dim=256, n_layers=6, n_heads=4)
    model = model.to(device)

    # Dataloaders
    batch_size = train_cfg.get("micro_batch", 8)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_classifier_batch(b, vocab),
    )
    eval_loader_fn = lambda: DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_classifier_batch(b, vocab),
    )

    # Optimizer + scheduler
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

    print(f"event_only_classifier: trainable params = {n_trainable:,}")
    print(f"train: {len(train_ds)} examples; eval: {len(eval_ds)} examples")
    print(f"total_steps={total_steps} lr={train_cfg.get('lr', 1e-4)}")

    # Training loop
    model.train()
    step = 0
    t_start = time.time()
    loss_fn = torch.nn.CrossEntropyLoss()
    losses: list[float] = []

    train_iter = iter(train_loader)
    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        logits = model(
            event_type_ids=batch["event_type_ids"],
            bucket_ids=batch["bucket_ids"],
            delta_t=batch["delta_t"],
            event_mask=batch["event_mask"],
        )
        loss = loss_fn(logits, batch["labels"])
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.append(loss.detach().item())
        if step % 100 == 0:
            print(f"step {step:>5} loss {loss.item():.4f}")
        step += 1

    # Final eval: produce per-mode predictions (all identical for this
    # baseline; structured-only score doesn't depend on eval modes).
    run_dir = args.config.parent
    model.eval()

    def _structured_input_fn(ex_list):
        b = collate_classifier_batch(ex_list, vocab)
        return {
            "event_type_ids": b["event_type_ids"],
            "bucket_ids": b["bucket_ids"],
            "delta_t": b["delta_t"],
            "event_mask": b["event_mask"],
        }

    from src.train.eval_runner import run_classifier_eval
    eval_summary = run_classifier_eval(
        model, eval_ds, run_dir,
        structured_input_fn=_structured_input_fn,
        batch_size=batch_size,
    )

    # Write metrics.json
    metrics = {
        "status": "ok",
        "arm": "event_only",
        "final_train_loss": losses[-1] if losses else None,
        "n_steps": step,
        "wall_clock_sec": time.time() - t_start,
        "n_trainable": n_trainable,
        "predictions": eval_summary,
        "max_gate_magnitude": None,  # no gates in this arm
    }
    metrics_path = run_dir / "metrics.json"
    tmp = metrics_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics, indent=2))
    tmp.rename(metrics_path)
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
