"""Three-mode eval pass producing predictions_<mode>.jsonl per run dir.

Called by every LM-based trainer at end-of-training (and optionally at
eval_steps intervals). For each of the three eval modes in
src/eval/eval_modes.py — stripped, opaque, full — runs:

  1. Apply the mode transform to each example's text.
  2. Tokenize with the trainer's tokenizer.
  3. Forward through the model.
  4. At the token aligned to "<risk_verdict>\\nlabel:" + 1, extract
     logits.
  5. Score = logP(' fraud') - logP(' legit') from those logits.
  6. Emit one record per example to predictions_<mode>.jsonl.

scripts/run_next_experiment.py then invokes eval/score_risk.py and
eval/bootstrap_ci.py against each of those files.

This module also writes a single aggregate ci_report.json by joining
the per-mode CI files (the wiring lives in run_next_experiment.py, not
here, but this module produces the inputs).

Signature:
  run_three_mode_eval(model, eval_dataset, tokenizer, run_dir, *,
                      modes, batch_size, device, ...)

For the event-only classifier baseline (no LM), use
run_classifier_eval instead — it has the same output shape but takes
a classifier model with .forward(events) -> logits.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from src.train.common import (
    LABEL_MARKER_STR, find_label_token_position, get_label_token_ids,
    write_predictions_jsonl,
)


def run_three_mode_eval(
    model,
    eval_dataset,
    tokenizer,
    run_dir: str | Path,
    *,
    modes: list[str] | None = None,
    batch_size: int = 4,
    max_length: int = 2048,
    device: str | None = None,
    structured_input_fn=None,
):
    """Run the three-mode eval pass.

    Args:
      model:              the LM (or LM wrapper). Must be in eval mode
                          before this call.
      eval_dataset:       HF Dataset with fields {text, label,
                          journey_family, actor_family, is_hard_negative,
                          structured_events (for x-attn trainers)}.
      tokenizer:          tokenizer with custom tokens installed.
      run_dir:            directory under which predictions_<mode>.jsonl
                          is written.
      modes:              eval modes to run; default
                          ["stripped", "opaque", "full"].
      batch_size:         eval batch size.
      max_length:         max input tokens per example.
      device:             torch device. Defaults to model's device.
      structured_input_fn:
                          optional callable(batch_of_examples) -> dict of
                          tensors for the wrapper's `event_*` args. None
                          for non-x-attn trainers.

    Returns: dict {mode: n_records_written}.
    """
    import torch

    if modes is None:
        modes = ["stripped", "opaque", "full"]

    if device is None:
        device = next(model.parameters()).device

    fraud_id, legit_id = get_label_token_ids(tokenizer)
    run_dir = Path(run_dir)

    summary: dict[str, int] = {}
    for mode in modes:
        # Apply mode transform per example. Use a per-example RNG so the
        # opaque mode's random mapping is reproducible (seeded from
        # example index for stable ordering across runs).
        from eval import eval_modes

        records: list[dict] = []
        # Batch iteration
        for batch_start in range(0, len(eval_dataset), batch_size):
            batch_examples = eval_dataset[batch_start : batch_start + batch_size]
            # HF Dataset slicing returns dict-of-lists. Normalize to
            # list-of-dicts for per-example handling.
            n = len(batch_examples["text"])
            ex_list = [
                {k: batch_examples[k][i] for k in batch_examples}
                for i in range(n)
            ]

            # Apply eval-mode transform per example (with per-example RNG
            # for opaque)
            transformed: list[str] = []
            for i, ex in enumerate(ex_list):
                rng = random.Random(batch_start + i) if mode == "opaque" else None
                new_text, _ = eval_modes.apply(ex["text"], mode, rng=rng)  # type: ignore[arg-type]
                # Truncate at the score position +1 token so the model
                # doesn't see the ground-truth label during scoring.
                # We keep up to and including the "label:" marker.
                marker_idx = new_text.find(LABEL_MARKER_STR)
                if marker_idx >= 0:
                    new_text = new_text[: marker_idx + len(LABEL_MARKER_STR)]
                transformed.append(new_text)

            # Tokenize
            enc = tokenizer(
                transformed,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # Optional structured-side input for x-attn wrapper
            structured_kwargs: dict[str, Any] = {}
            if structured_input_fn is not None:
                structured_kwargs = structured_input_fn(ex_list)
                structured_kwargs = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in structured_kwargs.items()
                }

            # Forward
            with torch.no_grad():
                out = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    **structured_kwargs,
                )
            logits = out.logits  # (B, T, V)

            # For each example, find the position-of-label-token,
            # extract its logits, score = logP(fraud) - logP(legit).
            for i, ex in enumerate(ex_list):
                # Token position of the FIRST predicted token (right
                # after the "label:" marker)
                input_ids_i = enc["input_ids"][i].tolist()
                # We tokenized text that ends with "label:". The
                # model's next-token prediction is at the LAST
                # non-pad position.
                attn_i = enc["attention_mask"][i]
                last_real = int(attn_i.sum().item()) - 1
                # log-softmax for numerical stability
                log_probs = torch.log_softmax(logits[i, last_real], dim=-1)
                score = float(
                    (log_probs[fraud_id] - log_probs[legit_id]).item()
                )
                records.append({
                    "score": score,
                    "label": ex["label"],
                    "journey_family": ex["journey_family"],
                    "actor_family": ex["actor_family"],
                    "is_hard_negative": ex["is_hard_negative"],
                })

        out_path = run_dir / f"predictions_{mode}.jsonl"
        write_predictions_jsonl(records, out_path)
        summary[mode] = len(records)

    return summary


def run_classifier_eval(
    model,
    eval_dataset,
    run_dir: str | Path,
    *,
    structured_input_fn,
    batch_size: int = 16,
    device: str | None = None,
):
    """Eval pass for the event-only classifier baseline. The classifier
    consumes only the structured event stream, so there's no eval-mode
    surface to apply (the journey/actor tokens live in the TEXT side,
    which this baseline does not see). We still emit per-mode files —
    all three modes contain identical predictions — so downstream
    scoring + bootstrap-CI tooling is shape-compatible with the LM
    trainers' outputs.

    Args:
      structured_input_fn: callable(batch_of_examples) -> dict with
                          'event_type_ids', 'bucket_ids', 'delta_t',
                          'event_mask' tensors.
    """
    import torch

    if device is None:
        device = next(model.parameters()).device

    run_dir = Path(run_dir)
    base_records: list[dict] = []

    for batch_start in range(0, len(eval_dataset), batch_size):
        batch_examples = eval_dataset[batch_start : batch_start + batch_size]
        n = len(batch_examples["label"])
        ex_list = [
            {k: batch_examples[k][i] for k in batch_examples}
            for i in range(n)
        ]
        structured = structured_input_fn(ex_list)
        structured = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in structured.items()
        }
        with torch.no_grad():
            logits = model(**structured)  # (B, 2) — fraud, legit
        log_probs = torch.log_softmax(logits, dim=-1)
        for i, ex in enumerate(ex_list):
            score = float((log_probs[i, 0] - log_probs[i, 1]).item())
            base_records.append({
                "score": score,
                "label": ex["label"],
                "journey_family": ex["journey_family"],
                "actor_family": ex["actor_family"],
                "is_hard_negative": ex["is_hard_negative"],
            })

    summary: dict[str, int] = {}
    for mode in ("stripped", "opaque", "full"):
        write_predictions_jsonl(base_records, run_dir / f"predictions_{mode}.jsonl")
        summary[mode] = len(base_records)
    return summary
