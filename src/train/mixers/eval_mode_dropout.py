"""Training-time eval-mode dropout.

For each training example, sample one of {full, opaque, stripped} with
probabilities {0.50, 0.25, 0.25} and apply the corresponding eval transform
*before* tokenization. This ensures the model sees all three distributions
during training and stripped-mode eval is not OOD.

Usage:
    from src.train.mixers.eval_mode_dropout import EvalModeDropoutCollator

    collator = EvalModeDropoutCollator(
        tokenizer=tokenizer,
        text_field="text",
        seed=42,
        probabilities={"full": 0.50, "opaque": 0.25, "stripped": 0.25},
    )
    dataloader = DataLoader(dataset, collate_fn=collator, batch_size=4)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping

import torch

from eval import eval_modes

DEFAULT_PROBS: dict[str, float] = {"full": 0.50, "opaque": 0.25, "stripped": 0.25}


@dataclass
class EvalModeDropoutCollator:
    """HF Trainer-compatible collator that applies eval-mode dropout per example.

    The tokenizer must accept the post-transform text. Custom tokens (journey,
    actor, bucketed-feature, event, PII) must be registered with the tokenizer
    *before* this collator is invoked (see `src/tokenizer/custom_tokens.py`).
    """

    tokenizer: object  # PreTrainedTokenizer
    text_field: str = "text"
    max_length: int = 2048
    pad_to_multiple_of: int | None = 8
    seed: int = 0
    probabilities: Mapping[str, float] = field(default_factory=lambda: DEFAULT_PROBS)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        modes = list(self.probabilities.keys())
        weights = list(self.probabilities.values())
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities must sum to 1.0, got {total}")
        for m in modes:
            if m not in {"full", "opaque", "stripped"}:
                raise ValueError(f"unknown mode: {m}")
        self._modes = modes
        self._weights = weights

    def _sample_mode(self) -> str:
        return self._rng.choices(self._modes, weights=self._weights, k=1)[0]

    def _transform(self, text: str) -> tuple[str, str]:
        mode = self._sample_mode()
        # For opaque mode, pass a fresh per-example RNG so the journey/actor ID
        # permutation is randomized between training examples. Without this,
        # opaque is effectively just renamed full and the model can decode the
        # stable mapping during training (review finding #6).
        per_example_rng = random.Random(self._rng.getrandbits(64)) if mode == "opaque" else None
        new_text, _ = eval_modes.apply(text, mode, rng=per_example_rng)  # type: ignore[arg-type]
        return new_text, mode

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        modes_used: list[str] = []
        for ex in batch:
            t = ex[self.text_field]
            new_t, mode = self._transform(t)
            texts.append(new_t)
            modes_used.append(mode)

        enc = self.tokenizer(  # type: ignore[operator]
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Labels are input_ids with pad masked to -100 (standard HF causal-LM convention).
        labels = enc["input_ids"].clone()
        pad_id = self.tokenizer.pad_token_id  # type: ignore[attr-defined]
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id is None; set tokenizer.pad_token before training")
        labels[labels == pad_id] = -100

        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }
        # Pass structured-stream and metadata through unchanged if present.
        for key in ("structured_events", "journey_family", "actor_family"):
            if key in batch[0]:
                out[key] = [ex.get(key) for ex in batch]  # type: ignore[assignment]
        # Diagnostic only — not consumed by the model.
        out["_eval_modes_sampled"] = modes_used  # type: ignore[assignment]
        return out
