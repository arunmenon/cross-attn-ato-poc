"""Qwen3-8B + cross-attention wrapper.

Wraps a frozen Qwen3 causal-LM with the cross-attention apparatus:
  - Side-stream encoder (small_transformer)
  - Perceiver-Resampler (compresses encoder output to K slots)
  - N gated cross-attention blocks, inserted at decoder layers per
    `insertion_pattern`
  - Optional fresh LoRA on the self-attention Q-projection (Stage-1 LoRA
    per PLAN.md "Stages and adapter lifecycle")

Per PLAN.md "Stages and adapter lifecycle":
  - Stage-0 (CPT-light) LoRA is merged into the base BEFORE this wrapper
    is constructed. `base_checkpoint` therefore points to the merged
    `qwen3-8b-cpt-light-merged` directory, not the raw Qwen3-8B.
  - The base is then frozen here. The trainable params are:
      side-stream encoder
      Perceiver-Resampler
      gated cross-attention blocks
      Stage-1 LoRA-on-Q (fresh)

Hook-based injection design:
  We attach a forward hook to each selected decoder layer in
  `base.model.layers[i]`. The hook fires AFTER the layer's forward,
  receives its output (a tuple in HF's transformer convention; we read
  the hidden state at index 0), runs the cross-attention block on it,
  and returns the modified output. The cross-attention K/V is cached
  once per `forward()` call (we precompute the resampler output before
  invoking the base model).

This pattern is robust to HF API drift and avoids subclassing
Qwen3DecoderLayer directly. Trade-off: hooks add a small overhead vs an
in-place modification, and they make the model harder to serialize
cleanly (state-dict-only saves are fine; full pickle is not). For a
3-day POC this is the right trade-off.

PLAN.md sweep dial:
  insertion_pattern ∈ {every_4, every_8, late_only}
  gate_init        ∈ {zero, small_0.01}
  resampler_slots  ∈ {64, 128}
  encoder          ∈ {small_transformer}
  lora_r_on_q      = 16
"""

from __future__ import annotations

import sys
from typing import Sequence

from src.model.cross_attn_block import GatedCrossAttnDense, GATE_INIT_VALUES
from src.model.encoders.small_transformer import (
    EventVocab, SmallTransformerEncoder, tokenize_events,
)
from src.model.resampler import PerceiverResampler


# ---------------------------------------------------------------------------
# Insertion-pattern math (pure-Python; no torch needed)
# ---------------------------------------------------------------------------

def compute_insertion_layers(pattern: str, n_hidden_layers: int) -> list[int]:
    """Resolve a sweep-dial string to a sorted list of layer indices.

    Qwen3-8B has 36 layers (per HF config; verified in
    scripts/preflight_check.py). Insertion strategies per PLAN.md
    Architecture:

      every_4:    indices 12, 16, 20, 24, 28, 32  (6 layers; starts at
                  the middle of the stack so x-attn doesn't disturb
                  early token-feature extraction)
      every_8:    indices 12, 20, 28                (3 layers; sparser)
      late_only:  indices last 4 layers of the stack (4 layers; signal
                  applied right before LM head)
    """
    if pattern == "every_4":
        return list(range(12, n_hidden_layers, 4))
    if pattern == "every_8":
        return list(range(12, n_hidden_layers, 8))
    if pattern == "late_only":
        return list(range(max(0, n_hidden_layers - 4), n_hidden_layers))
    raise ValueError(
        f"unknown insertion_pattern: {pattern!r}; "
        f"allowed: every_4, every_8, late_only"
    )


# ---------------------------------------------------------------------------
# Wrapper module (PyTorch — runs on the pod)
# ---------------------------------------------------------------------------

def _build_wrapper(
    base_model,
    *,
    insertion_layers: Sequence[int],
    n_slots: int,
    gate_init: str,
    encoder_hidden_dim: int,
    encoder_n_layers: int,
    encoder_n_heads: int,
    resampler_n_layers: int,
    resampler_n_heads: int,
    vocab: EventVocab,
):
    """Inner factory — torch + the constituent submodules are imported lazily."""
    import torch
    import torch.nn as nn

    hidden_size = base_model.config.hidden_size
    encoder = SmallTransformerEncoder(
        hidden_dim=encoder_hidden_dim,
        n_heads=encoder_n_heads,
        n_layers=encoder_n_layers,
        vocab=vocab,
    )
    resampler = PerceiverResampler(
        hidden_dim=encoder_hidden_dim,
        n_slots=n_slots,
        n_layers=resampler_n_layers,
        n_heads=resampler_n_heads,
    )
    if encoder_hidden_dim != hidden_size:
        kv_projection = nn.Linear(encoder_hidden_dim, hidden_size, bias=False)
    else:
        kv_projection = nn.Identity()

    xattn_blocks = nn.ModuleList([
        GatedCrossAttnDense(
            hidden_dim=hidden_size,
            n_heads=8,
            gate_init=gate_init,
        )
        for _ in insertion_layers
    ])

    class _QwenXAttnWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base_model
            for p in self.base.parameters():
                p.requires_grad = False

            self.encoder = encoder
            self.resampler = resampler
            self.kv_projection = kv_projection
            self.xattn_blocks = xattn_blocks
            self.insertion_layers = list(insertion_layers)

            self._cached_kv = None
            self._cached_kv_key_pad = None

            self._hook_handles = []
            for xattn_idx, layer_idx in enumerate(self.insertion_layers):
                layer = self.base.model.layers[layer_idx]
                handle = layer.register_forward_hook(self._make_hook(xattn_idx))
                self._hook_handles.append(handle)

        def _make_hook(self, xattn_idx: int):
            block = self.xattn_blocks[xattn_idx]

            def _hook(module, args, output):
                if isinstance(output, tuple):
                    h = output[0]
                    rest = output[1:]
                else:
                    h = output
                    rest = ()
                if self._cached_kv is None:
                    return output
                new_h = block(h, self._cached_kv, self._cached_kv_key_pad)
                if isinstance(output, tuple):
                    return (new_h, *rest)
                return new_h

            return _hook

        def precompute_kv(self, event_type_ids, bucket_ids, delta_t, event_mask):
            """Run encoder + resampler + projection. Caches the result
            for the upcoming base.forward() call. Caller must invoke
            this before each forward() pass requiring cross-attention.
            """
            enc_out = self.encoder(event_type_ids, bucket_ids, event_mask)
            kv_enc = self.resampler(enc_out, delta_t, event_mask)
            kv = self.kv_projection(kv_enc)
            self._cached_kv = kv
            self._cached_kv_key_pad = torch.zeros(
                kv.size(0), kv.size(1), dtype=torch.bool, device=kv.device,
            )
            return kv

        def clear_kv_cache(self) -> None:
            self._cached_kv = None
            self._cached_kv_key_pad = None

        def forward(self, *, input_ids, attention_mask, labels=None,
                    event_type_ids=None, bucket_ids=None, delta_t=None,
                    event_mask=None, **kwargs):
            """Convenience forward that wires K/V precomputation + base.forward() in one call.

            For training (with structured side present), all `event_*`
            args must be provided. For text-only inference (no
            cross-attention), pass `event_type_ids=None`.
            """
            if event_type_ids is not None:
                self.precompute_kv(event_type_ids, bucket_ids, delta_t, event_mask)
            else:
                self.clear_kv_cache()
            try:
                out = self.base(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs,
                )
            finally:
                self.clear_kv_cache()
            return out

        def attach_lora_on_q(self, r: int = 16, alpha: int = 16, dropout: float = 0.0):
            """Apply PEFT LoRA to the self-attention Q-projection only.

            Returns the PEFT-wrapped module that adds Stage-1 adapters
            on top of the (otherwise frozen) base.
            """
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj"],
            )
            self.base = get_peft_model(self.base, lora_config)
            return self.base

        def gate_diagnostics(self) -> list[tuple[int, float, float]]:
            """Per-block (layer_idx, |tanh(α_attn)|, |tanh(α_ffn)|).
            Used by the trainer's gate-trajectory logging."""
            out = []
            for layer_idx, block in zip(self.insertion_layers, self.xattn_blocks):
                a, f = block.gate_magnitudes()
                out.append((layer_idx, a, f))
            return out

        def trainable_param_summary(self) -> dict:
            """Counts of trainable vs frozen params, broken out by component."""
            counts = {
                "base_total": sum(p.numel() for p in self.base.parameters()),
                "base_trainable": sum(p.numel() for p in self.base.parameters()
                                      if p.requires_grad),
                "encoder": sum(p.numel() for p in self.encoder.parameters()),
                "resampler": sum(p.numel() for p in self.resampler.parameters()),
                "kv_projection": sum(p.numel() for p in self.kv_projection.parameters()),
                "xattn_blocks": sum(p.numel() for p in self.xattn_blocks.parameters()),
            }
            counts["xattn_machinery_total"] = (
                counts["encoder"] + counts["resampler"]
                + counts["kv_projection"] + counts["xattn_blocks"]
            )
            counts["total_trainable"] = (
                counts["base_trainable"] + counts["xattn_machinery_total"]
            )
            return counts

    return _QwenXAttnWrapper()


def QwenXAttnWrapper(
    base_model,
    *,
    insertion_pattern: str = "every_4",
    n_slots: int = 64,
    gate_init: str = "small_0.01",
    encoder_hidden_dim: int = 256,
    encoder_n_layers: int = 6,
    encoder_n_heads: int = 4,
    resampler_n_layers: int = 2,
    resampler_n_heads: int = 8,
    vocab: EventVocab | None = None,
):
    """Public factory.

    Args:
      base_model: a pre-loaded Qwen3-style HF causal-LM (typically the
                  merged `qwen3-8b-cpt-light-merged` checkpoint).
      insertion_pattern: 'every_4' | 'every_8' | 'late_only'.
      n_slots: Perceiver-Resampler K (64 or 128 per PLAN.md sweep).
      gate_init: 'zero' | 'small_0.01' (per PLAN.md sweep).
      encoder_*, resampler_*: side-stream encoder / resampler shapes.
      vocab: EventVocab instance (auto-created if None).

    Caller is responsible for moving the wrapper to the desired
    device/dtype (e.g., `.to(device).to(torch.bfloat16)`).
    """
    if gate_init not in GATE_INIT_VALUES:
        raise ValueError(
            f"unknown gate_init: {gate_init!r}; "
            f"allowed: {sorted(GATE_INIT_VALUES)}"
        )

    vocab = vocab or EventVocab()
    n_layers = base_model.config.num_hidden_layers
    insertion_layers = compute_insertion_layers(insertion_pattern, n_layers)

    if not insertion_layers:
        raise ValueError(
            f"insertion_pattern {insertion_pattern!r} resolves to no "
            f"layers for a base with {n_layers} hidden layers"
        )

    return _build_wrapper(
        base_model=base_model,
        insertion_layers=insertion_layers,
        n_slots=n_slots,
        gate_init=gate_init,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_n_layers=encoder_n_layers,
        encoder_n_heads=encoder_n_heads,
        resampler_n_layers=resampler_n_layers,
        resampler_n_heads=resampler_n_heads,
        vocab=vocab,
    )


# ---------------------------------------------------------------------------
# Self-test (uses a tiny mock HF model so we don't need real Qwen3 weights)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # 1. Insertion-layer math (no torch needed)
    assert compute_insertion_layers("every_4", 36) == [12, 16, 20, 24, 28, 32]
    assert compute_insertion_layers("every_8", 36) == [12, 20, 28]
    assert compute_insertion_layers("late_only", 36) == [32, 33, 34, 35]
    try:
        compute_insertion_layers("bogus", 36)
        raise AssertionError("expected ValueError on unknown pattern")
    except ValueError:
        pass
    print("compute_insertion_layers OK for every_4/every_8/late_only")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("torch not installed; wrapper integration self-test skipped "
              "(runs on the pod)")
        return

    # 2. Mock a minimal Qwen3-like HF model and exercise the wrapper.
    class _Cfg:
        hidden_size = 32
        num_hidden_layers = 8

    class _MockDecoderLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(_Cfg.hidden_size, _Cfg.hidden_size)
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(_Cfg.hidden_size, _Cfg.hidden_size)

        def forward(self, x, **kw):
            return (self.linear(x),)

    class _MockBase(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _Cfg()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(
                [_MockDecoderLayer() for _ in range(_Cfg.num_hidden_layers)]
            )

        def forward(self, input_ids, attention_mask, labels=None, **kw):
            B, T = input_ids.shape
            x = torch.randn(B, T, _Cfg.hidden_size, device=input_ids.device)
            for layer in self.model.layers:
                x = layer(x)[0]
            class _Out:
                pass
            o = _Out()
            o.loss = x.sum() if labels is not None else None
            o.logits = x
            return o

    base = _MockBase()
    # late_only because 8-layer mock won't have insertion layers under every_4.
    wrapper = QwenXAttnWrapper(
        base, insertion_pattern="late_only", n_slots=8,
        encoder_hidden_dim=_Cfg.hidden_size, encoder_n_layers=2,
        encoder_n_heads=2, resampler_n_layers=1, resampler_n_heads=2,
    )

    assert wrapper.insertion_layers == [4, 5, 6, 7], wrapper.insertion_layers
    assert len(wrapper.xattn_blocks) == 4
    print(f"hooks attached at layers {wrapper.insertion_layers}")

    vocab = EventVocab()
    events = [
        {"t": 0, "event": "login", "ip_risk": "<ip_risk=low>"},
        {"t": 30, "event": "txn", "amount_bucket": "<amount_bucket=high>"},
    ]
    toks = tokenize_events(events, vocab, max_events=4)
    event_type_ids = torch.tensor([toks["event_type_ids"]])
    bucket_ids = torch.tensor([toks["bucket_ids"]])
    delta_t = torch.tensor([toks["delta_t"]])
    event_mask = torch.tensor([toks["attention_mask"]])

    input_ids = torch.randint(0, 100, (1, 16))
    attn_mask = torch.ones(1, 16, dtype=torch.long)

    out = wrapper(
        input_ids=input_ids, attention_mask=attn_mask, labels=input_ids,
        event_type_ids=event_type_ids, bucket_ids=bucket_ids,
        delta_t=delta_t, event_mask=event_mask,
    )
    assert hasattr(out, "logits"), "expected HF-style output with .logits"
    print(f"forward with cross-attn OK; logits shape {tuple(out.logits.shape)}")

    diag = wrapper.gate_diagnostics()
    assert len(diag) == 4
    print(f"gate diagnostics: {diag}")

    summary = wrapper.trainable_param_summary()
    assert summary["base_trainable"] == 0, \
        f"base should be fully frozen, got {summary['base_trainable']} trainable"
    assert summary["xattn_machinery_total"] > 0
    print(f"trainable param summary: {summary}")


if __name__ == "__main__":
    _self_test()
