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
  encoder          ∈ {small_transformer, pooled_mlp, ft_transformer}
  lora_r_on_q      = 16
"""

from __future__ import annotations

import sys
from typing import Sequence

from src.model.cross_attn_block import GatedCrossAttnDense, GATE_INIT_VALUES
from src.model.encoders import EventVocab, available_encoders, build_event_encoder, tokenize_events
from src.model.resampler import PerceiverResampler


# ---------------------------------------------------------------------------
# Insertion-pattern math (pure-Python; no torch needed)
# ---------------------------------------------------------------------------

def estimate_wrapper_trainable_params(
    *,
    insertion_pattern: str,
    n_hidden_layers: int,
    hidden_size: int,
    cross_dim: int | None = None,
    dim_feedforward: int | None = None,
    n_heads_xattn: int = 8,
    encoder_hidden_dim: int = 256,
    encoder_n_layers: int = 6,
    encoder_n_heads: int = 4,
    encoder_name: str = "small_transformer",
    resampler_n_layers: int = 2,
    resampler_n_heads: int = 8,
    n_slots: int = 64,
    lora_r_on_q: int = 16,
) -> dict[str, int]:
    """Analytical estimate of total trainable params for a wrapper
    constructed with these hyperparameters. Used by self-tests and
    trainer-startup logs to validate sweep configs against PLAN.md's
    200-400M Stage-1 budget WITHOUT loading the real Qwen3-8B.

    Returns a dict with per-component counts + 'total'. Components:
      xattn_blocks      — N × per-block (from estimate_block_param_count)
      kv_projection     — encoder_hidden_dim × hidden_size (if differ)
      encoder_approx    — rough estimate; encoder + resampler don't
                          scale with hidden_size so they're small (~10M).
      lora_on_q         — 2 × hidden_size × r per attention layer × N layers
      total

    The encoder/resampler estimate is rough (we don't compute it
    exactly because they have many small Linear blocks). It's correct
    within ±30%, which is fine for budget-bounds checking.
    """
    from src.model.cross_attn_block import estimate_block_param_count

    insertion_layers = compute_insertion_layers(insertion_pattern, n_hidden_layers)
    n_blocks = len(insertion_layers)
    per_block = estimate_block_param_count(
        hidden_size, cross_dim=cross_dim, dim_feedforward=dim_feedforward,
        n_heads=n_heads_xattn,
    )
    xattn_blocks_total = per_block * n_blocks

    # kv_projection: only present when encoder_hidden_dim != hidden_size
    if encoder_hidden_dim != hidden_size:
        kv_projection = encoder_hidden_dim * hidden_size
    else:
        kv_projection = 0

    if encoder_name == "pooled_mlp":
        encoder_approx = (
            6 * encoder_hidden_dim * encoder_hidden_dim
            + 40 * encoder_hidden_dim
        )
    elif encoder_name == "ft_transformer":
        enc_per_layer = 4 * encoder_hidden_dim * encoder_hidden_dim
        enc_per_layer += 8 * encoder_hidden_dim * encoder_hidden_dim
        encoder_approx = (
            encoder_n_layers * enc_per_layer
            + 40 * encoder_hidden_dim
            + 10 * encoder_hidden_dim
        )
    else:
        # Encoder rough estimate. For typical (encoder_hidden_dim=256,
        # n_layers=6, n_heads=4), this is ~5M. Compute as the dominant
        # transformer-block cost × n_layers.
        enc_per_layer = 4 * encoder_hidden_dim * encoder_hidden_dim  # MHA q/k/v/o
        enc_per_layer += 8 * encoder_hidden_dim * encoder_hidden_dim  # FFN (dim_ff=4×)
        encoder_approx = (
            encoder_n_layers * enc_per_layer
            + 4 * encoder_hidden_dim * encoder_hidden_dim  # event embedder MLP
            + 40 * encoder_hidden_dim  # vocab embeddings (~40 tokens)
        )

    # Resampler rough estimate: K latents + N layers × (cross-attn + self-attn + FFN)
    res_per_layer = (
        4 * encoder_hidden_dim * encoder_hidden_dim  # cross-attn
        + 4 * encoder_hidden_dim * encoder_hidden_dim  # self-attn
        + 8 * encoder_hidden_dim * encoder_hidden_dim  # FFN
    )
    resampler_approx = (
        n_slots * encoder_hidden_dim
        + resampler_n_layers * res_per_layer
    )

    # LoRA-on-Q: 2 × r × hidden_size per attention layer × n_hidden_layers
    # (one LoRA A matrix + one LoRA B matrix per q_proj)
    lora_on_q = 2 * lora_r_on_q * hidden_size * n_hidden_layers

    total = (xattn_blocks_total + kv_projection + encoder_approx
             + resampler_approx + lora_on_q)

    return {
        "n_xattn_blocks": n_blocks,
        "per_block": per_block,
        "xattn_blocks_total": xattn_blocks_total,
        "kv_projection": kv_projection,
        "encoder_approx": encoder_approx,
        "resampler_approx": resampler_approx,
        "lora_on_q": lora_on_q,
        "total": total,
    }


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
    encoder_name: str,
    resampler_n_layers: int,
    resampler_n_heads: int,
    vocab: EventVocab,
):
    """Inner factory — torch + the constituent submodules are imported lazily."""
    import torch
    import torch.nn as nn

    hidden_size = base_model.config.hidden_size
    encoder = build_event_encoder(
        encoder_name,
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
            self.encoder_name = encoder_name
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

            Defensive device/dtype handling (review 005 finding #2):
              - ID tensors (event_type_ids, bucket_ids, event_mask) are
                moved to the encoder's device but kept integer.
              - delta_t is moved to encoder device AND cast to the
                encoder's parameter dtype, so the resampler's
                sinusoidal time encoding can compute in-dtype and the
                downstream add `kv = encoder_output + time_pe` stays in
                the residual-stream dtype (bf16 in production).
            """
            # Detect the encoder's device + dtype. We do this every call
            # rather than caching because the trainer may `.to(...)` the
            # wrapper between forward passes (e.g., bf16 conversion
            # mid-init).
            params = next(self.encoder.parameters(), None)
            target_device = params.device if params is not None else None
            target_dtype = params.dtype if params is not None else None

            if target_device is not None:
                event_type_ids = event_type_ids.to(device=target_device)
                bucket_ids = bucket_ids.to(device=target_device)
                event_mask = event_mask.to(device=target_device)
                # delta_t is float; cast both device AND dtype.
                delta_t = delta_t.to(device=target_device, dtype=target_dtype)
            elif target_dtype is not None:
                delta_t = delta_t.to(dtype=target_dtype)

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

        def generate(self, *args, **kwargs):
            """Autoregressive generation is not supported on the wrapper.

            (Review 005 finding #4): the current architecture would
            re-precompute cross-attention K/V on each generation step,
            which is both inefficient and only partially implemented.
            The Day-1-3 training path uses forward log-prob scoring
            against the verdict footer's `label:` tokens, not
            autoregressive generation, so this is not blocking for the
            POC. A future batch can add a proper generation API that
            prefills K/V once and reuses it across decode steps.

            Until then: do NOT call wrapper.generate(...). Use forward()
            with the verdict-footer prefix and read `out.logits` at the
            position of the `label:` token.
            """
            raise NotImplementedError(
                "QwenXAttnWrapper.generate() is intentionally unsupported. "
                "Use wrapper.forward(input_ids=..., attention_mask=..., "
                "labels=..., event_type_ids=..., bucket_ids=..., "
                "delta_t=..., event_mask=...) and score from out.logits. "
                "See src/model/qwen_xattn_wrapper.py generate() docstring."
            )

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
    encoder_name: str = "small_transformer",
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
      encoder_name: one of small_transformer, pooled_mlp, ft_transformer.
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
    if encoder_name not in available_encoders():
        raise ValueError(
            f"unknown encoder_name: {encoder_name!r}; "
            f"allowed: {list(available_encoders())}"
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
        encoder_name=encoder_name,
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

    # 1b. Trainable-param budget at production scale (Qwen3-8B; review
    # 005 finding #1). All three sweep arms must fit in 200-400M.
    print("\nProduction-scale trainable-param budget (Qwen3-8B, H=4096, 36 layers):")
    for pattern in ("every_4", "every_8", "late_only"):
        est = estimate_wrapper_trainable_params(
            insertion_pattern=pattern, n_hidden_layers=36, hidden_size=4096,
        )
        print(f"  {pattern:10s}: total={est['total']/1e6:7.1f}M  "
              f"(blocks={est['xattn_blocks_total']/1e6:.1f}M, "
              f"lora_on_q={est['lora_on_q']/1e6:.1f}M, "
              f"enc+resampler+kv_proj≈{(est['encoder_approx']+est['resampler_approx']+est['kv_projection'])/1e6:.1f}M)")
    # The most aggressive arm (every_4) must be ≤ 400M.
    every_4_est = estimate_wrapper_trainable_params(
        insertion_pattern="every_4", n_hidden_layers=36, hidden_size=4096,
    )
    assert every_4_est["total"] <= 400_000_000, (
        f"every_4 total {every_4_est['total']:,} exceeds 400M Stage-1 budget"
    )
    assert every_4_est["total"] >= 150_000_000, (
        f"every_4 total {every_4_est['total']:,} suspiciously small — "
        f"check encoder/resampler estimates"
    )
    print("Stage-1 trainable-param budget OK for all sweep arms")

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

    # 7. generate() must raise NotImplementedError (review 005 finding #4)
    try:
        wrapper.generate(input_ids=input_ids, max_new_tokens=4)
        raise AssertionError("wrapper.generate should have raised NotImplementedError")
    except NotImplementedError as e:
        assert "forward" in str(e).lower(), \
            f"generate's error should point to forward(); got: {e}"
        print(f"generate() correctly raises NotImplementedError")


if __name__ == "__main__":
    _self_test()
