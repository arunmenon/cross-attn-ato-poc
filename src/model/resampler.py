"""Perceiver-Resampler with sinusoidal-on-Δt time encoding.

Takes the side-stream encoder's variable-length output (B, N, H) and
compresses to fixed-length K (B, K, H), where K = `n_slots`. The
compressed output is what the cross-attention layers in the Qwen3 stack
attend to (cache-once per session).

Per PLAN.md "Architecture":
  - K ∈ {64, 128} (sweepable).
  - Sinusoidal-on-Δt time encoding lives INSIDE the resampler, not in
    the encoder. Pair-2 consensus rationale: time is what makes fraud
    sequences different from other token streams (FraudTransformer), so
    its representation should sit alongside the bottleneck rather than
    be diffused into the encoder's per-event MLP.

Architecture (Perceiver-Resampler, Jaegle 2021 / Flamingo 2022):
  - Learned latents: (K, H), initialized with small random normal.
  - Stack of cross-attention blocks. Queries = latents (or previous
    block's output); keys/values = encoder output + time encoding.
  - Self-attention over latents between cross-attention rounds (Perceiver
    paper's pattern).

Public API:
  - sinusoidal_time_encoding(delta_t, hidden_dim) -> (B, N, H) tensor
  - PerceiverResampler(...) — nn.Module producing (B, K, H).
"""

from __future__ import annotations

import math
import sys


def _build_resampler(hidden_dim: int, n_slots: int, n_layers: int,
                    n_heads: int, dim_feedforward: int, dropout: float,
                    time_base: float):
    """Inner factory — imports torch lazily."""
    import torch
    import torch.nn as nn

    def sinusoidal_time_encoding(delta_t, hidden_dim: int,
                                 time_base: float = time_base):
        """Map (B, N) Δt-in-seconds to (B, N, hidden_dim) sinusoidal PE.

        Uses the standard transformer PE formula but with `pos = delta_t`
        (a continuous value) rather than an integer index, and
        `time_base` instead of 10_000. Cumulative time within a session
        is reconstructed by the model from the per-event Δt at attention
        time.
        """
        device = delta_t.device
        # Cumulative time relative to session start (sec). Δt is per-event
        # gap, so cumsum gives absolute time within the session.
        t = delta_t.cumsum(dim=1)                                       # (B, N)
        half = hidden_dim // 2
        freqs = torch.exp(
            -math.log(time_base) * torch.arange(0, half, device=device, dtype=t.dtype) / half
        )                                                                # (half,)
        # (B, N, half)
        args = t.unsqueeze(-1) * freqs.view(1, 1, -1)
        pe = torch.zeros(*t.shape, hidden_dim, device=device, dtype=t.dtype)
        pe[..., 0::2] = torch.sin(args)
        pe[..., 1::2] = torch.cos(args)
        return pe

    class _CrossAttnBlock(nn.Module):
        """Pre-norm cross-attention + self-attention + FFN, Perceiver
        style."""

        def __init__(self) -> None:
            super().__init__()
            self.norm_q1 = nn.LayerNorm(hidden_dim)
            self.norm_kv1 = nn.LayerNorm(hidden_dim)
            self.cross = nn.MultiheadAttention(
                hidden_dim, n_heads, dropout=dropout, batch_first=True
            )
            self.norm_q2 = nn.LayerNorm(hidden_dim)
            self.self_attn = nn.MultiheadAttention(
                hidden_dim, n_heads, dropout=dropout, batch_first=True
            )
            self.norm_ffn = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, hidden_dim),
                nn.Dropout(dropout),
            )

        def forward(self, latents, kv, kv_key_padding_mask):
            # latents: (B, K, H); kv: (B, N, H); kv_key_padding_mask: (B, N)
            q = self.norm_q1(latents)
            k = v = self.norm_kv1(kv)
            attn_out, _ = self.cross(
                q, k, v, key_padding_mask=kv_key_padding_mask, need_weights=False,
            )
            latents = latents + attn_out

            q2 = self.norm_q2(latents)
            sa_out, _ = self.self_attn(q2, q2, q2, need_weights=False)
            latents = latents + sa_out

            f = self.ffn(self.norm_ffn(latents))
            latents = latents + f
            return latents

    class _PerceiverResampler(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.latents = nn.Parameter(torch.randn(n_slots, hidden_dim) * 0.02)
            self.layers = nn.ModuleList([_CrossAttnBlock() for _ in range(n_layers)])
            self.norm_out = nn.LayerNorm(hidden_dim)

        def forward(self, encoder_output, delta_t, attention_mask):
            """encoder_output: (B, N, H), delta_t: (B, N), attention_mask: (B, N)
            Returns: (B, K, H) compressed latents.
            """
            # Add time encoding to K/V before attending.
            time_pe = sinusoidal_time_encoding(delta_t, hidden_dim)     # (B, N, H)
            kv = encoder_output + time_pe

            B = encoder_output.size(0)
            latents = self.latents.unsqueeze(0).expand(B, -1, -1)       # (B, K, H)
            # PyTorch wants True where masked-out
            key_pad = ~attention_mask.bool()
            for layer in self.layers:
                latents = layer(latents, kv, key_pad)
            return self.norm_out(latents)

    # Expose the sinusoidal helper for downstream use (e.g., the wrapper
    # may want it for diagnostics).
    return _PerceiverResampler(), sinusoidal_time_encoding


def PerceiverResampler(
    *,
    hidden_dim: int = 256,
    n_slots: int = 64,
    n_layers: int = 2,
    n_heads: int = 8,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    time_base: float = 10_000.0,
):
    """Factory for the Perceiver-Resampler module.

    Time encoding scale: `time_base` is the denominator in the sinusoidal
    PE formula. With `time_base=10_000` and Δt in seconds, the first
    frequency component has period ≈ 2π·10_000 ≈ 17.4 hours, which spans
    the expected session-length range (seconds to hours) richly enough.
    """
    resampler, _ = _build_resampler(
        hidden_dim=hidden_dim, n_slots=n_slots, n_layers=n_layers,
        n_heads=n_heads, dim_feedforward=dim_feedforward,
        dropout=dropout, time_base=time_base,
    )
    return resampler


def sinusoidal_time_encoding(delta_t, hidden_dim: int, time_base: float = 10_000.0):
    """Standalone helper. delta_t: tensor of shape (B, N) in seconds."""
    _, fn = _build_resampler(
        hidden_dim=hidden_dim, n_slots=1, n_layers=1, n_heads=1,
        dim_feedforward=1, dropout=0.0, time_base=time_base,
    )
    return fn(delta_t, hidden_dim, time_base)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    try:
        import torch
    except ImportError:
        print("torch not installed; resampler self-test skipped "
              "(runs on the pod)")
        return

    H, K, N = 32, 16, 12

    # 1. Time-encoding shape + finite
    dt = torch.tensor([[0.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0,
                        1920.0, 3840.0, 7680.0, 15360.0, 30720.0]])
    pe = sinusoidal_time_encoding(dt, H, time_base=10_000.0)
    assert pe.shape == (1, N, H), f"expected (1, {N}, {H}), got {pe.shape}"
    assert torch.isfinite(pe).all(), "time PE contains NaN/Inf"
    # Larger Δt -> noticeably different PE rows
    assert not torch.allclose(pe[0, 1], pe[0, N - 1]), "PE collapses over Δt range"
    print(f"sinusoidal_time_encoding OK; shape {tuple(pe.shape)}")

    # 2. Resampler forward
    resampler = PerceiverResampler(
        hidden_dim=H, n_slots=K, n_layers=2, n_heads=4,
        dim_feedforward=64,
    )
    encoder_output = torch.randn(2, N, H)
    delta_t = torch.tensor([[0.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0,
                             1920.0, 3840.0, 7680.0, 15360.0, 30720.0]] * 2)
    attention_mask = torch.tensor([[1] * N, [1] * 6 + [0] * 6])  # second row half-padded
    out = resampler(encoder_output, delta_t, attention_mask)
    assert out.shape == (2, K, H), f"expected (2, {K}, {H}), got {out.shape}"
    assert torch.isfinite(out).all(), "resampler output contains NaN/Inf"
    print(f"PerceiverResampler forward OK; output shape {tuple(out.shape)}")

    # 3. Padded positions should not change the output meaningfully if we
    # swap values at padded positions (since key_padding_mask should
    # exclude them from cross-attention).
    encoder_output_b = encoder_output.clone()
    encoder_output_b[1, 6:] = 999.0  # set padded positions to extreme values
    out_b = resampler(encoder_output_b, delta_t, attention_mask)
    # Row 0 (no padding) should be unchanged
    assert torch.allclose(out[0], out_b[0], atol=1e-5), "padding mask leaks into unpadded row"
    # Row 1 (half-padded) should also be unchanged since padded positions
    # shouldn't have been attended to.
    assert torch.allclose(out[1], out_b[1], atol=1e-5), \
        "padding mask not enforced — padded positions affect output"
    print("padding mask enforced OK")


if __name__ == "__main__":
    _self_test()
