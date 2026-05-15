"""Gated cross-attention dense block (Flamingo GATED XATTN-DENSE).

Inserted at selected layers in the Qwen3-8B decoder. At insertion, the
block:

  1. Takes the LM's hidden state `h` of shape (B, T, H).
  2. Cross-attends `h` (as queries) into the resampler output
     `kv` of shape (B, K, H) (keys + values).
  3. Adds the cross-attention output back to `h`, scaled by
     tanh(α_attn).
  4. Applies an FFN on the result, scaled by tanh(α_ffn), again
     residual-added.

The two gates `α_attn` and `α_ffn` are scalar (or 1-d) parameters,
initialized at `gate_init`:

  - "zero":       α = 0   (tanh(0) = 0 — true Flamingo init; the block is
                  the identity at step 0 and the frozen LM is preserved
                  exactly).
  - "small_0.01": α = 0.01 (tanh(0.01) ≈ 0.01 — a faint nudge to break
                  symmetry; sometimes trains faster).

PLAN.md sweep dial:
  insertion_pattern ∈ {every_4, every_8, late_only}
  gate_init        ∈ {zero, small_0.01}
"""

from __future__ import annotations

import sys


GATE_INIT_VALUES: dict[str, float] = {
    "zero":       0.0,
    "small_0.01": 0.01,
}


def _build_gated_xattn(hidden_dim: int, n_heads: int, dim_feedforward: int,
                     dropout: float, gate_init: str):
    """Inner factory — torch is imported lazily."""
    if gate_init not in GATE_INIT_VALUES:
        raise ValueError(
            f"unknown gate_init: {gate_init!r}; "
            f"allowed: {sorted(GATE_INIT_VALUES)}"
        )
    init_val = GATE_INIT_VALUES[gate_init]

    import torch
    import torch.nn as nn

    class _GatedCrossAttnDense(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm_q = nn.LayerNorm(hidden_dim)
            self.norm_kv = nn.LayerNorm(hidden_dim)
            self.cross = nn.MultiheadAttention(
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
            # Per-block gate parameters. Scalar so the gate magnitude is
            # interpretable in the training log (a single value per
            # block, easy to plot).
            self.alpha_attn = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
            self.alpha_ffn = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

        def forward(self, h, kv, kv_key_padding_mask):
            """h: (B, T, H); kv: (B, K, H); kv_key_padding_mask: (B, K).

            Returns: (B, T, H) — same shape as the input hidden state, so
            the wrapper can drop this in as a residual-additive layer.
            """
            q = self.norm_q(h)
            k = v = self.norm_kv(kv)
            attn_out, _ = self.cross(
                q, k, v, key_padding_mask=kv_key_padding_mask, need_weights=False,
            )
            # tanh-gated residual add. dtype-cast to h.dtype so we never
            # accidentally upcast in bf16/fp16 runs.
            gate_a = torch.tanh(self.alpha_attn).to(h.dtype)
            h = h + gate_a * attn_out

            f = self.ffn(self.norm_ffn(h))
            gate_f = torch.tanh(self.alpha_ffn).to(h.dtype)
            h = h + gate_f * f
            return h

        def gate_magnitudes(self) -> tuple[float, float]:
            """Diagnostic helper: returns (|tanh(α_attn)|, |tanh(α_ffn)|)
            as Python floats. Used by the trainer to log per-step gate
            trajectories for the convergence/zero-gates halt checks.
            """
            return (
                float(torch.tanh(self.alpha_attn).detach().abs().item()),
                float(torch.tanh(self.alpha_ffn).detach().abs().item()),
            )

    return _GatedCrossAttnDense()


def GatedCrossAttnDense(
    *,
    hidden_dim: int,
    n_heads: int = 8,
    dim_feedforward: int | None = None,
    dropout: float = 0.0,
    gate_init: str = "small_0.01",
):
    """Factory. `gate_init` must be one of GATE_INIT_VALUES keys."""
    if dim_feedforward is None:
        dim_feedforward = 4 * hidden_dim
    return _build_gated_xattn(
        hidden_dim=hidden_dim, n_heads=n_heads,
        dim_feedforward=dim_feedforward, dropout=dropout,
        gate_init=gate_init,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    try:
        import torch
    except ImportError:
        print("torch not installed; cross_attn_block self-test skipped "
              "(runs on the pod)")
        return

    H, T, K = 32, 10, 16

    # 1. zero-gate init: block is exactly identity at step 0
    block = GatedCrossAttnDense(hidden_dim=H, n_heads=4, dim_feedforward=64,
                                gate_init="zero")
    h = torch.randn(2, T, H, requires_grad=True)
    kv = torch.randn(2, K, H)
    kv_mask = torch.zeros(2, K, dtype=torch.bool)  # nothing masked
    out = block(h, kv, kv_mask)
    assert out.shape == h.shape
    # With α=0 everywhere, output should equal input bit-for-bit
    assert torch.allclose(out, h, atol=1e-6), \
        "zero-gate init must produce identity at step 0"
    gates = block.gate_magnitudes()
    assert gates == (0.0, 0.0), f"expected (0, 0) gates, got {gates}"
    print(f"zero-gate identity OK; gates={gates}")

    # 2. small_0.01-gate: block is NOT identity (gates ~0.01)
    block2 = GatedCrossAttnDense(hidden_dim=H, n_heads=4, dim_feedforward=64,
                                 gate_init="small_0.01")
    out2 = block2(h, kv, kv_mask)
    assert not torch.allclose(out2, h, atol=1e-3), \
        "small_0.01-gate should not be identity (block has signal)"
    gates2 = block2.gate_magnitudes()
    # tanh(0.01) ≈ 0.00999 — close to 0.01
    assert 0.005 < gates2[0] < 0.02, f"unexpected attn gate magnitude: {gates2[0]}"
    print(f"small_0.01 gate produces signal OK; gates={gates2}")

    # 3. Gradient flows through the block
    loss = out2.sum()
    loss.backward()
    assert block2.alpha_attn.grad is not None, "alpha_attn has no gradient"
    assert block2.alpha_ffn.grad is not None, "alpha_ffn has no gradient"
    assert h.grad is not None, "input hidden state has no gradient"
    print("gradient flow OK")

    # 4. KV padding mask is honored
    kv_mask2 = torch.zeros(2, K, dtype=torch.bool)
    kv_mask2[0, K // 2:] = True  # half-mask row 0
    out_a = block2(h, kv, kv_mask2)
    # Re-run with the masked positions corrupted; row 0 output should be
    # unchanged.
    kv_corrupt = kv.clone()
    kv_corrupt[0, K // 2:] = 999.0
    out_b = block2(h, kv_corrupt, kv_mask2)
    assert torch.allclose(out_a[0], out_b[0], atol=1e-5), \
        "kv padding mask not enforced in row 0"
    print("kv padding mask enforced OK")


if __name__ == "__main__":
    _self_test()
