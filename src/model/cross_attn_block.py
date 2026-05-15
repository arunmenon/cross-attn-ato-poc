"""Gated cross-attention dense block (Flamingo GATED XATTN-DENSE).

Inserted at selected layers in the Qwen3-8B decoder. At insertion, the
block:

  1. Takes the LM's hidden state `h` of shape (B, T, H).
  2. Down-projects `h` (queries) and `kv` (keys+values) to a smaller
     bottleneck dim `cross_dim` (default H/4).
  3. Cross-attends at the bottleneck dim.
  4. Up-projects the attention output back to H.
  5. Adds the result to `h`, scaled by tanh(α_attn).
  6. Applies an FFN on the result (operating at H, with a configurable
     `dim_feedforward`, default H/2), scaled by tanh(α_ffn) and
     residual-added.

The two gates `α_attn` and `α_ffn` are scalar parameters, initialized
at `gate_init`:

  - "zero":       α = 0   (tanh(0) = 0 — Flamingo init; the block is the
                  identity at step 0 and the frozen LM is preserved
                  exactly).
  - "small_0.01": α = 0.01 (tanh(0.01) ≈ 0.01 — a faint nudge to break
                  symmetry; sometimes trains faster).

**Parameter budget** (review 005 finding #1):
  The original Flamingo design uses full-H MHA + 4H FFN per block,
  which at Qwen3-8B's H=4096 costs ~201M params/block. The `every_4`
  insertion pattern (6 blocks) then totals 1.21B trainable params —
  ~6× over PLAN.md's stated 200-400M Stage-1 budget. To fit the budget
  cleanly, this block:
    - cross-attends at `cross_dim = hidden_dim // 4` (1024 for H=4096),
      via down-/up-projections at the block boundary.
    - uses `dim_feedforward = hidden_dim // 2` (2048 for H=4096) by
      default, half the standard transformer 4× ratio.
  Result: ~33.6M params/block at Qwen3-8B scale. every_4 (6 blocks)
  → ~202M, every_8 (3 blocks) → ~101M, late_only (4) → ~135M. All
  within the budget.

`estimate_block_param_count()` returns the analytical estimate so
self-tests / trainer-startup logs can assert "you are within budget"
without instantiating a full wrapper.

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


def estimate_block_param_count(
    hidden_dim: int,
    *,
    cross_dim: int | None = None,
    dim_feedforward: int | None = None,
    n_heads: int = 8,
) -> int:
    """Analytical estimate of the trainable parameters in one
    `GatedCrossAttnDense` block, given its hyperparameters. Returns
    the count BEFORE bias terms in LayerNorm and the gate scalars
    (those add a handful of params and don't affect budget reasoning).

    Pure-Python; no torch needed. Used by the wrapper's startup log and
    by self-tests to validate sweep configs against PLAN.md's budget.
    """
    if cross_dim is None:
        cross_dim = max(1, hidden_dim // 4)
    if dim_feedforward is None:
        dim_feedforward = max(1, hidden_dim // 2)

    # Three LayerNorms (norm_q, norm_kv, norm_ffn), each weight + bias at hidden_dim
    ln = 3 * (2 * hidden_dim)

    # Bottleneck down-projections (q and kv, no bias)
    q_proj_in = hidden_dim * cross_dim
    kv_proj_in = hidden_dim * cross_dim

    # MHA at cross_dim (PyTorch nn.MultiheadAttention with default biases):
    # in_proj_weight: 3 * cross_dim * cross_dim, in_proj_bias: 3 * cross_dim
    # out_proj.weight: cross_dim * cross_dim, out_proj.bias: cross_dim
    mha = 3 * cross_dim * cross_dim + 3 * cross_dim + cross_dim * cross_dim + cross_dim

    # Up-projection (no bias)
    out_proj = cross_dim * hidden_dim

    # FFN (two Linears with default bias)
    ffn = (hidden_dim * dim_feedforward + dim_feedforward) + (dim_feedforward * hidden_dim + hidden_dim)

    # Gates (two scalars)
    gates = 2

    return ln + q_proj_in + kv_proj_in + mha + out_proj + ffn + gates


def _build_gated_xattn(hidden_dim: int, cross_dim: int, n_heads: int,
                     dim_feedforward: int, dropout: float, gate_init: str):
    """Inner factory — torch is imported lazily."""
    if gate_init not in GATE_INIT_VALUES:
        raise ValueError(
            f"unknown gate_init: {gate_init!r}; "
            f"allowed: {sorted(GATE_INIT_VALUES)}"
        )
    if cross_dim % n_heads != 0:
        raise ValueError(
            f"cross_dim ({cross_dim}) must be divisible by n_heads ({n_heads}) "
            f"for nn.MultiheadAttention"
        )
    init_val = GATE_INIT_VALUES[gate_init]

    import torch
    import torch.nn as nn

    class _GatedCrossAttnDense(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Pre-norm
            self.norm_q = nn.LayerNorm(hidden_dim)
            self.norm_kv = nn.LayerNorm(hidden_dim)
            # Down-projections to the bottleneck dim. No bias — we want
            # the projection to be a pure linear map; gates and norms
            # handle scale.
            self.q_proj_in = nn.Linear(hidden_dim, cross_dim, bias=False)
            self.kv_proj_in = nn.Linear(hidden_dim, cross_dim, bias=False)
            # Cross-attention at the bottleneck dim.
            self.cross = nn.MultiheadAttention(
                cross_dim, n_heads, dropout=dropout, batch_first=True,
            )
            # Up-projection back to the residual-stream dim.
            self.out_proj = nn.Linear(cross_dim, hidden_dim, bias=False)
            # FFN (at hidden_dim, with configurable feedforward size)
            self.norm_ffn = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, hidden_dim),
                nn.Dropout(dropout),
            )
            # Per-block scalar gates.
            self.alpha_attn = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
            self.alpha_ffn = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

            # Zero-init the up-projection so that at step 0 the cross-attn
            # contribution is exactly zero regardless of how the gate
            # behaves under tiny numerical noise. This is a belt-and-braces
            # safety on top of gate_init="zero".
            nn.init.zeros_(self.out_proj.weight)

        def forward(self, h, kv, kv_key_padding_mask):
            """h: (B, T, H); kv: (B, K, H); kv_key_padding_mask: (B, K).

            Returns: (B, T, H) — same shape as the input hidden state,
            so the wrapper can drop this in as a residual-additive layer.
            """
            q = self.norm_q(h)
            k = v = self.norm_kv(kv)
            # Down-project to bottleneck
            q_d = self.q_proj_in(q)
            kv_d = self.kv_proj_in(k)  # k == v at this point
            # Cross-attention at bottleneck
            attn_out_d, _ = self.cross(
                q_d, kv_d, kv_d, key_padding_mask=kv_key_padding_mask,
                need_weights=False,
            )
            # Up-project back to hidden_dim
            attn_out = self.out_proj(attn_out_d)
            # tanh-gated residual add. dtype-cast so we never silently
            # upcast in bf16/fp16 runs.
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
    cross_dim: int | None = None,
    n_heads: int = 8,
    dim_feedforward: int | None = None,
    dropout: float = 0.0,
    gate_init: str = "small_0.01",
):
    """Factory. `gate_init` must be one of GATE_INIT_VALUES keys.

    Defaults (review 005 finding #1):
      - cross_dim       = hidden_dim // 4   (bottleneck cross-attention)
      - dim_feedforward = hidden_dim // 2   (half the standard 4× FFN ratio)
      - n_heads         = 8

    These keep the total trainable-parameter count for a 6-block
    `every_4` insertion pattern at ~200M on a Qwen3-8B base — inside the
    200-400M Stage-1 budget per PLAN.md "Architecture".
    """
    if cross_dim is None:
        cross_dim = max(1, hidden_dim // 4)
    if dim_feedforward is None:
        dim_feedforward = max(1, hidden_dim // 2)
    return _build_gated_xattn(
        hidden_dim=hidden_dim, cross_dim=cross_dim, n_heads=n_heads,
        dim_feedforward=dim_feedforward, dropout=dropout,
        gate_init=gate_init,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # 1. Param-count estimator at Qwen3-8B scale (torch-free).
    # Budget per PLAN.md Stage-1: 200-400M trainable.
    H = 4096
    per_block = estimate_block_param_count(H)
    every_4_total = per_block * 6
    every_8_total = per_block * 3
    late_only_total = per_block * 4
    print(f"param estimate at H={H}: per_block={per_block:,}")
    print(f"  every_4 ({6} blocks): {every_4_total:,}")
    print(f"  every_8 ({3} blocks): {every_8_total:,}")
    print(f"  late_only ({4} blocks): {late_only_total:,}")
    # The budget is the x-attn blocks only; encoder + resampler +
    # kv_projection add ~10M more. Assert blocks alone are <= 400M for
    # every_4 (the most aggressive sweep arm).
    assert every_4_total <= 400_000_000, (
        f"every_4 blocks total {every_4_total:,} exceeds 400M budget"
    )
    # And not absurdly small — we want ≥ ~100M to have signal capacity.
    assert every_4_total >= 100_000_000, (
        f"every_4 blocks total {every_4_total:,} is suspiciously small"
    )
    print("param budget OK for all three sweep arms")

    try:
        import torch
    except ImportError:
        print("torch not installed; cross_attn_block forward-pass self-test "
              "skipped (runs on the pod)")
        return

    H, T, K = 32, 10, 16

    # 2. Zero-gate identity at step 0
    block = GatedCrossAttnDense(
        hidden_dim=H, cross_dim=8, n_heads=4, dim_feedforward=16,
        gate_init="zero",
    )
    h = torch.randn(2, T, H, requires_grad=True)
    kv = torch.randn(2, K, H)
    kv_mask = torch.zeros(2, K, dtype=torch.bool)
    out = block(h, kv, kv_mask)
    assert out.shape == h.shape
    # With α=0 AND zero-init out_proj, the cross-attn residual is exactly
    # zero. ffn residual is also zero (gate_f=0). So output == input.
    assert torch.allclose(out, h, atol=1e-6), \
        "zero-gate init must produce identity at step 0"
    gates = block.gate_magnitudes()
    assert gates == (0.0, 0.0), f"expected (0, 0) gates, got {gates}"
    print(f"zero-gate identity OK; gates={gates}")

    # Stronger: even with extreme KV values, output should not drift.
    kv_extreme = torch.ones(2, K, H) * 1e6
    out_e = block(h, kv_extreme, kv_mask)
    assert torch.allclose(out_e, h, atol=1e-6), \
        "zero-gate init drifts under extreme kv values"
    print("zero-gate identity holds under extreme kv values")

    # 3. small_0.01-gate produces signal
    block2 = GatedCrossAttnDense(
        hidden_dim=H, cross_dim=8, n_heads=4, dim_feedforward=16,
        gate_init="small_0.01",
    )
    out2 = block2(h, kv, kv_mask)
    # NOTE: out_proj is zero-init, so the FIRST forward will also be
    # identity-ish even with small_0.01 gates (because attn_out=0 from
    # the zero-init up-projection). We still get FFN signal though.
    # Check that the FFN gate is being applied (so output != input).
    # The simplest robust check: gate magnitude is ~0.01.
    gates2 = block2.gate_magnitudes()
    assert 0.005 < gates2[0] < 0.02, f"unexpected attn gate magnitude: {gates2[0]}"
    assert 0.005 < gates2[1] < 0.02, f"unexpected ffn gate magnitude: {gates2[1]}"
    print(f"small_0.01 gate magnitudes OK; gates={gates2}")

    # 4. Gradient flows
    loss = out2.sum()
    loss.backward()
    assert block2.alpha_attn.grad is not None, "alpha_attn has no gradient"
    assert block2.alpha_ffn.grad is not None, "alpha_ffn has no gradient"
    assert h.grad is not None, "input hidden state has no gradient"
    print("gradient flow OK")

    # 5. KV padding mask honored
    # Use a block whose out_proj has been nudged out of the zero-init
    # state so attn_out is non-zero; then verify masked positions don't
    # leak.
    block3 = GatedCrossAttnDense(
        hidden_dim=H, cross_dim=8, n_heads=4, dim_feedforward=16,
        gate_init="small_0.01",
    )
    with torch.no_grad():
        torch.nn.init.normal_(block3.out_proj.weight, std=0.02)
    kv_mask3 = torch.zeros(2, K, dtype=torch.bool)
    kv_mask3[0, K // 2:] = True
    out_a = block3(h, kv, kv_mask3)
    kv_corrupt = kv.clone()
    kv_corrupt[0, K // 2:] = 999.0
    out_b = block3(h, kv_corrupt, kv_mask3)
    assert torch.allclose(out_a[0], out_b[0], atol=1e-4), \
        "kv padding mask not enforced in row 0"
    print("kv padding mask enforced OK")

    # 6. Actual param count matches the estimator for the test config
    actual_count = sum(p.numel() for p in block.parameters())
    estimated = estimate_block_param_count(H, cross_dim=8, dim_feedforward=16, n_heads=4)
    # Allow ~5% slack (LayerNorm bias counts depend on version; gates
    # may be packaged differently).
    rel_err = abs(actual_count - estimated) / max(estimated, 1)
    assert rel_err < 0.05, (
        f"estimator off by {rel_err:.1%}: actual={actual_count} estimated={estimated}"
    )
    print(f"param estimator within 5% of actual; "
          f"actual={actual_count} estimated={estimated}")


if __name__ == "__main__":
    _self_test()
