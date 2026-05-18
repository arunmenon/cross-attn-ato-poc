# exp_xa_round2_007 notes

Config: every_8 / slots=64 / gate=zero / small_transformer (1500 steps).
Direct sibling: round1_002 (same arch, gate_init=small_0.01).

Trained cleanly (final train loss 1.317, wall-clock 46.6 min, leakage_clean).
Max-gate magnitude at step 1500: **0.00385** — below the (lowered) 0.005
"gate open" threshold. Gates barely moved from true-zero init: this counts
as one zero_gate xattn run; one more zero-init run with the same regression
would trip the consecutive_threshold=2 halt.

Worst-family HN-FPR-stripped 0.0608 (hn_account_recovery) CI [0.0475, 0.0716];
mean 0.0254 CI [0.0234, 0.0277]. AUC-stripped 1.000 (saturated, sanity-only).

Key comparison (the one this experiment was designed for):
- round2_007 (gate=zero,   max_gate 0.0038) → HN-FPR-worst 0.0608 [0.0475, 0.0716]
- round1_002 (gate=0.01,   max_gate 0.0112) → HN-FPR-worst 0.0524 [0.0420, 0.0647]

Point estimate is **worse** with zero-init, but CIs overlap heavily
([0.048-0.065]) — formally tied. The interesting part is the *gate magnitude
delta*: 3× larger max-gate (0.0112 vs 0.0038) bought a non-significant
~0.008 absolute on worst-family HN-FPR. That implies the **base CPT-light-merged
LM is doing most of the discrimination**; cross-attn provides marginal-at-best
lift even when gates are at the upper end of what Round-1 produced.

Per-family pattern unchanged from every prior x-attn run: hn_account_recovery
is the worst family (~0.06), hn_large_purchase mid (~0.016), hn_travel zeroed.
The failure mode is invariant to gate init *and* gate magnitude in this
regime.

Halt status: `halted: false`. Convergence halt remains disabled in
budget.yaml; zero_gate cascade is at 1/2 after this run.

Next: propose Round-2 top-2 perturbation, every_4 / slots=64 / gate=zero
(matched sibling to round1_001 at HN-FPR-worst 0.0572). Two reasons:
(1) it completes the gate_init=zero leg on the second-best architecture,
which is the data we need to claim or rule out the init-bias narrative
broadly rather than just on every_8; (2) if it *also* lands max_gate < 0.005,
the zero_gate halt fires legitimately, which itself is the right
conclusion for the writeup — gates don't open meaningfully under
pure-learning-pressure on this task, so cross-attn-via-gated-residual is
not a lift mechanism here.
