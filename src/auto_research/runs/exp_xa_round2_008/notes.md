# exp_xa_round2_008 notes

Config: every_4 / slots=64 / gate=zero / small_transformer (1500 steps).
Direct sibling: round1_001 (same arch, gate_init=small_0.01).

Trained cleanly (final train loss 1.414, wall-clock 47.6 min, leakage_clean).
Max-gate magnitude at step 1500: **0.00412** — second consecutive xattn
run below the 0.005 "gate open" threshold. **Launcher tripped the
zero_gate_activation halt: `halted: true`, halt_reason `last 2 x-attn
runs had max gate < 0.005`.** No more experiments will run unless the
halt is overridden.

Worst-family HN-FPR-stripped 0.0594 (hn_account_recovery) CI [0.0470, 0.0708];
mean 0.0256 CI [0.0236, 0.0277]. AUC-stripped 1.000 (saturated).

Key comparison (the matched pair this experiment was designed for):
- round2_008 (gate=zero,    max_gate 0.0041) → HN-FPR-worst 0.0594 [0.0470, 0.0708]
- round1_001 (gate=0.01,    max_gate ~0.011)  → HN-FPR-worst 0.0572 [matches Round-1]

Statistically tied (CIs overlap heavily). Same finding as round2_007 →
round1_002, now on the every_4 architecture: gates barely opening
(0.0041 vs 0.011, ~2.6× smaller) produces a statistically indistinguishable
HN-FPR. The base CPT-light-merged LM is doing essentially all of the
discrimination; cross-attn provides at most marginal lift that we cannot
detect at this eval scale (n=4466 after clean-eval).

This rules out the alternative hypothesis from the round2_007 notes
(that every_8 was structurally hostile to zero-init learning). every_4
has 2× the x-attn layers (6 vs 3), so 2× the parameters getting gradient
signal, yet gates *still* did not open from zero in 1500 steps. The
small_0.01 starting bias in Round-1 was carrying the gate-magnitude
story: gates aren't learning to open under pure-learning-pressure on
this dataset in this regime; they ride whatever bias they start with.

This is the answer to one of the Day-3 questions, arrived at on Day 2:
the Round-1 "gates opened to ~0.011" result is an init-bias artifact,
not learned cross-attn signal. Combined with the HN-FPR being unchanged
between max_gate 0.011 and max_gate 0.004 runs, cross-attn-via-gated-residual
is providing no detectable classification lift on this task at this
training scale. The structured-stream story is going to have to come
from the structured_as_text baseline comparison (5k fast eval) and a
Day-3 medium-eval verification, not from x-attn lift.

Halt status: launcher reports `halted: true`. Stopping. Writing Day-2
README section next.
