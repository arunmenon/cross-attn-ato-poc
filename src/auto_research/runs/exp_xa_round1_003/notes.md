# exp_xa_round1_003 notes

Config: late_only / slots=64 / gate=small_0.01 / small_transformer (Round-1 cell #3)

Trained cleanly in 46 min wall-clock. Final train loss **1.308** — notably
higher than every_8/64 (1.150) and every_4/64 (round1_001 final loss not
explicit but model converged). Gates open at max_gate_magnitude=0.0109,
above the 0.005 halt floor. Worst-family HN-FPR-stripped **0.0586**
(CI [0.0460, 0.0683]; worst family: hn_account_recovery — same as the
other two x-attn runs at slots=64). Mean HN-FPR-stripped 0.0256.
AUC-stripped 1.000 (saturated). Per-family: hn_account_recovery 0.0586,
hn_large_purchase 0.0184, hn_travel 0.000.

Round-1 slots=64 sweep complete. Side-by-side:

| insertion_pattern | HN-FPR-worst-stripped [CI]            | HN-FPR-mean | train_loss |
|-------------------|----------------------------------------|-------------|------------|
| every_4 (r1_001)  | 0.0572 [0.046, 0.069]                 | 0.0258      | n/a (~1.2) |
| every_8 (r1_002)  | 0.0524 [0.042, 0.065]                 | 0.0262      | 1.150      |
| late_only (r1_003)| 0.0586 [0.046, 0.068]                 | 0.0256      | 1.308      |

All three pattern CIs overlap heavily on worst-family. Mean HN-FPR is
within 0.001 across all three. hn_account_recovery is the bottleneck in
every case at ~0.05-0.06. This is consistent with outcome (b) from the
rationale: the slots=64 insertion_pattern dial is dominated by the
hn_account_recovery noise floor on this clean-eval surface.

Baseline deltas (worst-family HN-FPR, stripped, clean eval) for late_only:
- vs structured_as_text_v2 (0.0507): **tied within CIs**.
- vs lora_text_v2 (0.0701): nominally better by 0.011; CIs touch.
- vs event_only_v2 (0.0730): better by 0.015; CIs slightly separated.

Training-loss read: late_only ends ~0.16 nats higher than every_8 despite
both seeing the same data and gate-init regime. So the architectural
choice does matter for the LM optimization — but the gain doesn't
translate into worst-family HN-FPR lift, because that metric is bottlenecked
upstream by data signal in hn_account_recovery.

Next: continue to Round-1 slots=128 spread (cells #4, #5, #6). If
slots=128 also lands in the 0.05-0.06 worst-family band across all three
patterns, the headline finding solidifies as data-shaped saturation
rather than architectural search winning. After cell #6, n_xattn_runs=7
and convergence-halt becomes eligible.
