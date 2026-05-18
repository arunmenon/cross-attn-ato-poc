# exp_xa_round1_006 notes

Config: late_only / slots=128 / gate=small_0.01 / small_transformer

Trained cleanly (final train loss 1.368, wall-clock 47.0 min). Gates opened
to a maximum magnitude of 0.0109 at step 1500 — same order as every other
round-1 run, no surprises. Worst-family HN-FPR-stripped 0.0604 (hn_account_recovery)
with CI [0.0472, 0.0709]; mean 0.0255 with CI [0.0235, 0.0277]. AUC-stripped
saturated at 1.000 (sanity-only).

Comparisons (overlapping CIs = statistically tied):
- vs round1_003 (late_only / slots=64): worst 0.0604 vs 0.0586, CIs
  [0.0472,0.0709] vs [0.0460,0.0683] — tied. slots=128 did NOT help on the
  smallest-x-attn-budget pattern, where it had the most charitable shot.
- vs round1_002 (every_8 / slots=64, current best): 0.0604 vs 0.0524, CIs
  overlap [0.0472,0.0709] vs [0.0420,0.0647] — tied. Not a new leader.
- vs round1_004 (every_4 / slots=128): 0.0604 vs 0.0608 — tied (also slots=128).

Per-family pattern matches every other x-attn run: hn_account_recovery is the
worst family (~0.05-0.07), hn_large_purchase mid (~0.01-0.03), hn_travel
zeroed. This shape held across all six x-attn variants — the failure mode
is the same family regardless of insertion pattern or resampler size.

The slots dial is effectively dead on this 5k-eval surface: across all three
direct slots=64 vs slots=128 paired comparisons (every_4: 0.0572 vs 0.0608,
every_8: only slots=64 valid since round1_005 failed, late_only: 0.0586 vs
0.0604), the larger resampler is neutral-to-mildly-harmful with overlapping
CIs. Most likely: clean-eval surface is too easy / AUC-saturated to discriminate
between 64- and 128-slot capacity. Defer slots to Day-4 medium-eval if it
ever matters.

Halt: launcher tripped convergence after this run (6 valid x-attn runs;
last 4 worst-family HN-FPR moved 0 vs the >=0.005 threshold). No more
experiments. Round-1 winner = round1_002 (every_8 / slots=64) at 0.0524.

Next: write Day-2 README section. No Round-2 perturbation will run today
unless the user explicitly extends the budget.
