# exp_v5_p1_every4_64 notes

Config: every_4 / slots=64 / gate=small_0.01 / small_transformer

Trained cleanly (final loss 0.493, 41.4 min wall-clock). V5 adversarial error
0.151 CI [0.123, 0.189], driven by: phish_takeover recall 1.000 (perfect),
phish_takeover_mfa_phished recall 0.986 (miss rate halved vs baseline: 0.014
vs 0.028), hn_recovery_high_amount FPR 0.438 CI [0.357, 0.552] (essentially
unchanged vs baseline's 0.451 CI [0.369, 0.562]). Worst-family HN-FPR-stripped
0.438 is the secondary risk metric.

Gates remained small throughout: max_gate_magnitude=0.017 at step 1499, below
the 0.05 "gates open" threshold and actually slightly lower than the every_8
baseline (0.022). This is noteworthy: inserting 6 cross-attn layers instead of
3 did not increase per-layer gate magnitude — the model distributed influence
more broadly but each gate stayed small. The CI overlap between this run
([0.123, 0.189]) and exp_xattn_v4_001 ([0.131, 0.194]) means the improvement
is not statistically significant. The new state_best by point value, but not a
declared win.

The phish_takeover_mfa_phished improvement (recall 0.986 vs 0.972) is
encouraging but sits inside CI; the hn_recovery_high_amount family remains
stuck at ~44% FPR across both routing patterns, suggesting this family's
difficulty is not primarily about x-attn insertion frequency.

Next: proceed to exp_v5_p1_late_64 (late_only insertion / slots=64 /
gate=small_0.01) — tests whether concentrating x-attn in the last 4 layers
(where the LM's abstraction is richest) reduces HN-FPR more than uniform
spacing.
