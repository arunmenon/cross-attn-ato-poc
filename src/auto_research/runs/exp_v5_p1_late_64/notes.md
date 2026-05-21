# exp_v5_p1_late_64 notes

Config: late_only / slots=64 / gate=small_0.01 / small_transformer

Trained cleanly (final loss 0.483, 38.8 min wall-clock, 147M trainable — fewer
than every_4's 214M due to only 4 insertion points). V5 adversarial error 0.155
CI [0.128, 0.193], between every_4 (0.151) and every_8 (0.160). Components:
phish_takeover recall 1.000, phish_takeover_mfa_phished recall 0.986
(miss=0.014, same improvement as every_4), hn_recovery_high_amount FPR 0.451
CI [0.369, 0.562] — exactly back to the every_8 baseline value.

Gates: max_gate_magnitude=0.018, consistent with every_4 (0.017) and every_8
(0.022). Late layers did not produce larger or more informative gates.

Key finding: hn_recovery_high_amount FPR is locked at ~0.44-0.45 across all
three insertion patterns (every_4=0.438, late_only=0.451, every_8=0.451). This
family's difficulty is insensitive to where in the network cross-attn is
inserted, suggesting the issue is elsewhere — possibly resampler capacity,
gate_init, learning rate, or an intrinsic ambiguity in the event stream for
this family. phish_takeover_mfa_phished recall uniformly improved (from 0.972
to 0.986) for both non-baseline insertion patterns — modest but consistent.

Rankings so far: every_4 > late_only > every_8 by point v5_adv_error, but
all CIs overlap substantially. Not a declared win.

Next: exp_v5_p1_zero_64 (every_8 / slots=64 / gate_init=zero) to probe
initialization sensitivity with the baseline routing. If zero-init closes the
hn_recovery gap, it would suggest small_0.01 creates a training-dynamics
asymmetry at this family.
