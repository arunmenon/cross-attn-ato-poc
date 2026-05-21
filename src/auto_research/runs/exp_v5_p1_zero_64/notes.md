# exp_v5_p1_zero_64 notes

Config: every_8 / slots=64 / gate=zero / small_transformer

Trained cleanly (final loss 0.495, 39.4 min wall-clock, 113M trainable — the
smallest model so far, as every_8 inserts only 3 x-attn layers). V5 adversarial
error 0.15059 CI [0.124, 0.187], exactly tied with exp_v5_p1_every4_64 on
point value. Tiebreak on hn_fpr_mean_stripped: 0.1094 vs 0.1099 — zero_64
wins and becomes current_best. Components: phish_takeover recall 1.000,
phish_takeover_mfa_phished recall 0.986 (miss=0.014), hn_recovery_high_amount
FPR 0.438 CI [0.360, 0.545].

Gate story: max_gate_magnitude=0.013, the lowest observed across all runs
including the small_0.01 init runs (0.017-0.022). Zero-init gates opened from
exactly 0 to ~0.013 by step 1500 — the model found the cross-attn pathway
useful but opened gates more conservatively than when primed with small_0.01.
Interestingly, gate size did not predict quality: zero_64 matches every_4 in
v5_adv_error despite smaller gates.

Key finding: gate_init=zero achieves the same adversarial-family quality as
the every_4 (6-layer) configuration using only 3 x-attn layers (113M vs 214M
trainable params). This is a parameter-efficiency win. The hn_recovery_high_
amount family at ~44% FPR remains the hard limit — gate initialization does
not resolve it, consistent with the pattern across all insertion variants.

The phish_takeover_mfa_phished improvement (miss=0.014 vs baseline 0.028)
now appears across every_4, late_only, and zero_64 — it is a robust signal,
not a quirk of one configuration.

Next: exp_v5_p1_slots128 (every_8 / small_0.01 / slots=128) — tests whether
increasing the resampler from 64 to 128 slots reduces hn_recovery_high_amount
FPR, which has been the immovable variable across all routing/gate experiments.
