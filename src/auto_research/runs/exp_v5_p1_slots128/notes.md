# exp_v5_p1_slots128 notes

Config: every_8 / slots=128 / gate=small_0.01 / small_transformer

Trained cleanly (final loss 0.500, 40.1 min wall-clock, 113M trainable — same
as every_8/zero_64 since the trainable count is dominated by LoRA on the LM,
not the resampler). V5 adversarial error 0.1526 CI [0.128, 0.189], 3rd place.
Components: phish_takeover recall 1.000, phish_takeover_mfa_phished recall
0.980 (miss=0.020, slightly worse than the 0.014 leaders but within CI),
hn_recovery_high_amount FPR 0.438 CI [0.362, 0.552].

Important: hn_recovery_high_amount FPR = 0.438 — this matches the 0.438 seen
in zero_64 and every4_64, breaking below the 0.451 plateau of the baseline and
late_64. So slots=128 does improve this family vs the 64-slot baseline (0.451),
but the improvement is within CI [0.362, 0.552] and the same point value was
achieved by both zero_64 and every4_64 with only 64 slots. Doubling resampler
capacity does not clearly help hn_recovery_high_amount beyond what the better
gate/routing configs already achieve.

Gate: max_gate_magnitude=0.021, the largest observed — more slots giving the
gates more to summarize. Still well below the 0.05 "open" threshold.

Summary: slots=128 is not better than slots=64 with the right gate/routing
config. The 0.438 FPR floor appears to be a property of these 4 configurations
together (every_4, zero, 128 slots) and is not driven by any single dial.

Next: exp_v5_p1_slots32 (every_8 / small_0.01 / slots=32) — completing the
slots sweep. If slots=32 drops below 0.438, that would be surprising and
informative; if it returns to 0.451, it confirms the baseline slots=64 was
already near-optimal for the FPR floor.
