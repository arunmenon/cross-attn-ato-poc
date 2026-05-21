# exp_v5_p1_slots32 notes

Config: every_8 / slots=32 / gate=small_0.01 / small_transformer

Trained cleanly (final loss 0.485, 40.2 min wall-clock, 113M trainable).
V5 adversarial error 0.1656 CI [0.141, 0.202] — the worst of the slots sweep,
ranking below both slots=64 (0.1596 seed baseline) and slots=128 (0.1526).
Components: phish_takeover recall 1.000, phish_takeover_mfa_phished recall 0.954
(miss=0.046 — noticeably worse than the 0.014 leaders), hn_recovery_high_amount
FPR 0.451 CI [0.369, 0.561] — back to the baseline plateau, not the 0.438 seen
in the better configs.

This result completes the slots sweep {32, 64, 128}: 64 and 128 both reach the
0.438 FPR floor on hn_recovery_high_amount (when paired with good gate/routing),
while 32 slots is insufficient to carry enough event information — it degrades
both the adversarial family recall and the HN FPR. The resampler is not
over-parameterized at 64; it needs at least that capacity for this task.

Gate: max_gate_magnitude=0.018, consistent with other configs. All gates remain
well below the 0.05 "open" threshold across the sweep.

Slots sweep conclusion: 64 is the lower bound for useful event compression here.
Going lower is detrimental; going higher (128) offers no improvement vs the best
gate/routing configs at 64.

Next: exp_v5_p1_fastlr — best-of-first-five base (exp_v5_p1_zero_64:
every_8/gate=zero/slots=64) with lr=3e-4 and warmup_steps=100.
