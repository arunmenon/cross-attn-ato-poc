# exp_v5_p1_fastlr notes

Config: every_8 / slots=64 / gate=zero / small_transformer / lr=3e-4 / warmup=100

Training failure (practical, not mechanical). Status=ok (no NaN crash), but
model completely degraded: v5_adv_error=0.752 CI [0.721, 0.783] — 5x worse than
the 0.151 baseline. AUC collapsed to 0.972 (vs 0.9997 for all prior runs),
r@1%FPR=0.753 (vs 0.997). Final loss 0.532, meaningfully higher than the
0.485–0.500 cluster of successful runs. Gate magnitude=0.006 — the gates barely
opened, suggesting the LM weights were destabilized before the cross-attention
pathway could learn.

Diagnosis: lr=3e-4 with only 100 warmup steps is too aggressive for this
architecture. The warmup is ~7% of training steps (100/1500) vs the usual 33%
(500/1500), so the optimizer ramps to 3x the standard LR before the model has
adjusted. Catastrophic forgetting of the fine-tuned ATO decision surface is the
likely mechanism.

Note: hn_recovery_high_amount FPR=0.425 is actually the best observed across
all runs (point, within wide CI [0.342, 0.528]) — but the phish recall at 0.753
recall makes the run unusable. The FPR observation is noise given the collapsed
fraud detection.

Not counted toward NaN cascade (status=ok), but effectively a failed config.

Next: exp_v5_p1_slowlr — same base (every_8/gate=zero/slots=64) with lr=3e-5
and warmup_steps=500, testing the lower end of the LR range.
