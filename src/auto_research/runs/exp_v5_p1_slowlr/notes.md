# exp_v5_p1_slowlr notes

Config: every_8 / slots=64 / gate=zero / small_transformer / lr=3e-5 / warmup=500

Trained cleanly (final loss 0.488, 37.4 min wall-clock, 113M trainable), but
underperformed the baseline. V5 adversarial error 0.2015 CI [0.167, 0.244] —
worse than both the 0.151 default-LR baseline and the 0.160 seed
(exp_xattn_v4_001). Components: phish_takeover recall 1.000, but
phish_takeover_mfa_phished recall only 0.846 (miss=0.154 CI [0.064, 0.238] —
significantly worse than the 0.014 leaders), hn_recovery_high_amount FPR 0.451
(back to the baseline plateau). AUC 0.9993, r@1%FPR=0.983 — both slightly
below the 0.9997–0.9999 cluster of the strong configs.

Diagnosis: lr=3e-5 is 3x below the default (1e-4) and is too slow for 1500
training steps. The model has not fully adapted its decision boundary for the
harder phish_takeover_mfa_phished family within the training budget. Unlike the
fastlr failure (catastrophic forgetting at 3e-4), this is a classic
under-training failure.

LR sweep conclusion: 1e-4 is the robust choice for this architecture and step
budget. 3e-4 overshoots catastrophically; 3e-5 undertrains. The sensitivity
is asymmetric — the failure mode from going 3x too high is much worse than
from going 3x too low.

Next: exp_v5_p1_rank32 — best-so-far base (exp_v5_p1_zero_64:
every_8/gate=zero/slots=64) with lora_r_on_q=32 instead of the default 16.
This is the last Phase 1 item.
