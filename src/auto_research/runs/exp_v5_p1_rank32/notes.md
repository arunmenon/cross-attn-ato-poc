# exp_v5_p1_rank32 notes

Config: every_8 / slots=64 / gate=zero / small_transformer / lora_r_on_q=32

Trained cleanly (final loss 0.512, 36.3 min wall-clock, 118M trainable — 4.7M
more than rank16 configs, as expected from doubling LoRA rank). V5 adversarial
error 0.1617 CI [0.135, 0.198] — worse than the Phase 1 leaders at rank16
(0.1506). Components: phish_takeover recall 1.000, phish_takeover_mfa_phished
recall 0.965 (miss=0.035, worse than the 0.014 at the best rank16 configs),
hn_recovery_high_amount FPR 0.451 CI [0.369, 0.562] — back to the baseline
plateau, not the 0.438 seen in the top-3.

Gate: max_gate_magnitude=0.009 — the smallest of all clean-training runs.
The doubled LoRA rank may be creating optimization competition between the LoRA
pathway and the gate pathway, causing the gates to open less. Or the higher
parameter count simply needs more training steps.

Rank sweep conclusion: lora_r_on_q=16 is better than 32 in 1500 steps for this
architecture. The additional capacity from rank=32 does not help; it may
actually hurt by slowing gate learning. If rank is revisited, it should be
paired with more steps.

PHASE 1 COMPLETE. Phase 1 winner: exp_v5_p1_zero_64 (every_8 / gate=zero /
slots=64 / small_transformer / lora_r_on_q=16 / lr=1e-4 / v5_adv_error=0.1506).
Tied with every4_64 on v5_adv_error; wins the hn_fpr_mean tiebreak (0.1094 vs
0.1099).

Next: Phase 2 encoder sweep — exp_v5_p2_pooled_mlp and exp_v5_p2_ft_transformer,
pending encoder registry check in train_xattn.py.
