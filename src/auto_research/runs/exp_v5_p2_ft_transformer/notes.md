# exp_v5_p2_ft_transformer notes

Config: every_8 / slots=64 / gate=zero / ft_transformer

Trained cleanly in ~41 min. V5 adversarial error 0.1736 (CI [0.1448, 0.2136]),
driven by phish_takeover recall 1.0, phish_takeover_mfa_phished recall 0.9296
(miss=0.0704), and hn_recovery_high_amount FPR 0.4505. This is the worst Phase
2 result, and notably worse than the Phase 1 winner exp_v5_p1_zero_64
(v5_adv_error=0.1506). The ft_transformer's larger parameter count (113M
trainable vs ~109M for pooled_mlp, ~105M for small_transformer) did not help
and actually hurt mfa_phished recall: 0.9296 vs 0.9859 for the winner and 0.9859
for pooled_mlp. Gate magnitude 0.0157 is consistent with the rest of Phase 1-2
— the encoder architecture does not meaningfully affect gate activation level.
AUC-stripped 0.9996 is saturated and sanity-only.

Phase 2 stop rule fires: neither pooled_mlp (0.1549) nor ft_transformer (0.1737)
beat the Phase 1 winner (0.1506) by ≥0.005 absolute v5_adv_error. Encoder
architecture does not matter for this task in this regime. Writing final
synthesis.
