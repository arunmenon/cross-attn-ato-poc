# exp_v5_p2_pooled_mlp notes

Config: every_8 / slots=64 / gate=zero / pooled_mlp

Trained cleanly in ~37.9 min. V5 adversarial error 0.1549 (CI [0.128, 0.194]),
driven by phish_takeover recall 1.0, phish_takeover_mfa_phished recall 0.9859
(miss=0.0141), and hn_recovery_high_amount FPR 0.4505. This is slightly worse
than the Phase 1 winner exp_v5_p1_zero_64 (v5_adv_error=0.1506, FPR=0.4377) but
the CIs fully overlap — no meaningful separation. pooled_mlp does not clear the
0.005 absolute improvement bar needed for Phase 2 continuation. The hn_recovery
FPR actually edged up (0.4505 vs 0.4377 for winner), confirming that collapsing
attention-over-events to mean/max pooling + MLP does not help this hard-negative
family. Gate magnitude 0.0146 is consistent with the rest of Phase 1 — gates are
learning at roughly the same sparse level regardless of encoder architecture.
AUC-stripped 0.9997 is saturated and sanity-only.

Next: run exp_v5_p2_ft_transformer (Phase 2, item 2). If ft_transformer also
fails to beat Phase 1 winner by 0.005, the Phase 2 stop rule fires and we write
final synthesis.
