# exp_xa_round1_002 notes

Config: every_8 / slots=64 / gate=small_0.01 / small_transformer (Round-1 cell #2)

Trained cleanly to final loss 1.150 in 46 min wall-clock. Gates opened to
max_gate_magnitude=0.0112 by step 1500 — comparable to round1_001's 0.0106 and
above the 0.005 halt floor. Worst-family HN-FPR-stripped **0.0524**
(CI [0.0420, 0.0647]; worst family: hn_account_recovery) becomes the new
current_best, edging exp_xa_smoke_001_v2 (0.0537) and exp_xa_round1_001
(0.0572). Mean HN-FPR-stripped 0.0262 (tiebreak). AUC-stripped 1.000
(saturated, sanity-only). Per-family: hn_account_recovery 0.052,
hn_large_purchase 0.026, hn_travel 0.000 — same family-shape as both prior
x-attn runs (account_recovery is the binding constraint).

Baseline deltas (worst-family HN-FPR, stripped, clean eval):
- vs CPT-light-merged: bar not directly comparable yet — Stage-0 baseline did
  not record v2 worst-family on the same clean-eval surface; treat as
  qualitative for now.
- vs lora_text_v2 (0.0701, CI [0.056, 0.085]): nominally better by 0.018,
  but CIs touch (lora_lo=0.0564 vs xattn_hi=0.0647). Marginal separation,
  not decisive.
- vs structured_as_text_v2 (0.0507, CI [0.041, 0.063]): **tied within CI**.
  Cross-attn is still not beating the load-bearing concat baseline.
- vs event_only_v2 (0.0730, CI [0.067, 0.080]): better and CIs do NOT
  overlap (event_lo=0.0667 > xattn_hi=0.0647? — event_lo=0.0667 vs
  xattn_hi=0.0647, so CIs barely separated, xattn wins by ~0.02).

Architecture read: every_8 (3 x-attn layers) tied/marginally beat every_4
(6 x-attn layers) at slots=64, suggesting insertion density past 3 layers
gives diminishing returns on this synthetic surface. Gate magnitudes are
similar across both, so the additional layers in every_4 may be redundant
rather than collapsed.

Next: finish Round-1 spread cell #3 (late_only / 64). If late_only lands in
the same 0.05-0.06 band as the other two patterns, the worst-family
bottleneck is data-shaped (hn_account_recovery saturation), not
architectural — which would be the headline finding.
