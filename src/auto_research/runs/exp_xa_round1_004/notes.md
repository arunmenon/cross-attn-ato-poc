# exp_xa_round1_004 notes

Config: every_4 / slots=128 / gate=small_0.01 / small_transformer (Round-1 cell #4)

Trained in 48 min, final train loss **1.367** — highest of all x-attn runs
so far. 214M trainable params (vs 113M for every_8/64), pushing the gate
of what 1500 steps can converge cleanly. Gates open at
max_gate_magnitude=0.0112 (above floor). Worst-family HN-FPR-stripped
**0.0608** [0.048, 0.072]; mean 0.0254; worst family hn_account_recovery
(consistent). Per-family: hn_account_recovery 0.0608, hn_large_purchase
0.0156, hn_travel 0.000. AUC-stripped 1.000 (saturated).

Direct slots dial isolation vs round1_001 (every_4 / slots=64, the only
other every_4 run):
- slots=64  (round1_001): 0.0572 [0.046, 0.069]
- slots=128 (round1_004): 0.0608 [0.048, 0.072]
- Delta: +0.0036 worse with 128; CIs heavily overlap. Doubling slots
  did NOT help and the trend is mildly worse, consistent with the
  214M-param model being under-converged at 1500 steps.

Combined slots=64 vs slots=128 cross-pattern read so far:
| pattern   | slots | HN-FPR-worst [CI]             | train_loss | n_trainable |
|-----------|-------|--------------------------------|------------|-------------|
| every_4   | 64    | 0.0572 [0.046, 0.069]         | ~1.2       | ~113M       |
| every_8   | 64    | 0.0524 [0.042, 0.065]         | 1.150      | 113M        |
| late_only | 64    | 0.0586 [0.046, 0.068]         | 1.308      | n/a         |
| every_4   | 128   | 0.0608 [0.048, 0.072]         | 1.367      | 214M        |

hn_account_recovery is the bottleneck in every case at ~0.052-0.061.
Mean HN-FPR is within 0.001 across all four runs. The data-shaped
saturation hypothesis is strengthening: regardless of
insertion_pattern or slots, worst-family converges on this band.

Next: round1_005 (every_8 / slots=128). every_8 has the lowest param
count of the insertion patterns (3 x-attn layers), so it should
converge cleanest at slots=128. If every_8/128 still doesn't break
below 0.045 with CI separation, the slots dial is confidently dead
on this surface and Round-2 can focus on gate_init perturbation of
the every_8/64 leader.
