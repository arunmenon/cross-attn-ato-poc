# Integration-Friction Catalog

**Engineering lessons from v3 that made v4 and v5 credible.**

**v2 · 2026-05-22** (reframed; supersedes v1 dated 2026-05-18)

Ten things that would have eaten a week each, in roughly the order we hit them — every one of them surfaced during the v3 sweep. They are the reason the v4 pivot and the v5 robustness sweep produced credible results rather than another round of false signal. Without items 5 (narrator leakage), 6 (AUC saturation), and 7 (sklearn cliff) being caught in v3, the v4 architectural win would have been computed on a leaky surface and dismissed as overfitting; without items 1 (Blackwell), 3 (paged-8bit), and 8 (convergence halt), the v5 11-run sweep would have either taken twice as long or stopped before producing the dial-robustness finding. Each item is a place an engineer would have burned real time, with the root cause and the fix that landed in the repo. The pattern across all ten: none of them is a research result, but each one would have invalidated a research result if it had gone unnoticed. Catalog them once, generalize them, save the next POC from rediscovering them.

---

## 1. Blackwell architecture + bitsandbytes silently dropping to non-paged optimizer

**Symptom.** Default RunPod H100 image runs `bitsandbytes 0.43`. The trainer launches without error. Throughput is ~2× worse than expected. No error message in the log.

**Root cause.** Hopper-arch H100s require bnb 0.45+ for paged-8bit optimizers (`paged_adamw_8bit`) to actually use the paged path. On bnb 0.43 the optimizer silently falls back to a non-paged FP32 path, which works but is ~2× slower and uses ~3× the optimizer VRAM. No log line announces the fallback; the only signal is throughput.

**Fix.** Pin `bitsandbytes >= 0.45` + CUDA 12.4 in the pod image. Add a hard-fail in `scripts/preflight_xattn.py` that checks `bnb.__version__` and exits non-zero before `accelerate launch` ever loads weights. Documented in `RUNBOOK.md` §Blackwell.

**Lesson.** Treat throughput as a metric, not a property. If a run is unexpectedly slow, suspect environment before code. And: any preflight check that catches "the run will succeed but produce wrong/slow results" is worth its weight twice over.

---

## 2. Stage-0 LoRA must be merged into base before Stage-1 cross-attention training

**Symptom.** Early x-attn training showed gate-magnitude drift that looked like a learning-rate bug. Reducing LR helped, but the drift wasn't fully explained.

**Root cause.** The trainer was loading the Stage-0 CPT-light output as a live PEFT adapter on top of the frozen base, then attaching the Stage-1 cross-attention + fresh LoRA-on-Q. Both adapters were still in their respective gradient graphs at different layers, so the Stage-0 adapter was *itself* updating during Stage-1 training. The cross-attention gates were training against a moving target — the base + the changing Stage-0 weights — which produced the magnitude drift.

**Fix.** Mandatory `scripts/merge_stage0_lora.py` run after Stage-0 completes. The script merges Stage-0 LoRA weights into the base and emits `qwen3-8b-cpt-light-merged` — a single set of weights, no adapter attached. Stage-1 starts from that merged checkpoint. The merge precondition is hard-coded in trainer defaults and documented in PLAN.md §Stage 1.

**Lesson.** Adapter composition is not commutative. If two adapters are both in the gradient graph, you cannot reason cleanly about which one is moving. Bake the merge into the architecture's preconditions, not into the human's checklist.

---

## 3. paged_adamw_8bit / Accelerate / DataParallel silently degrading to FP32

**Symptom.** A wider-than-expected VRAM footprint on a config that should fit. No error.

**Root cause.** `accelerate launch` with multi-process settings (even if the user only wants single-process) initializes a DataParallel wrapper that the `paged_adamw_8bit` optimizer doesn't recognize. The optimizer silently falls back to FP32 optimizer states for the unrecognized parameter groups. The fallback is benign in that training doesn't crash; it is malignant in that the VRAM math no longer holds and the next slightly-larger config OOMs unexpectedly.

**Fix.** `scripts/preflight_xattn.py` now hard-asserts single-process + paged-8bit + no DataParallel wrapping before the trainer touches weights. `src/train/accelerate_configs/single_h100.yaml` is the only blessed Accelerate config; any deviation fails preflight.

**Lesson.** "Silent fallback" is the worst failure mode in ML infrastructure. Optimizer libraries should error on unrecognized parameter groups, not fall back; until they do, preflight is the only line of defense.

---

## 4. Synthetic-narrator throughput — the binding constraint on dataset regeneration

**Symptom.** Initial narrator runs (OpenAI gpt-5.4-nano, serial calls) couldn't saturate the $200 narrator budget in any reasonable wall-clock. Day-1 narrative generation projected to overrun Day-1's data budget.

**Root cause.** Two provider-specific quirks. (1) The gpt-5.4 family requires `max_completion_tokens`, not `max_tokens` — the legacy parameter is silently ignored and the response is truncated at the model's default. (2) Serial calls leave ~95% of the rate-limit budget on the table; the narrator's wall-clock is dominated by API latency, not by total token throughput.

**Fix.** `ThreadPoolExecutor` concurrency landed in commit `a39cc5f`, with `max_completion_tokens` set in `data/gen/narrative_generator.py`. The narrator now saturates the rate-limit budget and produces 25k narratives in ~63 minutes for ~$5.67.

**Lesson.** When the bottleneck is API throughput, the fix is concurrency, not bigger machines. Build the concurrency-aware narrator path before the first dataset regeneration, not after — the time to discover this is during scaffolding, not during scale-up.

---

## 5. Narrative leakage through narrator caching by structured-events-hash

**Symptom.** Day-2 baseline scrutiny revealed 10.7% of eval rows shared narrative text with train rows. Concentrations were family-specific: `hn_large_purchase` 35.3%, `hn_account_recovery` 16.4%, `clean` 13.2%.

**Root cause.** The narrator (`data/gen/narrative_generator.py::_journey_cache_key`) caches generated narratives by `(structured_events_hash, model, temperature)` for cost-efficiency — the same structured events should produce the same narrative deterministically. But the cache lookup runs **before** the train/eval split, so two distinct journeys with the same structured-events footprint (which is common in the synthetic generator's bucket-combination space) share a single cached narrative across the split. The eval row then matches a train row word-for-word.

**Fix.** Two-part. (1) `eval/leakage_checks.py::compute_clean_eval_mask` drops the 534 leaked rows from the eval surface; the launcher applies the mask automatically for every new run via `scripts/run_next_experiment.py::run_post_processing`. (2) `data/gen/build_dataset.py` now stratifies on the pre-narration `structured_events_hash` so identical-skeleton rows go to the same side of the split, and adds a post-narration `text_hash` dedup invariant so future regenerations cannot reintroduce the leak.

**Lesson.** Caching is a leakage vector on synthetic data when the cache key is derived from anything stored in the example. The fix is to split first, cache second. (And: never accept synthetic-data results without running a text-hash + structured-hash overlap audit. The audit is fast — `scripts/diagnose_data_overlap.py` runs in seconds.)

---

## 6. AUC saturation across all variants — metric had to pivot mid-POC

**Symptom.** Stage-0 CPT-light hit AUC=1.0 on every eval mode (stripped, opaque, full) on every eval size (5k, 15k, 50k templated). PLAN.md §Risks had flagged this as possible; Day-1 confirmed it.

**Root cause.** The synthetic data generator's bucket-combination space is small enough that the label is deterministic in observed support — `H(label | bucket-event skeleton) = 0` over 2,454 skeletons, zero mixed-label skeletons. Any sufficiently expressive model can memorize the mapping. AUC, which integrates over the score distribution, saturates at 1.0 because the model can rank every positive above every negative on the observed support.

**Fix.** Headline metric pivoted to **worst-family hard-negative FPR at 1% legit FPR** — a metric that forces the model to discriminate within the three structurally-confused hard-negative families. The pivot was in PLAN.md before Day-1; we just had to actually pull the trigger when AUC hit 1.0 the first time.

**Lesson.** Saturation is a property of the data and the metric, not the model. If you can predict the saturation in the planning stage (we did), the fix is "pivot the metric, not the data, until you have evidence the data is the bottleneck." Pre-commit to the fallback metric before you start training.

---

## 7. sklearn `recall_at_fpr` cliff under tied score distributions

**Symptom.** Day-2 first-cut leaderboard had `event_only` apparently crushing the LM baselines at worst HN-FPR = 0.820% — a 5-7× advantage. The Codex review pass flagged it as implausible before Task #37 dispatched.

**Root cause.** sklearn's `_binary_clf_curve` returns thresholds at the boundaries between unique score values. For models with bimodal score distributions and large tied masses (like `event_only`, which converged to train loss 1e-5), the "largest achievable FPR ≤ target" rule lands different models at materially different achieved legit-FPRs. `event_only` was being measured at achieved FPR = 0.114% (a tenth of the 1% budget the LM baselines were given). The reported numbers were not at the same operating point.

**Fix.** New `recall_at_fpr` in `eval/score_risk.py` that walks descending scores until the cumulative legit count hits exactly `target_fpr * n_legit` (kept as a float, not rounded), computes an `alpha` fraction of tied-at-threshold rows to weight, and reports `(threshold, alpha, achieved_fpr, n_above, n_tied, tie_fraction)` so the operating point is verifiable from JSON. `eval/bootstrap_ci.py` recomputes `(threshold, alpha)` per resample. All v1 rows were rescored under `metric_version: 2` (idempotent via `scripts/rescore_baselines.py`); `update_sweep_state` filters ranking to `metric_version >= 2`. Cost: most of Day-2's second half.

**Lesson.** Off-the-shelf metric implementations make assumptions about score distributions (mostly that they're continuous). Models that don't satisfy those assumptions will land at different operating points without saying so. A tie-aware, exact-target metric is the right default for any "operating-point-controlled" metric (R@FPR, P@R, etc.), not just for low-FPR regimes.

---

## 8. Convergence halt firing prematurely

**Symptom.** Day-2's convergence halt (`no AUC-stripped improvement of ≥0.005 over last 4 valid runs AND ≥6 valid runs completed`) fired after Round-1, before any Round-2 perturbation could probe gate-init sensitivity. The auto-loop stopped one experiment short of the question Round-2 was designed to answer.

**Root cause.** The halt windowed on the last 4 valid runs. The Round-1 leader (`exp_xa_round1_002`) sat in slot 1 of that window. By construction, any later sibling in the window had to beat the leader by ≥0.005 to keep the loop running — which made the halt impossible to *not* fire as soon as the window was full. The halt was firing on the metric's halt logic, not on actual convergence of the research question.

**Fix.** Halt logic is now configurable in `budget.yaml` with per-halt enable flags. v3 Day-3 disabled `convergence: enabled: false` and left `nan_cascade` + `zero_gate_activation` + budget caps as the real stops. Zero-gate fired cleanly on the second v3 Round-2 run and produced the gate-bias finding. v4 and v5 inherited the same halt configuration without modification, and v5 added an `early-exit-on-success` rule on top (pivot to local perturbations if any single run records `max_gate ≥ 0.05` AND beats current best with non-overlapping CIs).

**Lesson.** Halt conditions need to be designed against the question being asked, not against generic optimization heuristics. A "no improvement" halt that ignores the *what* of the next experiment will always over-fire. The fix is "halt on things that mean we are out of useful work to do," not on rolling-window deltas.

---

## 9. `max_gate_magnitude` halt threshold tuning

**Symptom.** Original `zero_gate_activation` halt threshold was `magnitude < 0.05`. With `gate_init=small_0.01` initializing at exactly 0.01 and 1500 steps lifting only to ~0.011, every run was tripping the halt threshold despite gates being technically "open" relative to their init.

**Root cause.** The 0.05 threshold was set in PLAN.md as the "gates are doing real work" bar, taken from the Flamingo paper's reported magnitudes. But the Flamingo magnitudes were on a different scale (image-text task, much higher gate magnitudes) and were not directly applicable to the structured-stream-on-LM regime. The threshold was wrong by an order of magnitude for this task.

**Fix.** Threshold lowered to `0.005` in `budget.yaml` after `exp_xa_smoke_001` and `exp_xa_round1_001` both landed in the 0.010-0.011 band. The lowered threshold still catches a true zero-collapse (Round-2 zero-init landed at 0.0038-0.0041, correctly tripping the halt) without false-firing on Round-1 small-init runs.

**Lesson.** Halt thresholds taken from papers on different tasks are starting points, not specs. Calibrate them against the actual smoke-run magnitudes before letting the auto-loop enforce them. (And: don't let a halt threshold suppress evidence — the 0.005 floor is now the documented "below this is structurally dead" mark, and Round-2 hitting it became the gate-bias finding.)

---

## 10. Launcher / agent ownership split

**Symptom.** Early versions of the auto-loop had the agent appending to `experiments.jsonl` directly. Within two days, the agent rows and launcher rows had different field orders, different float formatting, and one case where the agent emitted `null` for a numeric field that should have been omitted entirely. Tooling that read the history (`update_sweep_state`, the rescore script) had to handle both formats with defensive coding.

**Root cause.** Two writers, one file, no schema enforcement. Worse: the agent's natural mode is to write Markdown-like prose and slip occasional JSON, so the agent's JSON drift was idiosyncratic rather than uniform.

**Fix.** Clean ownership split documented in `AGENT_INSTRUCTIONS.md`: launcher owns `experiments.jsonl`, `sweep_state.yaml`, and all per-run JSON artifacts (`metrics.json`, `ci_report.json`, `gate_trajectory.json`, `leakage_report.json`). Agent owns `config.yaml` (one per planned run) and `notes.md` (one per completed run) and the README's Day-2 / Day-3 / Final sections. Source code edits require `git add -A && git commit -m "snapshot before <change>"` first.

**Lesson.** When two agents (literal or figurative) can write to the same file, eventually they will write incompatibly. The fix is single-writer-per-file, not "both writers should be careful." This generalizes well beyond auto-research: any pipeline with an LLM-in-the-loop should have an enforced ownership map and a deterministic gate.

---

## Cross-cutting patterns

Reading the catalog as a whole, three patterns emerge:

**Silent failures are the dangerous ones.** Items 1, 3, and 6 are all "the run succeeded but the result was wrong." None of them produced an error message. The fix in each case was a preflight check or a metric audit that *would have* errored. **Engineering effort should bias toward preflight detection over post-hoc debugging.** A preflight check that catches one silent failure is worth ten retries on a noisy alert.

**Caching is a leakage vector on synthetic data.** Items 5 (narrator cache) and adjacent (the bucket-event skeleton overlap in `docs/day-2-data-diagnostic.md`) both come from the same family: when the cache key is derived from anything stored in the example, the cache becomes a leakage channel between train and eval. **The structural fix is "split first, cache second"** — always. Any data pipeline that caches by a content-derived key needs an overlap audit before the data ships.

**Metrics need to be operating-point-exact, not operating-point-approximate.** Item 7 (sklearn cliff) is the worst case, but item 6 (AUC saturation) is the same shape: the metric is reporting a property of the score distribution, not a property of the model. Operating-point-controlled metrics (R@FPR, worst-family HN-FPR) must be tie-aware and exact-target by construction. Off-the-shelf implementations rarely are. **Budget the time to write the metric correctly the first time** — the cost of fixing it mid-POC is half a day plus the loss of confidence in everything reported before the fix.

---

## How to use this catalog

For the **AI Leadership team**: this is the realism check on what 3-day-POC velocity costs in integration friction. The headline cross-attention finding (no detectable classification lift) is one piece of evidence; the ten items above are ten more, each independently reusable. **Treat the catalog as a Foundation Science capability artifact** — every future POC that runs on the auto-research loop inherits the fixes for items 1, 3, 5, 7, 8, 9, and 10 for free, because they are now in the shared scaffolding.

For the **next engineer running a POC**: read items 1-3 before booting a pod, items 4-5 before generating data, items 6-7 before reporting any baseline numbers, items 8-9 before turning on auto-loop halt conditions, and item 10 before writing the loop's owner map. Roughly two hours of reading prevents a week of lost work.
