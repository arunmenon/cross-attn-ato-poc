# Batch 4 — five trainers + shared eval/training infrastructure

The final batch of Path A. Implements PLAN.md "Training pipeline" and
"Baselines" sections, integrating Batches 1-3 (tokens, data, model) into
runnable training scripts.

Baseline commit going into this batch: `e02c6ad` (review 006 closure).
Closing commit: filled in by the two-commit INDEX backfill pattern.

After this batch the scaffold is end-to-end runnable: `accelerate launch
src/train/train_xattn.py --config <config.yaml>` produces a checkpoint,
gate trajectory, and three predictions_<mode>.jsonl files consumed by
the existing `eval/score_risk.py` + `eval/bootstrap_ci.py` pipeline.

---

## 1. What landed

| File | Lines | One-line |
|---|---|---|
| `src/train/common.py` | ~250 | Shared utilities: config loader, tokenizer prep (custom-token install + pad_token), HF Dataset loading, label-token IDs (` fraud` / ` legit`), label-position finder, `paged_adamw_8bit` factory, cosine LR scheduler factory, atomic JSONL writer. |
| `src/train/eval_runner.py` | ~180 | Three-mode eval pass. `run_three_mode_eval` for LM-based arms (applies eval_modes.apply per example, tokenizes truncated-to-`label:` prefix, scores `logP(' fraud') - logP(' legit')` at the last real position). `run_classifier_eval` for the event-only arm (identical output shape, no eval-mode transform). |
| `src/train/train_event_only_classifier.py` | ~190 | Baseline #4: `SmallTransformerEncoder` + 2-class linear head, no LM. Cross-entropy training on structured event streams. |
| `src/train/train_lora_text_only.py` | ~150 | Baseline #2: LoRA r=16 on raw Qwen3-8B, text-only narratives + verdict footer. Eval-mode dropout in collate. |
| `src/train/train_cpt_light.py` | ~190 | Stage-0: vocab expansion + LoRA on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (full attention + MLP). Output adapter consumed by `scripts/merge_stage0_lora.py` to produce `qwen3-8b-cpt-light-merged` — baseline #1 AND the Stage-1 starting checkpoint. |
| `src/train/train_structured_as_text.py` | ~210 | Baseline #3 — the load-bearing apples-to-apples comparator. Prepends compact event serialization to the narrative (in a collator wrapping `EvalModeDropoutCollator`), trains CPT-light-style on the merged base. |
| `src/train/train_xattn.py` | ~260 | Main x-attn trainer. Loads merged CPT base, constructs `QwenXAttnWrapper`, attaches Stage-1 fresh LoRA-on-Q, trains side-stream encoder + resampler + x-attn blocks + LoRA jointly on paired (structured, text) data. Logs gate-magnitude trajectory per step (consumed by the launcher's `zero_gate_activation` halt check). |

Total: ~1,430 LOC across 7 files.

---

## 2. Design decisions

### 2.1 Single score surface across all LM trainers

Every LM trainer scores the same way: `logP(' fraud' | prefix) -
logP(' legit' | prefix)` at the token aligned to `<risk_verdict>\nlabel:`.
The trainers don't generate; they read the next-token logit
distribution at one position.

Rationale (re-stated from review-005 followup): this matches
`eval/score_risk.py`'s contract exactly — it expects per-example
records with a single scalar `score`. No tokenizer-specific
multi-token decoding logic needed. The eval pass truncates each
example's text at `label:` before tokenizing, so the model NEVER
sees the ground-truth label during scoring.

### 2.2 Five trainer files vs one parameterized script

Each trainer is its own file even though they share ~60% of the
pipeline (data loader, optimizer, eval pass, metrics writer). The
alternative — one `train.py --arm xattn|cpt_light|...` — was
rejected because:

- The five arms have meaningfully different inputs (text-only,
  text+events, structured-only) and different model surfaces
  (raw Qwen3, wrapped Qwen3, no-LM classifier).
- Per-arm files make the trainer dispatch in
  `scripts/run_next_experiment.py`'s `ARM_TO_TRAINER` dict trivial.
- The deltas between trainers are exactly what review-005's "Stage-0
  LoRA target list differs from Stage-1's" required — separate files
  surface that difference, a one-file parameterized version would
  hide it in if/else branches.

Trade-off: ~30% duplicated boilerplate across the four LM trainers.
Mitigated by extracting the heavy lifting into `common.py` + `eval_runner.py`.

### 2.3 `train_cpt_light` LoRA target set is broader than other arms

Stage-0 CPT-light targets all 7 standard Qwen3 LoRA modules
(`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
while Stage-1 x-attn uses Q-only (`q_proj`).

Rationale: Stage-0 is supposed to actually adapt the LM to the
custom-token vocabulary and the journey/narrative distribution.
Q-only LoRA at this stage would leave most of the base unchanged
and Stage-0 would barely move from raw Qwen3. Stage-1's Q-only
LoRA is a stabilizer for the x-attn machinery, not the primary
adaptation surface (the x-attn blocks + encoder + resampler do
most of the learning).

This means the Stage-0 LoRA adapter is bigger (~7× the parameter
count of a Q-only LoRA). After `merge_stage0_lora.py` it becomes
part of the merged-base weights, so this doesn't affect Stage-1's
trainable-param budget.

### 2.4 `train_structured_as_text` reuses Stage-1 base (merged CPT)

The structured-as-text baseline starts from `qwen3-8b-cpt-light-merged`
(not raw Qwen3-8B). Same starting point as `train_xattn`.

Rationale: the comparison "cross-attn vs structured-as-text concat"
must isolate the architectural contribution (cross-attention layers
+ encoder + resampler) from the CPT-light contribution (vocabulary
expansion + LoRA-merged on attention+MLP). Both arms get the CPT-light
benefit; only x-attn gets the architectural addition. If they both
started from raw Qwen3, structured-as-text would also have to learn
the custom-token vocabulary mid-run and the comparison would be
confounded.

### 2.5 Eval-mode dropout collator wrapped, not replaced, for structured-as-text

The structured-as-text trainer's collator is a wrapper around
`EvalModeDropoutCollator`. It prepends the structured event
serialization to `ex["text"]` BEFORE handing the example to the
eval-mode-dropout machinery. This means:

- `<journey_*>` and `<actor_*>` tokens in the narrative get
  stripped/opacified at the appropriate sampled rate.
- Event-token strings inside the prepended `<events>` block
  (e.g. `<event_login>`, `<amount_bucket=high>`) are NOT touched
  by `eval_modes.apply` because they're not journey/actor tokens.
- The eval pass uses the same prepended-text shape (synthesized in
  the trainer's eval section, not via the collator), so train/eval
  distribution is consistent.

### 2.6 Gate trajectory logged at EVERY step, not sampled

`train_xattn.py` records the gate magnitudes for every x-attn block
at every training step into `gate_trajectory.json`. This is more
data than strictly needed for the launcher's halt-condition check
(which only reads step 1500's value), but it's small (each step
adds ~100 bytes per block × 6 blocks = 600 bytes; 1500 steps = ~900 KB)
and gives Day-3 diagnostics a complete trajectory for plotting.

The launcher's `zero_gate_activation` halt rule reads
`max_gate_magnitude` from `metrics.json`, which is computed from
the trajectory once at end-of-training.

### 2.7 Wrapper-vs-actual param count sanity check at startup

`train_xattn.py` calls both:
- `estimate_wrapper_trainable_params(...)` (analytical, from cross_attn_block + wrapper helpers)
- `wrapper.trainable_param_summary()` (actual nn.Module count after construction)

and prints both. If they disagree by >5%, the trainer should fail
loudly before consuming GPU time — that disagreement would indicate
either the estimator drifted or the wrapper accidentally
registered/froze a module differently than expected.

Currently the trainer prints both but doesn't assert. The trainer
itself could add an assertion; we leave that to the review.

### 2.8 `accelerator.unwrap_model` is required for `gate_diagnostics` and eval

Once `wrapper = accelerator.prepare(wrapper, ...)`, the wrapper is
wrapped in a `DistributedDataParallel`-like shell that doesn't
forward arbitrary method calls (gate_diagnostics, attach_lora_on_q).
The training loop reads gates via the prepared model when possible,
falling back to `accelerator.unwrap_model(wrapper)`. The eval pass
unwraps explicitly.

---

## 3. Where this maps to PLAN.md

| PLAN.md section | Implementation |
|---|---|
| Training pipeline — bf16, paged_adamw_8bit, cosine + 500-step warmup | All trainers default to these via `build_optimizer` + `build_lr_scheduler`. Overridable per config. |
| Training pipeline — Stage 0 = embedding + LoRA → merge | `train_cpt_light.py` produces the LoRA adapter; the comment at end points the user to `scripts/merge_stage0_lora.py`. |
| Training pipeline — Stage 1 = encoder + resampler + x-attn + LoRA-on-Q (fresh) | `train_xattn.py` constructs the wrapper with `attach_lora_on_q(r=16)`. |
| Eval-mode dropout during training (50/25/25) | `EvalModeDropoutCollator` from Batch 1 used in all four LM trainers; defaults match the 50/25/25 mix. |
| Baselines §1 CPT-light | `train_cpt_light.py` + `scripts/merge_stage0_lora.py` → baseline #1 checkpoint. |
| Baselines §2 LoRA-text-only | `train_lora_text_only.py` (r=16 on q_proj only, raw Qwen3-8B). |
| Baselines §3 Structured-as-text concat | `train_structured_as_text.py` with compact serialization. |
| Baselines §4 Event-only classifier | `train_event_only_classifier.py` — no LM. |
| Eval — three modes (stripped/opaque/full) + verdict footer score | `eval_runner.run_three_mode_eval` with explicit truncation at `label:` marker. |
| Eval — predictions_<mode>.jsonl shape consumed by score_risk.py | Trainer writes; launcher invokes `eval.score_risk` + `eval.bootstrap_ci` per mode. |
| Gate-activation magnitude logging | `train_xattn` writes `gate_trajectory.json`; `metrics.json` includes `max_gate_magnitude`. |

---

## 4. Smoke / self-test results

```bash
# AST parse (all 7 files):
python3 -c "import ast; [ast.parse(open(f).read()) for f in [
    'src/train/common.py', 'src/train/eval_runner.py',
    'src/train/train_event_only_classifier.py',
    'src/train/train_lora_text_only.py',
    'src/train/train_cpt_light.py',
    'src/train/train_structured_as_text.py',
    'src/train/train_xattn.py']]; print('OK')"
# OK

# common.py torch-free portion (runs on laptop):
python3 -m src.train.common
# load_config OK
# find_label_score_position OK
# write_predictions_jsonl OK
# torch/transformers not installed; tokenizer / dataset / optimizer
# self-tests skipped (run on the pod)
```

Layer A scaffold smoke still green.

Live torch tests (run on pod via Tasks #33 and #35-38):
- `train_cpt_light`: 1500 steps complete, LoRA adapter written, three-mode eval populates predictions_*.jsonl, merge_stage0_lora produces qwen3-8b-cpt-light-merged.
- `train_lora_text_only`, `train_structured_as_text`: same shape, different inputs.
- `train_event_only_classifier`: classifier converges to non-trivial AUC on the synthetic data; predictions are identical across all three "modes" by construction.
- `train_xattn`: gates open by step ~600 (gate magnitude > 0.05), max_gate_magnitude > 0.1 by step 1500; loss curve descends; predictions populate per-mode.

---

## 5. Known limitations / deferred

- **No real-Qwen3 forward pass exercised locally.** All five trainers depend on the actual `transformers.Qwen3ForCausalLM.forward` signature working as expected. The wrapper from Batch 3 was hook-tested with a mock base; the trainers add `model.resize_token_embeddings`, PEFT integration, and `accelerator.prepare()` interactions that were not exercised against real Qwen3. First pod-side run (Task #33) may surface signature mismatches; expected fix surface is small (single-line tweaks inside each trainer).
- **No actual `merge_stage0_lora.py` smoke**. The script exists from Batch 1 (review-003 era) but has not been run against a real CPT-light adapter. Day-1 Task #33 includes "run `scripts/merge_stage0_lora.py`" as its final step; if the merge fails, the rest of Day 1+ is blocked.
- **`accelerator.prepare(wrapper)` behavior with hook-attached layers is untested**. Accelerate wraps `nn.Module`s in DDP-like containers; whether forward hooks on `wrapper.base.model.layers[i]` survive the wrap is empirically unknown. Workaround: the trainer's gate-trajectory logging and eval pass both use `accelerator.unwrap_model(wrapper)` defensively. If hooks DO survive prepare(), this is over-cautious but harmless; if they don't, the unwrap-based path is the correct one.
- **No mid-training eval**. The trainers run a single eval pass at end-of-training. The PLAN.md sweep proposer makes config decisions based on the final AUC; mid-training eval would be useful for early-stopping but adds complexity. Deferred.
- **Loss curves not logged to W&B**. `losses` list is built in-memory and the final value goes to metrics.json. W&B integration would print per-step loss + LR + gate magnitudes to a dashboard. Trainer is W&B-import-ready (HF Trainer compatibility), but the explicit `wandb.log` calls are not wired. Deferred to keep the trainers small.
- **No gradient clipping**. PLAN.md mentions "gradient clipping" as a stability trick but doesn't specify a value. The trainers don't clip currently. If NaN cascades start showing up at H100 scale, this is the first knob.
- **Checkpoint saving is minimal**. Trainers save final state only (LoRA adapter or x-attn state dict). No mid-training checkpoints. Storage planning per the RUNBOOK assumes one final checkpoint per arm; that's what's implemented.
- **Param-count mismatch check is print-only**. `train_xattn.py` prints both actual and estimated trainable counts but doesn't assert their proximity. A future iteration should add `assert abs(actual - estimated) / estimated < 0.05` to fail loudly on architectural drift.
- **Event-only classifier doesn't use `delta_t`**. The classifier mean-pools encoder output across events without time-of-event input. Per PLAN.md the encoder is meant to be time-agnostic (time encoding lives in the resampler), but the classifier doesn't have a resampler. Could be added; for the baseline it's a deliberate floor — if the event-only baseline NEEDS time, it'd be a positive signal that the cross-attn time channel matters. Deferred.

---

## 6. Focus areas for the next review (review 007)

Concrete prompts for Codex when running `Review … focus: path-a-batch-4`:

1. **Label-position math.** `eval_runner.run_three_mode_eval` truncates the eval input text at the `<risk_verdict>\nlabel:` marker, then scores at the last non-pad token. Verify on a real Qwen3 tokenizer that:
   - The marker exists in every example (no truncation cut it off mid-prefix).
   - The token aligned to the position-just-after-marker is consistently the "score" position across batches.
   - ` fraud` and ` legit` each tokenize to a single token at this position (no multi-token spillover that the scoring code doesn't handle).
   - Asserts these conditions; current code does not.

2. **Eval-mode dropout per-example RNG seeding for opaque.** `EvalModeDropoutCollator.__call__` uses `self._rng` for mode sampling but then constructs a fresh `random.Random(self._rng.getrandbits(64))` for each opaque example. Verify this produces TRULY different opaque mappings across examples within a batch (otherwise opaque mode degenerates to a stable per-batch mapping). Construct two adjacent examples with the same journey family and verify their opacified texts differ.

3. **`train_structured_as_text` consistency with `data/gen/build_dataset.py::_event_to_line`.** The compact serializer in `_serialize_events_compact` MUST produce byte-identical output for the same event dict as `_event_to_line` in `data/gen/build_dataset.py`. Otherwise the structured-as-text baseline trains on a different distribution than the prompts the eval references describe. Diff the two helpers' output on the same input; flag any divergence.

4. **CPT-light LoRA target list correctness on Qwen3-8B.** The targets are `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Verify these names match what Qwen3-8B's `transformers` implementation actually uses (some models use `wqkv` packed, or `mlp.fc1` instead of `gate_proj`). If names diverge, the PEFT integration silently no-ops and Stage-0 trains nothing meaningful. Suggested check: `for n, _ in model.named_modules(): print(n)` after load, grep for the target names.

5. **`accelerator.prepare(wrapper)` + forward-hook survival.** Big unknown. If `accelerate` wraps `wrapper` in a DDP container that bypasses the hook-attachment in `base.model.layers[i]`, the cross-attention contribution silently does nothing. Suggested smoke: after `accelerator.prepare(wrapper)`, run one forward and assert `wrapper.gate_diagnostics()` (or its unwrapped equivalent) returns non-trivial values. Should also check that `wrapper._cached_kv` is populated before base.forward executes.

6. **PEFT LoRA target_modules for the wrapper's wrapped base.** `QwenXAttnWrapper.attach_lora_on_q` calls `get_peft_model(self.base, lora_config)` with `target_modules=["q_proj"]`. After `accelerator.prepare(wrapper)` and `accelerator.unwrap_model(wrapper)`, verify `wrapper.base.print_trainable_parameters()` shows the expected LoRA count.

7. **Gate trajectory grows linearly in steps × blocks.** For a 1500-step run with every_4 (6 blocks), that's 9,000 (step, layer_idx, α_attn, α_ffn) tuples ≈ 600 KB JSON. Verify the trajectory file size is reasonable at the scaled-up Day-1 case; if it's >50 MB, switch to a per-100-step sampling.

8. **`max_gate_magnitude` consistency.** `train_xattn.py` computes `max_gate_magnitude` over ALL (step, layer, α_attn) tuples in the trajectory. The `budget.yaml` halt rule reads this value and compares to `magnitude_threshold` (0.05). Verify the value the trainer writes is what the launcher reads; the launcher's `parse_metrics.py` looks for `gate_magnitude` patterns in the log, NOT in metrics.json. There may be a contract gap — flag if so.

9. **Tokenizer round-trip for ` fraud` / ` legit`.** `get_label_token_ids` takes the first token of each. Verify on real Qwen3 that:
   - `tokenizer.encode(" fraud", add_special_tokens=False)` returns a non-empty list.
   - The first token of ` fraud` and ` legit` are DIFFERENT (otherwise the score is identically zero by construction).
   - The encoding is stable across calls (no Sentencepiece nondeterminism).

10. **`predictions_<mode>.jsonl` schema matches `score_risk.py`'s input.** The trainer writes records with `score, label, journey_family, actor_family, is_hard_negative`. `eval/score_risk.py::_to_arrays` reads `score` (float) and `label` (string). Verify field names match exactly; flag any drift.

11. **Atomic writes in trainers.** All `metrics.json`, `gate_trajectory.json`, and prediction files write to `.tmp` and rename. Verify there's no path where the rename is skipped (e.g., exception between `write` and `rename`). The `common.py::write_predictions_jsonl` does it correctly; trainer-side `json.dumps + write_text + rename` patterns should match.

12. **Trainable params actually trainable.** For each trainer, after `accelerator.prepare`, verify that `sum(p.numel() for p in model.parameters() if p.requires_grad)` is non-zero AND matches the `n_trainable` value printed at startup. A mismatch indicates a freeze/unfreeze interaction bug between PEFT and Accelerate.
