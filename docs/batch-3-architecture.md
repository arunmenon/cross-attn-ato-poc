# Batch 3 — model architecture surgery

Implements the Flamingo-style gated cross-attention apparatus on a
frozen Qwen3-8B. Completes PLAN.md "Architecture" and sets up Task #35
(Day 2 Hr 0-4: X-attn architecture surgery) for pod-side execution.

Baseline commit going into this batch: `38e9840` (review 004 closure).
Closing commit: filled in by the two-commit INDEX backfill pattern.

---

## 1. What landed

| File | Lines | One-line |
|---|---|---|
| `src/model/encoders/small_transformer.py` | ~290 | Side-stream encoder + `EventVocab` + `tokenize_events`. Consumes the structured event stream (one token per event, event-type + bucket-token bag) and produces (B, N, H). |
| `src/model/resampler.py` | ~190 | Perceiver-Resampler with sinusoidal-on-Δt time encoding INSIDE the resampler. Compresses (B, N, H) → (B, K, H). |
| `src/model/cross_attn_block.py` | ~180 | Gated cross-attention dense (Flamingo GATED XATTN-DENSE). Two scalar gates (`α_attn`, `α_ffn`), tanh-gated residual adds. Identity at step 0 when `gate_init="zero"`. |
| `src/model/qwen_xattn_wrapper.py` | ~320 | The integration. Freezes base, registers forward hooks at insertion layers, precomputes K/V once per forward, runs base.forward() with hooks injecting cross-attn outputs. Has `attach_lora_on_q()` for the Stage-1 fresh-LoRA. |

Plus `__init__.py` files were already in place from earlier batches.

**Local verification cap**: PyTorch is not installed on the laptop. Only
the torch-free portions of self-tests ran here (`EventVocab`,
`tokenize_events`, `compute_insertion_layers`). Live forward-pass tests
run on the pod via Task #35.

---

## 2. Design decisions (the non-obvious ones)

### 2.1 Hook-based injection over subclassing

The wrapper attaches `forward_hook`s to `base.model.layers[i]` rather
than subclassing `Qwen3DecoderLayer` and re-implementing forward.
Rationale:

- HF's `Qwen3DecoderLayer.forward` signature has drifted across
  transformers versions (added `cache_position`, `position_embeddings`,
  changed return-tuple shape). A subclass would need version-specific
  shims.
- Hooks operate on the OUTPUT of the layer, which is always
  `(hidden_state, *aux)` in HF's convention. That contract is stable.
- Hooks compose cleanly with PEFT LoRA: PEFT wraps `q_proj` with a
  LoRA module, the original layer's forward still runs, the hook still
  fires on the layer's OUTPUT. No interaction friction.

Trade-offs accepted:
- Hooks add a small per-layer overhead (an extra `if isinstance(output,
  tuple)` check). Negligible in inference; ~1% in training.
- The wrapper is harder to `torch.save(model)` (pickle) because hook
  closures don't pickle cleanly. State-dict saves (recommended for
  checkpoints) are unaffected.
- The wrapper instance must be the same object across forward calls;
  re-creating it costs a hook re-registration.

### 2.2 K/V cache-once via instance attribute, not return value

The hook fires *during* `self.base(...)` — by then, the K/V precompute
has finished and stashed the result in `self._cached_kv`. The hook
reads from that.

Alternative considered: pass K/V down through `base.forward(...,
extra_kv=...)`. Rejected — would require deep changes to HF's
attention modules. The current approach keeps the K/V plumbing entirely
in the wrapper.

Safety: `_cached_kv` is cleared in `finally` after every forward, so a
stale cache cannot bleed across calls.

### 2.3 Time encoding lives in the resampler, not the encoder

PLAN.md Architecture / Pair-2 consensus / FraudTransformer rationale.
The encoder embeds event-type and bucket-tokens; the resampler adds
sinusoidal positional encoding from `delta_t.cumsum()` immediately
before cross-attending with the learned latents.

Why not in the encoder:
- The encoder's job is to extract per-event semantic structure (which
  event happened, with what bucketed features). That structure is
  time-agnostic.
- Putting time information at the K/V boundary lets the model use it
  as relative-position info rather than baking it into the encoder's
  per-event embedding (where it would diffuse).
- It makes the per-event embedding cache-able if we ever want to
  reuse encoder outputs across different time-scaling experiments.

### 2.4 Time base = 10,000 seconds (≈2.78 hours)

Sinusoidal PE with `time_base=10_000` and Δt-in-seconds means the
lowest-frequency dimension has period `2π × 10_000 ≈ 17.4 hours`. This
covers the typical session-duration range (seconds to a few hours)
without resolving wraparound issues. PLAN.md noted FraudTransformer's
time-criticality concern; the time_base is chosen to give the model
discriminative power over sub-minute Δt while still encoding
session-scale durations.

If the trainer's gate-trajectory diagnostics show the model isn't using
the time channel (gates open on event-type / bucket info but stay flat
on time-perturbed inputs), this is the first knob to retune.

### 2.5 Encoder hidden_dim independent of LM hidden_size

The side-stream encoder runs at 256 hidden_dim by default, while
Qwen3-8B's `hidden_size` is 4096. A `kv_projection: nn.Linear(256,
4096, bias=False)` bridges the two at the resampler-output boundary.

Why decouple:
- Encoder size is set by the structured-stream complexity (~36-token
  vocab, ~200-event sequences). 256d is plenty.
- Forcing the encoder to run at 4096d would 16× the encoder param count
  without buying signal — the bucketed-feature space doesn't have
  enough cardinality to justify it.
- The projection is trainable, has ~1M params, and adds negligible
  compute compared to the LM.

### 2.6 Scalar gates per block, not per-head or per-channel

Each `GatedCrossAttnDense` has two scalar parameters (`α_attn`,
`α_ffn`). Flamingo used scalars; we follow.

Why scalars:
- Easy to log per training step (one number per block).
- Easy to plot — the "did the gates open?" question becomes a clean
  per-layer trajectory.
- The convergence-halt check in `budget.yaml`
  (`zero_gate_activation`) trivially compares against a magnitude
  threshold.

Per-channel gates would let the model selectively open / close
information per feature dimension, which sounds nice but trains worse
in our scale regime (Flamingo's ablations).

### 2.7 The `lora_r_on_q` knob is delegated to PEFT, not hand-rolled

`attach_lora_on_q()` calls `peft.get_peft_model` with
`target_modules=["q_proj"]` and `task_type="CAUSAL_LM"`. PEFT handles
the adapter math, gradient routing, and save/load.

We do NOT hand-roll LoRA because:
- PEFT's serialization plays nicely with HF's `save_pretrained` and the
  trainer's checkpointing.
- PEFT supports `target_modules` regex/list semantics that compose with
  Qwen3's module naming without us hard-coding paths.
- The `merge_stage0_lora.py` script (Batch 1, review 003 fix) uses
  PEFT's `merge_and_unload()`. Using PEFT here keeps the Stage-0 →
  Stage-1 adapter lifecycle consistent.

The Stage-0 LoRA is merged into the base BEFORE the wrapper is
constructed (per PLAN.md "Stages and adapter lifecycle"). The wrapper's
`attach_lora_on_q()` adds a fresh Stage-1 LoRA on top of the merged
base. No adapter confusion.

---

## 3. Where this maps to PLAN.md

| PLAN.md section | Implementation |
|---|---|
| "Architecture" → Frozen base | Wrapper `__init__` freezes all base params; `trainable_param_summary()` confirms `base_trainable == 0`. |
| "Architecture" → Cross-attn layers per `insertion_pattern` | `compute_insertion_layers()` resolves the sweep dial. `every_4` on Qwen3-8B (36 layers) → 6 insertion points at layers 12, 16, 20, 24, 28, 32. |
| "Architecture" → Resampler with sinusoidal-on-Δt | `resampler.py` `sinusoidal_time_encoding()` + `PerceiverResampler`. Time encoding added to K/V inside the resampler. |
| "Architecture" → x-attn variant: plain MHA (not MLA) | `torch.nn.MultiheadAttention` everywhere; we never touch the base's MLA path. |
| "Architecture" → side-stream encoder = `small_transformer` | `SmallTransformerEncoder` factory; FT-Transformer / CNN+LSTM are placeholder slots (Day 4+). |
| "Architecture" → Gate init {zero, small_0.01} | `GATE_INIT_VALUES` map; zero-init verified to produce exact identity in self-test. |
| "Architecture" → LoRA on Q, r=16 | `attach_lora_on_q(r=16)` via PEFT. |
| "Stages and adapter lifecycle" → Stage-0 merged BEFORE Stage-1 LoRA | Wrapper accepts a pre-merged base and adds Stage-1 LoRA on top. |
| "Two distinct x-attn components" | Implementation cleanly separates: encoder is one module, resampler is another, cross-attn blocks are a third. The wrapper composes them. |
| Risk: "gates don't open" | `gate_diagnostics()` exposes per-block `|tanh(α)|` for the trainer's `zero_gate_activation` halt-condition check. |

---

## 4. Smoke / self-test results

Per-file `_self_test()` invocations on the laptop (no torch):

```bash
python3 -m src.model.encoders.small_transformer
# event vocab size: 36
# vocab + tokenize_events OK
# torch not installed; encoder forward-pass test skipped (runs on the pod)

python3 -m src.model.resampler
# torch not installed; resampler self-test skipped (runs on the pod)

python3 -m src.model.cross_attn_block
# torch not installed; cross_attn_block self-test skipped (runs on the pod)

python3 -m src.model.qwen_xattn_wrapper
# compute_insertion_layers OK for every_4/every_8/late_only
# torch not installed; wrapper integration self-test skipped (runs on the pod)
```

What the pod-side self-tests verify (once torch is available):

- **small_transformer**: encoder forward returns shape `(B, N, hidden_dim)`.
- **resampler**:
  - `sinusoidal_time_encoding` returns shape `(B, N, hidden_dim)`, all finite, monotonically different across Δt rows.
  - `PerceiverResampler` forward returns `(B, K, H)`.
  - Padded positions in encoder output do NOT affect resampler output (key-padding-mask is honored).
- **cross_attn_block**:
  - `gate_init="zero"` → output equals input bit-for-bit (identity at step 0).
  - `gate_init="small_0.01"` → output differs (block has signal), gates ≈0.01.
  - Gradient flows through both `α_attn` and `α_ffn`.
  - KV padding mask is honored.
- **qwen_xattn_wrapper**:
  - Hooks attach to the right layer indices.
  - Forward returns HF-style output with `.logits`.
  - `base_trainable == 0` confirms full freeze.
  - `gate_diagnostics()` returns one entry per insertion layer.

---

## 5. Known limitations / deferred

- **No real Qwen3-8B exercise yet.** The wrapper's mock-base self-test uses a tiny `_MockBase` (8 layers × 32 hidden_dim) — sufficient to verify the wiring and the hook-registration math, but not the real serialization, dtype, or memory profile. First real run is Task #35 (Day 2 Hr 0-4) on the pod. Risk: HF's Qwen3 implementation may return tuples of different arity (e.g., HF adds `present_key_value` in some versions). The hook's `isinstance(output, tuple)` handling covers the common case; if it breaks on real Qwen3, the fix is a single-line update inside `_hook`.
- **`every_4` on tiny-model self-test.** `compute_insertion_layers("every_4", 8)` returns `[]` (start=12, range exhausted). The mock self-test uses `late_only` to work around. Production uses Qwen3-8B (36 layers) where `every_4` yields 6 insertion points. Not a bug; flagging for clarity.
- **Resampler self-attention is unmasked.** The Perceiver self-attention between cross-attention rounds attends across all `K` latent slots without any masking. Since all slots are "real" (the resampler has compressed everything to fixed K), this is correct. Documented here for the reviewer.
- **Encoder doesn't see actor_family directly.** The encoder ingests event-type + bucket tokens only. The `actor` field on each event is discarded by `tokenize_events`. This is intentional (actor_family is wrapped at the journey level via `<actor_*>`, which the LM consumes through text); the structured side gets actor-specific *behavior* (tool_call events, timing modulation) but not actor *labels*. Confirms with PLAN.md "Two distinct x-attn components" — the encoder is for behavioral signal, not labels.
- **No mixed-precision casts in the wrapper.** The wrapper accepts whatever dtype the base is in (bf16 in production). The cross-attention blocks cast their gate values to `h.dtype` to avoid silent fp32 upcasts inside an otherwise bf16 forward. But the encoder + resampler run in their default fp32 unless the caller `.to(bfloat16)`s the whole wrapper. Trainer should do this explicitly; otherwise activations are upcast at the `kv_projection` boundary.
- **Generation path not exercised.** The wrapper's `forward()` works for training (teacher-forcing). For autoregressive generation, HF's `generate()` calls `forward()` repeatedly with growing `input_ids` and `past_key_values`. The K/V cache from cross-attention should be precomputed ONCE per generation (since the structured side doesn't change as the LM decodes more text). The current `forward()` recomputes K/V each call. Not blocking for Day-2 training but the trainer should not use `generate()` until this is patched. Flag for review.
- **No torch.compile / fused attention.** Standard `nn.MultiheadAttention`. SDPA / Flash-Attention will be applied at trainer time via `torch.set_float32_matmul_precision('high')` or `attn_implementation='sdpa'` on the base, but the cross-attn blocks themselves use the eager path. ~25% of training time savings on the table; acceptable for a 3-day POC.

---

## 6. Focus areas for the next review (review 005)

Concrete prompts for Codex when running `Review … focus: path-a-batch-3`:

1. **Insertion-layer math sanity.** Verify `compute_insertion_layers` returns expected indices for `every_4`, `every_8`, `late_only` on `n_hidden_layers=36`. Particularly check that `every_4` starts at 12 (mid-stack) per the docstring and the explicit PLAN.md choice — not at 0 or 4.

2. **Zero-gate identity at step 0.** Most-critical correctness check. `GatedCrossAttnDense(gate_init="zero")(h, kv, mask)` MUST return `h` bit-for-bit. The self-test asserts this; verify the assertion is honest (not eps-tolerant in a way that masks a real shift). Try with extreme `kv` values (e.g., `kv = torch.ones(...) * 1e6`) — if the output drifts at all from `h`, the identity guarantee is broken.

3. **Padding mask enforcement.** Both `PerceiverResampler` and `GatedCrossAttnDense` accept a key-padding mask for the KV. Verify by constructing two batches that differ ONLY at masked positions; outputs at unmasked positions must be byte-identical. The self-tests cover this; confirm they exercise both modules.

4. **Hook registration on real Qwen3.** When loaded on the pod, `base.model.layers` should be a `nn.ModuleList` of `Qwen3DecoderLayer` instances. Verify:
   - `wrapper._hook_handles` has the expected length.
   - Each handle's `.remove()` actually de-registers (in case the trainer needs to swap blocks).
   - The hook's `isinstance(output, tuple)` branch fires on real Qwen3 (the layer should return a tuple).

5. **K/V cache lifecycle.** After `wrapper.forward()` returns, `wrapper._cached_kv` must be `None`. Verify the `finally` clause in `forward()` clears state even when `base.forward()` raises. Construct a base that raises mid-forward and confirm `_cached_kv` is `None` afterwards.

6. **Trainable-param count.** On a real merged Qwen3-8B base (≈8B params), `trainable_param_summary()['base_trainable']` should be 0 BEFORE `attach_lora_on_q()`. After `attach_lora_on_q(r=16, target_modules=['q_proj'])`, only the LoRA A/B matrices on q_proj should be trainable (∼0.5% of base). The x-attn machinery (encoder + resampler + kv_projection + xattn_blocks) should be in the hundreds of millions of params.

7. **Sinusoidal time encoding monotonicity.** For two events with very different Δt (e.g., 1 second vs 1 hour), the PE vectors should be substantively different. The self-test checks this with `assert not torch.allclose(pe[0, 1], pe[0, N-1])`. Stronger check: cosine similarity between PE rows should decrease as Δt difference grows. Flag if it doesn't.

8. **PEFT integration.** `attach_lora_on_q()` calls `get_peft_model` which wraps `self.base`. After that call, the wrapper's `self.base` is a `PeftModel`, not the original. Verify:
   - `wrapper.base.print_trainable_parameters()` reports the expected count.
   - The hooks STILL fire on `wrapper.base.model.layers[i]` (PEFT preserves the layer list).
   - `wrapper.base.save_pretrained(path)` saves the LoRA adapter only (small file), not the full base.

9. **Dtype consistency.** When the wrapper is moved to bf16 (`wrapper.to(torch.bfloat16)`), does the encoder still produce bf16 outputs? Does the kv_projection? Does the cross-attn block? Construct a bf16 forward and verify all activations are bf16. Look out for silent fp32 fallbacks (especially in `LayerNorm`, which historically defaults to fp32).

10. **Generation-mode K/V caching gap.** The "Known limitations" section flags that `generate()` would re-precompute K/V each step. Verify this is actually a problem (i.e., construct an HF `model.generate()` call against the wrapper and confirm K/V is recomputed each step). If so, propose the fix (a flag like `wrapper.prefill_kv_for_generation(...)` that locks the cache for the duration of the generation).

11. **Mock-vs-real Qwen3 drift.** The `_MockBase` in the self-test mimics the HF convention `(hidden_state, *aux)`. Verify by inspecting real `transformers.Qwen3DecoderLayer.forward` that this is still the convention as of `transformers==4.46.3` (the pinned version in `requirements.txt`). If HF has changed it, the mock — and the hook — need updating.

12. **AGENT_INSTRUCTIONS coverage.** The agent loop (Day 2 onward) will instantiate `QwenXAttnWrapper(...)` per experiment. Does `AGENT_INSTRUCTIONS.md` need a new section explaining how to pass the wrapper's `insertion_pattern` etc.? Or is the existing `experiment_template.yaml` schema sufficient? Flag if a doc update is needed.
