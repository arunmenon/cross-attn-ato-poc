# Cross-Attention Mechanism — Architecture, Hypothesis, and Alternatives

What the cross-attention pathway in this POC actually is, what we
hypothesized it would do, what the moving parts inside it are, and what
other architectures we considered for the same job.

Companion docs:
- `docs/experiments-log.md` — what was actually run and what we learned.
- `docs/auto-research-loop.md` — how the auto-loop wired up the experiments.
- Code: `src/model/{cross_attn_block, resampler, qwen_xattn_wrapper}.py`,
  `src/model/encoders/small_transformer.py`.

---

## The hypothesis (one paragraph)

The synthetic ATO data has two streams per session: a **structured event
list** (login, txn, device_add, … each carrying bucketed risk features
like `<amount_bucket=high>` / `<geo_distance=international>`) and a
**narrative text** that a strong LLM wrote to describe the session
analytically. If we serialize the events into the text prompt
(`structured_as_text` baseline) the LM sees the signal, but it costs
tokens and the LM has to re-parse structure on every step. **Our claim
was that a Flamingo-style gated cross-attention pathway** — frozen LM
queries attending to a side-stream encoder's output — **gives the LM a
high-bandwidth route to the same signal without polluting context, and
the per-layer gates let the LM learn *where* in its stack the side
stream is useful**. If true, we'd expect cross-attn to match (cheap win)
or beat (real win) the structured-as-text concat baseline.

What the 15-arm sweep so far says: **no detectable classification lift
over `structured_as_text` within 95% CIs**, and gate magnitudes are
init-bias-carried (gates don't open beyond their starting point on this
surface). See `docs/experiments-log.md` for the run-by-run evidence.
This document is about the architecture, not the verdict.

---

## Top-level data flow

```
                   ┌─────────────────────────────────────────────────┐
                   │ Text stream (narrative + verdict footer)        │
                   │ 100-300 words, bucketed feature tokens, fenced  │
                   │ PII. Length ~256-512 tokens.                    │
                   └────────────────────┬────────────────────────────┘
                                        │ tokenize
                                        ▼
                  ┌─────────────────────────────────────────┐
                  │  Qwen3-8B (FROZEN after Stage-0 merge)  │
                  │  36 transformer layers, hidden=4096     │
                  │                                         │
                  │  ┌────────────┐    ┌────────────────┐   │
                  │  │ self-attn  │    │ ⊕ x-attn block │◀──┼──── K/V slots
                  │  │ + LoRA-on-Q│    │  tanh(α)·attn  │   │  ┌─────────────┐
                  │  │ (r=16,     │    │  + tanh(α)·ffn │   │  │ Perceiver-  │
                  │  │  trainable)│    │  residual add  │   │  │  Resampler  │
                  │  └────────────┘    └────────────────┘   │  │  K slots    │
                  │       │                  ▲              │  │  (64 or 128)│
                  │       └──────────────────┴──── (every N │  │             │
                  │            layers, per insertion_pattern)│  │  + time PE  │
                  └─────────────────────────────────────────┘  └──────┬──────┘
                                                                      ▲
                                                                      │ encoded events
                                                                      │ (B, N_events, H)
                                                       ┌──────────────┴───────────────┐
                                                       │  small_transformer encoder   │
                                                       │  6 layers × 4 heads, H=4096  │
                                                       │  tied to Qwen's hidden_dim   │
                                                       │  reads bucketed event tokens │
                                                       └──────────────────────────────┘
                                                                      ▲
                                                                      │
                                ┌─────────────────────────────────────┴────────────────┐
                                │ Structured event stream (the "side" stream)          │
                                │ JSON-event list: t=0 login {…}, t=2 device_add {…},  │
                                │ t=7 txn {amount_bucket=high, …}. 5-200 events with   │
                                │ per-event Δt timestamps.                             │
                                └──────────────────────────────────────────────────────┘
```

Read it from the bottom: events get tokenized → encoded by a small
transformer → compressed to K fixed slots by a Perceiver-Resampler → those
slots become K/V for cross-attention blocks inserted every N layers of
Qwen3, gated by a per-block scalar tanh(α). The LM produces a verdict at
the end of the prompt; the loss is next-token CE on the text stream.

---

## The pieces, in detail

### 1. The base model — Qwen3-8B, frozen after Stage-0 merge

| | |
|---|---|
| Model | Qwen3-8B (36 layers, hidden=4096, 32 attn heads, 8 KV heads) |
| Stage 0 | CPT-light = embedding + LoRA, trained on 1.5k LLM-narrated narratives, then merged into the base |
| Result of Stage 0 | `qwen3-8b-cpt-light-merged` — a single set of weights, no adapter attached |
| In Stage 1 | The merged base is **frozen** entirely (no LoRA on it, no weight updates). Only the cross-attn surgery is trained. |

**Why merge?** Because Stage-1 adds a fresh LoRA-on-Q. Stacking a second
LoRA on top of a Stage-0 LoRA creates "adapter confusion": gradients
through the merged-adapter path are ambiguous. Merging Stage-0 first
gives Stage-1 a clean, frozen starting point.

### 2. The side-stream encoder — `small_transformer`

`src/model/encoders/small_transformer.py`

| | |
|---|---|
| Type | Vanilla pre-norm transformer encoder |
| Default config | `n_layers=6, n_heads=4` |
| Hidden dim | **Tied to the LM's `hidden_dim=4096`** (so the resampler doesn't have to project) |
| Input | Per-event tokens via `EventVocab`: event-type token + bucketed-feature tokens, with Δt per event |
| Output | (B, N_events, 4096) — one vector per event |
| Parameters | ~few hundred million (mostly the 4096-wide FFN) |

Trained from scratch jointly with the cross-attn layers. The encoder is
the cheapest part to swap (alternatives below).

### 3. The Perceiver-Resampler

`src/model/resampler.py`

| | |
|---|---|
| Job | Compress N variable-length encoder outputs to **K fixed slots** (K ∈ {64, 128}) |
| Mechanism | K learnable latents attend to (encoder_output + sinusoidal-time-PE) via cross-attention, then self-attention, then FFN — Perceiver style |
| Time encoding | Sinusoidal on **cumulative Δt** between events (continuous, not integer-indexed). `time_base=10000`. |
| Mixed-precision care | Time encoding is computed in **fp32** then cast to bf16 to keep sub-minute resolution at session-scale times (review 006 finding #2 — bf16's ~7-bit mantissa would otherwise collapse 30-60s deltas at t≈30k seconds) |
| Output | (B, K, 4096) — fixed-size K/V for downstream cross-attn |

The resampler matters because Qwen's self-attention works on a sequence
of fixed length; if every cross-attn block saw N variable events, the
attention cost would scale poorly. Compressing to K once at the bottom
of the side stream lets every cross-attn block re-use the same K/V slots.

### 4. The gated cross-attention dense block

`src/model/cross_attn_block.py`

This is the Flamingo "gated cross-attention dense" block. Per insertion
point in the LM stack:

```
h ← h                                                  # the LM hidden state
h ← h + tanh(α_attn) · OutProj( CrossAttn( norm(h), norm(KV) ) )
h ← h + tanh(α_ffn)  · FFN( norm(h) )
```

Key choices:

| Choice | Rationale |
|---|---|
| **Pre-norm** | LayerNorm before Q and KV projections (Flamingo + modern transformer convention) |
| **Bottleneck cross dim** | `cross_dim = hidden_dim // 4 = 1024` to keep parameter count inside the 200-400M Stage-1 budget |
| **FFN width** | `dim_feedforward = hidden_dim // 2 = 2048` (half the standard 4× ratio) |
| **Gate: `tanh(α)` scalar per block** | A single learnable scalar per block, separately for attn and ffn. `tanh` bounds the gate to (-1, 1). |
| **`gate_init`** | Two values: `zero` (α=0 → gate exactly 0 → exact identity at step 0) or `small_0.01` (α=0.01 → gate ≈ 0.01) |
| **`OutProj` uses standard init (NOT zero)** | Review 006 finding #1: zero-init on both `out_proj` AND `α_attn` makes both gradients vanish (`dL/dα ∝ attn_out = 0` if `out_proj=0`, AND `dL/dout_proj ∝ gate ≈ 0`). Only the gate is zero-init; `out_proj` uses standard init. Otherwise gate=zero arms become permanently dead. |

The `tanh(α)`-gated residual is the Flamingo invariant: when α=0 the
block is exactly the identity. The LM can ignore the side stream
entirely by keeping α near 0; if α opens up, the side-stream signal
flows through. The size of `|tanh(α)|` is what we mean by "max_gate
magnitude" — the headline diagnostic we tracked across the sweep.

### 5. LoRA-on-Q (Stage-1 only)

A **fresh** LoRA r=16 on Qwen's self-attention **query projections** only.
This is separate from the Stage-0 LoRA (which was merged into the base
weights). Trained jointly with the cross-attn surgery.

Why on Q only and not K/V? Because the LM's self-attention is doing its
own work on the text stream; LoRA-on-Q lets the LM modulate *what it
asks for* without changing the relationships in the embedding space.
Cheaper than full LoRA on Q+K+V, and avoids interfering with the
attention pattern over the text.

### 6. Insertion patterns

`src/model/qwen_xattn_wrapper.py` (`compute_insertion_layers`)

Where in Qwen's 36-layer stack do we put cross-attn blocks?

| Pattern | Layer indices (0-indexed) | # blocks | Idea |
|---|---|---|---|
| `every_4` | 12, 16, 20, 24, 28, 32 | 6 | dense conditioning, starts mid-stack |
| `every_8` | 12, 20, 28 | 3 | sparser, ~2× cheaper |
| `late_only` | 32, 33, 34, 35 | 4 | put all conditioning near the verdict (signal injection right before logits) |
| (deferred) `every_2` | 12, 14, 16, … 32 | 11 | too many parameters for the 200-400M budget |

Starting at layer 12 (not 0) is a Flamingo convention: early layers do
token-level work; conditioning is more useful where the LM is doing
semantic/sequence-level work.

---

## Parameter budget breakdown

The 200-400M Stage-1 trainable cap (per PLAN.md) comes from VRAM math
on a single H100. Distribution per arm:

| Component | Params |
|---|---|
| Frozen Qwen3-8B base | ~8.0B (no gradients, no optimizer state) |
| Side-stream encoder (small_transformer 6×4) | ~200-300M |
| Perceiver-Resampler (~2-3 layers × hidden=4096) | ~80-120M |
| Gated cross-attn blocks (every_4 = 6 × ~25M each) | ~150M |
| LoRA-on-Q r=16 across 36 self-attn layers | ~5M |
| **Stage-1 trainable total** | **~400-450M** for `every_4`, less for `every_8` / `late_only` |

`estimate_block_param_count()` and `estimate_wrapper_trainable_params()`
in the code give exact numbers for a given hidden_dim / pattern / slot
count. The Round-1 grid sat comfortably inside VRAM at micro-batch=4,
grad-accum=8 → effective batch 32, bf16.

---

## What the sweep tells us about the hypothesis

After 15 valid x-attn arms across Round 1 + Round 2 + Expanded Sweep
Phases 1-3, the picture is consistent:

| What we tested | What we hypothesized | What we got |
|---|---|---|
| `every_4` vs `every_8` vs `late_only` (insertion density) | denser conditioning → better signal access | inert within CIs |
| `slots=64` vs `slots=128` (K/V budget) | more slots → finer compression → better signal access | inert within CIs |
| `gate_init=small_0.01` vs `zero` (whether gates start open) | gates will learn to open if init=0 — init shouldn't matter long-term | **gates ride init**: max_gate stays near init value through 1500 steps |
| `lr=3e-5` vs `1e-4` vs `3e-4` (does LR throttle gate learning?) | higher LR → more gate motion → more side-stream use | gate motion is **LR-invariant** in this range; lowest LR produced highest max_gate (the dial isn't LR) |
| `warmup=100` vs `500` (does warmup gate the early signal?) | shorter warmup → gates open earlier → larger final magnitude | inert |
| `seq_len=4096` + `steps=3000` (does long context unlock signal?) | yes if the text has untapped long-range structure | **failed mechanically** (180m session boundary, F8 v2 patch); retry possible |

The empirical conclusion-so-far isn't "cross-attn doesn't work" — it's
"on **this** synthetic surface at **this** scale (5k clean eval), the
side-stream signal is already in the text via the bucketed-feature
tokens, and structured-as-text concat captures the same information at
roughly the same `hn_fpr_worst`." The cross-attn pathway adds capacity
but no extra signal because the signal is already routed.

Whether that conclusion holds on a real (or much larger) ATO dataset is
a Day-4+ question.

---

## Event-encoder alternatives — what else we considered

The encoder is the cheapest piece to swap. Three options were
documented in PLAN.md; only one was implemented in this POC.

### What's implemented: `small_transformer`

`src/model/encoders/small_transformer.py` — vanilla pre-norm transformer
encoder, 6 layers, 4 heads. **This is what every x-attn arm in the
sweep used.**

Why it was chosen for the POC:

1. **Inductive bias matches the data**: events are a short sequence
   (5-200 items), each with structured per-event features. Transformers
   handle variable length and global attention naturally.
2. **Hidden-dim parity with the LM**: 4096-wide, so the resampler
   doesn't have to project. Cleaner code.
3. **Fast to implement**: nn.TransformerEncoderLayer + EventVocab
   tokenizer + ~3 days. Day-1 deliverable.

Limitations we accepted by picking this:

1. Sequential order doesn't matter much beyond the Δt encoding — a
   permutation-invariant encoder (Set Transformer) might be more
   parameter-efficient.
2. The bucketed feature tokens are treated as if they were word tokens
   (single embedding per bucket value). For richer tabular data, an
   FT-Transformer style per-feature encoder would be more expressive.

### Deferred: FT-Transformer (FT-Tx)

Tabular-foundation-model approach. **Stub file exists at
`src/model/encoders/ft_transformer.py` (currently empty).**

What it would look like:
- Each event becomes a vector of (feature, value) pairs.
- A per-feature `FeatureTokenizer` learns embeddings for the feature
  name × value-bucket combinations.
- A transformer attends across the resulting feature-tokens within an
  event AND across events.

When it matters:
- When events have **many heterogeneous features** with complex
  inter-feature relationships (e.g., "high amount + new recipient + bursty
  velocity" is fraudish; one of those alone isn't). The current vocabulary
  treats each bucket as an opaque token, so the model has to discover
  feature × feature interactions through self-attention without an
  explicit inductive bias.

Why deferred: the synthetic generator's event schema is small enough
(~10 feature families × 3-4 buckets each) that `small_transformer`
exposes them adequately. On a real PayPal-internal feature schema (100s
of features), FT-Tx would likely earn its keep.

### Deferred: CNN + LSTM

**Stub file exists at `src/model/encoders/cnn_lstm.py` (currently empty).**

What it would look like:
- 1-D CNN over the event sequence captures **local n-gram patterns**
  (e.g., "device_add → pw_reset → txn within 5 minutes").
- LSTM on top captures longer-range dependencies.
- Outputs to the resampler as (B, N, H) like the transformer.

When it matters:
- When the fraud signal is in **specific event sequences with local
  ordering** (not "this happened" but "this then that then that").
- Fewer parameters than a transformer at the same hidden dim.

Why deferred: the transformer captures the same patterns via attention,
and we wanted a single-encoder POC to keep the sweep dial space small.

### Not considered (worth knowing about)

| Architecture | Why we didn't pick it |
|---|---|
| **Set Transformer** (Lee et al. 2019) | Permutation-invariant — ignores event order. Useful if "set of risk signals" is the right abstraction, but our hypothesis was that time-ordering matters (cred-stuff → device-add → txn ≠ txn → device-add → cred-stuff). Δt-aware ordering is a stronger inductive bias for ATO. |
| **GRU/LSTM-only** (no transformer) | Smaller, but couldn't attend globally across events. Would force the resampler to do all the global integration. |
| **Mamba / S4 / state-space models** | Could handle very long event sequences (1000s) efficiently. Our events cap at 200 — overkill. Also a younger codebase ecosystem; not worth the integration risk for a 3-day POC. |
| **Per-event embedding lookup, no encoder** | No interaction between events. The resampler would have to do all the heavy lifting. Would test whether the resampler alone is enough — interesting null-model variant, but the POC's hypothesis was that the encoder matters. |
| **Hierarchical (event-level + session-level)** | Two-stage encoder: per-event features → event embedding → cross-event transformer. Would be cleaner conceptually but doubles the encoder complexity. Worth doing on a real dataset. |
| **Use the LM itself as the encoder** | "Just put events in the text prompt" — that's the `structured_as_text` baseline. Already in the head-to-head. |

---

## What dials we left on the table

The current sweep_space holds the encoder fixed at `small_transformer`
and varies architecture / init / training dials around it. A logical
Day-4 expansion would be:

1. **Encoder swap**: `small_transformer` → FT-Tx → CNN+LSTM on the same
   leader cell. Tests whether the encoder is the bottleneck.
2. **Bigger encoder**: `n_layers=6 → 12` and/or `n_heads=4 → 8`. Tests
   whether the encoder is under-parameterized.
3. **Resampler depth**: more layers in the Perceiver-Resampler. Tests
   whether the K-slot compression is the bottleneck.
4. **Direct cross-attn (skip the resampler)**: cross-attention directly
   over the variable-length encoder output, no compression. Tests
   whether the resampler is destroying information.

None of these are budgeted in the current POC's 3-day window. They're
the natural follow-ups if a real-data Day-4 POC takes this code further.

---

## Why we picked this architecture (not just for show)

Three reasons cross-attn was the architecture-of-choice over alternatives
that would have given the LM access to the same signal:

1. **Decouples context length from signal access**. The text stream stays
   short (narrative + verdict footer ≈ 256-512 tokens). All event signal
   flows through K/V slots, separately from the LM's KV cache.

2. **Gates let the LM learn where it wants the signal**. The per-block
   `tanh(α)` gate is the diagnostic that tells us *whether the LM is
   using cross-attn at all*. A max_gate of 0.01 across all blocks means
   "the LM does not need the side stream"; a max_gate that opens up to,
   say, 0.5 means "the LM is leaning heavily on cross-attn at this layer."
   This is a much more interpretable signal than "is the loss lower."

3. **Apples-to-apples with structured-as-text**. Both arms see the same
   `qwen3-8b-cpt-light-merged` base. The difference is *how* the events
   reach the LM — concat-into-text vs cross-attn. So a clean comparison
   tests **the conditioning mechanism**, not the model + data.

The first two were borne out — we got clean gate magnitude readouts that
let us conclude "the gates are inert" with confidence. The third was the
load-bearing comparison the sweep was designed for, and the answer (so
far) is that the two mechanisms are tied within CIs on this surface.

---

## Where to read the code

| File | What's in it |
|---|---|
| `src/model/cross_attn_block.py` | `GatedCrossAttnDense` — the per-block factory. The gating math, the bottleneck dim, the init choices. ~400 lines. |
| `src/model/resampler.py` | `PerceiverResampler` — K learnable latents attending to (encoder_output + sinusoidal-time-PE). The mixed-precision time-encoding subtlety. ~325 lines. |
| `src/model/qwen_xattn_wrapper.py` | `QwenXAttnWrapper` — the full assembly. Loads the frozen base, inserts cross-attn blocks per `insertion_pattern`, attaches the Stage-1 LoRA-on-Q, wires the side-stream encoder + resampler. ~600 lines. |
| `src/model/encoders/small_transformer.py` | `SmallTransformerEncoder` + `EventVocab` tokenizer. ~330 lines. |
| `src/model/encoders/ft_transformer.py` | (empty stub, deferred) |
| `src/model/encoders/cnn_lstm.py` | (empty stub, deferred) |
| `src/train/train_xattn.py` | The Stage-1 trainer that consumes all of the above. |

If you're hunting for a specific design choice's rationale, every
non-obvious choice has a review-finding comment (`review 005 #1`,
`review 006 #1`, etc.) pointing to the review thread that justified it.
