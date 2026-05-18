# Cross-Attention Mechanism — Architecture, Hypothesis, and Alternatives

This doc explains what the cross-attention pathway in this POC actually
is, what we hoped it would do, and what other shapes it could have
taken. Written for a reader who knows transformers exist but hasn't
shipped Flamingo-style architectures before — terms are defined the
first time they appear, and there's a glossary at the bottom if you
want to jump.

Companion docs:
- `docs/experiments-log.md` — what was actually run, what we found.
- `docs/auto-research-loop.md` — how the auto-loop ran the experiments.
- Code: `src/model/{cross_attn_block, resampler, qwen_xattn_wrapper}.py`,
  `src/model/encoders/small_transformer.py`.

---

## The setup, in plain terms

Each fraud session in the data has **two views of the same activity**:

1. A **narrative paragraph** that reads like an analyst's note —
   "user logged in from a new device, reset password, then transferred
   $4,500 to a recipient they added 20 minutes ago."

2. A **structured event list** that reads like a database log —
   `login (geo=international)`, `device_add (age=new)`,
   `pw_reset (auth=password)`, `txn (amount=high, recipient=newly_added)`.

The two views encode the same underlying session. The narrative is good
for a language model — that's what LMs are built for. The events are
where the *features* are: amount buckets, geo distances, IP risk scores.
Fraud detection has historically lived on the event side.

**The question this POC asks**: can we get a frozen language model to
*read both at once*, so it gets the narrative as input but also has a
side channel to peek at the events whenever it wants?

That side channel is what cross-attention is.

---

## What is "cross-attention," intuitively?

In a normal transformer language model, each layer does **self-attention**:
every token in the input looks at every other token in the input. The
text talks to itself.

**Cross-attention** is the same operation, but pointed at a *different*
sequence. The text tokens (queries) look at — and pull information
from — a separate set of vectors (keys + values). The text still
"reads itself" via self-attention as usual; cross-attention is an
*additional* pathway, sandwiched between self-attention layers.

If self-attention is "the LM thinking about its prompt," cross-attention
is "the LM glancing over at a side document while thinking." We give
the LM a side document — the encoded event stream — and we let it
glance whenever it wants.

The Flamingo paper (Alayrac et al. 2022) introduced the specific shape
of cross-attention we use here. Two design choices from Flamingo matter:

1. **The base LM is frozen**. We don't change its weights at all.
   Cross-attn is added *on top* of a fixed LM. That keeps the LM's
   language ability intact and lets us swap the side channel in and out.

2. **Each cross-attn block is gated** by a learnable scalar `tanh(α)`.
   At initialization the gate is set to ~0, so the block is a
   no-op — the LM ignores the side channel entirely. During training,
   `α` can grow if the side channel is useful. The gate's size tells
   us *whether the LM is actually using cross-attn*.

This second property is the diagnostic we cared most about. If the
gates stay near 0, the LM is saying "I don't need the side channel."
If they grow, the LM is saying "I'm leaning on cross-attn at this layer."

---

## The hypothesis (what we expected)

> Cross-attention will give the LM a high-bandwidth route to the
> structured event signal *without* requiring us to serialize events
> into the prompt. The gates will tell us where in the LM stack the
> event signal is useful — likely later layers, near the verdict.
> Result: cross-attn will match or beat the `structured_as_text`
> baseline (which just concatenates serialized events into the prompt).

What the 15-arm sweep so far shows:

- The gates **don't open** beyond their starting value (~0.01 if they
  started at 0.01, ~0.004 if they started at 0).
- HN-FPR-worst on the held-out eval is **statistically tied** with the
  `structured_as_text` baseline (CIs overlap).
- This holds across every architectural and training-dial perturbation
  we tested.

The bullet conclusion: **on this synthetic surface at this scale, the
LM doesn't need the side channel** — the event signal is already routed
through the bucketed-feature tokens in the text. Whether the conclusion
holds on real data is a Day-4+ question.

This doc is about *what we built*, not about *whether it should have
worked*. The "did it work" verdict lives in `docs/experiments-log.md`.

---

## A 30-second tour of the architecture

```
                  ┌──────────────────────────────────────────────────┐
                  │  Narrative text (the LM's input)                  │
                  │  100-300 words: "user logged in from a new        │
                  │  device, reset password, then transferred…"       │
                  └──────────────────────┬───────────────────────────┘
                                         │ tokenize
                                         ▼
       ┌─────────────────────────────────────────────────────────────┐
       │  Qwen3-8B (FROZEN: no gradient flows here)                  │
       │  36 transformer layers, hidden size 4096                    │
       │                                                             │
       │   layer_0 → layer_1 → ... → layer_11 → layer_12 ━━━┐         │
       │                                                    │         │
       │     ┌─────────────────────────────────────────────▼─┐       │
       │     │ ⊕  Gated cross-attn block                    │        │
       │     │    LM queries look at side-channel K/V       │        │
       │     │    Output is scaled by tanh(α), added back   │        │
       │     │    α starts at 0 (or 0.01) — block is ≈ no-op│        │
       │     └─────────────────────────────────────────────┘ │        │
       │                       │   ▲                                  │
       │                       ▼   │ (every N layers,                 │
       │   layer_13 → ... → layer_16 → ⊕ another x-attn block,        │
       │   ...                          per insertion_pattern)        │
       └─────────────────────────────────────────┬───────────────────┘
                                                 │
                                                 │ K/V slots
                                                 │ (the "side channel")
                                                 ▼
                              ┌──────────────────────────────┐
                              │ Perceiver-Resampler          │
                              │ Compresses N variable events │
                              │ to K fixed slots (K=64 or 128)│
                              │ Adds a time encoding         │
                              └──────────────┬───────────────┘
                                             ▲
                                             │ per-event vectors
                                             │ (one per event)
                              ┌──────────────┴───────────────┐
                              │ small_transformer encoder    │
                              │ 6 layers, 4 heads, H=4096    │
                              │ Reads the structured events  │
                              └──────────────┬───────────────┘
                                             ▲
                                             │
       ┌─────────────────────────────────────┴────────────────┐
       │ Structured event stream:                              │
       │ t=0  login         geo=international, ip_risk=high    │
       │ t=2  device_add    age=new                            │
       │ t=4  pw_reset      auth=password_only                 │
       │ t=7  txn           amount=high, recipient=new         │
       │ ... (5-200 events per session, with Δt timestamps)    │
       └───────────────────────────────────────────────────────┘
```

Read the diagram bottom-up:

1. **Events come in** as a list of (event-type, features, timestamp).
2. **An encoder turns each event into a vector** (`small_transformer`).
3. **A "resampler" compresses** the variable-length encoder output
   down to a fixed number of slots (K=64 or 128).
4. **Those slots become the side channel** — the K/V (keys + values)
   that cross-attention blocks pull from.
5. **Every few layers** in the LM stack, a cross-attention block reads
   from those slots and adds the result (scaled by a learnable gate)
   back into the LM's hidden state.
6. **The LM produces a verdict** at the end of its prompt; the loss is
   next-token cross-entropy on the text.

The next sections unpack each piece.

---

## Piece 1 — The frozen LM (Qwen3-8B)

| | |
|---|---|
| Model | Qwen3-8B (36 layers, hidden 4096, 32 attention heads) |
| Stage 0 (before this matters) | Light continued pretraining ("CPT-light") on the synthetic narratives, merged back into the base weights |
| Stage 1 (the cross-attn work) | The merged base is **frozen entirely** — no gradients flow through Qwen's original weights |

The point of freezing: we want to test **the conditioning mechanism**
(cross-attn) without confounding it with "the LM learned to do fraud
detection." If we let the LM update, we couldn't tell whether
improvements came from cross-attn or from the LM picking up patterns.

**Why merge Stage-0 first?** Stage-0 used a small adapter
(a "LoRA" — see glossary). If we stacked Stage-1's new adapter on top
of an unmerged Stage-0 adapter, gradients flowing back through the
old adapter would be confusing. Merging Stage-0 into the base before
Stage-1 starts gives Stage-1 a clean, frozen foundation.

---

## Piece 2 — The event encoder (`small_transformer`)

Code: `src/model/encoders/small_transformer.py`

| | |
|---|---|
| What it does | Turns a list of structured events into a list of vectors, one vector per event |
| How | A 6-layer, 4-head transformer encoder over event-token sequences |
| Input | Tokenized events: each event becomes an event-type token (`<event_txn>`) plus its bucketed-feature tokens (`<amount_bucket=high>`, `<geo_distance=international>`, etc.) |
| Output | `(batch, n_events, 4096)` — one 4096-wide vector per event |
| Hidden dim | **4096** — matched to Qwen's hidden dim on purpose, so the resampler doesn't need a projection layer |

Why a transformer (and not, say, an LSTM)? Because the encoder needs
to look across *all events in a session at once* to capture patterns
like "device_add at t=2 plus pw_reset at t=4 plus large_txn at t=7."
Self-attention does that naturally. Alternatives are discussed later.

---

## Piece 3 — The Perceiver-Resampler

Code: `src/model/resampler.py`

This is the trickiest piece, and it has a clean job description:
**take a variable number of event vectors and produce a fixed number of
slot vectors.**

Why "fixed number of slots"? Because cross-attention in the LM is
going to read from those slots at every cross-attn layer. If the
side-channel size varied every batch, the K/V cache logistics would
get unpleasant. K (slots) is a hyperparameter: we test 64 and 128.

How it works:

1. We declare **K learnable "latent" vectors** (K=64 or 128). Think of
   them as empty buckets the resampler will fill with summarized event
   signal.
2. Each latent does cross-attention against the encoder output: each
   bucket asks the events "what's relevant to me?" and pulls in
   whatever they offer.
3. A bit of self-attention between latents, an FFN — standard
   Perceiver-style block. Stack a few of these.
4. Output: `(batch, K, 4096)` — exactly K vectors, regardless of how
   many events came in.

There's one wrinkle: **events have timestamps**. A login at t=0 and a
txn at t=7 (seconds) shouldn't be treated as identical events at the
same position. We add a **sinusoidal time encoding** to each event's
vector before the resampler reads it — same idea as transformer
positional encodings but using continuous time (Δt) rather than
integer positions.

(There's a mixed-precision detail here: bf16's mantissa is too narrow
to keep sub-minute resolution at session-scale times, so the time
encoding is computed in fp32 then cast back to bf16. Review 006 finding
#2 caught the bug. Code comment explains.)

---

## Piece 4 — The gated cross-attention dense block

Code: `src/model/cross_attn_block.py`

This is the heart of the architecture. Every cross-attn block does
two things:

```
h  ←  h  +  tanh(α_attn) ·  out_proj( CrossAttn( pre_norm(h),  K/V slots ) )
h  ←  h  +  tanh(α_ffn)  ·  FFN( pre_norm(h) )
```

Translating to English:

- **`h`** is the LM's hidden state at this layer (one 4096-vector per
  token in the prompt).
- **CrossAttn** has `h`'s tokens ask questions about the K/V slots and
  get a pulled-from-slots answer.
- **`tanh(α_attn)`** is a single learnable scalar that *gates* how
  much of that answer flows back into `h`. If `α=0`, the gate is 0
  and the LM ignores cross-attn entirely. If `α=1`, the gate is
  `tanh(1) ≈ 0.76` and the answer is ~76% added back.
- **`out_proj`** is a linear layer that maps the cross-attn output
  back to Qwen's hidden dim (4096). The cross-attn itself runs at a
  smaller "bottleneck" dim of 1024 to keep parameter count down.
- Then the same gating story for a feed-forward (FFN) sub-block.

Key implementation choices, with the why:

| Choice | Why |
|---|---|
| **Pre-norm** (LayerNorm before Q and K/V) | Modern transformer convention; stabilizes training. |
| **Bottleneck `cross_dim = 1024`** (= H/4) | Keeps parameter count inside the 200-400M Stage-1 budget. Cross-attention at the full 4096 would be too expensive. |
| **`tanh(α)` scalar gate per block** | `tanh` bounds gate magnitude to (-1, 1). Scalar (not per-head, not per-channel) means there's one diagnostic number per block. |
| **`gate_init ∈ {zero, small_0.01}`** | Two options: gate starts at exactly 0 (block is initially a perfect no-op) or at 0.01 (block starts contributing slightly). The sweep tested both. |
| **`out_proj` uses *standard* init, not zero** | Tempting to zero-initialize both `α` and `out_proj` for extra safety. But that makes BOTH gradients vanish: `dL/dα` is proportional to the cross-attn output (which is 0 if `out_proj=0`), and `dL/dout_proj` is proportional to the gate (which is 0). The gate-zero arm would be permanently dead. Only the gate is zero-init; `out_proj` uses standard init. |

The gate magnitude `|tanh(α)|` is the headline diagnostic we logged
every training step. It's what the convergence/halt logic reads. It's
the number all the experiment-log tables call "max_gate."

---

## Piece 5 — LoRA-on-Q (the only learnable bit of Qwen)

The frozen LM has one tiny escape valve: a small **LoRA adapter** is
attached to Qwen's self-attention **query projections** (the "Q" in
"Q/K/V") in every layer. LoRA = Low-Rank Adaptation: a pair of small
matrices that get added to Qwen's existing weights, multiplying out to
a low-rank update.

| | |
|---|---|
| Where it lives | Self-attention Q projection of every Qwen layer |
| Rank | r=16 (most arms); we vary this in Phase 4 of the sweep |
| Why Q only (not K/V) | The LM's K/V are about the structural relationships in the embedding space — we don't want to disturb those. Q is about "what is each token asking for"; modulating Q lets the LM update *what it queries for* without changing the geometry of the embedding space. |
| Why it exists at all | Without it, the entirely-frozen Qwen would only have cross-attn to lean on — but the LM was never pretrained on fraud-detection tasks specifically. LoRA-on-Q is a tiny capacity for the LM to adapt its question-asking. |

LoRA-on-Q is the smallest trainable piece — ~5M params across all 36
Qwen layers, vs the encoder's ~200M and cross-attn's ~150M.

---

## Piece 6 — Where do we put the cross-attn blocks?

We have 36 Qwen layers and don't want to insert a cross-attn block
between every pair (too expensive). We tested three patterns:

| Pattern | Where the blocks go | How many | Idea |
|---|---|---|---|
| **`every_4`** | Layers 12, 16, 20, 24, 28, 32 | 6 blocks | Dense conditioning; spaced every 4 layers; starts at layer 12 |
| **`every_8`** | Layers 12, 20, 28 | 3 blocks | Sparser; ~2× cheaper |
| **`late_only`** | Layers 32, 33, 34, 35 | 4 blocks | All conditioning near the verdict — inject the signal right before the model produces its label |

Why start at layer 12 and not layer 0? Convention from Flamingo: early
layers do token-level work (which character / which morpheme), while
mid-to-late layers do the semantic/sequence-level work where extra
context is more useful. There's no first-principles proof of this;
it's a strong prior in the literature that we inherited.

---

## Putting the parameter count in perspective

| Component | Approximate trainable params |
|---|---|
| Frozen Qwen3-8B base | ~8.0B (no gradients) |
| Side-stream encoder (small_transformer 6×4) | ~200-300M |
| Perceiver-Resampler | ~80-120M |
| Gated cross-attn blocks (`every_4` = 6 blocks × ~25M) | ~150M |
| LoRA-on-Q r=16 across 36 layers | ~5M |
| **Stage-1 trainable total** | **~400-450M** for `every_4`, less for sparser patterns |

The 200-400M ceiling is a VRAM budget on one H100. At micro-batch=4 and
grad-accum=8 (effective batch 32), bf16 precision, this comfortably
fits.

---

## What the sweep actually told us

After 15 valid arms across Round 1 + Round 2 + Expanded Sweep Phases
1-3, the picture is consistent. Here's what each dial was supposed to
test, and what we got:

| Dial we moved | Expected behavior | What happened |
|---|---|---|
| `insertion_pattern` (every_4 / every_8 / late_only) | Denser conditioning → larger gates and better HN-FPR | All three within CIs of each other |
| `resampler_slots` (64 vs 128) | More slots → finer compression → better signal access | Inert; slots=128 cells slightly worse on point estimate |
| `gate_init` (small_0.01 vs zero) | Gates will learn to open; init shouldn't matter long-term | **Gates ride their init**: max_gate stays near init value |
| `lr` (3e-5 / 1e-4 / 3e-4) | Higher LR → more gate motion | LR-invariant in this range; lowest LR gave the *highest* max_gate |
| `warmup_steps` (100 vs 500) | Shorter warmup → gates open earlier | Inert |
| `seq_len` (2048 → 4096) and `steps` (1500 → 3000) | Long context might unlock signal | Mechanical failure (180m session boundary); patch deployed (F8 v2); retry possible |

The honest read: **the model doesn't want to use the side channel** on
this synthetic surface. Why might that be? Because the structured
events are *already* in the text — the bucketed-feature tokens
(`<amount_bucket=high>`, etc.) appear in the narrative version of each
session. The "high-bandwidth side channel" cross-attn is supposed to
provide isn't doing more than the text already does, because the text
already contains the same signal.

This is a real finding, even if it's a null one. It says: **the
information bottleneck on this surface is not the LM's access to event
features**; it's the eval set's intrinsic difficulty on
`hn_account_recovery`-style hard negatives. Architecture won't fix
that — better data will.

---

## Event-encoder alternatives we considered

The encoder (Piece 2 above) is the easiest part to swap. The POC
implemented one option and stubbed two more. Other options exist in
the literature but were ruled out for time / scope reasons.

### What's implemented and used in every sweep arm

**`small_transformer`** — pre-norm transformer encoder, 6 layers, 4
heads, hidden dim tied to Qwen's 4096.

- Pros: handles variable-length sequences, attends globally across
  events, fast to write.
- Cons: treats each bucketed-feature token as an opaque word — no
  explicit "this is a feature with a value" structure.

### Stubbed-but-deferred

The PLAN.md called these out as Day-4+ options. Empty placeholder
files exist:

**`ft_transformer.py`** — FT-Transformer (Gorishniy et al. 2021), a
tabular-foundation-model approach.

- Idea: each event is a vector of (feature, value) pairs. A per-feature
  embedding learns "amount_bucket high" as a distinct concept from
  "amount_bucket low" and from "geo_distance high."
- When this helps: when events have **many features with complex
  inter-feature interactions** that a vocabulary-style encoder can't
  capture. The synthetic generator's schema is small enough that this
  didn't seem to matter; on a real PayPal feature set it might.

**`cnn_lstm.py`** — Convolutional layer + LSTM stack.

- Idea: 1-D CNN captures **local n-gram patterns** in event order
  (`device_add → pw_reset → txn` within a 5-minute window). LSTM picks
  up longer dependencies.
- When this helps: when the fraud signal is in **specific temporal
  patterns** more than in the events themselves. Cheaper than a
  transformer.

### Not considered (and why)

These are real alternatives in the literature. We didn't run them.

| Architecture | Why we didn't pick it |
|---|---|
| **Set Transformer** (Lee et al. 2019) — permutation-invariant attention | Ignores event order. Our prior was that ordering matters for ATO (e.g., `device_change → pw_reset → large_txn` is a sequence pattern). Hard to walk that back. |
| **GRU-only / LSTM-only** (no transformer) | Smaller and cheaper, but no global view. Would force the resampler to do all the inter-event work. |
| **Mamba / S4 / state-space models** | Designed for very long sequences. Our event sequences cap at 200. Overkill, and a younger ecosystem with integration risk on a 3-day POC. |
| **No encoder, embedding lookup only** | Per-event embedding with no encoder. The resampler would have to do everything. Would be a useful null-model variant to isolate "is the encoder doing real work?" — but the POC's hypothesis was that the encoder does matter. |
| **Hierarchical event + session encoder** | Two-stage: per-event features → event embedding → cross-event transformer. Conceptually cleaner but doubles encoder complexity. Probably the right shape for real data. |
| **"Just put events in the text"** | That's the `structured_as_text` baseline already in the head-to-head. Not a cross-attn variant. |

### Dials we left on the table

If the sweep budget allowed (it doesn't — we're at 4 GPU-hours left),
the natural Day-4 expansions would be:

1. **Encoder swap**: run the same x-attn arm with FT-Tx, then with
   CNN+LSTM. Tells us whether the encoder is the bottleneck.
2. **Bigger encoder**: `n_layers=6 → 12`, `n_heads=4 → 8`. Tests
   whether the encoder is under-capacity.
3. **Deeper resampler**: more layers in the Perceiver-Resampler.
4. **Skip the resampler entirely**: cross-attn directly over the
   variable-length encoder output, no compression. Tests whether the
   K-slot compression is destroying information.

Prior: none of these are likely to overturn the current finding,
because the gates aren't opening regardless of architecture. But they'd
close the question.

---

## Why we picked this design (and not something simpler)

Three concrete reasons cross-attn was worth the engineering complexity
versus alternatives that would have given the LM access to the same
signal:

1. **It decouples context length from signal access.** The text stream
   stays at ~256-512 tokens. All event signal flows through K/V slots
   separately. If a real session has 200 events with rich features,
   that's still K=64 slots into the LM, not +200 tokens of context.

2. **The gates are an interpretable diagnostic.** "Did cross-attn
   matter?" is answerable: read the max_gate magnitude. A max_gate of
   0.01 means "the LM is not using cross-attn"; 0.5 means "the LM is
   leaning on cross-attn heavily." This is a much sharper signal than
   "is the loss lower?" and it's what told us — early — that the
   pathway is inert on this data.

3. **It's apples-to-apples with the `structured_as_text` baseline.**
   Both arms see the same `qwen3-8b-cpt-light-merged` frozen base. The
   only difference is *how* the events reach the LM: concatenated into
   text, or via cross-attn. That isolates the conditioning mechanism
   from everything else.

The first two were vindicated. The third is the load-bearing
comparison, and the answer (so far) is "the two mechanisms tie."

---

## Glossary

Terms used above, with one-line definitions and links to where they
appear in the code.

**Cross-attention** — The transformer attention operation where queries
come from one sequence and keys+values come from a different one. In
this POC, queries come from the LM's hidden state, keys+values come
from the event stream encoder output (via the resampler).

**Self-attention** — The standard transformer attention where Q, K, V
all come from the same sequence. Qwen's layers do self-attention on the
text stream; we don't touch this.

**Gate (`tanh(α)`)** — A learnable scalar that scales how much
cross-attn contributes to the LM's hidden state. `α=0 → gate=0 →
cross-attn is a no-op`. The gate's size is the headline diagnostic.

**Residual stream** — In a transformer, each layer doesn't replace the
hidden state; it adds to it. The running sum is the "residual stream."
Cross-attn blocks are residual-additive: they add their output to the
stream rather than replacing it.

**Flamingo-style** — The specific shape of frozen-base + gated
cross-attn introduced in the Flamingo paper (Alayrac et al. 2022,
DeepMind). Our cross-attn block is a direct lift of their "Gated
XATTN-DENSE" block.

**LoRA (Low-Rank Adaptation)** — A small adapter on top of a frozen
weight matrix. Instead of updating `W` directly, you train a low-rank
correction `BA` (with `rank(BA) = r`, small) and use `W + BA` at
inference. Cheap to train; restorable by setting `B=0` or merging
`BA` back into `W`.

**Stage-0 LoRA, Stage-1 LoRA** — Two separate LoRAs trained at
different times. Stage-0 was on Qwen's continued pretraining; merged
back into the base before Stage-1. Stage-1 is the cross-attn surgery's
own LoRA on Q projections, fresh.

**K/V slots** — The keys and values that cross-attention reads from.
In this POC, K/V are the output of the Perceiver-Resampler — a fixed
number (K=64 or 128) of vectors per session.

**Perceiver-Resampler** — A small module that takes a variable-length
sequence and compresses it to a fixed-length sequence using cross-
attention against learnable latent vectors. From the Perceiver paper
family (Jaegle et al. 2021).

**Pre-norm** — Convention where LayerNorm is applied *before* the
attention/FFN, not after. More stable in deep transformers.

**Bottleneck dim** — Cross-attention is computed at a smaller hidden
dim (1024) than the LM's hidden dim (4096), with linear projections
in and out. Saves parameters and compute.

**Sinusoidal time encoding** — Continuous-valued position encoding
where the "position" is the timestamp of an event (in seconds), not
its integer index. Uses sine/cosine basis functions at varying
frequencies; same shape as standard transformer positional encodings
but with continuous input.

**bf16 / fp32** — Numerical precisions. bf16 has a smaller mantissa
than fp32; sub-minute time resolution can be lost in cumulative-time
math if we're not careful (review 006 finding #2).

**HN-FPR-worst** — Hard-negative false positive rate at the worst
family. The headline metric: at a target legit-FPR of 1%, what fraction
of the hardest hard-negative family does the model flag as fraud?
Lower is better. Calculated with tie-aware exact-target interpolation
and 1000-resample bootstrap CI.

**`structured_as_text` baseline** — A baseline arm where the events
are serialized into the text prompt (`<events>` block before the
`<narrative>` block) and the LM trains on the concatenation. No cross-
attn. The load-bearing comparison against the x-attn arms.

---

## Code pointers

| File | What's in it |
|---|---|
| `src/model/cross_attn_block.py` (~400 lines) | The gated cross-attn block factory. The `tanh(α)` math, the bottleneck design, the init choices. |
| `src/model/resampler.py` (~325 lines) | The Perceiver-Resampler. K learnable latents, the sinusoidal time encoding, the fp32-then-cast-to-bf16 mixed-precision care. |
| `src/model/qwen_xattn_wrapper.py` (~600 lines) | The full assembly. Loads the frozen Qwen base, inserts cross-attn blocks per `insertion_pattern`, attaches the Stage-1 LoRA-on-Q, wires in the encoder and resampler. |
| `src/model/encoders/small_transformer.py` (~330 lines) | The side-stream encoder + `EventVocab` tokenizer. |
| `src/model/encoders/ft_transformer.py` | Empty stub (deferred). |
| `src/model/encoders/cnn_lstm.py` | Empty stub (deferred). |
| `src/train/train_xattn.py` | The Stage-1 trainer that puts it all together. |

Every non-obvious design choice has a `review NNN finding M` comment
next to it pointing to the review thread that justified it. Worth
grepping `review 006` if you want to see the gate-init / out_proj
vanishing-gradient discussion in detail.
