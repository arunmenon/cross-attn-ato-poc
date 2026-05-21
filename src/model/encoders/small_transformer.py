"""Side-stream encoder for the cross-attention ATO POC.

Consumes the *structured* event stream (a list of event dicts produced by
data/gen/journey_templates) and produces a sequence of embeddings that
the Perceiver-Resampler downstream then compresses to fixed K/V slots
for cross-attention into Qwen3.

Design (per PLAN.md "Two distinct x-attn components", Architecture
section):

  - The encoder is the *analog of Flamingo's vision encoder* — but
    trained from scratch on synthetic ATO data, since no pretrained
    fraud-encoder exists to drop in.
  - The encoder is time-AGNOSTIC. Time encoding lives in the resampler
    (sinusoidal-on-Δt) per Pair-2 consensus / PLAN.md Architecture.
  - One token per event (event_type + bucket-token bag, combined
    via a small MLP). Sequence length = number of events.
  - Vocabulary is small (~40 tokens) so the encoder has its own
    embedding table, NOT shared with Qwen3's tokenizer.

Public API:
  - EventVocab — single source of truth for encoder token IDs.
  - tokenize_events(events: list[dict]) -> dict of int-id tensors.
  - SmallTransformerEncoder — nn.Module producing (B, N, hidden).

The Day-4 trainer wires `tokenize_events` into the dataloader collate
to produce batched tensors.
"""

from __future__ import annotations

import sys
from typing import Any

# Public constants — used by both encoder and tokenizer
EVENT_NAMES: list[str] = [
    "login", "txn", "pw_reset", "device_add",
    "recipient_add", "chat_to_support", "tool_call",
]

# Bucketed-feature tokens that may appear on events. Same as
# src.tokenizer.custom_tokens.BUCKET_TOKENS but kept inline here so the
# encoder does not depend on the Qwen-side tokenizer at all.
BUCKET_FAMILIES: dict[str, list[str]] = {
    "amount_bucket":   ["low", "medium", "high", "extreme"],
    "geo_distance":    ["local", "domestic_far", "international"],
    "ip_risk":         ["low", "medium", "high"],
    "device_age":      ["known", "new", "rare"],
    "merchant_risk":   ["normal", "elevated"],
    "txn_velocity":    ["normal", "bursty", "extreme"],
    "recipient_age":   ["known", "newly_added"],
    "session_dwell":   ["short", "normal", "extended"],
    "auth_strength":   ["mfa_strong", "password_only", "cookie_only"],
}


class EventVocab:
    """Encoder-side vocabulary. Maps event-type names and bucket-token
    strings to small contiguous integer IDs.

    Vocabulary layout (deterministic, do not change without bumping
    cache/dataset versions):

      Special tokens:
        0  = PAD     (used for padding short event sequences)
        1  = UNK     (unknown event type or bucket — should never fire
                      in practice; catches bugs)
        2  = ABSENT  (placeholder for "this event has no value for this
                      bucket family", e.g. login events have no
                      amount_bucket)

      Event types: indices 3 .. 3+|EVENT_NAMES|-1
      Bucket values, family-by-family: contiguous blocks after event types.
    """

    PAD_ID = 0
    UNK_ID = 1
    ABSENT_ID = 2

    def __init__(self) -> None:
        # Build event-type id map
        self.event_to_id: dict[str, int] = {}
        next_id = 3
        for name in EVENT_NAMES:
            self.event_to_id[name] = next_id
            next_id += 1

        # Per-family bucket-value id maps.
        # Each family has its own slot range so the encoder's embedding
        # of "amount_bucket=low" is distinct from "ip_risk=low".
        self.bucket_to_id: dict[str, dict[str, int]] = {}
        for family, values in BUCKET_FAMILIES.items():
            self.bucket_to_id[family] = {}
            for v in values:
                self.bucket_to_id[family][v] = next_id
                next_id += 1

        self.vocab_size = next_id

    def event_id(self, event_name: str) -> int:
        return self.event_to_id.get(event_name, self.UNK_ID)

    def bucket_id(self, family: str, value: str | None) -> int:
        if value is None:
            return self.ABSENT_ID
        family_map = self.bucket_to_id.get(family)
        if family_map is None:
            return self.UNK_ID
        return family_map.get(value, self.UNK_ID)


def _parse_bucket_string(raw: str) -> tuple[str, str] | None:
    """Parse `<amount_bucket=high>` -> ('amount_bucket', 'high'). Returns
    None on malformed input.
    """
    if not raw.startswith("<") or not raw.endswith(">"):
        return None
    inner = raw[1:-1]
    if "=" not in inner:
        return None
    family, value = inner.split("=", 1)
    return family, value


def tokenize_events(
    events: list[dict[str, Any]],
    vocab: EventVocab,
    max_events: int = 200,
) -> dict[str, list[int] | list[list[int]]]:
    """Convert a list of event dicts into integer-ID tensors (as Python
    lists; the trainer's collate will stack/pad and convert to torch).

    Returns a dict with three parallel arrays:
      - event_type_ids: list[int] of length max_events (PAD-padded)
      - bucket_ids:     list[list[int]] of shape (max_events, N_FAMILIES),
                        each row is the bucket value IDs for that event,
                        ABSENT_ID where the family is not present on the
                        event.
      - delta_t:        list[float] of length max_events, seconds since
                        the previous event in this stream (0 for the
                        first event, 0 for padding positions). Used by
                        the resampler.
      - attention_mask: list[int] of length max_events, 1 for real events,
                        0 for padding.
    """
    families = list(BUCKET_FAMILIES.keys())

    event_type_ids: list[int] = []
    bucket_ids: list[list[int]] = []
    delta_t: list[float] = []
    attention_mask: list[int] = []

    prev_t: int | None = None
    for ev in events[:max_events]:
        event_type_ids.append(vocab.event_id(ev.get("event", "")))

        row = []
        for family in families:
            raw = ev.get(family)
            if raw is None:
                row.append(vocab.ABSENT_ID)
                continue
            # Bucket-token strings look like "<amount_bucket=high>"; the
            # event dict stores the FULL token string, so parse it.
            if isinstance(raw, str) and raw.startswith("<") and raw.endswith(">"):
                parsed = _parse_bucket_string(raw)
                if parsed is None:
                    row.append(vocab.UNK_ID)
                else:
                    _, val = parsed
                    row.append(vocab.bucket_id(family, val))
            else:
                # Defensive: shouldn't happen, but covers raw-value events
                row.append(vocab.UNK_ID)
        bucket_ids.append(row)

        t = ev.get("t", 0)
        delta_t.append(0.0 if prev_t is None else float(t - prev_t))
        prev_t = t
        attention_mask.append(1)

    # Pad to max_events
    while len(event_type_ids) < max_events:
        event_type_ids.append(vocab.PAD_ID)
        bucket_ids.append([vocab.PAD_ID] * len(families))
        delta_t.append(0.0)
        attention_mask.append(0)

    return {
        "event_type_ids": event_type_ids,
        "bucket_ids": bucket_ids,
        "delta_t": delta_t,
        "attention_mask": attention_mask,
    }


# ---------------------------------------------------------------------------
# Encoder module (PyTorch — runs on the pod)
# ---------------------------------------------------------------------------

def _build_encoder_module(hidden_dim: int, n_heads: int, n_layers: int,
                         dim_feedforward: int, dropout: float, vocab_size: int,
                         n_bucket_families: int):
    """Inner factory — imported lazily so this module can be imported on
    laptops without torch installed (only the vocab + tokenize_events are
    needed off-GPU).
    """
    import torch
    import torch.nn as nn

    class _EventEmbedder(nn.Module):
        """Combines event-type embedding + bucket-token bag into one
        embedding per event."""

        def __init__(self) -> None:
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=EventVocab.PAD_ID)
            # MLP combines (event_type_emb + summed_bucket_embs) -> hidden_dim
            self.combine = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, event_type_ids, bucket_ids):
            # event_type_ids: (B, N), bucket_ids: (B, N, F)
            et = self.token_emb(event_type_ids)                       # (B, N, H)
            bk = self.token_emb(bucket_ids).sum(dim=2)                # (B, N, H)
            x = torch.cat([et, bk], dim=-1)                            # (B, N, 2H)
            return self.combine(x)                                     # (B, N, H)

    class _SmallTransformerEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedder = _EventEmbedder()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        def forward(self, event_type_ids, bucket_ids, attention_mask):
            # attention_mask: (B, N) with 1 for real, 0 for pad
            x = self.embedder(event_type_ids, bucket_ids)              # (B, N, H)
            # PyTorch's src_key_padding_mask expects True where PAD.
            key_pad = ~attention_mask.bool()
            out = self.encoder(x, src_key_padding_mask=key_pad)        # (B, N, H)
            return out * attention_mask.to(dtype=out.dtype).unsqueeze(-1)

    return _SmallTransformerEncoder()


def _build_pooled_mlp_module(hidden_dim: int, dropout: float, vocab_size: int):
    """Cheap event encoder: per-event MLP plus masked global context."""
    import torch
    import torch.nn as nn

    class _PooledMLPEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=EventVocab.PAD_ID)
            self.event_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.context_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, event_type_ids, bucket_ids, attention_mask):
            et = self.token_emb(event_type_ids)
            bk = self.token_emb(bucket_ids).sum(dim=2)
            x = self.event_mlp(torch.cat([et, bk], dim=-1))
            mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (x * mask).sum(dim=1) / denom
            context = self.context_mlp(pooled).unsqueeze(1)
            return (x + context) * mask

    return _PooledMLPEncoder()


def _build_ft_transformer_module(
    hidden_dim: int,
    n_heads: int,
    n_layers: int,
    dim_feedforward: int,
    dropout: float,
    vocab_size: int,
    n_bucket_families: int,
):
    """Feature-token transformer.

    Each event becomes F+1 feature tokens: event type plus one token per
    bucket family. A small transformer mixes those feature tokens inside
    each event, then pooled feature states form the event embedding.
    """
    import torch
    import torch.nn as nn

    class _FTTransformerEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_features = n_bucket_families + 1
            self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=EventVocab.PAD_ID)
            self.feature_pos = nn.Parameter(torch.zeros(1, self.n_features, hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.feature_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.out_norm = nn.LayerNorm(hidden_dim)

        def forward(self, event_type_ids, bucket_ids, attention_mask):
            bsz, n_events = event_type_ids.shape
            feature_ids = torch.cat([event_type_ids.unsqueeze(-1), bucket_ids], dim=-1)
            x = self.token_emb(feature_ids) + self.feature_pos.to(dtype=self.token_emb.weight.dtype)
            x = x.reshape(bsz * n_events, self.n_features, -1)
            x = self.feature_encoder(x)
            x = x.mean(dim=1).reshape(bsz, n_events, -1)
            x = self.out_norm(x)
            return x * attention_mask.to(dtype=x.dtype).unsqueeze(-1)

    return _FTTransformerEncoder()


def SmallTransformerEncoder(
    *,
    hidden_dim: int = 256,
    n_heads: int = 4,
    n_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    vocab: EventVocab | None = None,
):
    """Factory for the encoder module. Lazy-imports torch so the module
    file is import-safe on laptops without torch installed.
    """
    vocab = vocab or EventVocab()
    n_families = len(BUCKET_FAMILIES)
    return _build_encoder_module(
        hidden_dim=hidden_dim, n_heads=n_heads, n_layers=n_layers,
        dim_feedforward=dim_feedforward, dropout=dropout,
        vocab_size=vocab.vocab_size, n_bucket_families=n_families,
    )


def PooledMLPEncoder(
    *,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    vocab: EventVocab | None = None,
    **_: object,
):
    """Factory for the cheap feature-only pooled MLP encoder."""
    vocab = vocab or EventVocab()
    return _build_pooled_mlp_module(
        hidden_dim=hidden_dim,
        dropout=dropout,
        vocab_size=vocab.vocab_size,
    )


def FTTransformerEncoder(
    *,
    hidden_dim: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    vocab: EventVocab | None = None,
    **_: object,
):
    """Factory for the feature-token-aware event encoder."""
    vocab = vocab or EventVocab()
    return _build_ft_transformer_module(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        vocab_size=vocab.vocab_size,
        n_bucket_families=len(BUCKET_FAMILIES),
    )


# ---------------------------------------------------------------------------
# Self-test (requires torch; skipped gracefully otherwise)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # Vocab + tokenize_events do not require torch — always test them.
    vocab = EventVocab()
    print(f"event vocab size: {vocab.vocab_size}")
    assert vocab.event_id("login") != vocab.UNK_ID
    assert vocab.bucket_id("amount_bucket", "high") != vocab.UNK_ID
    assert vocab.bucket_id("amount_bucket", "ginormous") == vocab.UNK_ID  # unknown value
    assert vocab.bucket_id("nonsense_family", "anything") == vocab.UNK_ID
    assert vocab.bucket_id("amount_bucket", None) == vocab.ABSENT_ID

    sample_events = [
        {"t": 0, "event": "login", "ip_risk": "<ip_risk=low>",
         "geo_distance": "<geo_distance=local>",
         "auth_strength": "<auth_strength=mfa_strong>"},
        {"t": 60, "event": "txn", "amount_bucket": "<amount_bucket=high>",
         "txn_velocity": "<txn_velocity=normal>"},
        {"t": 120, "event": "pw_reset", "auth_strength": "<auth_strength=password_only>"},
    ]
    toks = tokenize_events(sample_events, vocab, max_events=8)
    assert len(toks["event_type_ids"]) == 8
    assert toks["event_type_ids"][0] == vocab.event_id("login")
    assert toks["event_type_ids"][3] == vocab.PAD_ID, "pad position should be PAD_ID"
    assert toks["attention_mask"] == [1, 1, 1, 0, 0, 0, 0, 0]
    # Δt: first is 0, then 60, then 60
    assert toks["delta_t"][0] == 0.0
    assert toks["delta_t"][1] == 60.0
    assert toks["delta_t"][2] == 60.0
    print("vocab + tokenize_events OK")

    # PyTorch-dependent path
    try:
        import torch
    except ImportError:
        print("torch not installed; encoder forward-pass test skipped "
              "(runs on the pod)")
        return

    et = torch.tensor([toks["event_type_ids"]])               # (1, 8)
    bk = torch.tensor([toks["bucket_ids"]])                   # (1, 8, F)
    am = torch.tensor([toks["attention_mask"]])               # (1, 8)
    for name, factory in {
        "small_transformer": SmallTransformerEncoder,
        "pooled_mlp": PooledMLPEncoder,
        "ft_transformer": FTTransformerEncoder,
    }.items():
        enc = factory(hidden_dim=64, n_heads=4, n_layers=2)
        out = enc(et, bk, am)
        assert out.shape == (1, 8, 64), f"{name}: expected (1, 8, 64), got {out.shape}"
        assert torch.isfinite(out).all(), f"{name}: output contains NaN/Inf"
        assert torch.allclose(out[:, 3:], torch.zeros_like(out[:, 3:])), (
            f"{name}: padded positions should be zeroed"
        )
        print(f"{name} forward OK; output shape {tuple(out.shape)}")


if __name__ == "__main__":
    _self_test()
