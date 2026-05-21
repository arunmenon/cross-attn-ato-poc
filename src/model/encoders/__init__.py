"""Event encoder registry for x-attn experiments."""

from __future__ import annotations

from .small_transformer import (
    EventVocab,
    FTTransformerEncoder,
    PooledMLPEncoder,
    SmallTransformerEncoder,
    tokenize_events,
)

ENCODER_REGISTRY = {
    "small_transformer": SmallTransformerEncoder,
    "pooled_mlp": PooledMLPEncoder,
    "ft_transformer": FTTransformerEncoder,
}


def available_encoders() -> tuple[str, ...]:
    return tuple(sorted(ENCODER_REGISTRY))


def build_event_encoder(
    name: str,
    *,
    hidden_dim: int = 256,
    n_heads: int = 4,
    n_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    vocab: EventVocab | None = None,
):
    try:
        factory = ENCODER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown event encoder {name!r}; allowed: {list(available_encoders())}"
        ) from exc
    return factory(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        vocab=vocab,
    )


__all__ = [
    "EventVocab",
    "SmallTransformerEncoder",
    "PooledMLPEncoder",
    "FTTransformerEncoder",
    "tokenize_events",
    "available_encoders",
    "build_event_encoder",
]
