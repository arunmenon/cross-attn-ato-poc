"""Synthetic-data-side PII fencing.

When the journey generator (journey_templates.py + agent_actor_mixer.py)
emits an event, raw identifier-valued fields (acct_id, email, ip, etc.)
are replaced with their fenced placeholders BEFORE the event is added to
the training corpus. This module owns the policy.

Distinction from src/tokenizer/fencer.py:
  - src/tokenizer/fencer.py:    regex-based scrub of LLM-generated NARRATIVE
                                text (defense-in-depth against the narrator
                                accidentally producing literal-looking PII).
  - data/gen/pii_fencer.py:     emits fenced placeholders directly during
                                synthetic event construction. There are
                                never any real identifiers in the structured
                                stream — they live only in metadata for
                                downstream eval/debugging.

Usage:
    from data.gen.pii_fencer import fenced_event_dict, MetadataKeeper

    keeper = MetadataKeeper()
    ev = fenced_event_dict(
        t=42,
        event_name="txn",
        actor="agent_compromised",
        keeper=keeper,
        # any raw fields get the keeper:
        amount_usd=1247.50,
        recipient_seed=12345,
        merchant_seed=678,
    )
    # ev["acct_id"] == "<acct_id>"     (token, not the real value)
    # ev["amount_bucket"] == "<amount_bucket=high>"  (bucket-token; signal)
    # keeper.amounts[42] == 1247.50    (real value, only in metadata)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetadataKeeper:
    """Stores the actual synthetic values that were fenced, so downstream
    eval / debug can join back to ground truth without polluting the
    training corpus.
    """
    amounts: dict[int, float] = field(default_factory=dict)        # t -> usd
    recipients: dict[int, str] = field(default_factory=dict)       # t -> recipient_id
    merchants: dict[int, str] = field(default_factory=dict)        # t -> merchant_id
    ips: dict[int, str] = field(default_factory=dict)              # t -> ip
    device_ids: dict[int, str] = field(default_factory=dict)       # t -> device fingerprint


def _deterministic_synth_id(prefix: str, seed: int) -> str:
    """Produce a stable but obviously-synthetic ID for metadata use.

    Hash-based so the same seed always produces the same id; downstream
    eval can rely on this for joins.
    """
    h = hashlib.sha256(f"{prefix}::{seed}".encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


def _synth_ip(seed: int) -> str:
    """Produce a synthetic-looking IPv4 from a seed (deterministic)."""
    h = hashlib.sha256(f"ip::{seed}".encode()).digest()
    return f"{h[0]}.{h[1]}.{h[2]}.{h[3]}"


def fenced_event_dict(
    *,
    t: int,
    event_name: str,
    actor: str,
    keeper: MetadataKeeper,
    amount_usd: float | None = None,
    recipient_seed: int | None = None,
    merchant_seed: int | None = None,
    ip_seed: int | None = None,
    device_seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a single event with PII fenced + relevant metadata stored.

    Returns a dict suitable for the structured event stream (and for the
    compact structured-as-text serialization). All identifier values are
    replaced with fence tokens; bucketed-feature derivations happen in the
    caller (journey_templates.py).

    The `extra` dict is merged in as-is (no fencing applied) — caller is
    responsible for not putting raw PII there.
    """
    ev: dict[str, Any] = {
        "t": t,
        "event": event_name,
        "actor": actor,
    }

    if amount_usd is not None:
        keeper.amounts[t] = amount_usd
        # No raw amount in the event; the caller will attach
        # amount_bucket separately via feature_bucketer.

    if recipient_seed is not None:
        keeper.recipients[t] = _deterministic_synth_id("recipient", recipient_seed)
        ev["recipient"] = "<recipient>"

    if merchant_seed is not None:
        keeper.merchants[t] = _deterministic_synth_id("merchant", merchant_seed)
        ev["merchant"] = "<merchant>"

    if ip_seed is not None:
        keeper.ips[t] = _synth_ip(ip_seed)
        ev["ip"] = "<ip>"

    if device_seed is not None:
        keeper.device_ids[t] = _deterministic_synth_id("device", device_seed)
        ev["device_id"] = "<device_id>"

    if extra:
        ev.update(extra)

    return ev


def assert_no_raw_pii_in_event(ev: dict[str, Any]) -> None:
    """Defensive check: an event ready for the training corpus must not
    contain any raw identifier-looking values. Called by build_dataset.py
    on every event before writing.
    """
    forbidden_keys = ("recipient_real", "merchant_real", "ip_real",
                      "device_id_real", "email", "phone", "acct_id_real")
    for k in forbidden_keys:
        if k in ev:
            raise AssertionError(
                f"event at t={ev.get('t')} contains raw PII key {k!r}; "
                f"use MetadataKeeper instead"
            )

    # Spot-check string values for obvious leaks
    for key, val in ev.items():
        if not isinstance(val, str):
            continue
        if "@" in val and "<" not in val:
            raise AssertionError(f"event has bare email-like value at {key}: {val!r}")
        if val.count(".") == 3 and any(c.isdigit() for c in val):
            # crude IPv4 detection
            if "<" not in val:
                raise AssertionError(f"event has bare IPv4-like value at {key}: {val!r}")


def _self_test() -> None:
    keeper = MetadataKeeper()
    ev = fenced_event_dict(
        t=10,
        event_name="txn",
        actor="agent_compromised",
        keeper=keeper,
        amount_usd=1247.50,
        recipient_seed=42,
        merchant_seed=99,
        ip_seed=7,
        device_seed=3,
    )
    assert ev["t"] == 10
    assert ev["event"] == "txn"
    assert ev["recipient"] == "<recipient>"
    assert ev["merchant"] == "<merchant>"
    assert ev["ip"] == "<ip>"
    assert ev["device_id"] == "<device_id>"
    assert keeper.amounts[10] == 1247.50
    assert keeper.recipients[10].startswith("recipient_")
    assert_no_raw_pii_in_event(ev)

    # Negative case
    bad = dict(ev)
    bad["email"] = "alice@evil.com"
    try:
        assert_no_raw_pii_in_event(bad)
        raise AssertionError("expected AssertionError")
    except AssertionError as e:
        assert "raw PII" in str(e) or "email" in str(e)

    print("pii_fencer self-test OK")


if __name__ == "__main__":
    _self_test()
