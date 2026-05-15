"""Shared types for the data-generation pipeline.

Centralized so journey_templates / agent_actor_mixer / narrative_generator /
build_dataset can agree on the shape of a single training example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from data.gen.pii_fencer import MetadataKeeper

Label = Literal["fraud", "legit"]


@dataclass
class Journey:
    """One synthetic ATO session, ready for narrative attachment + serialization.

    Attributes:
        events:           list of fenced event dicts (output of
                          pii_fencer.fenced_event_dict + bucket-token
                          attachment by journey_templates).
        journey_family:   one of JOURNEY_FAMILIES (clean, cred_stuff,
                          sim_swap, phish_takeover, malware_rat,
                          mule_chain, hn_travel, hn_large_purchase,
                          hn_account_recovery).
        actor_family:     one of ACTOR_FAMILIES (human, agent_buying,
                          agent_finance, agent_compromised,
                          agent_adversarial, hybrid).
        label:            "fraud" | "legit". Derived from journey_family
                          by journey_templates.
        is_hard_negative: True iff journey_family starts with "hn_".
        metadata:         MetadataKeeper holding the actual synthetic
                          values for IDs/IPs/amounts. NOT part of the
                          training corpus; only used for downstream eval
                          / debugging joins.
        narrative:        the LLM-generated or templated narrative text
                          (body only; the verdict footer is appended
                          separately by the serializer).
        seed:             the rng seed used to generate this journey,
                          for reproducibility.
    """
    events: list[dict[str, Any]]
    journey_family: str
    actor_family: str
    label: Label
    is_hard_negative: bool
    metadata: MetadataKeeper = field(default_factory=MetadataKeeper)
    narrative: str | None = None
    seed: int | None = None

    def session_summary(self) -> dict[str, Any]:
        """Extract a small dict useful for templated narrative generation."""
        return {
            "n_events": len(self.events),
            "first_event": self.events[0]["event"] if self.events else None,
            "last_event": self.events[-1]["event"] if self.events else None,
            "duration_seconds": (
                self.events[-1]["t"] - self.events[0]["t"]
                if len(self.events) >= 2 else 0
            ),
            "journey_family": self.journey_family,
            "actor_family": self.actor_family,
        }


__all__ = ["Journey", "Label"]
