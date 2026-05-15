"""Eval-mode text transforms.

Three modes for the fraud-classification eval surface:

- full:     all tokens visible (debug only — never headline)
- opaque:   journey_* and actor_* tokens replaced with neutral IDs
            **per-example randomized mapping** so the model cannot learn
            ID->name across examples; within an example, open/close share
            an ID so structural (boundary) signal is preserved.
- stripped: journey_* and actor_* tokens removed entirely.

Bucketed-feature tokens (amount_bucket, geo_distance, etc.) and PII-fence
tokens (acct_id, ip, etc.) are NEVER touched by these transforms — they're
the signal and the hygiene boundary, not the label.

Pure-function module. No model deps. Regex-only.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Literal

Mode = Literal["full", "opaque", "stripped"]

# Tokens that get stripped/opacified in non-full modes.
JOURNEY_OPEN = re.compile(r"<journey_([a-z_]+)>")
JOURNEY_CLOSE = re.compile(r"</journey_([a-z_]+)>")
ACTOR = re.compile(r"<actor_([a-z_]+)>")

# Maximum number of distinct journey/actor families per example. Used to size
# the per-example random permutation pool. Real data has <20 each, 99 is slack.
_MAX_FAMILIES = 99


@dataclass(frozen=True)
class TransformReport:
    """What got changed. Used by leakage_checks.verify_strip()."""
    mode: Mode
    journey_open_hits: int
    journey_close_hits: int
    actor_hits: int


def apply(text: str, mode: Mode, rng: random.Random | None = None) -> tuple[str, TransformReport]:
    """Apply the eval-mode transform. Returns (new_text, report).

    For `opaque` mode, a per-call random permutation is used. Pass `rng` for
    reproducibility (e.g., a per-example seed during training); without it,
    a fresh non-deterministic permutation is sampled.

    Within a single call, open/close tags for the same journey family share
    the same opaque ID — so the model can still detect journey boundaries,
    just not decode WHICH family it is. This addresses review finding #6:
    the previous stable mapping leaked the name via training-time dropout.

    Bucketed-feature tokens and PII-fence tokens are intentionally untouched.
    """
    if mode == "full":
        return text, TransformReport("full", 0, 0, 0)

    if mode == "stripped":
        new, jo = JOURNEY_OPEN.subn("", text)
        new, jc = JOURNEY_CLOSE.subn("", new)
        new, ah = ACTOR.subn("", new)
        # Collapse double-spaces / orphan whitespace created by substitution.
        new = re.sub(r"[ \t]{2,}", " ", new)
        new = re.sub(r"\n{3,}", "\n\n", new)
        return new, TransformReport("stripped", jo, jc, ah)

    if mode == "opaque":
        rng = rng or random.Random()

        # Per-call random permutation of two-digit IDs.
        journey_pool = [f"{i:02d}" for i in range(1, _MAX_FAMILIES + 1)]
        actor_pool = [f"{i:02d}" for i in range(1, _MAX_FAMILIES + 1)]
        rng.shuffle(journey_pool)
        rng.shuffle(actor_pool)

        # On-the-fly per-family assignment. Same name -> same ID within this call.
        journey_map: dict[str, str] = {}
        actor_map: dict[str, str] = {}

        def _get_journey_id(name: str) -> str:
            if name not in journey_map:
                if not journey_pool:
                    raise RuntimeError(f"opaque pool exhausted on journey {name!r}")
                journey_map[name] = journey_pool.pop()
            return journey_map[name]

        def _get_actor_id(name: str) -> str:
            if name not in actor_map:
                if not actor_pool:
                    raise RuntimeError(f"opaque pool exhausted on actor {name!r}")
                actor_map[name] = actor_pool.pop()
            return actor_map[name]

        def _journey_open_sub(m: re.Match) -> str:
            return f"<journey_type_{_get_journey_id(m.group(1))}>"

        def _journey_close_sub(m: re.Match) -> str:
            return f"</journey_type_{_get_journey_id(m.group(1))}>"

        def _actor_sub(m: re.Match) -> str:
            return f"<actor_type_{_get_actor_id(m.group(1))}>"

        new, jo = JOURNEY_OPEN.subn(_journey_open_sub, text)
        new, jc = JOURNEY_CLOSE.subn(_journey_close_sub, new)
        new, ah = ACTOR.subn(_actor_sub, new)
        return new, TransformReport("opaque", jo, jc, ah)

    raise ValueError(f"unknown mode: {mode!r}")


__all__ = ["Mode", "TransformReport", "apply"]
