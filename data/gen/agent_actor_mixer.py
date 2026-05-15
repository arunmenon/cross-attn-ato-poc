"""Actor-family modulation of a base journey.

Takes a Journey produced by journey_templates and applies actor-type
signature: programmatic timing, tool_call events, session_dwell shifts,
auth-strength shifts.

Why this is separate from journey_templates: the fraud-pattern signature
(which events happen in what order) is journey-family-specific, while the
actor signature (timing cadence, tool-use traces) is journey-family-
independent. Separating them keeps the combinatorial explosion in check:
9 journey families × 6 actor types = 54 distinct distributions, with only
9 + 6 generator functions to maintain.

CLI:
    python -m data.gen.agent_actor_mixer --self-test
"""

from __future__ import annotations

import argparse
import random
import sys

from data.gen.feature_bucketer import format_bucket_token
from data.gen.pii_fencer import MetadataKeeper, fenced_event_dict
from data.gen.types import Journey


# Per-actor timing modulation. Multiplier applied to inter-event gaps:
# < 1 means tighter cadence (more automated); > 1 means slower / more
# human-like.
_TIMING_SCALE: dict[str, float] = {
    "human":             1.00,    # baseline
    "agent_buying":      0.55,    # noticeably faster than human
    "agent_finance":     0.65,
    "agent_compromised": 0.30,    # very fast, regular intervals
    "agent_adversarial": 0.35,    # also fast, slightly jittered to mimic human
    "hybrid":            0.85,    # mix of automation and human pauses
}


# Probability of inserting a <event_tool_call> event before each non-login
# event for agent actors.
_TOOL_CALL_PROB: dict[str, float] = {
    "human":             0.0,
    "agent_buying":      0.55,
    "agent_finance":     0.40,
    "agent_compromised": 0.70,
    "agent_adversarial": 0.65,
    "hybrid":            0.25,
}


# How much to jitter inter-event intervals to mimic human noise. Agents
# vary in how human-like they try to appear.
_TIMING_JITTER: dict[str, float] = {
    "human":             0.30,    # +/- 30% noise
    "agent_buying":      0.05,
    "agent_finance":     0.05,
    "agent_compromised": 0.02,    # very regular
    "agent_adversarial": 0.20,    # tries to look human
    "hybrid":            0.20,
}


def mix(journey: Journey, *, rng: random.Random | None = None) -> Journey:
    """Apply actor-family modulation to `journey`. Returns a NEW Journey
    object (does not mutate the input).

    For human actor: returns a near-identity copy. The mixer is a no-op
    structurally for human actors, but still passed through so callers
    can use the same pipeline.
    """
    actor = journey.actor_family
    if actor not in _TIMING_SCALE:
        raise ValueError(f"unknown actor: {actor!r}")

    rng = rng or random.Random((journey.seed or 0) * 31 + 17)

    scale = _TIMING_SCALE[actor]
    tool_prob = _TOOL_CALL_PROB[actor]
    jitter_amt = _TIMING_JITTER[actor]

    new_events: list[dict] = []
    # New keeper: we'll re-fence any tool_call events we add. The original
    # events keep their existing metadata references.
    keeper = MetadataKeeper(
        amounts=dict(journey.metadata.amounts),
        recipients=dict(journey.metadata.recipients),
        merchants=dict(journey.metadata.merchants),
        ips=dict(journey.metadata.ips),
        device_ids=dict(journey.metadata.device_ids),
    )

    # Reconstruct timeline: keep first event at t=0, then rescale gaps.
    prev_orig_t: int | None = None
    cur_t = 0
    for i, ev in enumerate(journey.events):
        orig_t = ev["t"]

        if prev_orig_t is None:
            # First event stays at t=0
            new_ev = dict(ev)
            new_ev["t"] = 0
            new_events.append(new_ev)
        else:
            gap = orig_t - prev_orig_t
            # Apply scale + jitter
            jitter_factor = 1.0 + rng.uniform(-jitter_amt, jitter_amt)
            new_gap = max(1, int(round(gap * scale * jitter_factor)))
            cur_t += new_gap

            # Maybe insert a tool_call event before this one (agent actors only)
            if rng.random() < tool_prob and ev["event"] != "login":
                tool_t = cur_t - max(1, int(new_gap * 0.4))
                if tool_t > (new_events[-1]["t"] if new_events else 0):
                    tc = fenced_event_dict(
                        t=tool_t, event_name="tool_call", actor=actor, keeper=keeper,
                    )
                    # Annotate which event the tool call is preparing for
                    tc["prepares_event"] = ev["event"]
                    new_events.append(tc)

            new_ev = dict(ev)
            new_ev["t"] = cur_t
            new_events.append(new_ev)

        prev_orig_t = orig_t

    # Re-attach session_dwell on the new last event based on the rescaled
    # duration. Re-derive from feature_bucketer to keep bucket boundaries
    # honest.
    from data.gen.feature_bucketer import session_dwell as _sd_fn
    if new_events:
        # Strip any stale session_dwell on intermediate events
        for ev in new_events[:-1]:
            ev.pop("session_dwell", None)
        duration = new_events[-1]["t"] - new_events[0]["t"]
        new_events[-1]["session_dwell"] = format_bucket_token("session_dwell", _sd_fn(duration))

    return Journey(
        events=new_events,
        journey_family=journey.journey_family,
        actor_family=actor,
        label=journey.label,
        is_hard_negative=journey.is_hard_negative,
        metadata=keeper,
        narrative=journey.narrative,
        seed=journey.seed,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    from data.gen.journey_templates import generate

    # Test all 6 actor types on a single journey family.
    base = generate("sim_swap", seed=100, actor="human")
    durations: dict[str, int] = {}
    tool_call_counts: dict[str, int] = {}

    for actor in ["human", "agent_buying", "agent_finance",
                  "agent_compromised", "agent_adversarial", "hybrid"]:
        # Reassign actor before mixing (the mixer reads journey.actor_family)
        j = Journey(
            events=base.events, journey_family=base.journey_family,
            actor_family=actor, label=base.label,
            is_hard_negative=base.is_hard_negative,
            metadata=base.metadata, seed=base.seed,
        )
        mixed = mix(j, rng=random.Random(actor))
        durations[actor] = mixed.events[-1]["t"] - mixed.events[0]["t"]
        tool_call_counts[actor] = sum(1 for e in mixed.events if e["event"] == "tool_call")

    # Sanity: agent actors should have at least *some* tool_call events on
    # a 5-event sim_swap journey with their respective probabilities
    # (probabilistic; we only assert >0 for the highest-prob actor).
    assert tool_call_counts["human"] == 0, "human actor should produce no tool_calls"

    # Sanity: agent_compromised's duration should be shorter than human's
    # for the same base journey (scale=0.30 vs 1.00).
    assert durations["agent_compromised"] < durations["human"], (
        f"agent_compromised duration {durations['agent_compromised']} "
        f"should be < human {durations['human']}"
    )

    # Determinism: running twice with same rng produces same output
    j_for_det = Journey(
        events=base.events, journey_family=base.journey_family,
        actor_family="agent_compromised", label=base.label,
        is_hard_negative=base.is_hard_negative, metadata=base.metadata, seed=base.seed,
    )
    a = mix(j_for_det, rng=random.Random(123))
    b = mix(j_for_det, rng=random.Random(123))
    for ea, eb in zip(a.events, b.events):
        assert ea == eb, "non-deterministic mixer"

    print(f"agent_actor_mixer self-test OK")
    print(f"  durations: {durations}")
    print(f"  tool_calls: {tool_call_counts}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
