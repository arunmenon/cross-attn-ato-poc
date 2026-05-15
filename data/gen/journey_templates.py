"""Per-journey-family generators.

Each generator function takes (seed, actor_family) and returns a Journey
with `events` populated with fenced + bucket-tokenized event dicts. The
journey family's signature lives in the event ordering, inter-event
timing, and bucketed-feature distributions.

Design notes:
  - Every event has bucket tokens for the features that meaningfully
    distinguish *this* event type. (We don't attach all 9 bucket families
    to every event; that would be noise.)
  - Inter-event timing is in seconds; agent_actor_mixer may rescale.
  - Each generator is deterministic given (seed, actor_family).

CLI:
    python -m data.gen.journey_templates --self-test
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Callable

from data.gen.feature_bucketer import (
    amount_bucket, format_bucket_token, geo_distance, ip_risk, device_age,
    merchant_risk, txn_velocity, recipient_age, session_dwell, auth_strength,
)
from data.gen.pii_fencer import MetadataKeeper, fenced_event_dict
from data.gen.types import Journey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bt(family: str, value: str) -> str:
    """Short alias for format_bucket_token."""
    return format_bucket_token(family, value)


def _login_event(
    t: int,
    actor: str,
    keeper: MetadataKeeper,
    *,
    rng: random.Random,
    ip_risk_value: str = "low",
    geo_distance_value: str = "local",
    auth: str = "password_only",
    device_age_value: str = "known",
) -> dict:
    ev = fenced_event_dict(
        t=t, event_name="login", actor=actor, keeper=keeper,
        ip_seed=rng.randint(0, 10**9),
        device_seed=rng.randint(0, 10**9),
    )
    ev["ip_risk"] = _bt("ip_risk", ip_risk_value)
    ev["geo_distance"] = _bt("geo_distance", geo_distance_value)
    ev["auth_strength"] = _bt("auth_strength", auth)
    ev["device_age"] = _bt("device_age", device_age_value)
    return ev


def _txn_event(
    t: int,
    actor: str,
    keeper: MetadataKeeper,
    *,
    rng: random.Random,
    amount_usd: float,
    velocity: str = "normal",
    rcpt_age: str = "known",
    merch_risk: str = "normal",
) -> dict:
    ev = fenced_event_dict(
        t=t, event_name="txn", actor=actor, keeper=keeper,
        amount_usd=amount_usd,
        recipient_seed=rng.randint(0, 10**9),
        merchant_seed=rng.randint(0, 10**9),
    )
    ev["amount_bucket"] = _bt("amount_bucket", amount_bucket(amount_usd))
    ev["txn_velocity"] = _bt("txn_velocity", velocity)
    ev["recipient_age"] = _bt("recipient_age", rcpt_age)
    ev["merchant_risk"] = _bt("merchant_risk", merch_risk)
    return ev


def _pw_reset_event(t: int, actor: str, keeper: MetadataKeeper) -> dict:
    ev = fenced_event_dict(t=t, event_name="pw_reset", actor=actor, keeper=keeper)
    ev["auth_strength"] = _bt("auth_strength", "password_only")
    return ev


def _device_add_event(t: int, actor: str, keeper: MetadataKeeper, *, rng: random.Random,
                     device_age_value: str = "new") -> dict:
    ev = fenced_event_dict(
        t=t, event_name="device_add", actor=actor, keeper=keeper,
        device_seed=rng.randint(0, 10**9),
    )
    ev["device_age"] = _bt("device_age", device_age_value)
    return ev


def _recipient_add_event(t: int, actor: str, keeper: MetadataKeeper, *, rng: random.Random,
                        age: str = "newly_added") -> dict:
    ev = fenced_event_dict(
        t=t, event_name="recipient_add", actor=actor, keeper=keeper,
        recipient_seed=rng.randint(0, 10**9),
    )
    ev["recipient_age"] = _bt("recipient_age", age)
    return ev


def _attach_session_dwell(events: list[dict]) -> None:
    """Attach session_dwell bucket to the LAST event (representing
    overall session duration).
    """
    if not events:
        return
    duration = events[-1]["t"] - events[0]["t"]
    events[-1]["session_dwell"] = _bt("session_dwell", session_dwell(duration))


# ---------------------------------------------------------------------------
# Journey generators (9 families)
# ---------------------------------------------------------------------------

def gen_clean(seed: int, actor: str = "human") -> Journey:
    """Normal user behavior: routine login + a few small/medium transactions."""
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng,
                               ip_risk_value="low", geo_distance_value="local",
                               auth="mfa_strong", device_age_value="known"))

    n_txns = rng.randint(1, 4)
    for _ in range(n_txns):
        t += rng.randint(60, 600)
        amount = rng.choice([5, 12, 25, 45, 80, 120, 200])
        events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=amount,
                                 velocity="normal", rcpt_age="known", merch_risk="normal"))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family="clean", actor_family=actor,
                   label="legit", is_hard_negative=False, metadata=keeper, seed=seed)


def gen_cred_stuff(seed: int, actor: str = "human") -> Journey:
    """Credential stuffing: many login attempts in quick succession from
    rotating IPs. No completed transactions (mostly failed logins).
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []

    t = 0
    n_attempts = rng.randint(8, 25)
    for _ in range(n_attempts):
        events.append(_login_event(t, actor, keeper, rng=rng,
                                   ip_risk_value="high",  # VPN/datacenter
                                   geo_distance_value="international",
                                   auth="password_only",
                                   device_age_value="rare"))
        # Rapid retries — 1-15 seconds apart
        t += rng.randint(1, 15)

    _attach_session_dwell(events)
    return Journey(events=events, journey_family="cred_stuff", actor_family=actor,
                   label="fraud", is_hard_negative=False, metadata=keeper, seed=seed)


def gen_sim_swap(seed: int, actor: str = "human") -> Journey:
    """SIM-swap takeover: new device → password reset → large transaction
    to a newly-added recipient.
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng,
                               ip_risk_value="medium",
                               geo_distance_value="domestic_far",
                               auth="password_only",
                               device_age_value="new"))

    t += rng.randint(30, 180)
    events.append(_device_add_event(t, actor, keeper, rng=rng, device_age_value="new"))

    t += rng.randint(15, 90)
    events.append(_pw_reset_event(t, actor, keeper))

    t += rng.randint(60, 300)
    events.append(_recipient_add_event(t, actor, keeper, rng=rng, age="newly_added"))

    t += rng.randint(30, 180)
    big_amount = rng.uniform(5000, 18000)  # extreme bucket
    events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=big_amount,
                             velocity="bursty", rcpt_age="newly_added", merch_risk="normal"))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family="sim_swap", actor_family=actor,
                   label="fraud", is_hard_negative=False, metadata=keeper, seed=seed)


def gen_phish_takeover(seed: int, actor: str = "human") -> Journey:
    """Phishing-collected credentials → fast monetization.

    Distinct from sim_swap: no device-add / pw-reset (the attacker has
    the password already). Goes straight to high-value txns from a
    suspicious IP.
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng,
                               ip_risk_value="high",
                               geo_distance_value="international",
                               auth="password_only",
                               device_age_value="rare"))

    n_txns = rng.randint(2, 5)
    velocity_seq = ["bursty"] * (n_txns - 1) + ["extreme"]
    for vel in velocity_seq:
        t += rng.randint(20, 180)
        amount = rng.uniform(500, 8000)
        events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=amount,
                                 velocity=vel, rcpt_age="newly_added", merch_risk="elevated"))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family="phish_takeover", actor_family=actor,
                   label="fraud", is_hard_negative=False, metadata=keeper, seed=seed)


def gen_malware_rat(seed: int, actor: str = "human") -> Journey:
    """Remote-access-trojan style: known device + correct auth, but
    anomalous behavior (atypical timing, unusual recipients).
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng,
                               ip_risk_value="low",         # known device, known IP
                               geo_distance_value="local",
                               auth="mfa_strong",
                               device_age_value="known"))

    # Then anomalous activity: large new-recipient transaction at odd hour.
    t += rng.randint(30, 120)
    events.append(_recipient_add_event(t, actor, keeper, rng=rng, age="newly_added"))

    t += rng.randint(15, 90)
    amount = rng.uniform(2000, 9000)
    events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=amount,
                             velocity="normal", rcpt_age="newly_added", merch_risk="elevated"))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family="malware_rat", actor_family=actor,
                   label="fraud", is_hard_negative=False, metadata=keeper, seed=seed)


def gen_mule_chain(seed: int, actor: str = "human") -> Journey:
    """Mule account: receives funds then forwards to multiple recipients
    in quick succession.
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng,
                               ip_risk_value="low",
                               geo_distance_value="local",
                               auth="password_only",
                               device_age_value="known"))

    # Receive funds (incoming txn) — represented as a single high-amount entry
    t += rng.randint(60, 300)
    incoming = rng.uniform(2000, 12000)
    ev = _txn_event(t, actor, keeper, rng=rng, amount_usd=incoming,
                    velocity="normal", rcpt_age="known", merch_risk="normal")
    ev["direction"] = "incoming"
    events.append(ev)

    # Forward to N recipients rapidly
    n_forwards = rng.randint(3, 7)
    for _ in range(n_forwards):
        t += rng.randint(5, 45)
        events.append(_recipient_add_event(t, actor, keeper, rng=rng, age="newly_added"))
        t += rng.randint(2, 15)
        amount = incoming / n_forwards * rng.uniform(0.8, 1.0)
        ev = _txn_event(t, actor, keeper, rng=rng, amount_usd=amount,
                        velocity="extreme", rcpt_age="newly_added", merch_risk="normal")
        ev["direction"] = "outgoing"
        events.append(ev)

    _attach_session_dwell(events)
    return Journey(events=events, journey_family="mule_chain", actor_family=actor,
                   label="fraud", is_hard_negative=False, metadata=keeper, seed=seed)


# --- Hard negatives ---


def gen_hn_travel(seed: int, actor: str = "human") -> Journey:
    """Hard negative: legitimate user logging in from an international
    location while traveling. Looks like a geo anomaly but otherwise
    routine.
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng,
                               ip_risk_value="low",
                               geo_distance_value="international",
                               auth="mfa_strong",
                               device_age_value="known"))

    n_txns = rng.randint(2, 5)
    for _ in range(n_txns):
        t += rng.randint(120, 1200)
        amount = rng.choice([15, 35, 80, 150, 300])
        events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=amount,
                                 velocity="normal", rcpt_age="known", merch_risk="normal"))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family="hn_travel", actor_family=actor,
                   label="legit", is_hard_negative=True, metadata=keeper, seed=seed)


def gen_hn_large_purchase(seed: int, actor: str = "human") -> Journey:
    """Hard negative: legitimate large purchase (vacation, car, etc.).
    High amount but otherwise normal session.
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng,
                               ip_risk_value="low",
                               geo_distance_value="local",
                               auth="mfa_strong",
                               device_age_value="known"))

    t += rng.randint(300, 1800)  # user took time to decide
    big_amount = rng.uniform(3000, 15000)
    events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=big_amount,
                             velocity="normal", rcpt_age="known", merch_risk="normal"))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family="hn_large_purchase", actor_family=actor,
                   label="legit", is_hard_negative=True, metadata=keeper, seed=seed)


def gen_hn_account_recovery(seed: int, actor: str = "human") -> Journey:
    """Hard negative: legitimate password reset (user forgot password).
    Triggers pw_reset + maybe device_add but is a genuine recovery flow.
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng,
                               ip_risk_value="low",
                               geo_distance_value="local",
                               auth="mfa_strong",      # MFA used during recovery
                               device_age_value="known"))

    t += rng.randint(60, 600)
    events.append(_pw_reset_event(t, actor, keeper))

    # Maybe add a new device (e.g., new phone)
    if random.Random(seed + 1).random() < 0.5:
        t += rng.randint(30, 180)
        events.append(_device_add_event(t, actor, keeper, rng=rng, device_age_value="new"))

    # No transactions (or one small one) — recovery flows usually don't end in big spends
    if random.Random(seed + 2).random() < 0.4:
        t += rng.randint(60, 600)
        events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=rng.uniform(5, 100),
                                 velocity="normal", rcpt_age="known", merch_risk="normal"))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family="hn_account_recovery", actor_family=actor,
                   label="legit", is_hard_negative=True, metadata=keeper, seed=seed)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

JOURNEY_GENERATORS: dict[str, Callable[[int, str], Journey]] = {
    "clean": gen_clean,
    "cred_stuff": gen_cred_stuff,
    "sim_swap": gen_sim_swap,
    "phish_takeover": gen_phish_takeover,
    "malware_rat": gen_malware_rat,
    "mule_chain": gen_mule_chain,
    "hn_travel": gen_hn_travel,
    "hn_large_purchase": gen_hn_large_purchase,
    "hn_account_recovery": gen_hn_account_recovery,
}


def generate(family: str, seed: int, actor: str = "human") -> Journey:
    """Top-level dispatch. `family` must be a key of JOURNEY_GENERATORS."""
    if family not in JOURNEY_GENERATORS:
        raise ValueError(f"unknown journey family: {family!r}")
    return JOURNEY_GENERATORS[family](seed, actor)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    from data.gen.pii_fencer import assert_no_raw_pii_in_event

    for family in JOURNEY_GENERATORS:
        j = generate(family, seed=42, actor="human")
        # All generators produce at least one event
        assert j.events, f"{family} produced no events"
        # Family/label consistency
        if family == "clean" or family.startswith("hn_"):
            assert j.label == "legit", f"{family} should be legit, got {j.label}"
        else:
            assert j.label == "fraud", f"{family} should be fraud, got {j.label}"
        # Hard-negative flag consistency
        assert j.is_hard_negative == family.startswith("hn_"), f"hn flag wrong for {family}"
        # All events PII-fenced
        for ev in j.events:
            assert_no_raw_pii_in_event(ev)
        # Determinism: re-running with same seed produces identical events
        j2 = generate(family, seed=42, actor="human")
        # Note: keeper has its own dicts with same values — compare events only
        for a, b in zip(j.events, j2.events):
            assert a == b, f"{family} non-deterministic at t={a.get('t')}"

    print(f"journey_templates self-test OK ({len(JOURNEY_GENERATORS)} families)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--sample", type=str, help="generate one journey of this family and print it")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor", type=str, default="human")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    if args.sample:
        import json
        j = generate(args.sample, args.seed, args.actor)
        print(json.dumps({
            "journey_family": j.journey_family,
            "actor_family": j.actor_family,
            "label": j.label,
            "is_hard_negative": j.is_hard_negative,
            "events": j.events,
            "seed": j.seed,
        }, indent=2, default=str))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
