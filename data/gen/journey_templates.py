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
# v4 feature-distribution table (data-v4-pivot-plan.md Change 2)
# ---------------------------------------------------------------------------
#
# v3 generators used DETERMINISTIC feature values per family (every
# hn_account_recovery had auth=mfa_strong, device_age=known, ip_risk=low,
# geo_distance=local). That made hn vs fraud families perfectly
# separable at the event-feature level — and is the second-cause
# explanation for the v3 sweep's null result.
#
# v4 samples each feature from a per-family distribution. Families
# still BIAS toward their characteristic signature (hn families lean
# legitimate, fraud families lean suspicious), but there's now
# meaningful overlap. Some hn_account_recovery samples have
# password_only auth and rare devices; some phish_takeover samples
# have mfa_strong auth (the attacker phished MFA) or known devices
# (compromised primary device).
#
# Schema: _FEATURE_DIST[family][phase][feature] = {value: weight}
# Values are normalized by random.choices().
_FEATURE_DIST: dict[str, dict[str, dict[str, dict[str, float]]]] = {
    "clean": {
        "login": {
            "ip_risk":     {"low": 0.85, "medium": 0.12, "high": 0.03},
            "geo_distance":{"local": 0.75, "domestic_far": 0.22, "international": 0.03},
            "auth":        {"mfa_strong": 0.75, "password_only": 0.22, "cookie_only": 0.03},
            "device_age":  {"known": 0.88, "new": 0.10, "rare": 0.02},
        },
        "txn": {
            "rcpt_age":    {"known": 0.95, "newly_added": 0.05},
            "merch_risk":  {"normal": 0.98, "elevated": 0.02},
        },
    },
    "cred_stuff": {
        "login": {
            "ip_risk":     {"high": 0.70, "medium": 0.25, "low": 0.05},
            "geo_distance":{"international": 0.60, "domestic_far": 0.30, "local": 0.10},
            "auth":        {"password_only": 0.85, "cookie_only": 0.10, "mfa_strong": 0.05},
            "device_age":  {"rare": 0.55, "new": 0.30, "known": 0.15},
        },
    },
    "sim_swap": {
        "login": {
            "ip_risk":     {"medium": 0.45, "high": 0.35, "low": 0.20},
            "geo_distance":{"domestic_far": 0.45, "international": 0.30, "local": 0.25},
            "auth":        {"password_only": 0.65, "mfa_strong": 0.20, "cookie_only": 0.15},
            "device_age":  {"new": 0.50, "rare": 0.30, "known": 0.20},
        },
        "device_add": {"device_age": {"new": 0.70, "rare": 0.30}},
        "txn":        {"rcpt_age": {"newly_added": 0.75, "known": 0.25},
                       "merch_risk":  {"normal": 0.80, "elevated": 0.20},
                       "velocity":    {"bursty": 0.50, "extreme": 0.40, "normal": 0.10}},
    },
    "phish_takeover": {
        "login": {
            "ip_risk":     {"high": 0.55, "medium": 0.30, "low": 0.15},
            "geo_distance":{"international": 0.50, "domestic_far": 0.30, "local": 0.20},
            "auth":        {"password_only": 0.55, "mfa_strong": 0.25, "cookie_only": 0.20},
            "device_age":  {"rare": 0.45, "new": 0.30, "known": 0.25},
        },
        "txn":        {"rcpt_age":    {"newly_added": 0.65, "known": 0.35},
                       "merch_risk":  {"elevated": 0.55, "normal": 0.45},
                       "velocity":    {"bursty": 0.50, "extreme": 0.40, "normal": 0.10}},
    },
    "malware_rat": {
        "login": {
            "ip_risk":     {"low": 0.65, "medium": 0.25, "high": 0.10},
            "geo_distance":{"local": 0.60, "domestic_far": 0.30, "international": 0.10},
            "auth":        {"mfa_strong": 0.65, "password_only": 0.25, "cookie_only": 0.10},
            "device_age":  {"known": 0.70, "new": 0.20, "rare": 0.10},
        },
        "txn":        {"rcpt_age":    {"newly_added": 0.55, "known": 0.45},
                       "merch_risk":  {"elevated": 0.40, "normal": 0.60},
                       "velocity":    {"normal": 0.85, "bursty": 0.15}},
    },
    "mule_chain": {
        "login": {
            "ip_risk":     {"low": 0.50, "medium": 0.30, "high": 0.20},
            "geo_distance":{"local": 0.50, "domestic_far": 0.30, "international": 0.20},
            "auth":        {"password_only": 0.45, "mfa_strong": 0.35, "cookie_only": 0.20},
            "device_age":  {"known": 0.55, "new": 0.30, "rare": 0.15},
        },
        "txn":        {"rcpt_age":    {"newly_added": 0.70, "known": 0.30},
                       "merch_risk":  {"normal": 0.85, "elevated": 0.15},
                       "velocity":    {"extreme": 0.55, "bursty": 0.40, "normal": 0.05}},
    },
    "hn_travel": {
        "login": {
            "ip_risk":     {"low": 0.80, "medium": 0.15, "high": 0.05},
            "geo_distance":{"international": 1.0},  # by definition
            "auth":        {"mfa_strong": 0.85, "password_only": 0.10, "cookie_only": 0.05},
            "device_age":  {"known": 0.85, "new": 0.10, "rare": 0.05},
        },
        "txn":        {"rcpt_age":    {"known": 0.90, "newly_added": 0.10},
                       "merch_risk":  {"normal": 0.95, "elevated": 0.05}},
    },
    "hn_large_purchase": {
        "login": {
            "ip_risk":     {"low": 0.80, "medium": 0.15, "high": 0.05},
            "geo_distance":{"local": 0.70, "domestic_far": 0.25, "international": 0.05},
            "auth":        {"mfa_strong": 0.80, "password_only": 0.15, "cookie_only": 0.05},
            "device_age":  {"known": 0.85, "new": 0.10, "rare": 0.05},
        },
        "txn":        {"rcpt_age":    {"known": 0.90, "newly_added": 0.10},
                       "merch_risk":  {"normal": 0.90, "elevated": 0.10}},
    },
    "hn_account_recovery": {
        "login": {
            "ip_risk":     {"low": 0.65, "medium": 0.25, "high": 0.10},
            "geo_distance":{"local": 0.55, "domestic_far": 0.30, "international": 0.15},
            "auth":        {"mfa_strong": 0.55, "password_only": 0.30, "cookie_only": 0.15},
            "device_age":  {"known": 0.60, "new": 0.30, "rare": 0.10},
        },
        "device_add": {"device_age": {"new": 0.85, "rare": 0.15}},
        "txn":        {"rcpt_age":    {"known": 0.85, "newly_added": 0.15},
                       "merch_risk":  {"normal": 0.90, "elevated": 0.10}},
    },
    # Adversarial subtypes — Change 3
    # hn_recovery_high_amount: looks behaviorally like sim_swap (device
    # add + pw reset + large transfer to new recipient) but events
    # carry legitimacy markers. Label = legit.
    "hn_recovery_high_amount": {
        "login": {
            "ip_risk":     {"low": 0.70, "medium": 0.25, "high": 0.05},
            "geo_distance":{"local": 0.60, "domestic_far": 0.30, "international": 0.10},
            "auth":        {"mfa_strong": 0.70, "password_only": 0.20, "cookie_only": 0.10},
            "device_age":  {"new": 0.50, "known": 0.40, "rare": 0.10},
        },
        "device_add": {"device_age": {"new": 0.85, "rare": 0.15}},
        "txn":        {"rcpt_age":    {"newly_added": 0.55, "known": 0.45},  # 45% known: the "new" recipient is actually self
                       "merch_risk":  {"normal": 0.90, "elevated": 0.10},
                       "velocity":    {"normal": 0.70, "bursty": 0.30}},
    },
    # phish_takeover_mfa_phished: looks behaviorally like clean (login
    # + a few txns) but one txn goes to a newly-added recipient AND
    # the device fingerprint is anomalous. Label = fraud.
    "phish_takeover_mfa_phished": {
        "login": {
            "ip_risk":     {"low": 0.55, "medium": 0.35, "high": 0.10},
            "geo_distance":{"local": 0.50, "domestic_far": 0.35, "international": 0.15},
            "auth":        {"mfa_strong": 0.70, "password_only": 0.20, "cookie_only": 0.10},
            "device_age":  {"rare": 0.45, "known": 0.30, "new": 0.25},  # subtle device anomaly
        },
        "txn":        {"rcpt_age":    {"known": 0.55, "newly_added": 0.45},  # one txn slipped to newly-added
                       "merch_risk":  {"normal": 0.65, "elevated": 0.35},
                       "velocity":    {"normal": 0.75, "bursty": 0.25}},
    },
}


def _sample_value(rng: random.Random, family: str, phase: str, feature: str,
                  fallback: str) -> str:
    """Sample one value for (family, phase, feature) from _FEATURE_DIST.

    Returns `fallback` if no distribution is registered (defensive — the
    table covers all known family+phase+feature triples).
    """
    dist = _FEATURE_DIST.get(family, {}).get(phase, {}).get(feature)
    if not dist:
        return fallback
    keys = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(keys, weights=weights, k=1)[0]


def _login_features(rng: random.Random, family: str) -> dict:
    """Sample the four login-event features for a given journey family."""
    return {
        "ip_risk_value":      _sample_value(rng, family, "login", "ip_risk",     "low"),
        "geo_distance_value": _sample_value(rng, family, "login", "geo_distance","local"),
        "auth":               _sample_value(rng, family, "login", "auth",        "password_only"),
        "device_age_value":   _sample_value(rng, family, "login", "device_age",  "known"),
    }


def _txn_features(rng: random.Random, family: str, default_velocity: str = "normal") -> dict:
    """Sample txn features for a given family. `default_velocity` is
    used when no velocity distribution is registered (e.g., hn families
    that don't carry an explicit velocity bias)."""
    return {
        "rcpt_age":   _sample_value(rng, family, "txn", "rcpt_age",   "known"),
        "merch_risk": _sample_value(rng, family, "txn", "merch_risk", "normal"),
        "velocity":   _sample_value(rng, family, "txn", "velocity",   default_velocity),
    }


# ---------------------------------------------------------------------------
# Journey generators (9 v3 families + 2 v4 adversarial subtypes)
# ---------------------------------------------------------------------------

def gen_clean(seed: int, actor: str = "human") -> Journey:
    """Normal user behavior: routine login + a few small/medium transactions."""
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []
    fam = "clean"

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng, **_login_features(rng, fam)))

    n_txns = rng.randint(1, 4)
    for _ in range(n_txns):
        t += rng.randint(60, 600)
        amount = rng.choice([5, 12, 25, 45, 80, 120, 200])
        txn_feat = _txn_features(rng, fam)
        events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=amount, **txn_feat))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family=fam, actor_family=actor,
                   label="legit", is_hard_negative=False, metadata=keeper, seed=seed)


def gen_cred_stuff(seed: int, actor: str = "human") -> Journey:
    """Credential stuffing: many login attempts in quick succession from
    rotating IPs. No completed transactions (mostly failed logins).
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []
    fam = "cred_stuff"

    t = 0
    n_attempts = rng.randint(8, 25)
    for _ in range(n_attempts):
        # Each login attempt independently samples features. Most will
        # be high-risk per the distribution; a few will look benign —
        # reflecting that real cred-stuffing campaigns try many vectors.
        events.append(_login_event(t, actor, keeper, rng=rng, **_login_features(rng, fam)))
        t += rng.randint(1, 15)  # rapid retries

    _attach_session_dwell(events)
    return Journey(events=events, journey_family=fam, actor_family=actor,
                   label="fraud", is_hard_negative=False, metadata=keeper, seed=seed)


def gen_sim_swap(seed: int, actor: str = "human") -> Journey:
    """SIM-swap takeover: new device → password reset → large transaction
    to a newly-added recipient.
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []
    fam = "sim_swap"

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng, **_login_features(rng, fam)))

    t += rng.randint(30, 180)
    dev_age = _sample_value(rng, fam, "device_add", "device_age", "new")
    events.append(_device_add_event(t, actor, keeper, rng=rng, device_age_value=dev_age))

    t += rng.randint(15, 90)
    events.append(_pw_reset_event(t, actor, keeper))

    t += rng.randint(60, 300)
    rcpt_age_for_add = _sample_value(rng, fam, "txn", "rcpt_age", "newly_added")
    events.append(_recipient_add_event(t, actor, keeper, rng=rng, age=rcpt_age_for_add))

    t += rng.randint(30, 180)
    big_amount = rng.uniform(5000, 18000)
    txn_feat = _txn_features(rng, fam, default_velocity="bursty")
    events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=big_amount, **txn_feat))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family=fam, actor_family=actor,
                   label="fraud", is_hard_negative=False, metadata=keeper, seed=seed)


def gen_phish_takeover(seed: int, actor: str = "human") -> Journey:
    """Phishing-collected credentials → fast monetization.

    Distinct from sim_swap: no device-add / pw-reset (the attacker has
    the password already). Goes straight to high-value txns. Note:
    v4 stochastic features mean that ~25% of phish_takeover samples
    now use mfa_strong (the attacker phished the MFA too) and ~25%
    use a known device (compromised primary device) — making this
    family no longer perfectly separable from hn families at the
    event-feature level.
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []
    fam = "phish_takeover"

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng, **_login_features(rng, fam)))

    n_txns = rng.randint(2, 5)
    for _ in range(n_txns):
        t += rng.randint(20, 180)
        amount = rng.uniform(500, 8000)
        txn_feat = _txn_features(rng, fam, default_velocity="bursty")
        events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=amount, **txn_feat))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family=fam, actor_family=actor,
                   label="fraud", is_hard_negative=False, metadata=keeper, seed=seed)


def gen_malware_rat(seed: int, actor: str = "human") -> Journey:
    """Remote-access-trojan style: known device + correct auth, but
    anomalous behavior (atypical timing, unusual recipients).
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []
    fam = "malware_rat"

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng, **_login_features(rng, fam)))

    t += rng.randint(30, 120)
    rcpt_age_for_add = _sample_value(rng, fam, "txn", "rcpt_age", "newly_added")
    events.append(_recipient_add_event(t, actor, keeper, rng=rng, age=rcpt_age_for_add))

    t += rng.randint(15, 90)
    amount = rng.uniform(2000, 9000)
    txn_feat = _txn_features(rng, fam)
    events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=amount, **txn_feat))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family=fam, actor_family=actor,
                   label="fraud", is_hard_negative=False, metadata=keeper, seed=seed)


def gen_mule_chain(seed: int, actor: str = "human") -> Journey:
    """Mule account: receives funds then forwards to multiple recipients
    in quick succession.
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []
    fam = "mule_chain"

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng, **_login_features(rng, fam)))

    # Receive funds (incoming txn) — represented as a single high-amount entry.
    # The incoming side keeps a "neutral" velocity since it's just a deposit;
    # the outbound spree below is what carries the fraud signal.
    t += rng.randint(60, 300)
    incoming = rng.uniform(2000, 12000)
    ev = _txn_event(t, actor, keeper, rng=rng, amount_usd=incoming,
                    velocity="normal", rcpt_age="known", merch_risk="normal")
    ev["direction"] = "incoming"
    events.append(ev)

    # Forward to N recipients rapidly — stochastic features per forward
    n_forwards = rng.randint(3, 7)
    for _ in range(n_forwards):
        t += rng.randint(5, 45)
        rcpt_age_for_add = _sample_value(rng, fam, "txn", "rcpt_age", "newly_added")
        events.append(_recipient_add_event(t, actor, keeper, rng=rng, age=rcpt_age_for_add))
        t += rng.randint(2, 15)
        amount = incoming / n_forwards * rng.uniform(0.8, 1.0)
        txn_feat = _txn_features(rng, fam, default_velocity="extreme")
        ev = _txn_event(t, actor, keeper, rng=rng, amount_usd=amount, **txn_feat)
        ev["direction"] = "outgoing"
        events.append(ev)

    _attach_session_dwell(events)
    return Journey(events=events, journey_family=fam, actor_family=actor,
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
    fam = "hn_travel"

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng, **_login_features(rng, fam)))

    n_txns = rng.randint(2, 5)
    for _ in range(n_txns):
        t += rng.randint(120, 1200)
        amount = rng.choice([15, 35, 80, 150, 300])
        txn_feat = _txn_features(rng, fam)
        events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=amount, **txn_feat))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family=fam, actor_family=actor,
                   label="legit", is_hard_negative=True, metadata=keeper, seed=seed)


def gen_hn_large_purchase(seed: int, actor: str = "human") -> Journey:
    """Hard negative: legitimate large purchase (vacation, car, etc.).
    High amount but otherwise normal session.
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []
    fam = "hn_large_purchase"

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng, **_login_features(rng, fam)))

    t += rng.randint(300, 1800)  # user took time to decide
    big_amount = rng.uniform(3000, 15000)
    txn_feat = _txn_features(rng, fam)
    events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=big_amount, **txn_feat))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family=fam, actor_family=actor,
                   label="legit", is_hard_negative=True, metadata=keeper, seed=seed)


def gen_hn_account_recovery(seed: int, actor: str = "human") -> Journey:
    """Hard negative: legitimate password reset (user forgot password).
    Triggers pw_reset + maybe device_add but is a genuine recovery flow.

    v4: features now stochastic — ~30% of samples have password_only
    auth, ~10% have high ip_risk, ~30% have a new device added during
    the session. These overlap with phish_takeover and sim_swap feature
    distributions, so a feature-level classifier can no longer
    perfectly separate them.
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []
    fam = "hn_account_recovery"

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng, **_login_features(rng, fam)))

    t += rng.randint(60, 600)
    events.append(_pw_reset_event(t, actor, keeper))

    if random.Random(seed + 1).random() < 0.5:
        t += rng.randint(30, 180)
        dev_age = _sample_value(rng, fam, "device_add", "device_age", "new")
        events.append(_device_add_event(t, actor, keeper, rng=rng, device_age_value=dev_age))

    if random.Random(seed + 2).random() < 0.4:
        t += rng.randint(60, 600)
        txn_feat = _txn_features(rng, fam)
        events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=rng.uniform(5, 100),
                                 **txn_feat))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family=fam, actor_family=actor,
                   label="legit", is_hard_negative=True, metadata=keeper, seed=seed)


# ---------------------------------------------------------------------------
# v4 adversarial subtypes (Change 3)
# ---------------------------------------------------------------------------
# These two families are designed so that a SINGLE-STREAM classifier
# (text-only OR event-only) cannot reliably distinguish them from their
# adversarial counterparts. The disambiguating signal requires looking
# at BOTH the event sequence AND the per-event features.
#
# hn_recovery_high_amount: event sequence reads like sim_swap (login +
# device_add + pw_reset + transfer to recipient) but the events carry
# legitimacy markers (mfa_strong, low ip_risk, recipient is actually
# a known account in disguise). Label = legit.
#
# phish_takeover_mfa_phished: event sequence reads like clean
# (login + a few transfers) but one transfer goes to a newly-added
# recipient and the device fingerprint is subtly anomalous. Label =
# fraud.

def gen_hn_recovery_high_amount(seed: int, actor: str = "human") -> Journey:
    """Adversarial hard negative: sim_swap-style event sequence with
    legitimacy markers. The 'newly-added' recipient is actually a known
    account (e.g., the user transferring to their other account).
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []
    fam = "hn_recovery_high_amount"

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng, **_login_features(rng, fam)))

    t += rng.randint(30, 180)
    dev_age = _sample_value(rng, fam, "device_add", "device_age", "new")
    events.append(_device_add_event(t, actor, keeper, rng=rng, device_age_value=dev_age))

    t += rng.randint(15, 90)
    events.append(_pw_reset_event(t, actor, keeper))

    t += rng.randint(60, 300)
    rcpt_age_for_add = _sample_value(rng, fam, "txn", "rcpt_age", "newly_added")
    events.append(_recipient_add_event(t, actor, keeper, rng=rng, age=rcpt_age_for_add))

    t += rng.randint(30, 180)
    big_amount = rng.uniform(3000, 12000)
    txn_feat = _txn_features(rng, fam)
    events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=big_amount, **txn_feat))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family=fam, actor_family=actor,
                   label="legit", is_hard_negative=True, metadata=keeper, seed=seed)


def gen_phish_takeover_mfa_phished(seed: int, actor: str = "human") -> Journey:
    """Adversarial fraud: clean-style event sequence (login + transfers)
    but the attacker phished the MFA AND one transfer goes to a
    newly-added recipient. Sub-feature signal: device fingerprint shows
    a subtle anomaly (sampled into device_age=rare more often than
    typical clean sessions).
    """
    rng = random.Random(seed)
    keeper = MetadataKeeper()
    events: list[dict] = []
    fam = "phish_takeover_mfa_phished"

    t = 0
    events.append(_login_event(t, actor, keeper, rng=rng, **_login_features(rng, fam)))

    n_txns = rng.randint(2, 4)
    for i in range(n_txns):
        t += rng.randint(120, 600)
        amount = rng.uniform(150, 4000)
        txn_feat = _txn_features(rng, fam)
        # Force AT LEAST ONE transfer to a newly-added recipient — that's
        # the disambiguating feature.
        #
        # Review 021 finding #3 fix: _txn_features returns RAW values
        # like "known"/"newly_added"; _txn_event later converts them
        # into bucket tokens like "<recipient_age=known>". The v0
        # comparison `txn_feat["rcpt_age"] == _bt("recipient_age",
        # "known")` was always False because the LHS is "known" not
        # "<recipient_age=known>". That meant ~16% of samples (160/1000
        # in spot check) had no newly-added recipient at all — the
        # adversarial feature was silently absent for those samples.
        # Fix: compare against the raw value. The last txn ALWAYS gets
        # a newly_added recipient now, even if the prior sample was
        # already newly_added (idempotent).
        if i == n_txns - 1:
            txn_feat["rcpt_age"] = "newly_added"
        events.append(_txn_event(t, actor, keeper, rng=rng, amount_usd=amount, **txn_feat))

    _attach_session_dwell(events)
    return Journey(events=events, journey_family=fam, actor_family=actor,
                   label="fraud", is_hard_negative=False, metadata=keeper, seed=seed)


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
    # v4 adversarial subtypes (Change 3)
    "hn_recovery_high_amount": gen_hn_recovery_high_amount,
    "phish_takeover_mfa_phished": gen_phish_takeover_mfa_phished,
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

    # Review 021 finding #3 regression: phish_takeover_mfa_phished must
    # ALWAYS contain at least one event with recipient_age=newly_added.
    # The v0 implementation used the wrong comparison (raw "known" vs
    # bucket token "<recipient_age=known>") so ~16% of samples (160 of
    # 1000) silently lacked the disambiguating feature. After the fix,
    # the last txn unconditionally sets rcpt_age="newly_added".
    newly_added_token = format_bucket_token("recipient_age", "newly_added")
    n_check = 100
    misses = []
    for s in range(n_check):
        j = generate("phish_takeover_mfa_phished", seed=s, actor="human")
        has_na = any(ev.get("recipient_age") == newly_added_token for ev in j.events)
        if not has_na:
            misses.append(s)
    assert not misses, (
        f"phish_takeover_mfa_phished is missing newly_added recipient in "
        f"{len(misses)}/{n_check} samples — review 021 finding #3 regression. "
        f"First miss seeds: {misses[:5]}"
    )

    print(f"journey_templates self-test OK ({len(JOURNEY_GENERATORS)} families, "
          f"adversarial-feature invariant: {n_check}/{n_check})")


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
