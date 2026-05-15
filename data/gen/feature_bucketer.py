"""Derive bucketed-feature tokens from raw event values.

Maps raw values (USD amounts, distances in km, device-age-in-days, etc.) to
the bucket-token strings registered in `src.tokenizer.custom_tokens`.

These tokens carry the actual fraud signal in privacy-safe form. Both the
text side (narrative) and the structured side (events for the side-stream
encoder) emit them.

Each bucketer returns the bucket *value* (e.g. "high"); use
`format_bucket_token(family, value)` to produce the full token string.

CLI:
    python -m data.gen.feature_bucketer --self-test
"""

from __future__ import annotations

import argparse
import sys
from typing import Literal

# Single source of truth for bucket boundaries. Synced with
# src.tokenizer.custom_tokens.BUCKET_FAMILIES.

# ---------------------------------------------------------------------------
# amount_bucket
# ---------------------------------------------------------------------------

def amount_bucket(usd: float) -> Literal["low", "medium", "high", "extreme"]:
    """USD amount → bucket. Thresholds: <$50, <$500, <$5000, >=$5000."""
    if usd < 50:
        return "low"
    if usd < 500:
        return "medium"
    if usd < 5000:
        return "high"
    return "extreme"


# ---------------------------------------------------------------------------
# geo_distance
# ---------------------------------------------------------------------------

def geo_distance(km_from_baseline: float, crossed_country_boundary: bool = False) -> Literal["local", "domestic_far", "international"]:
    """Distance from user's baseline location.

    - international: crossed country boundary (regardless of km)
    - local: < 50 km
    - domestic_far: 50-... (domestic but far from baseline)
    """
    if crossed_country_boundary:
        return "international"
    if km_from_baseline < 50:
        return "local"
    return "domestic_far"


# ---------------------------------------------------------------------------
# ip_risk
# ---------------------------------------------------------------------------

def ip_risk(*, is_vpn: bool = False, is_tor: bool = False,
            is_datacenter_asn: bool = False, residential_reputation: float = 1.0) -> Literal["low", "medium", "high"]:
    """IP-reputation bucket.

    - high:   Tor exit, known VPN, or datacenter ASN
    - medium: residential_reputation in [0.3, 0.7]
    - low:    everything else (clean residential, mobile carrier)
    """
    if is_tor or is_vpn or is_datacenter_asn:
        return "high"
    if 0.3 <= residential_reputation <= 0.7:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# device_age
# ---------------------------------------------------------------------------

def device_age(days_first_seen: float, times_seen: int) -> Literal["known", "new", "rare"]:
    """Device-fingerprint familiarity.

    - rare:  seen <= 2 times historically
    - new:   first seen within the last 7 days
    - known: >= 30 days old AND seen >= 3 times
    - (gap between new and known defaults to "new")
    """
    if times_seen <= 2:
        return "rare"
    if days_first_seen < 7:
        return "new"
    if days_first_seen >= 30 and times_seen >= 3:
        return "known"
    return "new"


# ---------------------------------------------------------------------------
# merchant_risk
# ---------------------------------------------------------------------------

def merchant_risk(*, category_risk_score: float = 0.0, sanctioned: bool = False) -> Literal["normal", "elevated"]:
    """Merchant-category risk. Coarse.

    - elevated: sanctioned OR category_risk_score >= 0.5 (synthetic threshold)
    - normal:   otherwise
    """
    if sanctioned or category_risk_score >= 0.5:
        return "elevated"
    return "normal"


# ---------------------------------------------------------------------------
# txn_velocity
# ---------------------------------------------------------------------------

def txn_velocity(count_last_1h: int, count_last_5min: int) -> Literal["normal", "bursty", "extreme"]:
    """Transaction-count velocity.

    - extreme: >5 in last 5 min
    - bursty:  >5 in last hour
    - normal:  otherwise
    """
    if count_last_5min > 5:
        return "extreme"
    if count_last_1h > 5:
        return "bursty"
    return "normal"


# ---------------------------------------------------------------------------
# recipient_age
# ---------------------------------------------------------------------------

def recipient_age(hours_since_added: float) -> Literal["known", "newly_added"]:
    """Recipient (payee) graph age.

    - newly_added: < 24 hours in the account's payee list
    - known:       >= 24 hours
    """
    if hours_since_added < 24:
        return "newly_added"
    return "known"


# ---------------------------------------------------------------------------
# session_dwell
# ---------------------------------------------------------------------------

def session_dwell(seconds: float) -> Literal["short", "normal", "extended"]:
    """Session duration bucket.

    - short:    < 30 s (often automated)
    - extended: > 1800 s (30 min, often human review/deliberation)
    - normal:   in between
    """
    if seconds < 30:
        return "short"
    if seconds > 1800:
        return "extended"
    return "normal"


# ---------------------------------------------------------------------------
# auth_strength
# ---------------------------------------------------------------------------

def auth_strength(*, used_mfa: bool, used_password: bool, used_cookie_only: bool) -> Literal["mfa_strong", "password_only", "cookie_only"]:
    """Authentication method bucket.

    Priority: mfa > password > cookie.
    """
    if used_mfa:
        return "mfa_strong"
    if used_password:
        return "password_only"
    if used_cookie_only:
        return "cookie_only"
    # Fallback — should not happen in synthetic generation
    return "cookie_only"


# ---------------------------------------------------------------------------
# Token formatter
# ---------------------------------------------------------------------------

def format_bucket_token(family: str, value: str) -> str:
    """Produce the registered bucket-token string (e.g. `<amount_bucket=high>`).

    Validates against `src.tokenizer.custom_tokens.BUCKET_FAMILIES`.
    """
    # Lazy import to keep this module CLI-runnable without the tokenizer.
    from src.tokenizer.custom_tokens import BUCKET_FAMILIES

    if family not in BUCKET_FAMILIES:
        raise ValueError(f"unknown bucket family: {family!r}")
    if value not in BUCKET_FAMILIES[family]:
        raise ValueError(
            f"unknown value {value!r} for family {family!r}; "
            f"expected one of {BUCKET_FAMILIES[family]}"
        )
    return f"<{family}={value}>"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    assert amount_bucket(10) == "low"
    assert amount_bucket(200) == "medium"
    assert amount_bucket(1500) == "high"
    assert amount_bucket(10000) == "extreme"

    assert geo_distance(5) == "local"
    assert geo_distance(500) == "domestic_far"
    assert geo_distance(50, crossed_country_boundary=True) == "international"

    assert ip_risk(is_tor=True) == "high"
    assert ip_risk(is_vpn=True) == "high"
    assert ip_risk(residential_reputation=0.5) == "medium"
    assert ip_risk(residential_reputation=0.95) == "low"

    assert device_age(0.5, 1) == "rare"
    assert device_age(2, 5) == "new"
    assert device_age(60, 10) == "known"

    assert merchant_risk() == "normal"
    assert merchant_risk(sanctioned=True) == "elevated"

    assert txn_velocity(2, 1) == "normal"
    assert txn_velocity(8, 1) == "bursty"
    assert txn_velocity(8, 6) == "extreme"

    assert recipient_age(1) == "newly_added"
    assert recipient_age(100) == "known"

    assert session_dwell(10) == "short"
    assert session_dwell(120) == "normal"
    assert session_dwell(2000) == "extended"

    assert auth_strength(used_mfa=True, used_password=True, used_cookie_only=False) == "mfa_strong"
    assert auth_strength(used_mfa=False, used_password=True, used_cookie_only=False) == "password_only"

    # Token formatter
    assert format_bucket_token("amount_bucket", "high") == "<amount_bucket=high>"
    try:
        format_bucket_token("amount_bucket", "ginormous")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    print("feature_bucketer self-test OK")


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
