"""Runtime PII fencing for narrative text.

Use case: the LLM narrator (data/gen/narrative_generator.py) may produce a
narrative that contains literal-looking PII (emails, IPs, phone numbers,
amounts in dollars). Before the narrative is written to the training
corpus, this module replaces those patterns with the corresponding fenced
placeholders from `src.tokenizer.custom_tokens.PII_TOKENS`.

This is a defense-in-depth layer on top of:
  1. The narrator's prompt explicitly telling it to use fenced tokens
     directly. (Primary defense.)
  2. `eval/leakage_checks.narrative_leakage_scan` which catches class-name
     leakage. (Separate concern.)

The patterns here intentionally err on the side of over-fencing — false
positives produce slightly-too-fenced narratives, which is benign. False
negatives leak real-looking identifiers into the corpus.

NOT used at eval time (eval inputs come from the data pipeline which has
already been fenced).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Email: word@word.tld (loose)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Phone: e.g. +1-555-123-4567, (555) 123-4567, 555.123.4567, 5551234567
PHONE_RE = re.compile(
    r"(?:(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})"
)

# IPv4 (rough; doesn't validate octets <= 255 — over-matching is acceptable)
IPV4_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

# IPv6 (loose hex+colon)
IPV6_RE = re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b")

# Dollar amounts: $1, $1.50, $1,000, $1,234.56
USD_RE = re.compile(r"\$\s?[\d,]+(?:\.\d{1,2})?")

# Account/merchant/recipient names of the form "MERCHANT_12345" or "Acct-X8Y"
# that the narrator might generate. Best-effort.
#
# `(?<!<)` prevents re-matching inside an already-fenced token like
# `<acct_id>`, where `acct_id` would otherwise satisfy the pattern and
# produce a double-fenced `<<acct_id>>` on idempotent re-application.
ACCT_LIKE_RE = re.compile(
    r"(?<!<)\b(?:acct|account|recipient|payee|merchant|customer|user|payer)[_\-#]?[A-Z0-9_\-]{3,}\b",
    re.IGNORECASE,
)

# Browser user-agent fragments (overmatches; OK for fencing)
BROWSER_RE = re.compile(
    r"\b(?:Mozilla|Chrome|Safari|Firefox|Edge|Opera)/[\d.]+(?:\s+\([^)]*\))?",
)

# Device IDs that look like UUIDs or long alphanumeric strings.
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Fencer
# ---------------------------------------------------------------------------

# Application order matters: more-specific patterns first to avoid IP being
# misread as a phone number etc.
_FENCE_ORDER: list[tuple[re.Pattern, str]] = [
    (EMAIL_RE, "<email>"),
    (UUID_RE, "<device_id>"),
    (IPV6_RE, "<ip>"),
    (IPV4_RE, "<ip>"),
    (USD_RE, "<amount>"),   # narrative-level catch; structured side uses bucket tokens
    (PHONE_RE, "<phone>"),
    (BROWSER_RE, "<browser>"),
    (ACCT_LIKE_RE, "<acct_id>"),
]


def fence(text: str) -> tuple[str, dict[str, int]]:
    """Replace PII-looking patterns in `text` with fenced placeholders.

    Returns (fenced_text, hits_per_pattern_name).
    Idempotent: running fence() twice produces the same output as once
    (the fenced tokens never re-match any pattern).
    """
    hits: dict[str, int] = {}
    out = text
    for pat, replacement in _FENCE_ORDER:
        out, n = pat.subn(replacement, out)
        if n:
            hits[replacement] = hits.get(replacement, 0) + n
    return out, hits


def fence_file(in_path: str, out_path: str) -> dict[str, int]:
    """Convenience: fence an entire file. Returns aggregate hit counts."""
    with open(in_path) as f:
        text = f.read()
    fenced, hits = fence(text)
    with open(out_path, "w") as f:
        f.write(fenced)
    return hits


# Sentinel — used by tests to assert the patterns work end-to-end without
# importing pytest.
def _self_test() -> None:
    sample = (
        "User alice@example.com logged in from 192.168.1.5 on Chrome/120.0 "
        "and sent $1,247.50 to MERCHANT_9F2A. Device id: "
        "12345678-1234-1234-1234-123456789012. Phone: (555) 123-4567."
    )
    fenced, hits = fence(sample)
    assert "alice@example.com" not in fenced
    assert "192.168.1.5" not in fenced
    assert "$1,247.50" not in fenced
    assert "MERCHANT_9F2A" not in fenced
    assert "(555) 123-4567" not in fenced
    assert "<email>" in fenced
    assert "<ip>" in fenced
    assert "<amount>" in fenced
    # Idempotence
    fenced2, _ = fence(fenced)
    assert fenced2 == fenced
    print("fencer self-test OK:", hits)


if __name__ == "__main__":
    _self_test()
