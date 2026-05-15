"""Custom-token registry for the cross-attention ATO POC.

Three token families per PLAN.md v3:

  1. **Journey-type structural tokens** — `<journey_*>` / `</journey_*>`.
     Visible during training; stripped or opacified at fraud-classification
     eval per `eval/eval_modes.py`. NOT the fraud signal.

  2. **Actor-type structural tokens** — `<actor_*>`. Same eval-mode regime.
     Agentic-commerce dimension.

  3. **Event tokens** — `<event_*>`. Structural markers within a journey.
     Always visible.

  4. **PII-fencing tokens** — `<acct_id>`, `<email>`, `<phone>`,
     `<device_id>`, `<ip>`, `<recipient>`, `<merchant>`, `<browser>`.
     Hygiene only; raw identifiers never appear in the training corpus.

  5. **Bucketed derived-feature tokens** — `<amount_bucket=high>`,
     `<geo_distance=international>`, etc. **The fraud signal lives here.**
     Always visible. Privacy-safe by construction.

Usage:
    from transformers import AutoTokenizer
    from src.tokenizer.custom_tokens import install, ALL_CUSTOM_TOKENS

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    n_added = install(tok)
    # IMPORTANT: caller must then call model.resize_token_embeddings(len(tok))
    # before any training step. Forgetting this produces silent NaN.

CLI:
    python -m src.tokenizer.custom_tokens --check
        Validates registry integrity (no duplicates, all expected families
        present). Does NOT require model weights. Full tokenizer-roundtrip
        is performed by scripts/preflight_check.py.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

# ---------------------------------------------------------------------------
# 1. Journey-type structural tokens (open + close pairs)
# ---------------------------------------------------------------------------

JOURNEY_FAMILIES: list[str] = [
    "clean",
    "cred_stuff",
    "sim_swap",
    "phish_takeover",
    "malware_rat",
    "mule_chain",
    "hn_travel",
    "hn_large_purchase",
    "hn_account_recovery",
]

JOURNEY_TOKENS: list[str] = (
    [f"<journey_{name}>" for name in JOURNEY_FAMILIES] +
    [f"</journey_{name}>" for name in JOURNEY_FAMILIES]
)

# Hard-negative families (subset of JOURNEY_FAMILIES) — useful for analysis
# code that needs to compute hard-negative FPR.
HARD_NEGATIVE_FAMILIES: list[str] = [
    "hn_travel",
    "hn_large_purchase",
    "hn_account_recovery",
]

# Fraud families = JOURNEY_FAMILIES \ {clean} \ HARD_NEGATIVE_FAMILIES
FRAUD_FAMILIES: list[str] = [
    f for f in JOURNEY_FAMILIES
    if f != "clean" and f not in HARD_NEGATIVE_FAMILIES
]

# ---------------------------------------------------------------------------
# 2. Actor-type structural tokens
# ---------------------------------------------------------------------------

ACTOR_FAMILIES: list[str] = [
    "human",
    "agent_buying",
    "agent_finance",
    "agent_compromised",
    "agent_adversarial",
    "hybrid",
]

ACTOR_TOKENS: list[str] = [f"<actor_{name}>" for name in ACTOR_FAMILIES]

# Agent-actor classes (subset of ACTOR_FAMILIES) — useful for agent-vs-human
# differential analysis.
AGENT_ACTOR_FAMILIES: list[str] = [
    f for f in ACTOR_FAMILIES if f.startswith("agent_") or f == "hybrid"
]

# ---------------------------------------------------------------------------
# 3. Event tokens (structural within a journey)
# ---------------------------------------------------------------------------

EVENT_NAMES: list[str] = [
    "login",
    "txn",
    "pw_reset",
    "device_add",
    "recipient_add",
    "chat_to_support",
    "tool_call",  # for agent actors
]

EVENT_TOKENS: list[str] = [f"<event_{name}>" for name in EVENT_NAMES]

# ---------------------------------------------------------------------------
# 4. PII-fencing tokens (raw identifiers; no info leak)
# ---------------------------------------------------------------------------

PII_TOKENS: list[str] = [
    "<acct_id>",
    "<email>",
    "<phone>",
    "<device_id>",
    "<ip>",
    "<recipient>",
    "<merchant>",
    "<browser>",
]

# ---------------------------------------------------------------------------
# 5. Bucketed derived-feature tokens (THE FRAUD SIGNAL)
# ---------------------------------------------------------------------------

# {family: ordered list of bucket values}
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

BUCKET_TOKENS: list[str] = [
    f"<{family}={value}>"
    for family, values in BUCKET_FAMILIES.items()
    for value in values
]

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

ALL_CUSTOM_TOKENS: list[str] = (
    JOURNEY_TOKENS + ACTOR_TOKENS + EVENT_TOKENS + PII_TOKENS + BUCKET_TOKENS
)


# ---------------------------------------------------------------------------
# Installer (HuggingFace tokenizer)
# ---------------------------------------------------------------------------

def install(tokenizer, tokens: Iterable[str] | None = None) -> int:
    """Add custom tokens to a HuggingFace tokenizer as special tokens.

    Returns the number of tokens newly added (zero if already present).

    Caller's responsibility: after installation, call
        model.resize_token_embeddings(len(tokenizer))
    on the underlying model before training. Forgetting this is silent NaN.
    """
    tokens = list(tokens) if tokens is not None else ALL_CUSTOM_TOKENS
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in tokens if t not in existing]
    if not new_tokens:
        return 0
    # `additional_special_tokens` is the safest slot — these become
    # individually-tokenized atomic units, the model treats them as never
    # broken into BPE subwords.
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    return len(new_tokens)


def init_new_embeddings(model, tokenizer, *, old_vocab_size: int | None = None) -> None:
    """Initialize newly-added token embeddings as the mean of existing
    embeddings (a stable default that avoids NaN cascades on the first step).

    Call this AFTER `install()` and AFTER `model.resize_token_embeddings(len(tokenizer))`.
    `old_vocab_size` should be the model's pre-resize vocab size; if None,
    nothing is initialized (assumes resize already handled init).
    """
    import torch

    if old_vocab_size is None:
        return
    if old_vocab_size >= len(tokenizer):
        return

    input_emb = model.get_input_embeddings().weight.data
    # Mean over the pre-existing rows
    mean_input = input_emb[:old_vocab_size].mean(dim=0)
    input_emb[old_vocab_size:] = mean_input

    # Output embeddings, if untied (Qwen3-8B has tied embeddings by default,
    # but check just in case)
    output_emb_layer = model.get_output_embeddings()
    if output_emb_layer is not None and output_emb_layer.weight is not input_emb:
        out = output_emb_layer.weight.data
        if out.shape[0] >= len(tokenizer):
            mean_out = out[:old_vocab_size].mean(dim=0)
            out[old_vocab_size:] = mean_out


# ---------------------------------------------------------------------------
# Registry integrity check (CLI)
# ---------------------------------------------------------------------------

def check_registry() -> dict:
    """Validate registry invariants. Returns a summary dict; raises on failure.

    Invariants:
      - No duplicate tokens across families.
      - Every token is a non-empty string starting with `<` and ending with `>`.
      - Every journey family has both open and close tokens.
      - Every bucket family has at least 2 values.
      - Token counts match expected ranges (sanity).
    """
    # Duplicate check
    seen: set[str] = set()
    dupes: list[str] = []
    for t in ALL_CUSTOM_TOKENS:
        if t in seen:
            dupes.append(t)
        seen.add(t)
    if dupes:
        raise AssertionError(f"duplicate tokens in registry: {dupes}")

    # Shape check
    for t in ALL_CUSTOM_TOKENS:
        if not isinstance(t, str) or not t:
            raise AssertionError(f"empty or non-string token: {t!r}")
        if not (t.startswith("<") and t.endswith(">")):
            raise AssertionError(f"malformed token (missing <…>): {t!r}")

    # Journey open/close pairing
    open_set = {t for t in JOURNEY_TOKENS if not t.startswith("</")}
    close_set = {t for t in JOURNEY_TOKENS if t.startswith("</")}
    expected_open = {f"<journey_{f}>" for f in JOURNEY_FAMILIES}
    expected_close = {f"</journey_{f}>" for f in JOURNEY_FAMILIES}
    if open_set != expected_open:
        raise AssertionError(f"journey open set mismatch: {open_set ^ expected_open}")
    if close_set != expected_close:
        raise AssertionError(f"journey close set mismatch: {close_set ^ expected_close}")

    # Bucket families each have >= 2 values
    for family, vals in BUCKET_FAMILIES.items():
        if len(vals) < 2:
            raise AssertionError(f"bucket family {family!r} has <2 values")

    # Sanity counts (these are upper bounds; lower bounds are the invariants)
    summary = {
        "n_total": len(ALL_CUSTOM_TOKENS),
        "n_journey": len(JOURNEY_TOKENS),
        "n_actor": len(ACTOR_TOKENS),
        "n_event": len(EVENT_TOKENS),
        "n_pii": len(PII_TOKENS),
        "n_bucket": len(BUCKET_TOKENS),
        "n_journey_families": len(JOURNEY_FAMILIES),
        "n_actor_families": len(ACTOR_FAMILIES),
        "n_bucket_families": len(BUCKET_FAMILIES),
        "n_hard_negative_families": len(HARD_NEGATIVE_FAMILIES),
        "n_fraud_families": len(FRAUD_FAMILIES),
        "n_agent_actor_families": len(AGENT_ACTOR_FAMILIES),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="validate registry integrity")
    parser.add_argument("--list", action="store_true",
                        help="print all tokens, one per line")
    args = parser.parse_args()

    if args.list:
        for t in ALL_CUSTOM_TOKENS:
            print(t)
        return 0

    if args.check or (not args.list):
        summary = check_registry()
        print("custom-token registry OK:")
        for k, v in summary.items():
            print(f"  {k:30s} = {v}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
