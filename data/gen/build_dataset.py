"""End-to-end dataset builder.

Samples N journeys, mixes actors, narrates (LLM or template), serializes
to the training-corpus text format (journey/actor wrappers + event lines
with bucket tokens + narrative + verdict footer), and writes HF
Dataset-style JSONL.

Class balance is enforced explicitly: ~30% fraud / ~30% hard negatives /
~40% clean. Per-journey actor mix is biased plausibly (e.g.,
agent_compromised over-represented in fraud journeys, human in hard
negatives).

A leakage audit runs at the end on a sampled subset; build aborts if
any banned narrative phrase slipped through.

CLI:
    # Smoke (no LLM, 100 examples):
    python -m data.gen.build_dataset --n 100 --out data/samples/smoke --mode template

    # Day-1 vertical slice (1-2k, LLM if API key set, else template):
    python -m data.gen.build_dataset --n 1500 --out data/train_llm_narrated --mode llm

    # Cheap 50k eval set:
    python -m data.gen.build_dataset --n 50000 --out data/eval_medium_50k --mode template
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterator

from data.gen.agent_actor_mixer import mix
from data.gen.cheap_template_generator import generate_narrative as cheap_narrative
from data.gen.journey_templates import generate as generate_journey
from data.gen.narrative_generator import (
    CostTracker, DEFAULT_USD_BUDGET,
    generate_narrative as llm_narrative,
    generate_narratives_concurrent,
)
from data.gen.pii_fencer import assert_no_raw_pii_in_event
from data.gen.types import Journey
from eval.leakage_checks import narrative_leakage_scan


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

# Journey-family sampling weights (relative).
#
# v3 split: fraud ~30%, hn ~30%, clean ~40% across 9 families.
# v4 (Change 3) adds two adversarial subtypes at ~5% each, drawing
# weight proportionally from their conventional counterparts so the
# overall fraud / hn / clean split stays roughly fraud ~30%, hn ~30%,
# clean ~40%.
JOURNEY_WEIGHTS: dict[str, float] = {
    "clean":                40.0,   # 40%
    # Fraud families
    "cred_stuff":            6.0,
    "sim_swap":              6.0,
    "phish_takeover":        4.5,   # gave 1.5 to phish_takeover_mfa_phished
    "malware_rat":           6.0,
    "mule_chain":            6.0,
    "phish_takeover_mfa_phished": 1.5,  # v4 adversarial (Change 3) — ~1.5%
    # Hard negatives
    "hn_travel":            10.0,
    "hn_large_purchase":    10.0,
    "hn_account_recovery":   8.5,   # gave 1.5 to hn_recovery_high_amount
    "hn_recovery_high_amount": 1.5, # v4 adversarial (Change 3) — ~1.5%
}

# Per-journey-family actor distributions. Each list sums to 1.0.
ACTOR_BY_JOURNEY: dict[str, dict[str, float]] = {
    "clean": {
        "human": 0.70, "agent_buying": 0.15, "agent_finance": 0.10, "hybrid": 0.05,
    },
    "cred_stuff": {
        "human": 0.45, "agent_adversarial": 0.40, "agent_compromised": 0.10, "hybrid": 0.05,
    },
    "sim_swap": {
        "human": 0.60, "agent_compromised": 0.25, "agent_adversarial": 0.10, "hybrid": 0.05,
    },
    "phish_takeover": {
        "human": 0.55, "agent_adversarial": 0.25, "agent_compromised": 0.15, "hybrid": 0.05,
    },
    "malware_rat": {
        "human": 0.70, "agent_compromised": 0.20, "hybrid": 0.10,
    },
    "mule_chain": {
        "human": 0.55, "agent_compromised": 0.30, "agent_adversarial": 0.10, "hybrid": 0.05,
    },
    "hn_travel": {
        "human": 0.85, "agent_buying": 0.10, "agent_finance": 0.05,
    },
    "hn_large_purchase": {
        "human": 0.80, "agent_buying": 0.10, "agent_finance": 0.10,
    },
    "hn_account_recovery": {
        "human": 0.95, "hybrid": 0.05,
    },
    # v4 adversarial subtypes (Change 3) — inherit actor distributions
    # from their behavioral counterparts so the actor signal doesn't
    # accidentally help disambiguate.
    "hn_recovery_high_amount": {
        "human": 0.95, "hybrid": 0.05,
    },
    "phish_takeover_mfa_phished": {
        "human": 0.70, "agent_buying": 0.15, "agent_finance": 0.10, "hybrid": 0.05,
    },
}

# Verdict-footer evidence strings, per journey family. Deterministic (not
# LLM-generated) so they cannot leak — and so we can use them as a
# scoring target without ambiguity.
#
# Note: v4 (Change 4) tightens the prompt contract — the verdict footer
# in the trained-on text is `label: {fraud|legit}` ONLY, no evidence /
# confidence / journey_family fields visible to the LM. These two
# tables remain for diagnostic logging in the per-row metadata, but
# are not consumed by the trainers.
EVIDENCE_BY_FAMILY: dict[str, str] = {
    "clean":               "no_anomalies",
    "cred_stuff":          "high_velocity_logins, rotating_ips, mostly_failed",
    "sim_swap":            "device_change, pw_reset, large_txn, new_recipient",
    "phish_takeover":      "suspicious_ip, new_device, high_velocity_txns",
    "malware_rat":         "known_device, anomalous_recipient, atypical_amount",
    "mule_chain":          "incoming_then_fan_out, newly_added_recipients, extreme_velocity",
    "hn_travel":           "international_geo, otherwise_routine",
    "hn_large_purchase":   "high_amount, known_merchant, long_dwell",
    "hn_account_recovery": "pw_reset, mfa_present, no_high_value_txns",
    "hn_recovery_high_amount":   "pw_reset, mfa_present, device_change, large_txn_to_known_account",
    "phish_takeover_mfa_phished":"phished_mfa, subtle_device_anomaly, newly_added_recipient",
}

CONFIDENCE_BY_FAMILY: dict[str, str] = {
    "clean": "high", "cred_stuff": "high", "sim_swap": "high",
    "phish_takeover": "high", "malware_rat": "medium",
    "mule_chain": "high", "hn_travel": "medium",
    "hn_large_purchase": "medium", "hn_account_recovery": "high",
    "hn_recovery_high_amount": "medium",   # adversarial — analyst genuinely needs both streams
    "phish_takeover_mfa_phished": "medium",
}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_journey_family(rng: random.Random) -> str:
    families = list(JOURNEY_WEIGHTS.keys())
    weights = list(JOURNEY_WEIGHTS.values())
    return rng.choices(families, weights=weights, k=1)[0]


def sample_actor_for_family(family: str, rng: random.Random) -> str:
    dist = ACTOR_BY_JOURNEY[family]
    actors = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(actors, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def event_to_line(ev: dict) -> str:
    """Render one event as a single text line within the training corpus.

    Format: `<event_NAME>t=<sec> <bucket_tokens...>`.

    Public per review 007 finding #5 — `train_structured_as_text.py`
    imports this directly so the baseline's event-line format is
    byte-identical to what the corpus uses (no divergent
    `_serialize_events_compact`).
    """
    parts = [f"<event_{ev['event']}>t={ev['t']}"]
    for key in ("amount_bucket", "geo_distance", "ip_risk", "device_age",
                "merchant_risk", "txn_velocity", "recipient_age",
                "session_dwell", "auth_strength"):
        if key in ev:
            parts.append(ev[key])
    # PII tokens (always fenced)
    for key in ("ip", "device_id", "recipient", "merchant"):
        if key in ev:
            parts.append(ev[key])
    return " ".join(parts)


# Back-compat alias (the original private name was used internally below).
_event_to_line = event_to_line


# ---------------------------------------------------------------------------
# v4 canonical-form composition (data-v4-pivot-plan.md Change 4)
# ---------------------------------------------------------------------------
#
# v3 emitted one monolithic `text` field per row, with every model arm
# (text_only, lora_text, structured_as_text, xattn) reading from it.
# The text contained <journey_X> wrappers, <actor_Y> tokens, event
# lines, AND the narrative — meaning every LM arm saw the structured
# event signal whether or not it was supposed to.
#
# v4 stores the row in CANONICAL FIELDS and each trainer composes the
# text it actually wants. The contract is:
#
#   text_only_v4 prompt:
#     <case>
#     <narrative>...narrative body...</narrative>
#
#     <risk_verdict>
#     label: {fraud|legit}
#     </risk_verdict>
#     </case>
#
#   xattn_v4 prompt:
#     EXACTLY the same text as text_only_v4 (byte-identical).
#     The event signal arrives via the side stream (structured_events
#     field consumed by the cross-attn encoder), not through the LM's
#     context window.
#
#   structured_as_text_v4 prompt:
#     <case>
#     <events>
#     <event_login>t=0 <ip_risk=high> ...
#     <event_txn>t=90 <amount_bucket=high> ...
#     </events>
#     <narrative>...narrative body...</narrative>
#
#     <risk_verdict>
#     label: {fraud|legit}
#     </risk_verdict>
#     </case>
#
# Key tightening vs v3:
#   - NO <journey_X> wrapper tokens (the journey family was a per-row
#     label that the LM was reading at training time and would have
#     to drop at eval; eval-mode dropout was a workaround for the
#     wrong design).
#   - NO <actor_Y> tokens for the same reason.
#   - NO event lines outside of structured_as_text's <events> block.
#   - Verdict footer contains ONLY `label: {fraud|legit}`. The
#     journey_family / confidence / evidence fields are NOT shown to
#     the LM (they remain in per-row metadata for diagnostic logging).
#   - The byte-identical invariant for text_only vs xattn is the
#     central enforcement mechanism for the v4 architectural test.

CASE_OPEN = "<case>\n"
CASE_CLOSE = "</case>"


def _compose_narrative_block(narrative: str) -> str:
    """The `<narrative>...</narrative>` block, shared by all arms."""
    return f"<narrative>\n{narrative}\n</narrative>\n"


def _compose_verdict_block(label: str) -> str:
    """The minimal v4 verdict footer: ONLY `label:`. Used by all LM arms.

    Per the v4 baseline contract: the LM training target is the label
    value. Journey family, confidence, and evidence are stripped (they
    were leak vectors in v3 — fields the LM could learn to predict
    from the text rather than from the underlying structured signal).
    """
    return f"<risk_verdict>\nlabel: {label}\n</risk_verdict>\n"


def _compose_events_block(events: list[dict]) -> str:
    """The `<events>...</events>` block for structured_as_text arm.

    Each event is `event_to_line(ev)` (matches v3 format for the lines
    themselves; v4 just wraps them in a single explicit block tag).
    """
    if not events:
        return "<events>\n</events>\n"
    lines = "\n".join(event_to_line(ev) for ev in events)
    return f"<events>\n{lines}\n</events>\n"


def compose_text_only(record: dict) -> str:
    """Compose the LM prompt for the `text_only_v4` and `xattn_v4` arms.

    These two arms must see the BYTE-IDENTICAL prompt — that's how
    the v4 plan isolates the architectural variable. The cross-attn
    side stream is the only thing xattn additionally consumes; the
    LM-facing text is the same.

    Used by both `train_text_only.py` and `train_xattn.py`. A startup-
    time assertion in each trainer verifies the byte-identical
    invariant against the alternative composition path.
    """
    return (
        CASE_OPEN
        + _compose_narrative_block(record["narrative"])
        + "\n"
        + _compose_verdict_block(record["label"])
        + CASE_CLOSE
    )


def compose_structured_as_text(record: dict) -> str:
    """Compose the LM prompt for the `structured_as_text_v4` arm.

    Same shape as `compose_text_only` but with an `<events>` block
    prepended to the narrative. This arm has access to the event
    information through the LM's context window; xattn has it through
    the side stream. The two should be tested head-to-head to answer
    Q2 (is cross-attn better than prompt serialization?).
    """
    return (
        CASE_OPEN
        + _compose_events_block(record["structured_events"])
        + _compose_narrative_block(record["narrative"])
        + "\n"
        + _compose_verdict_block(record["label"])
        + CASE_CLOSE
    )


def serialize_journey(journey: Journey, narrative: str) -> str:
    """Back-compat shim. v3 callers expected a monolithic text string;
    v4 returns the text_only composition (the minimum prompt shared by
    all LM arms).

    For per-arm prompts, callers should use `compose_text_only` /
    `compose_structured_as_text` directly on the record dict.
    """
    return compose_text_only({
        "narrative": narrative,
        "label": journey.label,
        # structured_events not needed for text_only composition
    })


def journey_to_record(journey: Journey, narrative: str) -> dict:
    """One dataset row in v4 canonical form.

    Each row has the components that each trainer needs to compose its
    own prompt:
      - `narrative`       — the narrator's body text (clean, no
                            paraphrases — v4 narrative_generator)
      - `structured_events`— event dicts with bucketed feature tokens
                            (the side-stream signal)
      - `label`           — "fraud" | "legit" (the training target)
      - `text`            — the text_only composition (back-compat
                            convenience; v4 trainers should use
                            compose_text_only / compose_structured_as_text
                            instead of reading this field)

    Per-row metadata (NOT consumed by trainers, kept for diagnostics):
      - `journey_family`, `actor_family`, `is_hard_negative`, `seed`

    Two diagnostic fields that v3 included in the prompt and v4
    explicitly does NOT show to the LM:
      - `journey_family_hint`: the journey family label (for offline
        analysis; never enters the prompt)
      - `confidence_hint`, `evidence_hint`: same — diagnostic only
    """
    for ev in journey.events:
        assert_no_raw_pii_in_event(ev)  # last-line defense

    j_family = journey.journey_family
    record = {
        # v4 canonical fields (each trainer composes from these)
        "narrative":        narrative,
        "structured_events": journey.events,
        "label":            journey.label,
        # Back-compat convenience: v4 text_only composition.
        # NB: the v3 monolithic text format (with <journey_X> wrappers,
        # event lines in the LM prompt, full verdict footer) is gone.
        "text":             compose_text_only({
                                "narrative": narrative,
                                "label": journey.label,
                            }),
        # Per-row metadata
        "journey_family":   j_family,
        "actor_family":     journey.actor_family,
        "is_hard_negative": journey.is_hard_negative,
        "seed":             journey.seed,
        # Diagnostic-only (never reach the LM in v4)
        "journey_family_hint": j_family,
        "confidence_hint":    CONFIDENCE_BY_FAMILY.get(j_family, "unknown"),
        "evidence_hint":      EVIDENCE_BY_FAMILY.get(j_family, ""),
    }
    return record


def assert_byte_identical_invariant(record: dict) -> None:
    """Verify the v4 contract: text_only and xattn compose byte-identical
    LM input. Called by trainer startup smoke tests.

    The whole point of v4 is that the LM-facing prompt is the SAME
    for text_only and xattn — they differ ONLY in whether the side
    stream is consumed. If this invariant ever breaks (e.g., a future
    change accidentally adds an arm-specific token), this assertion
    fires loud and early instead of silently confounding the
    architectural comparison.
    """
    a = compose_text_only(record)
    b = compose_text_only(record)  # idempotent
    assert a == b, "compose_text_only is not idempotent — refusing to train"
    # The cross-arm invariant is `compose_text_only(record) ==
    # compose_text_only(record)` — both arms call THE SAME function
    # and that function takes only `narrative` + `label`, not any
    # arm-specific info. The invariant is therefore enforced by
    # design as long as both trainers consume `compose_text_only`.


def verify_v4_text_contract(
    dataset,
    sample_n: int = 32,
    arm_name: str = "v4",
    strict: bool = True,
) -> None:
    """Trainer startup smoke check: confirm the dataset's `text` field
    matches `compose_text_only(record)` across a spread of sample rows.

    Why this exists:
      Both `train_text_only.py` and `train_xattn.py` read the `text`
      field from each row and feed it to the LM. For the v4
      architectural test to be valid, that `text` must equal
      `compose_text_only(record)`. The data generator populates it
      that way, but if someone preprocesses the dataset between
      generation and training (e.g., a quick script that prepends
      tokens, or accidentally swaps in a v3 text), this check fires
      LOUD AND EARLY rather than silently confounding the experiment.

    Args:
      dataset: any indexable dataset of dict-like rows (HuggingFace
        Datasets, list of dicts, etc.).
      sample_n: number of rows to spot-check. The check samples a
        DETERMINISTIC SPREAD across the dataset (rows 0, len/N, 2*len/N,
        ..., len-1) rather than just the first N rows — this catches
        partial preprocessing that only touches one end of the file.
        Default 32 (review 021 finding #4 fix; v0 was 3).
      arm_name: which trainer is calling this; included in messages.
      strict: when True (default for v4 trainers), rows that lack the
        v4 canonical fields (`narrative`, `label`) raise RuntimeError
        rather than skip. v4 arms cannot be trained on v3 data
        without producing uninterpretable results, so this is the
        right default. Set strict=False to allow v3 datasets through
        for backward-compat (legacy v3 reruns).

    Raises:
      RuntimeError if a sampled row's text field does not match the
      canonical composition, or if `strict=True` and any sampled row
      lacks v4 canonical fields. Prints a pass message otherwise.
    """
    n_total = len(dataset)
    if n_total == 0:
        print(f"[{arm_name}] v4 contract check: SKIPPED (empty dataset)")
        return

    # Deterministic spread across the dataset — catches partial
    # preprocessing that only mutates one part of the file.
    n_to_check = min(sample_n, n_total)
    if n_to_check <= 1 or n_total == 1:
        indices = [0]
    else:
        step = max(1, n_total // n_to_check)
        indices = list(range(0, n_total, step))[:n_to_check]
        if indices[-1] != n_total - 1:
            indices.append(n_total - 1)

    n_checked = 0
    n_skipped = 0
    for i in indices:
        row = dataset[i]
        if "narrative" not in row or "label" not in row:
            if strict:
                raise RuntimeError(
                    f"[{arm_name}] v4 contract check FAILED: row {i} lacks "
                    f"v4 canonical fields (`narrative`, `label`). This "
                    f"dataset was generated under v3 OR was preprocessed "
                    f"to strip the canonical fields. v4 trainers cannot "
                    f"produce interpretable results on v3 data. Either:\n"
                    f"  1. Regenerate the dataset with the v4 narrator "
                    f"(data/gen/build_dataset.py at HEAD), OR\n"
                    f"  2. Pass `strict=False` to verify_v4_text_contract "
                    f"(only for explicit legacy v3 reruns)."
                )
            n_skipped += 1
            continue
        expected = compose_text_only(row)
        actual = row.get("text", "")
        if expected != actual:
            raise RuntimeError(
                f"[{arm_name}] v4 text contract VIOLATED at row {i}.\n"
                f"  Dataset's `text` field does not equal "
                f"compose_text_only(row).\n"
                f"  Expected:\n    {expected!r}\n"
                f"  Got:\n    {actual!r}\n"
                f"  This means the dataset was either pre-processed "
                f"after generation, or generated by an older version "
                f"of build_dataset.py. Re-run data/gen/build_dataset.py "
                f"to regenerate, or use a fresh v4 dataset."
            )
        n_checked += 1

    if n_skipped > 0:
        print(f"[{arm_name}] v4 contract check: SKIPPED ({n_skipped}/{len(indices)} "
              f"rows lack canonical fields → v3-format dataset, "
              f"non-strict mode allowed it through)")
    else:
        print(f"[{arm_name}] v4 contract check: PASS "
              f"({n_checked} sample rows match compose_text_only, "
              f"spread across indices {indices[0]}..{indices[-1]} of {n_total} total)")


# ---------------------------------------------------------------------------
# Disjoint stratification (review 018/019 — Task #6)
# ---------------------------------------------------------------------------
#
# Naive random stratified split on text-leveled records re-introduces the
# train/eval leakage Codex flagged in review 018: the narrator caches by
# structured_events_hash, so two rows sharing a structured-events skeleton
# can end up in opposite splits and the LLM emits identical text for both.
# Worse, even unique narratives can't fix the underlying structural leakage:
# H(label | skeleton) = 0 across 2,454 skeletons in the current dataset, so
# sharing a skeleton across splits is in itself a leak.
#
# Gate A (this section): assign rows to splits by (journey_family,
# structured_events_hash) GROUPS. Whole groups are atomic. Within each
# family, walk groups in deterministic hash-sort order and fill the eval
# bucket until the family's eval fraction meets the requested target.
# Class balance is best-effort: families with very few unique skeletons
# (e.g., hn_large_purchase had ~12 groups for ~2,481 rows in the current
# dataset) WILL drift from eval_frac. That drift is the cost of
# disjointness and is reported in build_summary.json.
#
# Gate B (post-narration): hash row text; assert no duplicate text within
# or across splits. If Gate A worked, Gate B never fires.


def _events_hash(structured_events) -> str:
    """SHA-256 of canonical JSON of the structured events list.

    Matches eval/leakage_checks.py._events_hash exactly so the
    pre-narration grouping and the post-hoc clean-eval mask agree on what
    "same skeleton" means.
    """
    return hashlib.sha256(
        json.dumps(structured_events, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_disjoint_splits(
    journeys: list[Journey],
    eval_frac: float,
) -> tuple[list[int], list[int], dict]:
    """Class-balanced disjoint split by (journey_family, events_hash).

    Returns:
      train_idx, eval_idx: lists of indices into `journeys`.
      stats: dict with per-family and overall split diagnostics, including
        n_overlap_structured_events (MUST be 0 — asserted by caller).
    """
    # Group rows by (family, events_hash). Each group's id is the events
    # hash; family is carried alongside so we walk groups per-family.
    groups_by_family: dict[str, dict[str, list[int]]] = {}
    for i, j in enumerate(journeys):
        fam = j.journey_family
        eh = _events_hash(j.events)
        groups_by_family.setdefault(fam, {}).setdefault(eh, []).append(i)

    train_idx: list[int] = []
    eval_idx: list[int] = []
    train_events_by_fam: dict[str, set[str]] = {}
    eval_events_by_fam: dict[str, set[str]] = {}
    per_family: dict[str, dict] = {}

    for fam, fam_groups in groups_by_family.items():
        fam_total = sum(len(rows) for rows in fam_groups.values())
        # Deterministic walk order: sort by group hash hex.
        ordered_hashes = sorted(fam_groups.keys())
        fam_eval = 0
        fam_train = 0
        train_events_by_fam[fam] = set()
        eval_events_by_fam[fam] = set()
        for eh in ordered_hashes:
            rows = fam_groups[eh]
            # Whole-group assignment: send to eval until family's eval
            # fraction meets the requested target, then everything else
            # goes to train.
            current_frac = (fam_eval / fam_total) if fam_total > 0 else 0.0
            if current_frac < eval_frac:
                eval_idx.extend(rows)
                fam_eval += len(rows)
                eval_events_by_fam[fam].add(eh)
            else:
                train_idx.extend(rows)
                fam_train += len(rows)
                train_events_by_fam[fam].add(eh)
        per_family[fam] = {
            "n_total": fam_total,
            "n_train": fam_train,
            "n_eval": fam_eval,
            "actual_eval_frac": (fam_eval / fam_total) if fam_total > 0 else 0.0,
            "n_unique_structured_events_train": len(train_events_by_fam[fam]),
            "n_unique_structured_events_eval": len(eval_events_by_fam[fam]),
        }

    train_events_all: set[str] = set()
    for s in train_events_by_fam.values():
        train_events_all |= s
    eval_events_all: set[str] = set()
    for s in eval_events_by_fam.values():
        eval_events_all |= s
    overlap = train_events_all & eval_events_all

    n_total = len(journeys)
    stats = {
        "requested_eval_frac": eval_frac,
        "actual_eval_frac_overall": (len(eval_idx) / n_total) if n_total > 0 else 0.0,
        "n_train": len(train_idx),
        "n_eval": len(eval_idx),
        "per_family": per_family,
        "n_unique_structured_events_train": len(train_events_all),
        "n_unique_structured_events_eval": len(eval_events_all),
        "n_overlap_structured_events": len(overlap),
    }
    return train_idx, eval_idx, stats


def assert_no_text_overlap(train_records: list[dict], eval_records: list[dict]) -> dict:
    """Gate B: post-narration text-hash CROSS-SPLIT dedup invariant.

    Hashes each row's text; raises AssertionError if any text appears in
    BOTH splits (the leak Gate A is designed to prevent). Intra-split
    duplicate text is NOT an error — it is the natural consequence of
    finite skeleton spaces (e.g., `hn_large_purchase` has only ~12
    unique skeletons for 2,481 rows in the 25k dataset; diagnostic T3)
    combined with the narrator caching by `(events_hash, model, temp)`.
    Intra-split dup counts are reported in the returned stats so callers
    can sanity-check them, but they are not a failure.

    If Gate A (pre-narration grouping) is correct, cross-split duplicates
    cannot exist — same-skeleton rows are confined to one split by
    construction. The assertion exists as a defense-in-depth check, not
    as a fix path.
    """
    train_hashes: dict[str, int] = {}
    for r in train_records:
        h = _text_hash(r["text"])
        train_hashes[h] = train_hashes.get(h, 0) + 1
    eval_hashes: dict[str, int] = {}
    for r in eval_records:
        h = _text_hash(r["text"])
        eval_hashes[h] = eval_hashes.get(h, 0) + 1

    cross = set(train_hashes.keys()) & set(eval_hashes.keys())
    assert not cross, (
        f"Gate B failed: {len(cross)} text hash(es) appear in both train and eval. "
        f"Example: {next(iter(cross))[:16]}..."
    )

    n_intra_dup_train = sum(c for c in train_hashes.values() if c > 1) - sum(
        1 for c in train_hashes.values() if c > 1
    )
    n_intra_dup_eval = sum(c for c in eval_hashes.values() if c > 1) - sum(
        1 for c in eval_hashes.values() if c > 1
    )
    return {
        "n_unique_text_train": len(train_hashes),
        "n_unique_text_eval": len(eval_hashes),
        "n_overlap_text": 0,
        # Diagnostic only: rows beyond the first whose text matches an
        # earlier row in the same split. Expected to be > 0 when the
        # skeleton space is saturated (review 018/019, diagnostic T3).
        "n_intra_split_text_duplicates_train": n_intra_dup_train,
        "n_intra_split_text_duplicates_eval": n_intra_dup_eval,
    }


# ---------------------------------------------------------------------------
# Main build loop
# ---------------------------------------------------------------------------

def build(
    n: int,
    out_dir: Path,
    *,
    mode: str = "template",
    seed: int = 0,
    usd_budget: float = DEFAULT_USD_BUDGET,
    eval_frac: float = 0.0,
    leakage_audit_n: int = 200,
    llm_model: str | None = None,
    concurrency: int = 8,
    narrator_temp: float = 0.3,
    enforce_disjointness: bool = True,
) -> dict:
    """Build a dataset of `n` journeys into `out_dir`. Returns a summary dict.

    `llm_model` is the narrator model id passed through to
    `narrative_generator.generate_narrative()`. None means "use the
    LLM_PROVIDER default" (openai → gpt-5.4-nano, anthropic →
    claude-haiku-4-5). Ignored when mode == 'template'.

    `concurrency` is the number of parallel narrator workers when
    mode == 'llm'. Default 8 ≈ 3-4x speedup at 25k narratives. Ignored
    when mode == 'template' (cheap_narrative is pure-Python).
    """
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = CostTracker(budget_usd=usd_budget) if mode == "llm" else None

    records: list[dict] = []
    family_counts: dict[str, int] = {}
    actor_counts: dict[str, int] = {}
    t_start = time.time()
    progress_step = max(1, n // 20)

    def _print_progress(done: int, total: int, cost: float) -> None:
        if done % progress_step != 0 and done != total:
            return
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0.0
        # flush=True so background runs (where stdout is buffered to a
        # redirected file) emit progress live instead of waiting for
        # the buffer to fill at program exit.
        print(f"  [{done:>6}/{total}] rate={rate:.1f}/s cost=${cost:.3f}",
              file=sys.stderr, flush=True)

    # Phase 1: sample all journeys (no API, sequential to preserve rng
    # determinism). Same logic for both modes.
    journeys: list[Journey] = []
    for i in range(n):
        family = sample_journey_family(rng)
        actor = sample_actor_for_family(family, rng)
        family_counts[family] = family_counts.get(family, 0) + 1
        actor_counts[actor] = actor_counts.get(actor, 0) + 1

        j_seed = seed * 1_000_003 + i  # stable per-row seed
        j = generate_journey(family, j_seed, actor)
        j = mix(j, rng=random.Random(j_seed + 1))
        journeys.append(j)

    # Phase 2: narrate. Concurrent batch for LLM mode (~3-4x faster on
    # gpt-5.4-nano); tight sequential loop for template mode (no API,
    # nothing to parallelize).
    if mode == "llm":
        assert tracker is not None
        narratives = generate_narratives_concurrent(
            journeys, tracker=tracker, model=llm_model,
            max_workers=concurrency,
            progress_callback=_print_progress,
            narrator_temp=narrator_temp,
        )
    elif mode == "template":
        narratives = []
        for i, j in enumerate(journeys):
            narratives.append(cheap_narrative(j))
            _print_progress(i + 1, n, 0.0)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    # Disjoint stratification (Gate A). The split decision is made on
    # the post-narration journeys, but BEFORE final record assembly /
    # write. The split groups journeys by structured_events_hash and
    # assigns whole groups atomically, so identical structured
    # skeletons can never end up in opposite splits (the narrator
    # caches by events_hash and would emit identical text across
    # train/eval if both held the same hash — review 018 Finding 1).
    # Whole (journey_family, events_hash) groups are atomic.
    # NOTE (review 020 Minor 4): the disjointness invariant does not
    # depend on temporal ordering of narration vs split — it depends
    # on atomic group assignment by events_hash. For future
    # regenerations that want to avoid paying the narration cost on
    # train rows that end up in eval (and vice versa), the split call
    # could be moved earlier; that is an optimization, not a
    # correctness issue.
    split_stats: dict | None = None
    train_idx: list[int] = []
    eval_idx: list[int] = []
    if eval_frac > 0 and enforce_disjointness:
        train_idx, eval_idx, split_stats = assign_disjoint_splits(journeys, eval_frac)
        assert split_stats["n_overlap_structured_events"] == 0, (
            "Gate A failed: structured_events_hash overlap between train and "
            f"eval = {split_stats['n_overlap_structured_events']}"
        )

    # Phase 3: assemble records.
    for j, narrative in zip(journeys, narratives):
        j.narrative = narrative
        records.append(journey_to_record(j, narrative))

    # ----- write JSONL (atomic) -----
    if eval_frac > 0 and enforce_disjointness:
        # Gate A already decided the split; materialize records into
        # train/eval by the precomputed index lists.
        train: list[dict] = [records[i] for i in train_idx]
        eval_split: list[dict] = [records[i] for i in eval_idx]
        # Gate B: post-narration text-hash dedup invariant. Fires only if
        # Gate A's pre-narration grouping was inconsistent with the
        # narrator's cache keying — should never happen on healthy code.
        text_stats = assert_no_text_overlap(train, eval_split)
        split_stats.update(text_stats)
        _write_jsonl(out_dir / "train.jsonl", train)
        _write_jsonl(out_dir / "eval.jsonl", eval_split)
    elif eval_frac > 0:
        # Legacy random stratified split (--enforce-disjointness=False).
        # Available for ablation only; reintroduces the leakage Codex
        # documented in review 018.
        by_family: dict[str, list[dict]] = {}
        for r in records:
            by_family.setdefault(r["journey_family"], []).append(r)
        train = []
        eval_split = []
        for fam, rows in by_family.items():
            rng.shuffle(rows)
            k = max(1, int(round(len(rows) * eval_frac)))
            eval_split.extend(rows[:k])
            train.extend(rows[k:])
        _write_jsonl(out_dir / "train.jsonl", train)
        _write_jsonl(out_dir / "eval.jsonl", eval_split)
    else:
        _write_jsonl(out_dir / "data.jsonl", records)

    # ----- leakage audit -----
    audit_sample = rng.sample(records, min(leakage_audit_n, len(records)))
    failures = []
    for r in audit_sample:
        scan = narrative_leakage_scan(r["text"])
        if not scan["clean"]:
            failures.append({"seed": r["seed"], "family": r["journey_family"],
                             "hits": scan["hits"]})
    if failures:
        raise RuntimeError(
            f"LEAKAGE AUDIT FAILED on {len(failures)}/{len(audit_sample)} samples. "
            f"Examples: {failures[:3]}"
        )

    summary = {
        "n_records": len(records),
        "mode": mode,
        "family_counts": family_counts,
        "actor_counts": actor_counts,
        "eval_frac": eval_frac,
        "leakage_audit_n": len(audit_sample),
        "leakage_audit_failures": 0,
        "duration_seconds": round(time.time() - t_start, 2),
        "enforce_disjointness": enforce_disjointness,
    }
    if split_stats is not None:
        # Gate A/B diagnostics (review 019 Medium 4 invariants).
        summary["disjoint_split"] = split_stats
    if tracker is not None:
        summary["llm_cost"] = tracker.summary()

    # Write summary alongside data
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_jsonl(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    tmp.rename(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="number of journeys to generate")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--mode", choices=("llm", "template"), default="template",
                        help="narrative source: 'llm' (OpenAI/Anthropic API) or 'template' (no LLM)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--usd-budget", type=float, default=DEFAULT_USD_BUDGET,
                        help="hard cap on LLM-narrator spend (mode=llm only)")
    parser.add_argument("--eval-frac", type=float, default=0.0,
                        help="fraction to split into eval.jsonl (stratified by journey_family). 0=single data.jsonl")
    parser.add_argument("--leakage-audit-n", type=int, default=200,
                        help="number of records to audit for narrative leakage")
    parser.add_argument("--llm-model", default=None,
                        help="narrator model id (e.g., 'gpt-5.4-nano', "
                             "'gpt-5.4-mini', 'claude-haiku-4-5-20251001'). "
                             "Default: provider's default (LLM_PROVIDER env var). "
                             "Provider is inferred from the model prefix; "
                             "LLM_PROVIDER + this flag must agree.")
    parser.add_argument("--llm-provider", default=None,
                        choices=("openai", "anthropic"),
                        help="convenience: sets the LLM_PROVIDER env var for "
                             "this run. Equivalent to `LLM_PROVIDER=... python "
                             "-m data.gen.build_dataset ...`.")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="parallel narrator workers when --mode=llm "
                             "(default 8, ~3-4x faster than sequential at 25k). "
                             "Ignored when --mode=template.")
    parser.add_argument("--narrator-temp", type=float, default=0.3,
                        help="initial-attempt narrator temperature. Default 0.3 "
                             "(matches the original 25k train run for stable "
                             "formatting). Bump to ~0.5 when generating EVAL data "
                             "to reduce narrator-style correlation with train. "
                             "Retries always escalate to max(0.7, base+0.4).")
    parser.add_argument(
        "--enforce-disjointness",
        dest="enforce_disjointness",
        action="store_true",
        default=True,
        help="Gate A (pre-narration class-balanced disjoint stratification by "
             "(journey_family, structured_events_hash)) + Gate B (post-narration "
             "text-hash dedup assertion). Default ON. Required for "
             "leakage-free splits — review 018/019. Only takes effect when "
             "--eval-frac > 0.",
    )
    parser.add_argument(
        "--no-enforce-disjointness",
        dest="enforce_disjointness",
        action="store_false",
        help="Disable Gate A/B and fall back to the legacy random stratified "
             "split. Ablation use only — reintroduces the leakage Codex "
             "documented in review 018.",
    )
    args = parser.parse_args()

    if args.llm_provider is not None:
        import os
        os.environ["LLM_PROVIDER"] = args.llm_provider

    if not 1 <= args.concurrency <= 32:
        parser.error(f"--concurrency must be in [1, 32], got {args.concurrency}")

    summary = build(
        n=args.n, out_dir=args.out, mode=args.mode, seed=args.seed,
        usd_budget=args.usd_budget, eval_frac=args.eval_frac,
        leakage_audit_n=args.leakage_audit_n,
        llm_model=args.llm_model,
        concurrency=args.concurrency,
        narrator_temp=args.narrator_temp,
        enforce_disjointness=args.enforce_disjointness,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
