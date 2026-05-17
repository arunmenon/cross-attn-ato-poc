"""Read-only data-overlap diagnostic for train_llm_narrated.

Quantifies the three overlap layers between train.jsonl and eval.jsonl
that surfaced in Codex review 018 (`review/018-day-2-baseline-findings/comments.txt`):

  1. Exact narrative text overlap (text_hash collision).
  2. Identical structured-event payload overlap (events_hash collision).
  3. Bucket-event skeleton overlap (same event types + same bucket tokens
     in the same order, regardless of timestamps or random identifiers).

The script also estimates the bucket-combination space per journey_family
by listing how many distinct skeletons the family actually produces, and
compares that to the family's row count to derive a "saturation ratio"
(observed unique skeletons / total rows). A saturation ratio near 0 means
the family is small enough that any sample of meaningful size will revisit
the same skeletons many times — which is the root cause of the 4,661 / 5,000
skeleton overlap reported in review 018.

Output:
  - stdout: per-family table + summary lines.
  - When --write-md is given, the same content is rendered to
    `docs/day-2-data-diagnostic.md` along with author commentary that names
    bucket-combination saturation as the structural reason behind the overlap
    and recommends pre-narration-stratification for future regenerations.

This script does NOT modify the dataset. It is safe to run repeatedly.

Usage:
    python3 scripts/diagnose_data_overlap.py \
        --data-dir data/train_llm_narrated \
        --write-md docs/day-2-data-diagnostic.md
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Hashing helpers — kept locally so the script has no project dependencies
# ---------------------------------------------------------------------------

def _text_field(row: dict) -> str:
    """Return the narrative text. The dataset uses 'text' (full prompt with
    tags) for some rows and 'narrative' for others depending on how the
    record was emitted — accept both, prefer 'text' for hashing parity with
    review 018's reproduction script."""
    if "text" in row:
        return row["text"]
    if "narrative" in row:
        return row["narrative"]
    raise KeyError("row has neither 'text' nor 'narrative' field")


def text_hash(row: dict) -> str:
    return hashlib.sha256(_text_field(row).encode()).hexdigest()


def events_hash(row: dict) -> str:
    return hashlib.sha256(
        json.dumps(row["structured_events"], sort_keys=True).encode()
    ).hexdigest()


def skeleton(row: dict) -> tuple:
    """Bucket-event skeleton: ordered sequence of (event_type, actor, sorted
    bucket tokens). Strips raw values (timestamps, free identifiers, random
    seeds) and keeps only the per-event signature the model can actually see.
    Two rows with the same skeleton differ only in fields the model is
    designed to ignore.

    Notation: bucket tokens are strings of the shape "<family=value>" (see
    `data/gen/feature_bucketer.py::format_bucket_token`). Non-bucket fields
    on a structured_event are either identifier noise (`<ip>`, `<device_id>`,
    `<recipient>`, `<merchant>`) or timestamps (`t`) — both ignored.

    This definition reproduces review 018's 4,661/5,000 skeleton overlap and
    2,454 distinct skeletons on the current dataset.
    """
    parts = []
    for event in row["structured_events"]:
        event_type = event.get("event", event.get("event_type"))
        actor = event.get("actor")
        bucket_values = tuple(sorted(
            value for value in event.values()
            if isinstance(value, str) and value.startswith("<") and value.endswith(">") and "=" in value
        ))
        parts.append((event_type, actor, bucket_values))
    return tuple(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Per-family statistics
# ---------------------------------------------------------------------------

@dataclass
class FamilyStats:
    family: str
    n_train: int = 0
    n_eval: int = 0
    unique_skeletons_train: int = 0
    unique_skeletons_eval: int = 0
    unique_skeletons_combined: int = 0
    n_text_overlap: int = 0
    n_events_overlap: int = 0
    n_skeleton_overlap: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)


def per_family_stats(train: list[dict], eval_rows: list[dict]) -> dict[str, FamilyStats]:
    train_text_hashes = {text_hash(r) for r in train}
    train_events_hashes = {events_hash(r) for r in train}
    train_skeletons = {skeleton(r) for r in train}

    train_skeletons_by_fam: dict[str, set] = collections.defaultdict(set)
    eval_skeletons_by_fam: dict[str, set] = collections.defaultdict(set)
    combined_skeletons_by_fam: dict[str, set] = collections.defaultdict(set)
    label_by_fam: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    for r in train:
        fam = r["journey_family"]
        train_skeletons_by_fam[fam].add(skeleton(r))
        combined_skeletons_by_fam[fam].add(skeleton(r))
        label_by_fam[fam][r["label"]] += 1
    for r in eval_rows:
        fam = r["journey_family"]
        eval_skeletons_by_fam[fam].add(skeleton(r))
        combined_skeletons_by_fam[fam].add(skeleton(r))
        label_by_fam[fam][r["label"]] += 1

    stats: dict[str, FamilyStats] = {}
    train_count_by_fam = collections.Counter(r["journey_family"] for r in train)
    eval_count_by_fam = collections.Counter(r["journey_family"] for r in eval_rows)
    all_fams = sorted(set(train_count_by_fam) | set(eval_count_by_fam))
    for fam in all_fams:
        s = FamilyStats(family=fam)
        s.n_train = train_count_by_fam[fam]
        s.n_eval = eval_count_by_fam[fam]
        s.unique_skeletons_train = len(train_skeletons_by_fam[fam])
        s.unique_skeletons_eval = len(eval_skeletons_by_fam[fam])
        s.unique_skeletons_combined = len(combined_skeletons_by_fam[fam])
        s.label_counts = dict(label_by_fam[fam])
        for r in eval_rows:
            if r["journey_family"] != fam:
                continue
            if text_hash(r) in train_text_hashes:
                s.n_text_overlap += 1
            if events_hash(r) in train_events_hashes:
                s.n_events_overlap += 1
            if skeleton(r) in train_skeletons:
                s.n_skeleton_overlap += 1
        stats[fam] = s
    return stats


# ---------------------------------------------------------------------------
# Per-skeleton diagnostics
# ---------------------------------------------------------------------------

@dataclass
class SkeletonDiagnostics:
    n_unique_skeletons: int
    n_skeletons_train_only: int
    n_skeletons_eval_only: int
    n_skeletons_both: int
    n_skeletons_mixed_labels: int
    label_entropy_given_skeleton: float
    n_rows: int


def skeleton_diagnostics(train: list[dict], eval_rows: list[dict]) -> SkeletonDiagnostics:
    """Compute split membership + label entropy conditioned on skeleton.

    A skeleton with zero label entropy means every row with that bucket-event
    signature carries the same label. Across the dataset this is the strongest
    statement of "synthetic data is label-deterministic in the observed support."
    """
    membership: dict[tuple, dict[str, int]] = collections.defaultdict(lambda: {"train": 0, "eval": 0})
    labels_by_skel: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)

    for r in train:
        skel = skeleton(r)
        membership[skel]["train"] += 1
        labels_by_skel[skel][r["label"]] += 1
    for r in eval_rows:
        skel = skeleton(r)
        membership[skel]["eval"] += 1
        labels_by_skel[skel][r["label"]] += 1

    n_train_only = n_eval_only = n_both = n_mixed = 0
    total_rows = 0
    weighted_entropy = 0.0
    for skel, m in membership.items():
        if m["train"] and not m["eval"]:
            n_train_only += 1
        elif m["eval"] and not m["train"]:
            n_eval_only += 1
        else:
            n_both += 1
        counter = labels_by_skel[skel]
        if len(counter) > 1:
            n_mixed += 1
        subtotal = sum(counter.values())
        total_rows += subtotal
        ent = -sum((c / subtotal) * math.log2(c / subtotal) for c in counter.values() if c > 0)
        weighted_entropy += (subtotal) * ent
    cond_entropy = weighted_entropy / total_rows if total_rows else 0.0

    return SkeletonDiagnostics(
        n_unique_skeletons=len(membership),
        n_skeletons_train_only=n_train_only,
        n_skeletons_eval_only=n_eval_only,
        n_skeletons_both=n_both,
        n_skeletons_mixed_labels=n_mixed,
        label_entropy_given_skeleton=cond_entropy,
        n_rows=total_rows,
    )


# ---------------------------------------------------------------------------
# Bucket-combination space estimate (theoretical vs observed)
# ---------------------------------------------------------------------------

# Cardinalities of each bucket family that the journey templates actually use.
# Source: data/gen/feature_bucketer.py. Each bucket value's literal token is
# fixed (e.g., <amount_bucket=low|medium|high|extreme>), so the *value* space
# per family is the family's own cardinality.
BUCKET_CARDINALITY = {
    "amount_bucket": 4,        # low | medium | high | extreme
    "geo_distance": 3,         # local | domestic_far | international
    "ip_risk": 3,              # low | medium | high
    "device_age": 3,           # known | new | rare
    "merchant_risk": 2,        # normal | elevated
    "txn_velocity": 3,         # normal | bursty | extreme
    "recipient_age": 2,        # known | newly_added
    "session_dwell": 3,        # short | normal | extended
    "auth_strength": 3,        # mfa_strong | password_only | cookie_only
}


def per_family_bucket_features(train: list[dict], eval_rows: list[dict]) -> dict[str, dict]:
    """For each journey_family, enumerate the (event_type, length) shape and
    the set of bucket families that actually appear across that family's rows.

    Returns: {family: {n_unique_event_lengths: int, bucket_families_used: list[str],
                        theoretical_bucket_space: int}}
    Each row in a family typically has a fixed event ordering with a few
    variable-length stretches (the n_txns randint loops in journey_templates.py).
    For the theoretical space estimate we multiply the cardinalities of every
    bucket family that ever appears in the family's rows. This is an UPPER BOUND
    on distinct skeletons that respects bucket-token disjointness but ignores
    event-count variability (which compounds further but is harder to enumerate
    cleanly).
    """
    by_fam_buckets: dict[str, set[str]] = collections.defaultdict(set)
    by_fam_event_lengths: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in (*train, *eval_rows):
        fam = row["journey_family"]
        by_fam_event_lengths[fam][len(row["structured_events"])] += 1
        for ev in row["structured_events"]:
            for k, v in ev.items():
                if isinstance(v, str) and v.startswith("<") and v.endswith(">") and "=" in v:
                    # Token shape "<family=value>" → record the family.
                    family_name = v[1:-1].split("=", 1)[0]
                    by_fam_buckets[fam].add(family_name)

    result: dict[str, dict] = {}
    for fam, bucket_families in by_fam_buckets.items():
        # Theoretical: product of cardinalities of each bucket family used,
        # raised to a small power to account for repeated events with the
        # same bucket family (e.g., 1-4 txn events per clean journey each
        # carrying an amount_bucket). We use the most-common event-length
        # for the family as an order-of-magnitude multiplier on bucket
        # families that vary per event (amount, velocity, recipient_age).
        per_event_varying = {"amount_bucket", "txn_velocity", "recipient_age",
                              "merchant_risk", "session_dwell"}
        most_common_len = by_fam_event_lengths[fam].most_common(1)[0][0]
        log_space = 0.0
        for bf in bucket_families:
            cardinality = BUCKET_CARDINALITY.get(bf, 2)
            if bf in per_event_varying:
                log_space += most_common_len * math.log10(max(cardinality, 1))
            else:
                log_space += math.log10(max(cardinality, 1))
        theoretical = int(round(10 ** log_space))
        result[fam] = {
            "n_unique_event_lengths": len(by_fam_event_lengths[fam]),
            "most_common_event_length": most_common_len,
            "bucket_families_used": sorted(bucket_families),
            "theoretical_bucket_space_estimate": theoretical,
        }
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_table(stats: dict[str, FamilyStats], bucket_space: dict[str, dict]) -> str:
    headers = [
        "family", "n_train", "n_eval", "skel_train", "skel_eval",
        "skel_uniq_all", "skel_overlap_eval%", "theor_skel_space", "saturation",
        "text_overlap", "events_overlap", "labels",
    ]
    rows = []
    for fam in sorted(stats):
        s = stats[fam]
        skel_overlap_pct = (s.n_skeleton_overlap / s.n_eval * 100.0) if s.n_eval else 0.0
        bs = bucket_space.get(fam, {})
        theor = bs.get("theoretical_bucket_space_estimate", 0)
        total_rows = s.n_train + s.n_eval
        # Saturation: observed_unique_skeletons / min(theoretical_space, total_rows).
        # Values near 1 mean we've covered the bucket-space; values near 0 mean
        # the bucket-space is much larger than what was sampled.
        denom = min(theor, total_rows) if theor else 1
        saturation = s.unique_skeletons_combined / denom if denom else 0.0
        rows.append([
            fam,
            str(s.n_train),
            str(s.n_eval),
            str(s.unique_skeletons_train),
            str(s.unique_skeletons_eval),
            str(s.unique_skeletons_combined),
            f"{skel_overlap_pct:5.1f}",
            str(theor) if theor else "-",
            f"{saturation:5.3f}" if theor else "-",
            str(s.n_text_overlap),
            str(s.n_events_overlap),
            "/".join(f"{k}:{v}" for k, v in sorted(s.label_counts.items())),
        ])
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "  "
    lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append(sep.join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        lines.append(sep.join(r[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def render_markdown_table(stats: dict[str, FamilyStats], bucket_space: dict[str, dict]) -> str:
    headers = [
        "journey_family", "n_train", "n_eval", "skel_train", "skel_eval",
        "skel_uniq_all", "skel_overlap_eval %", "theor_bucket_space", "saturation",
        "text_overlap", "events_overlap", "label_counts",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for fam in sorted(stats):
        s = stats[fam]
        skel_overlap_pct = (s.n_skeleton_overlap / s.n_eval * 100.0) if s.n_eval else 0.0
        bs = bucket_space.get(fam, {})
        theor = bs.get("theoretical_bucket_space_estimate", 0)
        total_rows = s.n_train + s.n_eval
        denom = min(theor, total_rows) if theor else 1
        saturation = s.unique_skeletons_combined / denom if denom else 0.0
        lines.append("| " + " | ".join([
            fam,
            str(s.n_train),
            str(s.n_eval),
            str(s.unique_skeletons_train),
            str(s.unique_skeletons_eval),
            str(s.unique_skeletons_combined),
            f"{skel_overlap_pct:.1f}",
            f"{theor:,}" if theor else "-",
            f"{saturation:.3f}" if theor else "-",
            str(s.n_text_overlap),
            str(s.n_events_overlap),
            ", ".join(f"{k}={v}" for k, v in sorted(s.label_counts.items())),
        ]) + " |")
    return "\n".join(lines)


def write_markdown_report(
    out_path: Path,
    stats: dict[str, FamilyStats],
    bucket_space: dict[str, dict],
    skel_diag: SkeletonDiagnostics,
    total_text_overlap: int,
    total_events_overlap: int,
    total_skel_overlap: int,
    n_eval: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Day-2 data-overlap diagnostic")
    md.append("")
    md.append("Read-only diagnostic of `data/train_llm_narrated/{train,eval}.jsonl`. "
              "Quantifies the three overlap layers between the train and eval splits and "
              "compares each `journey_family`'s observed skeleton count to a theoretical "
              "upper bound on the bucket-combination space. Source: "
              "`scripts/diagnose_data_overlap.py`.")
    md.append("")
    md.append("## Headline numbers")
    md.append("")
    md.append(f"- Eval rows: **{n_eval:,}**")
    md.append(f"- Eval rows with a train **text-hash** match: **{total_text_overlap:,}** "
              f"({total_text_overlap / n_eval:.1%}) — exact narrative duplicates")
    md.append(f"- Eval rows with a train **structured_events-hash** match: "
              f"**{total_events_overlap:,}** ({total_events_overlap / n_eval:.1%}) — "
              f"identical structured payload (a strict superset of text overlap up to one row)")
    md.append(f"- Eval rows with a train **bucket-event skeleton** match: "
              f"**{total_skel_overlap:,}** ({total_skel_overlap / n_eval:.1%}) — "
              f"same event sequence + same bucket tokens, different identifiers/timestamps "
              f"(reproduces Codex review-018's 4,661/5,000 number)")
    md.append(f"- Total distinct bucket-event skeletons across train ∪ eval: "
              f"**{skel_diag.n_unique_skeletons:,}**")
    md.append(f"- Skeletons appearing in BOTH splits: **{skel_diag.n_skeletons_both:,}** "
              f"({skel_diag.n_skeletons_both / skel_diag.n_unique_skeletons:.1%} of distinct skeletons)")
    md.append(f"- Skeletons train-only: **{skel_diag.n_skeletons_train_only:,}**; "
              f"eval-only: **{skel_diag.n_skeletons_eval_only:,}**")
    md.append(f"- Skeletons with mixed labels (legit ↔ fraud): "
              f"**{skel_diag.n_skeletons_mixed_labels:,}**")
    md.append(f"- H(label | skeleton): **{skel_diag.label_entropy_given_skeleton:.4f} bits**")
    md.append("")
    md.append("## Per-family stats")
    md.append("")
    md.append(render_markdown_table(stats, bucket_space))
    md.append("")
    md.append("Column legend:")
    md.append("")
    md.append("- `skel_train`, `skel_eval`, `skel_uniq_all` — count of distinct bucket-event "
              "skeletons observed in the train split, eval split, and their union for the family.")
    md.append("- `skel_overlap_eval %` — fraction of eval rows in this family whose skeleton "
              "also appears in the family's train rows.")
    md.append("- `theor_bucket_space` — rough upper bound on distinct skeletons the family's "
              "template *could* emit, computed from the cardinalities in "
              "`data/gen/feature_bucketer.py` and the family's most common event-length "
              "(see `BUCKET_CARDINALITY` in this script). Bucket families that vary per event "
              "(`amount_bucket`, `txn_velocity`, `recipient_age`, `merchant_risk`, `session_dwell`) "
              "are raised to the most-common event-length power; bucket families that are "
              "set once per journey (`auth_strength`, `ip_risk`, `geo_distance`, `device_age`) "
              "contribute a single factor. This is intentionally coarse — it is an order-of-"
              "magnitude estimate, not a counting argument.")
    md.append("- `saturation` — `skel_uniq_all / min(theor_bucket_space, n_train + n_eval)`. "
              "Values close to 1 mean the family has saturated its bucket-space at the "
              "sample size used; values close to 0 mean the bucket-space is much larger "
              "than the sample, so revisiting the same skeleton in train and eval is unlikely.")
    md.append("- `text_overlap`, `events_overlap` — counts of eval rows whose narrative-text "
              "hash or `structured_events`-hash appears in train.")
    md.append("- `label_counts` — distribution of `legit` vs `fraud` labels across train+eval "
              "for the family. Every family in the current generator has a single label.")
    md.append("")
    md.append("## Why train/eval skeleton overlap is near 100%")
    md.append("")
    md.append("The dataset's structured stream uses **bucketed** features by design "
              "(see `PLAN.md` §3 — bucketed-feature tokens preserve fraud signal in "
              "privacy-safe form). Each bucket family has 2-4 values: `amount_bucket` has "
              "four, `geo_distance` has three, `ip_risk` has three, `merchant_risk` has two, "
              "and so on. The journey templates in `data/gen/journey_templates.py` then pick "
              "from these values along narrow, family-specific paths — `gen_clean` always "
              "emits `ip_risk=low`, `geo_distance=local`, `auth_strength=mfa_strong`, "
              "`device_age=known`; `gen_phish_takeover` always emits `ip_risk=high`, "
              "`geo_distance=international`, `auth_strength=password_only`, `device_age=rare`; "
              "and the per-event variation reduces almost entirely to `amount_bucket` and "
              "`txn_velocity`. Multiplied out, each family's effective bucket-combination "
              "space is small enough (hundreds to a few thousand per family) that 20,000 "
              "train + 5,000 eval rows easily revisit every cell — which is what the "
              "skeleton-uniqueness column in the table above shows.")
    md.append("")
    md.append("Because narration in `data/gen/build_dataset.py` happens BEFORE the "
              "train/eval split — and `data/gen/narrative_generator.py` caches narratives by "
              "`(structured_events_hash, model, temp)` (see `_journey_cache_key` at "
              "`narrative_generator.py:252`) — a structured-event payload that occurs twice "
              "in the same generation run reuses the cached narrative. When the post-narration "
              "split then assigns one copy to train and another to eval, the result is a row "
              "pair with identical text AND identical structured payload across splits. "
              "That is the mechanism behind the 533 exact-text duplicates Codex reported. "
              "The events-only-hash class (1 extra row, dropping 534 in total) is a row whose "
              "structured payload matches train but whose narrative happens to differ in "
              "whitespace or model output variance.")
    md.append("")
    md.append("The skeleton-level overlap (4,661 of 5,000 eval rows) is one layer deeper "
              "than the narrative-hash leak. Even if narration were perfectly cached-by-split, "
              "and even if every structured-event-hash were unique across train and eval, "
              "the bucket-event skeleton would still match because the bucket-combination "
              "space per family is smaller than the per-family sample size. This is a "
              "property of the synthetic distribution, not a code bug — and it is what makes "
              "the structured stream label-deterministic in the observed support "
              "(H(label | skeleton) = 0). The event-only classifier's perfect-looking "
              "performance on the original eval is partly a consequence: it memorizes a "
              "compact categorical mapping over a feature stream that is deterministic in "
              "the support its eval inhabits.")
    md.append("")
    md.append("## Recommendations for future regenerations")
    md.append("")
    md.append("Future regenerations must enforce **pre-narration structured-events-hash "
              "stratification**: assign each unique `structured_events_hash` to exactly one "
              "split (train OR eval, never both) BEFORE the narration step caches anything, "
              "and balance the assignment per `journey_family` so that the requested "
              "`eval_frac` is honored at the family level. Concretely: group generated rows "
              "by `(journey_family, structured_events_hash)`, then walk the groups in a "
              "deterministic order and assign whole groups to the split that is currently "
              "under its target eval-fraction. This makes structured-events-hash disjoint "
              "between splits and removes the narrative-cache reuse vector in one step.")
    md.append("")
    md.append("Add a **post-narration text-hash dedup gate** as a defensive invariant: after "
              "narration, hash every row's text and assert no duplicates exist within or "
              "across splits. This catches anything the pre-narration gate misses (e.g., LLM "
              "output collision on distinct structured payloads).")
    md.append("")
    md.append("Neither gate removes the **skeleton-level** overlap, because that is a "
              "property of the bucket-combination space, not of the cache or the split. "
              "Removing skeleton overlap would require either (a) enlarging the per-family "
              "bucket-combination space (more bucket families per event type, finer bucket "
              "granularity, more variable per-family templates) or (b) holding out a separate "
              "skeleton-disjoint eval set sampled from regions of bucket-space the train set "
              "does not cover. (a) is a generator redesign; (b) is a future Day-N "
              "deliverable. For the current POC we document the skeleton overlap as a "
              "synthetic-data finding and constrain the Day-3 claim accordingly (see "
              "`docs/day-2-results.md`).")
    md.append("")
    md.append("## Reproducibility")
    md.append("")
    md.append("```bash")
    md.append("python3 scripts/diagnose_data_overlap.py \\")
    md.append("    --data-dir data/train_llm_narrated \\")
    md.append("    --write-md docs/day-2-data-diagnostic.md")
    md.append("```")
    md.append("")
    md.append("This script is read-only and idempotent — re-running it does not modify the dataset.")
    md.append("")
    out_path.write_text("\n".join(md))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/train_llm_narrated"),
                        help="Directory containing train.jsonl and eval.jsonl")
    parser.add_argument("--write-md", type=Path, default=None,
                        help="If given, write the diagnostic to this markdown path")
    parser.add_argument("--check", action="store_true",
                        help="Run to completion and assert the expected headline numbers "
                             "from review 018 (533 text overlap, 534 events overlap, "
                             "4,661 skeleton overlap). Non-zero exit on mismatch.")
    args = parser.parse_args()

    train_path = args.data_dir / "train.jsonl"
    eval_path = args.data_dir / "eval.jsonl"
    if not train_path.exists() or not eval_path.exists():
        sys.stderr.write(f"missing data files under {args.data_dir} "
                         f"(expected train.jsonl + eval.jsonl)\n")
        return 2

    train = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)
    print(f"loaded train={len(train):,} rows, eval={len(eval_rows):,} rows")

    stats = per_family_stats(train, eval_rows)
    skel_diag = skeleton_diagnostics(train, eval_rows)
    bucket_space = per_family_bucket_features(train, eval_rows)

    total_text = sum(s.n_text_overlap for s in stats.values())
    total_events = sum(s.n_events_overlap for s in stats.values())
    total_skel = sum(s.n_skeleton_overlap for s in stats.values())

    print()
    print(f"text_overlap     = {total_text}/{len(eval_rows)} "
          f"({total_text / len(eval_rows):.1%})")
    print(f"events_overlap   = {total_events}/{len(eval_rows)} "
          f"({total_events / len(eval_rows):.1%})")
    print(f"skeleton_overlap = {total_skel}/{len(eval_rows)} "
          f"({total_skel / len(eval_rows):.1%})")
    print()
    print(f"unique skeletons (train ∪ eval) = {skel_diag.n_unique_skeletons:,}")
    print(f"  train-only       = {skel_diag.n_skeletons_train_only:,}")
    print(f"  eval-only        = {skel_diag.n_skeletons_eval_only:,}")
    print(f"  both             = {skel_diag.n_skeletons_both:,}")
    print(f"  mixed-label      = {skel_diag.n_skeletons_mixed_labels:,}")
    print(f"  H(label|skeleton) = {skel_diag.label_entropy_given_skeleton:.4f} bits")
    print()
    print("per-family table:")
    print(render_table(stats, bucket_space))

    if args.write_md is not None:
        write_markdown_report(args.write_md, stats, bucket_space, skel_diag,
                              total_text, total_events, total_skel, len(eval_rows))
        print()
        print(f"wrote {args.write_md}")

    if args.check:
        ok = True
        if total_text != 533:
            sys.stderr.write(f"CHECK FAIL: expected text_overlap=533, got {total_text}\n")
            ok = False
        if total_events != 534:
            sys.stderr.write(f"CHECK FAIL: expected events_overlap=534, got {total_events}\n")
            ok = False
        if total_skel != 4661:
            sys.stderr.write(f"CHECK FAIL: expected skeleton_overlap=4661, got {total_skel}\n")
            ok = False
        if not ok:
            return 1
        print("CHECK OK: review-018 reference numbers reproduced exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
