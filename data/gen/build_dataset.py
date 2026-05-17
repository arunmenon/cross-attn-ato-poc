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

# Journey-family sampling weights (relative). Total fraud ~30%, hn ~30%, clean ~40%.
JOURNEY_WEIGHTS: dict[str, float] = {
    "clean":                40.0,   # 40%
    "cred_stuff":            6.0,
    "sim_swap":              6.0,
    "phish_takeover":        6.0,
    "malware_rat":           6.0,
    "mule_chain":            6.0,   # total fraud = 30%
    "hn_travel":            10.0,
    "hn_large_purchase":    10.0,
    "hn_account_recovery":  10.0,   # total hn = 30%
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
}

# Verdict-footer evidence strings, per journey family. Deterministic (not
# LLM-generated) so they cannot leak — and so we can use them as a
# scoring target without ambiguity.
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
}

CONFIDENCE_BY_FAMILY: dict[str, str] = {
    "clean": "high", "cred_stuff": "high", "sim_swap": "high",
    "phish_takeover": "high", "malware_rat": "medium",
    "mule_chain": "high", "hn_travel": "medium",
    "hn_large_purchase": "medium", "hn_account_recovery": "high",
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


def serialize_journey(journey: Journey, narrative: str) -> str:
    """Produce the full corpus-format text for a journey.

    Structure:
      <journey_X>
      <actor_Y>
      <event_login>t=0 <bucket tokens...>
      ...
      <event_txn>t=N <bucket tokens...>

      <narrative>
      ...body...
      </narrative>

      <risk_verdict>
      label: fraud|legit
      journey_family: X
      confidence: ...
      evidence: ...
      </risk_verdict>
      </journey_X>
    """
    j_family = journey.journey_family
    open_tag = f"<journey_{j_family}>"
    close_tag = f"</journey_{j_family}>"
    actor_tag = f"<actor_{journey.actor_family}>"

    event_lines = "\n".join(_event_to_line(ev) for ev in journey.events)
    verdict_block = (
        f"<risk_verdict>\n"
        f"label: {journey.label}\n"
        f"journey_family: {j_family}\n"
        f"confidence: {CONFIDENCE_BY_FAMILY[j_family]}\n"
        f"evidence: {EVIDENCE_BY_FAMILY[j_family]}\n"
        f"</risk_verdict>"
    )

    return (
        f"{open_tag}\n"
        f"{actor_tag}\n"
        f"{event_lines}\n\n"
        f"<narrative>\n{narrative}\n</narrative>\n\n"
        f"{verdict_block}\n"
        f"{close_tag}"
    )


def journey_to_record(journey: Journey, narrative: str) -> dict:
    """One dataset row: full HF-Dataset-style dict."""
    for ev in journey.events:
        assert_no_raw_pii_in_event(ev)  # last-line defense
    return {
        "text": serialize_journey(journey, narrative),
        "structured_events": journey.events,
        "journey_family": journey.journey_family,
        "actor_family": journey.actor_family,
        "label": journey.label,
        "is_hard_negative": journey.is_hard_negative,
        "seed": journey.seed,
    }


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
