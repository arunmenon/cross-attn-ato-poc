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
        )
    elif mode == "template":
        narratives = []
        for i, j in enumerate(journeys):
            narratives.append(cheap_narrative(j))
            _print_progress(i + 1, n, 0.0)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    # Phase 3: assemble records.
    for j, narrative in zip(journeys, narratives):
        j.narrative = narrative
        records.append(journey_to_record(j, narrative))

    # ----- write JSONL (atomic) -----
    if eval_frac > 0:
        # Stratify by journey_family
        by_family: dict[str, list[dict]] = {}
        for r in records:
            by_family.setdefault(r["journey_family"], []).append(r)
        train: list[dict] = []
        eval_split: list[dict] = []
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
    }
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
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
