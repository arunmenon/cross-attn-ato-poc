"""Leakage detection across two axes:

1. Token-leakage: confirms that `stripped` and `opaque` eval modes actually
   removed/opacified every journey/actor token. Catches synonyms, numerals,
   partial matches.

2. Narrative-leakage: the LLM narrator may write phrases like "this is a
   SIM-swap" or "fraudulent activity" — which the text-only baselines can
   trivially read as the label. Banned-phrase scan + post-generation gate.

CLI:
    python eval/leakage_checks.py --dataset DATA_DIR --modes stripped,opaque,full --narrative-scan
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Token-leakage detection
# ---------------------------------------------------------------------------

_ANY_JOURNEY = re.compile(r"<journey_[a-z_]+>|</journey_[a-z_]+>")
_ANY_ACTOR = re.compile(r"<actor_[a-z_]+>")
_OPAQUE_JOURNEY = re.compile(r"<journey_type_\d{2}>|</journey_type_\d{2}>")
_OPAQUE_ACTOR = re.compile(r"<actor_type_\d{2}>")


def verify_strip(text: str) -> dict:
    """Confirm no journey/actor tokens remain in `text`."""
    return {
        "has_journey": bool(_ANY_JOURNEY.search(text)),
        "has_actor": bool(_ANY_ACTOR.search(text)),
        "has_opaque_journey": bool(_OPAQUE_JOURNEY.search(text)),
        "has_opaque_actor": bool(_OPAQUE_ACTOR.search(text)),
    }


def verify_opaque(text: str) -> dict:
    """Confirm journey/actor tokens are opacified — no raw names remain."""
    return {
        "has_raw_journey": bool(_ANY_JOURNEY.search(text)),
        "has_raw_actor": bool(_ANY_ACTOR.search(text)),
        "has_opaque_journey": bool(_OPAQUE_JOURNEY.search(text)),
        "has_opaque_actor": bool(_OPAQUE_ACTOR.search(text)),
    }


# ---------------------------------------------------------------------------
# Narrative-leakage detection (v3)
# ---------------------------------------------------------------------------

# Banned phrases that would let a text-only baseline read the label.
# Word boundaries enforced; case-insensitive.
#
# Stems matter: "fraud" catches fraud, fraudulent, fraudster, defraud, etc.
# unless an explicit exception is added.
_BANNED_STEMS = [
    # Class-name labels (label leakage if any of these appear in narrative)
    r"\bfraud\w*",        # fraud, fraudulent, fraudster, defraud
    r"\blegit\w*",        # legit, legitimate, legitimately
    r"\bgenuine\w*",      # genuine, genuinely
    r"\baccount\s+takeover\b",
    r"\bATO\b",
    r"\bhard[\s-]+negative\w*",
    r"\bsim[\s-]+swap\w*",
    r"\bphishing\b|\bphish\b",
    r"\bmule\b|\bmule[\s-]+chain\b",
    r"\bmalware\b|\bremote\s+access\s+trojan\b|\bRAT\b",
    r"\bcredential[\s-]+stuffing\b",
    r"\btakeover\b",
    r"\blegitimate\s+travel\b",
    r"\blegitimate\s+(large\s+)?purchase\b",
    r"\blegitimate\s+(account\s+)?recovery\b",
    # Actor-class labels (review 004 #1 — these are stripped/opacified at
    # eval, so the narrative body must not name them either).
    r"\bcompromised\b",
    r"\badversarial\b",
    r"\bmalicious\s+(agent|bot|tool|assistant)\b",
    r"\b(buying|shopping)\s+(assistant|agent|bot|tool)\b",
    r"\b(finance|financial)\s+(assistant|agent|bot|tool)\b",
    r"\bhybrid\s+(actor|agent|user|session)\b",
    # Raw class-name tokens (should never appear in narrative body)
    r"\bagent_(buying|finance|compromised|adversarial)\b",
]

# Phrases that look bannable but are operational evidence (allow-listed).
# Always require the banned-stem regex to fire AND the allowlist regex to NOT match.
_ALLOW_PHRASES = [
    r"\bgenuine\s+identifier\b",       # narrator may say "genuine identifier" as a synthetic-data marker
]

_BANNED_RE = re.compile("|".join(_BANNED_STEMS), re.IGNORECASE)
_ALLOW_RE = re.compile("|".join(_ALLOW_PHRASES), re.IGNORECASE) if _ALLOW_PHRASES else None


# ---------------------------------------------------------------------------
# Bucket-paraphrase detection (v4)
# ---------------------------------------------------------------------------
# v3 catches class labels and journey-family names. v4 adds a second
# layer: catch value-laden adjectives that paraphrase the BUCKETED
# FEATURE TOKENS (amount_bucket, device_age, recipient_age, ip_risk,
# txn_velocity, auth_strength, session_dwell). These paraphrases are
# the bug v4 is fixing — they leak the structured-stream signal into
# the narrative.
#
# Patterns are mostly bigrams (`<value-adjective> <target-noun>`) to
# avoid false positives on common English ("a large number of attempts"
# doesn't match because "number" isn't an amount-bearing noun).

_BUCKET_PARAPHRASE_STEMS = [
    # Amount paraphrases — bigrams of (value-adjective, amount-bearing noun)
    r"\bhigh[-\s]value\b",
    r"\blow[-\s]value\b",
    r"\bbig[-\s]ticket\b",
    r"\b(large|small|sizable|sizeable|hefty|modest|substantial|significant|minor|tiny)\s+(transfer|transaction|payment|purchase|amount|sum|deposit|withdrawal)s?\b",
    # Device paraphrases
    r"\b(previously[-\s]unseen|unfamiliar|new|rare|known|trusted|recognised|recognized|primary|secondary)\s+(device|browser|phone|laptop|machine|hardware|computer)\b",
    # IP / network paraphrases
    r"\b(high|low|elevated)[-\s]risk\s+(network|location|ip|address|connection|origin)\b",
    r"\bsuspicious\s+(network|location|ip|address|connection|origin)\b",
    r"\bdatacenter\s+(ip|network|connection|origin)\b",
    r"\b(vpn|tor|anonymizing)\s+(network|connection|service|exit|node|relay|address|ip)\b",
    # Recipient paraphrases
    r"\b(freshly|newly|just)[-\s]added\s+(recipient|payee|contact|beneficiary)\b",
    r"\bnew\s+(recipient|payee|contact|beneficiary)\b",
    r"\brecent\s+(recipient|payee|contact|beneficiary)\b",
    r"\bunfamiliar\s+(recipient|payee|contact|beneficiary)\b",
    r"\bunknown\s+(recipient|payee|contact|beneficiary)\b",
    # Velocity paraphrases
    r"\b(bursty|rapid|extreme|very[-\s]fast|fast[-\s]paced|compressed|quick)\s+(cadence|sequence|succession|frequency|pace|velocity)\b",
    r"\b(rapid|extreme|fast|quick)\s+(succession|sequence)\s+of\b",
    # Auth paraphrases — partially standalone since auth-tokens are stronger signals
    r"\b(multi[-\s]factor|MFA)\b",
    r"\b(strong|weak)\s+(authentication|auth)\b",
    r"\b(no|without)\s+(multi[-\s]factor|MFA)\b",
    r"\bpassword[-\s]only\b",
    r"\bcookie[-\s]only\b",
    # Session-dwell paraphrases
    r"\b(short|brief|extended|long|prolonged)\s+session\b",
    r"\b(brief|extended)\s+dwell\b",
]

_BUCKET_PARAPHRASE_RE = re.compile("|".join(_BUCKET_PARAPHRASE_STEMS), re.IGNORECASE)


def paraphrase_leakage_scan(text: str) -> dict:
    """Detect bucket-paraphrase phrases in narrative body (v4).

    Returns dict with 'clean' boolean and 'hits' list of (phrase, span)
    tuples — same shape as `narrative_leakage_scan` so the build_dataset
    retry loop can treat both scanners uniformly.

    These paraphrases are banned in v4 because they leak the
    structured-stream signal into the narrative — the very bug v4 is
    designed to fix. If the LLM narrator emits "large transfer" or
    "newly-added recipient" or "MFA" in the narrative body, the
    text-only baseline can read fraud signal that should live only in
    the side stream.

    See data/gen/narrative_generator.py SYSTEM_PROMPT rule 5 for the
    full list of banned adjective families and target nouns. This
    scanner is the enforcement layer for that rule.
    """
    body = re.sub(r"<risk_verdict>.*?</risk_verdict>", "", text, flags=re.DOTALL)

    hits: list[tuple[str, tuple[int, int]]] = []
    for m in _BUCKET_PARAPHRASE_RE.finditer(body):
        phrase = m.group(0)
        hits.append((phrase, (m.start(), m.end())))

    return {
        "clean": len(hits) == 0,
        "hits": hits,
        "n_hits": len(hits),
    }


def narrative_leakage_scan(text: str, include_paraphrase: bool = True) -> dict:
    """Detect banned phrases in narrative body.

    Returns dict with 'clean' boolean and 'hits' list of (phrase, span) tuples.

    The verdict footer is allowed to contain `label: fraud` / `label: legit` —
    those are scoring targets, not narrative leakage. So strip the verdict
    footer before scanning.

    v4 (2026-05-18): by default this scanner ALSO checks for
    bucket-paraphrase leakage (see `paraphrase_leakage_scan`). Callers
    that want only the v3 class/actor scan (e.g., for analyzing legacy
    v3 datasets) can pass `include_paraphrase=False`. Hits from both
    scanners are merged into the returned `hits` list.
    """
    # Strip the verdict footer (between <risk_verdict> and </risk_verdict>).
    body = re.sub(r"<risk_verdict>.*?</risk_verdict>", "", text, flags=re.DOTALL)

    hits: list[tuple[str, tuple[int, int]]] = []
    for m in _BANNED_RE.finditer(body):
        phrase = m.group(0)
        # Check allowlist
        if _ALLOW_RE is not None:
            ctx_start = max(0, m.start() - 30)
            ctx_end = min(len(body), m.end() + 30)
            ctx = body[ctx_start:ctx_end]
            if _ALLOW_RE.search(ctx):
                continue
        hits.append((phrase, (m.start(), m.end())))

    if include_paraphrase:
        para = paraphrase_leakage_scan(text)
        hits.extend(para["hits"])

    return {
        "clean": len(hits) == 0,
        "hits": hits,
        "n_hits": len(hits),
    }


# ---------------------------------------------------------------------------
# Dataset-level audit
# ---------------------------------------------------------------------------

def audit_dataset(
    dataset_path: Path,
    modes: Iterable[str],
    narrative_scan: bool = True,
    sample_n: int | None = None,
) -> dict:
    """Walk a dataset directory of jsonl files. Apply leakage checks to each example.

    Returns a summary dict (counts of failures, per-mode and per-narrative).
    """
    from . import eval_modes  # local import to avoid eager import on CLI tools

    files = sorted(dataset_path.glob("**/*.jsonl"))
    if not files:
        return {"error": f"no jsonl files in {dataset_path}"}

    summary = {
        "total_examples": 0,
        "narrative_leakage_failures": 0,
        "narrative_leakage_sample_hits": [],
        "per_mode": {m: {"strip_failures": 0, "opaque_failures": 0} for m in modes},
    }

    for path in files:
        with path.open() as f:
            for line in f:
                ex = json.loads(line)
                text = ex.get("text") or ex.get("input") or ""
                summary["total_examples"] += 1

                if sample_n is not None and summary["total_examples"] > sample_n:
                    break

                if narrative_scan:
                    scan = narrative_leakage_scan(text)
                    if not scan["clean"]:
                        summary["narrative_leakage_failures"] += 1
                        if len(summary["narrative_leakage_sample_hits"]) < 5:
                            summary["narrative_leakage_sample_hits"].append({
                                "file": str(path),
                                "hits": scan["hits"],
                            })

                for mode in modes:
                    transformed, _ = eval_modes.apply(text, mode)  # type: ignore[arg-type]
                    if mode == "stripped":
                        v = verify_strip(transformed)
                        if v["has_journey"] or v["has_actor"]:
                            summary["per_mode"][mode]["strip_failures"] += 1
                    elif mode == "opaque":
                        v = verify_opaque(transformed)
                        if v["has_raw_journey"] or v["has_raw_actor"]:
                            summary["per_mode"][mode]["opaque_failures"] += 1
                    # mode == "full": no-op
    return summary


# ---------------------------------------------------------------------------
# Train/eval text-hash + structured-events-hash leakage (review 018/019)
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _events_hash(structured_events) -> str:
    return hashlib.sha256(
        json.dumps(structured_events, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compute_clean_eval_mask(
    train_path: Path,
    eval_path: Path,
) -> tuple[list[bool], dict]:
    """Return (mask, stats) for cleaning the eval set against train leakage.

    Each eval row passes iff BOTH:
      - SHA-256(text) NOT IN train text hashes
      - SHA-256(json.dumps(structured_events, sort_keys=True)) NOT IN train events hashes

    Both `text` and `structured_events` are required on every row.

    Returns:
      mask: list[bool] of length n_eval; mask[i] == True means the i-th eval
            row survives (keep it).
      stats: {
        "n_train": int,
        "n_eval": int,
        "n_kept": int,
        "n_dropped": int,
        "n_text_overlap": int,           # text-hash hits (incl. ones also in events)
        "n_events_overlap": int,         # events-hash hits (incl. ones also in text)
        "n_text_only": int,              # text-hash hit but NOT events-hash hit
        "n_events_only": int,            # events-hash hit but NOT text-hash hit
        "n_both": int,                   # text-hash AND events-hash hit
        "n_unique_train_text_hashes": int,
        "n_unique_train_events_hashes": int,
        "per_family": {
          "<journey_family>": {
            "n_eval": int, "n_dropped": int, "n_kept": int,
            "n_text_only": int, "n_events_only": int, "n_both": int,
          }, ...
        }
      }

    Hard contract (review 018, current dataset): n_dropped == 534
    (533 both, 1 events-only). The CLI mode prints the per-family table.
    """
    train_text = set()
    train_events = set()
    n_train = 0
    for r in _iter_jsonl(train_path):
        train_text.add(_text_hash(r["text"]))
        train_events.add(_events_hash(r["structured_events"]))
        n_train += 1

    mask: list[bool] = []
    per_family: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"n_eval": 0, "n_dropped": 0, "n_kept": 0,
                 "n_text_only": 0, "n_events_only": 0, "n_both": 0}
    )
    n_eval = 0
    n_text_overlap = 0
    n_events_overlap = 0
    n_text_only = 0
    n_events_only = 0
    n_both = 0

    for r in _iter_jsonl(eval_path):
        n_eval += 1
        fam = r.get("journey_family", "<unknown>")
        per_family[fam]["n_eval"] += 1

        text_hit = _text_hash(r["text"]) in train_text
        ev_hit = _events_hash(r["structured_events"]) in train_events

        if text_hit and ev_hit:
            n_both += 1
            n_text_overlap += 1
            n_events_overlap += 1
            per_family[fam]["n_both"] += 1
        elif text_hit:
            n_text_only += 1
            n_text_overlap += 1
            per_family[fam]["n_text_only"] += 1
        elif ev_hit:
            n_events_only += 1
            n_events_overlap += 1
            per_family[fam]["n_events_only"] += 1

        passes = not (text_hit or ev_hit)
        mask.append(passes)
        if passes:
            per_family[fam]["n_kept"] += 1
        else:
            per_family[fam]["n_dropped"] += 1

    n_dropped = sum(1 for m in mask if not m)
    n_kept = n_eval - n_dropped

    stats = {
        "n_train": n_train,
        "n_eval": n_eval,
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "n_text_overlap": n_text_overlap,
        "n_events_overlap": n_events_overlap,
        "n_text_only": n_text_only,
        "n_events_only": n_events_only,
        "n_both": n_both,
        "n_unique_train_text_hashes": len(train_text),
        "n_unique_train_events_hashes": len(train_events),
        "per_family": dict(per_family),
    }
    return mask, stats


def _print_overlap_diagnostic(train_path: Path, eval_path: Path) -> None:
    mask, stats = compute_clean_eval_mask(train_path, eval_path)
    print(f"# train/eval overlap diagnostic")
    print(f"# train: {train_path}")
    print(f"# eval:  {eval_path}")
    print(f"n_train={stats['n_train']}  n_eval={stats['n_eval']}")
    print(f"n_kept={stats['n_kept']}  n_dropped={stats['n_dropped']}")
    print(
        f"n_text_overlap={stats['n_text_overlap']}  "
        f"n_events_overlap={stats['n_events_overlap']}  "
        f"n_both={stats['n_both']}  "
        f"n_text_only={stats['n_text_only']}  "
        f"n_events_only={stats['n_events_only']}"
    )
    print(f"n_unique_train_text_hashes={stats['n_unique_train_text_hashes']}")
    print(f"n_unique_train_events_hashes={stats['n_unique_train_events_hashes']}")
    print()
    print(f"{'family':<28} {'n_eval':>7} {'drop':>6} {'keep':>6} {'both':>6} {'txt_o':>6} {'ev_o':>6}")
    for fam in sorted(stats["per_family"]):
        f = stats["per_family"][fam]
        print(
            f"{fam:<28} {f['n_eval']:>7} {f['n_dropped']:>6} {f['n_kept']:>6} "
            f"{f['n_both']:>6} {f['n_text_only']:>6} {f['n_events_only']:>6}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, help="dataset directory of jsonl files (for --narrative-scan audit)")
    p.add_argument("--modes", default="stripped,opaque,full", help="comma-separated eval modes to audit")
    p.add_argument("--narrative-scan", action="store_true", help="run narrative-leakage scan too")
    p.add_argument("--sample-n", type=int, default=None, help="cap examples scanned (for fast smoke tests)")
    p.add_argument("--strict", action="store_true", help="exit non-zero on any failure")
    p.add_argument(
        "--train-eval-overlap",
        type=Path,
        help="path to dataset dir with train.jsonl + eval.jsonl; "
             "prints text-hash + structured_events-hash overlap diagnostic",
    )
    args = p.parse_args()

    if args.train_eval_overlap is not None:
        d = args.train_eval_overlap
        _print_overlap_diagnostic(d / "train.jsonl", d / "eval.jsonl")
        return 0

    if args.dataset is None:
        p.error("--dataset is required (or pass --train-eval-overlap DIR)")

    modes = [m.strip() for m in args.modes.split(",")]
    summary = audit_dataset(args.dataset, modes, narrative_scan=args.narrative_scan, sample_n=args.sample_n)

    print(json.dumps(summary, indent=2))

    if args.strict:
        any_fail = summary.get("narrative_leakage_failures", 0) > 0 or any(
            v["strip_failures"] > 0 or v["opaque_failures"] > 0
            for v in summary.get("per_mode", {}).values()
        )
        if any_fail:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
