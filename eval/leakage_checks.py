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


def narrative_leakage_scan(text: str) -> dict:
    """Detect banned phrases in narrative body.

    Returns dict with 'clean' boolean and 'hits' list of (phrase, span) tuples.

    The verdict footer is allowed to contain `label: fraud` / `label: legit` —
    those are scoring targets, not narrative leakage. So strip the verdict
    footer before scanning.
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
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path, help="dataset directory of jsonl files")
    p.add_argument("--modes", default="stripped,opaque,full", help="comma-separated eval modes to audit")
    p.add_argument("--narrative-scan", action="store_true", help="run narrative-leakage scan too")
    p.add_argument("--sample-n", type=int, default=None, help="cap examples scanned (for fast smoke tests)")
    p.add_argument("--strict", action="store_true", help="exit non-zero on any failure")
    args = p.parse_args()

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
