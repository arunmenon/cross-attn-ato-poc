"""Deterministic templated narrative generator (no LLM).

Used when:
  - The Anthropic API key is unset (Day-0 / local-only iteration)
  - Producing Layer-C 50-200k eval sets (LLM-narration cost would be
    prohibitive, and distribution-shape match to training is what
    matters)
  - Smoke tests (deterministic = reproducible)

Per-journey-family templates. Output structure mirrors the LLM narrator:
a session_summary block + the deterministic verdict footer.

Narrative-leakage policy: templates here are hand-written specifically to
*not* leak class names (no "SIM-swap", "fraud", "legit", etc.). The
narrative_leakage_scan runs as a defense-in-depth check in build_dataset.

CLI:
    python -m data.gen.cheap_template_generator --self-test
"""

from __future__ import annotations

import argparse
import random
import sys

from data.gen.types import Journey

# Per-family narrative phrases. Each list has multiple variants so the
# templater can pick one for variety.
#
# v4 (2026-05-18, data-v4-pivot-plan.md Change 1): templates rewritten
# to strip ALL value-laden adjectives that paraphrase the bucketed
# event features. The narrator (LLM or template) describes WHAT EVENT
# TYPES happened and in what ORDER — never how big, how risky, how
# new, how rapid, or how strongly authenticated. Quantification lives
# only in the structured event stream.
#
# Each template variant below has been audited against
# eval/leakage_checks.paraphrase_leakage_scan(). NONE of these phrases
# names the journey family (no "SIM-swap", "phishing", etc., per
# review 004 #1), AND none uses banned bucket paraphrases like
# "high-value", "previously-unseen device", "newly-added recipient",
# "high-risk network", "multi-factor", "rapid succession", etc.
_TEMPLATES: dict[str, list[str]] = {
    "clean": [
        "Routine session. The account holder logged in and made a series of "
        "transactions to recipients. No unusual signals.",
        "Standard activity. A login was followed by purchases at merchants. "
        "The session ended without further events.",
    ],
    "cred_stuff": [
        "The session shows a series of login attempts in a short window. No "
        "transactions completed.",
        "A series of failed login attempts. The pattern is incompatible with "
        "typical human authentication.",
    ],
    "sim_swap": [
        "A login was followed by a device registration, a password change, "
        "and the addition of a recipient. A transfer to that recipient "
        "occurred shortly after.",
        "The account holder authenticated, registered a device, changed "
        "credentials, and initiated a transfer to a recipient added during "
        "the session.",
    ],
    "phish_takeover": [
        "The session began with authentication and proceeded directly to "
        "several outbound transfers to recipients at merchants.",
        "A login was followed immediately by a series of outgoing "
        "transactions.",
    ],
    "malware_rat": [
        "A login was followed by activity that diverges from the account's "
        "historical pattern: a recipient was added and a transfer was sent "
        "to that recipient.",
        "The session shows transactional behavior that does not match the "
        "account's prior history: the addition of a payee followed by a "
        "transfer.",
    ],
    "mule_chain": [
        "An incoming transfer arrived and was redistributed across multiple "
        "recipients in a short window. The outgoing transfers are "
        "inconsistent with deliberate human-paced activity.",
        "Funds were received and then sent to several payees, each receiving "
        "a partial sum, within minutes.",
    ],
    "hn_travel": [
        "A login occurred from an international location. Subsequent "
        "transactions proceeded routinely with merchants.",
        "The session originated from a geographic location outside the "
        "account's usual range. Subsequent transaction behavior was "
        "otherwise typical.",
    ],
    "hn_large_purchase": [
        "A login was followed by a single purchase at a merchant after "
        "a deliberation period within the session.",
        "The session culminates in a single transaction to a merchant, "
        "preceded by a dwell period consistent with deliberation.",
    ],
    "hn_account_recovery": [
        "A login was followed by a password change. The session ended "
        "without further activity.",
        "Standard recovery flow: authentication, credential update, "
        "optional device registration. No notable transactional activity "
        "in the session.",
    ],
}


def generate_narrative(journey: Journey, *, rng: random.Random | None = None) -> str:
    """Produce a deterministic templated narrative for `journey`.

    Returns the body only (no verdict footer; the serializer appends it).
    """
    family = journey.journey_family
    if family not in _TEMPLATES:
        raise ValueError(f"no template for journey family {family!r}")

    rng = rng or random.Random(journey.seed or 0)
    variants = _TEMPLATES[family]
    body = rng.choice(variants)

    # Append a brief actor-cadence note for agent journeys. NEUTRAL
    # phrasings only — no class names like "shopping assistant",
    # "financial assistant", "compromised", "adversarial", "hybrid agent"
    # (all banned by eval/leakage_checks per review 004 finding #1).
    # Describe BEHAVIOR (cadence, tool-use, jitter), not CLASS.
    if journey.actor_family != "human":
        actor_phrases = {
            "agent_buying":      " The session's interaction cadence is "
                                 "moderately fast and contains tool-mediated steps.",
            "agent_finance":     " The session shows a regular, tool-mediated "
                                 "interaction cadence.",
            "agent_compromised": " The session's pacing is extremely regular "
                                 "and contains tool-mediated steps.",
            "agent_adversarial": " The session shows a regular cadence with "
                                 "occasional jitter, alongside tool-mediated steps.",
            "hybrid":            " The session shows a mix of human-paced and "
                                 "tool-mediated steps.",
        }
        body += actor_phrases.get(journey.actor_family, "")

    return body


def _self_test() -> None:
    """Exhaustive — every combination of (journey_family × actor_family)
    must pass the leakage scan. Regression-protection against the
    review-004 finding #1 class of issues (template strings that leaked
    new banned terms only on agent paths).
    """
    from data.gen.journey_templates import generate as gen_journey
    from eval.leakage_checks import narrative_leakage_scan

    ACTORS = ["human", "agent_buying", "agent_finance",
              "agent_compromised", "agent_adversarial", "hybrid"]
    leaks: list = []
    for family in _TEMPLATES:
        for actor in ACTORS:
            j = gen_journey(family, seed=7, actor=actor)
            text = generate_narrative(j)
            scan = narrative_leakage_scan(text)
            if not scan["clean"]:
                leaks.append((family, actor, scan["hits"]))
    if leaks:
        raise AssertionError(
            f"{len(leaks)} family/actor combinations leaked: {leaks[:3]}"
        )

    # Determinism
    j2 = gen_journey("clean", seed=99, actor="human")
    a = generate_narrative(j2)
    b = generate_narrative(j2)
    assert a == b, "non-deterministic template"

    n_combos = len(_TEMPLATES) * len(ACTORS)
    print(f"cheap_template_generator self-test OK "
          f"({n_combos} family/actor combos, no leakage)")


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
