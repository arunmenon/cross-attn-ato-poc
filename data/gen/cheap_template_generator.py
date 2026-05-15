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
# templater can pick one for variety. NONE of these phrases name the
# journey family (no "SIM-swap", "phishing", etc.).
_TEMPLATES: dict[str, list[str]] = {
    "clean": [
        "Routine session. The account holder logged in from a familiar device "
        "and made a small number of low-value transactions to known recipients. "
        "No unusual signals.",
        "Standard activity. A single login was followed by typical, low-amount "
        "purchases to previously-seen merchants. Session length was within the "
        "user's normal range.",
    ],
    "cred_stuff": [
        "The session shows an unusually large number of login attempts in a "
        "very short window, from a high-risk network location. No transactions "
        "completed.",
        "A rapid series of failed login attempts originating from a network "
        "category associated with elevated risk. The pattern is incompatible "
        "with typical human authentication.",
    ],
    "sim_swap": [
        "A login from a previously-unseen device was followed by a new device "
        "registration, a password change, and the addition of a new payee. "
        "Shortly thereafter, an unusually large transfer occurred to that "
        "newly-registered payee.",
        "The account holder authenticated, registered a new hardware "
        "fingerprint, changed credentials, and within minutes initiated a "
        "high-value transfer to a freshly-added recipient on the account.",
    ],
    "phish_takeover": [
        "The session began with authentication from a high-risk network and "
        "an unfamiliar device. The actor proceeded directly to several "
        "high-velocity transfers to newly-added recipients at elevated-risk "
        "merchants.",
        "Login originated from a network and device combination not previously "
        "associated with this account. The session immediately progressed to "
        "outgoing transactions in quick succession.",
    ],
    "malware_rat": [
        "Login from a familiar device with strong authentication, but the "
        "subsequent activity diverges from the account's historical pattern: "
        "a freshly-added recipient receives a high-value transfer that does "
        "not match prior spending behavior.",
        "Although the device and authentication signals are clean, the "
        "session's transactional behavior is anomalous: an addition of a new "
        "payee followed by a transfer well above the account's normal range.",
    ],
    "mule_chain": [
        "An incoming transfer arrived and was rapidly redistributed across "
        "multiple newly-added recipients in a short window. The outgoing "
        "transfers show an extreme cadence inconsistent with deliberate "
        "human-paced activity.",
        "Funds were received and then immediately fanned out to several fresh "
        "payees, each receiving a partial sum, all within minutes.",
    ],
    "hn_travel": [
        "Login from an international location, but the device and "
        "authentication signals are consistent with the account holder. "
        "Subsequent transactions are routine: low to moderate amounts to "
        "known merchants.",
        "Session originated from a geographic location outside the account's "
        "usual range, but device, authentication, and transaction behavior "
        "are otherwise typical for this user.",
    ],
    "hn_large_purchase": [
        "Routine login from a familiar device with strong authentication. "
        "Followed by a single high-value purchase at a known merchant after "
        "a long deliberation period within the session.",
        "Standard authentication and device signals. The session culminates "
        "in a single large-amount transaction to a previously-seen merchant, "
        "preceded by an extended dwell consistent with deliberation.",
    ],
    "hn_account_recovery": [
        "Login from a known device with multi-factor verification was "
        "followed by a password change. The session is short and contains no "
        "high-value activity.",
        "Standard recovery flow: authentication with strong factors, "
        "credential update, optional device registration. No notable "
        "transactional activity in the session.",
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

    # Append a brief actor-cadence note for agent journeys (still no class
    # names). Helps the model use actor signature without hand-holding.
    if journey.actor_family != "human":
        actor_phrases = {
            "agent_buying": " The session's interaction cadence is consistent "
                            "with a programmatic shopping assistant.",
            "agent_finance": " The session shows a regular, tool-mediated "
                             "interaction cadence consistent with an automated "
                             "financial assistant.",
            "agent_compromised": " The session's pacing is extremely regular "
                                 "and contains tool-mediated steps.",
            "agent_adversarial": " The session shows a regular cadence with "
                                 "occasional jitter, suggestive of automation "
                                 "attempting to mimic human pacing.",
            "hybrid": " The session shows a mix of human-paced and "
                      "tool-mediated steps.",
        }
        body += actor_phrases.get(journey.actor_family, "")

    return body


def _self_test() -> None:
    from data.gen.journey_templates import generate as gen_journey
    from data.gen.agent_actor_mixer import mix
    from eval.leakage_checks import narrative_leakage_scan

    for family in _TEMPLATES:
        j = gen_journey(family, seed=7, actor="human")
        text = generate_narrative(j)
        scan = narrative_leakage_scan(text)
        assert scan["clean"], f"{family} template leaks: {scan['hits']}"

    # Agent-actor variant
    j = gen_journey("sim_swap", seed=11, actor="agent_compromised")
    j = mix(j, rng=random.Random(11))
    text = generate_narrative(j)
    assert "regular" in text or "tool" in text, "agent cadence note missing"
    scan = narrative_leakage_scan(text)
    assert scan["clean"], f"agent variant leaks: {scan['hits']}"

    # Determinism
    j2 = gen_journey("clean", seed=99, actor="human")
    a = generate_narrative(j2)
    b = generate_narrative(j2)
    assert a == b, "non-deterministic template"

    print(f"cheap_template_generator self-test OK ({len(_TEMPLATES)} families, no leakage)")


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
