"""LLM-narrated paired-text generator.

Calls Anthropic's API (Claude) to produce an analyst-style narrative for
each Journey. Defense-in-depth against narrative-leakage:

  1. **Prompt-side ban**: the system prompt explicitly forbids the
     narrator from using class names (fraud, legit, ATO, SIM-swap,
     phishing, mule, malware, takeover, hard negative, etc.).
  2. **Post-gen scrub**: every narrative passes through
     `src.tokenizer.fencer.fence()` to fence any residual literal PII the
     narrator inadvertently produced.
  3. **Post-gen scan**: every narrative passes through
     `eval.leakage_checks.narrative_leakage_scan()`. Failed narratives
     are regenerated up to `--max-retries` times, then either dropped
     or hard-failed depending on policy.
  4. **Disk cache**: narratives keyed by SHA-256 of the structured
     stream are cached on disk, so re-runs don't re-bill.

The verdict footer is NOT LLM-generated. It is deterministically
appended by the pipeline based on `journey.label` and `journey.journey_family`.

If `ANTHROPIC_API_KEY` is not set, this module raises with a clear
message pointing the user to `cheap_template_generator` for the offline
path.

CLI:
    python -m data.gen.narrative_generator --self-test
        (uses a stub when no API key is set; passes without LLM call)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from data.gen.types import Journey
from eval.leakage_checks import narrative_leakage_scan
from src.tokenizer.fencer import fence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path("data/cache/narratives")
DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # cheap; ~$0.005/narrative target
DEFAULT_MAX_TOKENS = 300
DEFAULT_TIMEOUT_SEC = 30
DEFAULT_USD_BUDGET = 200.0
DEFAULT_MAX_RETRIES = 2

# Rough cost estimates (USD per call, Haiku 4.5):
# - Input ~400 tokens × $0.80/M = $0.00032
# - Output ~250 tokens × $4.00/M = $0.00100
# Total: ~$0.0013/call (well under $0.005 target).
_HAIKU_INPUT_USD_PER_TOKEN = 0.80 / 1_000_000
_HAIKU_OUTPUT_USD_PER_TOKEN = 4.00 / 1_000_000


SYSTEM_PROMPT = """You are an internal risk-analyst describing a payments session in plain English.
You will receive a structured event sequence and must produce a 2-4 sentence narrative
suitable for an analyst's case log.

HARD RULES (any violation is a critical failure):
1. Do NOT use any of these words or their stems, in any form (singular/plural/verb/adjective):
   - fraud, fraudulent, fraudster, defraud
   - legit, legitimate, legitimately, genuine (except "genuine identifier")
   - account takeover, ATO, takeover
   - hard negative, hard-negative
   - SIM swap, SIM-swap
   - phishing, phish
   - mule, mule chain
   - malware, remote access trojan, RAT
   - credential stuffing
   - compromised, adversarial, malicious (when describing the actor)
   - buying assistant, shopping assistant, finance assistant, financial assistant
   - hybrid actor, hybrid agent, hybrid user, hybrid session
2. Describe what HAPPENED operationally — events, sequences, timing — without LABELING the
   session or the actor's intent.
3. Refer to the actor as one of: "the account holder", "the actor", "the session". For
   agent-mediated sessions you may use "the agent" or "the tool-mediated agent". You must
   NEVER use any class label (compromised, adversarial, buying assistant, finance assistant,
   hybrid agent, etc.). Describe BEHAVIOR (cadence, tool-use trace), not CLASS.
4. Do not invent any PII (no email addresses, IPs, phone numbers, account IDs, full names). If
   you need to refer to a recipient or device, use phrases like "a newly-added recipient" or
   "a previously-unseen device".
5. Use 2-4 sentences. No bullet points, no headers, no markdown.

Examples of compliant phrasing:
- "Login from a previously-unseen device was followed by a credential change and a
   high-value transfer to a freshly-added recipient."
- "A large number of authentication attempts in a very short window, from a high-risk
   network location. No outgoing transactions completed."
- "Routine session. A single login was followed by typical low-amount purchases to
   known merchants."
- "The session showed a highly regular cadence with tool-mediated steps."   (agent case;
   note: no class label used)
"""


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------

@dataclass
class CostTracker:
    """Running USD spend on narrator calls. build_dataset polls this to
    abort if the budget cap is exceeded mid-batch.
    """
    spent_usd: float = 0.0
    n_calls: int = 0
    n_cache_hits: int = 0
    budget_usd: float = DEFAULT_USD_BUDGET
    breakdown: dict[str, float] = field(default_factory=dict)

    def charge(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if model.startswith("claude-haiku"):
            cost = (input_tokens * _HAIKU_INPUT_USD_PER_TOKEN
                    + output_tokens * _HAIKU_OUTPUT_USD_PER_TOKEN)
        else:
            # Conservative default for other models
            cost = (input_tokens * 3.0 / 1_000_000
                    + output_tokens * 15.0 / 1_000_000)
        self.spent_usd += cost
        self.n_calls += 1
        self.breakdown[model] = self.breakdown.get(model, 0.0) + cost
        return cost

    def cache_hit(self) -> None:
        self.n_cache_hits += 1

    def over_budget(self) -> bool:
        return self.spent_usd >= self.budget_usd

    def summary(self) -> dict:
        return {
            "spent_usd": round(self.spent_usd, 4),
            "budget_usd": self.budget_usd,
            "n_calls": self.n_calls,
            "n_cache_hits": self.n_cache_hits,
            "by_model": {k: round(v, 4) for k, v in self.breakdown.items()},
        }


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _journey_cache_key(journey: Journey, model: str) -> str:
    """SHA-256 over the structured stream + model name. Stable across
    runs; cache is safe to share between regenerations of the same data.
    """
    canonical = {
        "model": model,
        "journey_family": journey.journey_family,
        "actor_family": journey.actor_family,
        "events": journey.events,
    }
    blob = json.dumps(canonical, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


def _cache_path(cache_dir: Path, key: str) -> Path:
    # Two-level prefix to keep directory listings manageable at 200k+ entries.
    return cache_dir / key[:2] / key[2:4] / f"{key}.txt"


def _cache_get(cache_dir: Path, key: str) -> str | None:
    p = _cache_path(cache_dir, key)
    if p.exists():
        return p.read_text()
    return None


def _cache_put(cache_dir: Path, key: str, body: str) -> None:
    p = _cache_path(cache_dir, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(body)
    tmp.rename(p)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Per-actor neutral cadence descriptors. The LLM sees BEHAVIOR (cadence,
# tool-use, jitter), never the class name. Review 004 finding #1.
_ACTOR_CADENCE_DESCRIPTORS: dict[str, str] = {
    "human":             "human-paced; irregular inter-event intervals",
    "agent_buying":      "programmatic; tool-mediated steps; moderately fast cadence",
    "agent_finance":     "programmatic; tool-mediated steps; regular cadence",
    "agent_compromised": "very fast; highly regular cadence; tool-mediated steps",
    "agent_adversarial": "fast; regular cadence with mimicry jitter; tool-mediated steps",
    "hybrid":            "mixed human-paced and tool-mediated steps",
}


def _serialize_events_for_prompt(journey: Journey) -> str:
    """Compact textual rendering of the structured stream, fed into the
    narrator's user message. Keep this stable — cache keys depend on the
    event content.

    Actor descriptor is NEUTRAL (behavioral, not class-named) per review
    004 finding #1.
    """
    lines = []
    actor = journey.actor_family
    cadence = _ACTOR_CADENCE_DESCRIPTORS.get(actor, "unknown cadence")
    lines.append(f"Interaction cadence: {cadence}.")
    lines.append(f"Number of events: {len(journey.events)}.")
    lines.append("Event sequence:")
    for ev in journey.events:
        bits = [f"t={ev['t']}s", f"event={ev['event']}"]
        for key in ("amount_bucket", "geo_distance", "ip_risk", "device_age",
                    "merchant_risk", "txn_velocity", "recipient_age",
                    "session_dwell", "auth_strength", "direction",
                    "prepares_event"):
            if key in ev:
                bits.append(f"{key}={ev[key]}")
        lines.append("  " + " ".join(bits))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The LLM call
# ---------------------------------------------------------------------------

# Type alias for callable injection (used in tests / cheap-template fallback).
# Review 004 finding #4: retries now pass `attempt` index and `prev_hits` so
# the narrator can vary its prompt + temperature to escape leaky-output
# fixed points.
NarratorFn = Callable[[Journey, int, list], tuple[str, int, int]]
"""Function: (journey, attempt_num, prev_hits) -> (narrative_body, input_tokens, output_tokens).

attempt_num is 0 for the initial call, 1 for first retry, etc.
prev_hits is the list of (phrase, span) tuples from the previous attempt's
leakage scan; empty list on the initial call.
"""


def _real_anthropic_call(model: str, max_tokens: int, timeout: int) -> NarratorFn:
    """Build a NarratorFn that calls the Anthropic API. Raises if the
    SDK / API key is unavailable.
    """
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic package not installed. Run `pip install -r requirements.txt` "
            "or use cheap_template_generator for offline narration."
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it, or use "
            "cheap_template_generator for offline narration."
        )

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def call(journey: Journey, attempt: int, prev_hits: list) -> tuple[str, int, int]:
        user_msg = _serialize_events_for_prompt(journey)

        # On retry: append targeted feedback + raise temperature for variation.
        # Review 004 finding #4: without this the retry will likely
        # reproduce the same leaky output.
        system = SYSTEM_PROMPT
        if attempt > 0 and prev_hits:
            banned_phrases = sorted({h[0] for h in prev_hits})
            system = (
                f"{SYSTEM_PROMPT}\n\n"
                f"RETRY FEEDBACK (attempt {attempt}): a previous attempt at "
                f"this narrative was REJECTED because it used these banned "
                f"phrases or stems: {banned_phrases}. Rewrite the narrative "
                f"without using any of them or their variants. Stay strictly "
                f"within the HARD RULES above."
            )
        # Initial attempt uses low temperature for stable formatting; retries
        # use higher temperature to escape stuck outputs.
        temperature = 0.3 if attempt == 0 else 0.7

        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        body = resp.content[0].text if resp.content else ""
        in_toks = resp.usage.input_tokens if resp.usage else 0
        out_toks = resp.usage.output_tokens if resp.usage else 0
        return body.strip(), in_toks, out_toks

    return call


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_narrative(
    journey: Journey,
    *,
    tracker: CostTracker,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    narrator_fn: NarratorFn | None = None,
) -> str:
    """Produce one LLM narrative for `journey`. Returns the body string
    (no verdict footer; serializer appends that).

    Args:
        journey:        the Journey to narrate.
        tracker:        CostTracker that records spend; consult
                        `tracker.over_budget()` BEFORE calling to avoid
                        blowing the cap.
        narrator_fn:    optional injected narrator (for testing); when
                        None, the real Anthropic client is used.

    Raises:
        RuntimeError if budget is exceeded or narrator produces leaky
        text that cannot be regenerated within max_retries.
    """
    if tracker.over_budget():
        raise RuntimeError(
            f"narrator budget exceeded: ${tracker.spent_usd:.2f} >= "
            f"${tracker.budget_usd:.2f}"
        )

    key = _journey_cache_key(journey, model)
    cached = _cache_get(cache_dir, key)
    if cached is not None:
        tracker.cache_hit()
        return cached

    call = narrator_fn or _real_anthropic_call(model, max_tokens, timeout)

    last_body = ""
    prev_hits: list = []
    for attempt in range(max_retries + 1):
        body, in_toks, out_toks = call(journey, attempt, prev_hits)
        tracker.charge(model, in_toks, out_toks)

        # Hard budget cap, post-charge (review 004 finding #3).
        # The current call's cost is sunk, but we reject the narrative
        # so it cannot be cached or consumed by the build.
        if tracker.spent_usd > tracker.budget_usd:
            raise RuntimeError(
                f"narrator budget exceeded by this call: "
                f"${tracker.spent_usd:.4f} > ${tracker.budget_usd:.2f}. "
                f"Aborting before caching the narrative. Sunk cost from "
                f"this overshoot is the in-flight call only."
            )

        # Post-gen scrub: fence any literal-looking PII the narrator produced.
        body, _hits = fence(body)
        # Post-gen scan: did the narrator leak class names?
        scan = narrative_leakage_scan(body)
        if scan["clean"]:
            _cache_put(cache_dir, key, body)
            return body
        last_body = body
        prev_hits = scan["hits"]
        # Loop continues; next call will regenerate with feedback (real
        # narrator) or stable output (stub narrator).

    raise RuntimeError(
        f"narrative-leakage scan failed after {max_retries + 1} attempts "
        f"for journey seed={journey.seed} family={journey.journey_family}. "
        f"Hits: {prev_hits}. Last body: {last_body!r}"
    )


# ---------------------------------------------------------------------------
# Stub narrator for tests
# ---------------------------------------------------------------------------

def _stub_narrator() -> NarratorFn:
    """A narrator that returns deterministic compliant text without calling
    the API. Useful for self-test and offline CI. Accepts (and ignores)
    the retry context — review 004 finding #4 signature change.
    """
    from data.gen.cheap_template_generator import generate_narrative as cheap

    def call(journey: Journey, attempt: int, prev_hits: list) -> tuple[str, int, int]:
        body = cheap(journey)
        # Pretend it cost 300 input / 250 output tokens.
        return body, 300, 250

    return call


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    from data.gen.journey_templates import generate as gen_journey
    import tempfile

    with tempfile.TemporaryDirectory() as tdir:
        cache_dir = Path(tdir) / "narr"
        tracker = CostTracker(budget_usd=10.0)

        j = gen_journey("sim_swap", seed=42, actor="human")
        text = generate_narrative(
            j, tracker=tracker, cache_dir=cache_dir,
            narrator_fn=_stub_narrator(),
        )
        assert "SIM" not in text and "fraud" not in text.lower(), \
            f"stub narrator leaked: {text!r}"

        # Cache hit
        text2 = generate_narrative(
            j, tracker=tracker, cache_dir=cache_dir,
            narrator_fn=_stub_narrator(),
        )
        assert text2 == text, "cache should return identical body"
        assert tracker.n_cache_hits == 1, f"expected 1 cache hit, got {tracker.n_cache_hits}"
        assert tracker.n_calls == 1, f"expected 1 API call, got {tracker.n_calls}"

        # Budget cap
        cap = CostTracker(budget_usd=0.0)
        try:
            generate_narrative(j, tracker=cap, cache_dir=cache_dir,
                               narrator_fn=_stub_narrator())
            raise AssertionError("expected budget overage")
        except RuntimeError as e:
            assert "budget" in str(e).lower()

    print("narrative_generator self-test OK (using stub narrator)")
    print(f"  tracker summary: {tracker.summary()}")


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
