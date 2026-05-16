"""LLM-narrated paired-text generator.

Calls OpenAI's API by default (gpt-5.4-nano — cheapest in the GPT-5.4
family at ~$0.0004/call on our token shape, ~$10 for the full 25k
training set). Anthropic remains supported as an alternative provider
via the `LLM_PROVIDER` env var or by passing a `claude-*` model name.

Defense-in-depth against narrative-leakage:

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

Provider selection:
  - LLM_PROVIDER=openai (default) + OPENAI_API_KEY → gpt-5.4-nano
  - LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY → claude-haiku-4-5
  - Provider can also be inferred from the model name's prefix
    (gpt-* → openai, claude-* → anthropic).
  - If neither key is set, this module raises with a clear pointer
    to cheap_template_generator for the offline path.

CLI:
    python -m data.gen.narrative_generator --self-test
        (uses a stub when no API key is set; passes without LLM call)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
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
DEFAULT_MAX_TOKENS = 300
DEFAULT_TIMEOUT_SEC = 30
DEFAULT_USD_BUDGET = 200.0
DEFAULT_MAX_RETRIES = 2

# Provider switch — set LLM_PROVIDER env var to "openai" (default) or "anthropic".
# Defaults are chosen for cost: gpt-5.4-nano is ~10x cheaper than gpt-5.4-mini
# and ~3.4x cheaper than claude-haiku-4-5 at the 25k-narrative scale (see
# docs/batch-2-data-generators.md cost section).
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# Pricing table, USD per token. Last verified 2026-05-16 from
# developers.openai.com/api/docs/models/* (gpt-5.4 family). Anthropic
# pricing carried over from the prior default.
#
# Keys must match the `model` string the trainer passes to CostTracker.charge().
# "cached_in" is the price for tokens served from OpenAI's prompt-cache (90%
# off standard input). The tracker treats cache hits as `cached_in` priced.
_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4-nano":              {"in": 0.20  / 1e6, "out": 1.25 / 1e6, "cached_in": 0.02  / 1e6},
    "gpt-5.4-mini":              {"in": 0.75  / 1e6, "out": 4.50 / 1e6, "cached_in": 0.075 / 1e6},
    "gpt-5.4":                   {"in": 2.50  / 1e6, "out": 15.0 / 1e6, "cached_in": 0.25  / 1e6},
    "claude-haiku-4-5-20251001": {"in": 0.80  / 1e6, "out": 4.00 / 1e6, "cached_in": 0.08  / 1e6},
}

# Back-compat alias. New call sites should pass model=None and let
# _resolve_provider_and_model() pick the provider's default.
DEFAULT_MODEL = DEFAULT_OPENAI_MODEL


def _provider_for_model(model: str) -> str:
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    raise ValueError(f"cannot infer provider from model name {model!r}")


def _default_model_for_provider(provider: str) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    if provider == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL
    raise ValueError(
        f"unknown LLM_PROVIDER={provider!r}; expected 'openai' or 'anthropic'"
    )


def _resolve_provider_and_model(
    model: str | None,
    env_override: str | None,
) -> tuple[str, str]:
    """Pick (provider, model) given an optional caller-supplied model and
    an optional LLM_PROVIDER env override. Review 009 finding #1.

    Resolution rules:
      - No model + no env  → DEFAULT_PROVIDER + that provider's default model.
      - Env only           → env's provider + that provider's default model.
      - Model only         → infer provider from the model-name prefix.
      - Both               → infer provider from the model; if that disagrees
                             with the env override, raise. This catches the
                             "LLM_PROVIDER=anthropic + model=gpt-..." footgun
                             that review 009 caught (Anthropic was being
                             asked to serve an OpenAI model name).
    """
    env_override = env_override.lower() if env_override else None

    if model is None and env_override is None:
        provider = DEFAULT_PROVIDER
        return provider, _default_model_for_provider(provider)

    if model is None:
        # env_override is set, by elimination
        return env_override, _default_model_for_provider(env_override)

    inferred = _provider_for_model(model)
    if env_override is not None and env_override != inferred:
        raise ValueError(
            f"LLM_PROVIDER={env_override!r} disagrees with model={model!r} "
            f"(which implies provider={inferred!r}). Pass a model name that "
            f"matches LLM_PROVIDER, or unset LLM_PROVIDER and rely on "
            f"model-prefix inference."
        )
    return inferred, model


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

    def charge(self, model: str, input_tokens: int, output_tokens: int,
               cached_input_tokens: int = 0) -> float:
        """Charge for one API call. `cached_input_tokens` is the subset of
        `input_tokens` that was served from the provider's prompt cache
        (OpenAI's prompt caching reports this in `usage.prompt_tokens_details.cached_tokens`;
        Anthropic uses `cache_read_input_tokens`).

        Cost = (input - cached) × in_price + cached × cached_in_price + output × out_price.
        """
        prices = _PRICING.get(model)
        if prices is None:
            # Conservative default for unknown models — assume frontier pricing.
            cost = (input_tokens * 3.0 / 1e6 + output_tokens * 15.0 / 1e6)
        else:
            uncached_in = max(0, input_tokens - cached_input_tokens)
            cost = (uncached_in * prices["in"]
                    + cached_input_tokens * prices["cached_in"]
                    + output_tokens * prices["out"])
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


def _real_openai_call(
    model: str, max_tokens: int, timeout: int,
    base_temperature: float = 0.3,
) -> NarratorFn:
    """Build a NarratorFn that calls the OpenAI API. Default provider.

    The OpenAI Chat Completions API uses a different message shape than
    Anthropic — system + user roles inside `messages`, not a separate
    `system` arg. Prompt caching is automatic on supported models when
    the SAME prefix is sent for >1024 tokens (our system prompt is
    ~350 tokens, below threshold; caching may not fire — see
    docs/batch-2-data-generators.md cost discussion).

    `base_temperature` controls the initial-attempt temperature (default
    0.3 = stable formatting, matches the original 25k train run). Bump
    to 0.5 for eval generation to reduce narrator-style correlation
    with the training distribution. Retries always escalate to
    max(0.7, base + 0.4) so leaky outputs get escape velocity.
    """
    try:
        import openai
    except ImportError as e:
        raise RuntimeError(
            "openai package not installed. Run `pip install openai>=1.0` "
            "or use cheap_template_generator for offline narration."
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it, or use "
            "cheap_template_generator for offline narration."
        )

    client = openai.OpenAI(api_key=api_key, timeout=timeout)

    def call(journey: Journey, attempt: int, prev_hits: list) -> tuple[str, int, int]:
        user_msg = _serialize_events_for_prompt(journey)

        # Retry-time feedback — same shape as Anthropic path.
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
        temperature = base_temperature if attempt == 0 else max(0.7, base_temperature + 0.4)

        # gpt-5.4 family uses `max_completion_tokens`, not the legacy
        # `max_tokens` param (same naming shift OpenAI made for o-series
        # reasoning models — the older keyword returns HTTP 400 with
        # 'Unsupported parameter'). Caught by the first live API gate
        # after the OpenAI switch.
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )
        body = resp.choices[0].message.content if resp.choices else ""
        # Token counts — OpenAI reports prompt_tokens / completion_tokens.
        # Cached-token sub-count (when present) is consumed by
        # CostTracker.charge via the `cached_input_tokens` kwarg, but the
        # narrator interface stays single-int. We expose total input here
        # and rely on the SDK's standard `prompt_tokens` for the cache-naive
        # cost approximation. (Tighter accounting would refactor the
        # NarratorFn signature to return cached_in too — deferred.)
        usage = resp.usage
        in_toks = usage.prompt_tokens if usage else 0
        out_toks = usage.completion_tokens if usage else 0
        return (body or "").strip(), in_toks, out_toks

    return call


def _real_anthropic_call(
    model: str, max_tokens: int, timeout: int,
    base_temperature: float = 0.3,
) -> NarratorFn:
    """Build a NarratorFn that calls the Anthropic API. Raises if the
    SDK / API key is unavailable.

    `base_temperature` matches the OpenAI factory — see _real_openai_call.
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
        # Initial attempt uses base_temperature for stable formatting;
        # retries escalate to escape stuck outputs.
        temperature = base_temperature if attempt == 0 else max(0.7, base_temperature + 0.4)

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

def generate_narratives_concurrent(
    journeys: list[Journey],
    *,
    tracker: CostTracker,
    model: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    narrator_fn: NarratorFn | None = None,
    max_workers: int = 8,
    progress_callback: Callable[[int, int, float], None] | None = None,
    narrator_temp: float = 0.3,
) -> list[str]:
    """Concurrent batch wrapper around generate_narrative.

    `result[i]` corresponds to `journeys[i]` (ordering preserved).

    Why this exists: the sequential per-record path runs at ~30/min on
    gpt-5.4-nano (network-bound), so 25k narratives would take ~13 hr.
    With max_workers=8 we get ~3-4x speedup (the OpenAI SDK serializes
    the response-parsing step under the GIL, so super-linear scaling
    stops past ~8 workers in our measurements).

    Thread-safety:
      - The HTTP client is built ONCE up front and shared (openai.OpenAI
        is documented thread-safe).
      - CostTracker mutations are serialized behind a single lock.
      - The on-disk cache uses atomic temp-rename writes (`_cache_put`),
        so concurrent puts to the same key are race-free at the
        filesystem level.
      - (provider, model) is resolved once before any thread starts, so
        the LLM_PROVIDER/model agreement check (review 009 finding #1)
        still fires before any API spend.

    Budget cap behavior under concurrency:
      The cap is enforced post-charge inside each worker. Up to
      max_workers in-flight calls may complete between the cap-trip and
      the executor halting other workers, so the worst-case overshoot
      is `max_workers` calls past the cap, not one. Document this in
      RUNBOOK §9 if max_workers > 1.

    progress_callback: optional `(n_done, n_total, spent_usd)` hook
    called from inside the as_completed loop. Build_dataset wires this
    up to print the same 5%-stride progress lines the sequential path
    emits.
    """
    if not journeys:
        return []

    # Resolve once — single agreement check, single client build.
    provider, resolved_model = _resolve_provider_and_model(
        model, os.environ.get("LLM_PROVIDER"),
    )

    if narrator_fn is not None:
        call = narrator_fn
    elif provider == "openai":
        call = _real_openai_call(resolved_model, max_tokens, timeout,
                                 base_temperature=narrator_temp)
    elif provider == "anthropic":
        call = _real_anthropic_call(resolved_model, max_tokens, timeout,
                                    base_temperature=narrator_temp)
    else:
        raise ValueError(f"no narrator dispatch for provider={provider!r}")

    tracker_lock = threading.Lock()
    n_total = len(journeys)
    results: list[str | None] = [None] * n_total

    def _one(idx: int, journey: Journey) -> tuple[int, str]:
        key = _journey_cache_key(journey, resolved_model)
        cached = _cache_get(cache_dir, key)
        if cached is not None:
            with tracker_lock:
                tracker.cache_hit()
            return idx, cached

        # Pre-check budget (best-effort; the post-charge check below is
        # the hard gate).
        with tracker_lock:
            if tracker.over_budget():
                raise RuntimeError(
                    f"narrator budget exceeded before idx={idx}: "
                    f"${tracker.spent_usd:.4f} >= ${tracker.budget_usd:.2f}"
                )

        last_body = ""
        prev_hits: list = []
        for attempt in range(max_retries + 1):
            body, in_toks, out_toks = call(journey, attempt, prev_hits)
            with tracker_lock:
                tracker.charge(resolved_model, in_toks, out_toks)
                if tracker.spent_usd > tracker.budget_usd:
                    raise RuntimeError(
                        f"narrator budget exceeded at idx={idx}: "
                        f"${tracker.spent_usd:.4f} > ${tracker.budget_usd:.2f}. "
                        f"Up to {max_workers - 1} other in-flight calls may "
                        f"also complete before the executor halts."
                    )
            body, _hits = fence(body)
            scan = narrative_leakage_scan(body)
            if scan["clean"]:
                _cache_put(cache_dir, key, body)
                return idx, body
            last_body = body
            prev_hits = scan["hits"]

        raise RuntimeError(
            f"narrative-leakage scan failed after {max_retries + 1} attempts "
            f"for journey idx={idx} seed={journey.seed} "
            f"family={journey.journey_family}. Hits: {prev_hits}. "
            f"Last body: {last_body!r}"
        )

    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_one, i, j) for i, j in enumerate(journeys)]
        for fut in concurrent.futures.as_completed(futures):
            idx, body = fut.result()
            results[idx] = body
            n_done += 1
            if progress_callback is not None:
                # Read tracker outside the lock — eventual-consistent is
                # fine for progress reporting.
                progress_callback(n_done, n_total, tracker.spent_usd)

    assert all(r is not None for r in results), \
        "concurrent narrator path left holes — investigate before relying on output"
    return results  # type: ignore[return-value]


def generate_narrative(
    journey: Journey,
    *,
    tracker: CostTracker,
    model: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    narrator_fn: NarratorFn | None = None,
    narrator_temp: float = 0.3,
) -> str:
    """Produce one LLM narrative for `journey`. Returns the body string
    (no verdict footer; serializer appends that).

    Args:
        journey:        the Journey to narrate.
        tracker:        CostTracker that records spend; consult
                        `tracker.over_budget()` BEFORE calling to avoid
                        blowing the cap.
        model:          explicit model id ("gpt-5.4-nano" / "claude-haiku-4-5-...").
                        Default None means: pick the default model for whichever
                        provider LLM_PROVIDER selects (openai → gpt-5.4-nano,
                        anthropic → claude-haiku-4-5-20251001). Caller-supplied
                        model + LLM_PROVIDER override that disagree raise.
                        Review 009 finding #1.
        narrator_fn:    optional injected narrator (for testing); when
                        None, the OpenAI or Anthropic client is chosen
                        based on LLM_PROVIDER env var or model prefix.

    Raises:
        RuntimeError if budget is exceeded or narrator produces leaky
        text that cannot be regenerated within max_retries.
        ValueError   if LLM_PROVIDER and the model name imply different
        providers.
    """
    if tracker.over_budget():
        raise RuntimeError(
            f"narrator budget exceeded: ${tracker.spent_usd:.2f} >= "
            f"${tracker.budget_usd:.2f}"
        )

    provider, model = _resolve_provider_and_model(
        model, os.environ.get("LLM_PROVIDER"),
    )

    key = _journey_cache_key(journey, model)
    cached = _cache_get(cache_dir, key)
    if cached is not None:
        tracker.cache_hit()
        return cached

    # Provider dispatch — resolved above, so the model string and the
    # client factory always agree.
    if narrator_fn is not None:
        call = narrator_fn
    elif provider == "openai":
        call = _real_openai_call(model, max_tokens, timeout,
                                 base_temperature=narrator_temp)
    elif provider == "anthropic":
        call = _real_anthropic_call(model, max_tokens, timeout,
                                    base_temperature=narrator_temp)
    else:
        # Defensive: _resolve_provider_and_model already raises on unknown
        # providers, so reaching here means someone added a new value
        # without updating the dispatch.
        raise ValueError(
            f"no narrator dispatch for provider={provider!r}"
        )

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

    # Review 009 finding #1: provider/model dispatch resolution. These run
    # without any API key and without the openai/anthropic SDKs being
    # installed (we only construct the (provider, model) tuple — we never
    # build a client).
    assert _resolve_provider_and_model(None, None) == ("openai", DEFAULT_OPENAI_MODEL)
    assert _resolve_provider_and_model(None, "anthropic") == ("anthropic", DEFAULT_ANTHROPIC_MODEL)
    assert _resolve_provider_and_model(None, "openai") == ("openai", DEFAULT_OPENAI_MODEL)
    assert _resolve_provider_and_model("gpt-5.4-mini", None) == ("openai", "gpt-5.4-mini")
    assert _resolve_provider_and_model("claude-haiku-4-5-20251001", None) \
        == ("anthropic", "claude-haiku-4-5-20251001")
    # Agreement: env matches model -> OK.
    assert _resolve_provider_and_model("gpt-5.4-nano", "openai") == ("openai", "gpt-5.4-nano")
    # Disagreement: env says anthropic, model is gpt-* -> must raise.
    try:
        _resolve_provider_and_model("gpt-5.4-nano", "anthropic")
        raise AssertionError("expected ValueError for provider/model disagreement")
    except ValueError as e:
        assert "disagree" in str(e), f"unexpected message: {e}"
    print("  provider/model resolution OK (review 009 finding #1)")

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

        # Anthropic-default path: setting LLM_PROVIDER=anthropic without an
        # explicit model must route the stub call with the haiku model
        # (the bug review 009 caught was that this used to pass gpt-5.4-nano).
        prev = os.environ.get("LLM_PROVIDER")
        os.environ["LLM_PROVIDER"] = "anthropic"
        try:
            ant_tracker = CostTracker(budget_usd=10.0)
            generate_narrative(
                j, tracker=ant_tracker, cache_dir=Path(tdir) / "ant",
                narrator_fn=_stub_narrator(),
            )
            # Stub doesn't track which model it was called with, but the
            # tracker.breakdown is keyed by the resolved model string.
            charged_models = list(ant_tracker.breakdown.keys())
            assert charged_models == [DEFAULT_ANTHROPIC_MODEL], (
                f"expected Anthropic default to bill {DEFAULT_ANTHROPIC_MODEL!r}, "
                f"got {charged_models}"
            )
        finally:
            if prev is None:
                os.environ.pop("LLM_PROVIDER", None)
            else:
                os.environ["LLM_PROVIDER"] = prev
        print("  Anthropic-default dispatch OK (review 009 finding #1)")

        # Budget cap
        cap = CostTracker(budget_usd=0.0)
        try:
            generate_narrative(j, tracker=cap, cache_dir=cache_dir,
                               narrator_fn=_stub_narrator())
            raise AssertionError("expected budget overage")
        except RuntimeError as e:
            assert "budget" in str(e).lower()

        # Concurrent batch path — same stub narrator, 32 journeys, 4 workers.
        # Verifies ordering, tracker accounting, and that the batch wrapper
        # does not double-charge (cache hits don't fire in a fresh cache
        # dir, so n_calls should equal n_journeys after the first batch).
        conc_tracker = CostTracker(budget_usd=10.0)
        conc_cache = Path(tdir) / "conc"
        batch_journeys = [
            gen_journey("clean", seed=100 + k, actor="human") for k in range(32)
        ]
        progress_events: list[tuple[int, int, float]] = []
        bodies = generate_narratives_concurrent(
            batch_journeys, tracker=conc_tracker, cache_dir=conc_cache,
            narrator_fn=_stub_narrator(), max_workers=4,
            progress_callback=lambda d, t, s: progress_events.append((d, t, s)),
        )
        assert len(bodies) == 32, f"expected 32 bodies, got {len(bodies)}"
        assert all(b for b in bodies), "concurrent path returned an empty body"
        # Ordering: bodies[i] corresponds to batch_journeys[i]. Stub narrator
        # returns family-specific text, so bodies are not all identical; but
        # they are deterministic per-journey, so a re-call returns the same.
        assert conc_tracker.n_calls == 32, \
            f"expected 32 charged calls, got {conc_tracker.n_calls}"
        assert conc_tracker.n_cache_hits == 0, \
            f"fresh cache should have 0 hits, got {conc_tracker.n_cache_hits}"
        # Progress callback fired once per completion.
        assert len(progress_events) == 32, \
            f"expected 32 progress events, got {len(progress_events)}"
        # Final event reports n_done == n_total.
        assert progress_events[-1][0] == 32 and progress_events[-1][1] == 32

        # Re-run the same batch: cache should serve everything, zero charges.
        replay_tracker = CostTracker(budget_usd=10.0)
        replay_bodies = generate_narratives_concurrent(
            batch_journeys, tracker=replay_tracker, cache_dir=conc_cache,
            narrator_fn=_stub_narrator(), max_workers=4,
        )
        assert replay_bodies == bodies, "cache replay produced different bodies"
        assert replay_tracker.n_calls == 0, \
            f"cache replay should bill 0 calls, got {replay_tracker.n_calls}"
        assert replay_tracker.n_cache_hits == 32
        print("  concurrent batch path OK (32 journeys, 4 workers, cache replay clean)")

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
