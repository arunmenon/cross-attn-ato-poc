# Review index

Single-page view of all reviews. See `review/README.md` for conventions.

| #   | Slug                 | Date       | Findings | Status | Closing commit |
|-----|----------------------|------------|----------|--------|----------------|
| 001 | plan-and-scaffold    | 2026-05-15 | 6        | partial-closed (5 fixed, 1 deferred → Path A) | b903809 |
| 002 | followup-on-001      | 2026-05-15 | 3        | closed | 67ee1e7 + b903809 |

---

## Status legend

- `open` — comments.txt exists, no followup.txt yet.
- `in-progress` — followup.txt exists but at least one finding still in PARTIAL.
- `closed` — all findings FIXED or ACKNOWLEDGED.
- `partial-closed` — most findings closed; at least one DEFERRED (out of scope for this pass).
- `deferred` — entire review deferred (rare).

## Latest open question

001 Finding 1 → Path A (build Day-1/2 implementation files locally) is the
follow-on work. Batch 1 of 4 shipped at commit `b903809`. Batches 2-4 still
to come. No reviewer action required until Batch 2 lands.
