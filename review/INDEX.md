# Review index

Single-page view of all reviews. See `review/README.md` for conventions.

**Running a new review?** Point Codex at `review/CODEX_PROMPT.md` (or
paste: `Review the cross_attn_ato_poc repo per review/CODEX_PROMPT.md.
Follow it exactly.`).

**Convention:** `Closing commit` is exactly one short git hash — the latest
commit that advanced this review's status. Multi-commit context belongs in
the corresponding `followup.txt`, not here.

| #   | Slug                 | Date       | Findings | Status         | Closing commit |
|-----|----------------------|------------|----------|----------------|----------------|
| 001 | plan-and-scaffold    | 2026-05-15 | 6        | partial-closed | b903809        |
| 002 | followup-on-001      | 2026-05-15 | 3        | closed         | 67ee1e7        |
| 003 | recent-repo-state    | 2026-05-15 | 4        | closed         | 87070cf        |
| 004 | path-a-batch-2       | 2026-05-15 | 7        | closed         | 5916a53        |
| 005 | path-a-batch-3       | 2026-05-15 | 5        | closed         | a126761        |
| 006 | followup-on-005      | 2026-05-15 | 3        | closed         | 3981d96        |
| 007 | path-a-batch-4       | 2026-05-15 | 5        | closed         | d9a4f7f        |

---

## Status legend

- `open` — comments.txt exists, no followup.txt yet.
- `in-progress` — followup.txt exists but at least one finding still in PARTIAL.
- `closed` — all findings FIXED or ACKNOWLEDGED.
- `partial-closed` — most findings closed; at least one DEFERRED (out of scope for this pass).
- `deferred` — entire review deferred (rare).

## Latest open question

001 Finding 1 → Path A (build Day-1/2 implementation files locally) is the
follow-on work. Batch 1 of 4 is shipped; Batches 2-4 still to come. The
Closing commit column above carries the per-review baseline hash; do not
read hashes out of this prose section.
