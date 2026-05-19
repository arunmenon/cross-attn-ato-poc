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
| 008 | pod-readiness        | 2026-05-16 | 6        | closed         | 8852a59        |
| 009 | narrator-switch-and-prep | 2026-05-16 | 4    | closed         | 3648259        |
| 010 | blackwell-compat-patch | 2026-05-16 | 3    | closed         | a95eb79        |
| 011 | vslice-debug-and-xattn-preflight | 2026-05-17 | 5 | closed     | db723c1        |
| 012 | day-1-writeup        | 2026-05-17 | 5        | closed         | d6aa58f        |
| 013 | auto-loop-readiness  | 2026-05-17 | 7        | closed         | 2e325e2        |
| 014 | auto-loop-prelaunch-audit | 2026-05-17 | 24   | partial-closed | 7d04064        |
| 015 | audit-014-meta-review | 2026-05-17 | 11      | closed         | 7d04064        |
| 016 | fix-pass-014-015     | 2026-05-17 | 2        | closed         | 16549b0        |
| 017 | review-016-response  | 2026-05-17 | 1        | closed         | 7e7201d        |
| 018 | day-2-baseline-findings | 2026-05-17 | 3      | closed         | a373c14        |
| 019 | plan-review-baseline-correction | 2026-05-17 | 6 | closed       | a373c14        |
| 020 | baseline-correction-implementation | 2026-05-17 | 5 | closed   | e36c7a8        |
| 021 | v4-phase1-phase2     | 2026-05-19 | pending  | open           | 330e150        |

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
