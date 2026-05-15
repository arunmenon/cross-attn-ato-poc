# Review folder conventions

This folder is the **paired-review protocol** between two roles:

- **Reviewer** (Codex, or any agent invoked for review): reads the repo and
  PLAN.md, writes findings to a new review folder.
- **Maintainer** (Claude Code, or the human): reads the findings, applies
  fixes, writes a follow-up file in the same folder, updates the index, and
  commits.

Reviewer and Maintainer alternate as the project advances. INDEX.md is the
single page anyone can open to see the state of all reviews.

---

## Folder layout

```
review/
├── README.md                          ← this file
├── INDEX.md                           ← table of all reviews + status
├── 001-plan-and-scaffold/
│   ├── comments.txt                   ← reviewer writes
│   └── followup.txt                   ← maintainer writes
├── 002-followup-on-001/
│   ├── comments.txt
│   └── followup.txt
└── NNN-<short-slug>/
    └── …
```

Rules:

- One folder per review pair. Name: `NNN-<short-slug>/` where `NNN` is
  zero-padded and increments by 1. Pick the next free number.
- Inside the folder, filenames are always `comments.txt` and `followup.txt`.
  No variants. No date suffixes (the date lives in the file header).
- Plain text (`.txt`), not Markdown. The format is readable on any pager
  and avoids markdown-renderer divergence.
- Folder is **append-only** in spirit: once a comments.txt is written, the
  reviewer does not edit it later — instead, a new review folder is opened.

---

## Reviewer protocol (what Codex does)

When invoked to review:

1. **Pick the next folder number.** `ls review/` → find the highest `NNN-…`
   prefix → use `NNN+1`. Pick a short slug (3-6 words, kebab-case)
   describing the *scope* of this review (not its result). Examples:
   `plan-and-scaffold`, `followup-on-001`, `data-generators`,
   `architecture-surgery`.

2. **Create `review/NNN-<slug>/comments.txt`.** Use the schema below.

3. **Do not modify** anything else under `review/` (no INDEX update, no
   edits to prior `comments.txt` or `followup.txt`). The Maintainer owns
   those.

4. **Do not commit** unless explicitly instructed. The Maintainer's
   follow-up commit typically includes the new comments.txt anyway.

### Schema for `comments.txt`

```
Review comments NNN
Date: YYYY-MM-DD
Subject: <one-line description of what was reviewed>

Reviewed file(s):
    <absolute path 1>
    <absolute path 2 (if any)>

Repo root used for comparison:
    <absolute path to repo root>

Summary:
<2-5 line summary. Tone: opinionated, not hedging. State the bottom line.>

Verification run:                      [OPTIONAL — when commands were executed]

N. <Short description of the check>

Command:
    <exact command run>

Actual output:
    <verbatim output, indented>

Disposition:
    PASS | FAIL | <short description>

Review findings:

N. <Severity>: <one-line summary>.

Status: OPEN, <doc wording only | blocker | etc>

Details:
<paragraph-form details of the finding>

Recommendation:
<concrete action the Maintainer should take>

(repeat per finding; number them N=1,2,3,...)

Bottom line:
<1-3 line wrap-up. What's the high-bit takeaway?>
```

Severity vocabulary (use exactly one per finding, in the title line):

- **Blocker** — must be fixed before the next forward step.
- **High** — should be fixed before the next forward step.
- **Medium** — should be fixed before the next *milestone*.
- **Minor** — nice-to-have; doc wording, small ergonomics.
- **Informational** — no action required; for awareness.

---

## Maintainer protocol (what Claude Code does)

When asked to address a review:

1. **Read** `review/NNN-<slug>/comments.txt` in full.

2. **Address** each finding in repo code/docs/configs.

3. **Write** `review/NNN-<slug>/followup.txt` using the schema below.

4. **Update** `review/INDEX.md` — add the row for this review with its
   status, finding count, and (when applicable) the closing commit hash.

5. **Commit** with a message that mentions `review NNN` (so `git log
   --grep "review NNN"` finds the conversation around it).

### Schema for `followup.txt`

```
Review comments NNN — follow-up
Date: YYYY-MM-DD
Subject: Disposition of findings in NNN-<slug>/comments.txt
Author: scaffold maintainer

Original review file:
    review/NNN-<slug>/comments.txt

Repo state at follow-up:
    <absolute path to repo root>
Baseline commit (optional):
    <git short hash>

Summary:
<2-5 line summary: how many findings, what was accepted, what was deferred.>

Disposition per finding:

N. <Severity>: <copy the finding's one-line summary>.

Status: FIXED | PARTIAL | ACKNOWLEDGED | DEFERRED | REJECTED

Action:
<what was done, or why nothing was done>

Verify:
    <copy-pasteable command that confirms the fix>

(repeat per finding; numbers must match comments.txt)

Open question for the user (optional):
<if a finding requires a user decision before closure>
```

Status vocabulary:

- **FIXED** — finding is closed; the verify command above demonstrates it.
- **PARTIAL** — partially addressed; remaining work documented inline.
- **ACKNOWLEDGED** — accepted but no code change required (e.g., already
  the case, or doc-only).
- **DEFERRED** — accepted but deferred to a later batch/sprint; document
  *why* and *when*.
- **REJECTED** — disagree with the finding; explain the reasoning.

---

## INDEX.md update workflow

After writing `followup.txt`, the Maintainer updates `review/INDEX.md`:

- Add or update the row for this review.
- Status values: `open` / `in-progress` / `closed` / `partial-closed` /
  `deferred`.
- `Closing commit` is the short git hash of the commit that contains the
  fixes. Use `—` when there is no closing commit (e.g., status is
  `acknowledged` or `deferred`).

Keep the index short. One row per review, one line each. Don't expand
into prose — that's what `followup.txt` is for.

---

## Best practices

- **Reviewer**: cite verbatim command output where possible. Quoted output
  beats paraphrasing every time.
- **Maintainer**: end every FIXED finding with a copy-pasteable verify
  command. The next review will use it.
- **Both**: keep severity honest. If everything is a "Blocker", nothing is.
- **Both**: a `followup.txt` does not have to address everything in one
  pass. Marking a finding `DEFERRED` with a reason is better than rushing
  a fix.

---

## Adding a new review (quick reference for Codex)

```bash
# From the repo root:
N=$(ls review/ | grep -E '^[0-9]+-' | sort | tail -1 | cut -d'-' -f1)
NEXT=$(printf "%03d" $((10#${N} + 1)))
SLUG="<your-short-slug>"
mkdir -p "review/${NEXT}-${SLUG}"
# Now write review/${NEXT}-${SLUG}/comments.txt per the schema above.
```
