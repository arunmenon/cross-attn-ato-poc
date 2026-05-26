#!/usr/bin/env python3
"""
Merge the five whitepaper markdown documents into a single comprehensive file.

Output: whitepaper-comprehensive.md (in the whitepaper/ folder, alongside the
source documents and the figures/ directory so relative figure paths work).

Design:
  - Master document (00) becomes Part I and keeps its full structure: front
    matter, abstract, claims-at-a-glance, introduction, related work, method
    overview, experimental arc, results, discussion, limitations, conclusion,
    references.
  - Companion documents (01-04) become Parts II-V respectively. Each gets a
    "Part N" divider page and a brief preamble noting which §3.X section it
    expands.
  - Cross-references like "see companion `01-data-curation-and-distribution.md`"
    are rewritten to "see Part II".
  - File-level metadata headers (the "v1.2 · 2026-05-22" lines that appear at
    the top of each companion) are dropped since the unified doc has one such
    line on the title page.
  - The References block from 00 is preserved at the end. References mentioned
    only in companion files are merged in.
  - Figure embeds use the original relative path `figures/X.svg` since the
    merged doc lives in `whitepaper/` next to `whitepaper/figures/`. This keeps
    GitHub's web preview rendering correctly AND lets pandoc resolve images
    when `build_pdf.sh` runs from the whitepaper/ folder.
"""

import re
from pathlib import Path

# Script-relative path: the merger lives in <repo>/whitepaper/, so the repo
# root is two parents up. Works wherever the repo is cloned.
SCRIPT_DIR = Path(__file__).resolve().parent
WHITEPAPER_DIR = SCRIPT_DIR
REPO_ROOT = WHITEPAPER_DIR.parent
FIGURES_DIR = WHITEPAPER_DIR / "figures"

# We will produce the merged doc inside whitepaper/ so the relative
# figure path "figures/..." continues to work for both markdown
# readers and pandoc.
OUTPUT_MD = WHITEPAPER_DIR / "whitepaper-comprehensive.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_file_header(text: str) -> str:
    """Remove the file-level metadata header from a companion document.

    Each companion starts with:
        # <Title>

        **Whitepaper companion document · v1.2 · 2026-05-22**

        <description paragraph>

        ---

    We want to drop the version stamp line (since the unified doc has one
    title-page version stamp) but keep the title and the description.
    """
    lines = text.splitlines()
    out = []
    for line in lines:
        if line.startswith("**Whitepaper companion document"):
            continue  # drop the version line
        out.append(line)
    return "\n".join(out)


def adjust_cross_refs(text: str) -> str:
    """Rewrite cross-references to companion documents into Part references."""
    mapping = [
        # Order matters: long forms before short forms.
        (r"`00-whitepaper-main\.md`", "Part I (Master Narrative)"),
        (r"`01-data-curation-and-distribution\.md`", "Part II (Data Curation and Distribution)"),
        (r"`02-agentic-experiment-harness\.md`", "Part III (Agentic Experiment Harness)"),
        (r"`03-eval-strategy\.md`", "Part IV (Eval Strategy)"),
        (r"`04-cross-attention-experiments\.md`", "Part V (Cross-Attention Experiments)"),
        # Shorter forms found in companion footers like "see `01-data` §3".
        # Be careful: only match the exact short tags, not arbitrary content.
        (r"\b01-data §", "Part II §"),
        (r"\b02-harness §", "Part III §"),
        (r"\b03-eval §", "Part IV §"),
        (r"\b04 §", "Part V §"),
    ]
    for pat, repl in mapping:
        text = re.sub(pat, repl, text)
    return text


def normalize_figure_paths(text: str) -> str:
    """Keep figure embeds as relative `figures/X.svg` paths.

    The merged markdown lives in `whitepaper/` and the figures live in
    `whitepaper/figures/`, so the original relative path resolves correctly
    for both GitHub's web preview and pandoc (when `build_pdf.sh` runs from
    the whitepaper/ folder via `cd "$(dirname "$0")"`).

    Currently this is a no-op preserved as a hook in case future merges need
    path rewriting (e.g., if the merged doc moves to a different folder).
    """
    return text


def fix_figure_captions(text: str) -> str:
    """Strip the leading "Figure N. " prefix from image alt text so pandoc's
    auto-numbered "Figure N:" caption doesn't double up.

    Before:
        ![Figure 2. Auto-research loop dataflow](figures/fig2-auto-research-loop.svg)
    After:
        ![Auto-research loop dataflow](figures/fig2-auto-research-loop.svg){ width=100% }

    The {width=100%} attribute enlarges the figure in the PDF (so dense
    diagrams don't render postage-stamp-sized in portrait) without affecting
    GitHub / Typora markdown rendering, which ignores unknown attributes.
    """
    pattern = r"!\[Figure\s+\d+\.\s+([^\]]+)\]\((figures/[^)]+)\)"
    return re.sub(pattern, r"![\1](\2){ width=100% }", text)


# Canonical figure numbers that match the labels embedded inside the SVG
# title bars and the prose references throughout the paper. Order of FIRST
# appearance in the merged document does NOT determine numbering — these
# fixed assignments do, so the in-image "Figure N" label always matches the
# PDF caption number.
FIGURE_CANONICAL_NUMBERS = {
    "fig1-architecture": 1,
    "fig2-auto-research-loop": 2,
    "fig3-data-distribution": 3,
    "fig4-sweep-results": 4,
}


# Header-line substrings that uniquely identify the worst wide tables.
# When the merger sees one of these as the header row of a markdown table,
# it converts that table to a raw LaTeX landscape + scriptsize tabular
# block so the columns fit and the page rotates to give the table extra
# horizontal room.
WIDE_TABLE_SIGNATURES = [
    "| Run | Config | Worst-family HN-FPR",   # v3 leaderboard
    "| exp_id | Config | v5_adv_error",       # v5 Phase-1 sweep
]


def _md_cell_to_latex(cell: str) -> str:
    """Convert a markdown table cell to LaTeX. Strips leading/trailing
    whitespace; converts inline `code` -> \\texttt{}, **bold** -> \\textbf{};
    escapes LaTeX specials (_, &, %, #) everywhere, including inside the
    \\texttt{} and \\textbf{} groups (LaTeX still interprets _ as subscript
    inside \\textbf{}; only \\texttt{} naturally handles certain specials,
    but underscores still need escaping there too).
    """
    s = cell.strip()
    # First, convert markdown markers to LaTeX, but use SENTINELS instead
    # of the real \texttt{}/\textbf{} so the escape pass below treats them
    # uniformly.
    s = re.sub(r"`([^`]+)`", r"§CODE§\1§/CODE§", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"§BOLD§\1§/BOLD§", s)
    # Now escape all LaTeX specials in the whole string (including inside
    # the sentinels — they will be wrapped in \texttt{}/\textbf{} which
    # do NOT auto-escape underscores).
    s = s.replace("\\", "\\textbackslash{}")  # raw backslashes (rare)
    for ch in ("&", "_", "%", "#", "$"):
        s = s.replace(ch, "\\" + ch)
    # Restore the markdown sentinels as LaTeX commands.
    s = s.replace("§CODE§", "\\texttt{").replace("§/CODE§", "}")
    s = s.replace("§BOLD§", "\\textbf{").replace("§/BOLD§", "}")
    return s


def _md_table_to_latex(table_lines: list[str]) -> str:
    """Convert a markdown table (header, separator, rows) into a raw LaTeX
    landscape + scriptsize tabular block."""
    # Split each line on '|', drop the empty first/last elements.
    rows = []
    for line in table_lines:
        cells = [c for c in line.split("|")]
        # leading and trailing empty strings (from outer pipes) — drop them
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]
        rows.append(cells)
    if len(rows) < 2:
        return "\n".join(table_lines)
    header_cells = rows[0]
    # rows[1] is the markdown separator (---); skip it
    data_rows = rows[2:]
    n_cols = len(header_cells)
    col_spec = "l" * n_cols  # all left-aligned (CIs read better left-aligned in narrow space)

    parts = [
        "",
        "\\begin{landscape}",
        "\\begin{center}",
        "\\scriptsize",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(f"\\textbf{{{_md_cell_to_latex(c)}}}" for c in header_cells) + " \\\\",
        "\\hline",
    ]
    for row in data_rows:
        # pad/truncate to n_cols
        if len(row) < n_cols:
            row = row + [""] * (n_cols - len(row))
        elif len(row) > n_cols:
            row = row[:n_cols]
        parts.append(" & ".join(_md_cell_to_latex(c) for c in row) + " \\\\")
    parts.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{center}",
        "\\end{landscape}",
        "",
    ])
    return "\n".join(parts)


def wrap_wide_tables_in_landscape(text: str) -> str:
    """Convert the v3 leaderboard and v5 Phase-1 tables to raw LaTeX
    landscape + scriptsize tabular blocks.

    Source markdown stays portrait (so GitHub renders cleanly). Only the
    comprehensive PDF gets the rotation. We detect the table by its header
    row's substring signature, then collect every consecutive line that
    starts with `|` (the markdown table block) and replace the whole block
    with a raw LaTeX equivalent.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("|") and any(sig in line for sig in WIDE_TABLE_SIGNATURES):
            # Collect this table — every consecutive |-prefixed line.
            j = i
            while j < n and lines[j].startswith("|"):
                j += 1
            table_lines = lines[i:j]
            out.append(_md_table_to_latex(table_lines))
            i = j
        else:
            out.append(line)
            i += 1
    return "\n".join(out)


def assign_figure_numbers(text: str) -> str:
    """Force canonical figure numbers (matching in-image labels) and mark
    repeat appearances as un-numbered "Repeated from Figure N" blocks.

    The merged whitepaper embeds each figure multiple times (once in Part I
    where it's first introduced, once again in the dedicated companion Part
    that expands on it). Pandoc's default numbering scheme assigns every
    appearance a new number, which (a) makes the second copy disagree with
    the in-image title bar, and (b) re-orders the canonical 1/2/3/4 because
    the first occurrences in the master narrative are out of canonical
    order.

    Strategy:
      * First appearance of fig{1,2,3,4}: emit a raw LaTeX \\setcounter{figure}
        line that primes the counter so pandoc's \\begin{figure} wrapper
        increments to the canonical number.
      * Subsequent appearances: emit a raw LaTeX \\begin{center}\\includegraphics
        block (NOT wrapped in \\begin{figure}), so the figure counter is
        untouched, with an italic "Repeated from Figure N — <caption>"
        line below.
    """
    seen: set[str] = set()

    def repl(match: re.Match) -> str:
        alt = match.group(1).strip()
        path = match.group(2)
        # detect figure key from path basename
        path_match = re.search(r"(fig[1-9]-[a-z0-9-]+)", path)
        if not path_match:
            return match.group(0)
        key = path_match.group(1)
        canonical_n = FIGURE_CANONICAL_NUMBERS.get(key)
        if canonical_n is None:
            return match.group(0)

        if key in seen:
            # Repeated figure — render inline, no figure counter increment.
            # Escape LaTeX specials in alt text: & is the most common offender
            # in our captions ("Data distribution & eval-mode mix").
            alt_latex = alt.replace("&", "\\&").replace("_", "\\_")
            return (
                "\n\n\\begin{center}\n"
                f"\\includegraphics[width=\\textwidth]{{{path}}}\\\\[0.3em]\n"
                f"\\textit{{Repeated from Figure {canonical_n} — {alt_latex}.}}\n"
                "\\end{center}\n\n"
            )
        seen.add(key)
        # First appearance — prime the counter so \begin{figure} increments
        # to the canonical figure number. width=100% kept for PDF sizing.
        return (
            f"\n\n\\setcounter{{figure}}{{{canonical_n - 1}}}\n\n"
            f"![{alt}]({path}){{ width=100% }}\n\n"
        )

    # Match standalone images (with optional {width=...} or { width=...} attrs)
    pattern = r"!\[([^\]]+)\]\(([^)]+)\)(?:\s*\{[^}]*\})?"
    return re.sub(pattern, repl, text)


# NOTE on PDF rendering: the comprehensive markdown intentionally keeps
# Unicode glyphs like ★, α, ≥, ≤, ∈ since GitHub / VS Code / Typora all
# render them fine.  For PDF build (xelatex + DejaVu Serif), the build
# script applies a small sed substitution for ★ specifically (the only
# glyph DejaVu Serif lacks).  See `whitepaper/build_pdf.sh`.


# ---------------------------------------------------------------
# Part headers (page-break separators between major sections)
# ---------------------------------------------------------------

def part_divider(roman: str, label: str, title: str, blurb: str) -> str:
    """Produce a Pandoc-friendly part divider that will translate into a
    visually distinct section in both Markdown viewers and the PDF.
    """
    return f"""

\\newpage

# Part {roman} — {title}

> *{label}*
>
> {blurb}

\\vspace{{1em}}

"""


# ---------------------------------------------------------------
# Master document handling
# ---------------------------------------------------------------

def process_master(text: str) -> str:
    """The master document is mostly preserved as-is.  We just adjust
    cross-refs, drop the file-level version stamp (it goes on the title
    page), and absolutize figure paths.
    """
    # Drop the file-level version stamp lines
    lines = text.splitlines()
    out = []
    skip_next_blank = False
    for line in lines:
        if line.startswith("**Whitepaper · v1.2"):
            skip_next_blank = True
            continue
        if skip_next_blank and not line.strip():
            skip_next_blank = False
            continue
        out.append(line)
    text = "\n".join(out)
    text = adjust_cross_refs(text)
    text = normalize_figure_paths(text)
    text = fix_figure_captions(text)
    return text


def process_companion(text: str) -> str:
    """Companion documents: strip the file-level version stamp, rewrite
    cross-refs, absolutize figures.
    """
    text = strip_file_header(text)
    text = adjust_cross_refs(text)
    text = normalize_figure_paths(text)
    text = fix_figure_captions(text)
    return text


# ---------------------------------------------------------------
# Title page
# ---------------------------------------------------------------

TITLE_PAGE = """---
title: |
  Cross-Attention for Account-Takeover Detection

  A Three-Generation Study Driven by an Agentic Experiment Harness
author: Arun Menon · Foundation Science · PayPal
date: "v1.2 (comprehensive) · 2026-05-25"
documentclass: report
geometry: margin=1in
fontsize: 11pt
linkcolor: NavyBlue
urlcolor: NavyBlue
toc: true
toc-depth: 3
numbersections: false
colorlinks: true
header-includes:
  - \\usepackage{titlesec}
  - \\titleformat{\\chapter}[display]{\\normalfont\\huge\\bfseries}{\\chaptertitlename\\ \\thechapter}{20pt}{\\Huge}
  - \\usepackage{fancyhdr}
  - \\pagestyle{fancy}
  - \\fancyhf{}
  - \\fancyhead[L]{\\nouppercase{\\leftmark}}
  - \\fancyhead[R]{Cross-Attention for ATO · v1.2}
  - \\fancyfoot[C]{\\thepage}
  - \\renewcommand{\\headrulewidth}{0.4pt}
  # --- Table rendering: shrink to footnotesize and allow long-pipe-table cells
  # to wrap on underscores in monospace identifiers (the v4/v5 result tables
  # are the worst offenders: phish_takeover_mfa_phished + multi-CI columns
  # overflow the page at the default size).
  - \\usepackage{longtable}
  - \\AtBeginEnvironment{longtable}{\\footnotesize}
  - \\AtBeginEnvironment{tabular}{\\footnotesize}
  # Make underscores in monospace identifiers a line-break opportunity so
  # column-wrap defects (e.g., "phish_takeover_mfa phishedtakeover recall")
  # no longer happen.  This relies on \\seqsplit from the seqsplit package.
  - \\usepackage{seqsplit}
  # --- Landscape pages for the widest leaderboard tables (v3, v5 Phase-1)
  # — these have 6–7 columns including long CI strings and overflow at any
  # portrait font size. The landscape wrapper is injected by merge_whitepaper.py
  # only for the comprehensive PDF (source markdown stays unaffected).
  - \\usepackage{pdflscape}
  # --- Code blocks: switch to fvextra so long lines wrap instead of clipping
  # off the right margin (multiple bash commands and JSON paths were
  # truncated in the previous build).
  - \\usepackage{fvextra}
  - \\fvset{breaklines=true,breakanywhere=true,fontsize=\\small}
  # Also shrink the plain `verbatim` environment that pandoc uses for
  # indented code blocks (fvextra does NOT patch base LaTeX's verbatim).
  # The narrator-event sample lines are ~107 chars wide; scriptsize fits
  # comfortably in a 6.5" text width.
  - \\AtBeginEnvironment{verbatim}{\\scriptsize}
  # --- Figures: a little extra room above captions so the auto-numbered
  # "Figure N:" doesn't crowd the alt text.
  - \\setlength{\\abovecaptionskip}{6pt}
---

\\thispagestyle{empty}

\\vspace*{2cm}

\\begin{center}
\\Huge\\textbf{Cross-Attention for Account-Takeover Detection}\\\\[0.5em]
\\Large\\textit{A Three-Generation Study Driven by an Agentic Experiment Harness}\\\\[2em]
\\large Arun Menon \\\\
Foundation Science · PayPal \\\\[1em]
\\normalsize v1.2 (comprehensive) · 2026-05-25 \\\\[2em]
\\end{center}

\\begin{center}
\\large\\textbf{Abstract}
\\end{center}

**The question.** Account-takeover (ATO) detection has two natural views of every session — an analyst-style narrative of what happened, and a chronological log of structured behavioral events (logins, transactions, device changes, with bucketed features such as `amount_bucket=high`). Language models read narratives well; the event stream is where the fraud signal historically lives. This paper asks whether **Flamingo-style gated cross-attention** — a frozen LM that attends to a side stream of structured events through inserted gated layers — can bridge that modality gap, on a **synthetic** ATO dataset modeled on PayPal session schemas. All results are on synthetic data; production transfer is explicitly out of scope.

**The arc.** The work went through three sweep generations, which we label v3, v4, and v5 throughout the paper. **v3** was the original 3-day proof-of-concept; it returned a null result — no architectural lift over the baseline. **v4** was a redesign of the synthetic dataset, after a code audit revealed that v3's null was caused by the way the data was generated, not by the architecture (the LLM that wrote the narratives could see the structured events and ended up describing them in the text — so the two views the architecture was meant to combine ended up carrying the same information). **v5** then ran 11 variations of the architecture on the v4 data to test whether the win held up — it did, robustly — and exposed a new data-side bottleneck. The labels track on-disk artifacts (experiments.jsonl rows, dataset versions, metric schemas) so every claim is reproducible against a specific generation.

**What the paper contributes.** *First, the research system itself.* An AI agent picks the next experiment to run, and a strict Python script does everything else — validates the config, locks the GPU, launches the job, parses the metrics, computes confidence intervals, and writes one tamper-proof line to a history log. Only one piece of code is ever allowed to write to that log, so the records can't get scrambled even when the agent retries or several runs queue up. The system ran 30 experiments across the three generations with no scrambled records, no clashing writes, and no manual cleanup. *Second, what we discovered with that system about the v3 synthetic data.* Two flaws in how the data was generated had quietly made v3 unable to test what it was supposed to test: (a) the LLM that wrote the narratives could see the structured events, so it copied the fraud-signal features into the text — leaving the two views the architecture was meant to combine carrying the same information; (b) the hard-negative templates used fixed feature combinations, so a simple feature-only classifier could already separate fraud from legitimate at near-perfect accuracy — meaning the benchmark cross-attention was being compared against wasn't measuring model capability, it was measuring a template artifact. Once both flaws were fixed in the v4 data pivot, the same cross-attention configuration that produced a null result in v3 produced a clear win on adversarial cross-modal fraud. On `phish_takeover_mfa_phished` — a deliberately hard scenario family we built into the eval, where the phisher has also captured the victim's MFA code, so the narrative reads like a routine login — the **text-only baseline** (the same frozen LM reading only the narrative, with no access to the structured events) caught 0% of cases and cross-attention caught 97.2% (95% confidence interval [93.1%, 100%]; the bracketed range is the band where the true detection rate most likely sits, and its lower bound sitting well above zero is what makes the win statistically reliable rather than a lucky run). The v5 sweep then ran 11 variations of the architecture and confirmed the win held up across all of them, while revealing one stubborn family of cases (large-amount legitimate account recoveries) where no architectural change moved the false-positive rate — a data-side problem, not a model problem. The general recipe — agent-driven experiment loop, honest confidence intervals on every number, and synthetic data that gets fixed when it misleads — applies well beyond this architecture and task.

\\textbf{The cross-attention finding is the worked example; the loop is the reusable artifact.}

\\newpage

"""

# ---------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------

def build():
    master = process_master(read(WHITEPAPER_DIR / "00-whitepaper-main.md"))
    data = process_companion(read(WHITEPAPER_DIR / "01-data-curation-and-distribution.md"))
    harness = process_companion(read(WHITEPAPER_DIR / "02-agentic-experiment-harness.md"))
    eval_doc = process_companion(read(WHITEPAPER_DIR / "03-eval-strategy.md"))
    experiments = process_companion(read(WHITEPAPER_DIR / "04-cross-attention-experiments.md"))

    # The master document already has its own title/abstract.  We replace
    # the title-page block of TITLE_PAGE with our own setup, then concat
    # the master verbatim, then the four companions with dividers.
    #
    # Strip the first H1 of the master (the title) since the title-page
    # already shows it — but keep the rest verbatim.
    master_lines = master.splitlines()
    # Drop everything from the first "# ..." line through the line just
    # before the abstract section.  Specifically, drop lines until we
    # reach "## Executive summary" (which is master's first major heading
    # after the front-matter).
    start_idx = 0
    for i, line in enumerate(master_lines):
        if line.startswith("## Executive summary"):
            start_idx = i
            break
    master_body = "\n".join(master_lines[start_idx:])

    parts = [
        TITLE_PAGE,
        # Part I — the master narrative (executive summary + claims table + how to read + abstract + §1-§8 + references)
        part_divider(
            "I", "MASTER NARRATIVE",
            "The full story end-to-end",
            "Executive summary, claims-at-a-glance, how to read this paper, abstract, related work, method overview, "
            "the v3 → v4 → v5 experimental arc, results, discussion, limitations, and conclusion. Parts II–V expand "
            "the four methodological pillars in turn."
        ),
        master_body,
        part_divider(
            "II", "DATA CURATION AND DISTRIBUTION",
            "Expands Part I §3.1",
            "The synthetic-data pipeline: three token families, journey × actor schema, bucketed features, narrator policy, "
            "leakage controls, the v3→v4 four-change pivot, and the v4/v5 data distribution audited against the schema."
        ),
        data,
        part_divider(
            "III", "AGENTIC EXPERIMENT HARNESS",
            "Expands Part I §3.2",
            "The Karpathy-style auto-research loop: agent / launcher ownership split, 10-step launcher pipeline, halt-condition design, "
            "dedup tuple, expanded-sweep directive, the three-generation evolution of the harness, and what generalizes beyond this POC."
        ),
        harness,
        part_divider(
            "IV", "EVAL STRATEGY",
            "Expands Part I §3.3",
            "Three eval modes (stripped / opaque / full), three eval-set sizes (5k / 15k / 50k), the headline-metric evolution "
            "(metric_version 1 → 2 → 5), the tie-aware exact-target operating-point computation with worked example, bootstrap-CI "
            "derivation, the v3 sklearn-cliff finding, and the leakage-control regime applied at eval time."
        ),
        eval_doc,
        part_divider(
            "V", "CROSS-ATTENTION EXPERIMENTS",
            "Expands Part I §3.4 and §4",
            "Architecture detail (Qwen3-8B + side encoder + Perceiver-Resampler + gated cross-attention + LoRA-on-Q), training recipe, "
            "the full v3+v4+v5 leaderboard with bootstrap CIs and gate magnitudes, ablation reads per dial, the gates story across three "
            "generations, the data-ceiling diagnostic, and the concrete next-step recommendations."
        ),
        experiments,
    ]

    merged = "\n\n".join(parts)
    # Apply canonical-number + repeated-figure post-processing globally so the
    # state (which figures we've already seen) spans master + all companions.
    merged = assign_figure_numbers(merged)
    # Wrap the widest leaderboard tables (v3, v5 Phase-1) in landscape pages so
    # their 6–7 columns no longer overflow.
    merged = wrap_wide_tables_in_landscape(merged)
    OUTPUT_MD.write_text(merged, encoding="utf-8")

    print(f"Wrote {OUTPUT_MD}")
    print(f"  size: {OUTPUT_MD.stat().st_size:,} bytes")
    print(f"  approximate words: {len(merged.split()):,}")


if __name__ == "__main__":
    build()
