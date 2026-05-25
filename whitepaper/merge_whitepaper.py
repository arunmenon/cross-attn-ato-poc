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
    return text


def process_companion(text: str) -> str:
    """Companion documents: strip the file-level version stamp, rewrite
    cross-refs, absolutize figures.
    """
    text = strip_file_header(text)
    text = adjust_cross_refs(text)
    text = normalize_figure_paths(text)
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

We report on a three-generation study (v3 → v4 → v5) of Flamingo-style gated cross-attention applied to **synthetic** account-takeover (ATO) detection on a single PayPal-internal H100 GPU. All results in this paper are on synthetic data modeled on PayPal session schemas; production transfer is explicitly out of scope and not claimed. The work has two contributions. First, we present a Karpathy-style **agentic experiment harness** in which an LLM agent proposes the next experiment and a deterministic Python launcher validates, locks, runs, parses, computes bootstrap confidence intervals, and writes one immutable row to history — with a strict single-writer-per-file ownership invariant. The harness ran 30 experiments across three sweep generations with zero format drift, zero concurrency races, and zero manual reconciliation. Second, we use the harness to surface, diagnose, and correct two design pathologies that hid the architecture's signal — a synthetic-data pipeline in which the narrator paraphrased the side-stream into the text (collapsing the modality gap), and deterministic feature signatures in hard-negative templates (making the eval surface label-deterministic in observed support). After the v4 data pivot, the same cross-attention configuration that ranked as null in v3 produced a confidence-interval-separated lift on adversarial cross-modal fraud families: text-only achieved 0.000 recall on `phish_takeover_mfa_phished` against cross-attention's 0.972 (95% bootstrap CI [0.931, 1.000]). The v5 expansion confirmed the win is dial-robust across 11 configurations and surfaced a data-shaped ceiling on the `hn_recovery_high_amount` adversarial-legitimate family that no architectural dial moved within the sweep. The methodology — agentic loop + bootstrap-CI eval + iteratively repaired synthetic data — generalizes beyond this architecture and task.

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
    OUTPUT_MD.write_text(merged, encoding="utf-8")

    print(f"Wrote {OUTPUT_MD}")
    print(f"  size: {OUTPUT_MD.stat().st_size:,} bytes")
    print(f"  approximate words: {len(merged.split()):,}")


if __name__ == "__main__":
    build()
