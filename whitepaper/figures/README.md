# Figures

Two audiences, two figure variants where the dense version is too detailed for a slide.

| File | Audience | Role |
|---|---|---|
| `fig1-architecture.svg` | Whitepaper (technical) | Full architecture — Qwen3-8B + side encoder + Perceiver-Resampler + gated x-attn, with insertion-pattern detail, layer indices, parameter inventory |
| `fig1b-architecture-simple.svg` | Leadership / slides | 5-stage simplified flow — Text path → Frozen LM → Gated cross-attention → Score, with side-stream feeding the fusion block. No insertion-pattern detail, ≥18 pt fonts |
| `fig2-auto-research-loop.svg` | Both | Loop dataflow + ownership invariant. Already slide-friendly; no separate variant needed |
| `fig3-data-distribution.svg` | Whitepaper (technical) | Full distribution — class balance, journey × actor heatmap, eval-mode dropout, token families |
| `fig3b-data-distribution-simple.svg` | Leadership / slides | Three-card causal story — `clean` (both solve) / `phish_takeover_mfa_phished` (x-attn wins, CI-separated) / `hn_recovery_high_amount` (both fail, v5 ceiling) |
| `fig4-sweep-results.svg` | Both | v5 sweep leaderboard with broken-axis treatment for the catastrophic regression. Already slide-friendly |

## Embedding in markdown

The whitepaper markdown files (`00`–`04`) embed only the dense / both-audience variants:

```markdown
![Figure 1. Cross-attention surgery on Qwen3-8B](figures/fig1-architecture.svg)
![Figure 3. Data distribution & eval-mode mix](figures/fig3-data-distribution.svg)
```

The simple variants (`fig1b`, `fig3b`) are not referenced from the whitepaper. They live here for the leadership readout PPTX and any future slide decks.

## PNG renders

PNGs of every SVG live alongside the SVGs (`fig*.png`). They are regenerated whenever the SVG changes. Use cases:

- **PPTX inserts** — PowerPoint and Keynote handle PNG more predictably than SVG (especially for cross-platform decks).
- **Documents in viewers without SVG support** — older Markdown previewers, some PDF generators, certain wikis.
- **Quick visual diffs** — easier to spot regressions in a binary image than a text-diff of XML.

The SVGs are the source of truth. Re-render PNGs with:

```bash
cd whitepaper/figures
for f in *.svg; do soffice --headless --convert-to png "$f"; done
```

## Naming convention

| Pattern | Meaning |
|---|---|
| `figN-<topic>.svg` | Dense technical figure for the whitepaper |
| `figNb-<topic>-simple.svg` | Simplified variant of the same figure for slide consumption |
| `figN-<topic>.png` | PNG render of the corresponding SVG (regenerated on SVG change) |

If you add a third variant (e.g., dark-mode for a different deck theme), use a third suffix: `fig1c-architecture-dark.svg`.
