#!/usr/bin/env bash
# Build the comprehensive whitepaper PDF from whitepaper-comprehensive.md.
#
# Prerequisites:
#   - pandoc  (>= 2.9)
#   - xelatex (from texlive-xetex; tested on TeX Live 2022)
#   - DejaVu Serif + DejaVu Sans Mono fonts (Linux: fonts-dejavu;
#     macOS: brew install --cask font-dejavu)
#
# Usage (from this folder):
#   ./build_pdf.sh
#
# Output: whitepaper-comprehensive.pdf in this folder.

set -euo pipefail

cd "$(dirname "$0")"

INPUT="whitepaper-comprehensive.md"
OUTPUT="whitepaper-comprehensive.pdf"

if [[ ! -f "$INPUT" ]]; then
  echo "Missing $INPUT. First run merge_whitepaper.py to generate it." >&2
  exit 1
fi

# Path-substitute the figures from .svg to .png for the LaTeX backend
# (xelatex via pandoc handles PNG natively but does not handle SVG;
# the figures/ folder contains both .svg and .png side-by-side).
# Also substitute the ★ glyph for "[winner]" since the default
# DejaVu Serif font lacks it.
#
# Note: BSD mktemp (darwin) does not support GNU's --suffix flag, so we
# create a tempfile then rename it. Works on both Linux and macOS.
_TMP_RAW="$(mktemp)"
TMP_MD="${_TMP_RAW}.md"
mv "$_TMP_RAW" "$TMP_MD"
trap 'rm -f "$TMP_MD"' EXIT

sed -e 's|figures/\([a-z0-9-]*\)\.svg|figures/\1.png|g' \
    -e 's|★|[winner]|g' \
    "$INPUT" > "$TMP_MD"

pandoc "$TMP_MD" \
  -o "$OUTPUT" \
  --pdf-engine=xelatex \
  --toc \
  -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono"

echo "Wrote $OUTPUT  ($(du -h "$OUTPUT" | cut -f1))"
