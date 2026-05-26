#!/usr/bin/env bash
# Build the Executive Edition PDF from executive-edition.md.
#
# Prerequisites (same as build_pdf.sh):
#   - pandoc  (>= 2.9)
#   - xelatex (from texlive-xetex)
#   - DejaVu Serif + DejaVu Sans Mono fonts
#
# Usage (from this folder):
#   ./build_executive_pdf.sh
#
# Output: executive-edition.pdf in this folder.

set -euo pipefail

cd "$(dirname "$0")"

INPUT="executive-edition.md"
OUTPUT="executive-edition.pdf"

if [[ ! -f "$INPUT" ]]; then
  echo "Missing $INPUT." >&2
  exit 1
fi

# Same SVG -> PNG path-substitute as build_pdf.sh, and replace ★ with [winner]
# in case any leadership-style markers crept in.
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
  -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono"

echo "Wrote $OUTPUT  ($(du -h "$OUTPUT" | cut -f1))"
