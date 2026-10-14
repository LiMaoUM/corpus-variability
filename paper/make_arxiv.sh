#!/usr/bin/env bash
# Build the signed arXiv source package: paper/arxiv-how-many-documents.tar.gz
# (arxiv.tex defines \arxivversion; arXiv does not run BibTeX, so the .bbl is included).
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode arxiv.tex >/dev/null
T=$(mktemp -d)
mkdir -p "$T/sections" "$T/figures"
cp arxiv.tex main.tex acl.sty acl_natbib.bst arxiv.bbl "$T/"
cp sections/*.tex "$T/sections/"
cp figures/*.pdf "$T/figures/"
(cd "$T" && pdflatex -interaction=nonstopmode arxiv.tex >/dev/null && pdflatex -interaction=nonstopmode arxiv.tex >/dev/null && rm -f arxiv.pdf arxiv.log arxiv.aux arxiv.out)
tar czf arxiv-how-many-documents.tar.gz -C "$T" .
rm -rf "$T"
echo "wrote paper/arxiv-how-many-documents.tar.gz"
