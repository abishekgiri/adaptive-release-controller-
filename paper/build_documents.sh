#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pandoc adaptive-deployment-control.md --standalone --resource-path=. --variable geometry:margin=0.65in --variable fontsize=10pt --variable colorlinks=true -o adaptive-deployment-control.tex
python3 finish_tex.py adaptive-deployment-control.tex
pdflatex -interaction=nonstopmode -halt-on-error adaptive-deployment-control.tex > /tmp/arc-main-tex.log
pdflatex -interaction=nonstopmode -halt-on-error adaptive-deployment-control.tex > /tmp/arc-main-tex.log
pandoc adaptive-deployment-control.md --resource-path=. --reference-doc=word-reference.docx -o adaptive-deployment-control.docx
python3 finish_docx.py adaptive-deployment-control.docx
cd supplementary
pandoc supplementary.md --standalone --resource-path=.:.. --variable geometry:margin=0.85in --variable fontsize=10pt -o supplementary.tex
python3 ../finish_tex.py supplementary.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex > /tmp/arc-supp-tex.log
pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex > /tmp/arc-supp-tex.log
