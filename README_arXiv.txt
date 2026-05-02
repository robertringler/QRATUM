Title: The Anti-holographic Entangled Universe
Author: Robert Ringler (Independent Researcher)
Date: October 20, 2025

Primary category: gr-qc (General Relativity and Quantum Cosmology)
Cross-lists: hep-th (High Energy Physics - Theory), quant-ph (Quantum Physics)

Files included in this package:
- main_revtex_new.pdf       (PRX-style PDF of manuscript, generated from RevTeX wrapper)
- main_revtex_new.tex       (RevTeX wrapper - clean submission file)
- main.tex                  (original arXiv-style article-class source for reference)
- references.bib            (BibTeX database)
- figures/                 (folder with figures; includes figure_phase_diagram.tex and fig_longrange_entropy.pdf)
- simulations/             (reproducibility artifacts: smoke_test.py and smoke_test_output.png)
- README_arXiv.txt         (this file)
- ONE_PAGE_SUMMARY.pdf     (one-page technical summary)

Build notes:
- The PDF `main_revtex_new.pdf` was produced from `main_revtex_new.tex` with pdflatex + bibtex + pdflatex x2.
- If re-compilation is required, run from the `manuscript/` directory:

  pdflatex -halt-on-error -interaction=nonstopmode main_revtex_new.tex
  bibtex main_revtex_new
  pdflatex -halt-on-error -interaction=nonstopmode main_revtex_new.tex
  pdflatex -halt-on-error -interaction=nonstopmode main_revtex_new.tex

- If you prefer the original arXiv article-class source, `main.tex` uses biblatex; to build it use a biblatex workflow (or convert to natbib). The RevTeX wrapper is the recommended upload for PRX-style formatting.

Contact:
Robert Ringler
Independent Researcher
Phone: +1 (561) 539-9324
Email: complexnumbers11@gmail.com
