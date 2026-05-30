# Formulation docs

Self-contained write-up of the week-long Berth Allocation MILP and how it sits
in the literature.

## Contents
- **`bap_formulation.tex`** — the extended MILP: sets/parameters/variables,
  objective, all constraints (compatible-berth assignment, arrival release,
  hard/soft no-wait service windows, sequencing, big-M precedence), the
  single-week solve scope, and the decision-focused-learning integration.
- **`related_formulations.tex`** — `\input` by the main doc: the "Related
  formulations" section (prose), a comparison table against the closest
  published BAP models, and a novelty statement (including what is *not* novel).
- **`references.bib`** — bibliography (Imai 2001, Cordeau 2005, Golias 2007,
  Alzaabi & Diabat 2016, Ursavas 2022, Chu 2025/2026, Pu 2025, Pogančić 2020,
  Elmachtoub & Grigas 2022).
- **`decisions_and_questions.md`** — running register of non-obvious modeling
  decisions and open questions, each tagged with its PtO/DFL implication.

## Build the PDF
```
cd prediction_models/docs/formulation
latexmk -pdf bap_formulation.tex      # preferred (runs bibtex automatically)
# or, manually:
pdflatex bap_formulation && bibtex bap_formulation && pdflatex bap_formulation && pdflatex bap_formulation
```

## Notes
- The literature comparison was assembled from a multi-source review; a few
  primary papers were paywalled, so verify exact equation symbols / bibliographic
  details before external publication (see the caveats in the project notes).
- The notation matches `optimizers/src/bap_optim/discrete_bap.py` and the style of
  `meetings/latex/dfl_explainer.tex`.
