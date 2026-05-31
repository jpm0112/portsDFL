# meetings

Self-contained LaTeX figures for committee meetings on the third paper
of the dissertation: *Decision-Focused Learning for Berth Allocation*.

The Bayesian sub-project is intentionally not covered here; figures
focus on the DFL pipeline (`prediction_models/`).

## Build

```powershell
pdflatex dfl_explainer.tex
pdflatex dfl_explainer.tex   # second pass for cross-references
```

Output: `dfl_explainer.pdf` (9 pages, landscape): title, "30-second
story", six figures (one combined PtO+DFL+evaluation figure as the
centerpiece), notation cheat-sheet.

## Lifting figures into Beamer

Each `tikzpicture` block is self-contained; copy it into a Beamer frame
together with the preamble's `\usetikzlibrary{...}`, color definitions,
and `\tikzset{...}` block.

## Updating Figure 8 with real numbers

The bar chart and CDF use placeholder coordinates. Replace them with
values from:

- `prediction_models/results/dfl_real_bap/decision_summary.csv`
  (aggregate regret per model).
- `prediction_models/results/dfl_real_bap/pto_per_instance.csv` and
  `dfl_per_instance.csv` (per-instance regret, for the CDF).
