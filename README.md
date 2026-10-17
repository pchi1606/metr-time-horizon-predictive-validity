# Reproduction package: Does the AI Time Horizon Predict Anything?

One-command reproduction of the preregistered predictive-validity shootout for
the METR AI time-horizon metric (TMLR submission).

## Quick start

```bash
bash reproduce.sh
```

This runs the full pipeline (5-fold cross-validated shootout, descriptives,
strengthening analyses, figures) and then verifies every headline number in
the paper against the recomputed outputs. Expected runtime: ~25 minutes on a
modest CPU (2 cores). Requires Python 3 with `numpy`, `scipy`, `pandas`,
`matplotlib`, `pyyaml`.

## What it does

1. `code/run_cv.py` — 5-fold CV over task families: P1 (spline horizon),
   P2 (mean accuracy), P3 (IRT ability). Saves `output/cv_scores.json`.
2. `code/descriptives.py` — full-data P1 fit, per-AI 50% horizons, fitted
   time-to-difficulty curve. Saves `output/descriptives.json`.
3. `code/strengthen.py` — post-hoc analyses: P0 logistic baseline,
   calibration/ECE, equal-task-weight sensitivity, per-AI gains, horizon
   fold-spreads. Saves `output/strengthen.json`.
3b. `code/p3_diagnostics.py` — per-fold P3 (IRT) fit diagnostics for the
   miscalibration mechanism (Appendix D). Saves `output/p3_diagnostics.json`.
4. `code/figures.py` — regenerates all paper figures from the outputs.
5. `verify.py` — recomputes every headline number from the outputs and fails
   loudly on any mismatch beyond tolerance.

## Data

`data/runs.jsonl` — METR Time Horizon 1.1 public run data (23,235 AI runs
after excluding the human baseline; 228 tasks; 79 families; 20 AIs), bundled
from `https://github.com/METR/eval-analysis-public`
(`reports/time-horizon-1-1/data/raw/runs.jsonl`, commit `52cb829c`).
`data/benchmark_results_1_1.yaml` — METR's published horizon estimates
(`https://metr.org/assets/benchmark_results_1_1.yaml`), used for the
descriptive comparison in Figure 4.

## Layout

- `reproduce.sh` — one-command pipeline + verifier
- `verify.py` — headline-number verifier
- `code/` — analysis code (PEP 8 comments)
- `data/` — bundled public data
- `output/` — regenerated analysis outputs (created by the pipeline)
- `figs/` — regenerated figures (created by the pipeline)
- `paper.pdf` — the manuscript
- `PREREGISTRATION.md` — the locked analysis plan (see project root)

## Author

Prabhu Rangaraju. See `CITATION.md`.
