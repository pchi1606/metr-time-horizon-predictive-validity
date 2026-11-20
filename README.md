# Reproduction package: Duration Conditioning Predicts Held-Out-Family Success for Already-Observed Models on METR Time Horizon 1.1 — A Shootout Against Mean Accuracy

One-command reproduction of the preregistered predictive-validity shootout for
the METR AI time-horizon metric (SN Computer Science submission).

Package version: v1.0.0, archived 2026-11-20 with the SN Computer Science
submission manuscript build 82c2f8fe (2026-11-20). The bundled METR run
data is not redistributed in this archive — see data/DATA-README.md. The verifier (verify.py) covers the
locked pipeline and every revision add-on's printed numbers, including the
regenerated alternative-partition values and the round-11 stage-1 penalty
sensitivity (Appendix D); the superseded historical values
discussed in the manuscript's Limitations are not retained and are not
covered. A companion verifier (verify_printed.py) checks the numbers as
printed in the compiled PDF, and code/p3_convergence.py demonstrates the
P3 fits' convergence to a printed tolerance (Appendix D).

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
6. `verify_printed.py` — extracts the compiled PDF's text layer and checks
   the printed values themselves (transcription errors on the page).
   `code/p3_convergence.py` — per-fold P3 convergence demonstration
   (analytic gradient at the locked solutions; tightened warm-started
   refits), saving `output/p3_convergence.json`. It reads
   `output/cv_scores.json` (produced by step 1) as its anchor and
   refits stage 1 per fold; expect a few minutes on top of the
   pipeline's ~25 minutes.

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
