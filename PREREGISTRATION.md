# P1 — Predictive-validity shootout for the METR AI time-horizon metric: PREREGISTRATION

**Locked:** 2026-10-11 23:06 CDT (America/Chicago).
**Status:** Analysis plan locked BEFORE any model fitting or comparison. Only descriptive
data verification (counts, fields, weight checks) was performed prior to locking.
No success-curve model has been fit; no comparison metric has been computed.

## 1. Research question

METR's 50% time horizon is widely quoted as *the* AI capability number. Nguyen & Fithian
(arXiv:2610.12466, submitted 2026-10-08) improved how the number is *estimated* (splines +
IRT, relaxing log-linearity). Nobody has tested whether the number is actually *useful*:
does a model's spline-estimated 50% time horizon predict held-out task success better than
simpler summaries of the same data?

## 2. Hypotheses

- **H1:** The spline-based horizon predictor beats a per-model mean-accuracy predictor on
  held-out task outcomes (cross-validated log-loss).
- **H2:** The spline-based horizon predictor beats an IRT-ability-based predictor on
  held-out task outcomes (cross-validated log-loss).

Both are directional superiority claims for the horizon predictor. A failure to beat
either baseline is reported as a negative result, not spun.

## 3. Data (verified 2026-10-11)

- Source: `https://github.com/METR/eval-analysis-public` (public repo),
  file `reports/time-horizon-1-1/data/raw/runs.jsonl`,
  commit `52cb829c7a2efb2d659285c4b1768d191d97f8d2`.
- Contents (verified by direct inspection): 24,008 runs; 228 tasks; 79 task families;
  21 aliases = 20 AIs + 1 human baseline; task suites HCAST (157 tasks), SWAA (66),
  RE-Bench (5). Fields per run include `task_id`, `task_family`, `alias`,
  `score_binarized` (0/1), `human_minutes` (>0 for all 24,008 runs), and
  `invsqrt_task_weight` (METR's sqrt-family weight; verified to sum to 1.0 per alias).
- Filters: the `human` alias (773 runs) is EXCLUDED everywhere (we predict AI success).
  Analysis set: 23,235 AI runs, 20 AIs, 228 tasks.
- Note: Nguyen & Fithian report "228 tasks and 26 AIs"; the public release contains
  20 AIs. The 6-model discrepancy is noted as a limitation (their fits may use
  non-public runs).

## 4. Predictors (all produce per-(task, AI) success probabilities on held-out tasks)

Let T_i = human_minutes of task i, x_i = log2(T_i), Y_ijr in {0,1} the r-th run of
AI j on task i, w_ijr = invsqrt_task_weight (per-run, sums to 1 per AI).

- **P1 — horizon/spline** (Nguyen & Fithian Model 1):
  logit p_j(t) = alpha_j − beta · f(log2 t), beta > 0,
  f a monotone nondecreasing cubic spline (PCHIP through strictly increasing knot
  values) with 4 interior knots at the 20/40/60/80% quantiles of x on the training
  data; identifiability f(min x) = min x, f(max x) = max x (knot values pinned at the
  endpoints, interior values strictly increasing via exp-gap parameterization).
  Fit per CV fold by minimizing weighted binary cross-entropy
  −Σ w_ijr [Y_ijr log p + (1−Y_ijr) log(1−p)] via L-BFGS-B.
  The 50% horizon per AI is th_0.5(AI_j) = 2^{f^{−1}(alpha_j/beta)} (numerical inverse;
  reported descriptively).
- **P2 — mean accuracy**: p_hat = weighted mean of Y over AI j's training runs
  (weights w_ijr). Constant across held-out tasks.
- **P3 — IRT ability** (1PL Rasch + explanatory regression):
  Fit logit p_ij = alpha_j − beta·theta_i on training tasks (theta_i per task,
  identifiability via centering theta inside the objective; beta > 0; weighted MLE).
  Then OLS: theta_i = a + b·x_i on training tasks.
  For held-out task i: theta_hat_i = a + b·x_i; p_hat_ij = sigmoid(alpha_j − beta·theta_hat_i).
  NOTE (preregistered expectation): this reduces approximately to a shared-slope
  logistic in log2(T); i.e., IRT ability is expected to carry no more predictive
  information than the linear baseline for unseen tasks. The comparison P1 vs P3
  therefore tests the spline nonlinearity specifically.

All probabilities clipped to [1e-12, 1−1e-12] for scoring.

## 5. Cross-validation scheme

- 5-fold CV over the 79 task families (the independent unit; matches Nguyen & Fithian
  §2.4.2). Families randomly assigned to 5 folds (~16 families each), seed 20261012.
  All runs of a task (hence family) go to a single fold. No stratification beyond this.
- Each predictor is refit from scratch on each fold's training tasks; predictions are
  made only on held-out tasks.

## 6. Scoring rules and decision criteria

Per fold, per predictor, on held-out runs:
- **Primary: weighted log-loss** (marginal log score, paper §2.4):
  −Σ w_ijr [Y log p_hat + (1−Y) log(1−p_hat)] / Σ w_ijr. Lower is better.
- **Secondary: weighted Brier score** Σ w_ijr (Y − p_hat)^2 / Σ w_ijr; and
  **micro-AUC** over held-out runs (Mann–Whitney rank statistic, unweighted).

**Decision criterion (locked):** P1 "wins" iff, on the primary metric, its mean paired
fold-difference vs P2 is negative AND vs P3 is negative, with |mean diff| > 2·SE in
both cases (SE = sd of the 5 paired fold-differences / √5). H1 requires the P1-vs-P2
win; H2 requires the P1-vs-P3 win. Secondary metrics are reported with the same
paired comparison but are descriptive (no win/loss declared on them).

## 7. Descriptive checks (not scored, reported for transparency)

- Per-AI 50% horizons from a P1 fit on all data, vs METR's published
  `benchmark_results_1_1.yaml` values (sanity check of implementation).
- Fitted f shape (visual check of the 2–30 min flat region reported by the paper).
- Monotonicity of fitted f on a fine grid (asserted in code).

## 8. Limitations (acknowledged upfront)

- Observational data; tasks are a fixed suite, not a random sample of "real work".
- 20 public AIs are a convenience sample (paper used 26).
- 29% of human times are expert estimates, not measurements (per the paper §2.2).
- Runs within a (task, AI) pair are non-independent; fold-level paired SEs are the
  unit of inference.
- Scoop/positioning note: Nguyen & Fithian already CV-compared spline vs logistic;
  this shootout's novel contribution is the mean-accuracy baseline (P2) and the
  predictive-validity framing of the horizon *metric* (not just the estimator).
  Their paper is prior art and will be cited, not duplicated.

## 9. Deviation log

(none yet; any deviation from this plan will be recorded here with date and reason)
