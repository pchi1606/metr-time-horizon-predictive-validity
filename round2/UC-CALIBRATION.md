# U-C simulation calibration — design, numbers, outcome

**Unit:** U-C / U-C-R2 (MAJOR). **Authority:** the user's ruling 2026-11-20
(calibration branch; ceilings ≤20 author-hours / ≤10 compute-hours) and the
amended outcome rule adopted in pass 2 (skeptic's required repair; see
ROUND2-RECORD.md). **Outcome: branch (c) — full demotion** of t-statistics,
t-CIs and critical-value comparisons at every mirror site (enacted in the
round-2 rebuild; the calibration itself is reported in a new appendix).

## Design

Grouped simulation mimicking the paper's own design, executed by
`uc_calibration.py` (sha256 bebf3f591d5fa9346dc7b5638d7c1dbb1396a55e296a36
c3822406418cc9da42):

- 79 families at their observed sizes, shipped sqrt-family weights, 23,235
  runs, 228 tasks, 20 AIs (the deposited run set, unmodified).
- Truths fit once on the full real data: **T1** (alternative) = the
  full-data P1 (SplineHorizon) fitted probabilities; **T0** (null) =
  per-AI weighted mean success (no duration effect).
- Random effects on the logit scale: task effects u_i ~ N(0, σ²_task) and
  family effects v_f ~ N(0, σ²_fam), variances estimated by delta-method
  moment matching on the T1 fit's residuals (group score residuals
  r_g = Σ(y−p)/Σ p(1−p), sampling variance 1/Σ p(1−p); task and family
  moments solved jointly, task nested in family). Estimates:
  **σ_task = 1.616 (variance 2.612), σ_fam = 0.0** (family-mean residual
  variation is fully accounted for by the constituent task effects).
  Diagnostics: raw task-effect variance 2.612; 45% of tasks have empirical
  success rates ≤0.05 or ≥0.95 — the heterogeneity is real, not a fitting
  artefact.
- Each replication: fresh u*, v*; y* ~ Bernoulli(sigmoid(η_truth + u*_i +
  v*_f)); a fresh 5-fold family partition; the exact locked pipeline per
  fold (P1 SplineHorizon, P2 mean, P3 1PL+OLS, P0 shared-slope logistic,
  Pmetr per-AI logistic); the five paired fold log-loss differences for
  H1 (P1−P2), H2 (P1−P3), D10 (P1−P0) and MP (Pmetr−P1); fold mean m_b and
  fold SE s_b = sd(diffs, ddof=1)/√5.
- B = 200 (T1) + 100 (T0); master seed 20261121 (per-replication seeds from
  SeedSequence([20261121, truth_bit, b])). One T1 replication (b=28)
  failed inside the spline fit on simulated data ("`y` must contain only
  finite values") and was recorded and skipped: **B = 199 (T1) + 100 (T0)**
  enter the summary. No criterion is close enough to a boundary for one
  replication to matter.
- P1−P3b is calibrated by the H2 series: P3b is the λ=0 endpoint of the
  P3r grid, whose construction (stage-1 IRT fit + OLS projection) is
  exactly P3's; identity verified on the locked partition (bracket λ=0
  outer mean −0.4182 vs locked H2 mean −0.418).
- P3's stage-1 fit inside the replications uses the analytic-gradient
  drop-in `fast_ridge.py` (sha256 e4ab019fe1110955269abf730d04210300ccf914
  5feacd165afabbd750a99be6), with fallback to the deposited
  finite-difference implementation where the exact gradient trips the
  line search. Validation (`validate_fast_ridge.py`, sha256
  35519e4e7b8407df185c7ede39a04cd581b25b34613c8d27c4ecd94a226f7cbb;
  results `fast-ridge-validation.json`, sha256 357de84120440ef81e4590e9d9
  f48bbe45ca0574ec9759f6698ae2cab08c83c5): on the locked partition the
  deposited side reproduces the deposited bracket record exactly
  (Δ = 0.00 on all five folds); the fast side selects the same λ
  (0.0003) on all five folds with outer log-loss differences ≤ 5.5e-5.
  Inner-mean differences are ≤ 0.0074 in the selection region
  (λ ∈ {1e-4, 3e-4, 1e-3}) and large only at λ = 0 — the unpenalised
  endpoint whose fit has no finite maximiser (the paper's own H2
  pathology), where no implementation-independent value exists. The H2
  series is therefore a calibration of the procedure, reported with this
  caveat; it gates nothing (the branch criteria run on H1 and P1−P0,
  neither of which involves the ridge fit).
- Plug-in caveat (printed in the appendix, per the amended rule): the
  calibration is local to this design and conditional on the fitted
  truths; under T1 the simulated H2 level (−0.016 mean) does not
  reproduce the real data's H2 level (−0.418), because a P1-generated
  truth does not reproduce the real baseline's termination pathology.
  The calibration measures the fold-difference SE's behaviour, not H2's
  level.

## Numbers (uc_summarize.py, sha256 39dde3af9688209397ea16a06c6814bbe83
928ca1387e38375f80385e9351de3; uc-summary.json, sha256 a4a956899e691f192
1eefa1fba08a2a73edd16f867bcf342bce9bd8747f01dc3)

ĉ = sd_b(m_b) / mean_b(s_b) under T1; MC CI = bootstrap percentile 95%
over replications (4,000 draws, seed 20261122); coverage = share of
replications with |m_b − mean(m)| ≤ 2.776 · f · s_b.

| comparison | ĉ | MC 95% CI | coverage at ×1.5 | coverage at ĉ |
|---|---|---|---|---|
| H1 (P1−P2) | 0.571 | [0.508, 0.635] | 0.995 | 0.955 |
| P1−P0 | 1.253 | [1.118, 1.394] | 0.955 | 0.955 |
| H2 (= P1−P3b) | 1.032 | [0.924, 1.141] | 0.985 | 0.950 |
| Pmetr−P1 | 1.253 | [1.114, 1.393] | 0.960 | 0.955 |

T0 (null): H1 two-test-bar false-clearance rate at the ×1.5-corrected bar
= 0.000 (MC SE 0.000; nominal 0.025 — passes). H1 ĉ under T0 = 1.048
[0.894, 1.191]: the factor the H1 fold SE needs depends on the truth
(0.57 under T1, 1.05 under T0), a further reason no single constant
calibrates it.

## Branch determination (amended rule)

- Branch (a) — ×1.5 stands: **fails for both graded comparisons.** H1:
  MC CI [0.508, 0.635] lies entirely below [1.35, 1.65], and coverage at
  ×1.5 (0.995) is outside [0.90, 0.99]. P1−P0: MC CI [1.118, 1.394]
  straddles the 1.35 band edge (amended rule (ii): a straddle is branch
  (c) for the affected comparison).
- Branch (b) — one corrected factor for both: **fails.** |ĉ_H1 − ĉ_D10|
  = 0.68 > 0.3; no common factor lies in both MC CIs (c* = 0.91 is in
  neither).
- **Branch (c) is therefore enacted**: the skeptic's demotion text at
  every mirror site in S3's inventory — t-statistics, t-CIs and
  critical-value comparisons deleted (Table 2's critical-value rows
  deleted, not relabelled); the five paired differences reported
  descriptively (range and median); §4.1's yardstick paragraph replaced
  by the skeptic's paste-ready text (its "Table 14" reference adapted to
  Table 15, the per-fold table's number in the current build — recorded
  here as the sole adaptation). The simulation and this outcome are
  reported in a new appendix, so the demotion's ground is itself a
  measured, dated record.

Compute used: ≈1.2 compute-hours wall on this VM across restarts
(300 replications, checkpointed in `uc-reps/`), inside the ≤10
compute-hour ceiling; author time inside the ≤20 author-hour ceiling.
