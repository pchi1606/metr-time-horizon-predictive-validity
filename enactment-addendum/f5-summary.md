# F-5 Partition-Only Sensitivity — Summary (P1, ENACTMENT-2026-11-20)

**Status: COMPLETE — run locally on this VM. 20 / 20 partitions.**

## Headline

- **H2 (P1 − P3 log-loss difference), partition-only range across 20 partitions: −0.487 to −0.363** (min to max of the per-partition means). Mean −0.417, SD 0.037, median −0.419, IQR [−0.442, −0.386].
- **H1 (P1 − mean-accuracy / P2 log-loss difference), partition-only range: −0.271 to −0.236.** Mean −0.253, SD 0.009, median −0.251, IQR [−0.257, −0.249].
- **Mechanical rule for H2: in 20 of 20 partitions (fraction 1.00), |per-partition mean| > 2 × per-partition SE** (SE = SD of the 5 fold differences / √5, the locked r4_partitions.py convention). The same holds for H1 (20/20).

## Distribution across the 20 partitions (per-partition means)

| Statistic | H1 | H2 |
|---|---|---|
| n partitions | 20 | 20 |
| Mean | −0.2528 | −0.4175 |
| SD | 0.0090 | 0.0366 |
| Min | −0.2707 | −0.4870 |
| Q1 | −0.2566 | −0.4418 |
| Median | −0.2513 | −0.4187 |
| Q3 | −0.2488 | −0.3859 |
| Max | −0.2363 | −0.3634 |
| Fraction \|mean\| > 2·SE | 1.00 (20/20) | 1.00 (20/20) |

Context: the locked partition (seed 20261012) and the three Table 27 alternative partitions gave H1 ≈ −0.246 to −0.272 and H2 ≈ −0.390 to −0.428; the 20 new partitions are consistent with both (H1 range overlaps almost exactly; H2 extends slightly wider, to −0.487 / −0.363).

## Method (exact extension of the locked procedure — no new estimator)

- Procedure is `repro/code/r4_partitions.py` (the script that produced Table 27's three alternative partitions), generalized from 3 seeds to 20 new seeds. Identical predictors (`SplineHorizon` = P1, `MeanPredictor` = P2/mean-accuracy, `IrtPredictor` = P3), identical fitting, identical scoring (`run_cv.scores_fold`: sqrt-family weighted held-out log-loss, weights `w = invsqrt_task_weight`).
- Partitions: `data.family_folds` — repeated random 5-fold partitions of all 79 task families. Each fold holds out 15–16 families, so every fold trains on **63–64 families** (verified in the CSV for all 100 folds: `n_train_families` ∈ {63, 64} throughout). This holds training-family count fixed at the locked level, unlike the bootstrap (mean 40.1 distinct training families).
- Per partition, H1/H2 are the mean of the 5 fold-level log-loss differences; per-partition SE = SD(fold differences, ddof=1)/√5 — exactly as in `r4_partitions.py`.
- Master seed **20261120**; partition seeds drawn deterministically as `np.random.default_rng(20261120).integers(0, 2**31−1)` in order, excluding the locked seed (20261012) and the three Table 27 seeds (20261013, 20261014, 42); no collision occurred. Seeds used, partitions 0–19: 1602163775, 1582770458, 993177791, 321768753, 889521776, 1789774899, 167864159, 1267217589, 1606862739, 1402071348, 997418360, 883869627, 316282813, 1986676874, 1120888242, 857164448, 711780634, 111960120, 1474962450, 980215524.
- Runner: `f5_run_partitions.py` (this directory); per-partition, per-fold results checkpointed to `f5-partitions.csv` after each partition (100 rows = 20 partitions × 5 folds). Stored per-partition means/SEs were re-verified against the fold differences after the run (exact to 1e-12).

## Per-partition results

| # | Seed | H1 mean | H1 SE | H2 mean | H2 SE | \|H2\| > 2·SE |
|---|---|---|---|---|---|---|
| 0 | 1602163775 | −0.2544 | 0.0096 | −0.4259 | 0.0743 | yes |
| 1 | 1582770458 | −0.2442 | 0.0395 | −0.4381 | 0.1135 | yes |
| 2 | 993177791 | −0.2700 | 0.0303 | −0.3780 | 0.0548 | yes |
| 3 | 321768753 | −0.2484 | 0.0381 | −0.4526 | 0.1144 | yes |
| 4 | 889521776 | −0.2386 | 0.0370 | −0.4358 | 0.1059 | yes |
| 5 | 1789774899 | −0.2363 | 0.0602 | −0.4870 | 0.1927 | yes |
| 6 | 167864159 | −0.2707 | 0.0485 | −0.4750 | 0.1211 | yes |
| 7 | 1267217589 | −0.2506 | 0.0396 | −0.4115 | 0.0772 | yes |
| 8 | 1606862739 | −0.2541 | 0.0420 | −0.3820 | 0.0809 | yes |
| 9 | 1402071348 | −0.2599 | 0.0365 | −0.3861 | 0.0375 | yes |
| 10 | 997418360 | −0.2569 | 0.0353 | −0.4340 | 0.0979 | yes |
| 11 | 883869627 | −0.2438 | 0.0463 | −0.4336 | 0.0886 | yes |
| 12 | 316282813 | −0.2565 | 0.0243 | −0.4540 | 0.0680 | yes |
| 13 | 1986676874 | −0.2500 | 0.0353 | −0.4076 | 0.0692 | yes |
| 14 | 1120888242 | −0.2636 | 0.0403 | −0.3684 | 0.1005 | yes |
| 15 | 857164448 | −0.2519 | 0.0291 | −0.4550 | 0.1391 | yes |
| 16 | 711780634 | −0.2505 | 0.0208 | −0.3896 | 0.0810 | yes |
| 17 | 111960120 | −0.2496 | 0.0330 | −0.3869 | 0.0706 | yes |
| 18 | 1474962450 | −0.2490 | 0.0526 | −0.3854 | 0.1121 | yes |
| 19 | 980215524 | −0.2565 | 0.0487 | −0.3634 | 0.1015 | yes |

## Cost

- Timing measurement first (required by the task): one partition end-to-end = **70.5 s** on this VM → projected 20 partitions ≈ 24 min, far below the ~10 h stop threshold, so the run proceeded locally; **no MacBook package was needed**.
- Actual: sum of per-partition compute = 1,446 s ≈ **24.1 min wall** total (partition 0 in the timing run; partitions 1–19 in one resumed run, 1,376 s). Price ceiling was 15 compute-hours — used ≈ 0.4 h (≈3%).
- Note: `f5-run.log` lines from the main run appear twice (the runner logs both to stdout, which was redirected to the same file, and to the file directly). The CSV is unaffected and is the authoritative record.

## Files

- `f5-partitions.csv` — per-partition, per-fold results (100 rows).
- `f5-summary.md` — this file.
- `f5-run.log` — run log.
- `f5_run_partitions.py` — the runner (resumable; skips partitions already in the CSV).
