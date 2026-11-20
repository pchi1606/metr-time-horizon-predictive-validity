# RUN RECORD — REG-P1-5 family-cluster bootstrap (canonical outputs)

**What:** B = 300 family-cluster bootstrap refitting the complete P1
pipeline per resample (driver: `repro/code/r22_family_refit_bootstrap.py`;
design note and estimand: `panel/round22-e70a5eec/INTEGRATION-MEMO-R22.md`,
Task C). Per resample: 79 families drawn with replacement, the 5-fold
partition re-derived inside the resample by the pipeline's own algorithm
(all copies of a family in one fold), P1/P2/P3 refit on every fold, the
five paired differences and the within-resample locked-rule verdict
recorded.

**Run:** Kaggle notebook `aipeppers/notebook36efd20a81` (dataset
`aipeppers/p1-r22-family-refit-bootstrap`), launched 2026-11-13 13:11 CST,
completed 2026-11-14 ~00:40 CST. Outputs harvested from the notebook's
output directory and downloaded in full; the harvest was staged at
`repro/output/bootstrap_r22_extract/bootstrap_r22/` and copied here
verbatim.

**Integrity:** 301 files (summary.json + resample_0000.json …
resample_0299.json) — 300/300 resample files present. md5 checksums of
the copy verified identical to the harvest for every file (sorted-hash
diff empty, 2026-11-14).

**Outcome:** 294 resamples completed ("ok"); 6 failed — indices 9, 193,
244, 283, 292, 293 — all with the identical recorded error
`ValueError: `y` must contain only finite values.` (degenerate family
draws). Failures are recorded in their resample files and in
summary.json; no statistic silently drops them (all summary statistics
are over the 294 completed resamples, as stated in the manuscript's
Appendix G).

**Headline results (summary.json):**
- H1 (P1−P2): bootstrap SE 0.0436, mean −0.2472, median −0.2486,
  95% percentile CI [−0.3272, −0.1546], frac resample means < 0 = 1.0,
  locked-rule pass rate 0.9932; locked value −0.2382
  (locked − bootstrap median = +0.0104).
- H2 (P1−P3): bootstrap SE 0.0929, mean −0.1972, median −0.1814,
  95% percentile CI [−0.4136, −0.0598], frac resample means < 0 = 1.0,
  locked-rule pass rate 0.8197; locked value −0.4182
  (locked − bootstrap median = −0.2368).
