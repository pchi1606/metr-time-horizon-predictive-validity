"""Round-22 REG-P1-5: family-level refit bootstrap for H1/H2 — the complete
pipeline refit under resampling of the task families.

=========================== DESIGN NOTE ===========================
(Also reproduced as a standalone section of the round-22 integration
memo. The justification is part of the deliverable.)

ESTIMAND. The locked headline numbers H1 = -0.238 and H2 = -0.418 are
means over five folds of per-fold paired log-loss differences. The
printed SEs (0.033 / 0.122) are the dispersion of those five dependent
fold scores; they are not sampling errors over any population. The
quantity this procedure estimates is the sampling distribution of the
*same statistic* — the 5-fold family-CV paired mean difference,
computed by the *same pipeline* — when the 79 task families are
treated as a sample from a population of task families. In plain
terms: "if the benchmark's task families had been a different draw of
79 families like these, how much would H1 and H2 move?" The 20 AIs and
the METR Time Horizon 1.1 release are held fixed throughout: the
procedure says nothing about sampling over AIs, over releases, or
over runs within a task. The bootstrap SE reported is the standard
deviation of the statistic across resamples; the interval is the
2.5%-97.5% percentile range across resamples.

RESAMPLING UNIT AND MULTIPLICITY. The resampling unit is the task
family (79 clusters), drawn with replacement, B times (default
B = 300). A family drawn c times contributes its runs with their
pipeline weights multiplied by c. This is exactly equivalent to
physically duplicating the family's rows: every fit in the pipeline
is a weighted BCE or a weighted mean keyed by task id, so duplicated
rows enter through a single shared task parameter with multiplied
run weights (stage 2 of P3 still sees each distinct task once, as in
the locked pipeline).

PARTITION. The 5-fold partition is RE-DERIVED inside each resample by
the pipeline's own algorithm (data.family_folds: shuffle the family
ids, round-robin into 5 folds), with a fresh per-resample seed. All
copies of a family land in the same fold — a task can never appear in
both train and test within a resample, so the grouped structure of
the locked design is preserved. Re-deriving (rather than freezing the
locked partition) is deliberate: the locked statistic is a functional
of the data AND the partition, the partition is part of the pipeline,
and freezing one arbitrary partition draw would understate the
dispersion (the alternative-partition refits of r4_partitions.py show
partition choice moving the SEs materially).

PROCEDURE PER RESAMPLE. Re-derive the partition; for each fold, refit
P1 (spline), P2 (mean), P3 (1PL + OLS) — and P0 (shared-slope
logistic) if --include-p0 — on the training folds with multiplicity-
weighted runs; score the held-out fold with multiplicity-weighted
runs (run_cv.scores_fold); record the five per-fold paired
differences and their mean (the resample's H1_b, H2_b). Also record
the within-resample locked-rule verdict (|mean| > 2 * sd/sqrt(5)
over the five fold differences), so the summary can report how often
the locked rule would have passed under resampling.

ASSUMPTIONS AND LIMITS, stated plainly. (i) Families are treated as
an iid sample from a family population; METR's suite was curated,
not sampled, so this is a modelling assumption — the resulting SEs
are "dispersion under family resampling", a calibration of the
headline numbers against family composition, not a frequentist
sampling error in the strict design-based sense. (ii) The statistic
inherits the coarseness of a 5-fold mean. (iii) Multiplicity
weighting assumes within-family runs are exchangeable replicates of
the family's contribution, which is the cluster bootstrap's standard
stance. These limits are why the paper's claim-change alternative
(full descriptive reframing) remains available; this procedure is
the strongest justified calibration the design supports.

CHECKPOINTING. One JSON per resample in --out-dir; existing files are
skipped, so an interrupted run resumes where it stopped. Failures
inside a resample are recorded in that resample's JSON (status field)
and counted in the summary, never silently dropped.
==================================================================

Usage:
    python3 r22_family_refit_bootstrap.py --B 300 [--include-p0]
        [--workers 4] [--out-dir ../output/bootstrap_r22]
        [--seed-base 20261113] [--start 0] [--end B]
    python3 r22_family_refit_bootstrap.py --summarize
        [--out-dir ../output/bootstrap_r22]

Post-hoc (round 22); not preregistered.
"""
import argparse
import json
import os
import time

import numpy as np

from data import load_runs, family_folds
from predictors import SplineHorizon, MeanPredictor, IrtPredictor, _sigmoid
from run_cv import scores_fold, SEED

DEFAULT_B = 300
DEFAULT_SEED_BASE = 20261113
DEFAULT_OUT = "../output/bootstrap_r22"


def paired_stats(diffs):
    d = np.asarray(diffs, float)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"per_fold": d.tolist(), "mean": m, "se_within": se,
            "locked_rule_pass": bool(abs(m) > 2 * se)}


def run_resample(df, b, seed_base, include_p0):
    """One bootstrap resample. df: full locked dataframe (with 'fold'
    column NOT required). Returns a record dict."""
    rng = np.random.default_rng([seed_base, b])
    fams = np.sort(df["f"].unique())
    draw = rng.choice(fams, size=len(fams), replace=True)
    counts = {int(f): int(np.sum(draw == f)) for f in fams}
    mult = df["f"].map(counts).to_numpy(dtype=float)

    # re-derive the partition on the distinct families, per-resample seed
    part_seed = int(rng.integers(0, 2 ** 31 - 1))
    foldmap = family_folds(df, n_folds=5, seed=part_seed)
    fold = df["f"].map(foldmap).to_numpy()

    n_ai = df["j"].nunique()
    d12, d13, d10 = [], [], []
    fold_scores = []
    for k in range(5):
        trm = fold != k
        tem = fold == k
        tr = df[trm].reset_index(drop=True)
        te = df[tem].reset_index(drop=True)
        m_tr = mult[trm]; m_te = mult[tem]
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr_ti = tr["i"].map(tmap).to_numpy()
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float)
        w_tr = tr["w"].to_numpy() * m_tr; j_tr = tr["j"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float)
        w_te = te["w"].to_numpy() * m_te; j_te = te["j"].to_numpy()

        p1 = SplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai)
        p2 = MeanPredictor().fit(y_tr, w_tr, j_tr, n_ai)
        p3 = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, tr_ti, n_ai, len(tmap))
        s1 = scores_fold(y_te, p1.predict(x_te, j_te), w_te)
        s2 = scores_fold(y_te, p2.predict(j_te), w_te)
        s3 = scores_fold(y_te, p3.predict(x_te, j_te), w_te)
        row = {"fold": k, "P1": s1, "P2": s2, "P3": s3}
        d12.append(s1["logloss"] - s2["logloss"])
        d13.append(s1["logloss"] - s3["logloss"])
        if include_p0:
            from strengthen import fit_p0
            th0 = fit_p0(x_tr, y_tr, w_tr, j_tr, n_ai)
            s0 = scores_fold(y_te, _sigmoid(th0[j_te] - np.exp(th0[n_ai]) * x_te), w_te)
            row["P0"] = s0
            d10.append(s1["logloss"] - s0["logloss"])
        fold_scores.append(row)
    rec = {"resample": b, "seed_base": seed_base, "partition_seed": part_seed,
           "family_counts": {str(k): v for k, v in counts.items()},
           "n_families_drawn_distinct": int(np.sum(np.array(list(counts.values())) > 0)),
           "folds": fold_scores,
           "H1_p1_minus_p2": paired_stats(d12),
           "H2_p1_minus_p3": paired_stats(d13),
           "status": "ok"}
    if include_p0:
        rec["P1_minus_P0"] = paired_stats(d10)
    return rec


_DF = None
_INCLUDE_P0 = False
_SEED_BASE = DEFAULT_SEED_BASE
_OUT = DEFAULT_OUT


def _init_worker(df, include_p0, seed_base, out_dir):
    global _DF, _INCLUDE_P0, _SEED_BASE, _OUT
    _DF, _INCLUDE_P0, _SEED_BASE, _OUT = df, include_p0, seed_base, out_dir


def _one(b):
    path = os.path.join(_OUT, f"resample_{b:04d}.json")
    if os.path.exists(path):
        return (b, "skipped")
    t0 = time.time()
    try:
        rec = run_resample(_DF, b, _SEED_BASE, _INCLUDE_P0)
    except Exception as e:  # recorded, never silently dropped
        rec = {"resample": b, "seed_base": _SEED_BASE, "status": "failed",
               "error": f"{type(e).__name__}: {e}"}
    rec["wall_s"] = time.time() - t0
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    return (b, rec["status"])


def summarize(out_dir):
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("resample_"))
    recs = [json.load(open(os.path.join(out_dir, f))) for f in files]
    ok = [r for r in recs if r.get("status") == "ok"]
    failed = [r for r in recs if r.get("status") != "ok"]
    out = {"n_files": len(recs), "n_ok": len(ok), "n_failed": len(failed),
           "failed_resamples": [r["resample"] for r in failed]}
    locked = None
    cv_path = "../output/cv_scores.json"
    if os.path.exists(cv_path):
        cv = json.load(open(cv_path))["folds"]
        locked = {
            "H1": float(np.mean([f["P1"]["logloss"] - f["P2"]["logloss"] for f in cv])),
            "H2": float(np.mean([f["P1"]["logloss"] - f["P3"]["logloss"] for f in cv])),
        }
        out["locked_values_from_cv_scores"] = locked
    for key, lock_key in [("H1_p1_minus_p2", "H1"), ("H2_p1_minus_p3", "H2"),
                          ("P1_minus_P0", None)]:
        if not ok or key not in ok[0]:
            continue
        vals = np.array([r[key]["mean"] for r in ok])
        entry = {"bootstrap_se": float(vals.std(ddof=1)),
                 "mean": float(vals.mean()), "median": float(np.median(vals)),
                 "percentile_ci95": [float(np.percentile(vals, 2.5)),
                                     float(np.percentile(vals, 97.5))],
                 "frac_resamples_mean_below_0": float(np.mean(vals < 0)),
                 "locked_rule_pass_rate": float(np.mean(
                     [r[key]["locked_rule_pass"] for r in ok]))}
        if locked is not None and lock_key is not None:
            entry["locked_value"] = locked[lock_key]
            entry["locked_minus_bootstrap_median"] = locked[lock_key] - entry["median"]
        out[key] = entry
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--include-p0", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.summarize:
        summarize(args.out_dir)
        return
    end = args.end if args.end is not None else args.B
    df = load_runs()
    idx = list(range(args.start, end))
    t0 = time.time()
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(df, args.include_p0, args.seed_base, args.out_dir)) as pool:
            for b, status in pool.imap_unordered(_one, idx):
                print(f"resample {b}: {status} ({time.time() - t0:.0f}s elapsed)",
                      flush=True)
    else:
        _init_worker(df, args.include_p0, args.seed_base, args.out_dir)
        for b in idx:
            print(f"resample {b}: {_one(b)[1]} ({time.time() - t0:.0f}s elapsed)",
                  flush=True)
    print(f"done range [{args.start}, {end}) in {time.time() - t0:.0f}s")
    summarize(args.out_dir)


if __name__ == "__main__":
    main()
