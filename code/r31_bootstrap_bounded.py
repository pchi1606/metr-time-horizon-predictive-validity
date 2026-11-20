"""Round-31 NW30-1 (adjudication ADJUDICATION-R30.md, Perplexity
MAJOR 5 accepted as new work; register REG-P1-13): completion of the
REG-P1-5 family-cluster bootstrap under the bounded implementation.

The deposited run (code/r22_family_refit_bootstrap.py;
output/bootstrap_r22/) completed 294 of 300 resamples; six failed
(indices 9, 193, 244, 283, 292, 293) when an optimizer trial point
in P1's exp-gap parameterisation reached u ~ +750, exp(u) overflowed,
and the PCHIP knot values became non-finite. This script reruns the
identical procedure -- same seed base (20261113), same per-resample
RNG stream, same family draws and re-derived partitions, same P2/P3
fits -- with P1's fit bounded exactly as the manuscript's Section 3
guidance and Appendix D test prescribe: the exp-gap parameter is
clipped at u <= 700 inside the objective (the semantics of
r30_p1_sensitivity.make_f's bound). Appendix D / fix F-13 verified
on the locked folds that the bounded variant reproduces the locked
fits exactly (d log-loss 0.0 on all five folds), so the bound is
inert wherever it does not bind; each resample record here carries
diagnostics (largest trial u per fold; whether the clip binds at
the fitted solution) so inertness is checkable per resample.

Outputs go to output/bootstrap_r31/ (the deposited bootstrap_r22/
record is not touched). Summary statistics follow the deposited
summarize() definitions exactly, plus a comparison block against
the deposited 294 resamples.

Usage:
    python3 r31_bootstrap_bounded.py [--B 300] [--workers 4]
        [--out-dir ../output/bootstrap_r31] [--start 0] [--end B]
    python3 r31_bootstrap_bounded.py --summarize
        [--out-dir ../output/bootstrap_r31]

Post-hoc (round 31); not preregistered.
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

from data import load_runs, family_folds
from predictors import MeanPredictor, IrtPredictor, _sigmoid, _bce, math_logit
from run_cv import scores_fold

BOUND = 700.0
DEFAULT_B = 300
DEFAULT_SEED_BASE = 20261113
DEFAULT_OUT = "../output/bootstrap_r31"
DEPOSITED_DIR = "../output/bootstrap_r22"


class BoundedSplineHorizon:
    """P1 (SplineHorizon) with the exp-gap parameter clipped at
    u <= BOUND inside the objective -- the bounded implementation of
    the manuscript's Section 3 guidance, as tested in Appendix D
    (r30_p1_sensitivity.py, fix F-13/F-21). Otherwise identical to
    predictors.SplineHorizon: same knots, same initialisation, same
    optimizer call. Instrumented: max_trial_u_ records the largest
    trial u the optimizer evaluated; clip_binds_ records whether
    any fitted gap sits at the bound."""

    def __init__(self):
        self.knots_ = None
        self.f_ = None
        self.alpha_ = None
        self.beta_ = None
        self.max_trial_u_ = -np.inf
        self.clip_binds_ = False

    def _make_f(self, knots, u):
        u = np.minimum(u, BOUND)
        gaps = np.exp(u)
        frac = np.cumsum(gaps) / np.sum(gaps)
        xmin, xmax = knots[0], knots[-1]
        values = xmin + (xmax - xmin) * np.concatenate([[0.0], frac[:-1], [1.0]])
        return PchipInterpolator(knots, values, extrapolate=True), values

    def fit(self, x, y, w, j, n_ai):
        xs = np.asarray(x, float)
        xmin, xmax = xs.min(), xs.max()
        qs = np.quantile(xs, [0.2, 0.4, 0.6, 0.8])
        knots = np.concatenate([[xmin], qs, [xmax]])
        for k in range(1, len(knots)):
            if knots[k] <= knots[k - 1]:
                knots[k] = knots[k - 1] + 1e-9
        u0 = np.log(np.diff(knots))
        a0 = np.zeros(n_ai)
        for jj in range(n_ai):
            m = (j == jj)
            pj = np.average(y[m], weights=w[m])
            pj = min(max(pj, 1e-3), 1 - 1e-3)
            a0[jj] = math_logit(pj)
        x0 = np.concatenate([a0, [0.0], u0])

        def obj(theta):
            u = theta[n_ai + 1:]
            if len(u):
                self.max_trial_u_ = max(self.max_trial_u_, float(np.max(u)))
            alpha = theta[:n_ai]
            beta = np.exp(theta[n_ai])
            f, _ = self._make_f(knots, u)
            p = _sigmoid(alpha[j] - beta * f(xs))
            return _bce(y, p, w)

        res = minimize(obj, x0, method="L-BFGS-B",
                       options={"maxiter": 20000, "maxfun": 200000})
        assert res.success, f"P1 fit failed: {res.message}"
        alpha = res.x[:n_ai]
        beta = np.exp(res.x[n_ai])
        f, values = self._make_f(knots, res.x[n_ai + 1:])
        grid = np.linspace(xmin, xmax, 2001)
        assert np.all(np.diff(f(grid)) >= -1e-9), "fitted f not monotone"
        self.clip_binds_ = bool(np.any(res.x[n_ai + 1:] >= BOUND))
        self.knots_, self.f_, self.alpha_, self.beta_ = knots, f, alpha, beta
        return self

    def predict(self, x, j):
        return _sigmoid(self.alpha_[j] - self.beta_ * self.f_(np.asarray(x, float)))


def paired_stats(diffs):
    d = np.asarray(diffs, float)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"per_fold": d.tolist(), "mean": m, "se_within": se,
            "locked_rule_pass": bool(abs(m) > 2 * se)}


def run_resample(df, b, seed_base):
    """One bootstrap resample -- r22_family_refit_bootstrap.run_resample
    verbatim except P1 is BoundedSplineHorizon and bounded diagnostics
    are recorded per fold."""
    rng = np.random.default_rng([seed_base, b])
    fams = np.sort(df["f"].unique())
    draw = rng.choice(fams, size=len(fams), replace=True)
    counts = {int(f): int(np.sum(draw == f)) for f in fams}
    mult = df["f"].map(counts).to_numpy(dtype=float)

    part_seed = int(rng.integers(0, 2 ** 31 - 1))
    foldmap = family_folds(df, n_folds=5, seed=part_seed)
    fold = df["f"].map(foldmap).to_numpy()

    n_ai = df["j"].nunique()
    d12, d13 = [], []
    fold_scores = []
    diag = []
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

        p1 = BoundedSplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai)
        p2 = MeanPredictor().fit(y_tr, w_tr, j_tr, n_ai)
        p3 = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, tr_ti, n_ai, len(tmap))
        s1 = scores_fold(y_te, p1.predict(x_te, j_te), w_te)
        s2 = scores_fold(y_te, p2.predict(j_te), w_te)
        s3 = scores_fold(y_te, p3.predict(x_te, j_te), w_te)
        fold_scores.append({"fold": k, "P1": s1, "P2": s2, "P3": s3})
        diag.append({"fold": k, "p1_max_trial_u": p1.max_trial_u_,
                     "p1_clip_binds_at_solution": p1.clip_binds_})
        d12.append(s1["logloss"] - s2["logloss"])
        d13.append(s1["logloss"] - s3["logloss"])
    rec = {"resample": b, "seed_base": seed_base, "partition_seed": part_seed,
           "family_counts": {str(k): v for k, v in counts.items()},
           "n_families_drawn_distinct": int(np.sum(np.array(list(counts.values())) > 0)),
           "folds": fold_scores,
           "H1_p1_minus_p2": paired_stats(d12),
           "H2_p1_minus_p3": paired_stats(d13),
           "implementation": "r31 bounded (exp-gap u clipped at 700 "
                             "inside the P1 objective)",
           "bounded_diagnostics": diag,
           "status": "ok"}
    return rec


_DF = None
_SEED_BASE = DEFAULT_SEED_BASE
_OUT = DEFAULT_OUT


def _init_worker(df, seed_base, out_dir):
    global _DF, _SEED_BASE, _OUT
    _DF, _SEED_BASE, _OUT = df, seed_base, out_dir


def _one(b):
    path = os.path.join(_OUT, f"resample_{b:04d}.json")
    if os.path.exists(path):
        return (b, "skipped")
    # One retry on failure, both attempts recorded: pre-flight showed
    # the local BLAS threading can flip an optimizer success-flag
    # assertion between identical runs; a failure that does not
    # reproduce is an environment artifact, and the record says so.
    attempts = []
    rec = None
    for attempt in range(2):
        t0 = time.time()
        try:
            rec = run_resample(_DF, b, _SEED_BASE)
            status, err = rec["status"], None
        except Exception as e:  # recorded, never silently dropped
            rec = {"resample": b, "seed_base": _SEED_BASE, "status": "failed",
                   "error": f"{type(e).__name__}: {e}"}
            status, err = "failed", rec["error"]
        attempts.append({"attempt": attempt + 1, "status": status,
                         "error": err, "wall_s": time.time() - t0})
        if status == "ok":
            break
    rec["attempts"] = attempts
    rec["wall_s"] = sum(a["wall_s"] for a in attempts)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    return (b, rec["status"])


def _stats_block(recs, key, locked_val):
    vals = np.array([r[key]["mean"] for r in recs])
    entry = {"bootstrap_se": float(vals.std(ddof=1)),
             "mean": float(vals.mean()), "median": float(np.median(vals)),
             "percentile_ci95": [float(np.percentile(vals, 2.5)),
                                 float(np.percentile(vals, 97.5))],
             "frac_resamples_mean_below_0": float(np.mean(vals < 0)),
             "locked_rule_pass_rate": float(np.mean(
                 [r[key]["locked_rule_pass"] for r in recs])),
             "locked_value": locked_val,
             "locked_minus_bootstrap_median": locked_val - float(np.median(vals))}
    return entry


def summarize(out_dir):
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("resample_"))
    recs = [json.load(open(os.path.join(out_dir, f))) for f in files]
    ok = [r for r in recs if r.get("status") == "ok"]
    failed = [r for r in recs if r.get("status") != "ok"]
    out = {"n_files": len(recs), "n_ok": len(ok), "n_failed": len(failed),
           "failed_resamples": [r["resample"] for r in failed],
           "implementation": "r31 bounded (exp-gap u clipped at 700 "
                             "inside the P1 objective)"}
    cv = json.load(open("../output/cv_scores.json"))["folds"]
    locked = {"H1": float(np.mean([f["P1"]["logloss"] - f["P2"]["logloss"]
                                    for f in cv])),
              "H2": float(np.mean([f["P1"]["logloss"] - f["P3"]["logloss"]
                                    for f in cv]))}
    out["locked_values_from_cv_scores"] = locked
    if ok:
        out["H1_p1_minus_p2"] = _stats_block(ok, "H1_p1_minus_p2", locked["H1"])
        out["H2_p1_minus_p3"] = _stats_block(ok, "H2_p1_minus_p3", locked["H2"])
    # comparison against the deposited (unbounded, Kaggle) run
    dep_dir = DEPOSITED_DIR
    if os.path.isdir(dep_dir):
        comp = {}
        for key in ["H1_p1_minus_p2", "H2_p1_minus_p3"]:
            diffs, worst = [], None
            for r in ok:
                dp = os.path.join(dep_dir, f"resample_{r['resample']:04d}.json")
                if not os.path.exists(dp):
                    continue
                d = json.load(open(dp))
                if d.get("status") != "ok":
                    continue
                delta = abs(r[key]["mean"] - d[key]["mean"])
                diffs.append(delta)
                if worst is None or delta > worst[1]:
                    worst = (r["resample"], delta)
            if diffs:
                comp[key] = {"n_compared": len(diffs),
                             "max_abs_mean_diff": float(max(diffs)),
                             "median_abs_mean_diff": float(np.median(diffs)),
                             "worst_resample": worst[0]}
        # newly completed resamples (failed in the deposited run)
        newly = {}
        for r in ok:
            dp = os.path.join(dep_dir, f"resample_{r['resample']:04d}.json")
            if os.path.exists(dp):
                d = json.load(open(dp))
                if d.get("status") != "ok":
                    newly[str(r["resample"])] = {
                        "H1_mean": r["H1_p1_minus_p2"]["mean"],
                        "H2_mean": r["H2_p1_minus_p3"]["mean"],
                        "H1_locked_rule_pass": r["H1_p1_minus_p2"]["locked_rule_pass"],
                        "H2_locked_rule_pass": r["H2_p1_minus_p3"]["locked_rule_pass"],
                        "max_trial_u_over_folds": max(
                            g["p1_max_trial_u"] for g in r["bounded_diagnostics"]),
                        "clip_binds_any_fold": any(
                            g["p1_clip_binds_at_solution"]
                            for g in r["bounded_diagnostics"])}
        comp["newly_completed"] = newly
        # clip diagnostics across all completed resamples
        binds = [r["resample"] for r in ok
                 if any(g["p1_clip_binds_at_solution"]
                        for g in r.get("bounded_diagnostics", []))]
        comp["resamples_where_clip_binds_at_solution"] = binds
        out["comparison_vs_deposited_r22"] = comp
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
                  initargs=(df, args.seed_base, args.out_dir)) as pool:
            for b, status in pool.imap_unordered(_one, idx):
                print(f"resample {b}: {status} ({time.time() - t0:.0f}s elapsed)",
                      flush=True)
    else:
        _init_worker(df, args.seed_base, args.out_dir)
        for b in idx:
            print(f"resample {b}: {_one(b)[1]} ({time.time() - t0:.0f}s elapsed)",
                  flush=True)
    print(f"done range [{args.start}, {end}) in {time.time() - t0:.0f}s")
    summarize(args.out_dir)


if __name__ == "__main__":
    main()
