"""U-C simulation calibration (P1 round 2, ENACTMENT-2026-11-20).

Calibrates the manuscript's fold-difference SE and the heuristic x1.5
overlap-correction factor under a grouped design mimicking the paper's
own: 79 families, observed family sizes and weights, 5-fold family
partition, correlated runs within family (task + family random effects
on the logit scale, variances estimated from the real data).

Truths (base linear predictors fit once on the full real data):
  T1 (alternative): full-data P1 (SplineHorizon) fitted probabilities.
  T0 (null for H1/D10): per-AI weighted mean success (no duration
     effect), i.e. P2's own model.
Both truths add task effects u_i ~ N(0, s2_task) and family effects
v_f ~ N(0, s2_fam), with s2 estimated by delta-method moment matching
on the T1 fit's residuals (see truth fit below).

Replication b: draw fresh u*, v*; p* = sigmoid(eta_base + u*_i + v*_f);
y* ~ Bernoulli(p*); draw a fresh 5-fold family partition; run the locked
pipeline (P1/P2/P3/P0/Pmetr per fold, run_cv.scores_fold); record the five
paired fold log-loss differences for H1 (P1-P2), H2 (P1-P3), D10 (P1-P0)
and MP (Pmetr-P1, the manuscript's printed direction), their mean m_b and
fold SE s_b = sd(diffs, ddof=1)/sqrt(5).

Amended outcome rule (pass-2 skeptic required repair, adopted 2026-11-20;
see ROUND2-RECORD.md): the P1-P3b comparison is calibrated by the H2
series — P3b is the lambda=0 endpoint of the P3r grid, whose construction
(stage-1 IRT fit + OLS difficulty regression + Rasch probabilities) is
exactly P3's; identity verified on the locked partition (bracket lambda=0
outer mean -0.4182 vs locked H2 mean -0.418). P3's stage-1 fit is computed
with the analytic-gradient drop-in ROUND2/fast_ridge.py (lam=0), whose
equivalence to the deposited finite-difference construction is checked by
validate_fast_ridge.py against the deposited bracket record. Pmetr is
r22_metr_spec.fit_metr per AI (stored weights renormalised within the
AI's training rows — the manuscript's construction; its two weight arms
coincide on the locked partition).

Calibration factor per comparison: c = sd_b(m_b) / mean_b(s_b).
Coverage and null clearance rates are computed by uc_summarize.py.

Checkpointing: one JSON per replication in ROUND2/uc-reps/; reruns skip
completed replications. Usage:
  python3 uc_calibration.py [n_main_T1] [n_null_T0] [n_workers]
Defaults: 200 / 100 / 2. Master seed 20261121.
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.expanduser("~/workspace/p1-retarget/repro/code")
sys.path.insert(0, REPRO)
os.chdir(REPRO)

from data import load_runs, family_folds            # noqa: E402
from predictors import SplineHorizon, MeanPredictor  # noqa: E402
from run_cv import scores_fold                      # noqa: E402
from strengthen import fit_p0                       # noqa: E402
from r22_p3r import predict_from_fit                # noqa: E402
from r22_metr_spec import fit_metr                  # noqa: E402
sys.path.insert(0, HERE)
from fast_ridge import fit_irt_ident_ridge_fast     # noqa: E402

MASTER_SEED = 20261121
REPS_DIR = os.path.join(HERE, "uc-reps")
TRUTH_PATH = os.path.join(HERE, "uc-truth.npz")
TRUTH_META = os.path.join(HERE, "uc-truth.json")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def fit_truth(df):
    """Fit T1/T0 base predictors + random-effect variances; cache to disk."""
    if os.path.exists(TRUTH_PATH) and os.path.exists(TRUTH_META):
        meta = json.load(open(TRUTH_META))
        return meta
    x = df["x"].to_numpy(); y = df["y"].to_numpy(dtype=float)
    w = df["w"].to_numpy(); j = df["j"].to_numpy()
    i = df["i"].to_numpy(); f = df["f"].to_numpy()
    n_ai = df["j"].nunique()
    p1 = SplineHorizon().fit(x, y, w, j, n_ai)
    phat = np.clip(p1.predict(x, j), 1e-9, 1 - 1e-9)
    eta_t1 = _logit(phat)
    # T0 base: per-AI weighted mean
    p0base = np.zeros(len(df))
    for jj in range(n_ai):
        m = j == jj
        p0base[m] = np.sum(w[m] * y[m]) / np.sum(w[m])
    eta_t0 = _logit(np.clip(p0base, 1e-9, 1 - 1e-9))
    # delta-method group residuals under T1: r_g = sum(y-p)/sum(p(1-p)),
    # sampling var of r_g ~ 1/sum(p(1-p)).
    resid = y - phat
    info = phat * (1 - phat)

    def group_effects(codes):
        r = {}
        for g in np.unique(codes):
            m = codes == g
            s_info = info[m].sum()
            r[g] = (resid[m].sum() / s_info, 1.0 / s_info)
        return r

    rt = group_effects(i)   # per task
    rf = group_effects(f)   # per family
    # task variance: E[r^2] = s2_task + s2_fam + noise (task nested in family;
    # family effect is common to the task's runs)
    m2_t = np.mean([v[0] ** 2 for v in rt.values()])
    nv_t = np.mean([v[1] for v in rt.values()])
    # family variance: family mean residual averages ~n_t tasks:
    # E[r_f^2] = s2_fam + s2_task / n_tasks(f) + noise_f
    tasks_per_fam = df.groupby("f")["i"].nunique()
    inv_nt = float(np.mean(1.0 / tasks_per_fam.to_numpy()))
    m2_f = np.mean([v[0] ** 2 for v in rf.values()])
    nv_f = np.mean([v[1] for v in rf.values()])
    # solve: s2_fam = m2_f - nv_f - s2_task * inv_nt ; iterate once with
    # s2_task from m2_t - nv_t - s2_fam (start s2_fam = 0)
    s2_fam = 0.0
    for _ in range(10):
        s2_task = max(0.0, m2_t - nv_t - s2_fam)
        s2_fam_new = max(0.0, m2_f - nv_f - s2_task * inv_nt)
        if abs(s2_fam_new - s2_fam) < 1e-12:
            s2_fam = s2_fam_new
            break
        s2_fam = s2_fam_new
    s2_task = max(0.0, m2_t - nv_t - s2_fam)
    np.savez(TRUTH_PATH, eta_t1=eta_t1, eta_t0=eta_t0)
    meta = {"s2_task": float(s2_task), "s2_fam": float(s2_fam),
            "sd_task": float(math.sqrt(s2_task)),
            "sd_fam": float(math.sqrt(s2_fam)),
            "n_runs": int(len(df)), "n_tasks": int(df["i"].nunique()),
            "n_families": int(df["f"].nunique()), "n_ai": int(n_ai)}
    json.dump(meta, open(TRUTH_META, "w"), indent=1)
    return meta


_DF = None
_ETA = None
_META = None


def _init():
    global _DF, _ETA, _META
    _DF = load_runs()
    z = np.load(TRUTH_PATH)
    _ETA = {"T1": z["eta_t1"], "T0": z["eta_t0"]}
    _META = json.load(open(TRUTH_META))


def run_replication(arg):
    truth, b = arg
    os.makedirs(REPS_DIR, exist_ok=True)
    ck = os.path.join(REPS_DIR, f"{truth}_{b:04d}.json")
    if os.path.exists(ck):
        return (truth, b, "cached")
    rng = np.random.default_rng(
        np.random.SeedSequence([MASTER_SEED, 1 if truth == "T1" else 0, b]))
    df = _DF
    n_task = df["i"].nunique(); n_fam = df["f"].nunique()
    n_ai = _META["n_ai"]
    u = rng.normal(0.0, _META["sd_task"], size=int(n_task))
    v = rng.normal(0.0, _META["sd_fam"], size=int(n_fam))
    i = df["i"].to_numpy(); f = df["f"].to_numpy()
    pstar = _sigmoid(_ETA[truth] + u[i] + v[f])
    ystar = (rng.random(len(df)) < pstar).astype(float)
    pseed = int(rng.integers(0, 2**31 - 1))
    d = df.copy()
    d["y"] = ystar
    folds = family_folds(d, seed=pseed)
    d["fold"] = d["f"].map(folds)
    diffs = {"H1": [], "H2": [], "D10": [], "MP": []}
    t0 = time.time()
    try:
        for k in range(5):
            tr = d[d["fold"] != k].reset_index(drop=True)
            te = d[d["fold"] == k].reset_index(drop=True)
            tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
            tr = tr.copy(); tr["ti"] = tr["i"].map(tmap)
            x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(dtype=float)
            w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy()
            i_tr = tr["ti"].to_numpy()
            x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(dtype=float)
            w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()
            p1 = SplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai)
            p2 = MeanPredictor().fit(y_tr, w_tr, j_tr, n_ai)
            fit3 = fit_irt_ident_ridge_fast(x_tr, y_tr, w_tr, j_tr, i_tr,
                                            n_ai, len(tmap), lam=0.0)
            p3_pred = predict_from_fit(fit3, x_te, j_te)
            th0 = fit_p0(x_tr, y_tr, w_tr, j_tr, n_ai)
            # Pmetr: per-AI logistic (r22_metr_spec), stored weights
            # renormalised within the AI's training rows
            pmetr_pred = np.zeros(len(te))
            for jj in range(n_ai):
                mtr = j_tr == jj
                wn = w_tr[mtr] / w_tr[mtr].sum()
                a_m, b_m, _ok = fit_metr(x_tr[mtr], y_tr[mtr], wn)
                mte = j_te == jj
                pmetr_pred[mte] = _sigmoid(a_m + b_m * x_te[mte])
            ll = {
                "P1": scores_fold(y_te, p1.predict(x_te, j_te), w_te)["logloss"],
                "P2": scores_fold(y_te, p2.predict(j_te), w_te)["logloss"],
                "P3": scores_fold(y_te, p3_pred, w_te)["logloss"],
                "P0": scores_fold(y_te, _sigmoid(th0[j_te] - np.exp(th0[n_ai]) * x_te), w_te)["logloss"],
                "Pmetr": scores_fold(y_te, pmetr_pred, w_te)["logloss"],
            }
            diffs["H1"].append(ll["P1"] - ll["P2"])
            diffs["H2"].append(ll["P1"] - ll["P3"])
            diffs["D10"].append(ll["P1"] - ll["P0"])
            diffs["MP"].append(ll["Pmetr"] - ll["P1"])
    except Exception as e:  # fit failure on simulated data: record, skip
        json.dump({"truth": truth, "rep": b, "failed": str(e)},
                  open(ck, "w"), indent=1)
        return (truth, b, f"FAILED {e}")
    rec = {"truth": truth, "rep": b, "partition_seed": pseed,
           "seconds": round(time.time() - t0, 1), "comparisons": {}}
    for name, dv in diffs.items():
        a = np.array(dv)
        rec["comparisons"][name] = {
            "diffs": [float(v) for v in dv], "mean": float(a.mean()),
            "se": float(a.std(ddof=1) / math.sqrt(5))}
    json.dump(rec, open(ck, "w"), indent=1)
    return (truth, b, "done")


def main():
    n_t1 = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_t0 = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    os.makedirs(REPS_DIR, exist_ok=True)
    df = load_runs()
    meta = fit_truth(df)
    print(f"truth: {meta}", flush=True)
    jobs = ([("T1", b) for b in range(n_t1)]
            + [("T0", b) for b in range(n_t0)])
    t0 = time.time()
    with Pool(n_workers, initializer=_init) as pool:
        for n, res in enumerate(pool.imap_unordered(run_replication, jobs)):
            if n % 10 == 0 or res[2] not in ("done", "cached"):
                print(f"[{time.time() - t0:7.0f}s] {n + 1}/{len(jobs)} {res}",
                      flush=True)
    print(f"ALL DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
