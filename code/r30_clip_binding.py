"""Round-30 fix F-10 rider (adjudication addendum, Perplexity F7):
clip-binding counts for the Appendix D convergence diagnostic.

The diagnostic (p3_convergence.py) evaluates its objective value
through the clipped helper (_bce clips p to [1e-12, 1-1e-12]) but
forms its analytic gradient on the unclipped p, so the reported
stationarity measure is exact only where the clip does not bind.
This script measures where the clip binds: on the stage-1 training
runs of each locked fold, at the locked solution and at the
tightened refit (the two solutions the diagnostic reports), it
counts the runs whose unclipped fitted probability lies at or
beyond the clip (p <= 1e-12 or p >= 1-1e-12) and their share of
the total shipped weight.

Fits replicate p3_convergence.py exactly (same replication path,
same anchors asserted against output/cv_scores.json and the
deposited output/p3_convergence.json). Saves
../output/r30_clip_binding.json.
"""
import json
import numpy as np
from scipy.optimize import minimize

from data import load_runs, family_folds
from predictors import IrtPredictor, _sigmoid, math_logit, EPS
from run_cv import SEED
from p3_convergence import stage1_obj, stage1_obj_jac, identified

CLIP_Z = float(np.log((1.0 - EPS) / EPS))  # |z| at which the clip binds


def binding_stats(theta, y, w, j, i, n_ai):
    alpha = theta[:n_ai]
    beta = np.exp(theta[n_ai])
    phi = theta[n_ai + 1:]
    th = phi - phi.mean()
    z = alpha[j] - beta * th[i]
    p = _sigmoid(z)
    bind = (p <= EPS) | (p >= 1.0 - EPS)
    return {
        "n_runs": int(len(y)),
        "n_binding": int(bind.sum()),
        "frac_binding": float(bind.mean()),
        "weight_total": float(w.sum()),
        "weight_binding": float(w[bind].sum()),
        "weight_share_binding": float(w[bind].sum() / w.sum()),
        "max_abs_z": float(np.max(np.abs(z))),
        "clip_z_threshold": CLIP_Z,
    }


def main():
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    cv = json.load(open("../output/cv_scores.json"))
    conv = json.load(open("../output/p3_convergence.json"))
    out = {"seed": SEED, "clip": EPS, "folds": []}
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        tr = tr.copy()
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr["ti"] = tr["i"].map(tmap)
        n_task = len(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(dtype=float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy()
        i_tr = tr["ti"].to_numpy()

        # locked fit via the deposited predictor (anchor), as in
        # p3_convergence.py
        locked = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr,
                                    n_ai, n_task)
        te = df[df["fold"] == k].reset_index(drop=True)
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(dtype=float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()
        id_locked, _ = identified(locked, x_te, y_te, w_te, j_te)
        assert abs(id_locked["logloss"]
                   - cv["folds"][k]["P3"]["logloss"]) < 1e-9

        # replicate the raw parameter vector (p3_convergence path)
        a0 = np.zeros(n_ai)
        for jj in range(n_ai):
            m = (j_tr == jj)
            pj = np.average(y_tr[m], weights=w_tr[m])
            a0[jj] = math_logit(min(max(pj, 1e-3), 1 - 1e-3))
        x0 = np.concatenate([a0, [0.0], np.zeros(n_task)])
        res = minimize(stage1_obj, x0, args=(y_tr, w_tr, j_tr, i_tr, n_ai),
                       method="L-BFGS-B",
                       options={"maxiter": 20000, "maxfun": 200000})
        assert res.success, f"fold {k}: locked replication failed"
        # anchor against the deposited convergence record
        assert abs(res.fun - conv["folds"][k]["locked"]["objective"]) < 1e-6

        b_lock = binding_stats(res.x, y_tr, w_tr, j_tr, i_tr, n_ai)

        # tightened refit, warm-started, analytic Jacobian (as deposited)
        res2 = minimize(stage1_obj_jac, res.x,
                        args=(y_tr, w_tr, j_tr, i_tr, n_ai, n_task),
                        method="L-BFGS-B", jac=True,
                        options={"maxiter": 20000, "maxfun": 200000,
                                 "ftol": 1e-13, "gtol": 1e-10})
        assert abs(res2.fun
                   - conv["folds"][k]["tightened"]["objective"]) < 1e-6
        b_tight = binding_stats(res2.x, y_tr, w_tr, j_tr, i_tr, n_ai)

        out["folds"].append({"fold": k, "locked": b_lock,
                             "tightened": b_tight})
        print(f"fold {k}: locked binding {b_lock['n_binding']}/"
              f"{b_lock['n_runs']} runs, weight share "
              f"{b_lock['weight_share_binding']:.3e}; tightened binding "
              f"{b_tight['n_binding']}/{b_tight['n_runs']}, weight share "
              f"{b_tight['weight_share_binding']:.3e}; "
              f"max|z| locked {b_lock['max_abs_z']:.1f}", flush=True)

    out["summary"] = {
        "total_binding_locked": int(sum(f["locked"]["n_binding"]
                                        for f in out["folds"])),
        "total_binding_tightened": int(sum(f["tightened"]["n_binding"]
                                           for f in out["folds"])),
        "max_weight_share_locked": max(
            f["locked"]["weight_share_binding"] for f in out["folds"]),
        "max_weight_share_tightened": max(
            f["tightened"]["weight_share_binding"] for f in out["folds"]),
    }
    with open("../output/r30_clip_binding.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("summary:", json.dumps(out["summary"], indent=1))
    print("saved ../output/r30_clip_binding.json")


if __name__ == "__main__":
    main()
