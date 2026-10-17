"""Post-hoc strengthening analyses (NOT preregistered; labeled as such in the paper).

S1. Calibration: reliability bins + ECE per predictor (pooled out-of-fold predictions).
S2. Weighting sensitivity: rescore CV predictions with equal-task weights.
S3. Per-AI gain: per-AI paired log-loss differences (P1 vs P2, P1 vs P3).
S4. Horizon uncertainty: per-AI 50% horizons from each fold's P1 fit (mean +/- sd).

Refits the predictors per fold (same seeds/splits as run_cv.py) and saves
../output/strengthen.json. Predictions are regenerated (not stored by run_cv).
"""
import json
import numpy as np
from scipy.optimize import minimize

from data import load_runs, family_folds
from predictors import SplineHorizon, MeanPredictor, IrtPredictor, _sigmoid, _bce, math_logit, EPS
from run_cv import scores_fold, SEED

N_BINS = 10


def ece_and_bins(y, p, w, n_bins=N_BINS):
    order = np.argsort(p)
    y, p, w = y[order], p[order], w[order]
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    bins = []
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            bins.append({"bin": b, "n": 0})
            continue
        wb = w[m] / w[m].sum()
        mpred = float(np.sum(wb * p[m]))
        mobs = float(np.sum(wb * y[m]))
        e += (w[m].sum() / w.sum()) * abs(mobs - mpred)
        bins.append({"bin": b, "n": int(m.sum()),
                     "mean_pred": mpred, "mean_obs": mobs})
    return float(e), bins


def fit_p0(x_tr, y_tr, w_tr, j_tr, n_ai):
    a0 = np.zeros(n_ai)
    for jj in range(n_ai):
        m = j_tr == jj
        pj = min(max(np.average(y_tr[m], weights=w_tr[m]), 1e-3), 1 - 1e-3)
        a0[jj] = math_logit(pj)

    def obj(th):
        return _bce(y_tr, _sigmoid(th[j_tr] - np.exp(th[n_ai]) * x_tr), w_tr)

    res = minimize(obj, np.concatenate([a0, [0.0]]), method="L-BFGS-B")
    assert res.success
    return res.x


def main():
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    ai_names = (df[["j", "alias"]].drop_duplicates()
                .sort_values("j")["alias"].tolist())

    oof = {m: [] for m in ["P1", "P2", "P3", "P0"]}
    yy, ww, jj_all = [], [], []
    # equal-task weights: 1 / n_ij per run
    pair_n = df.groupby(["i", "j"])["y"].transform("count").to_numpy()
    w_eq = 1.0 / pair_n
    eq_scores = {m: [] for m in ["P1", "P2", "P3", "P0"]}
    per_ai = {m: np.zeros((5, n_ai)) for m in ["P1", "P2", "P3"]}
    horizons_folds = np.zeros((5, n_ai))
    p0_sqrt_ll = []

    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr["ti"] = tr["i"].map(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()
        weq_te = (1.0 / te.groupby(["i", "j"])["y"].transform("count")).to_numpy()

        p1 = SplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai)
        p2 = MeanPredictor().fit(y_tr, w_tr, j_tr, n_ai)
        p3 = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, len(tmap))
        th0 = fit_p0(x_tr, y_tr, w_tr, j_tr, n_ai)

        preds = {
            "P1": p1.predict(x_te, j_te),
            "P2": p2.predict(j_te),
            "P3": p3.predict(x_te, j_te),
            "P0": _sigmoid(th0[j_te] - np.exp(th0[n_ai]) * x_te),
        }
        for m, p in preds.items():
            oof[m].append(p)
            eq_scores[m].append(scores_fold(y_te, p, weq_te)["logloss"])
        p0_sqrt_ll.append(scores_fold(y_te, preds["P0"], w_te)["logloss"])
        yy.append(y_te); ww.append(w_te); jj_all.append(j_te)
        for m, p in [("P1", preds["P1"]), ("P2", preds["P2"]), ("P3", preds["P3"])]:
            pc = np.clip(p, EPS, 1 - EPS)
            for jj in range(n_ai):
                mj = j_te == jj
                if mj.sum():
                    per_ai[m][k, jj] = float(
                        -np.sum(w_te[mj] * (y_te[mj] * np.log(pc[mj]) +
                                            (1 - y_te[mj]) * np.log(1 - pc[mj])))
                        / np.sum(w_te[mj]))
        for jj in range(n_ai):
            horizons_folds[k, jj] = p1.horizon(jj)
        print(f"fold {k} done", flush=True)

    out = {"note": "post-hoc, not preregistered"}
    cal = {}
    for m in ["P1", "P2", "P3", "P0"]:
        y = np.concatenate(yy); p = np.concatenate(oof[m]); w = np.concatenate(ww)
        e, bins = ece_and_bins(y, p, w)
        cal[m] = {"ece": e, "bins": bins}
    out["calibration"] = cal
    out["equal_task_weight_logloss"] = {
        m: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(5)),
            "per_fold": [float(x) for x in v]}
        for m, v in eq_scores.items()}
    out["p0_logloss_per_fold_sqrtweight"] = p0_sqrt_ll
    per_ai_diff = {}
    for a, b in [("P1", "P2"), ("P1", "P3")]:
        d = (per_ai[a] - per_ai[b]).mean(axis=0)
        per_ai_diff[f"{a}_vs_{b}"] = {ai_names[jj]: float(d[jj]) for jj in range(n_ai)}
    out["per_ai_logloss_diff"] = per_ai_diff
    out["horizon_fold_mean_sd_min"] = {
        ai_names[jj]: {"mean": float(horizons_folds[:, jj].mean()),
                       "sd": float(horizons_folds[:, jj].std(ddof=1))}
        for jj in range(n_ai)}
    with open("../output/strengthen.json", "w") as f:
        json.dump(out, f, indent=2)
    print("ECE:", {m: round(cal[m]["ece"], 4) for m in cal})
    print("equal-task logloss means:",
          {m: round(out["equal_task_weight_logloss"][m]["mean"], 4) for m in eq_scores})
    print("saved ../output/strengthen.json")


if __name__ == "__main__":
    main()
