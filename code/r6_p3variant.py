"""Round-7 sensitivity (post-hoc; referee M2 on the round-6 panel):
is H2's direction an artefact of P3's second stage?

P3b keeps P3's stage-1 1PL fit (alpha_j, beta, theta_i on train tasks)
exactly as locked, but replaces the unpenalised OLS projection
theta = a + b*x with a PREDICTION-FITTED projection: (a, b) chosen to
minimise the weighted binary cross-entropy of the train runs
themselves, logit p = alpha_j - beta*(a + b*x). Everything else
(folds, weights, clip, scoring) is the locked pipeline's.

Saves ../output/r6_p3variant.json with per-fold log-loss/Brier for
P3b, the P1-P3b paired differences, and each fold's fitted effective
slope beta*b (compare P3's OLS beta*b of 1.75-2.09 and P0's 0.54-0.63).
"""
import json
import numpy as np
from scipy.optimize import minimize

from data import load_runs, family_folds
from predictors import SplineHorizon, IrtPredictor, _sigmoid, _bce, EPS
from run_cv import scores_fold, SEED


def fit_projection_bce(alpha, beta, x, y, w, j, a0, b0):
    def obj(ab):
        a, b = ab
        p = _sigmoid(alpha[j] - beta * (a + b * x))
        return _bce(y, p, w)
    res = minimize(obj, np.array([a0, b0]), method="L-BFGS-B")
    assert res.success, f"projection fit failed: {res.message}"
    return float(res.x[0]), float(res.x[1])


def main():
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    cv = json.load(open("../output/cv_scores.json"))
    out = {"folds": []}
    d_ll, d_br = [], []
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr = tr.copy(); te = te.copy()
        tr["ti"] = tr["i"].map(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()

        p3 = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, len(tmap))
        a_b, b_b = fit_projection_bce(p3.alpha_, p3.beta_, x_tr, y_tr, w_tr,
                                      j_tr, p3.a_, p3.b_)
        p3b_te = _sigmoid(p3.alpha_[j_te] - p3.beta_ * (a_b + b_b * x_te))
        s3b = scores_fold(y_te, p3b_te, w_te)
        p1 = SplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai)
        s1 = scores_fold(y_te, p1.predict(x_te, j_te), w_te)
        # fidelity: P1 and P3 fold scores must match the locked run
        assert abs(s1["logloss"] - cv["folds"][k]["P1"]["logloss"]) < 1e-9
        d_ll.append(s1["logloss"] - s3b["logloss"])
        d_br.append(s1["brier"] - s3b["brier"])
        out["folds"].append({"fold": k, "P3b": s3b,
                             "beta_b_ols": float(p3.beta_ * p3.b_),
                             "beta_b_bce": float(p3.beta_ * b_b),
                             "P1_minus_P3b_logloss": d_ll[-1],
                             "P1_minus_P3b_brier": d_br[-1]})
        print(f"fold {k}: P3b ll={s3b['logloss']:.4f} beta*b OLS={p3.beta_*p3.b_:.2f} BCE={p3.beta_*b_b:.2f}", flush=True)
    d_ll = np.array(d_ll); d_br = np.array(d_br)
    out["P1_minus_P3b"] = {
        "logloss_mean": float(d_ll.mean()), "logloss_se": float(d_ll.std(ddof=1) / np.sqrt(5)),
        "brier_mean": float(d_br.mean()), "brier_se": float(d_br.std(ddof=1) / np.sqrt(5)),
        "per_fold_logloss": d_ll.tolist()}
    with open("../output/r6_p3variant.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["P1_minus_P3b"], indent=1))


if __name__ == "__main__":
    main()
