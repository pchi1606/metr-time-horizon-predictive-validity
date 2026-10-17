"""Diagnose P3 (IRT) extrapolation behavior per fold (test-23 C1/M1).

For each fold: fit P3 as in the preregistered pipeline, then report
  - fitted Rasch slope beta
  - train theta distribution (min/max/sd)
  - OLS extrapolation coefficients (a, b)
  - extrapolated theta_hat on HELD-OUT tasks (min/max/sd)
  - resulting predicted-probability extremes
Also fits P3r: ridge-penalized 1PL (L2 on theta, lambda=1.0) + same OLS
extrapolation, as an extrapolation-robustness variant.
Saves output/p3_diagnostics.json
"""
import json
import numpy as np
from scipy.optimize import minimize

from data import load_runs, family_folds
from predictors import _sigmoid, _bce, math_logit, EPS
from run_cv import scores_fold, SEED


def fit_irt(x, y, w, j, i, n_ai, n_task, ridge=0.0):
    xs = np.asarray(x, float)

    def obj(theta):
        alpha = theta[:n_ai]
        beta = np.exp(theta[n_ai])
        phi = theta[n_ai + 1:]
        th = phi - phi.mean()
        p = _sigmoid(alpha[j] - beta * th[i])
        val = _bce(y, p, w)
        if ridge > 0:
            val += ridge * np.sum(th ** 2)
        return val

    a0 = np.zeros(n_ai)
    for jj in range(n_ai):
        m = (j == jj)
        pj = min(max(np.average(y[m], weights=w[m]), 1e-3), 1 - 1e-3)
        a0[jj] = math_logit(pj)
    x0 = np.concatenate([a0, [0.0], np.zeros(n_task)])
    res = minimize(obj, x0, method="L-BFGS-B",
                   options={"maxiter": 20000, "maxfun": 200000})
    assert res.success, res.message
    phi = res.x[n_ai + 1:]
    theta = phi - phi.mean()
    beta = np.exp(res.x[n_ai])
    alpha = res.x[:n_ai]
    task_x = np.array([xs[i == t][0] for t in range(n_task)])
    X = np.column_stack([np.ones(n_task), task_x])
    (a, b), *_ = np.linalg.lstsq(X, theta, rcond=None)
    return {"alpha": alpha, "beta": beta, "theta": theta, "a": a, "b": b,
            "task_x": task_x}


def main():
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    out = {"folds": []}
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr["ti"] = tr["i"].map(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()

        fit = fit_irt(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, len(tmap), ridge=0.0)
        th_hat_te = fit["a"] + fit["b"] * x_te
        p = _sigmoid(fit["alpha"][j_te] - fit["beta"] * th_hat_te)
        s = scores_fold(y_te, p, w_te)

        fitr = fit_irt(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, len(tmap), ridge=1.0)
        th_hat_ter = fitr["a"] + fitr["b"] * x_te
        pr = _sigmoid(fitr["alpha"][j_te] - fitr["beta"] * th_hat_ter)
        sr = scores_fold(y_te, pr, w_te)

        rec = {
            "fold": k,
            "beta": float(fit["beta"]),
            "theta_train_min": float(fit["theta"].min()),
            "theta_train_max": float(fit["theta"].max()),
            "theta_train_sd": float(fit["theta"].std()),
            "ols_a": float(fit["a"]), "ols_b": float(fit["b"]),
            "thetahat_test_min": float(th_hat_te.min()),
            "thetahat_test_max": float(th_hat_te.max()),
            "thetahat_test_sd": float(th_hat_te.std()),
            "p_test_min": float(p.min()), "p_test_max": float(p.max()),
            "frac_extreme": float(np.mean((p < 0.01) | (p > 0.99))),
            "logloss": s["logloss"],
            "P3r_beta": float(fitr["beta"]),
            "P3r_thetahat_sd": float(th_hat_ter.std()),
            "P3r_frac_extreme": float(np.mean((pr < 0.01) | (pr > 0.99))),
            "P3r_logloss": sr["logloss"],
        }
        out["folds"].append(rec)
        print(f"fold {k}: beta={rec['beta']:.2f} th_te=[{rec['thetahat_test_min']:.2f},{rec['thetahat_test_max']:.2f}] "
              f"extreme={rec['frac_extreme']:.2f} ll={rec['logloss']:.3f} | "
              f"P3r: beta={rec['P3r_beta']:.2f} extreme={rec['P3r_frac_extreme']:.2f} ll={rec['P3r_logloss']:.3f}",
              flush=True)
    with open("output/p3_diagnostics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved output/p3_diagnostics.json")


if __name__ == "__main__":
    main()
