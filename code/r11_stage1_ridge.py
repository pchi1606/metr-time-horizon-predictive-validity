"""Round-11 stage-1 penalty sensitivity (post-hoc; preregistered in
panel/r11-record.md section 5, amended by sections 5b and 5c
before this version was run).

Question: is P3's extremeness an artefact of the stage-1 1PL fit
being unpenalised?

Design (section 5c, final): the deposited scaffold's own fitting
protocol (p3_diagnostics.fit_irt copied verbatim below -- same
parameterisation, init, and L-BFGS-B options), with the penalty
applied to the IDENTIFIED products: objective = weighted BCE +
lambda * sum((beta * theta_i)^2). The penalty is invariant along
the (beta, theta) scale ray, and at lambda=0 the objective is
exactly the scaffold's, so the run-validation anchor is the full
p3_diagnostics.json row set. (Two earlier designs were recorded
and abandoned in panel/r11-record.md 5b/5c: a theta-only penalty
is minimised along the scale ray -- the deposited P3r rows are
that pathology -- and a pinned-scale refit optimises materially
further than the scaffold, confounding penalty with optimiser
convergence.)

Grid: lambda in {0, 0.003, 0.01, 0.03, 0.1, 0.3} (calibrated to
sum((beta*theta)^2) ~ 2e4 and a BCE sum ~ 1e4 at the unpenalised
fit). Saves output/r11_stage1_ridge.json
"""
import json
import numpy as np
from scipy.optimize import minimize

from data import load_runs, family_folds
from predictors import _sigmoid, _bce, math_logit
from run_cv import scores_fold, SEED

GRID = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3]


def fit_irt_ident_ridge(x, y, w, j, i, n_ai, n_task, lam=0.0):
    """p3_diagnostics.fit_irt verbatim, plus lam * sum((beta*th)^2)."""
    xs = np.asarray(x, float)

    def obj(theta):
        alpha = theta[:n_ai]
        beta = np.exp(theta[n_ai])
        phi = theta[n_ai + 1:]
        th = phi - phi.mean()
        p = _sigmoid(alpha[j] - beta * th[i])
        val = _bce(y, p, w)
        if lam > 0:
            val += lam * np.sum((beta * th) ** 2)
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
    anchors = {f["fold"]: f for f in
               json.load(open("../output/p3_diagnostics.json"))["folds"]}
    out = {"grid": GRID, "folds": []}
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr["ti"] = tr["i"].map(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()

        rec = {"fold": k, "by_lambda": []}
        for lam in GRID:
            fit = fit_irt_ident_ridge(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai,
                                      len(tmap), lam=lam)
            th_hat_te = fit["a"] + fit["b"] * x_te
            p = _sigmoid(fit["alpha"][j_te] - fit["beta"] * th_hat_te)
            s = scores_fold(y_te, p, w_te)
            row = {
                "lambda": lam,
                "beta": float(fit["beta"]),
                "eff_slope": float(fit["beta"] * fit["b"]),
                "logit_span_test": float(fit["beta"] * (th_hat_te.max() - th_hat_te.min())),
                "thetahat_test_sd": float(th_hat_te.std()),
                "frac_extreme": float(np.mean((p < 0.01) | (p > 0.99))),
                "logloss": float(s["logloss"]),
            }
            rec["by_lambda"].append(row)
            print(f"fold {k} lambda={lam:g}: beta={row['beta']:.2f} "
                  f"eff={row['eff_slope']:.2f} span={row['logit_span_test']:.1f} "
                  f"extreme={row['frac_extreme']:.2f} ll={row['logloss']:.3f}",
                  flush=True)
        a0 = anchors[k]
        mine = rec["by_lambda"][0]
        anchor = {
            "beta": a0["beta"], "eff_slope": a0["beta"] * a0["ols_b"],
            "span": a0["beta"] * (a0["thetahat_test_max"] - a0["thetahat_test_min"]),
            "thetahat_test_sd": a0["thetahat_test_sd"],
            "frac_extreme": a0["frac_extreme"], "logloss": a0["logloss"],
        }
        rec["lambda0_anchor"] = anchor
        ok = (abs(mine["beta"] - anchor["beta"]) < 0.05 and
              abs(mine["eff_slope"] - anchor["eff_slope"]) < 0.05 and
              abs(mine["logit_span_test"] - anchor["span"]) < 0.5 and
              abs(mine["thetahat_test_sd"] - anchor["thetahat_test_sd"]) < 0.02 and
              abs(mine["frac_extreme"] - anchor["frac_extreme"]) < 0.02 and
              abs(mine["logloss"] - anchor["logloss"]) < 0.005)
        rec["lambda0_anchor_match"] = bool(ok)
        print(f"fold {k} lambda0 anchor match: {ok}", flush=True)
        out["folds"].append(rec)
    out["means"] = [
        {"lambda": lam,
         "eff_slope": float(np.mean([f["by_lambda"][i]["eff_slope"] for f in out["folds"]])),
         "logit_span_test": float(np.mean([f["by_lambda"][i]["logit_span_test"] for f in out["folds"]])),
         "frac_extreme": float(np.mean([f["by_lambda"][i]["frac_extreme"] for f in out["folds"]])),
         "logloss": float(np.mean([f["by_lambda"][i]["logloss"] for f in out["folds"]]))}
        for i, lam in enumerate(GRID)]
    with open("../output/r11_stage1_ridge.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved output/r11_stage1_ridge.json")


if __name__ == "__main__":
    main()
