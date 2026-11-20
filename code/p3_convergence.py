"""Round-14 add-on: P3 convergence demonstrated to a printed tolerance.

Appendix D previously rested the P3 (1PL + OLS) fits on the L-BFGS-B
success flag plus qualitative diagnostics. This script demonstrates
convergence directly, per fold, on the locked scaffold (seed 20261012,
identical fold construction to run_cv.py):

1. Refit stage 1 with the locked optimizer settings (deposited
   IrtPredictor code path, re-implemented here only to expose the
   optimizer result; the identified quantities are checked against
   the deposited predictor and against output/cv_scores.json).
2. Evaluate the ANALYTIC gradient of the stage-1 objective at the
   locked solution (the objective is invariant along the
   (beta, theta) scale ray, and the gradient is orthogonal to that
   ray identically, so its norm is a meaningful stationarity
   measure despite the non-identification).
3. Refit warm-started from the locked solution with an analytic
   Jacobian and tolerances tightened by orders of magnitude
   (ftol 1e-13, gtol 1e-10); report the change in every identified
   quantity and in the held-out log-loss.

Saves ../output/p3_convergence.json.
"""
import json
import numpy as np
from scipy.optimize import minimize

from data import load_runs, family_folds
from predictors import IrtPredictor, _sigmoid, _bce, math_logit
from run_cv import scores_fold, SEED


def stage1_obj(theta, y, w, j, i, n_ai):
    alpha = theta[:n_ai]
    beta = np.exp(theta[n_ai])
    phi = theta[n_ai + 1:]
    th = phi - phi.mean()
    p = _sigmoid(alpha[j] - beta * th[i])
    return _bce(y, p, w)


def stage1_obj_jac(theta, y, w, j, i, n_ai, n_task):
    alpha = theta[:n_ai]
    beta = np.exp(theta[n_ai])
    phi = theta[n_ai + 1:]
    th = phi - phi.mean()
    z = alpha[j] - beta * th[i]
    p = _sigmoid(z)
    val = _bce(y, p, w)
    g = w * (p - y)  # d obj / d z_r
    grad = np.zeros_like(theta)
    np.add.at(grad, j, g)  # alpha block: index j in [0, n_ai)
    grad[n_ai] = -beta * np.sum(g * th[i])  # log-beta
    per_task = np.zeros(n_task)
    np.add.at(per_task, i, g)
    grad[n_ai + 1:] = -beta * per_task + (beta / n_task) * np.sum(g)
    return val, grad


def identified(fit, x_te, y_te, w_te, j_te):
    """Identified quantities of a fitted IrtPredictor + held-out scores."""
    th_hat = fit.a_ + fit.b_ * x_te
    span = fit.beta_ * (th_hat.max() - th_hat.min())
    p = fit.predict(x_te, j_te)
    s = scores_fold(y_te, p, w_te)
    return {
        "beta": fit.beta_, "ols_a": fit.a_, "ols_b": fit.b_,
        "eff_slope_beta_b": fit.beta_ * fit.b_,
        "implied_logit_span_test": span,
        "frac_extreme": float(np.mean((p < 0.01) | (p > 0.99))),
        "logloss": s["logloss"], "brier": s["brier"],
    }, p


def main():
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    cv = json.load(open("../output/cv_scores.json"))
    out = {"seed": SEED, "folds": []}
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        tr = tr.copy(); te = te.copy()
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr["ti"] = tr["i"].map(tmap)
        n_task = len(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(dtype=float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy()
        i_tr = tr["ti"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(dtype=float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()

        # --- locked fit via the deposited predictor (anchor) ---
        locked = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, n_task)
        id_locked, p_locked = identified(locked, x_te, y_te, w_te, j_te)
        anchor = cv["folds"][k]["P3"]["logloss"]
        assert abs(id_locked["logloss"] - anchor) < 1e-9, (k, id_locked, anchor)

        # --- replicate the raw parameter vector for gradient work ---
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
        # the replicated solution must give the same identified quantities
        phi = res.x[n_ai + 1:]
        theta_c = phi - phi.mean()
        task_x = np.array([x_tr[i_tr == t][0] for t in range(n_task)])
        X = np.column_stack([np.ones(n_task), task_x])
        coef, *_ = np.linalg.lstsq(X, theta_c, rcond=None)
        beta_rep = np.exp(res.x[n_ai])
        assert abs(beta_rep * coef[1] - id_locked["eff_slope_beta_b"]) < 1e-6

        # --- analytic gradient at the locked solution ---
        obj_val, grad = stage1_obj_jac(res.x, y_tr, w_tr, j_tr, i_tr,
                                       n_ai, n_task)
        grad_inf = float(np.max(np.abs(grad)))

        # --- tightened refit, warm-started, analytic Jacobian ---
        res2 = minimize(stage1_obj_jac, res.x,
                        args=(y_tr, w_tr, j_tr, i_tr, n_ai, n_task),
                        method="L-BFGS-B", jac=True,
                        options={"maxiter": 20000, "maxfun": 200000,
                                 "ftol": 1e-13, "gtol": 1e-10})
        phi2 = res2.x[n_ai + 1:]
        theta2 = phi2 - phi2.mean()
        coef2, *_ = np.linalg.lstsq(X, theta2, rcond=None)
        tight = IrtPredictor()
        tight.alpha_ = res2.x[:n_ai]
        tight.beta_ = float(np.exp(res2.x[n_ai]))
        tight.a_, tight.b_ = float(coef2[0]), float(coef2[1])
        id_tight, p_tight = identified(tight, x_te, y_te, w_te, j_te)
        _, grad2 = stage1_obj_jac(res2.x, y_tr, w_tr, j_tr, i_tr,
                                  n_ai, n_task)

        rec = {
            "fold": k,
            "locked": {
                "objective": float(obj_val),
                "grad_inf_norm": grad_inf,
                "nit": int(res.nit),
                **id_locked,
            },
            "tightened": {
                "objective": float(res2.fun),
                "grad_inf_norm": float(np.max(np.abs(grad2))),
                "nit": int(res2.nit),
                "success": bool(res2.success),
                **id_tight,
            },
            "delta": {
                "objective": float(res2.fun - obj_val),
                "eff_slope_beta_b": abs(id_tight["eff_slope_beta_b"]
                                        - id_locked["eff_slope_beta_b"]),
                "implied_logit_span_test": abs(
                    id_tight["implied_logit_span_test"]
                    - id_locked["implied_logit_span_test"]),
                "logloss": abs(id_tight["logloss"] - id_locked["logloss"]),
                "max_abs_dp_test": float(np.max(np.abs(p_tight - p_locked))),
            },
        }
        out["folds"].append(rec)
        print(f"fold {k}: |grad|inf locked={grad_inf:.3e} "
              f"tight={rec['tightened']['grad_inf_norm']:.3e} "
              f"dObj={rec['delta']['objective']:.3e} "
              f"dLogLoss={rec['delta']['logloss']:.3e} "
              f"dBetaB={rec['delta']['eff_slope_beta_b']:.3e} "
              f"dSpan={rec['delta']['implied_logit_span_test']:.3e} "
              f"maxDp={rec['delta']['max_abs_dp_test']:.3e}", flush=True)

    out["summary"] = {
        "max_grad_inf_locked": max(f["locked"]["grad_inf_norm"]
                                   for f in out["folds"]),
        "max_grad_inf_tightened": max(f["tightened"]["grad_inf_norm"]
                                      for f in out["folds"]),
        "max_delta_objective": max(abs(f["delta"]["objective"])
                                   for f in out["folds"]),
        "max_delta_eff_slope": max(f["delta"]["eff_slope_beta_b"]
                                   for f in out["folds"]),
        "max_delta_span": max(f["delta"]["implied_logit_span_test"]
                              for f in out["folds"]),
        "max_delta_logloss": max(f["delta"]["logloss"] for f in out["folds"]),
        "max_delta_p": max(f["delta"]["max_abs_dp_test"] for f in out["folds"]),
        "printed_ranges": {
            "d_eff_slope": [min(f["tightened"]["eff_slope_beta_b"]
                                - f["locked"]["eff_slope_beta_b"]
                                for f in out["folds"]),
                            max(f["tightened"]["eff_slope_beta_b"]
                                - f["locked"]["eff_slope_beta_b"]
                                for f in out["folds"])],
            "d_span": [min(f["tightened"]["implied_logit_span_test"]
                           - f["locked"]["implied_logit_span_test"]
                           for f in out["folds"]),
                       max(f["tightened"]["implied_logit_span_test"]
                           - f["locked"]["implied_logit_span_test"]
                           for f in out["folds"])],
            "d_logloss": [min(f["tightened"]["logloss"]
                              - f["locked"]["logloss"]
                              for f in out["folds"]),
                          max(f["tightened"]["logloss"]
                              - f["locked"]["logloss"]
                              for f in out["folds"])],
        },
    }
    with open("../output/p3_convergence.json", "w") as f:
        json.dump(out, f, indent=2)
    print("summary:", json.dumps(out["summary"], indent=1))
    print("saved ../output/p3_convergence.json")


if __name__ == "__main__":
    main()
