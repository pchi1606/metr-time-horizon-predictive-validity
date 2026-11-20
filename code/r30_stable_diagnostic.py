"""Round-30 fix F-10 rider, second limb: the tightened convergence
diagnostic rerun with a numerically stable log-loss and its matching
derivative (triggered because r30_clip_binding.py found the clip
binding non-negligible: up to 10.8% of training weight at the locked
solutions, 10.6-14.7% at the tightened solutions).

The deposited diagnostic (p3_convergence.py) pairs the CLIPPED
objective value (_bce clips p to [1e-12, 1-1e-12]) with the analytic
gradient of the UNCLIPPED loss. This rerun pairs the unclipped loss,
evaluated stably as sum w * (logaddexp(0, z) - y*z), with its exact
analytic gradient (the same gradient formula the deposited
diagnostic uses -- it is the matching derivative of the unclipped
loss). The tightened refit is warm-started from the locked solution
at the same tightened tolerances (ftol 1e-13, gtol 1e-10).

Per fold it records: the clipped and stable objective values at the
locked solution; the stable objective and its gradient norm at the
stable-tightened solution; the objective improvement on the stable
objective; and the stable-tightened fit's identified quantities and
held-out scores, beside the deposited (clipped) tightened values
from output/p3_convergence.json. Saves
../output/r30_stable_diagnostic.json.
"""
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from data import load_runs, family_folds
from predictors import IrtPredictor, math_logit
from run_cv import SEED
from p3_convergence import stage1_obj, identified


def stable_obj_jac(theta, y, w, j, i, n_ai, n_task):
    alpha = theta[:n_ai]
    beta = np.exp(theta[n_ai])
    phi = theta[n_ai + 1:]
    th = phi - phi.mean()
    z = alpha[j] - beta * th[i]
    # unclipped BCE, stably: -[y log p + (1-y) log(1-p)]
    #                      = logaddexp(0, z) - y * z
    val = float(np.sum(w * (np.logaddexp(0.0, z) - y * z)))
    p = expit(z)
    g = w * (p - y)
    grad = np.zeros_like(theta)
    np.add.at(grad, j, g)
    grad[n_ai] = -beta * np.sum(g * th[i])
    per_task = np.zeros(n_task)
    np.add.at(per_task, i, g)
    grad[n_ai + 1:] = -beta * per_task + (beta / n_task) * np.sum(g)
    return val, grad


def main():
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    cv = json.load(open("../output/cv_scores.json"))
    conv = json.load(open("../output/p3_convergence.json"))
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

        locked = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr,
                                    n_ai, n_task)
        id_locked, _ = identified(locked, x_te, y_te, w_te, j_te)
        assert abs(id_locked["logloss"]
                   - cv["folds"][k]["P3"]["logloss"]) < 1e-9

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
        assert abs(res.fun - conv["folds"][k]["locked"]["objective"]) < 1e-6

        val_clip_lock = float(res.fun)
        val_stab_lock, grad_lock = stable_obj_jac(
            res.x, y_tr, w_tr, j_tr, i_tr, n_ai, n_task)
        grad_lock_inf = float(np.max(np.abs(grad_lock)))
        # the deposited diagnostic's gradient at locked is this same
        # unclipped gradient; anchor it
        assert abs(grad_lock_inf
                   - conv["folds"][k]["locked"]["grad_inf_norm"]) < 1e-9

        res2 = minimize(stable_obj_jac, res.x,
                        args=(y_tr, w_tr, j_tr, i_tr, n_ai, n_task),
                        method="L-BFGS-B", jac=True,
                        options={"maxiter": 20000, "maxfun": 200000,
                                 "ftol": 1e-13, "gtol": 1e-10})
        val_stab_tight, grad_tight = stable_obj_jac(
            res2.x, y_tr, w_tr, j_tr, i_tr, n_ai, n_task)
        phi2 = res2.x[n_ai + 1:]
        theta2 = phi2 - phi2.mean()
        task_x = np.array([x_tr[i_tr == t][0] for t in range(n_task)])
        X = np.column_stack([np.ones(n_task), task_x])
        coef2, *_ = np.linalg.lstsq(X, theta2, rcond=None)
        tight = IrtPredictor()
        tight.alpha_ = res2.x[:n_ai]
        tight.beta_ = float(np.exp(res2.x[n_ai]))
        tight.a_, tight.b_ = float(coef2[0]), float(coef2[1])
        id_stight, _ = identified(tight, x_te, y_te, w_te, j_te)

        dep_tight = conv["folds"][k]["tightened"]
        rec = {
            "fold": k,
            "locked": {"objective_clipped": val_clip_lock,
                       "objective_stable": float(val_stab_lock),
                       "grad_inf_norm_stable": grad_lock_inf},
            "stable_tightened": {
                "objective_stable": float(val_stab_tight),
                "grad_inf_norm_stable": float(np.max(np.abs(grad_tight))),
                "nit": int(res2.nit), "success": bool(res2.success),
                **id_stight},
            "improvement_stable_objective": float(val_stab_lock
                                                  - val_stab_tight),
            "deposited_clipped_tightened": {
                "objective": dep_tight["objective"],
                "eff_slope_beta_b": dep_tight["eff_slope_beta_b"],
                "implied_logit_span_test":
                    dep_tight["implied_logit_span_test"],
                "logloss": dep_tight["logloss"]},
        }
        out["folds"].append(rec)
        print(f"fold {k}: stable obj locked {val_stab_lock:.6f} "
              f"(clipped {val_clip_lock:.6f}) -> tightened "
              f"{val_stab_tight:.6f}; improvement "
              f"{rec['improvement_stable_objective']:.3e}; "
              f"|grad| tight {rec['stable_tightened']['grad_inf_norm_stable']:.3e}; "
              f"eff slope {id_stight['eff_slope_beta_b']:.2f} "
              f"(clipped-tight {dep_tight['eff_slope_beta_b']:.2f}); "
              f"held-out ll {id_stight['logloss']:.3f} "
              f"(clipped-tight {dep_tight['logloss']:.3f})", flush=True)

    out["summary"] = {
        "max_improvement_stable_objective": max(
            abs(f["improvement_stable_objective"]) for f in out["folds"]),
        "improvement_range": [
            min(f["improvement_stable_objective"] for f in out["folds"]),
            max(f["improvement_stable_objective"] for f in out["folds"])],
        "max_grad_inf_stable_tightened": max(
            f["stable_tightened"]["grad_inf_norm_stable"]
            for f in out["folds"]),
        "stable_tightened_logloss_range_vs_locked": [
            min(f["stable_tightened"]["logloss"] for f in out["folds"]),
            max(f["stable_tightened"]["logloss"] for f in out["folds"])],
    }
    with open("../output/r30_stable_diagnostic.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("summary:", json.dumps(out["summary"], indent=1))
    print("saved ../output/r30_stable_diagnostic.json")


if __name__ == "__main__":
    main()
