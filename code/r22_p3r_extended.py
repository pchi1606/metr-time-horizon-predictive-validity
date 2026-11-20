"""Round-23 STEP 0 (grid extension): P3r with the penalty grid extended
DOWNWARD -- {0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3}.

Identical protocol to r22_p3r.py in every other respect: identification
exactly as r11_stage1_ridge.py (stage-1 objective = weighted BCE +
lambda * sum((beta*theta_i)^2) on the identified products; stage 2 =
the locked OLS projection); lambda selected EXCLUSIVELY within each
outer fold's training families (inner 5-fold family CV, seed 20261012,
selection by mean inner held-out weighted log-loss); outer scoring
under the locked protocol (scores_fold) against the DEPOSITED P1 fold
scores; warm-started tightened-tolerance refit at the selected lambda
for stopping sensitivity; locked-arm stopping sensitivity recomputed
from the deposited output/p3_convergence.json.

Purpose: the round-22 selection landed on lambda = 0.003, the smallest
nonzero point of r11's deposited grid, in all five folds. This run
probes below that boundary. It is a post-hoc methods refinement of a
post-hoc arm and is disclosed as such in the paper (Appendix D).

Helpers are imported from r22_p3r / r11_stage1_ridge so the fitting
code path is byte-identical; only the GRID and the output path differ.
Saves ../output/r22_p3r_extended.json.
"""
import json
import time

import numpy as np

from data import load_runs, family_folds
from run_cv import scores_fold, SEED
from r11_stage1_ridge import fit_irt_ident_ridge
from r22_p3r import predict_from_fit, tightened_refit, mean_se

GRID = [0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


def main():
    t_start = time.time()
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    cv = json.load(open("../output/cv_scores.json"))
    p1_ll = np.array([f["P1"]["logloss"] for f in cv["folds"]])
    p1_br = np.array([f["P1"]["brier"] for f in cv["folds"]])
    p1_auc = np.array([f["P1"]["auc"] for f in cv["folds"]])

    out = {"grid": GRID, "seed": SEED, "folds": [],
           "grid_extension": "r22 grid {0, 0.003, 0.01, 0.03, 0.1, 0.3} "
                             "extended downward with 0.0003 and 0.001; "
                             "protocol otherwise identical to r22_p3r.py"}
    sel_lams, p3r_ll, p3r_br, p3r_auc = [], [], [], []
    tight_ll = []
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr["ti"] = tr["i"].map(tmap)
        n_task = len(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()

        # ---- inner 5-fold family CV over the TRAINING families only ----
        inner_map = family_folds(tr, n_folds=5, seed=SEED)
        tr = tr.copy(); tr["ifold"] = tr["f"].map(inner_map)
        inner_ll = {lam: [] for lam in GRID}
        for m in range(5):
            itr = tr[tr["ifold"] != m].reset_index(drop=True)
            ite = tr[tr["ifold"] == m].reset_index(drop=True)
            imap = {t: n for n, t in enumerate(sorted(itr["i"].unique()))}
            itr["ti"] = itr["i"].map(imap)
            xi = itr["x"].to_numpy(); yi = itr["y"].to_numpy(float)
            wi = itr["w"].to_numpy(); ji = itr["j"].to_numpy(); ii = itr["ti"].to_numpy()
            xe = ite["x"].to_numpy(); ye = ite["y"].to_numpy(float)
            we = ite["w"].to_numpy(); je = ite["j"].to_numpy()
            for lam in GRID:
                fit = fit_irt_ident_ridge(xi, yi, wi, ji, ii, n_ai, len(imap), lam=lam)
                s = scores_fold(ye, predict_from_fit(fit, xe, je), we)
                inner_ll[lam].append(s["logloss"])
        inner_mean = {lam: float(np.mean(v)) for lam, v in inner_ll.items()}
        lam_star = min(GRID, key=lambda lam: inner_mean[lam])
        sel_lams.append(lam_star)

        # ---- outer fits: selected lambda + full grid + tightened refit ----
        rec = {"fold": k, "selected_lambda": lam_star,
               "inner_mean_logloss": {str(lam): inner_mean[lam] for lam in GRID},
               "inner_per_fold_logloss": {str(lam): inner_ll[lam] for lam in GRID},
               "outer_by_lambda": {}}
        outer_ll = {}
        fit_star = None
        for lam in GRID:
            fit = fit_irt_ident_ridge(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, n_task, lam=lam)
            s = scores_fold(y_te, predict_from_fit(fit, x_te, j_te), w_te)
            outer_ll[lam] = s["logloss"]
            rec["outer_by_lambda"][str(lam)] = s
            if lam == lam_star:
                fit_star = fit
                rec["P3r"] = s
        p3r_ll.append(rec["P3r"]["logloss"]); p3r_br.append(rec["P3r"]["brier"])
        p3r_auc.append(rec["P3r"]["auc"])
        rec["oracle_lambda"] = min(GRID, key=lambda lam: outer_ll[lam])

        # stopping sensitivity of the new arm at the selected lambda
        fit_t = tightened_refit(fit_star, x_tr, y_tr, w_tr, j_tr, i_tr,
                                n_ai, n_task, lam_star)
        s_t = scores_fold(y_te, predict_from_fit(fit_t, x_te, j_te), w_te)
        tight_ll.append(s_t["logloss"])
        rec["tightened"] = {"scores": s_t,
                            "drift_logloss": s_t["logloss"] - rec["P3r"]["logloss"],
                            "drift_eff_slope": abs(fit_t["beta"] * fit_t["b"]
                                                   - fit_star["beta"] * fit_star["b"]),
                            "success": fit_t["success"]}
        rec["selected_eff_slope"] = float(fit_star["beta"] * fit_star["b"])
        out["folds"].append(rec)
        print(f"fold {k}: lambda*={lam_star:g} (oracle {rec['oracle_lambda']:g}) "
              f"P3r ll={rec['P3r']['logloss']:.4f} "
              f"P1-P3r={p1_ll[k] - rec['P3r']['logloss']:+.4f} "
              f"tight drift={rec['tightened']['drift_logloss']:+.2e}", flush=True)

    p3r_ll = np.array(p3r_ll); tight_ll = np.array(tight_ll)
    d = p1_ll - p3r_ll
    m, se = mean_se(d)
    d_t = p1_ll - tight_ll
    mt, set_ = mean_se(d_t)
    db, seb = mean_se(p1_br - np.array(p3r_br))
    da, sea = mean_se(p1_auc - np.array(p3r_auc))
    out["paired_p1_minus_p3r"] = {
        "logloss": {"per_fold": d.tolist(), "mean": m, "se": se,
                    "locked_rule_pass": bool(abs(m) > 2 * se)},
        "brier": {"per_fold": (p1_br - np.array(p3r_br)).tolist(), "mean": db, "se": seb},
        "auc": {"per_fold": (p1_auc - np.array(p3r_auc)).tolist(), "mean": da, "se": sea},
    }
    out["stopping_sensitivity_new_arm"] = {
        "paired_margin_standard": {"mean": m, "se": se},
        "paired_margin_tightened": {"per_fold": d_t.tolist(), "mean": mt, "se": set_},
        "max_abs_drift_logloss": float(np.max(np.abs(tight_ll - p3r_ll))),
    }
    out["selected_lambda_by_fold"] = sel_lams

    # ---- locked arm stopping sensitivity, from deposited p3_convergence ----
    conv = json.load(open("../output/p3_convergence.json"))
    d_lock = np.array([p1_ll[k] - conv["folds"][k]["locked"]["logloss"] for k in range(5)])
    d_lock_t = np.array([p1_ll[k] - conv["folds"][k]["tightened"]["logloss"] for k in range(5)])
    ml, sel_ = mean_se(d_lock); mlt, selt = mean_se(d_lock_t)
    out["stopping_sensitivity_locked_arm_from_p3_convergence"] = {
        "paired_margin_locked_tolerances": {"per_fold": d_lock.tolist(), "mean": ml, "se": sel_},
        "paired_margin_tightened": {"per_fold": d_lock_t.tolist(), "mean": mlt, "se": selt},
        "source": "output/p3_convergence.json (deposited) + output/cv_scores.json",
    }
    out["runtime_s"] = time.time() - t_start
    with open("../output/r22_p3r_extended.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"P1-P3r logloss (extended grid): {m:+.4f} +/- {se:.4f} "
          f"(rule pass: {abs(m) > 2 * se}); selected lambdas: {sel_lams}")
    print(f"tightened margin: {mt:+.4f} +/- {set_:.4f}; "
          f"locked arm: {ml:+.4f} -> {mlt:+.4f} under tightening")
    print(f"saved ../output/r22_p3r_extended.json in {out['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
