"""Round-30 fix F-13 (adjudication addendum, Perplexity F10;
register REG-P1-12): P1 optimization sensitivity.

Appendix D audits P3's fit exhaustively and P1's not at all. This
script runs the missing audit on the locked five folds, on the
predictors.py lineage (the fit is re-implemented here only to
expose optimizer controls; the locked arm is the deposited
SplineHorizon itself, anchored against output/cv_scores.json):

  locked    the deposited fit (default L-BFGS-B tolerances,
            identity-map initialisation, finite-difference
            gradients);
  tightened warm-started from the locked solution at
            ftol 1e-13 / gtol 1e-10 (finite-difference, as locked);
  init-half / init-double / init-flat
            cold refits from perturbed initialisations of the
            exp-gap vector u (x0.5, x2, equal gaps), alphas and
            log-beta initialised as locked;
  bounded   cold refit with the exp-gap parameter clipped at
            u <= 700 inside the objective (the Appendix D
            reproduction guidance, tested -- fix F-21).

Per fold and variant: held-out log-loss / Brier / AUC
(scores_fold), the paired H1/H2 margins against the deposited
P2/P3 fold scores, and the per-AI 50% horizon crossings.
Saves ../output/r30_p1_sensitivity.json.
"""
import json
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, brentq

from data import load_runs, family_folds
from predictors import SplineHorizon, _sigmoid, _bce, math_logit
from run_cv import scores_fold, SEED

AI_NAMES = ["Claude 3 Opus", "Claude 3.5 Sonnet (New)",
            "Claude 3.5 Sonnet (Old)", "Claude 3.7 Sonnet",
            "Claude 4 Opus", "Claude 4.1 Opus", "Claude Opus 4.5",
            "Claude Opus 4.6", "GPT-4 0314", "GPT-4 1106",
            "GPT-4 Turbo", "GPT-4o", "GPT-5", "GPT-5.1-Codex-Max",
            "GPT-5.2", "GPT-5.3-Codex", "Gemini 3 Pro", "o1",
            "o1-preview", "o3"]


def make_f(knots, u, bound=None):
    if bound is not None:
        u = np.minimum(u, bound)
    gaps = np.exp(u)
    frac = np.cumsum(gaps) / np.sum(gaps)
    xmin, xmax = knots[0], knots[-1]
    values = xmin + (xmax - xmin) * np.concatenate([[0.0], frac[:-1], [1.0]])
    return PchipInterpolator(knots, values, extrapolate=True), values


def fit_p1(x, y, w, j, n_ai, theta0=None, u0_kind="identity",
           ftol=None, gtol=None, bound=None):
    xs = np.asarray(x, float)
    xmin, xmax = xs.min(), xs.max()
    qs = np.quantile(xs, [0.2, 0.4, 0.6, 0.8])
    knots = np.concatenate([[xmin], qs, [xmax]])
    for k in range(1, len(knots)):
        if knots[k] <= knots[k - 1]:
            knots[k] = knots[k - 1] + 1e-9
    if theta0 is None:
        if u0_kind == "identity":
            u0 = np.log(np.diff(knots))
        elif u0_kind == "half":
            u0 = 0.5 * np.log(np.diff(knots))
        elif u0_kind == "double":
            u0 = 2.0 * np.log(np.diff(knots))
        elif u0_kind == "flat":
            u0 = np.zeros(len(knots) - 1)
        else:
            raise ValueError(u0_kind)
        a0 = np.zeros(n_ai)
        for jj in range(n_ai):
            m = (j == jj)
            pj = np.average(y[m], weights=w[m])
            a0[jj] = math_logit(min(max(pj, 1e-3), 1 - 1e-3))
        theta0 = np.concatenate([a0, [0.0], u0])
    opts = {"maxiter": 20000, "maxfun": 200000}
    if ftol is not None:
        opts["ftol"] = ftol
    if gtol is not None:
        opts["gtol"] = gtol

    def obj(theta):
        alpha = theta[:n_ai]
        beta = np.exp(theta[n_ai])
        f, _ = make_f(knots, theta[n_ai + 1:], bound=bound)
        p = _sigmoid(alpha[j] - beta * f(xs))
        return _bce(y, p, w)

    res = minimize(obj, theta0, method="L-BFGS-B", options=opts)
    alpha = res.x[:n_ai]
    beta = float(np.exp(res.x[n_ai]))
    f, _ = make_f(knots, res.x[n_ai + 1:], bound=bound)
    return {"theta": res.x, "alpha": alpha, "beta": beta, "f": f,
            "knots": knots, "success": bool(res.success),
            "nit": int(res.nit), "objective": float(res.fun),
            "max_u": float(np.max(res.x[n_ai + 1:]))}


def predict(fit, x, j):
    return _sigmoid(fit["alpha"][j] - fit["beta"] * fit["f"](np.asarray(x, float)))


def horizon(fit, jj):
    xmin, xmax = fit["knots"][0], fit["knots"][-1]
    rhs = fit["alpha"][jj] / fit["beta"]
    if rhs <= fit["f"](xmin):
        return float(2.0 ** xmin)
    if rhs >= fit["f"](xmax):
        return float(2.0 ** xmax)
    return float(2.0 ** brentq(lambda t: fit["f"](t) - rhs, xmin, xmax))


def main():
    t_start = time.time()
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    cv = json.load(open("../output/cv_scores.json"))
    variants = ["locked", "tightened", "init-half", "init-double",
                "init-flat", "bounded"]
    out = {"seed": SEED, "variants": variants, "folds": []}
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(dtype=float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(dtype=float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()

        # locked arm: the deposited predictor itself
        dep = SplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai)
        s_dep = scores_fold(y_te, dep.predict(x_te, j_te), w_te)
        anchor = cv["folds"][k]["P1"]
        assert abs(s_dep["logloss"] - anchor["logloss"]) < 1e-9, (k, s_dep, anchor)
        # The exp-gap u is recoverable only up to its normalised gaps:
        # gaps proportional to diffs of the knot values. Rebuild an
        # exactly-equivalent u from the knot values (the objective
        # depends on u only through the normalised cumulative sums).
        vals = dep.f_(dep.knots_)
        gaps = np.diff(vals)
        u_locked = np.log(gaps / gaps.sum())
        theta_locked = np.concatenate([dep.alpha_, [np.log(dep.beta_)],
                                       u_locked])

        fits = {}
        fits["locked"] = {"theta": theta_locked, "alpha": dep.alpha_,
                          "beta": float(dep.beta_), "f": dep.f_,
                          "knots": dep.knots_, "success": True,
                          "objective": None, "max_u": float(np.max(u_locked))}
        fits["tightened"] = fit_p1(x_tr, y_tr, w_tr, j_tr, n_ai,
                                   theta0=theta_locked,
                                   ftol=1e-13, gtol=1e-10)
        fits["init-half"] = fit_p1(x_tr, y_tr, w_tr, j_tr, n_ai,
                                   u0_kind="half")
        fits["init-double"] = fit_p1(x_tr, y_tr, w_tr, j_tr, n_ai,
                                     u0_kind="double")
        fits["init-flat"] = fit_p1(x_tr, y_tr, w_tr, j_tr, n_ai,
                                   u0_kind="flat")
        fits["bounded"] = fit_p1(x_tr, y_tr, w_tr, j_tr, n_ai,
                                 bound=700.0)

        rec = {"fold": k, "variants": {}}
        h_lock = [horizon(fits["locked"], jj) for jj in range(n_ai)]
        for v in variants:
            ft = fits[v]
            s = scores_fold(y_te, predict(ft, x_te, j_te), w_te)
            hs = [horizon(ft, jj) for jj in range(n_ai)]
            rec["variants"][v] = {
                "scores": s, "success": ft["success"],
                "objective": ft["objective"],
                "max_u": ft["max_u"],
                "horizons_min": hs,
                "d_logloss_vs_locked": s["logloss"] - s_dep["logloss"],
                "d_brier_vs_locked": s["brier"] - s_dep["brier"],
                "d_auc_vs_locked": s["auc"] - s_dep["auc"],
            }
        out["folds"].append(rec)
        msg = " ".join(f"{v}: dll={rec['variants'][v]['d_logloss_vs_locked']:+.2e}"
                       for v in variants[1:])
        print(f"fold {k} ({time.time() - t_start:.0f}s): {msg}", flush=True)

    # ---- summaries -------------------------------------------------
    p2_ll = np.array([f["P2"]["logloss"] for f in cv["folds"]])
    p3_ll = np.array([f["P3"]["logloss"] for f in cv["folds"]])

    def mean_se(v):
        v = np.asarray(v, float)
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))

    summ = {}
    for v in variants:
        dll = [f["variants"][v]["d_logloss_vs_locked"] for f in out["folds"]]
        ll = np.array([f["variants"][v]["scores"]["logloss"]
                       for f in out["folds"]])
        h1 = mean_se(ll - p2_ll)
        h2 = mean_se(ll - p3_ll)
        # horizon movement vs locked, over all (fold, AI) cells
        ratios, absdiff = [], []
        for f in out["folds"]:
            hl = f["variants"]["locked"]["horizons_min"]
            hv = f["variants"][v]["horizons_min"]
            for a, b in zip(hl, hv):
                ratios.append(b / a)
                absdiff.append(abs(b - a))
        ratios = np.array(ratios); absdiff = np.array(absdiff)
        summ[v] = {
            "d_logloss_per_fold": dll,
            "max_abs_d_logloss": float(np.max(np.abs(dll))),
            "max_abs_d_brier": float(max(
                abs(f["variants"][v]["d_brier_vs_locked"])
                for f in out["folds"])),
            "max_abs_d_auc": float(max(
                abs(f["variants"][v]["d_auc_vs_locked"])
                for f in out["folds"])),
            "H1_margin": {"mean": h1[0], "se": h1[1]},
            "H2_margin": {"mean": h2[0], "se": h2[1]},
            "horizon_ratio_max_dev": float(np.max(np.abs(ratios - 1.0))),
            "horizon_ratio_range": [float(ratios.min()), float(ratios.max())],
            "horizon_cells_moving_gt5pct": int(np.sum(np.abs(ratios - 1) > 0.05)),
            "horizon_cells_total": int(len(ratios)),
            "horizon_max_abs_diff_min": float(absdiff.max()),
            "all_success": bool(all(f["variants"][v]["success"]
                                    for f in out["folds"])),
        }
    out["summary"] = summ
    with open("../output/r30_p1_sensitivity.json", "w") as fh:
        json.dump(out, fh, indent=2)
    for v in variants:
        s = summ[v]
        print(f"{v}: max|dll|={s['max_abs_d_logloss']:.2e} "
              f"H1 {s['H1_margin']['mean']:+.3f} H2 {s['H2_margin']['mean']:+.3f} "
              f"horizon max ratio dev {s['horizon_ratio_max_dev']:.3f} "
              f"cells>5%: {s['horizon_cells_moving_gt5pct']}/100 "
              f"success all: {s['all_success']}")
    print(f"saved ../output/r30_p1_sensitivity.json in "
          f"{time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
