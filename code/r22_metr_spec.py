"""Round-22 REG-P1-6: refit of METR's ACTUAL horizon specification.

Source extraction (METR/eval-analysis-public @ 52cb829c7a2efb2d659285c4b1768d191d97f8d2,
cloned to ../../scratch-r22/metr-repo; the deposited data/runs.jsonl is
md5-identical to reports/time-horizon-1-1/data/raw/runs.jsonl at that commit):

* fig_params/figs.yaml, figs.wrangle_logistic.headline:
    {"weighting": "invsqrt_task_weight", "regularization": 1e-05,
     "exclude": [], "success_percents": [50, 80]}
* src/horizon/wrangle/logistic.py, run_logistic_regressions():
    "for agent, agent_runs in runs.groupby("agent", as_index=False):"
    "weights = agent_runs[wrangle_params["weighting"]].values"
    score_col defaults to "score_binarized".
* src/horizon/wrangle/logistic.py, agent_regression():
    "x = np.log2(x).reshape(-1, 1)"            (x = log2 human_minutes)
    "horizon = np.exp2(get_x_for_quantile(model, p / 100))"
* src/horizon/utils/logistic.py:
    "model = LogisticRegression(C=1 / regularization)"
    "model.fit(X, y, sample_weight=sample_weight)"
    get_x_for_quantile: "(np.log(quantile / (1 - quantile))
                          - model.intercept_[0]) / model.coef_[0][0]"
* src/horizon/compute_task_weights.py: invsqrt_task_weight is computed
  PER AGENT: (1 / n_runs_in_task) * (1 / sqrt(n_tasks_in_family)), then
  normalised to sum to 1 within the agent ("invsqrt_weight =
  invsqrt_weight / invsqrt_weight.sum()"). The weights stored in
  runs.jsonl already are these per-agent normalised weights (verified:
  ratio stored/recomputed = 1 +/- 1e-7 on sampled agents).

So METR's specification is: a SEPARATE sklearn LogisticRegression per AI
(own intercept AND own slope on log2 T), C = 1/1e-5 = 1e5, sample weights
= the per-agent normalised invsqrt weights. sklearn's L2 objective is
    C * sum_i s_i * bce_i + 0.5 * slope^2
(the intercept is NOT penalised), i.e. dividing by C:
    sum_i s_i * bce_i + (0.5 / C) * slope^2,   sum_i s_i = 1.
This script optimises exactly that objective (analytic gradient, tight
tolerances) and cross-checks against sklearn LogisticRegression itself
(default solver settings, and tightened) on the full data.

Parts:
(a) Full data: per-AI horizons; ratios vs METR's published values and vs
    this paper's spline refit (Table 8) and P0 control (Table 11);
    counterfactual intercept-only fit at the shared slope (slope-freedom
    test); Opus 4.6 diagnostics.
(b) Locked 5-fold family CV (seed 20261012): Pmetr as a predictive arm,
    paired fold differences vs P1 (deposited cv_scores.json) and vs P0
    (strengthen.json). Primary: METR weights recomputed within each
    training subset per agent (the weighting SCHEME applied to the data
    being fit; leak-free). Sensitivity: stored weights renormalised
    within the training subset per agent.

Saves ../output/r22_metr_spec.json. Post-hoc (round 22); not preregistered.
"""
import json
import time

import numpy as np
import yaml
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from data import load_runs, family_folds
from predictors import _sigmoid, math_logit
from run_cv import scores_fold, SEED

REG = 1e-5                    # METR headline regularization (figs.yaml)
C_SKL = 1.0 / REG             # sklearn C = 1e5
PEN = 0.5 / C_SKL             # coefficient on slope^2 in the mean-form objective
SHARED_BETA = 0.577858009657536   # P0 shared slope (output/r5_analyses.json)

PUB_MAP = {  # alias -> id in data/benchmark_results_1_1.yaml (from r5_analyses.py)
    "Claude 3 Opus (Inspect)": "claude_3_opus_inspect",
    "Claude 3.5 Sonnet (Old) (Inspect)": "claude_3_5_sonnet_20240620_inspect",
    "Claude 3.5 Sonnet (New) (Inspect)": "claude_3_5_sonnet_20241022_inspect",
    "Claude 3.7 Sonnet (Inspect)": "claude_3_7_sonnet_inspect",
    "Claude 4 Opus (Inspect)": "claude_4_opus_inspect",
    "Claude 4.1 Opus (Inspect)": "claude_4_1_opus_inspect",
    "Claude Opus 4.5 (Inspect)": "claude_opus_4_5_inspect",
    "Claude Opus 4.6 (Inspect)": "claude_opus_4_6_inspect",
    "GPT-4 0314": "gpt_4", "GPT-4 1106 (Inspect)": "gpt_4_1106_inspect",
    "GPT-4 Turbo (Inspect)": "gpt_4_turbo_inspect", "GPT-4o (Inspect)": "gpt_4o_inspect",
    "GPT-5 (Inspect)": "gpt_5_2025_08_07_inspect",
    "GPT-5.1-Codex-Max (Inspect)": "gpt_5_1_codex_max_inspect",
    "GPT-5.2": "gpt_5_2", "GPT-5.3-Codex": "gpt_5_3_codex",
    "Gemini 3 Pro": "gemini_3_pro", "o1 (Inspect)": "o1_inspect",
    "o1-preview": "o1_preview", "o3 (Inspect)": "o3_inspect",
}


def metr_weights(df_agent):
    """METR's invsqrt weights for one agent's run set, normalised to sum 1
    (compute_task_weights.compute_sample_weights, verbatim formula)."""
    n_runs_task = df_agent.groupby("task_id")["task_id"].transform("count")
    fam_sizes = (df_agent.drop_duplicates("task_id")
                 .groupby("family")["task_id"].count())
    raw = (1.0 / n_runs_task) * (1.0 / np.sqrt(df_agent["family"].map(fam_sizes)))
    return (raw / raw.sum()).to_numpy()


def fit_metr(x, y, wn, slope_fixed=None):
    """Fit logit p = a + b*x minimising sum(wn*bce) + PEN*b^2 (wn sums to 1).
    If slope_fixed is given, fit the intercept only at b = -slope_fixed
    (paper sign convention: logit = a - beta*x). Returns (a, b, success)."""
    ybar = float(np.sum(wn * y))
    a0 = math_logit(min(max(ybar, 1e-4), 1 - 1e-4))
    if slope_fixed is not None:
        b_fix = -float(slope_fixed)

        def og1(a):
            z = a[0] + b_fix * x
            val = float(np.sum(wn * (np.logaddexp(0.0, z) - y * z)))
            return val, np.array([np.sum(wn * (_sigmoid(z) - y))])

        r = minimize(og1, np.array([a0]), jac=True, method="L-BFGS-B",
                     options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000})
        return float(r.x[0]), b_fix, bool(r.success)

    def og(th):
        a, b = th
        z = a + b * x
        val = float(np.sum(wn * (np.logaddexp(0.0, z) - y * z))) + PEN * b * b
        g = wn * (_sigmoid(z) - y)
        return val, np.array([np.sum(g), np.sum(g * x) + 2 * PEN * b])

    r = minimize(og, np.array([a0, -SHARED_BETA]), jac=True, method="L-BFGS-B",
                 options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000})
    return float(r.x[0]), float(r.x[1]), bool(r.success)


def horizon_min(a, b):
    """50% horizon in minutes: 2^(-a/b) for b<0, else None."""
    if b >= 0:
        return None
    return float(2.0 ** (-a / b))


def mean_se(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def main():
    t_start = time.time()
    df = load_runs()
    n_ai = df["j"].nunique()
    ai_names = (df[["j", "alias"]].drop_duplicates()
                .sort_values("j")["alias"].tolist())
    desc = json.load(open("../output/descriptives.json"))
    p1_full = desc["horizons_mine_min"]
    r5 = json.load(open("../output/r5_analyses.json"))
    p0_full = {k: v["p0_horizon_min"] for k, v in r5["p0_control"]["per_ai"].items()}
    pub_yaml = yaml.safe_load(open("../data/benchmark_results_1_1.yaml"))["results"]
    pub = {a: float(pub_yaml[pid]["metrics"]["p50_horizon_length"]["estimate"])
           for a, pid in PUB_MAP.items()}

    out = {"config": {"regularization": REG, "sklearn_C": C_SKL, "seed": SEED,
                      "spec": "per-AI sklearn LogisticRegression(C=1e5), own "
                              "intercept+slope on log2 T, per-agent normalised "
                              "invsqrt weights, y=score_binarized"},
           "full_data": {}, "cv": {}}

    # ---------------- (a) full data ----------------
    rows = {}
    skl_dev_default, skl_dev_tight = [], []
    for jj, name in enumerate(ai_names):
        d = df[df["j"] == jj]
        x = d["x"].to_numpy(); y = d["y"].to_numpy(float)
        wn = d["w"].to_numpy(); wn = wn / wn.sum()  # stored, already per-agent normalised
        a, b, ok = fit_metr(x, y, wn)
        h = horizon_min(a, b)
        a_cf, _, _ = fit_metr(x, y, wn, slope_fixed=SHARED_BETA)
        h_cf = horizon_min(a_cf, -SHARED_BETA)
        # sklearn cross-checks (METR's own solver), defaults and tightened
        hs = {}
        for tag, kw in [("sklearn_default", {}),
                        ("sklearn_tight", {"tol": 1e-12, "max_iter": 10000})]:
            m = LogisticRegression(C=C_SKL, **kw)
            m.fit(x.reshape(-1, 1), y, sample_weight=wn)
            hs[tag] = horizon_min(float(m.intercept_[0]), float(m.coef_[0][0]))
            if h is not None and hs[tag]:
                dev = abs(np.log2(hs[tag] / h))
                (skl_dev_default if tag == "sklearn_default" else skl_dev_tight).append(dev)
        rows[name] = {
            "n_runs": int(len(d)), "intercept": a, "slope": b, "fit_ok": ok,
            "horizon_min": h,
            "horizon_sklearn_default_min": hs["sklearn_default"],
            "horizon_sklearn_tight_min": hs["sklearn_tight"],
            "metr_published_min": pub[name],
            "p1_spline_min": float(p1_full[name]),
            "p0_shared_slope_min": float(p0_full[name]),
            "ratio_spec_over_published": (h / pub[name]) if h else None,
            "ratio_spec_over_p1": (h / float(p1_full[name])) if h else None,
            "ratio_p0_over_published": float(p0_full[name]) / pub[name],
            "counterfactual_shared_slope_min": h_cf,
            "ratio_counterfactual_over_published": (h_cf / pub[name]) if h_cf else None,
        }
        print(f"full {name}: slope={b:+.3f} h={h if h is None else round(h,1)} "
              f"pub={pub[name]:.1f} ratio={(h / pub[name]) if h else float('nan'):.3f}",
              flush=True)
    out["full_data"]["per_ai"] = rows
    rs = np.array([v["ratio_spec_over_published"] for v in rows.values()])
    out["full_data"]["summary"] = {
        "ratio_spec_over_published_median": float(np.median(rs)),
        "ratio_spec_over_published_range": [float(rs.min()), float(rs.max())],
        "n_within_0.85_1.10": int(np.sum((rs >= 0.85) & (rs <= 1.10))),
        "max_abs_log2_dev_sklearn_default": float(np.max(skl_dev_default)),
        "max_abs_log2_dev_sklearn_tight": float(np.max(skl_dev_tight)),
    }

    # Opus 4.6 diagnostics: weighted empirical success by METR time buckets,
    # weight concentration, own-slope vs shared-slope counterfactual.
    name46 = "Claude Opus 4.6 (Inspect)"
    d = df[df["alias"] == name46]
    x = d["x"].to_numpy(); y = d["y"].to_numpy(float); wn = d["w"].to_numpy()
    wn = wn / wn.sum()
    buckets = [1, 4, 16, 64, 256, 960, 2880]
    emp = []
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        m = (2.0 ** x >= lo) & (2.0 ** x < hi)
        emp.append({"bucket_min": [lo, hi], "n_runs": int(m.sum()),
                    "weight_share": float(wn[m].sum()),
                    "weighted_success": (float(np.sum(wn[m] * y[m]) / wn[m].sum())
                                         if m.sum() else None)})
    r46 = rows[name46]
    out["opus46"] = {
        "own_slope": r46["slope"], "shared_slope_beta": SHARED_BETA,
        "horizon_spec_min": r46["horizon_min"],
        "horizon_counterfactual_shared_slope_min": r46["counterfactual_shared_slope_min"],
        "horizon_p0_min": r46["p0_shared_slope_min"],
        "metr_published_min": r46["metr_published_min"],
        "ratio_spec_over_published": r46["ratio_spec_over_published"],
        "ratio_counterfactual_over_published": r46["ratio_counterfactual_over_published"],
        "n_runs": r46["n_runs"],
        "weight_share_above_256min": float(wn[2.0 ** x >= 256].sum()),
        "weight_share_above_960min": float(wn[2.0 ** x >= 960].sum()),
        "empirical_buckets": emp,
    }
    print("Opus 4.6:", json.dumps(out["opus46"], indent=1)[:800], flush=True)

    # ---------------- (b) locked 5-fold CV ----------------
    foldmap = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(foldmap)
    cv_locked = json.load(open("../output/cv_scores.json"))["folds"]
    p0_cv = json.load(open("../output/strengthen.json"))["p0_logloss_per_fold_sqrtweight"]
    p0_fold_horiz = np.load("../output/r5_p0_fold_horizons.npy")  # (5, n_ai)?
    p1_fold_horiz = json.load(open("../output/strengthen.json"))["horizon_folds_min"]

    def run_cv_arm(weight_mode):
        folds_out = []
        for k in range(5):
            tr = df[df["fold"] != k]; te = df[df["fold"] == k]
            x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float)
            w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()
            p = np.zeros(len(te))
            h_folds = {}
            for jj, name in enumerate(ai_names):
                dtr = tr[tr["j"] == jj]
                if weight_mode == "recomputed":
                    wn_tr = metr_weights(dtr)
                else:  # stored weights renormalised within train per agent
                    wn_tr = dtr["w"].to_numpy(); wn_tr = wn_tr / wn_tr.sum()
                a, b, ok = fit_metr(dtr["x"].to_numpy(),
                                    dtr["y"].to_numpy(float), wn_tr)
                m_te = j_te == jj
                p[m_te] = _sigmoid(a + b * x_te[m_te])
                h_folds[name] = horizon_min(a, b)
            s = scores_fold(y_te, p, w_te)
            rec = {"fold": k, "Pmetr": s,
                   "P1_logloss_deposited": cv_locked[k]["P1"]["logloss"],
                   "P0_logloss_deposited": p0_cv[k],
                   "diff_metr_minus_p1_logloss": s["logloss"] - cv_locked[k]["P1"]["logloss"],
                   "diff_metr_minus_p0_logloss": s["logloss"] - p0_cv[k],
                   "opus46_horizon_min": h_folds[name46],
                   "opus46_p1_fold_horizon_min": p1_fold_horiz["Claude Opus 4.6 (Inspect)"][k]
                   if "Claude Opus 4.6 (Inspect)" in p1_fold_horiz else None}
            folds_out.append(rec)
            print(f"cv[{weight_mode}] fold {k}: Pmetr ll={s['logloss']:.4f} "
                  f"d_vs_P1={rec['diff_metr_minus_p1_logloss']:+.4f} "
                  f"d_vs_P0={rec['diff_metr_minus_p0_logloss']:+.4f}", flush=True)
        d1 = np.array([f["diff_metr_minus_p1_logloss"] for f in folds_out])
        d0 = np.array([f["diff_metr_minus_p0_logloss"] for f in folds_out])
        m1, s1 = mean_se(d1); m0, s0 = mean_se(d0)
        return {"folds": folds_out,
                "diff_vs_p1": {"per_fold": d1.tolist(), "mean": m1, "se": s1,
                               "locked_rule_pass_metr_better": bool(m1 < 0 and abs(m1) > 2 * s1)},
                "diff_vs_p0": {"per_fold": d0.tolist(), "mean": m0, "se": s0}}

    out["cv"]["primary_weights_recomputed"] = run_cv_arm("recomputed")
    out["cv"]["sensitivity_weights_stored"] = run_cv_arm("stored")
    out["cv"]["p0_fold_horizons_npy_shape"] = list(p0_fold_horiz.shape)
    out["runtime_s"] = time.time() - t_start

    with open("../output/r22_metr_spec.json", "w") as f:
        json.dump(out, f, indent=2)
    print("summary full:", json.dumps(out["full_data"]["summary"], indent=1))
    print(f"saved ../output/r22_metr_spec.json in {out['runtime_s']}")


if __name__ == "__main__":
    main()