"""R5: round-5 panel-response analyses. All post-hoc, on existing outputs/data.

1. LOFO (leave-one-fold-out) paired differences for H1/H2 from
   output/cv_scores.json (no refit), with SEs and locked-rule re-evaluation;
   fold-4 share of H2's total margin.
2. Refit scaffold identical to strengthen.py / r4_ece.py (same locked fits
   and seed), computing:
   - EW10 reliability-bin weight shares per predictor (share of total
     sqrt-family weight in each bin), so the printed ECE follows from the
     printed tables: ECE = sum_b share_b * |mean_obs_b - mean_pred_b|.
   - Pooled (run-level, sqrt-weighted) out-of-fold log-loss per predictor.
   - Per-AI per-fold paired log-loss differences -> mean +/- SE per AI
     (same estimand as strengthen.per_ai_logloss_diff, plus SEs).
3. Same-specification control (skeptic A7): P0 (shared-slope logistic, the
   paper pipeline's own implementation of METR's specification class) fit
   on ALL data; per-AI 50% horizons with the same range convention as P1
   (crossing clipped to the fitted log2-time range); ratios vs METR's
   published horizons for the 20 paper AIs.
"""
import json
import numpy as np
from data import load_runs, family_folds
from predictors import SplineHorizon, MeanPredictor, IrtPredictor, _sigmoid, EPS
from run_cv import SEED
from strengthen import fit_p0, ece_and_bins

OUT = {}


def mean_se(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


# ---------- 1. LOFO from cv_scores.json ----------
cv = json.load(open("../output/cv_scores.json"))
folds = cv["folds"]
d12 = np.array([f["P1"]["logloss"] - f["P2"]["logloss"] for f in folds])
d13 = np.array([f["P1"]["logloss"] - f["P3"]["logloss"] for f in folds])
lofo = {}
for name, d in [("H1", d12), ("H2", d13)]:
    rows = {}
    for k in range(5):
        rest = np.delete(d, k)
        m, se = mean_se(rest)
        rows[f"excl_fold_{k}"] = {"mean": m, "se": se,
                                 "rule_pass": bool(abs(m) > 2 * se)}
    lofo[name] = {"per_fold_diffs": [float(x) for x in d], "lofo": rows}
lofo["H2_fold4_share_of_margin"] = float(abs(d13[4]) / np.sum(np.abs(d13)))
lofo["H1_fold3_diff_unrounded"] = float(d12[3])
OUT["lofo"] = lofo
print("LOFO:", json.dumps(lofo, indent=1), flush=True)

# ---------- 2. Refit scaffold ----------
df = load_runs()
foldmap = family_folds(df, seed=SEED)
df["fold"] = df["f"].map(foldmap)
n_ai = df["j"].nunique()
ai_names = (df[["j", "alias"]].drop_duplicates()
            .sort_values("j")["alias"].tolist())
oof = {m: [] for m in ["P1", "P2", "P3", "P0"]}
yy, ww, jj_all = [], [], []
per_ai = {m: np.zeros((5, n_ai)) for m in ["P1", "P2", "P3"]}
per_ai_n = {m: np.zeros((5, n_ai), dtype=int) for m in ["P1", "P2", "P3"]}
for k in range(5):
    tr = df[df["fold"] != k].reset_index(drop=True)
    te = df[df["fold"] == k].reset_index(drop=True)
    tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
    tr["ti"] = tr["i"].map(tmap)
    x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float)
    w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
    x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float)
    w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()
    p1 = SplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai)
    p2 = MeanPredictor().fit(y_tr, w_tr, j_tr, n_ai)
    p3 = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, len(tmap))
    th0 = fit_p0(x_tr, y_tr, w_tr, j_tr, n_ai)
    preds = {"P1": p1.predict(x_te, j_te), "P2": p2.predict(j_te),
             "P3": p3.predict(x_te, j_te),
             "P0": _sigmoid(th0[j_te] - np.exp(th0[n_ai]) * x_te)}
    for m, p in preds.items():
        oof[m].append(p)
    yy.append(y_te); ww.append(w_te); jj_all.append(j_te)
    for m in ["P1", "P2", "P3"]:
        pc = np.clip(preds[m], EPS, 1 - EPS)
        for jj in range(n_ai):
            mj = j_te == jj
            per_ai_n[m][k, jj] = int(mj.sum())
            if mj.sum():
                per_ai[m][k, jj] = float(
                    -np.sum(w_te[mj] * (y_te[mj] * np.log(pc[mj]) +
                                        (1 - y_te[mj]) * np.log(1 - pc[mj])))
                    / np.sum(w_te[mj]))
    print(f"fold {k} done", flush=True)

y = np.concatenate(yy); w = np.concatenate(ww)

# ECE bin weight shares (EW10, identical binning to strengthen.ece_and_bins)
shares = {}
for m in ["P1", "P2", "P3", "P0"]:
    p = np.concatenate(oof[m])
    e, bins = ece_and_bins(y, p, w, n_bins=10)
    order = np.argsort(p)
    ps, ws = p[order], w[order]
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(ps, edges) - 1, 0, 9)
    sh = []
    for b in range(10):
        mb = idx == b
        sh.append(float(ws[mb].sum() / ws.sum()) if mb.sum() else 0.0)
    ece_from_table = float(sum(
        s * abs(bn["mean_obs"] - bn["mean_pred"])
        for s, bn in zip(sh, bins) if bn.get("n", 0) > 0))
    shares[m] = {"bin_weight_share": sh, "ece_direct": e,
                 "ece_from_printed_table": ece_from_table,
                 "bins_n": [bn.get("n", 0) for bn in bins]}
    print(m, "ECE direct", round(e, 4), "from table", round(ece_from_table, 4),
          "shares", [round(s, 4) for s in sh], flush=True)
OUT["ece_bin_weight_shares"] = shares

# Pooled run-level sqrt-weighted log-loss
pooled = {}
for m in ["P1", "P2", "P3", "P0"]:
    p = np.clip(np.concatenate(oof[m]), EPS, 1 - EPS)
    pooled[m] = float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))
                      / np.sum(w))
OUT["pooled_logloss_sqrtweighted"] = pooled
print("pooled log-loss:", {k: round(v, 4) for k, v in pooled.items()}, flush=True)

# Per-AI paired differences with SEs (estimand = strengthen's: mean over
# folds of per-fold per-AI weighted log-loss differences)
perai = {}
for a, b in [("P1", "P2"), ("P1", "P3")]:
    d = per_ai[a] - per_ai[b]  # (5, n_ai); folds where AI absent have n=0
    entry = {}
    for jj, name in enumerate(ai_names):
        mask = per_ai_n[a][:, jj] > 0
        vals = d[mask, jj]
        m, se = mean_se(vals)
        entry[name] = {"mean": m, "se": se, "n_folds": int(mask.sum())}
    perai[f"{a}_vs_{b}"] = entry
OUT["per_ai_diff_with_se"] = perai
print("per-AI diffs with SE computed", flush=True)
for pair in ["P1_vs_P2", "P1_vs_P3"]:
    print(pair, {k.replace(" (Inspect)", ""): (round(v["mean"], 3), round(v["se"], 3), v["n_folds"])
                 for k, v in perai[pair].items()}, flush=True)
json.dump(OUT, open("../output/r5_analyses_partial.json", "w"), indent=1)

# ---------- 3. P0 same-specification control (full data) ----------
x_all = df["x"].to_numpy(); y_all = df["y"].to_numpy(float)
w_all = df["w"].to_numpy(); j_all = df["j"].to_numpy()
th0 = fit_p0(x_all, y_all, w_all, j_all, n_ai)
beta0 = float(np.exp(th0[n_ai]))
xmin, xmax = float(x_all.min()), float(x_all.max())
desc = json.load(open("../output/descriptives.json"))
mine = desc["horizons_mine_min"]
import yaml
pub_yaml = yaml.safe_load(open("../data/benchmark_results_1_1.yaml"))["results"]
PUB_MAP = {
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
pub = {a: float(pub_yaml[pid]["metrics"]["p50_horizon_length"]["estimate"])
       for a, pid in PUB_MAP.items()}
ctrl = {}
for jj, name in enumerate(ai_names):
    xs = th0[jj] / beta0
    xs_c = min(max(xs, xmin), xmax)
    h = float(2.0 ** xs_c)
    ctrl[name] = {"p0_horizon_min": h, "metr_published_min": float(pub[name]),
                  "ratio_p0_over_metr": h / float(pub[name]),
                  "ratio_p1_over_metr": float(mine[name]) / float(pub[name]),
                  "clipped": bool(xs != xs_c)}
OUT["p0_control"] = {"beta_shared_slope": beta0, "x_range_log2min": [xmin, xmax],
                     "per_ai": ctrl}
r0 = np.array([v["ratio_p0_over_metr"] for v in ctrl.values()])
r1 = np.array([v["ratio_p1_over_metr"] for v in ctrl.values()])
OUT["p0_control"]["summary"] = {
    "p0_ratio_min": float(r0.min()), "p0_ratio_max": float(r0.max()),
    "p0_ratio_median": float(np.median(r0)),
    "p1_ratio_min": float(r1.min()), "p1_ratio_max": float(r1.max()),
    "p1_ratio_median": float(np.median(r1))}
print("P0 control summary:", OUT["p0_control"]["summary"], flush=True)

json.dump(OUT, open("../output/r5_analyses.json", "w"), indent=1)
print("saved ../output/r5_analyses.json")
