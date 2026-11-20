"""Clip sensitivity for H1/H2 + per-fold P1 horizons with censoring counts.
Reuses the repo's locked fold split and predictor fits; scores log-loss at
multiple clipping bounds from the SAME fitted predictions."""
import json, math
import numpy as np
from data import load_runs, family_folds
from predictors import SplineHorizon, MeanPredictor, IrtPredictor, EPS

CLIPS = [1e-2, 1e-4, 1e-8, 1e-12]

def wlogloss(y, p, w, clip):
    pc = np.clip(p, clip, 1 - clip)
    return float(-np.sum(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))) / np.sum(w))

df = load_runs(); folds = family_folds(df, seed=20261012); df["fold"] = df["f"].map(folds)
n_ai = df["j"].nunique()
ai_names = (df[["j", "alias"]].drop_duplicates().sort_values("j")["alias"].tolist())
clip_ll = {c: {"P1": [], "P2": [], "P3": []} for c in CLIPS}
horizons = {a: [] for a in ai_names}
censored = {a: {"low": 0, "high": 0} for a in ai_names}

for k in range(5):
    tr = df[df["fold"] != k].reset_index(drop=True); te = df[df["fold"] == k].reset_index(drop=True)
    tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
    tr["ti"] = tr["i"].map(tmap)
    x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float); w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
    x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float); w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()
    p1 = SplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai)
    p2 = MeanPredictor().fit(y_tr, w_tr, j_tr, n_ai)
    p3 = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, len(tmap))
    preds = {"P1": p1.predict(x_te, j_te), "P2": p2.predict(j_te), "P3": p3.predict(x_te, j_te)}
    for c in CLIPS:
        for m, p in preds.items():
            clip_ll[c][m].append(wlogloss(y_te, p, w_te, c))
    # horizons + censoring from this fold's P1 fit
    xmin, xmax = p1.knots_[0], p1.knots_[-1]
    flo, fhi = float(p1.f_(xmin)), float(p1.f_(xmax))
    for jj, a in enumerate(ai_names):
        rhs = p1.alpha_[jj] / p1.beta_
        if rhs <= flo: censored[a]["low"] += 1
        elif rhs >= fhi: censored[a]["high"] += 1
        horizons[a].append(float(p1.horizon(jj)))
    print(f"fold {k} done", flush=True)

out = {"clips": {}}
for c in CLIPS:
    d1 = np.array(clip_ll[c]["P1"]) - np.array(clip_ll[c]["P2"])
    d2 = np.array(clip_ll[c]["P1"]) - np.array(clip_ll[c]["P3"])
    out["clips"][str(c)] = {
        "P1": clip_ll[c]["P1"], "P2": clip_ll[c]["P2"], "P3": clip_ll[c]["P3"],
        "H1_mean_se": [float(d1.mean()), float(d1.std(ddof=1) / math.sqrt(5))],
        "H2_mean_se": [float(d2.mean()), float(d2.std(ddof=1) / math.sqrt(5))]}
out["fold_horizons_min"] = horizons
out["censored_counts"] = censored
json.dump(out, open("../output/clip_horizons.json", "w"), indent=1)
print("saved ../output/clip_horizons.json")
