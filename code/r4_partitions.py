"""R4: regenerate the three alternative-partition shootouts (seeds 20261013,
20261014, 42) with the locked pipeline of run_cv.py, capturing per-fold
log-loss (the original post-hoc re-runs retained means +/- SEs only).
Identical predictors, weights and scoring to run_cv.py; only the family
partition seed changes, exactly as reported in the manuscript."""
import json, math
import numpy as np
from data import load_runs, family_folds
from predictors import SplineHorizon, MeanPredictor, IrtPredictor
from run_cv import scores_fold

SEEDS = [20261013, 20261014, 42]

def run(seed):
    df = load_runs()
    folds = family_folds(df, seed=seed)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    rows = []
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        tr = tr.copy(); te = te.copy()
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr["ti"] = tr["i"].map(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(dtype=float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(dtype=float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()
        p1 = SplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai)
        p2 = MeanPredictor().fit(y_tr, w_tr, j_tr, n_ai)
        p3 = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, len(tmap))
        rows.append({"fold": k, "n_test_runs": int(len(te)),
                     "P1": scores_fold(y_te, p1.predict(x_te, j_te), w_te)["logloss"],
                     "P2": scores_fold(y_te, p2.predict(j_te), w_te)["logloss"],
                     "P3": scores_fold(y_te, p3.predict(x_te, j_te), w_te)["logloss"]})
        print(f"seed {seed} fold {k} done", flush=True)
    out = {"folds": rows}
    for a, b, name in [("P1", "P2", "H1"), ("P1", "P3", "H2")]:
        d = np.array([r[a] - r[b] for r in rows])
        out[name] = {"diffs": [float(x) for x in d],
                     "mean": float(d.mean()),
                     "se": float(d.std(ddof=1) / math.sqrt(5))}
    return out

res = {str(s): run(s) for s in SEEDS}
json.dump(res, open("../output/r4_partitions.json", "w"), indent=1)
for s, r in res.items():
    print(s, "H1 %.3f +/- %.3f" % (r["H1"]["mean"], r["H1"]["se"]),
          "H2 %.3f +/- %.3f" % (r["H2"]["mean"], r["H2"]["se"]))
print("saved ../output/r4_partitions.json")
