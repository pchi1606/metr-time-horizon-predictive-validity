"""5-fold CV shootout over task families: P1 (spline) vs P2 (mean) vs P3 (IRT).

Primary metric: weighted log-loss. Secondary: weighted Brier, micro-AUC.
Saves per-fold scores to ../output/cv_scores.json.
"""
import json
import numpy as np
from scipy.stats import rankdata

from data import load_runs, family_folds
from predictors import SplineHorizon, MeanPredictor, IrtPredictor, EPS

SEED = 20261012


def micro_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def scores_fold(y, p, w):
    pc = np.clip(p, EPS, 1 - EPS)
    logloss = float(-np.sum(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))) / np.sum(w))
    brier = float(np.sum(w * (y - pc) ** 2) / np.sum(w))
    return {"logloss": logloss, "brier": brier, "auc": micro_auc(y, pc)}


def main():
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    out = {"folds": [], "config": {"seed": SEED, "n_folds": 5,
                                  "n_ai": n_ai, "n_runs": len(df)}}
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        # task codes must be contiguous per fold for P3 -> remap
        tr = tr.copy(); te = te.copy()
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr["ti"] = tr["i"].map(tmap)
        n_task_tr = len(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(dtype=float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(dtype=float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()

        p1 = SplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai)
        p2 = MeanPredictor().fit(y_tr, w_tr, j_tr, n_ai)
        p3 = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, n_task_tr)

        s = {"fold": k, "n_test_runs": len(te),
             "P1": scores_fold(y_te, p1.predict(x_te, j_te), w_te),
             "P2": scores_fold(y_te, p2.predict(j_te), w_te),
             "P3": scores_fold(y_te, p3.predict(x_te, j_te), w_te)}
        out["folds"].append(s)
        print(f"fold {k}: " +
              " ".join(f"{m} ll={s[m]['logloss']:.4f} brier={s[m]['brier']:.4f} auc={s[m]['auc']:.4f}"
                        for m in ["P1", "P2", "P3"]), flush=True)

    with open("../output/cv_scores.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved ../output/cv_scores.json")


if __name__ == "__main__":
    main()
