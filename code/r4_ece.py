"""R4: ECE under alternative binnings. Regenerates pooled out-of-fold
predictions exactly as strengthen.py (same locked fits/seed), then scores
ECE under: 10 equal-width bins (the paper's binning; verification target),
5 and 20 equal-width bins, and 10 equal-count (quantile) bins. Weighting
convention identical to strengthen.ece_and_bins: sqrt-family weights
within bins and for the ECE aggregate."""
import json
import numpy as np
from data import load_runs, family_folds
from predictors import SplineHorizon, MeanPredictor, IrtPredictor, _sigmoid
from run_cv import SEED
from strengthen import fit_p0, ece_and_bins

def ece_edges(y, p, w, edges):
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    e = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        wb = w[m] / w[m].sum()
        mpred = float(np.sum(wb * p[m])); mobs = float(np.sum(wb * y[m]))
        e += (w[m].sum() / w.sum()) * abs(mobs - mpred)
    return float(e)

def ece_equal_count(y, p, w, n_bins=10):
    order = np.argsort(p)
    p_s = p[order]
    edges = np.quantile(p_s, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return ece_edges(y, p, w, edges)

def main():
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    oof = {m: [] for m in ["P1", "P2", "P3", "P0"]}
    yy, ww = [], []
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
        yy.append(y_te); ww.append(w_te)
        print(f"fold {k} done", flush=True)
    y = np.concatenate(yy); w = np.concatenate(ww)
    out = {}
    for m in ["P1", "P2", "P3", "P0"]:
        p = np.concatenate(oof[m])
        e10, _ = ece_and_bins(y, p, w, n_bins=10)
        out[m] = {
            "ece_ew10_verify": e10,
            "ece_ew5": ece_edges(y, p, w, np.linspace(0, 1, 6)),
            "ece_ew20": ece_edges(y, p, w, np.linspace(0, 1, 21)),
            "ece_ec10": ece_equal_count(y, p, w, 10),
        }
        print(m, {k: round(v, 4) for k, v in out[m].items()}, flush=True)
    json.dump(out, open("../output/r4_ece.json", "w"), indent=1)
    print("saved ../output/r4_ece.json")

if __name__ == "__main__":
    main()
