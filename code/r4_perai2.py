"""R4 check 2: per-AI paired diffs pooled under (a) unweighted, (b) equal-task weights."""
import json
import numpy as np
from data import load_runs, family_folds
from predictors import SplineHorizon, MeanPredictor, IrtPredictor, EPS
from run_cv import SEED

def main():
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    ai_names = (df[["j", "alias"]].drop_duplicates()
                .sort_values("j")["alias"].tolist())
    acc = {scheme: {m: [np.zeros(n_ai), np.zeros(n_ai)] for m in ["P1", "P2", "P3"]}
           for scheme in ["unweighted", "equaltask"]}
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr["ti"] = tr["i"].map(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float)
        j_te = te["j"].to_numpy()
        weq_te = (1.0 / te.groupby(["i", "j"])["y"].transform("count")).to_numpy()
        wsets = {"unweighted": np.ones_like(y_te), "equaltask": weq_te}
        preds = {
            "P1": SplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai).predict(x_te, j_te),
            "P2": MeanPredictor().fit(y_tr, w_tr, j_tr, n_ai).predict(j_te),
            "P3": IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, len(tmap)).predict(x_te, j_te),
        }
        for scheme, w_te in wsets.items():
            for m, p in preds.items():
                pc = np.clip(p, EPS, 1 - EPS)
                ll = -(y_te * np.log(pc) + (1 - y_te) * np.log(1 - pc))
                for jj in range(n_ai):
                    mj = j_te == jj
                    if mj.sum():
                        acc[scheme][m][0][jj] += np.sum(w_te[mj] * ll[mj])
                        acc[scheme][m][1][jj] += np.sum(w_te[mj])
        print(f"fold {k} done", flush=True)
    out = {}
    for scheme in acc:
        out[scheme] = {}
        for a, b in [("P1", "P2"), ("P1", "P3")]:
            d = acc[scheme][a][0] / acc[scheme][a][1] - acc[scheme][b][0] / acc[scheme][b][1]
            out[scheme][f"{a}_vs_{b}"] = {ai_names[jj]: float(d[jj]) for jj in range(n_ai)}
            print(scheme, a, "vs", b, "negatives:", int((d < 0).sum()), "of", n_ai, flush=True)
    with open("../output/r4_perai_variants.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved", flush=True)

if __name__ == "__main__":
    main()
