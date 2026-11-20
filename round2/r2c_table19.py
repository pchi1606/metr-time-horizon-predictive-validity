"""R2-C enactment tabulation: Table 19 (fold max/min horizon ratios)
recomputed at locked and tightened endpoints under the printed table's
own convention — the ceiling-censored fold (Claude Opus 4.6, one fold at
the 1800-min ceiling in the P1 locked fits) set aside for that AI.
P1 horizons: deposited output/r30_p1_sensitivity.json (locked/tightened).
P0 locked horizons: deposited output/r5_p0_fold_horizons.npy.
P0 tightened horizons: refit here under the Appendix D protocol exactly
as fixloop/sk048_run.py (L-BFGS-B, ftol 1e-13 / gtol 1e-10)."""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

REPRO = os.path.expanduser("~/workspace/p1-retarget/repro/code")
sys.path.insert(0, REPRO)
os.chdir(REPRO)
from data import load_runs, family_folds          # noqa: E402
from run_cv import SEED                            # noqa: E402
from strengthen import _sigmoid, _bce, math_logit  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "r2c-table19.json")

r30 = json.load(open("../output/r30_p1_sensitivity.json"))
p1_lock = np.array([f["variants"]["locked"]["horizons_min"] for f in r30["folds"]])
p1_tight = np.array([f["variants"]["tightened"]["horizons_min"] for f in r30["folds"]])
p0_lock = np.load("../output/r5_p0_fold_horizons.npy")

# censored cell: P1 locked horizon at the 1800-min ceiling
cens = np.argwhere(p1_lock >= 1799.9)
print("censored cells (fold, ai):", cens.tolist())
cens_ai = int(cens[0][1]) if len(cens) else None
cens_fold = int(cens[0][0]) if len(cens) else None


def fit_p0_tight(x, y, w, j, n_ai):
    a0 = np.zeros(n_ai)
    for jj in range(n_ai):
        m = j == jj
        pj = min(max(np.average(y[m], weights=w[m]), 1e-3), 1 - 1e-3)
        a0[jj] = math_logit(pj)
    th0 = np.concatenate([a0, [0.0]])

    def obj(th):
        return _bce(y, _sigmoid(th[j] - np.exp(th[n_ai]) * x), w)
    res = minimize(obj, th0, method="L-BFGS-B",
                   options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 10000})
    return res.x


df = load_runs()
folds = family_folds(df, seed=SEED)
df["fold"] = df["f"].map(folds)
n_ai = df["j"].nunique()
p0_tight = np.zeros_like(p0_lock)
for k in range(5):
    tr = df[df["fold"] != k].reset_index(drop=True)
    th = fit_p0_tight(tr["x"].to_numpy(), tr["y"].to_numpy(dtype=float),
                      tr["w"].to_numpy(), tr["j"].to_numpy(), n_ai)
    p0_tight[k] = 2.0 ** (th[:n_ai] / float(np.exp(th[n_ai])))
    print(f"fold {k} P0 tightened horizons done", flush=True)


def ratios(H, set_aside_ai=None, set_aside_fold=None):
    R = []
    for a in range(H.shape[1]):
        col = H[:, a]
        if set_aside_ai is not None and a == set_aside_ai:
            col = np.delete(col, set_aside_fold)
        R.append(col.max() / col.min())
    return np.array(R)


def summarise(name, Hp1, Hp0):
    R1 = ratios(Hp1, cens_ai, cens_fold)
    R0 = ratios(Hp0)
    s = {"p1_gt_p0_count": int((R1 > R0).sum()),
         "p1_median_ratio": float(np.median(R1)),
         "p1_worst_ratio": float(R1.max()),
         "p0_median_ratio": float(np.median(R0)),
         "p0_worst_ratio": float(R0.max()),
         "per_ai": [{"ai": int(a), "p1_ratio": float(R1[a]),
                     "p0_ratio": float(R0[a])} for a in range(len(R1))]}
    print(name, {k: s[k] for k in s if k != "per_ai"}, flush=True)
    return s


res = {"censored_cell_fold_ai": [cens_fold, cens_ai],
       "locked": summarise("locked", p1_lock, p0_lock),
       "tightened": summarise("tightened", p1_tight, p0_tight)}
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
