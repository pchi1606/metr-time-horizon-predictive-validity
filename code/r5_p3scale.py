"""R5 (PX-M4/M5): identified P3 quantities per fold on the locked scaffold.

P3's (beta, theta) pair is scale-redundant in the fitted objective
(location centring only; no scale constraint, penalty, or bounds in
IrtPredictor.fit), so beta_hat alone is not interpretable. The identified
quantities reported here:
  - beta*b per fold: P3's effective shared slope on log2 T for held-out
    tasks (held-out logit = (alpha_j - beta*a) - (beta*b) * x);
  - P0's jointly fitted shared slope per fold (same scaffold) and on all
    data, for comparison;
  - implied held-out logit span of P3 per fold: beta * (max-min of the
    projected thetahat over held-out tasks) -- the task-side span;
  - held-out logit range of P3 predictions per fold (all runs).
Also recomputes the tab:p3diag beta values to confirm scaffold fidelity.
"""
import json
import numpy as np
from data import load_runs, family_folds
from predictors import IrtPredictor, _sigmoid
from run_cv import SEED
from strengthen import fit_p0

df = load_runs()
foldmap = family_folds(df, seed=SEED)
df["fold"] = df["f"].map(foldmap)
n_ai = df["j"].nunique()
out = {"folds": []}
for k in range(5):
    tr = df[df["fold"] != k].reset_index(drop=True)
    te = df[df["fold"] == k].reset_index(drop=True)
    tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
    tr["ti"] = tr["i"].map(tmap)
    x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float)
    w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
    x_te = te["x"].to_numpy(); j_te = te["j"].to_numpy()
    p3 = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, len(tmap))
    th0 = fit_p0(x_tr, y_tr, w_tr, j_tr, n_ai)
    th_hat = p3.a_ + p3.b_ * x_te
    logits = p3.alpha_[j_te] - p3.beta_ * th_hat
    rec = {"fold": k,
           "beta": float(p3.beta_), "a": float(p3.a_), "b": float(p3.b_),
           "beta_times_b": float(p3.beta_ * p3.b_),
           "p0_slope_fold": float(np.exp(th0[n_ai])),
           "theta_hat_range": [float(th_hat.min()), float(th_hat.max())],
           "implied_logit_span_beta_x_thetahat_width":
               float(p3.beta_ * (th_hat.max() - th_hat.min())),
           "heldout_logit_range": [float(logits.min()), float(logits.max())],
           "frac_extreme": float(np.mean((_sigmoid(logits) < 0.01) |
                                         (_sigmoid(logits) > 0.99)))}
    out["folds"].append(rec)
    print(json.dumps(rec), flush=True)

th0_all = fit_p0(df["x"].to_numpy(), df["y"].to_numpy(float),
                 df["w"].to_numpy(), df["j"].to_numpy(), n_ai)
out["p0_slope_full_data"] = float(np.exp(th0_all[n_ai]))
print("P0 full-data slope:", out["p0_slope_full_data"], flush=True)
json.dump(out, open("../output/r5_p3scale.json", "w"), indent=1)
print("saved ../output/r5_p3scale.json")
