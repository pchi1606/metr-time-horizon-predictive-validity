"""Validate fast_ridge against the deposited P3r construction on the
locked partition: for each outer fold, run the inner 5-fold family-CV
selection over the final 13-point grid with BOTH implementations and
compare inner means, selected lambda, and the outer P3r log-loss /
P1-P3r margin against output/r26_p3r_bracket.json (deposited)."""
import json
import os
import sys
import time

import numpy as np

REPRO = os.path.expanduser("~/workspace/p1-retarget/repro/code")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPRO)
sys.path.insert(0, HERE)
os.chdir(REPRO)
from data import load_runs, family_folds          # noqa: E402
from run_cv import scores_fold, SEED              # noqa: E402
from r11_stage1_ridge import fit_irt_ident_ridge  # noqa: E402
from r22_p3r import predict_from_fit              # noqa: E402
from fast_ridge import fit_irt_ident_ridge_fast   # noqa: E402

GRID = [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3]
bracket = json.load(open("../output/r26_p3r_bracket.json"))
cv = json.load(open("../output/cv_scores.json"))
p1_ll = np.array([f["P1"]["logloss"] for f in cv["folds"]])

df = load_runs()
folds = family_folds(df, seed=SEED)
df["fold"] = df["f"].map(folds)
n_ai = df["j"].nunique()

REPORT_PATH = os.path.join(HERE, "fast-ridge-validation.json")
report = {"folds": []}
if os.path.exists(REPORT_PATH):
    report = json.load(open(REPORT_PATH))
done_folds = {fr["fold"] for fr in report["folds"]}
for k in range(5):
    if k in done_folds:
        print(f"fold {k}: cached, skipping", flush=True)
        continue
    tr = df[df["fold"] != k].reset_index(drop=True)
    te = df[df["fold"] == k].reset_index(drop=True)
    tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
    tr = tr.copy(); tr["ti"] = tr["i"].map(tmap); n_task = len(tmap)
    inner_map = family_folds(tr, n_folds=5, seed=SEED)
    tr["ifold"] = tr["f"].map(inner_map)
    x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float)
    w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()
    fold_rec = {"fold": k}
    for impl_name, impl in (("deposited", fit_irt_ident_ridge),
                            ("fast", fit_irt_ident_ridge_fast)):
        t0 = time.time()
        inner_ll = {lam: [] for lam in GRID}
        for m in range(5):
            itr = tr[tr["ifold"] != m].reset_index(drop=True)
            ite = tr[tr["ifold"] == m].reset_index(drop=True)
            imap = {t: n for n, t in enumerate(sorted(itr["i"].unique()))}
            itr = itr.copy(); itr["ti"] = itr["i"].map(imap)
            xi = itr["x"].to_numpy(); yi = itr["y"].to_numpy(float)
            wi = itr["w"].to_numpy(); ji = itr["j"].to_numpy()
            ii = itr["ti"].to_numpy()
            xe = ite["x"].to_numpy(); ye = ite["y"].to_numpy(float)
            we = ite["w"].to_numpy(); je = ite["j"].to_numpy()
            for lam in GRID:
                fit = impl(xi, yi, wi, ji, ii, n_ai, len(imap), lam=lam)
                inner_ll[lam].append(
                    scores_fold(ye, predict_from_fit(fit, xe, je), we)["logloss"])
        inner_mean = {lam: float(np.mean(v)) for lam, v in inner_ll.items()}
        lam_star = min(GRID, key=lambda lam: inner_mean[lam])
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy()
        i_tr = tr["ti"].to_numpy()
        fit = impl(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, n_task, lam=lam_star)
        outer = scores_fold(y_te, predict_from_fit(fit, x_te, j_te), w_te)["logloss"]
        fold_rec[impl_name] = {
            "selected_lambda": lam_star, "outer_logloss": float(outer),
            "margin": float(p1_ll[k] - outer),
            "inner_mean": {str(l): v for l, v in inner_mean.items()},
            "seconds": round(time.time() - t0, 1)}
        print(f"fold {k} {impl_name}: lam*={lam_star} outer={outer:.6f} "
              f"margin={p1_ll[k] - outer:+.6f} ({fold_rec[impl_name]['seconds']}s)",
              flush=True)
    dep_outer = bracket["folds"][k]["outer_by_lambda"][str(bracket["folds"][k]["selected_lambda"])]["logloss"]
    fold_rec["bracket_outer_at_selected"] = float(dep_outer)
    fold_rec["bracket_selected_lambda"] = bracket["folds"][k]["selected_lambda"]
    report["folds"].append(fold_rec)
    report["folds"].sort(key=lambda fr: fr["fold"])
    json.dump(report, open(REPORT_PATH, "w"), indent=1)

json.dump(report, open(REPORT_PATH, "w"), indent=1)
for fr in report["folds"]:
    d, f = fr["deposited"], fr["fast"]
    print(f"fold {fr['fold']}: lam* dep={d['selected_lambda']} fast={f['selected_lambda']} "
          f"bracket={fr['bracket_selected_lambda']}; "
          f"|dOuter dep-vs-bracket|={abs(d['outer_logloss'] - fr['bracket_outer_at_selected']):.2e}; "
          f"|dOuter fast-vs-dep|={abs(f['outer_logloss'] - d['outer_logloss']):.2e}; "
          f"max|dInnerMean|={max(abs(f['inner_mean'][l] - d['inner_mean'][l]) for l in d['inner_mean']):.2e}")
print("wrote fast-ridge-validation.json")
