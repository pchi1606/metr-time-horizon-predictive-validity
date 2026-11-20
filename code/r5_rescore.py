"""Round-5 rescoring add-on (post-panel analyses; NOT part of the locked
pipeline). One pass over the locked 5-fold family CV (same seed, same
predictor code as run_cv.py / strengthen.py) that additionally:

  * dumps per-run out-of-fold predictions for P0/P1/P2/P3
    (../output/oof_predictions.jsonl) and the fold assignment
    (../output/fold_assignment.json);
  * computes WITHIN-AI AUC per predictor (skeptic MAJOR-2: the pooled
    AUC mixes between-AI and within-AI ordering);
  * computes CROSS-FITTED recalibration of P3 (Platt + isotonic, each
    fitted on the other four folds' OOF predictions) — referee m1;
  * computes the headline paired differences restricted to tasks whose
    human times are MEASURED (human_source != 'estimate') — referee m2;
  * records P0's per-fold horizons (skeptic MINOR-5 fragility comparator);
  * checks P1's fold-3 extrapolation behaviour (skeptic MINOR-7): fold 3
    rescored with x clipped to the training range (true flat
    extrapolation) vs as published.

Everything is computed under the locked sqrt-family weights unless a
line says otherwise. Prints a JSON summary; also saved to
../output/r5_rescore.json.
"""
import json
import numpy as np
from collections import Counter, defaultdict

from data import load_runs, family_folds
from predictors import SplineHorizon, MeanPredictor, IrtPredictor, EPS
from run_cv import scores_fold, micro_auc, SEED
from strengthen import fit_p0, _sigmoid, ece_and_bins

MODELS = ["P1", "P2", "P3", "P0"]


def wlogloss(y, p, w):
    pc = np.clip(p, EPS, 1 - EPS)
    return float(-np.sum(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))) / np.sum(w))


def wbrier(y, p, w):
    pc = np.clip(p, EPS, 1 - EPS)
    return float(np.sum(w * (y - pc) ** 2) / np.sum(w))


def main():
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    ai_names = (df[["j", "alias"]].drop_duplicates()
                .sort_values("j")["alias"].tolist())

    # fold assignment dump (referee m3)
    code2name = (df[["f", "family"]].drop_duplicates()
                 .set_index("f")["family"].to_dict())
    fam_fold = {}
    for code, k in sorted(folds.items()):
        fam_fold.setdefault(str(int(k)), []).append(code2name[code])
    with open("../output/fold_assignment.json", "w") as f:
        json.dump({"seed": SEED, "families_per_fold": fam_fold,
                   "n_families": int(df["f"].nunique())}, f, indent=1)

    # task -> human_source (measured vs estimate), from the raw release file
    src = defaultdict(Counter)
    with open("../data/runs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            src[r["task_id"]][r["human_source"]] += 1
    task_src = {t: c.most_common(1)[0][0] for t, c in src.items()}
    df["hsrc"] = df["task_id"].map(task_src)
    assert df["hsrc"].notna().all()

    rows = []          # per-run OOF dump
    p0_fold_horizons = np.zeros((5, n_ai))
    pchip = {}
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr = tr.copy(); te = te.copy()
        tr["ti"] = tr["i"].map(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()

        p1 = SplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai)
        p2 = MeanPredictor().fit(y_tr, w_tr, j_tr, n_ai)
        p3 = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, len(tmap))
        th0 = fit_p0(x_tr, y_tr, w_tr, j_tr, n_ai)
        beta0 = float(np.exp(th0[n_ai]))
        p0_fold_horizons[k] = 2.0 ** (th0[:n_ai] / beta0)

        preds = {"P1": p1.predict(x_te, j_te), "P2": p2.predict(j_te),
                 "P3": p3.predict(x_te, j_te),
                 "P0": _sigmoid(th0[j_te] - beta0 * x_te)}
        # fidelity: fold scores must match the locked cv_scores.json
        chk = scores_fold(y_te, preds["P1"], w_te)
        print(f"fold {k}: P1 logloss {chk['logloss']:.4f}", flush=True)

        if k == 3:
            lo, hi = float(x_tr.min()), float(x_tr.max())
            out_mask = (x_te < lo) | (x_te > hi)
            p_flat = p1.predict(np.clip(x_te, lo, hi), j_te)
            pchip = {
                "n_test": int(len(te)), "n_out_of_range": int(out_mask.sum()),
                "n_below": int((x_te < lo).sum()), "n_above": int((x_te > hi).sum()),
                "train_range": [lo, hi],
                "logloss_as_published": wlogloss(y_te, preds["P1"], w_te),
                "logloss_flat": wlogloss(y_te, p_flat, w_te),
                "max_abs_dp_out_of_range": float(np.abs(p_flat[out_mask] - preds["P1"][out_mask]).max()) if out_mask.any() else 0.0,
            }

        for n in range(len(te)):
            rows.append({"fold": k, "j": int(j_te[n]), "task": te["task_id"].iloc[n],
                         "y": float(y_te[n]), "w": float(w_te[n]),
                         "hsrc": te["hsrc"].iloc[n],
                         **{m: float(preds[m][n]) for m in MODELS}})
    with open("../output/oof_predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    y = np.array([r["y"] for r in rows]); w = np.array([r["w"] for r in rows])
    jj = np.array([r["j"] for r in rows]); fk = np.array([r["fold"] for r in rows])
    P = {m: np.array([r[m] for r in rows]) for m in MODELS}
    hsrc = np.array([r["hsrc"] for r in rows])
    out = {"n_oof": int(len(rows))}

    # --- within-AI AUC: per (fold, AI) cell, then averaged over cells ---
    # (Pooling an AI's runs across folds would contaminate the estimand:
    # each fold's fit has its own level, e.g. P2's constant differs by
    # fold. Per cell, every duration-based predictor ranks by x alone,
    # so P1/P0/P3 coincide; P2 is exactly 0.5 by construction.)
    wa = {}
    for m in MODELS:
        cells = []
        for k in range(5):
            for a in range(n_ai):
                sel = (fk == k) & (jj == a)
                if 0 < y[sel].sum() < sel.sum():
                    cells.append(micro_auc(y[sel], P[m][sel]))
        wa[m] = {"mean": float(np.mean(cells)), "sd": float(np.std(cells, ddof=1)),
                 "min": float(np.min(cells)), "max": float(np.max(cells)),
                 "n_cells": len(cells)}
    out["within_ai_auc"] = wa

    # --- cross-fitted recalibration of P3 (Platt + isotonic) ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    p3 = np.clip(P["P3"], EPS, 1 - EPS)
    logit = np.log(p3 / (1 - p3)).reshape(-1, 1)
    rec = {"platt": np.zeros_like(p3), "isotonic": np.zeros_like(p3)}
    for k in range(5):
        trm, tem = fk != k, fk == k
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(logit[trm], y[trm])
        rec["platt"][tem] = lr.predict_proba(logit[tem])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p3[trm], y[trm])
        rec["isotonic"][tem] = iso.predict(p3[tem])
    rc = {"raw": {"logloss": wlogloss(y, p3, w), "brier": wbrier(y, p3, w),
                  "ece_ew10": ece_and_bins(y, p3, w, 10)[0]}}
    for name, pr in rec.items():
        rc[name] = {"logloss": wlogloss(y, pr, w), "brier": wbrier(y, pr, w),
                    "ece_ew10": ece_and_bins(y, pr, w, 10)[0]}
    out["p3_recalibration_crossfitted"] = rc

    # --- measured-time-only paired differences (per fold, locked weights) ---
    meas = hsrc != "estimate"
    md = {}
    for pair in [("P1", "P2"), ("P1", "P3")]:
        diffs = []
        for k in range(5):
            m = meas & (fk == k)
            diffs.append(wlogloss(y[m], P[pair[0]][m], w[m]) - wlogloss(y[m], P[pair[1]][m], w[m]))
        diffs = np.array(diffs)
        md[f"{pair[0]}-{pair[1]}"] = {"per_fold": diffs.tolist(),
                                      "mean": float(diffs.mean()),
                                      "se": float(diffs.std(ddof=1) / np.sqrt(5))}
    out["measured_only_logloss_diffs"] = md
    out["measured_share_runs"] = float(meas.mean())
    out["estimate_tasks"] = int(sum(1 for t, s in task_src.items() if s == "estimate"))

    # --- P0 fold horizons (fragility comparator) ---
    s = json.load(open("../output/strengthen.json"))
    p1h = np.array([s["horizon_folds_min"][a] for a in ai_names])  # AIs x folds
    p0h = p0_fold_horizons.T                                       # AIs x folds
    def spread(h):
        med = np.median(h, axis=1)
        return {"median_fold_median": float(np.median(med)),
                "max_ratio_foldmax_foldmin": float(np.max(h.max(axis=1) / h.min(axis=1))),
                "opus46_fold_values": h[ai_names.index("Claude Opus 4.6 (Inspect)")].tolist()}
    out["fold_horizon_spread"] = {"P1": spread(p1h), "P0": spread(p0h)}
    np.save("../output/r5_p0_fold_horizons.npy", p0_fold_horizons)
    out["pchip_fold3"] = pchip

    with open("../output/r5_rescore.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
