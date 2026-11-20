"""Round-30 fix F-24 (adjudication addendum, Perplexity F30):
calibration presentation for P3r.

The ECE tables cover P0, P1, P2 and P3; P3r -- the comparator that
nearly closes the log-loss gap -- appears in none of them. This
script regenerates P3r's out-of-fold predictions on the deposited
lineage (r22_p3r_extended.py's protocol: fit_irt_ident_ridge at
the selected lambda* = 0.0003 in every fold, locked OLS stage 2,
locked scoring), anchors each fold's scores against
output/r22_p3r_extended.json, and computes P3r's calibration
under the paper's ECE definition (10 equal-width bins, sqrt-family
weights within and across bins, pooled out-of-fold predictions --
the r27_ece_cancellation.py computation), together with the
alternative binnings of Table tab:ecealt and P3r's per-AI ECE.

Anchors: P1/P3 pooled ECE recomputed from
output/oof_predictions.jsonl must match output/r26_ece_ranges.json
to 1e-9 (the r27 assertion); P0's must match strengthen.json.
Saves ../output/r30_p3r_ece.json and the aligned P3r out-of-fold
predictions ../output/r30_p3r_oof.jsonl.
"""
import json

import numpy as np

from data import load_runs, family_folds
from run_cv import scores_fold, SEED
from r11_stage1_ridge import fit_irt_ident_ridge
from r22_p3r import predict_from_fit
from r4_ece import ece_edges, ece_equal_count

LAM = 0.0003
BINS = np.linspace(0.0, 1.0, 11)
AI_NAMES = ["Claude 3 Opus", "Claude 3.5 Sonnet (New)",
            "Claude 3.5 Sonnet (Old)", "Claude 3.7 Sonnet",
            "Claude 4 Opus", "Claude 4.1 Opus", "Claude Opus 4.5",
            "Claude Opus 4.6", "GPT-4 0314", "GPT-4 1106",
            "GPT-4 Turbo", "GPT-4o", "GPT-5", "GPT-5.1-Codex-Max",
            "GPT-5.2", "GPT-5.3-Codex", "Gemini 3 Pro", "o1",
            "o1-preview", "o3"]


def ece_cells(y, p, w, jj):
    """r27_ece_cancellation.cells, generalised to any predictor."""
    p = np.clip(p, 0.0, 1.0)
    idx = np.minimum(np.digitize(p, BINS[1:-1]), 9)
    Wjb = np.zeros((20, 10)); Sjb = np.zeros((20, 10))
    Njb = np.zeros((20, 10), dtype=int)
    Pb = np.zeros(10); Ob = np.zeros(10); Nb = np.zeros(10, dtype=int)
    Wb = np.zeros(10)
    for b in range(10):
        m = idx == b
        Nb[b] = int(m.sum()); Wb[b] = w[m].sum()
        if m.sum():
            wb = w[m] / w[m].sum()
            Pb[b] = float(np.sum(wb * p[m])); Ob[b] = float(np.sum(wb * y[m]))
    for j in range(20):
        for b in range(10):
            m = (jj == j) & (idx == b)
            Wjb[j, b] = w[m].sum(); Njb[j, b] = int(m.sum())
            Sjb[j, b] = float(np.sum(w[m] * (y[m] - p[m])))
    W = float(w.sum())
    Wj = Wjb.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(Wjb > 0, Sjb / np.where(Wjb > 0, Wjb, 1.0), 0.0)
    ece_j = (Wjb / Wj[:, None] * np.abs(gap)).sum(axis=1)
    Gb = Sjb.sum(axis=0) / Wb
    pooled = float((Wb / W * np.abs(Gb)).sum())
    return {"pooled_ece": pooled, "per_ai_ece": ece_j,
            "per_ai_median": float(np.median(ece_j)),
            "per_ai_min": float(ece_j.min()),
            "per_ai_max": float(ece_j.max()),
            "weight_averaged_per_ai_ece": float((Wj / W * ece_j).sum()),
            "bins": [{"bin": f"{BINS[b]:.1f}-{BINS[b+1]:.1f}",
                      "n": int(Nb[b]),
                      "weight_share": float(Wb[b] / W),
                      "pred": float(Pb[b]), "obs": float(Ob[b])}
                     for b in range(10)]}


def main():
    rows = [json.loads(l) for l in open("../output/oof_predictions.jsonl")]
    y = np.array([r["y"] for r in rows], float)
    w = np.array([r["w"] for r in rows], float)
    jj = np.array([r["j"] for r in rows], int)

    r26 = json.load(open("../output/r26_ece_ranges.json"))
    st = json.load(open("../output/strengthen.json"))
    for pred in ("P1", "P3"):
        p = np.array([r[pred] for r in rows], float)
        got = ece_cells(y, p, w, jj)["pooled_ece"]
        assert abs(got - r26["pooled"][pred]) < 1e-9, (pred, got)
        print(f"anchor {pred} pooled ECE {got:.6f} == r26 OK")
    p0 = np.array([r["P0"] for r in rows], float)
    got0 = ece_cells(y, p0, w, jj)["pooled_ece"]
    assert abs(got0 - st["calibration"]["P0"]["ece"]) < 5e-4, got0
    print(f"anchor P0 pooled ECE {got0:.6f} ~= strengthen OK")

    # ---- regenerate P3r OOF predictions at the selected lambda ----
    df = load_runs()
    folds = family_folds(df, seed=SEED)
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    ext = json.load(open("../output/r22_p3r_extended.json"))
    pred_map = {}
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr["ti"] = tr["i"].map(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy()
        i_tr = tr["ti"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()
        fit = fit_irt_ident_ridge(x_tr, y_tr, w_tr, j_tr, i_tr,
                                  n_ai, len(tmap), lam=LAM)
        p_te = predict_from_fit(fit, x_te, j_te)
        s = scores_fold(y_te, p_te, w_te)
        dep = ext["folds"][k]["P3r"]
        assert abs(s["logloss"] - dep["logloss"]) < 1e-9, (k, s, dep)
        assert abs(s["brier"] - dep["brier"]) < 1e-9, (k, s, dep)
        assert abs(s["auc"] - dep["auc"]) < 1e-9, (k, s, dep)
        for task, jv, pv in zip(te["task_id"].to_numpy(), j_te, p_te):
            pred_map[(k, task, int(jv))] = float(pv)
        print(f"fold {k}: P3r scores anchored to r22_p3r_extended "
              f"(ll {s['logloss']:.4f})", flush=True)

    p3r = np.array([pred_map[(r["fold"], r["task"], r["j"])] for r in rows])
    with open("../output/r30_p3r_oof.jsonl", "w") as fh:
        for r, pv in zip(rows, p3r):
            fh.write(json.dumps({"fold": r["fold"], "j": r["j"],
                                 "task": r["task"], "P3r": pv}) + "\n")

    res = ece_cells(y, p3r, w, jj)
    out = {"lambda": LAM, "source": "output/oof_predictions.jsonl + "
           "regenerated P3r predictions (r22_p3r_extended lineage)",
           "pooled_ece": res["pooled_ece"],
           "ece_ew5": ece_edges(y, p3r, w, np.linspace(0, 1, 6)),
           "ece_ew20": ece_edges(y, p3r, w, np.linspace(0, 1, 21)),
           "ece_ec10": ece_equal_count(y, p3r, w, 10),
           "per_ai_ece": {AI_NAMES[j]: float(res["per_ai_ece"][j])
                          for j in range(20)},
           "per_ai_median": res["per_ai_median"],
           "per_ai_min": res["per_ai_min"], "per_ai_max": res["per_ai_max"],
           "weight_averaged_per_ai_ece":
               res["weight_averaged_per_ai_ece"],
           "bins": res["bins"]}
    with open("../output/r30_p3r_ece.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"P3r pooled ECE {res['pooled_ece']:.4f} | EW5 "
          f"{out['ece_ew5']:.4f} | EW20 {out['ece_ew20']:.4f} | EC10 "
          f"{out['ece_ec10']:.4f} | per-AI median {res['per_ai_median']:.4f} "
          f"range {res['per_ai_min']:.4f}-{res['per_ai_max']:.4f} | "
          f"weight-avg per-AI {res['weight_averaged_per_ai_ece']:.4f}")
    print("saved ../output/r30_p3r_ece.json + ../output/r30_p3r_oof.jsonl")


if __name__ == "__main__":
    main()
