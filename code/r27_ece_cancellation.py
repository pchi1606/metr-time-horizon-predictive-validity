"""Round-27 REG-P1-9: cross-AI cancellation analysis of the pooled ECE.

The paper's pooled ECE pools all out-of-fold predictions across the 20
AIs before binning; within a bin, the signed gaps (observed minus
predicted, sqrt-family weighted) of different AIs can cancel, so the
pooled ECE can lie below every per-AI ECE. This script decomposes the
pooled ECE of P1 and P3 from the deposited out-of-fold predictions
(output/oof_predictions.jsonl) under the paper's printed ECE definition
(10 equal-width bins on predicted probability, sqrt-family weights
within and across bins; the definition of code/r26_ece_ranges.py):

  cell (j, b): W_jb = total weight of AI j's runs in bin b;
               gap_jb = weighted (obs - pred) within the cell.
  per-AI ECE_j = sum_b (W_jb / W_j) |gap_jb|        (== r26 ranges)
  pooled bin gap G_b = sum_j W_jb gap_jb / W_b;  pooled ECE
               = sum_b (W_b / W) |G_b|              (== printed ECE)
  no-cancellation pooled ECE* = sum_j (W_j / W) ECE_j
               = sum_b (W_b / W) A_b,  A_b = sum_j (W_jb / W_b) |gap_jb|
  cancellation = ECE* - pooled ECE = sum_b (W_b / W) (A_b - |G_b|) >= 0.

Recomposition identities are asserted to 1e-9 against both the direct
pooled computation and output/r26_ece_ranges.json before any number is
saved. Saves output/r27_ece_cancellation.json, including the full
per-AI signed-gap matrices.
"""
import json

import numpy as np

ROWS = [json.loads(l) for l in open("../output/oof_predictions.jsonl")]
PREDS = ["P1", "P3"]
BINS = np.linspace(0.0, 1.0, 11)
AI_NAMES = ["Claude 3 Opus", "Claude 3.5 Sonnet (New)",
            "Claude 3.5 Sonnet (Old)", "Claude 3.7 Sonnet",
            "Claude 4 Opus", "Claude 4.1 Opus", "Claude Opus 4.5",
            "Claude Opus 4.6", "GPT-4 0314", "GPT-4 1106",
            "GPT-4 Turbo", "GPT-4o", "GPT-5", "GPT-5.1-Codex-Max",
            "GPT-5.2", "GPT-5.3-Codex", "Gemini 3 Pro", "o1",
            "o1-preview", "o3"]
JS = sorted({r["j"] for r in ROWS})
assert JS == list(range(20)), JS

y = np.array([r["y"] for r in ROWS], float)
w = np.array([r["w"] for r in ROWS], float)
jj = np.array([r["j"] for r in ROWS], int)
W = float(w.sum())

r26 = json.load(open("../output/r26_ece_ranges.json"))


def cells(pred):
    p = np.clip(np.array([r[pred] for r in ROWS], float), 0.0, 1.0)
    idx = np.minimum(np.digitize(p, BINS[1:-1]), 9)
    Wjb = np.zeros((20, 10))
    Sjb = np.zeros((20, 10))  # weighted signed gap mass: sum w*(y-p)
    for j in range(20):
        for b in range(10):
            m = (jj == j) & (idx == b)
            Wjb[j, b] = w[m].sum()
            Sjb[j, b] = float(np.sum(w[m] * (y[m] - p[m])))
    return Wjb, Sjb


out = {"definition": "10 equal-width bins on predicted probability; "
                     "sqrt-family weights within and across bins; "
                     "pooled out-of-fold predictions; signed gap = "
                     "weighted observed rate minus weighted mean "
                     "prediction within the cell",
       "source": "output/oof_predictions.jsonl",
       "n_rows": len(ROWS), "total_weight": W, "predictors": {}}
for pred in PREDS:
    Wjb, Sjb = cells(pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(Wjb > 0, Sjb / np.where(Wjb > 0, Wjb, 1.0), 0.0)
    Wj = Wjb.sum(axis=1)
    Wb = Wjb.sum(axis=0)
    ece_j = (Wjb / Wj[:, None] * np.abs(gap)).sum(axis=1)
    Gb = Sjb.sum(axis=0) / Wb
    pooled = float((Wb / W * np.abs(Gb)).sum())
    Ab = (Wjb / Wb[None, :] * np.abs(gap)).sum(axis=0)
    star = float((Wj / W * ece_j).sum())
    star_bins = float((Wb / W * Ab).sum())
    canc_b = (Wb / W) * (Ab - np.abs(Gb))
    # --- recomposition assertions -------------------------------------
    assert abs(pooled - r26["pooled"][pred]) < 1e-9, (pred, pooled)
    for j in range(20):
        assert abs(ece_j[j] - r26["per_ai"][pred]["values"][str(j)]) < 1e-9
    assert abs(star - star_bins) < 1e-9, (pred, star, star_bins)
    assert abs(star - pooled - canc_b.sum()) < 1e-9
    assert (canc_b >= -1e-12).all()
    mixed = [int(b) for b in range(10)
             if (gap[:, b][Wjb[:, b] > 0] > 0).any()
             and (gap[:, b][Wjb[:, b] > 0] < 0).any()]
    rec = {"pooled_ece": pooled,
           "weight_averaged_per_ai_ece": star,
           "cancellation": float(star - pooled),
           "cancellation_share_of_star": float((star - pooled) / star),
           "per_ai_ece": {AI_NAMES[j]: float(ece_j[j]) for j in range(20)},
           "per_ai_median": float(np.median(ece_j)),
           "per_ai_min": float(ece_j.min()), "per_ai_max": float(ece_j.max()),
           "ai_weight_shares": {AI_NAMES[j]: float(Wj[j] / W)
                                for j in range(20)},
           "bins": [{"bin": f"{BINS[b]:.1f}-{BINS[b+1]:.1f}",
                     "weight_share": float(Wb[b] / W),
                     "pooled_signed_gap": float(Gb[b]),
                     "mean_abs_per_ai_gap": float(Ab[b]),
                     "cancelled": float(canc_b[b])} for b in range(10)],
           "mixed_sign_bins": mixed,
           "per_ai_signed_gaps": {AI_NAMES[j]: [float(gap[j, b])
                                                for b in range(10)]
                                  for j in range(20)},
           "per_ai_bin_weights": {AI_NAMES[j]: [float(Wjb[j, b])
                                                for b in range(10)]
                                  for j in range(20)}}
    out["predictors"][pred] = rec
    print(f"{pred}: pooled {pooled:.4f} | weight-avg per-AI {star:.4f} | "
          f"cancelled {star - pooled:.4f} "
          f"({100 * (star - pooled) / star:.1f}% of the uncancelled value) | "
          f"mixed-sign bins {mixed}")
with open("../output/r27_ece_cancellation.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved ../output/r27_ece_cancellation.json")
