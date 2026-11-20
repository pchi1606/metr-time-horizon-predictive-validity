"""Round-26 SP25-D: per-AI and per-fold ECE ranges for P1 and P3 (and
P0/P2 for the record), computed from the deposited out-of-fold
predictions (output/oof_predictions.jsonl) under the paper's ECE
definition: 10 equal-width bins on predicted probability, sqrt-family
weights within and across bins (Section 5 / Appendix G). The pooled
values must reproduce the printed ECE (P1 0.031, P3 0.155, P0 0.024,
P2 0.037) before any range is trusted. Saves output/r26_ece_ranges.json.
Also records the Claude Opus 4.6 sub-1-minute weight share behind
Table 13's caption (SW25-4), from output/r22_metr_spec.json.
"""
import json

import numpy as np

ROWS = [json.loads(l) for l in open("../output/oof_predictions.jsonl")]
PREDS = ["P1", "P3", "P0", "P2"]
BINS = np.linspace(0.0, 1.0, 11)


def ece(rows, pred):
    y = np.array([r["y"] for r in rows], float)
    w = np.array([r["w"] for r in rows], float)
    p = np.clip(np.array([r[pred] for r in rows], float), 0.0, 1.0)
    idx = np.minimum(np.digitize(p, BINS[1:-1]), 9)
    tot = w.sum()
    val = 0.0
    for b in range(10):
        m = idx == b
        if w[m].sum() == 0:
            continue
        obs = float(np.sum(w[m] * y[m]) / w[m].sum())
        prd = float(np.sum(w[m] * p[m]) / w[m].sum())
        val += (w[m].sum() / tot) * abs(obs - prd)
    return float(val)


out = {"definition": "10 equal-width bins on predicted probability; "
                     "sqrt-family weights within and across bins; "
                     "pooled out-of-fold predictions",
       "source": "output/oof_predictions.jsonl", "pooled": {},
       "per_ai": {}, "per_fold": {}}
for pred in PREDS:
    out["pooled"][pred] = ece(ROWS, pred)
    per_ai = {}
    for j in sorted({r["j"] for r in ROWS}):
        per_ai[str(j)] = ece([r for r in ROWS if r["j"] == j], pred)
    per_fold = {}
    for k in sorted({r["fold"] for r in ROWS}):
        per_fold[str(k)] = ece([r for r in ROWS if r["fold"] == k], pred)
    out["per_ai"][pred] = {"values": per_ai,
                           "min": min(per_ai.values()), "max": max(per_ai.values())}
    out["per_fold"][pred] = {"values": per_fold,
                             "min": min(per_fold.values()), "max": max(per_fold.values())}

spec = json.load(open("../output/r22_metr_spec.json"))
shares = [b["weight_share"] for b in spec["opus46"]["empirical_buckets"]]
out["opus46_table13"] = {
    "bucket_shares_sum": float(np.sum(shares)),
    "sub_1min_weight_share": float(1.0 - np.sum(shares)),
    "n_runs_in_buckets": int(np.sum([b["n_runs"] for b in spec["opus46"]["empirical_buckets"]])),
    "n_runs_total_opus46": int(spec["opus46"]["n_runs"]),
}
with open("../output/r26_ece_ranges.json", "w") as f:
    json.dump(out, f, indent=2)
for pred in PREDS:
    print(f"{pred}: pooled {out['pooled'][pred]:.4f} | per-AI range "
          f"{out['per_ai'][pred]['min']:.3f}-{out['per_ai'][pred]['max']:.3f} | "
          f"per-fold range {out['per_fold'][pred]['min']:.3f}-{out['per_fold'][pred]['max']:.3f}")
print("Opus46 Table 13: bucket shares sum", round(out["opus46_table13"]["bucket_shares_sum"], 4),
      "| sub-1-min share", round(out["opus46_table13"]["sub_1min_weight_share"], 4),
      "| runs in buckets", out["opus46_table13"]["n_runs_in_buckets"],
      "of", out["opus46_table13"]["n_runs_total_opus46"])
