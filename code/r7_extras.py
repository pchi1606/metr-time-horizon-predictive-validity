"""Round-8 extras (post-hoc; r7 panel items):
  * Family-cluster bootstrap of the pooled ECE contrast (referee m5):
    resample the 79 task families with replacement, recompute pooled
    sqrt-weighted EW10 ECE for P1/P3/P0/P2 and the P3-P1 contrast.
  * P2u sensitivity (skeptic m3): H1 against an UNWEIGHTED per-AI train
    mean (the practitioner's mean accuracy). P1 fold scores are taken
    from the locked cv_scores.json; only P2u is fitted here.
  * Fold-averaged pooled-across-AI AUC for P0 (for the AUC summary
    table), from the OOF dump; plus exact within-AI AUC means.
Saves ../output/r7_extras.json.
"""
import json
import numpy as np

from data import load_runs, family_folds
from run_cv import scores_fold, micro_auc, SEED
from strengthen import ece_and_bins

rng = np.random.default_rng(20261105)

# ---------- load OOF dump + family map ----------
rows = [json.loads(l) for l in open("../output/oof_predictions.jsonl")]
df = load_runs()
task2fam = (df[["task_id", "family"]].drop_duplicates()
            .set_index("task_id")["family"].to_dict())
fams = np.array([task2fam[r["task"]] for r in rows])
y = np.array([r["y"] for r in rows]); w = np.array([r["w"] for r in rows])
fk = np.array([r["fold"] for r in rows])
P = {m: np.array([r[m] for r in rows]) for m in ["P1", "P2", "P3", "P0"]}

# ---------- ECE family-cluster bootstrap ----------
uniq = np.unique(fams)
by_fam = {f: np.where(fams == f)[0] for f in uniq}
B = 400
boot = {m: [] for m in P}
for b in range(B):
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([by_fam[f] for f in pick])
    for m in P:
        boot[m].append(ece_and_bins(y[idx], P[m][idx], w[idx], 10)[0])
ece_boot = {}
for m in P:
    arr = np.array(boot[m])
    ece_boot[m] = {"ece": ece_and_bins(y, P[m], w, 10)[0],
                   "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]}
contrast = np.array(boot["P3"]) - np.array(boot["P1"])
ece_boot["P3_minus_P1"] = {"diff": ece_boot["P3"]["ece"] - ece_boot["P1"]["ece"],
                          "ci95": [float(np.percentile(contrast, 2.5)),
                                   float(np.percentile(contrast, 97.5))]}
print("ECE bootstrap:", json.dumps(ece_boot, indent=1), flush=True)

# ---------- AUC: fold-averaged for P0 + exact within-AI ----------
auc_fold = {}
for m in P:
    per_fold = [micro_auc(y[fk == k], P[m][fk == k]) for k in range(5)]
    auc_fold[m] = {"per_fold": per_fold, "mean": float(np.mean(per_fold)),
                   "se": float(np.std(per_fold, ddof=1) / np.sqrt(5))}
rs = json.load(open("../output/r5_rescore.json"))
within = {m: rs["within_ai_auc"][m]["mean"] for m in P}
pooled = {m: float(micro_auc(y, P[m])) for m in P}
print("AUC fold-avg:", {m: round(auc_fold[m]["mean"], 4) for m in P}, flush=True)
print("within-AI exact:", within, flush=True)

# ---------- P2u: unweighted train mean ----------
folds_map = family_folds(df, seed=SEED)
df["fold"] = df["f"].map(folds_map)
cv = json.load(open("../output/cv_scores.json"))
diffs = []
for k in range(5):
    tr = df[df["fold"] != k]; te = df[df["fold"] == k]
    mu = tr.groupby("j")["y"].mean()
    p2u = te["j"].map(mu).to_numpy()
    s = scores_fold(te["y"].to_numpy(float), p2u, te["w"].to_numpy())
    diffs.append(cv["folds"][k]["P1"]["logloss"] - s["logloss"])
    print(f"fold {k}: P2u ll={s['logloss']:.4f} diff={diffs[-1]:+.4f}", flush=True)
diffs = np.array(diffs)
p2u_out = {"per_fold": diffs.tolist(), "mean": float(diffs.mean()),
           "se": float(diffs.std(ddof=1) / np.sqrt(5))}

out = {"ece_bootstrap": ece_boot, "auc_foldavg": auc_fold,
       "auc_within_ai": within, "auc_pooled": pooled, "p1_minus_p2u": p2u_out}
with open("../output/r7_extras.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"p1_minus_p2u": p2u_out}, indent=1))
