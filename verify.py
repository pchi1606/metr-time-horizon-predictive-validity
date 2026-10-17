"""Verify headline numbers from the reproduction outputs.

Recomputes every quantitative claim of the paper from output/*.json and
fails loudly on any mismatch beyond tolerance. Run after the pipeline:
    python3 verify.py
"""
import json
import math
import sys

TOL = 0.002
fails = []


def check(name, actual, expected, tol=TOL):
    ok = abs(actual - expected) <= tol
    print(f"{'PASS' if ok else 'FAIL'}: {name} = {actual:.4f} (paper {expected})")
    if not ok:
        fails.append(name)


def mean_se(v):
    import numpy as np
    v = np.array(v, float)
    return v.mean(), v.std(ddof=1) / math.sqrt(len(v))


cv = json.load(open("output/cv_scores.json"))
st = json.load(open("output/strengthen.json"))
p0 = {"P0_logloss_per_fold": st["p0_logloss_per_fold_sqrtweight"]}

# Table 1: log-loss
for m, exp_m, exp_s in [("P1", 0.394, 0.018), ("P2", 0.632, 0.019),
                        ("P3", 0.812, 0.135)]:
    mu, se = mean_se([f[m]["logloss"] for f in cv["folds"]])
    check(f"{m} log-loss mean", mu, exp_m)
    check(f"{m} log-loss SE", se, exp_s, tol=0.003)
mu, se = mean_se(p0["P0_logloss_per_fold"])
check("P0 log-loss mean", mu, 0.390)
check("P0 log-loss SE", se, 0.025, tol=0.003)

# paired differences
import numpy as np
d12 = np.array([f["P1"]["logloss"] - f["P2"]["logloss"] for f in cv["folds"]])
check("P1-P2 diff mean", d12.mean(), -0.238)
check("P1-P2 diff SE", d12.std(ddof=1) / math.sqrt(5), 0.033, tol=0.003)
d13 = np.array([f["P1"]["logloss"] - f["P3"]["logloss"] for f in cv["folds"]])
check("P1-P3 diff mean", d13.mean(), -0.418)
check("P1-P3 diff SE", d13.std(ddof=1) / math.sqrt(5), 0.122, tol=0.003)

# Brier and AUC
for m, exp_m in [("P1", 0.128), ("P2", 0.220), ("P3", 0.178)]:
    mu, _ = mean_se([f[m]["brier"] for f in cv["folds"]])
    check(f"{m} Brier mean", mu, exp_m)
for m, exp_m in [("P1", 0.919), ("P2", 0.679), ("P3", 0.905)]:
    mu, _ = mean_se([f[m]["auc"] for f in cv["folds"]])
    check(f"{m} AUC mean", mu, exp_m)

# calibration
for m, exp in [("P1", 0.031), ("P2", 0.037), ("P3", 0.155), ("P0", 0.024)]:
    check(f"ECE {m}", st["calibration"][m]["ece"], exp)

# equal-task weighting
for m, exp in [("P1", 0.349), ("P0", 0.347), ("P2", 0.629), ("P3", 0.723)]:
    check(f"equal-task {m}", st["equal_task_weight_logloss"][m]["mean"], exp)

# P3 worse than P2 on 3/5 folds
n = sum(1 for f in cv["folds"] if f["P3"]["logloss"] > f["P2"]["logloss"])
check("P3> P2 folds", float(n), 3.0, tol=0.0)

# Brier P3-P2 paired difference (secondary metric, n.s.)
d32b = np.array([f["P3"]["brier"] - f["P2"]["brier"] for f in cv["folds"]])
check("P3-P2 Brier diff mean", d32b.mean(), -0.042)
check("P3-P2 Brier diff SE", d32b.std(ddof=1) / math.sqrt(5), 0.036, tol=0.003)

# P3 per-fold diagnostics (Appendix D)
p3d = json.load(open("output/p3_diagnostics.json"))
exp_beta = [8.61, 12.10, 7.01, 6.76, 7.69]
exp_ext = [0.85, 0.46, 0.50, 0.76, 0.67]
for f, eb, ee in zip(p3d["folds"], exp_beta, exp_ext):
    check(f"P3 beta fold {f['fold']}", f["beta"], eb, tol=0.05)
    check(f"P3 extreme frac fold {f['fold']}", f["frac_extreme"], ee, tol=0.02)

print()
if fails:
    print(f"VERIFICATION FAILED: {len(fails)} mismatches: {fails}")
    sys.exit(1)
print("ALL HEADLINE NUMBERS VERIFIED.")
