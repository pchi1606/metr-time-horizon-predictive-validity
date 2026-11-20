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

# ---- round-5 additions (panel responses) ----
r5 = json.load(open("output/r5_analyses.json"))
r5s = json.load(open("output/r5_p3scale.json"))

# LOFO means for both hypotheses (Table: LOFO) + fold-4 share of H2 margin
for hyp, exp in [("H1", [-0.217, -0.236, -0.250, -0.226, -0.264]),
                 ("H2", [-0.461, -0.425, -0.399, -0.492, -0.314])]:
    for k in range(5):
        check(f"LOFO {hyp} excl fold {k}",
              r5["lofo"][hyp]["lofo"][f"excl_fold_{k}"]["mean"], exp[k],
              tol=0.0015)
check("H2 fold-4 share of margin", r5["lofo"]["H2_fold4_share_of_margin"],
      0.400, tol=0.002)
check("H1 fold-3 diff (Table: per-fold scores)",
      r5["lofo"]["H1_fold3_diff_unrounded"], -0.287, tol=0.001)

# ECE recomputed from the printed bins via weight shares (Tables: ECE bins)
for m, exp in [("P1", 0.031), ("P2", 0.037), ("P3", 0.155), ("P0", 0.024)]:
    check(f"ECE {m} from printed bins",
          r5["ece_bin_weight_shares"][m]["ece_from_printed_table"], exp,
          tol=0.001)

# pooled (run-level) log-loss, same ordering as fold means
for m, exp in [("P1", 0.393), ("P0", 0.389), ("P2", 0.634), ("P3", 0.801)]:
    check(f"pooled log-loss {m}", r5["pooled_logloss_sqrtweighted"][m], exp,
          tol=0.0015)

# P0 same-specification control: ratios cluster at 1 (Table: P0 control)
summ = r5["p0_control"]["summary"]
check("P0 control ratio min", summ["p0_ratio_min"], 0.693)
check("P0 control ratio max", summ["p0_ratio_max"], 1.092)
check("P0 control ratio median", summ["p0_ratio_median"], 1.001)

# per-AI universality is a point-estimate statement: all 40 diffs negative
neg = sum(1 for pair in ["P1_vs_P2", "P1_vs_P3"]
          for v in r5["per_ai_diff_with_se"][pair].values() if v["mean"] < 0)
check("per-AI negative point estimates (of 40)", float(neg), 40.0, tol=0.0)

# P3 identified quantities: effective slope beta*b vs P0's fitted slope
bb = [f["beta_times_b"] for f in r5s["folds"]]
check("P3 beta*b min", min(bb), 1.751)
check("P3 beta*b max", max(bb), 2.094)
p0s = [f["p0_slope_fold"] for f in r5s["folds"]]
check("P0 slope min (folds)", min(p0s), 0.540)
check("P0 slope max (folds)", max(p0s), 0.635)
check("P0 slope full data", r5s["p0_slope_full_data"], 0.578)

# --- round 6: rescore add-on (r5_rescore.json) ---
rs = json.load(open("output/r5_rescore.json"))
wa = rs["within_ai_auc"]
check("within-AI AUC P1 (duration ranking)", 0.898, wa["P1"]["mean"], 0.002)
check("within-AI AUC P2 (constant within AI)", 0.500, wa["P2"]["mean"], 0.002)
rc = rs["p3_recalibration_crossfitted"]
check("P3 pooled OOF log-loss (raw)", 0.801, rc["raw"]["logloss"], 0.002)
check("P3 recalibrated (Platt, cross-fitted) log-loss", 0.425, rc["platt"]["logloss"], 0.002)
check("P3 recalibrated (Platt) ECE", 0.033, rc["platt"]["ece_ew10"], 0.002)
md = rs["measured_only_logloss_diffs"]
check("measured-only P1-P2 mean", -0.243, md["P1-P2"]["mean"], 0.002)
check("measured-only P1-P2 SE", 0.063, md["P1-P2"]["se"], 0.002)
check("measured-only P1-P3 mean", -0.337, md["P1-P3"]["mean"], 0.002)
check("measured-only P1-P3 SE", 0.139, md["P1-P3"]["se"], 0.002)
pc = rs["pchip_fold3"]
check("PCHIP fold3 out-of-range runs", 947, pc["n_out_of_range"], 0)
check("PCHIP fold3 log-loss flat", 0.3962, pc["logloss_flat"], 0.0005)
check("PCHIP fold3 log-loss as published", 0.3964, pc["logloss_as_published"], 0.0005)
check("PCHIP fold3 max |dp|", 0.014, pc["max_abs_dp_out_of_range"], 0.001)

# --- round 7: P3 variant + pooled AUC + fold-spread stats ---
v = json.load(open("output/r6_p3variant.json"))
check("P1-P3b log-loss mean", -0.028, v["P1_minus_P3b"]["logloss_mean"], 0.002)
check("P1-P3b log-loss SE", 0.009, v["P1_minus_P3b"]["logloss_se"], 0.002)
check("P3b beta*b min (BCE projection)", 0.73, min(f["beta_b_bce"] for f in v["folds"]), 0.02)
check("P3b beta*b max (BCE projection)", 0.86, max(f["beta_b_bce"] for f in v["folds"]), 0.02)
import numpy as _np
_p0h = _np.load(open("output/r5_p0_fold_horizons.npy", "rb")).T
_r0 = _np.array([_np.max(r) / _np.min(r) for r in _p0h])
check("P0 fold max/min worst", 1.97, float(_np.max(_r0)), 0.02)
check("P0 fold max/min median", 1.55, float(_np.median(_r0)), 0.02)

# --- round 8: extras (r7_extras.json) ---
ex = json.load(open("output/r7_extras.json"))
check("ECE contrast P3-P1 (bootstrap)", 0.124, ex["ece_bootstrap"]["P3_minus_P1"]["diff"], 0.002)
check("ECE contrast CI lower", 0.084, ex["ece_bootstrap"]["P3_minus_P1"]["ci95"][0], 0.002)
check("ECE contrast CI upper", 0.151, ex["ece_bootstrap"]["P3_minus_P1"]["ci95"][1], 0.002)
check("P1-P2u (unweighted mean) mean", -0.290, ex["p1_minus_p2u"]["mean"], 0.002)
check("P1-P2u SE", 0.029, ex["p1_minus_p2u"]["se"], 0.002)
check("AUC fold-avg P0", 0.920, ex["auc_foldavg"]["P0"]["mean"], 0.002)
check("AUC all-pooled P3", 0.917, ex["auc_pooled"]["P3"], 0.002)

# --- round 11: stage-1 penalty sensitivity (r11_stage1_ridge.json) ---
r11 = json.load(open("output/r11_stage1_ridge.json"))
_m = {row["lambda"]: row for row in r11["means"]}
check("R11 ridge lam=0 logloss", 0.812, _m[0.0]["logloss"], 0.002)
check("R11 ridge lam=0 extreme", 0.65, _m[0.0]["frac_extreme"], 0.006)
check("R11 ridge lam=0.003 logloss", 0.447, _m[0.003]["logloss"], 0.002)
check("R11 ridge lam=0.003 extreme", 0.00, _m[0.003]["frac_extreme"], 0.001)
check("R11 ridge lam=0.003 eff slope", 0.24, _m[0.003]["eff_slope"], 0.006)
check("R11 ridge lam=0.3 logloss", 0.623, _m[0.3]["logloss"], 0.002)
check("R11 ridge anchors all match", 5, sum(1 for f in r11["folds"] if f["lambda0_anchor_match"]), 0)

# --- round 14: P3 convergence demonstration (p3_convergence.json) ---
conv = json.load(open("output/p3_convergence.json"))
cs = conv["summary"]
check("R14 conv grad inf max (x1e-4)", cs["max_grad_inf_locked"] * 1e4, 2.504, tol=0.01)
check("R14 conv grad tightened max (x1e-6)", cs["max_grad_inf_tightened"] * 1e6, 3.723, tol=0.01)
check("R14 conv dObjective max (x1e-6)", cs["max_delta_objective"] * 1e6, 0.888, tol=0.01)
_ds = [f["tightened"]["eff_slope_beta_b"] - f["locked"]["eff_slope_beta_b"] for f in conv["folds"]]
_dl = [f["tightened"]["logloss"] - f["locked"]["logloss"] for f in conv["folds"]]
_dsp = [f["tightened"]["implied_logit_span_test"] - f["locked"]["implied_logit_span_test"] for f in conv["folds"]]
check("R14 conv dSlope min", min(_ds), 0.711)
check("R14 conv dSlope max", max(_ds), 1.146)
check("R14 conv dSpan min", min(_dsp), 9.294, tol=0.01)
check("R14 conv dSpan max", max(_dsp), 14.624, tol=0.01)
check("R14 conv dLogLoss min", min(_dl), 0.232)
check("R14 conv dLogLoss max", max(_dl), 1.042)
check("R14 conv tightened worse all folds", float(sum(1 for d in _dl if d > 0)), 5.0, tol=0.0)
_p1ll = np.array([f["P1"]["logloss"] for f in cv["folds"]])
_dtight = _p1ll - np.array([f["tightened"]["logloss"] for f in conv["folds"]])
check("R14 H2 margin at tightened points mean", _dtight.mean(), -1.041)
check("R14 H2 margin at tightened points SE", _dtight.std(ddof=1) / math.sqrt(5), 0.290, tol=0.003)

# --- round 22/23: METR-spec refit (r22_metr_spec.json), P3r (extended grid) ---
ms22 = json.load(open("output/r22_metr_spec.json"))
_s22 = ms22["full_data"]["summary"]
check("R22 spec ratio median", _s22["ratio_spec_over_published_median"], 1.000, tol=0.001)
check("R22 spec ratio min", _s22["ratio_spec_over_published_range"][0], 0.9999, tol=0.0005)
check("R22 spec ratio max", _s22["ratio_spec_over_published_range"][1], 1.0002, tol=0.0005)
check("R22 spec AIs within band (of 20)", float(_s22["n_within_0.85_1.10"]), 20.0, tol=0.0)
_pa22 = ms22["full_data"]["per_ai"]
_o46 = _pa22["Claude Opus 4.6 (Inspect)"]
check("R22 Opus 4.6 spec horizon", _o46["horizon_min"], 718.9, tol=0.05)
check("R22 Opus 4.6 own slope", _o46["slope"], -0.412, tol=0.001)
check("R22 Opus 4.6 counterfactual horizon", _o46["counterfactual_shared_slope_min"], 497.7, tol=0.05)
check("R22 Opus 4.6 P0 horizon", _o46["p0_shared_slope_min"], 497.8, tol=0.05)
_sl22 = sorted(d["slope"] for d in _pa22.values())
check("R22 flattest slope (Opus 4.6)", _sl22[-1], -0.412, tol=0.001)
check("R22 second-flattest slope", _sl22[-2], -0.465, tol=0.001)
check("R22 steepest slope", _sl22[0], -0.694, tol=0.001)
_cvd22 = ms22["cv"]["primary_weights_recomputed"]
check("R22 Pmetr-P1 diff mean", _cvd22["diff_vs_p1"]["mean"], 0.004)
check("R22 Pmetr-P1 diff SE", _cvd22["diff_vs_p1"]["se"], 0.008, tol=0.003)
check("R22 Pmetr-P0 diff mean", _cvd22["diff_vs_p0"]["mean"], 0.008)
check("R22 Pmetr-P0 diff SE", _cvd22["diff_vs_p0"]["se"], 0.003, tol=0.003)
_ob = ms22["opus46"]
check("R22 Opus 4.6 weight share >=256min", _ob["weight_share_above_256min"], 0.304, tol=0.001)
_bd = {tuple(x["bucket_min"]): x for x in _ob["empirical_buckets"]}
check("R22 Opus 4.6 bucket 256-960 success", _bd[(256, 960)]["weighted_success"], 0.60)
check("R22 Opus 4.6 bucket 256-960 weight share", _bd[(256, 960)]["weight_share"], 0.278, tol=0.001)
_fh22 = [f["opus46_horizon_min"] for f in _cvd22["folds"]]
for _k, _exp in enumerate([720.7, 966.1, 677.5, 869.6, 477.9]):
    check(f"R22 Opus 4.6 spec fold horizon {_k}", _fh22[_k], _exp, tol=0.05)

p3rx = json.load(open("output/r22_p3r_extended.json"))
check("R23 P3r folds selecting 0.0003 (of 5)",
      float(sum(1 for x in p3rx["selected_lambda_by_fold"] if x == 0.0003)), 5.0, tol=0.0)
_pp3 = p3rx["paired_p1_minus_p3r"]
check("R23 P1-P3r log-loss mean", _pp3["logloss"]["mean"], -0.012)
check("R23 P1-P3r log-loss SE", _pp3["logloss"]["se"], 0.012, tol=0.003)
check("R23 P1-P3r locked rule pass (0=False)", float(_pp3["logloss"]["locked_rule_pass"]), 0.0, tol=0.0)
check("R23 P1-P3r Brier mean", _pp3["brier"]["mean"], -0.004)
check("R23 P1-P3r Brier SE", _pp3["brier"]["se"], 0.005, tol=0.003)
check("R23 P1-P3r AUC mean", _pp3["auc"]["mean"], 0.007)
check("R23 P1-P3r AUC SE", _pp3["auc"]["se"], 0.006, tol=0.003)
for _met, _exp in [("logloss", 0.406), ("brier", 0.132), ("auc", 0.912)]:
    _mu, _se = mean_se([f["P3r"][_met] for f in p3rx["folds"]])
    check(f"R23 P3r {_met} mean (Table 2)", _mu, _exp)
check("R23 P3r max tightened drift (x1e-6)",
      p3rx["stopping_sensitivity_new_arm"]["max_abs_drift_logloss"] * 1e6, 3.47, tol=0.01)
_es3 = [f["selected_eff_slope"] for f in p3rx["folds"]]
check("R23 P3r eff slope min", min(_es3), 0.508, tol=0.002)
check("R23 P3r eff slope max", max(_es3), 0.544, tol=0.002)
p3r22 = json.load(open("output/r22_p3r.json"))
check("R22 P1-P3r (deposited grid) log-loss mean", p3r22["paired_p1_minus_p3r"]["logloss"]["mean"], -0.054)
check("R22 P1-P3r (deposited grid) log-loss SE", p3r22["paired_p1_minus_p3r"]["logloss"]["se"], 0.017, tol=0.003)
_dmax = max(abs(p3rx["folds"][k]["outer_by_lambda"]["0.003"]["logloss"]
                - p3r22["folds"][k]["P3r"]["logloss"]) for k in range(5))
check("R23 Step-0 replication max diff at lam=0.003", _dmax, 0.0, tol=1e-9)

# --- round 24: REG-P1-5 family-cluster bootstrap (bootstrap_r22/summary.json) ---
bs = json.load(open("output/bootstrap_r22/summary.json"))
check("R24 bootstrap n_files", float(bs["n_files"]), 300.0, tol=0.0)
check("R24 bootstrap n_ok", float(bs["n_ok"]), 294.0, tol=0.0)
check("R24 bootstrap n_failed", float(bs["n_failed"]), 6.0, tol=0.0)
assert bs["failed_resamples"] == [9, 193, 244, 283, 292, 293], bs["failed_resamples"]
print("PASS: R24 bootstrap failed indices = [9, 193, 244, 283, 292, 293]")
for _rs in bs["failed_resamples"]:
    _rf = json.load(open(f"output/bootstrap_r22/resample_{_rs:04d}.json"))
    assert _rf["status"] == "failed" and \
        _rf["error"] == "ValueError: `y` must contain only finite values.", _rf
print("PASS: R24 all 6 failed resamples carry the identical recorded error")
_h1b, _h2b = bs["H1_p1_minus_p2"], bs["H2_p1_minus_p3"]
check("R24 H1 bootstrap SE", _h1b["bootstrap_se"], 0.044, tol=0.001)
check("R24 H1 bootstrap mean", _h1b["mean"], -0.247)
check("R24 H1 bootstrap median", _h1b["median"], -0.249)
check("R24 H1 bootstrap CI lower", _h1b["percentile_ci95"][0], -0.327)
check("R24 H1 bootstrap CI upper", _h1b["percentile_ci95"][1], -0.155)
check("R24 H1 frac means < 0", _h1b["frac_resamples_mean_below_0"], 1.0, tol=0.0)
check("R24 H1 locked-rule pass rate", _h1b["locked_rule_pass_rate"], 0.993, tol=0.001)
check("R24 H1 locked value", _h1b["locked_value"], -0.238)
check("R24 H1 locked - median", _h1b["locked_minus_bootstrap_median"], 0.010, tol=0.001)
check("R24 H2 bootstrap SE", _h2b["bootstrap_se"], 0.093, tol=0.001)
check("R24 H2 bootstrap mean", _h2b["mean"], -0.197)
check("R24 H2 bootstrap median", _h2b["median"], -0.181)
check("R24 H2 bootstrap CI lower", _h2b["percentile_ci95"][0], -0.414)
check("R24 H2 bootstrap CI upper", _h2b["percentile_ci95"][1], -0.060)
check("R24 H2 frac means < 0", _h2b["frac_resamples_mean_below_0"], 1.0, tol=0.0)
check("R24 H2 locked-rule pass rate", _h2b["locked_rule_pass_rate"], 0.820, tol=0.001)
check("R24 H2 locked value", _h2b["locked_value"], -0.418)
check("R24 H2 locked - median", _h2b["locked_minus_bootstrap_median"], -0.237, tol=0.001)

# ===== Round 26: REG-P1-8 bracket run (output/r26_p3r_bracket.json) =====
_br = json.load(open("output/r26_p3r_bracket.json"))
check("R26 bracket grid size", float(len(_br["grid"])), 13.0, tol=0.0)
check("R26 grid smallest nonzero", min(g for g in _br["grid"] if g > 0), 1e-6, tol=0.0)
for _f in _br["folds"]:
    check(f"R26 fold {_f['fold']} selected lambda",
          _f["selected_lambda"], 0.0003, tol=0.0)
    _im = _f["inner_mean_logloss"]
    check(f"R26 fold {_f['fold']} bracket below (1e-4 - sel > 0)",
          1.0 if _im["0.0001"] > _im["0.0003"] else 0.0, 1.0, tol=0.0)
    check(f"R26 fold {_f['fold']} bracket above (1e-3 - sel > 0)",
          1.0 if _im["0.001"] > _im["0.0003"] else 0.0, 1.0, tol=0.0)
_r23 = json.load(open("output/r22_p3r_extended.json"))
for _m in ("logloss", "brier", "auc"):
    check(f"R26 paired P1-P3r {_m} mean unchanged vs r23",
          _br["paired_p1_minus_p3r"][_m]["mean"],
          _r23["paired_p1_minus_p3r"][_m]["mean"], tol=1e-9)
    check(f"R26 paired P1-P3r {_m} SE unchanged vs r23",
          _br["paired_p1_minus_p3r"][_m]["se"],
          _r23["paired_p1_minus_p3r"][_m]["se"], tol=1e-9)
check("R26 bracket rule pass (False = 0)",
      1.0 if _br["paired_p1_minus_p3r"]["logloss"]["locked_rule_pass"] else 0.0,
      0.0, tol=0.0)

# ===== Round 27: REG-P1-9 ECE cancellation analysis =====
# (output/r27_ece_cancellation.json; script code/r27_ece_cancellation.py)
_ec = json.load(open("output/r27_ece_cancellation.json"))
_bs = json.load(open("output/bootstrap_r22/summary.json"))
check("R27 n_rows analysed", float(_ec["n_rows"]), 23235.0, tol=0.0)
check("R27 total weight", _ec["total_weight"], 20.0, tol=1e-6)
check("R27 bootstrap n_ok", float(_bs["n_ok"]), 294.0, tol=0.0)
check("R27 bootstrap n_files", float(_bs["n_files"]), 300.0, tol=0.0)
for _h, _key, _wc in (("H1", "H1_p1_minus_p2", 0.973),
                      ("H2", "H2_p1_minus_p3", 0.803)):
    _passes = round(_bs[_key]["locked_rule_pass_rate"] * _bs["n_ok"])
    check(f"R27 {_h} worst-case pass rate (passes/300)",
          _passes / _bs["n_files"], _wc, tol=0.001)
import statistics as _st27
_ecr = json.load(open("output/r26_ece_ranges.json"))
for _p, _pool, _star, _canc, _share, _med in (
        ("P1", 0.031, 0.066, 0.035, 0.53, 0.064),
        ("P3", 0.155, 0.171, 0.016, 0.09, 0.177)):
    _r = _ec["predictors"][_p]
    check(f"R27 {_p} pooled ECE (printed)", _r["pooled_ece"], _pool, tol=0.001)
    check(f"R27 {_p} weight-averaged per-AI ECE (printed)",
          _r["weight_averaged_per_ai_ece"], _star, tol=0.001)
    check(f"R27 {_p} cancelled ECE (printed)", _r["cancellation"], _canc, tol=0.001)
    check(f"R27 {_p} cancellation share (printed)",
          _r["cancellation_share_of_star"], _share, tol=0.01)
    check(f"R27 {_p} identity star - pooled - cancelled",
          _r["weight_averaged_per_ai_ece"] - _r["pooled_ece"] - _r["cancellation"],
          0.0, tol=1e-9)
    check(f"R27 {_p} per-AI median ECE (printed)",
          _r["per_ai_median"], _med, tol=0.001)
    _vals = list(_r["per_ai_ece"].values())
    check(f"R27 {_p} per-AI ECE min vs r26 ranges",
          min(_vals), _ecr["per_ai"][_p]["min"], tol=0.015)
    check(f"R27 {_p} per-AI ECE max vs r26 ranges",
          max(_vals), _ecr["per_ai"][_p]["max"], tol=0.015)
    check(f"R27 {_p} sign-disagreement bins",
          float(len(_r["mixed_sign_bins"])),
          float({"P1": 10, "P3": 7}[_p]), tol=0.0)

# --- Round 28: SW27-2 distinct-family distribution, recomputed from
# the deposited resample records and the deposited partition
# algorithm (data.family_folds on each record's partition seed) ---
import glob as _glob28
import statistics as _st28

sys.path.insert(0, "code")
from data import family_folds as _ff28
import pandas as _pd28

_recs28 = []
for _f in sorted(_glob28.glob("output/bootstrap_r22/resample_*.json")):
    _d = json.load(open(_f))
    if _d.get("status") == "ok":
        _recs28.append(_d)
_nd28 = [_d["n_families_drawn_distinct"] for _d in _recs28]
check("R28 distinct families per resample: mean",
      _st28.mean(_nd28), 50.1, tol=0.05)
check("R28 distinct families per resample: sd",
      _st28.stdev(_nd28), 2.7, tol=0.05)
check("R28 distinct families per resample: median",
      float(_st28.median(_nd28)), 50.0, tol=0.0)
check("R28 distinct families per resample: min",
      float(min(_nd28)), 43.0, tol=0.0)
check("R28 distinct families per resample: max",
      float(max(_nd28)), 59.0, tol=0.0)
_fams28 = sorted(int(k) for k in _recs28[0]["family_counts"].keys())
_stub28 = _pd28.DataFrame({"f": _fams28})
_te28, _tr28 = [], []
for _d in _recs28:
    _fm = _ff28(_stub28, n_folds=5, seed=int(_d["partition_seed"]))
    _drawn = {int(k) for k, v in _d["family_counts"].items() if v > 0}
    for _k in range(5):
        _te28.append(len({f for f in _drawn if _fm[f] == _k}))
        _tr28.append(len(_drawn) - _te28[-1])
check("R28 test-fold distinct families: mean",
      _st28.mean(_te28), 10.0, tol=0.05)
check("R28 test-fold distinct families: min",
      float(min(_te28)), 4.0, tol=0.0)
check("R28 test-fold distinct families: max",
      float(max(_te28)), 15.0, tol=0.0)
check("R28 train-fold distinct families: mean",
      _st28.mean(_tr28), 40.1, tol=0.05)
check("R28 train-fold distinct families: min",
      float(min(_tr28)), 31.0, tol=0.0)
check("R28 train-fold distinct families: max",
      float(max(_tr28)), 49.0, tol=0.0)
check("R28 per-fold means sum to resample mean",
      _st28.mean(_te28) + _st28.mean(_tr28), _st28.mean(_nd28),
      tol=1e-9)

# --- Round 28: SW27-3 locked-vs-tightened crosswalk (Appendix D) ---
_pc28 = json.load(open("output/p3_convergence.json"))
_pr_obj = (3.6472, 3.4898, 3.3472, 3.4535, 3.4299)
for _f in _pc28["folds"]:
    _k = _f["fold"]
    check(f"R28 crosswalk fold {_k} objective locked (printed 4dp)",
          round(_f["locked"]["objective"], 4), _pr_obj[_k], tol=5e-5)
    check(f"R28 crosswalk fold {_k} objective tightened (printed 4dp)",
          round(_f["tightened"]["objective"], 4), _pr_obj[_k], tol=5e-5)
    check(f"R28 crosswalk fold {_k} eff slope locked",
          _f["locked"]["eff_slope_beta_b"],
          (1.75, 2.09, 1.89, 1.77, 1.85)[_k], tol=0.005)
    check(f"R28 crosswalk fold {_k} eff slope tightened",
          _f["tightened"]["eff_slope_beta_b"],
          (2.46, 3.03, 3.03, 2.66, 2.87)[_k], tol=0.005)
    check(f"R28 crosswalk fold {_k} log-loss locked",
          _f["locked"]["logloss"],
          (0.583, 0.765, 0.919, 0.519, 1.274)[_k], tol=0.0005)
    check(f"R28 crosswalk fold {_k} log-loss tightened",
          _f["tightened"]["logloss"],
          (0.885, 1.262, 1.960, 0.751, 2.316)[_k], tol=0.0005)
check("R28 max |delta objective| under 1e-6 (printed)",
      _pc28["summary"]["max_delta_objective"], 0.0, tol=1e-6)

# --- Round 28: SW27-7 equal-task paired P1-P0 (strengthen.json) ---
_et28 = json.load(open("output/strengthen.json"))[
    "equal_task_weight_logloss"]
import math as _math28
_d28 = [a - b for a, b in zip(_et28["P1"]["per_fold"],
                              _et28["P0"]["per_fold"])]
_m28 = _st28.mean(_d28)
_se28 = _st28.stdev(_d28) / _math28.sqrt(5)
check("R28 equal-task paired P1-P0 mean (printed)", _m28, 0.002,
      tol=0.001)
check("R28 equal-task paired P1-P0 SE (printed)", _se28, 0.009,
      tol=0.001)
check("R28 equal-task 90% CI lower (printed)", _m28 - 2.1318 * _se28,
      -0.016, tol=0.001)
check("R28 equal-task 90% CI upper (printed)", _m28 + 2.1318 * _se28,
      0.020, tol=0.001)

# --- Round 28: SW27-8 P3 mixed-sign bins are the middle bins 0.2-0.9
_mb28 = _ec["predictors"]["P3"]["mixed_sign_bins"]
check("R28 P3 mixed-sign bin count", float(len(_mb28)), 7.0, tol=0.0)
check("R28 P3 mixed-sign first bin index", float(min(_mb28)), 2.0,
      tol=0.0)
check("R28 P3 mixed-sign last bin index", float(max(_mb28)), 8.0,
      tol=0.0)

# ===== Round 29: consolidation pass (SW28-2 bounds; SW28-8 MC SEs) =====
# Bounds on the over-300 bootstrap location statistics under arbitrary
# placement of the six failed resamples, recomputed from the per-resample
# files (statistic = the resample's 'mean' for each hypothesis).
import glob as _glob
import numpy as _np
_r29 = {}
for _hyp in ("H1_p1_minus_p2", "H2_p1_minus_p3"):
    _vals = []
    for _fn in sorted(_glob.glob("output/bootstrap_r22/resample_*.json")):
        _d = json.load(open(_fn))
        if _d.get("status") == "ok":
            _vals.append(_d[_hyp]["mean"])
    _r29[_hyp] = _np.array(_vals)
_h1v, _h2v = _r29["H1_p1_minus_p2"], _r29["H2_p1_minus_p3"]
check("R29 H1 n completed", float(len(_h1v)), 294.0, tol=0.0)
check("R29 H2 n completed", float(len(_h2v)), 294.0, tol=0.0)
for _nm, _v, _pm, _plo, _phi in (
        ("H1", _h1v, (-0.2495, -0.2447), (-0.3435, -0.3270), (-0.1548, -0.1309)),
        ("H2", _h2v, (-0.2045, -0.1936), (-0.5014, -0.4136), (-0.0607, -0.0318))):
    _ms = [float(_np.mean(_np.concatenate([_v, [_v.min()] * _k,
                                            [_v.max()] * (6 - _k)])))
           for _k in range(7)]
    check(f"R29 {_nm} over-300 mean bound lo", min(_ms), _pm[0], tol=5e-5)
    check(f"R29 {_nm} over-300 mean bound hi", max(_ms), _pm[1], tol=5e-5)
    _los, _his = [], []
    for _k in range(7):
        _ext = _np.concatenate([_v, [_v.min()] * _k, [_v.max()] * (6 - _k)])
        _lo, _hi = _np.percentile(_ext, [2.5, 97.5])
        _los.append(_lo); _his.append(_hi)
    check(f"R29 {_nm} lower-endpoint range lo", min(_los), _plo[0], tol=5e-5)
    check(f"R29 {_nm} lower-endpoint range hi", max(_los), _plo[1], tol=5e-5)
    check(f"R29 {_nm} upper-endpoint range lo", min(_his), _phi[0], tol=5e-5)
    check(f"R29 {_nm} upper-endpoint range hi", max(_his), _phi[1], tol=5e-5)
# SW28-8: binomial Monte Carlo SEs of the pass rates (292/294, 241/294).
for _nm, _p, _printed in (("H1", 292 / 294, 0.005), ("H2", 241 / 294, 0.022)):
    check(f"R29 {_nm} pass-rate binomial MC SE",
          float(_np.sqrt(_p * (1 - _p) / 294)), _printed, tol=0.0015)

# --- Round-30 add-ons: saved-output invariants (the scripts assert their
# data anchors at run time; here the printed claims' load-bearing values) ---
_r30b = json.load(open("output/r30_clip_binding.json"))
_r30d = json.load(open("output/r30_stable_diagnostic.json"))
_r30s = json.load(open("output/r30_p1_sensitivity.json"))
_r30e = json.load(open("output/r30_p3r_ece.json"))
check("R30 locked clipped-vs-stable objective agreement",
      float(max(abs(f["locked"]["objective_clipped"]
                    - f["locked"]["objective_stable"])
                for f in _r30d["folds"])), 0.0, tol=1e-9)
check("R30 stable improvement max < 1e-6",
      float(max(f["improvement_stable_objective"] for f in _r30d["folds"])),
      0.0, tol=1e-6)
check("R30 H2 margin at stable-tightened points",
      float(_np.mean([cv["folds"][k]["P1"]["logloss"]
                      - _r30d["folds"][k]["stable_tightened"]["logloss"]
                      for k in range(5)])), -1.042, tol=1e-3)
check("R30 bounded-u variant identical to locked",
      _r30s["summary"]["bounded"]["max_abs_d_logloss"], 0.0, tol=0.0)
check("R30 sensitivity locked arm anchored to cv_scores",
      _r30s["summary"]["locked"]["max_abs_d_logloss"], 0.0, tol=0.0)
# R30 P3r fold anchors: recompute per-fold log-loss from the deposited
# r30 OOF predictions joined positionally to oof_predictions.jsonl
# (the generating script asserts the same join at run time).
_oof30 = [json.loads(l) for l in open("output/r30_p3r_oof.jsonl")]
_oofbase = [json.loads(l) for l in open("output/oof_predictions.jsonl")]
assert len(_oof30) == len(_oofbase) == 23235
for _f in range(5):
    _num = _den = 0.0
    for _a, _b in zip(_oof30, _oofbase):
        if _a["fold"] != _f:
            continue
        assert _a["task"] == _b["task"] and _a["j"] == _b["j"]
        _p = min(max(_a["P3r"], 1e-12), 1 - 1e-12)
        _num += _b["w"] * -(_b["y"] * _np.log(_p)
                            + (1 - _b["y"]) * _np.log(1 - _p))
        _den += _b["w"]
    check(f"R30 P3r anchor fold {_f}", _num / _den,
          p3rx["folds"][_f]["P3r"]["logloss"], tol=1e-9)
check("R30 P3r pooled ECE recomposes from its bins",
      float(sum(b["weight_share"] * abs(b["obs"] - b["pred"])
                for b in _r30e["bins"])), _r30e["pooled_ece"], tol=1e-9)
check("R30 binding runs fold 1", _r30b["folds"][1]["locked"]["n_binding"], 1390, tol=0)
check("R30 binding runs fold 2", _r30b["folds"][2]["locked"]["n_binding"], 366, tol=0)

# ===== Round 31: NW30-1 bounded bootstrap completion (bootstrap_r31) =====
# The completion run (code/r31_bootstrap_bounded.py) recomputed all 300
# resamples under the bounded P1 implementation at the deposited run's
# seed base, draws and partitions. The deposited bootstrap_r22 record is
# untouched and remains the basis of the retained worst-case bounds
# (R24/R29 above); the checks below key the printed all-300 statistics
# to the completion run's own records.
_b31 = json.load(open("output/bootstrap_r31/summary.json"))
check("R31 bootstrap n_files", float(_b31["n_files"]), 300.0, tol=0.0)
check("R31 bootstrap n_ok", float(_b31["n_ok"]), 300.0, tol=0.0)
check("R31 bootstrap n_failed", float(_b31["n_failed"]), 0.0, tol=0.0)
assert _b31["failed_resamples"] == []
assert _b31["comparison_vs_deposited_r22"][
    "resamples_where_clip_binds_at_solution"] == []
print("PASS: R31 no failed resamples; clip binds at no fitted solution")
_h1c, _h2c = _b31["H1_p1_minus_p2"], _b31["H2_p1_minus_p3"]
check("R31 H1 bootstrap SD", _h1c["bootstrap_se"], 0.043, tol=0.001)
check("R31 H1 bootstrap mean", _h1c["mean"], -0.247)
check("R31 H1 bootstrap median", _h1c["median"], -0.249)
check("R31 H1 bootstrap CI lower", _h1c["percentile_ci95"][0], -0.327)
check("R31 H1 bootstrap CI upper", _h1c["percentile_ci95"][1], -0.155)
check("R31 H1 locked-rule pass rate", _h1c["locked_rule_pass_rate"],
      0.993, tol=0.001)
check("R31 H1 locked - median", _h1c["locked_minus_bootstrap_median"],
      0.010, tol=0.001)
check("R31 H2 bootstrap SD", _h2c["bootstrap_se"], 0.095, tol=0.001)
check("R31 H2 bootstrap mean", _h2c["mean"], -0.198)
check("R31 H2 bootstrap median", _h2c["median"], -0.185)
check("R31 H2 bootstrap CI lower", _h2c["percentile_ci95"][0], -0.437)
check("R31 H2 bootstrap CI upper", _h2c["percentile_ci95"][1], -0.060)
check("R31 H2 locked-rule pass rate", _h2c["locked_rule_pass_rate"],
      0.817, tol=0.001)
check("R31 H2 locked - median", _h2c["locked_minus_bootstrap_median"],
      -0.233, tol=0.001)
for _hyp, _sum in (("H1_p1_minus_p2", _h1c), ("H2_p1_minus_p3", _h2c)):
    _ms, _ps = [], []
    for _i in range(300):
        _r = json.load(open(f"output/bootstrap_r31/resample_{_i:04d}.json"))
        assert _r["status"] == "ok"
        _ms.append(_r[_hyp]["mean"])
        _ps.append(bool(_r[_hyp]["locked_rule_pass"]))
    _ms = _np.array(_ms)
    check(f"R31 {_hyp} recomputed mean", float(_ms.mean()),
          _sum["mean"], tol=1e-9)
    check(f"R31 {_hyp} recomputed SD", float(_ms.std(ddof=1)),
          _sum["bootstrap_se"], tol=1e-9)
    check(f"R31 {_hyp} recomputed pass rate", float(_np.mean(_ps)),
          _sum["locked_rule_pass_rate"], tol=1e-12)
_same = 0
for _i in range(300):
    _a = json.load(open(f"output/bootstrap_r22/resample_{_i:04d}.json"))
    if _a.get("status") != "ok":
        continue
    _b = json.load(open(f"output/bootstrap_r31/resample_{_i:04d}.json"))
    assert _a["family_counts"] == _b["family_counts"] and \
        _a["partition_seed"] == _b["partition_seed"]
    _same += 1
assert _same == 294
print("PASS: R31 draws+partitions identical to the deposited run on all "
      "294 shared resamples")
_nd = _np.array([json.load(open(f"output/bootstrap_r31/resample_{_i:04d}.json"))[
    "n_families_drawn_distinct"] for _i in range(300)])
check("R31 distinct families mean", float(_nd.mean()), 50.1, tol=0.05)
check("R31 distinct families SD", float(_nd.std(ddof=1)), 2.7, tol=0.05)
check("R31 distinct families median", float(_np.median(_nd)), 50.0, tol=0.0)
check("R31 distinct families min", float(_nd.min()), 43.0, tol=0.0)
check("R31 distinct families max", float(_nd.max()), 59.0, tol=0.0)
_cmp = _b31["comparison_vs_deposited_r22"]
check("R31 H1 reproduction median |diff|",
      _cmp["H1_p1_minus_p2"]["median_abs_mean_diff"], 0.00002, tol=0.00001)
check("R31 H1 reproduction max |diff|",
      _cmp["H1_p1_minus_p2"]["max_abs_mean_diff"], 0.023, tol=0.0005)
check("R31 H2 reproduction median |diff|",
      _cmp["H2_p1_minus_p3"]["median_abs_mean_diff"], 0.0055, tol=0.0005)
check("R31 H2 reproduction max |diff|",
      _cmp["H2_p1_minus_p3"]["max_abs_mean_diff"], 0.087, tol=0.0005)
for _i, _v in _cmp["newly_completed"].items():
    _r = json.load(open(f"output/bootstrap_r31/resample_{int(_i):04d}.json"))
    check(f"R31 resample {_i} H1 mean", _r["H1_p1_minus_p2"]["mean"],
          _v["H1_mean"], tol=1e-9)
    check(f"R31 resample {_i} H2 mean", _r["H2_p1_minus_p3"]["mean"],
          _v["H2_mean"], tol=1e-9)
    assert _r["H1_p1_minus_p2"]["locked_rule_pass"] and \
        _r["H2_p1_minus_p3"]["locked_rule_pass"]
    assert not any(_d["p1_clip_binds_at_solution"]
                   for _d in _r["bounded_diagnostics"])
print("PASS: R31 all six newly completed resamples pass the locked rule "
      "for both hypotheses; the clip binds at none of their solutions")
assert -0.2495 <= _h1c["mean"] <= -0.2447 and \
    -0.2045 <= _h2c["mean"] <= -0.1936
print("PASS: R31 completed means inside the retained worst-case "
      "location bounds")
for _nm, _p, _printed in (("H1", 298 / 300, 0.005), ("H2", 245 / 300, 0.022)):
    check(f"R31 {_nm} pass-rate binomial MC SE",
          float(_np.sqrt(_p * (1 - _p) / 300)), _printed, tol=0.0015)

print()
if fails:
    print(f"VERIFICATION FAILED: {len(fails)} mismatches: {fails}")
    sys.exit(1)
print("ALL HEADLINE NUMBERS VERIFIED.")
