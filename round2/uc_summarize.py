"""Summarize U-C calibration replications (uc-reps/) into uc-summary.json.

Implements the AMENDED outcome rule (pass-2 skeptic required repair,
adopted 2026-11-20; see ROUND2-RECORD.md):
 (i) branch (a) for a graded comparison (H1, P1-P0) requires c-hat in the
     [1.35, 1.65] band AND x1.5-corrected t-interval coverage in
     [0.90, 0.99] under T1; for H1 also the T0 two-test-bar
     false-clearance rate within Monte Carlo error of nominal (0.025).
 (ii) the band test applies to a bootstrap Monte Carlo CI for c-hat; a CI
     straddling a band edge sends the affected comparison to branch (c)
     unless branch (b)'s single-factor test rescues both comparisons.
 (iii) P1-P3b is calibrated by the H2 series (identical construction);
     Pmetr-P1 by the MP series. Factors are applied only to calibrated
     comparisons.
Branch (b): the two graded c-hats within 0.3 of each other, both MC CIs
containing the common factor c*, and coverage at c* in [0.90, 0.99] for
both -> the printed factor is replaced by c* (2 dp) everywhere.
Otherwise branch (c) per affected comparison (demotion).
"""
import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
T975 = float(stats.t.ppf(0.975, 4))     # 2.776
T_BONF = float(stats.t.ppf(0.9875, 4))  # two-test Bonferroni crit, ~3.495
RNG = np.random.default_rng(20261122)
N_BOOT = 4000
COMPS = ("H1", "H2", "D10", "MP")


def boot_ci_c(m, s):
    """Bootstrap percentile CI for c = sd(m)/mean(s) over replications."""
    n = len(m)
    cs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        mb, sb = m[idx], s[idx]
        cs[b] = mb.std(ddof=1) / sb.mean() if sb.mean() > 0 else np.nan
    lo, hi = np.nanpercentile(cs, [2.5, 97.5])
    return float(lo), float(hi)


out = {}
for truth in ("T1", "T0"):
    recs = []
    failed = 0
    for p in sorted(glob.glob(os.path.join(HERE, "uc-reps", f"{truth}_*.json"))):
        r = json.load(open(p))
        if "failed" in r:
            failed += 1
        else:
            recs.append(r)
    entry = {"n_ok": len(recs), "n_failed": failed, "comparisons": {}}
    for comp in COMPS:
        if not recs or comp not in recs[0]["comparisons"]:
            continue
        m = np.array([r["comparisons"][comp]["mean"] for r in recs])
        s = np.array([r["comparisons"][comp]["se"] for r in recs])
        if len(m) < 10:
            continue
        sd_m = float(m.std(ddof=1))
        c_hat = sd_m / float(s.mean())
        lo, hi = boot_ci_c(m, s)
        cov = {}
        for f in (1.0, 1.5, round(c_hat, 4)):
            cov[str(f)] = float(np.mean(np.abs(m - m.mean()) <= T975 * f * s))
        e = {"mean_of_means": float(m.mean()), "sd_of_means": sd_m,
             "mean_se": float(s.mean()), "median_se": float(np.median(s)),
             "c_hat": float(c_hat), "c_mc_ci95": [lo, hi],
             "coverage_t975": cov}
        if truth == "T0":
            t = m / s
            bar_rate = float(np.mean(np.abs(m) > T_BONF * 1.5 * s))
            bar_mcse = float(np.sqrt(bar_rate * (1 - bar_rate) / len(m)))
            e["null_clearance"] = {
                "P_|mean|>2se": float(np.mean(np.abs(m) > 2 * s)),
                "P_|t|>2.776": float(np.mean(np.abs(t) > T975)),
                "P_|t|>bonf3.495": float(np.mean(np.abs(t) > T_BONF)),
                "two_test_bar_x1.5_rate": bar_rate,
                "two_test_bar_x1.5_mc_se": bar_mcse,
                "two_test_bar_pass": bool(bar_rate <= 0.025 + 1.96 * bar_mcse),
            }
        entry["comparisons"][comp] = e
    out[truth] = entry

# ---- amended-rule branch determination (graded comparisons H1, D10) ----
det = {}
t1 = out.get("T1", {}).get("comparisons", {})
t0 = out.get("T0", {}).get("comparisons", {})
graded = {}
for comp in ("H1", "D10"):
    e = t1.get(comp)
    if e is None:
        graded[comp] = {"branch": "UNDETERMINED — no T1 replications"}
        continue
    lo, hi = e["c_mc_ci95"]
    band_ok = lo >= 1.35 and hi <= 1.65
    cov15 = e["coverage_t975"].get("1.5")
    cov_ok = cov15 is not None and 0.90 <= cov15 <= 0.99
    null_ok = True
    if comp == "H1":
        nc = t0.get("H1", {}).get("null_clearance", {})
        null_ok = bool(nc.get("two_test_bar_pass", False))
    graded[comp] = {"c_hat": e["c_hat"], "c_mc_ci95": e["c_mc_ci95"],
                    "band_ok_ci_within_1.35_1.65": bool(band_ok),
                    "coverage_at_1.5": cov15, "coverage_ok": bool(cov_ok),
                    "null_bar_ok": bool(null_ok),
                    "branch_a": bool(band_ok and cov_ok and null_ok)}
if all(g.get("branch_a") for g in graded.values() if "branch_a" in g) \
        and len(graded) == 2:
    det["outcome"] = "(a) x1.5 stands as calibrated for H1 and P1-P0"
else:
    # branch (b) single-factor test
    try:
        c1, c2 = graded["H1"]["c_hat"], graded["D10"]["c_hat"]
        c_star = round((c1 + c2) / 2.0, 2)
        close = abs(c1 - c2) <= 0.3
        ci_ok = all(g["c_mc_ci95"][0] <= c_star <= g["c_mc_ci95"][1]
                    for g in graded.values())
        cov_ok_b = True
        for comp in ("H1", "D10"):
            e = t1[comp]
            covs = e["coverage_t975"]
            key = str(round(e["c_hat"], 4))
            # coverage at c* recomputed from the stored per-factor table is
            # unavailable for c* itself; use coverage at own c-hat as proxy
            # only if c* equals it to 2dp, else mark for manual recompute
            cov_ok_b = cov_ok_b and (key in covs)
        det["branch_b_test"] = {"c_hats": [c1, c2], "c_star": c_star,
                                "within_0.3": bool(close),
                                "both_mc_ci_contain_c_star": bool(ci_ok),
                                "note": "coverage at c* must be recomputed "
                                        "from uc-reps before branch (b) "
                                        "can be declared"}
        det["outcome"] = ("(b) CANDIDATE — single factor c*=%.2f; verify "
                          "coverage at c* in [0.90, 0.99] for both, else "
                          "(c) per affected comparison" % c_star) \
            if (close and ci_ok) else \
            "(c) demotion for each graded comparison failing branch (a)"
    except Exception as ex:  # noqa: BLE001
        det["outcome"] = f"UNDETERMINED ({ex})"
det["graded"] = graded
det["also_calibrated"] = {
    comp: {"c_hat": t1[comp]["c_hat"], "c_mc_ci95": t1[comp]["c_mc_ci95"],
           "coverage_at_1.5": t1[comp]["coverage_t975"].get("1.5")}
    for comp in ("H2", "MP") if comp in t1}
det["note_p3b"] = "P1-P3b is calibrated by the H2 series (identical construction)."
out["branch_determination"] = det
out["t_crit"] = {"t975_4df": T975, "t_bonf_two_test_4df": T_BONF}
json.dump(out, open(os.path.join(HERE, "uc-summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
