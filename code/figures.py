"""Generate report figures from real analysis outputs (no mockups)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 600})

cv = json.load(open("../output/cv_scores.json"))
de = json.load(open("../output/descriptives.json"))
st = json.load(open("../output/strengthen.json"))
p0 = {"P0_logloss_per_fold": st["p0_logloss_per_fold_sqrtweight"]}

# Fig 1: fitted time-to-difficulty curve f
xs = np.array(de["f_curve"]["x_log2min"]); f = np.array(de["f_curve"]["f"])
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(2.0 ** xs, f, lw=1.8)
ax.set_xscale("log")
ax.set_xlabel("human time (minutes, log scale)")
ax.set_ylabel("fitted difficulty f(log2 t)")
ax.set_title("Fitted time-to-difficulty curve (P1 spline, all data)")
ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig("../output/fig1_f_curve.png"); plt.close(fig)

# Fig 2: reliability diagram
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for m, ls, lab in [("P1", "-", "P1 spline"), ("P2", "--", "P2 mean"),
                   ("P3", "-.", "P3 IRT"), ("P0", ":", "P0 logistic")]:
    bins = st["calibration"][m]["bins"]
    xp = [b["mean_pred"] for b in bins if b["n"] > 0]
    yp = [b["mean_obs"] for b in bins if b["n"] > 0]
    ax.plot(xp, yp, marker="o", ms=3, ls=ls, label=f"{lab} (ECE={st['calibration'][m]['ece']:.3f})")
ax.plot([0, 1], [0, 1], "k-", lw=0.8)
ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed success rate")
ax.set_title("Reliability diagram (pooled out-of-fold predictions)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig("../output/fig2_reliability.png"); plt.close(fig)

# Fig 3: per-fold log-loss grouped bars
folds = cv["folds"]
x = np.arange(5); wdt = 0.2
fig, ax = plt.subplots(figsize=(6.2, 3.4))
for k, m in enumerate(["P1", "P0", "P2", "P3"]):
    vals = [f[m]["logloss"] if m != "P0" else p0["P0_logloss_per_fold"][fi]
            for fi, f in enumerate(folds)]
    ax.bar(x + (k - 1.5) * wdt, vals, wdt, label=m)
ax.set_xticks(x); ax.set_xticklabels([f"fold {i}" for i in range(5)])
ax.set_ylabel("held-out weighted log-loss")
ax.set_title("Cross-validated log-loss per fold (lower is better)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig("../output/fig3_fold_logloss.png"); plt.close(fig)

# Fig 4: horizons (mine, fold mean +/- sd) vs METR published
mine = de["horizons_mine_min"]
hsd = st["horizon_fold_mean_sd_min"]
pub_map = {
    "Claude 3 Opus (Inspect)": "claude_3_opus_inspect",
    "Claude 3.5 Sonnet (Old) (Inspect)": "claude_3_5_sonnet_20240620_inspect",
    "Claude 3.5 Sonnet (New) (Inspect)": "claude_3_5_sonnet_20241022_inspect",
    "Claude 3.7 Sonnet (Inspect)": "claude_3_7_sonnet_inspect",
    "Claude 4 Opus (Inspect)": "claude_4_opus_inspect",
    "Claude 4.1 Opus (Inspect)": "claude_4_1_opus_inspect",
    "Claude Opus 4.5 (Inspect)": "claude_opus_4_5_inspect",
    "Claude Opus 4.6 (Inspect)": "claude_opus_4_6_inspect",
    "GPT-4 0314": "gpt_4", "GPT-4 1106 (Inspect)": "gpt_4_1106_inspect",
    "GPT-4 Turbo (Inspect)": "gpt_4_turbo_inspect", "GPT-4o (Inspect)": "gpt_4o_inspect",
    "GPT-5.1-Codex-Max (Inspect)": "gpt_5_1_codex_max_inspect",
    "GPT-5.2": "gpt_5_2", "GPT-5.3-Codex": "gpt_5_3_codex",
    "Gemini 3 Pro": "gemini_3_pro", "o1 (Inspect)": "o1_inspect",
    "o1-preview": "o1_preview", "o3 (Inspect)": "o3_inspect",
}
import yaml
pub = yaml.safe_load(open("../data/benchmark_results_1_1.yaml"))["results"]
names, mh, sd, ph = [], [], [], []
for alias, pid in pub_map.items():
    names.append(alias.replace(" (Inspect)", ""))
    mh.append(mine[alias]); sd.append(hsd[alias]["sd"])
    ph.append(float(pub[pid]["metrics"]["p50_horizon_length"]["estimate"]))
order = np.argsort(mh)
names = [names[i] for i in order]; mh = np.array(mh)[order]
sd = np.array(sd)[order]; ph = np.array(ph)[order]
y = np.arange(len(names))
fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.errorbar(mh, y, xerr=sd, fmt="o", ms=3, capsize=2, label="this work (spline, fold mean +/- sd)")
ax.scatter(ph, y, marker="s", s=16, label="METR published (logistic)")
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel("50% time horizon (minutes, log scale)")
ax.set_title("50% horizons: spline refit vs METR published")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="x")
fig.tight_layout(); fig.savefig("../output/fig4_horizons.png"); plt.close(fig)
print("figures written")
