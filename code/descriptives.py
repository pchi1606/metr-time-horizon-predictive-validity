"""Descriptive checks (preregistered, not scored):
1. Fit P1 on ALL data; compute per-AI 50% horizons; compare vs METR's published
   benchmark_results_1_1.yaml.
2. Save the fitted time-to-difficulty curve f for plotting.
Saves ../output/descriptives.json.
"""
import json
import numpy as np
import yaml

from data import load_runs
from predictors import SplineHorizon


def main():
    df = load_runs()
    n_ai = df["j"].nunique()
    ai_names = (df[["j", "alias"]].drop_duplicates()
                .sort_values("j")["alias"].tolist())
    x = df["x"].to_numpy(); y = df["y"].to_numpy(float)
    w = df["w"].to_numpy(); j = df["j"].to_numpy()

    p1 = SplineHorizon().fit(x, y, w, j, n_ai)
    horizons = {ai_names[jj]: p1.horizon(jj) for jj in range(n_ai)}

    with open("../data/benchmark_results_1_1.yaml") as f:
        pub = yaml.safe_load(f)
    pub_h = {}
    for model_id, entry in pub.get("results", {}).items():
        try:
            pub_h[model_id] = float(
                entry["metrics"]["p50_horizon_length"]["estimate"])
        except (KeyError, TypeError):
            pass

    grid = np.linspace(x.min(), x.max(), 200)
    out = {
        "horizons_mine_min": {k: float(v) for k, v in horizons.items()},
        "horizons_published_min": pub_h,
        "f_curve": {"x_log2min": grid.tolist(),
                    "f": [float(v) for v in p1.f_(grid)]},
        "beta": float(p1.beta_),
        "alpha": {ai_names[jj]: float(p1.alpha_[jj]) for jj in range(n_ai)},
    }
    with open("../output/descriptives.json", "w") as f:
        json.dump(out, f, indent=2)
    print("horizons (min):")
    for k, v in sorted(horizons.items(), key=lambda kv: kv[1]):
        print(f"  {k}: {v:.1f}")
    print("saved ../output/descriptives.json")


if __name__ == "__main__":
    main()
