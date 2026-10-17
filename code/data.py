"""Load and prepare METR Time Horizon 1.1 run data.

Source: github.com/METR/eval-analysis-public,
reports/time-horizon-1-1/data/raw/runs.jsonl,
commit 52cb829c7a2efb2d659285c4b1768d191d97f8d2.
"""
import json
import math
import numpy as np
import pandas as pd

RUNS_PATH = "../data/runs.jsonl"


def load_runs(path=RUNS_PATH):
    """Return DataFrame of AI runs (human alias excluded) with derived fields."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["alias"] == "human":
                continue
            rows.append({
                "task_id": r["task_id"],
                "family": r["task_family"],
                "alias": r["alias"],
                "y": int(r["score_binarized"]),
                "human_minutes": float(r["human_minutes"]),
                "w": float(r["invsqrt_task_weight"]),
                "source": r["task_source"],
            })
    df = pd.DataFrame(rows)
    assert (df["human_minutes"] > 0).all(), "non-positive human_minutes found"
    df["x"] = np.log2(df["human_minutes"])
    # integer codes for models, tasks, families
    df["j"] = pd.Categorical(df["alias"]).codes
    df["i"] = pd.Categorical(df["task_id"]).codes
    df["f"] = pd.Categorical(df["family"]).codes
    return df


def family_folds(df, n_folds=5, seed=20261012):
    """Assign each task family to a fold; return dict family_code -> fold."""
    rng = np.random.default_rng(seed)
    fams = np.sort(df["f"].unique())
    rng.shuffle(fams)
    folds = {}
    for k, fam in enumerate(fams):
        folds[fam] = k % n_folds
    return folds
