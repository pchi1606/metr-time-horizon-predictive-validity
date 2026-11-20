"""F-5 partition-only sensitivity for P1 (ENACTMENT-2026-11-20).

Extends repro/code/r4_partitions.py EXACTLY: identical predictors
(SplineHorizon / MeanPredictor / IrtPredictor), identical scoring
(run_cv.scores_fold -> sqrt-family weighted held-out log-loss), identical
fitting procedure; only the family-partition seed changes, via
data.family_folds (79 families, 5 folds of 15-16 families -> 63-64
training families per fold).

Master seed: 20261120. Partition seeds are drawn once, deterministically:
  rng = np.random.default_rng(20261120)
  seeds = rng.integers(0, 2**31 - 1, size=N_PARTITIONS)
collisions with the locked seed (20261012) and the three Table 27
alternative-partition seeds (20261013, 20261014, 42) are rejected and
redrawn (none occurred in the drawn sequence; asserted in code).

Checkpointing: after each partition completes, its 5 per-fold rows are
appended to f5-partitions.csv (flush + fsync). Re-running skips
partitions already present in the CSV (resume).

Usage: python3 f5_run_partitions.py [n_partitions]
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO_CODE = os.path.expanduser("~/workspace/p1-retarget/repro/code")
sys.path.insert(0, REPRO_CODE)
# data.load_runs / predictors use paths relative to repro/code
os.chdir(REPRO_CODE)

from data import load_runs, family_folds          # noqa: E402
from predictors import SplineHorizon, MeanPredictor, IrtPredictor  # noqa: E402
from run_cv import scores_fold                    # noqa: E402

MASTER_SEED = 20261120
N_PARTITIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CSV_PATH = os.path.join(HERE, "f5-partitions.csv")
LOG_PATH = os.path.join(HERE, "f5-run.log")
EXCLUDED_SEEDS = {20261012, 20261013, 20261014, 42}

FIELDS = ["partition_index", "partition_seed", "fold", "n_test_families",
          "n_train_families", "n_test_runs", "P1_logloss", "P2_logloss",
          "P3_logloss", "H1_diff", "H2_diff",
          "partition_H1_mean", "partition_H1_se", "partition_H2_mean",
          "partition_H2_se", "partition_seconds"]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def partition_seeds(n):
    rng = np.random.default_rng(MASTER_SEED)
    seeds = []
    while len(seeds) < n:
        s = int(rng.integers(0, 2**31 - 1))
        if s in EXCLUDED_SEEDS or s in seeds:
            continue
        seeds.append(s)
    return seeds


def run_partition(df, seed):
    folds = family_folds(df, seed=seed)
    df = df.copy()
    df["fold"] = df["f"].map(folds)
    n_ai = df["j"].nunique()
    rows = []
    for k in range(5):
        tr = df[df["fold"] != k].reset_index(drop=True)
        te = df[df["fold"] == k].reset_index(drop=True)
        tr = tr.copy(); te = te.copy()
        tmap = {t: n for n, t in enumerate(sorted(tr["i"].unique()))}
        tr["ti"] = tr["i"].map(tmap)
        x_tr = tr["x"].to_numpy(); y_tr = tr["y"].to_numpy(dtype=float)
        w_tr = tr["w"].to_numpy(); j_tr = tr["j"].to_numpy(); i_tr = tr["ti"].to_numpy()
        x_te = te["x"].to_numpy(); y_te = te["y"].to_numpy(dtype=float)
        w_te = te["w"].to_numpy(); j_te = te["j"].to_numpy()
        p1 = SplineHorizon().fit(x_tr, y_tr, w_tr, j_tr, n_ai)
        p2 = MeanPredictor().fit(y_tr, w_tr, j_tr, n_ai)
        p3 = IrtPredictor().fit(x_tr, y_tr, w_tr, j_tr, i_tr, n_ai, len(tmap))
        ll1 = scores_fold(y_te, p1.predict(x_te, j_te), w_te)["logloss"]
        ll2 = scores_fold(y_te, p2.predict(j_te), w_te)["logloss"]
        ll3 = scores_fold(y_te, p3.predict(x_te, j_te), w_te)["logloss"]
        n_test_fam = int(te["f"].nunique())
        rows.append({"fold": k, "n_test_families": n_test_fam,
                     "n_train_families": int(tr["f"].nunique()),
                     "n_test_runs": int(len(te)),
                     "P1_logloss": ll1, "P2_logloss": ll2, "P3_logloss": ll3,
                     "H1_diff": ll1 - ll2, "H2_diff": ll1 - ll3})
    for name in ("H1", "H2"):
        d = np.array([r[f"{name}_diff"] for r in rows])
        for r in rows:
            r[f"partition_{name}_mean"] = float(d.mean())
            r[f"partition_{name}_se"] = float(d.std(ddof=1) / math.sqrt(5))
    return rows


def completed_partitions():
    if not os.path.exists(CSV_PATH):
        return set()
    done = set()
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            done.add(int(row["partition_index"]))
    return done


def main():
    seeds = partition_seeds(N_PARTITIONS)
    log(f"F-5 run start: master_seed={MASTER_SEED} n_partitions={N_PARTITIONS}")
    log(f"partition seeds: {seeds}")
    df = load_runs()
    log(f"loaded {len(df)} runs, {df['f'].nunique()} families, {df['j'].nunique()} AIs")
    done = completed_partitions()
    if done:
        log(f"resuming; already complete: {sorted(done)}")
    new_file = not os.path.exists(CSV_PATH)
    t_all = time.time()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader(); f.flush(); os.fsync(f.fileno())
        for idx, seed in enumerate(seeds):
            if idx in done:
                continue
            t0 = time.time()
            rows = run_partition(df, seed)
            dt = time.time() - t0
            assert all(63 <= r["n_train_families"] <= 64 for r in rows), \
                f"training-family count out of 63-64 range: {[r['n_train_families'] for r in rows]}"
            for r in rows:
                w.writerow({"partition_index": idx, "partition_seed": seed,
                            "partition_seconds": round(dt, 2), **r})
            f.flush(); os.fsync(f.fileno())
            log(f"partition {idx} seed={seed} done in {dt:.1f}s "
                f"H1={rows[0]['partition_H1_mean']:.4f}+/-{rows[0]['partition_H1_se']:.4f} "
                f"H2={rows[0]['partition_H2_mean']:.4f}+/-{rows[0]['partition_H2_se']:.4f} "
                f"train_fams={[r['n_train_families'] for r in rows]}")
    log(f"F-5 run complete: total wall {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
