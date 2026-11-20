"""Separation census of the P3 stage-1 training designs (added 2026-11-12,
round-17 adjudication addendum, demand D1).

For each fold's stage-1 training set (the runs stage 1 is actually fit on:
binary score_binarized, family folds, seed 20261012), count:
- tasks unanimous-failure (all runs y=0) and unanimous-success (all y=1):
  for such a task the unpenalised 1PL likelihood is monotone in its
  difficulty, so no finite joint maximiser exists;
- AIs unanimous either way;
- connected components of the AI x task incidence graph.
Pure counting; writes ../output/p3_separation_census.json.
"""
import json
import sys

sys.path.insert(0, ".")
from data import load_runs, family_folds


def components(pairs):
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    nodes = set()
    for a, b in pairs:
        union(a, b)
        nodes.add(a)
        nodes.add(b)
    return len({find(n) for n in nodes}), len(nodes)


def main():
    df = load_runs()
    folds = family_folds(df, seed=20261012)
    df["fold"] = df["f"].map(folds)
    out = {}
    for k in range(5):
        tr = df[df["fold"] != k]
        g_task = tr.groupby("task_id")["y"].agg(["sum", "count"])
        g_ai = tr.groupby("alias")["y"].agg(["sum", "count"])
        pairs = list(zip(("A:" + a for a in tr["alias"]),
                         ("T:" + t for t in tr["task_id"])))
        nc, nn = components(pairs)
        out[str(k)] = {
            "n_runs": int(len(tr)),
            "n_tasks": int(tr["task_id"].nunique()),
            "n_ais": int(tr["alias"].nunique()),
            "task_all_fail": int((g_task["sum"] == 0).sum()),
            "task_all_succ": int((g_task["sum"] == g_task["count"]).sum()),
            "ai_all_fail": int((g_ai["sum"] == 0).sum()),
            "ai_all_succ": int((g_ai["sum"] == g_ai["count"]).sum()),
            "components": nc,
            "nodes": nn,
        }
        print(k, out[str(k)])
    with open("../output/p3_separation_census.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
