#!/usr/bin/env bash
# Reproduce the P1 horizon-shootout results end to end.
# Usage: bash reproduce.sh
# Expects: python3 with numpy, scipy, pandas, matplotlib, pyyaml.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== checking python dependencies ==="
python3 - <<'EOF'
import importlib
for pkg in ["numpy", "scipy", "pandas", "matplotlib", "yaml"]:
    importlib.import_module(pkg)
print("dependencies OK")
EOF

mkdir -p output figs

echo "=== 1/5 cross-validated shootout (P1/P2/P3) ==="
cd code && python3 run_cv.py && cd ..

echo "=== 2/5 descriptives (full-data P1 fit, horizons) ==="
cd code && python3 descriptives.py && cd ..

echo "=== 3/5 strengthening analyses (calibration, P0, sensitivity) ==="
cd code && python3 strengthen.py && cd ..

echo "=== 3b/5 P3 diagnostics (test-23 review response) ==="
cd code && python3 p3_diagnostics.py && cd ..

echo "=== 4/5 figures ==="
cd code && python3 figures.py && cd ..
cp output/fig*.png figs/

echo "=== 5/5 revision add-ons (appendix robustness tables) ==="
# Each add-on refits on the locked scaffold; together they rebuild every
# robustness summary printed in the appendix (clip sensitivity and
# per-fold horizons; alternative partitions; ECE binnings; per-AI
# variants; round-5 LOFO / ECE weight shares / pooled scores / per-AI
# SEs / P0 same-specification control; P3 identified quantities).
cd code && python3 clip_and_horizons.py && cd ..
cd code && python3 r4_ece.py && cd ..
cd code && python3 r4_partitions.py && cd ..
cd code && python3 r4_perai.py && cd ..
cd code && python3 r4_perai2.py && cd ..
cd code && python3 r5_analyses.py && cd ..
cd code && python3 r5_p3scale.py && cd ..
cd code && python3 r5_rescore.py && cd ..
cd code && python3 r6_p3variant.py && cd ..
cd code && python3 r7_extras.py && cd ..
cd code && python3 r11_stage1_ridge.py && cd ..
cd code && python3 p3_convergence.py && cd ..

echo "=== verifying headline numbers ==="
python3 verify.py
echo "=== verifying printed numbers against the compiled PDF ==="
python3 verify_printed.py
echo "=== reproduction complete ==="
