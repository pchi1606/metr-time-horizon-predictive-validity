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

echo "=== 1/4 cross-validated shootout (P1/P2/P3) ==="
cd code && python3 run_cv.py && cd ..

echo "=== 2/4 descriptives (full-data P1 fit, horizons) ==="
cd code && python3 descriptives.py && cd ..

echo "=== 3/4 strengthening analyses (calibration, P0, sensitivity) ==="
cd code && python3 strengthen.py && cd ..

echo "=== 3b/4 P3 diagnostics (test-23 review response) ==="
cd code && python3 p3_diagnostics.py && cd ..

echo "=== 4/4 figures ==="
cd code && python3 figures.py && cd ..
cp output/fig*.png figs/

echo "=== verifying headline numbers ==="
python3 verify.py
echo "=== reproduction complete ==="
