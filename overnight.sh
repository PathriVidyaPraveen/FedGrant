#!/usr/bin/env bash
set -euo pipefail
echo "=== start $(date)"

echo "=== 1. tests"
python -m pytest -q

echo "=== 2. synthetic data (only if missing)"
for s in train val test_id test_ood; do
  [ -f data/mldata_synth/$s.npz ] || python -m fedgrant.ml.synthetic --split $s --out data/mldata_synth/
done

echo "=== 3. train 5 seeds x {gru, mlp}"
python -m fedgrant.ml.train_all --data data/mldata_synth --out models/ --resume

echo "=== 4. export production models"
python -m fedgrant.ml.export --models models/

echo "=== 5. evaluate all predictors"
python -m fedgrant.ml.evaluate --data data/mldata_synth --models models/ --splits test_id test_ood

echo "=== 6. latency benchmark"
python scripts/bench_predictors.py --models models/

echo "=== 7. figures"
python -m fedgrant.ml.plots_ml --summary results/summary --out results/figures

echo "=== 8. tests again (now with real models present)"
python -m pytest -q

echo "=== DONE $(date)"
