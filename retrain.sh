#!/usr/bin/env bash
set -euo pipefail
python -m fedgrant.ml.train_all --data data/mldata --out models/
python -m fedgrant.ml.export --models models/
python -m fedgrant.ml.evaluate --data data/mldata --models models/ --splits test_id test_ood
python scripts/bench_predictors.py --models models/
python -m fedgrant.ml.plots_ml
echo "=== DONE $(date)"
