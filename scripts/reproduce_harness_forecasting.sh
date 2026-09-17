#!/usr/bin/env bash
# Matched-harness comparison (Sec. 4.3): ours vs. retrained DecompSSM, six datasets,
# four horizons, three seeds per model, identical steps/samples/evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run.py --models DecompDict-K3-q1-align DecompSSM \
  --datasets ETTm1 ETTm2 ETTh1 ETTh2 Weather Exchange --horizons 96 192 336 720 --seeds 0 1 2 \
  --max-steps 20000 --n-samples 20 --max-test-windows 1024 --warmup-steps 500 --skip-completed --skip-reconstruct
