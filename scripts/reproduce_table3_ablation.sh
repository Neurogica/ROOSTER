#!/usr/bin/env bash
# Table 3 (right): the two proposed components removed from the full model, all four
# combinations, at the published protocol. Table 3 (left) is produced by
# reproduce_table4_reconstruction.sh (the reconstruction ablation shares its harness).
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="--seeds 0 --max-steps 20000 --n-samples 20 --warmup-steps 500 --skip-completed --skip-reconstruct"
python scripts/run.py --models DecompDict-K3 DecompDict-K3-q1 DecompDict-K3-align DecompDict-K3-q1-align \
  --datasets Weather ETTm2 --horizons 96 192 336 720 $COMMON
python scripts/run.py --models DecompDict-K3-paired-w192 DecompDict-K3-q1-paired-w192 DecompDict-K3-align-paired-w192 DecompDict-K3-q1-align-paired-w192 \
  --datasets PEMS04 --horizons 12 24 48 96 $COMMON
# Bias-form ablation (Sec. 4.5): matched harness, H = 96, three seeds.
python scripts/run.py --models DecompDict-K3-q1-align DecompDict-K3-q1-align-nobias DecompDict-K3-q1-align-bump DecompDict-K3-q1-align-free DecompDict-K3-q1-align-frozen \
  --datasets ETTm1 ETTm2 ETTh1 ETTh2 Weather Exchange --horizons 96 --seeds 0 1 2 \
  --max-steps 20000 --n-samples 20 --max-test-windows 1024 --warmup-steps 500 --skip-completed --skip-reconstruct
