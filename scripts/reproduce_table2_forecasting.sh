#!/usr/bin/env bash
# Table 2: MTS forecasting at the published protocol (20k steps, 20 samples, full test
# split, seed 0). One configuration per dataset, selected on validation; results are
# appended to results/results.jsonl and summarised in LEADERBOARD.md.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="--seeds 0 --max-steps 20000 --n-samples 20 --warmup-steps 500 --skip-completed --skip-reconstruct"
python scripts/run.py --models DecompDict-K3-q2-align-w512        --datasets Electricity --horizons 96 192 336 720 $COMMON
python scripts/run.py --models DecompDict-K3-q1-align             --datasets Weather ETTm2 --horizons 96 192 336 720 $COMMON
python scripts/run.py --models DecompDict-K3-q1-align-paired-w192 --datasets PEMS04 --horizons 12 24 48 96 $COMMON
