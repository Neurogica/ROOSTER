#!/usr/bin/env bash
# Table 1: PPG-to-vital-sign reconstruction under PENGUIN's metric protocol
# (Hamilton HR over 8 s windows, 60 s Fourier RR, subject-wise splits, seed 0).
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run_reconstruction.py --models FlowSSMRecon-align \
  --tasks DaLiA-ECG WildPPG-ECG BIDMC-RESP-60s WESAD-RESP --seeds 0 --out results/table1.jsonl
