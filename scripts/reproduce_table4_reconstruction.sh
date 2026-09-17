#!/usr/bin/env bash
# Table 4 (matched-harness control: ours vs. PENGUIN retrained from its original code,
# one rate estimator and one medoid readout) and Table 3 (left), the conditioning-scheme
# ablation on the same five tasks.
set -euo pipefail
cd "$(dirname "$0")/.."
TASKS="BIDMC-ECG CapnoBase-ECG BIDMC-RESP CapnoBase-RESP DaLiA-ECG"
python scripts/run_reconstruction.py --models FlowSSMRecon-align PENGUIN --tasks $TASKS --seeds 0 --out results/table4.jsonl
python scripts/run_reconstruction.py --models FlowSSMRecon FlowSSMRecon-skiponly FlowSSMRecon-align-nobias FlowSSMRecon-align \
  --tasks $TASKS --seeds 0 --out results/table3_reconstruction.jsonl
