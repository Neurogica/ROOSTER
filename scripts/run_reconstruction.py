#!/usr/bin/env python
"""Baseline table for the PPG-to-vital reconstruction tasks, with seed replication.

Seeds first, on purpose. The forecasting direction was pursued for weeks on
single-seed tables that turned out to be inside the noise
(docs/06_negative_result_cardinality.md), so on this side every number gets an
error bar before it gets interpreted.

`CopyInput` is in the table for the same reason it exists: on the clinical sets a
model that returns the PPG unchanged scores 1.53 bpm, so any result there is
already near-saturated. The first thing worth knowing about a new task is what
that trivial baseline scores on it -- measured, PPG-DaLiA gives 18.4 bpm, so
unlike BIDMC it has genuine headroom.

Usage:
    python scripts/run_reconstruction.py --tasks DaLiA-ECG --models CopyInput PENGUIN
"""

import argparse
import json
import pathlib
import statistics
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rooster.models.configs  # noqa: F401,E402  -- registers the candidate models
from rooster.evaluation.harness import run_reconstruction_cell  # noqa: E402
from rooster.evaluation.leaderboard import append_records  # noqa: E402
from rooster.models.baselines import TrainingBudget  # noqa: E402

# Reported for every cell. `rate_unresolved_frac` is not decoration: the rate MAE
# is a mean over windows whose rate could be resolved at all, so a model that
# fails on the hard windows can win the headline number by not competing for it.
# The two have to be read together.
REPORTED = ("rate_mae_bpm", "rate_unresolved_frac", "rmse", "mae", "pearson_r")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=["DaLiA-ECG"])
    parser.add_argument("--models", nargs="+", default=["CopyInput", "ConvReconstructor", "PENGUIN"])
    # One seed by default: this script is the search loop, and replicating every
    # exploratory cell costs 3x for information the search does not use. The noise
    # floor is a property of the task, measured once and recorded in
    # docs/07_physio_findings.md -- ~1.5-2.2 bpm on DaLiA-ECG, ~0.2-0.7 on the
    # clinical sets. Differences smaller than that are not read as differences.
    # Pass --seeds 0 1 2 for the finalists, which is the only place error bars
    # change a decision.
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-test-windows", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--out", default="results/physio_baselines.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        collected = {}
        for model_name in args.models:
            # A model with no parameters cannot depend on the seed, and the split
            # is deterministic, so replicating it would just cost time.
            seeds = [args.seeds[0]] if model_name == "CopyInput" else args.seeds
            budget = TrainingBudget(
                max_steps=1 if model_name == "CopyInput" else args.max_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=3,
                eval_every=max(args.max_steps // 10, 1),
                num_workers=args.num_workers,
                warmup_steps=args.warmup_steps,
                checkpoint_dir=args.checkpoint_dir,
            )
            for seed in seeds:
                try:
                    record = run_reconstruction_cell(
                        task, model_name, budget, seed, device,
                        n_samples=args.n_samples, max_test_windows=args.max_test_windows,
                    )
                except Exception as error:  # one model failing must not lose the table
                    print(f"  {model_name} seed={seed} FAILED: {type(error).__name__}: {error}", flush=True)
                    continue
                append_records([record])
                row = {"task": task, "model": model_name, "seed": seed, **record.metrics}
                with out.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                collected.setdefault(model_name, []).append(row)
                print(
                    f"  {model_name:20} seed={seed}  rate_mae={row.get('rate_mae_bpm', float('nan')):6.2f} bpm  "
                    f"unresolved={row.get('rate_unresolved_frac', float('nan')):.3f}  "
                    f"rmse={row.get('rmse', float('nan')):.4f}  r={row.get('pearson_r', float('nan')):+.3f}",
                    flush=True,
                )

        print(f"\n{task} -- mean +/- sd over seeds:", flush=True)
        for model_name, rows in collected.items():
            parts = []
            for metric in REPORTED:
                values = [row[metric] for row in rows if metric in row and row[metric] == row[metric]]
                if not values:
                    continue
                spread = f" +/- {statistics.stdev(values):.3f}" if len(values) > 1 else ""
                parts.append(f"{metric}={statistics.mean(values):.3f}{spread}")
            print(f"  {model_name:20} n={len(rows)}  " + "  ".join(parts), flush=True)


if __name__ == "__main__":
    main()
