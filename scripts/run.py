#!/usr/bin/env python
"""Run the benchmark suite and regenerate LEADERBOARD.md.

The intended workflow is to run this after every substantive change:

    python scripts/run_leaderboard.py                 # everything available
    python scripts/run_leaderboard.py --smoke         # tiny budget, wiring check
    python scripts/run_leaderboard.py --models DLinear --datasets ETTm2

Scope, by design (see docs/03_datasets.md):

* **Forecasting** -- every LTSF dataset present on disk, every standard horizon.
  Cheap enough to run in full.
* **PPG reconstruction** -- BIDMC and CapnoBase only. The heavy sets
  (PPG-DaLiA, WESAD, WildPPG, cuff-less BP) are excluded from this loop; they
  are for final-table runs, not per-commit ones.

Results append to `results/results.jsonl`; the newest record per cell wins.
Nothing is ever overwritten, so a bad run stays traceable.
"""

import argparse
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import rooster.models  # noqa: F401, E402  (registers every benchmark model)
from rooster.datasets.forecasting import FORECAST_SPECS, available_datasets, resolve_spec  # noqa: E402
from rooster.datasets.physio import available_tasks  # noqa: E402
from rooster.evaluation.harness import run_forecast_cell, run_reconstruction_cell  # noqa: E402
from rooster.evaluation.leaderboard import append_records, budget_label, git_sha, load_records, render  # noqa: E402
from rooster.evaluation.promotion import format_report, winning_candidates  # noqa: E402
from rooster.models.baselines import TrainingBudget  # noqa: E402

FORECAST_MODELS = ("RepeatLast", "LinearForecaster", "DLinear", "DecompSSM", "FlowSSM")
RECONSTRUCTION_MODELS = ("CopyInput", "ConvReconstructor", "PENGUIN", "FlowSSMRecon")

# Exit code for "the gate stopped this run". Distinct from a crash so the
# supervisor can tell "improve the model and re-run" from "retry".
GATE_STOPPED_EXIT = 2

# PEMS is excluded from candidate selection: four datasets of 170-883 correlated
# sensors accounted for over half the sweep's wall-clock, and the cross-variate
# question they answer is better asked once on the promoted winner than four
# times on every candidate.
ITERATE_EXCLUDED = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")

# Two standing configurations, so "which setting was this run under" is one word
# rather than eight flags that have to match between people.
#
#   iterate -- the cheap forecasting datasets at the primary horizon only, plus
#              the single lightest reconstruction task. For ranking candidates
#              against each other; NOT for a number that goes in the paper.
#   full    -- every dataset, every horizon, every reconstruction task including
#              the heavy physiological ones. For the promoted winner only.
#
# CapnoBase-RESP is the lightest reconstruction task (750k training samples,
# against BIDMC-ECG's 1.86M) and it is one where the trivial CopyInput baseline
# is genuinely weak, so it discriminates.
PROFILES = {
    # 20 000 steps, not 5 000: learning curves (scripts/learning_curve.py) show
    # FlowSSM's validation loss and test MSE both still improving at 20 000 on
    # every dataset measured, so selecting at 5 000 was selecting on how fast a
    # model starts rather than where it ends up.
    "iterate": {
        "exclude": ITERATE_EXCLUDED,
        "horizons": [96],
        "tasks": ["CapnoBase-RESP"],
        "max_steps": 20000,
        # 20, not 8. The fair-CRPS estimator is biased upward at small ensemble
        # sizes, so 8 samples was penalising our own model: on Weather, CRPS
        # measured 0.187 at 8 samples and 0.175 at 20, for the same weights.
        "n_samples": 20,
        "max_test_windows": 1024,
        "warmup_steps": 500,
    },
    "full": {
        "exclude": (),
        "horizons": None,
        "tasks": None,
        "max_steps": 20000,
        "n_samples": 20,
        "max_test_windows": 4096,
        "warmup_steps": 500,
    },
}

# Heaviest datasets last, so an interrupted run still leaves a broadly populated
# table rather than four horizons of one dataset.
# Cheapest first. Training cost scales with batch x variates, so the variate count
# dominates; length breaks ties. Running in this order means an early gate check
# happens after minutes rather than hours.
DATASET_ORDER = (
    "ILI", "Exchange", "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "Weather", "Solar", "PEMS08", "PEMS04", "PEMS03",
    "Electricity", "Traffic",
)  # fmt: skip

# PEMS is excluded from candidate selection: four datasets of 170-883 correlated
# sensors accounted for over half the sweep's wall-clock, and the cross-variate
# question they answer is better asked once on the promoted winner than four
# times on every candidate.
ITERATE_EXCLUDED = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")


def apply_profile(args):
    """Fill unset options from the chosen profile. Explicit flags always win."""
    if not args.profile:
        return args
    profile = PROFILES[args.profile]
    if args.horizons is None:
        args.horizons = profile["horizons"]
    if args.tasks is None and profile["tasks"] is not None:
        args.tasks = list(profile["tasks"])
    for key in ("max_steps", "n_samples", "max_test_windows", "warmup_steps"):
        if getattr(args, f"_{key}_explicit", False):
            continue
        setattr(args, key, profile[key])
    return args


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data", help="root holding forecast/ and physio/ (default: data)")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None, help="standing configuration; explicit flags still win")
    parser.add_argument("--models", nargs="*", default=None, help="subset of forecasting models to run")
    parser.add_argument("--recon-models", nargs="*", default=None, help="subset of reconstruction models to run")
    parser.add_argument("--datasets", nargs="*", default=None, help="subset of forecasting datasets to run")
    parser.add_argument("--tasks", nargs="*", default=None, help="subset of reconstruction tasks to run")
    parser.add_argument("--horizons", nargs="*", type=int, default=None, help="override each dataset's horizon set")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0], help="seeds per cell (default: 0)")
    parser.add_argument("--n-samples", type=int, default=20, help="ensemble size for CRPS; ignored by deterministic models")
    parser.add_argument("--max-steps", type=int, default=2000, help="matched training budget in optimiser steps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=0, help="linear LR warmup; worth using once max-steps is large")
    parser.add_argument("--cosine-decay", action="store_true", help="cosine-decay the LR to 1%% of peak over the budget")
    parser.add_argument("--selection", choices=("mse", "objective"), default="mse", help="what early stopping selects the checkpoint on: validation MSE (current) or the training objective incl. auxiliary terms (pre-Aug-3 behaviour)")
    parser.add_argument("--recipe", choices=("none", "tslib"), default="none", help="named training recipe; 'tslib' is the DecompSSM paper's: batch 32, lr 1e-4 halved every epoch, 10 epochs, patience 3 epochs")
    parser.add_argument("--epochs", type=int, default=10, help="epochs for an epoch-based --recipe")
    parser.add_argument("--checkpoint-dir", default=None, help="enable save/resume so an interrupted sweep continues where it stopped")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="skip cells already recorded at the current git SHA; makes restarting an interrupted sweep nearly free",
    )
    parser.add_argument("--max-test-windows", type=int, default=None, help="cap test windows (evenly strided); recorded in the protocol column")
    parser.add_argument(
        "--stop-if-losing",
        action="store_true",
        help="after each dataset, abort (exit 2) unless some candidate beats every baseline; for fail-fast candidate search",
    )
    parser.add_argument("--skip-forecast", action="store_true")
    parser.add_argument("--skip-reconstruct", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny budget on one dataset/task: wiring check, not a result")
    args = parser.parse_args()
    # Record which numeric options the user actually typed, so a profile fills in
    # the rest without silently overriding an explicit choice.
    typed = {argument.lstrip("-").replace("-", "_") for argument in sys.argv[1:] if argument.startswith("--")}
    for key in ("max_steps", "n_samples", "max_test_windows", "warmup_steps", "learning_rate"):
        setattr(args, f"_{key}_explicit", key in typed)
    return apply_profile(args)


def build_budget(args):
    if args.smoke:
        return TrainingBudget(max_steps=20, batch_size=16, learning_rate=args.learning_rate, patience=1, eval_every=10, num_workers=0)
    if args.recipe == "tslib":
        # The paper's run.py default is lr 1e-4; an explicit --learning-rate
        # probes the schedule at another peak and is recorded in the label.
        lr = args.learning_rate if args._learning_rate_explicit else 1e-4
        return TrainingBudget(
            max_steps=0,
            batch_size=32,
            learning_rate=lr,
            epochs=args.epochs,
            recipe="tslib" if lr == 1e-4 else f"tslib-lr{lr:g}",
            num_workers=args.num_workers,
            checkpoint_dir=args.checkpoint_dir,
            selection=args.selection,
        )
    return TrainingBudget(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=3,
        eval_every=max(args.max_steps // 10, 1),
        num_workers=args.num_workers,
        warmup_steps=args.warmup_steps,
        cosine_decay=args.cosine_decay,
        checkpoint_dir=args.checkpoint_dir,
        selection=args.selection,
    )


def select_forecast_datasets(args):
    present = set(available_datasets(os.path.join(args.data_root, "forecast")))
    if args.datasets:
        chosen = [resolve_spec(name).name for name in args.datasets]
        missing = [name for name in chosen if name not in present]
        if missing:
            raise SystemExit(f"requested datasets not found on disk: {missing}. See docs/03_datasets.md.")
    else:
        excluded = set(PROFILES[args.profile]["exclude"]) if args.profile else set()
        chosen = [name for name in DATASET_ORDER if name in present and name not in excluded]
        chosen += sorted(present - set(chosen) - excluded)  # anything added to the registry later
    if args.smoke:
        chosen = chosen[:1]
    return chosen


def select_reconstruction_tasks(args):
    present = available_tasks(os.path.join(args.data_root, "physio"))
    chosen = args.tasks if args.tasks else present
    missing = [name for name in chosen if name not in present]
    if missing:
        raise SystemExit(f"requested reconstruction tasks not available: {missing}. See docs/03_datasets.md.")
    return chosen[:1] if args.smoke else chosen


def main():
    args = parse_args()
    budget = build_budget(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The gate must judge only cells from THIS configuration; a leftover
    # 5 000-step result would otherwise decide a 20 000-step sweep.
    budget_signature = budget_label(budget.max_steps, args.n_samples, args.max_test_windows, selection=getattr(budget, "selection", "mse"), recipe=getattr(budget, "recipe", None), epochs=getattr(budget, "epochs", None))
    forecast_root = os.path.join(args.data_root, "forecast")
    physio_root = os.path.join(args.data_root, "physio")

    print(
        f"device={device}  profile={args.profile or 'none'}  budget={budget.max_steps} steps  batch={budget.batch_size}  "
        f"seeds={args.seeds}  n_samples={args.n_samples}  ckpt={budget.checkpoint_dir or 'off'}"
    )
    if args.smoke:
        # Cap evaluation too: a 20-step model against 79k full-stride test
        # sequences is a slow way to learn nothing.
        args.max_test_windows = args.max_test_windows or 256
        print("SMOKE MODE -- results are a wiring check, not a measurement.")

    # Each cell is persisted the moment it completes. A full sweep is long enough
    # that batching the writes until the end means a crash, a kill, or a session
    # ending loses every result -- which is exactly what happened once.
    n_recorded, failures = 0, []

    # Cells already recorded under this budget are skipped so an interrupted sweep
    # resumes cheaply. Deliberately NOT keyed on the git SHA: a sweep spans hours
    # and the code will have moved on, and requiring an exact SHA match threw away
    # a full day of finished cells for a change that touched no model. Cells from
    # an older commit are still skipped, but their count is reported so a stale
    # row is visible rather than assumed fresh -- and every row carries its own
    # SHA in the leaderboard. Clear results/results.jsonl to force a full re-run.
    completed = set()
    if args.skip_completed:
        sha = git_sha()
        existing = [r for r in load_records() if r.budget == budget_signature]
        completed = {r.key for r in existing}
        stale = sum(1 for r in existing if r.git_sha != sha)
        print(f"skip-completed: {len(completed)} cells already recorded at this budget ({stale} from an earlier commit than {sha})")

    def already_done(task, dataset, model, horizon, seed):
        return (task, dataset, model, horizon, seed) in completed

    def record_cell(record):
        """Persist and re-render immediately, so an interrupted sweep still
        leaves a valid, up-to-date leaderboard rather than nothing."""
        nonlocal n_recorded
        append_records([record])
        render(load_records())
        n_recorded += 1

    if not args.skip_forecast:
        models = args.models if args.models else list(FORECAST_MODELS)
        datasets = select_forecast_datasets(args)
        print(f"\nForecasting: {len(models)} models x {len(datasets)} datasets -> {datasets}")
        for dataset_name in datasets:
            horizons = args.horizons if args.horizons else FORECAST_SPECS[dataset_name].horizons
            if args.smoke:
                horizons = horizons[:1]
            for horizon in horizons:
                for model_name in models:
                    for seed in args.seeds:
                        label = f"forecast/{dataset_name}/h{horizon}/{model_name}/s{seed}"
                        if already_done("forecast", dataset_name, model_name, horizon, seed):
                            continue
                        try:
                            record = run_forecast_cell(
                                model_name=model_name,
                                dataset_name=dataset_name,
                                horizon=horizon,
                                budget=budget,
                                seed=seed,
                                device=device,
                                n_samples=args.n_samples,
                                data_root=forecast_root,
                                max_test_windows=args.max_test_windows,
                            )
                            record_cell(record)
                            metrics = record.metrics
                            print(f"  {label:52s} MSE={metrics['mse']:.4f} MAE={metrics['mae']:.4f} ({metrics['train_seconds']:.0f}s)")
                        except Exception as error:  # noqa: BLE001 -- one bad cell must not abort the sweep
                            failures.append((label, error))
                            print(f"  {label:52s} FAILED: {type(error).__name__}: {error}")

            # Gate: checked once per dataset, cheapest datasets first, so a
            # candidate that is not competitive is caught in minutes.
            if args.stop_if_losing:
                current = [r for r in load_records() if r.budget == budget_signature]
                winners = winning_candidates(current, "forecast")
                print(f"\n  -- gate after {dataset_name} --")
                print("  " + format_report(current, "forecast").replace("\n", "\n  "))
                if not winners:
                    print(f"\n  STOPPING: no candidate beats every baseline after {dataset_name}.")
                    print("  Improve the model, then re-run; --skip-completed will keep the finished cells.")
                    render(load_records())
                    return GATE_STOPPED_EXIT
                print(f"  winning: {winners}\n")

    if not args.skip_reconstruct:
        models = args.recon_models if args.recon_models else list(RECONSTRUCTION_MODELS)
        tasks = select_reconstruction_tasks(args)
        print(f"\nReconstruction: {len(models)} models x {len(tasks)} tasks -> {tasks}")
        for task_name in tasks:
            for model_name in models:
                for seed in args.seeds:
                    label = f"reconstruct/{task_name}/{model_name}/s{seed}"
                    # Reconstruction cells key on window length, which the harness
                    # derives, so completion is checked after the record exists.
                    try:
                        record = run_reconstruction_cell(
                            task_name=task_name,
                            model_name=model_name,
                            budget=budget,
                            seed=seed,
                            device=device,
                            n_samples=args.n_samples,
                            data_root=physio_root,
                            max_test_windows=args.max_test_windows,
                        )
                        record_cell(record)
                        rate = record.metrics["rate_mae_bpm"]
                        print(f"  {label:52s} rateMAE={rate:.2f}bpm r={record.metrics['pearson_r']:.3f} ({record.metrics['train_seconds']:.0f}s)")
                    except Exception as error:  # noqa: BLE001
                        failures.append((label, error))
                        print(f"  {label:52s} FAILED: {type(error).__name__}: {error}")

    path = render(load_records())

    print(f"\n{n_recorded} cells recorded, {len(failures)} failed. Leaderboard written to {path}")
    if failures:
        print("\nFailures (full tracebacks):")
        for label, error in failures:
            print(f"\n--- {label} ---")
            traceback.print_exception(type(error), error, error.__traceback__)
    return 1 if failures and not n_recorded else 0


if __name__ == "__main__":
    raise SystemExit(main())
