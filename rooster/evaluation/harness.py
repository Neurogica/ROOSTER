"""Runs one leaderboard cell end to end: load -> split -> train -> evaluate -> record.

Kept separate from `metrics.py` (which stays pure and trivially testable) and
from the entrypoint script (which stays thin). Everything that could make two
people's numbers incomparable -- split protocol, scaler placement, ensemble
size, which checkpoint gets evaluated -- is decided here, once.
"""

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from rooster.datasets.batches import FREQ_SECONDS
from rooster.datasets.forecasting import load_forecast_series
from rooster.datasets.physio import load_recordings, subject_split
from rooster.datasets.windows import build_forecast_windows, build_reconstruction_windows
from rooster.evaluation.leaderboard import RunRecord, budget_label
from rooster.evaluation.metrics import ForecastMetrics
from rooster.evaluation.physio_metrics import evaluate_reconstruction
from rooster.models.baselines import build_model, count_parameters
from rooster.utils.help_func import fix_seed

# Component selection replays this many validation windows once per candidate
# removal, so it is capped -- the criterion is a mean over windows and its
# standard error is already tiny at this size. The cap is recorded alongside the
# selected count, because the threshold IS the standard error: a different number
# of windows is a different criterion, not the same one measured less precisely.
_SELECTION_WINDOWS = 2048


@torch.no_grad()
def _stream_forecast_metrics(model, dataset, batch_size, n_samples, device):
    """Evaluate a forecasting split batch by batch, without materialising it.

    Traffic at horizon 720 would be ~9 GB of predictions in float32 and more
    again once metrics promote to float64, so the accumulator is a necessity
    rather than a nicety. Every metric is a mean over scalars, so this is
    exactly equal to the whole-array computation. Returns `(metrics, n_windows)`.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    accumulator = ForecastMetrics()
    n_windows = 0
    for x, y in loader:
        x = x.to(device)
        samples = model.sample(x, n_samples)
        # Point metrics come from whatever the model declares as its point
        # forecast; probabilistic metrics from the ensemble. For most models the
        # two coincide (forward() returns the ensemble mean), but a model with a
        # directly MSE-supervised head should be scored on that head.
        point = model(x).detach().cpu().numpy()
        accumulator.update(samples.detach().cpu().numpy(), y.numpy(), point=point)
        n_windows += len(y)
    return accumulator.compute(), n_windows


@torch.no_grad()
def _collect_predictions(model, dataset, batch_size, n_samples, device):
    """Run the model over a split, returning `(samples, targets)` as numpy.

    Only used for reconstruction, where a whole split is at most a few thousand
    windows of a few thousand samples -- comfortably in memory, and the physio
    metrics (spectral rate estimation) are not expressible as running sums.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sample_batches, target_batches = [], []
    for x, y in loader:
        samples = model.sample(x.to(device), n_samples)
        sample_batches.append(samples.detach().cpu().numpy().astype(np.float32))
        target_batches.append(y.numpy().astype(np.float32))
    return np.concatenate(sample_batches, axis=1), np.concatenate(target_batches, axis=0)


def _capped(dataset, max_windows):
    """Evenly-strided subset of a split, or the split itself when uncapped.

    Striding rather than taking a prefix keeps the subset spread over the whole
    test period instead of only its first days. Any cap is recorded in the
    leaderboard's protocol column, because a capped number is not comparable to
    an uncapped one.
    """
    if not max_windows or len(dataset) <= max_windows:
        return dataset, 1
    stride = math.ceil(len(dataset) / max_windows)
    return Subset(dataset, list(range(0, len(dataset), stride))), stride


@torch.no_grad()
def _measure_cardinality(model, dataset, batch_size, device, max_batches=16):
    """How many components the model actually used, averaged over validation.

    Validation rather than test: the count is the headline claim, and a claim
    read off the test split is a claim about the test split. Capped at a few
    batches because the count is a mean over windows and converges immediately.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    per_window, per_dataset, n_batches = 0.0, 0, 0
    for x, _y in loader:
        counts = model.cardinality(x.to(device))
        per_window += counts["per_window"]
        per_dataset = max(per_dataset, counts["per_dataset"])
        n_batches += 1
        if n_batches >= max_batches:
            break
    if not n_batches:
        return {}
    return {"k_per_window": per_window / n_batches, "k_per_dataset": per_dataset}


@torch.no_grad()
def _head_rate_error(model, dataset, batch_size, device, spec):
    """Rate error from a model's own rate head, against the reference waveform.

    The reference rate is estimated with the same estimator the waveform path
    uses, and windows it cannot resolve are excluded the same way, so the two
    numbers differ only in where the prediction came from.
    """
    from rooster.evaluation.physio_metrics import estimate_rate_bpm

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    errors, n_unresolved, n_total = [], 0, 0
    for x, y in loader:
        predicted = model.predict_rate_bpm(x.to(device)).cpu().numpy()
        for prediction, window in zip(predicted, y.numpy(), strict=True):
            n_total += 1
            reference = estimate_rate_bpm(np.asarray(window, dtype=np.float64), spec.target_fs, spec.target)
            if not np.isfinite(reference):
                n_unresolved += 1
                continue
            errors.append(abs(float(prediction) - reference))
    return {
        "rate_mae_bpm_head": float(np.mean(errors)) if errors else float("nan"),
        "rate_unresolved_frac_head": n_unresolved / max(n_total, 1),
    }


@torch.no_grad()
def _head_pinball(model, dataset, batch_size, device):
    """Pinball loss for both quantile readouts of the same model.

    `pinball_head` uses the model's quantile head; `pinball_ensemble` takes the
    empirical quantiles of its sampled ensemble at the same levels. Same weights,
    same windows, same levels -- so the difference is the readout.

    A quantile is a NONLINEAR functional of the predictive distribution, unlike
    the mean, which is why this is the forecasting-side test of the readout claim:
    the ensemble mean is already the MSE-optimal estimator, so MSE cannot show a
    readout gap even if one exists.
    """
    from rooster.models.quantile_head import pinball_loss

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    levels = list(model.quantile_levels)
    head_total, ensemble_total, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        head = model.predict_quantiles(x)
        samples = model.sample(x, model.n_point_samples)
        empirical = torch.quantile(samples, torch.tensor(levels, device=device, dtype=samples.dtype), dim=0)
        empirical = empirical.permute(*range(1, empirical.dim()), 0)
        head_total += float(pinball_loss(head.reshape(-1, len(levels)), y.reshape(-1), levels)) * y.numel()
        ensemble_total += float(pinball_loss(empirical.reshape(-1, len(levels)), y.reshape(-1), levels)) * y.numel()
        n += y.numel()
    if not n:
        return {}
    return {"pinball_head": head_total / n, "pinball_ensemble": ensemble_total / n}


def _alignment_metrics(model):
    """The learned condition-to-target offsets, so the claim is a recorded number.

    The mechanism's whole argument is that the correspondence between condition and
    target positions is learned rather than assumed -- identity for time-aligned
    translation, a seasonal lag for extrapolation. That is only a claim if the
    offsets travel with the results, so they are written into every cell that has
    them.
    """
    align = getattr(model, "relative_align", None)
    if align is None:
        return {}
    if not all(hasattr(align, name) for name in ("offset_centre", "log_period", "log_sharpness")):
        return {}
    centres = [float(c) for c in align.offset_centre.detach().cpu()]
    periods = [float(p) for p in align.log_period.detach().exp().cpu()]
    sharpness = [float(k) for k in align.log_sharpness.detach().exp().cpu()]
    # Read defensively: this reached the queue reading `log_width`, a parameter the
    # comb parameterisation had already removed, so every cell trained to completion
    # and then died in evaluation. Training is expensive and recording a diagnostic
    # is not, so a diagnostic must never be able to discard a finished run.
    return {
        "align_centres": centres,
        "align_periods": periods,
        "align_sharpness": sharpness,
        # The sharpest head carries the positional commitment; the flat ones are
        # closer to unstructured context.
        "align_centre_sharpest": centres[int(max(range(len(sharpness)), key=sharpness.__getitem__))],
        "align_period_sharpest": periods[int(max(range(len(sharpness)), key=sharpness.__getitem__))],
    }


def run_forecast_cell(model_name, dataset_name, horizon, budget, seed, device, n_samples=1, lookback=None, data_root=None, max_test_windows=None):
    """Train and evaluate one (model, dataset, horizon, seed) forecasting cell."""
    fix_seed(seed)
    series = load_forecast_series(dataset_name, **({"root": data_root} if data_root else {}))
    lookback = lookback if lookback is not None else series.spec.lookback
    splits, _scaler = build_forecast_windows(series, lookback, horizon)

    model = build_model(
        model_name,
        lookback=lookback,
        horizon=horizon,
        n_variates=series.n_variates,
        dt_seconds=FREQ_SECONDS.get(series.spec.freq, 1.0),
    )
    # A model whose penalty is derived from the dataset size (an MDL / BIC
    # criterion rather than a tuned coefficient) needs N. It comes from the
    # harness so the number is the protocol's, not something the model guesses.
    if hasattr(model, "n_train_samples"):
        model.n_train_samples = len(splits["train"]) * series.n_variates

    checkpoint_name = f"forecast_{series.spec.name}_h{horizon}_{model_name}_s{seed}"
    train_seconds = model.fit(splits["train"], splits["val"], budget, device, checkpoint_name=checkpoint_name)

    # Component selection happens here, between training and evaluation, and only
    # ever looks at validation. Doing it inside training would need a penalty
    # coefficient, which is the thing this replaces; doing it on test would make
    # the reported error a training metric.
    selection = None
    if getattr(model, "selects_cardinality", False):
        selection_batches = _capped(splits["val"], _SELECTION_WINDOWS)[0]
        loader = DataLoader(selection_batches, batch_size=budget.batch_size, shuffle=False, num_workers=0)
        selection = model.select_cardinality(loader, device=device)

    # A deterministic model is evaluated with a single sample: tiling one point
    # forecast N times would waste memory and cannot change any metric.
    effective_samples = n_samples if model.is_probabilistic else 1
    test_split, eval_stride = _capped(splits["test"], max_test_windows)
    metrics, n_test_windows = _stream_forecast_metrics(model, test_split, budget.batch_size, effective_samples, device)
    metrics.update(_alignment_metrics(model))
    if hasattr(model, "predict_quantiles") and getattr(model, "quantile_head", None) is not None:
        metrics.update(_head_pinball(model, test_split, budget.batch_size, device))
    metrics["train_seconds"] = train_seconds
    metrics["n_parameters"] = count_parameters(model)
    metrics["n_test_windows"] = n_test_windows
    # Recorded so that any choice BETWEEN cells -- which component count, which
    # variant -- can be made on validation and shown to have been made there.
    if getattr(model, "best_val_loss", None) is not None and math.isfinite(model.best_val_loss):
        metrics["val_mse"] = float(model.best_val_loss)
        metrics["best_step"] = int(model.best_step)
    if selection is not None:
        metrics["k_selected"] = selection["k_selected"]
        metrics["n_selection_windows"] = selection["n_val_windows"]
    if hasattr(model, "cardinality"):
        metrics.update(_measure_cardinality(model, splits["val"], budget.batch_size, device))

    return RunRecord(
        task="forecast",
        dataset=series.spec.name,
        model=model_name,
        horizon=horizon,
        seed=seed,
        metrics=metrics,
        lookback=lookback,
        protocol=f"{series.spec.boundaries}/scaled-space/{series.n_variates}var/stride{eval_stride}",
        budget=budget_label(budget.max_steps, n_samples, max_test_windows, selection=getattr(budget, 'selection', 'mse'), recipe=getattr(budget, 'recipe', None), epochs=getattr(budget, 'epochs', None)),
    )


def run_reconstruction_cell(task_name, model_name, budget, seed, device, n_samples=1, data_root=None, max_test_windows=None):
    """Train and evaluate one (model, PPG-reconstruction task, seed) cell."""
    fix_seed(seed)
    spec, recordings = load_recordings(task_name, **({"root": data_root} if data_root else {}))
    split_recordings = subject_split(recordings)
    window_samples = int(round(spec.window_seconds * spec.target_fs))

    # PENGUIN sizes its conv front end from the sample rate, so the real rate is
    # passed to any model that accepts it rather than left at a default.
    # `target_kind` reaches any model that supervises a rate: the rate of a
    # respiration trace lives in a different band from a heart rate, and a model
    # trained against the wrong band would be learning nothing. build_model drops
    # kwargs a model does not accept, so this is inert for the others.
    model = build_model(model_name, window_samples=window_samples, sample_rate=spec.target_fs, target_kind=spec.target)
    # The model is built before the windows because the windows' shape depends on
    # it: a model that declares `wants_covariate` receives channel-stacked
    # conditions (signal + motion), everything else the plain 1-D windows it has
    # always seen.
    splits = build_reconstruction_windows(spec, split_recordings, covariate=getattr(model, "wants_covariate", False))
    checkpoint_name = f"reconstruct_{spec.name}_{model_name}_s{seed}"
    train_seconds = model.fit(splits["train"], splits["val"], budget, device, checkpoint_name=checkpoint_name)

    effective_samples = n_samples if model.is_probabilistic else 1
    test_split, eval_stride = _capped(splits["test"], max_test_windows)
    samples, targets = _collect_predictions(model, test_split, budget.batch_size, effective_samples, device)
    metrics = evaluate_reconstruction(samples, targets, spec.target_fs, spec.target)
    # A model that predicts the rate directly gets that reading reported next to
    # the one taken off its generated waveform. Both come from the same weights on
    # the same windows, so the gap between them is the cost of the
    # reconstruct-then-measure path with everything else held fixed.
    metrics.update(_alignment_metrics(model))
    if hasattr(model, "predict_rate_bpm"):
        metrics.update(_head_rate_error(model, test_split, budget.batch_size, device, spec))
    metrics["train_seconds"] = train_seconds
    metrics["n_parameters"] = count_parameters(model)
    metrics["n_test_windows"] = int(targets.shape[0])

    n_subjects = {split: len(records) for split, records in split_recordings.items()}
    return RunRecord(
        task="reconstruct",
        dataset=spec.name,
        model=model_name,
        horizon=window_samples,
        seed=seed,
        metrics=metrics,
        lookback=window_samples,
        protocol=f"subject-wise {n_subjects['train']}/{n_subjects['val']}/{n_subjects['test']} @ {spec.target_fs:g}Hz/stride{eval_stride}",
        budget=budget_label(budget.max_steps, n_samples, max_test_windows, selection=getattr(budget, 'selection', 'mse'), recipe=getattr(budget, 'recipe', None), epochs=getattr(budget, 'epochs', None)),
    )
