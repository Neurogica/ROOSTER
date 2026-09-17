"""Sliding-window `torch.utils.data.Dataset` wrappers over loaded series.

Two window types, one per task, mirroring the project's framing that both tasks
are "generate a target sequence from an observed sequence":

* `ForecastWindows`   -- lookback of a series -> the next `horizon` steps of the
  *same* variates (the target follows the condition in time).
* `ReconstructionWindows` -- a PPG window -> the *same timespan* of a different
  signal (ECG or respiration).

Scaling is applied here rather than in the loaders so a single fitted scaler is
shared across all three splits, which is what makes the train-only fit
enforceable.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from rooster.datasets.splits import StandardScaler, windowed_bounds


class ForecastWindows(Dataset):
    """Stride-1 lookback/horizon windows drawn from one chronological split.

    Yields `(x, y)` with `x` `(lookback, n_variates)` and `y`
    `(horizon, n_variates)`, both standardised with the train-fitted scaler.
    All variates are returned together; a channel-independent model folds the
    variate axis into the batch itself.
    """

    def __init__(self, series, split, lookback, horizon, scaler):
        self.lookback = lookback
        self.horizon = horizon
        self.scaler = scaler

        start, end = windowed_bounds(series.boundaries, split, lookback)
        self.values = scaler.transform(series.values[start:end]).astype(np.float32)

        self.n_windows = len(self.values) - lookback - horizon + 1
        if self.n_windows <= 0:
            raise ValueError(f"{series.spec.name} split {split!r} holds {len(self.values)} rows, too few for lookback {lookback} + horizon {horizon}")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, index):
        cut = index + self.lookback
        return (
            torch.from_numpy(self.values[index:cut]),
            torch.from_numpy(self.values[cut : cut + self.horizon]),
        )


def build_forecast_windows(series, lookback, horizon):
    """Fit the scaler on train, then build all three splits from it.

    Returns `(splits, scaler)`. The scaler is handed back because physical-unit
    reporting and plotting need it -- the forecasting table itself stays in
    scaled space, per the LTSF convention.
    """
    train_start, train_end = series.boundaries.bounds("train")
    scaler = StandardScaler().fit(series.values[train_start:train_end])
    splits = {split: ForecastWindows(series, split, lookback, horizon, scaler) for split in ("train", "val", "test")}
    return splits, scaler


class ReconstructionWindows(Dataset):
    """Non-overlapping condition/target windows over the same timespan.

    Yields `(ppg, target)`, each `(window_samples,)`. Windows are
    non-overlapping rather than stride-1: consecutive stride-1 windows of an
    8-minute recording are almost identical, so overlapping them inflates the
    apparent dataset size without adding information, and makes the eval set
    correlated.

    Each window is z-scored **individually**. That is the convention in the
    PPG->X literature (PPG has no meaningful absolute scale -- it depends on
    sensor contact and skin tone), and it is what makes waveform metrics
    comparable across recordings.
    """

    def __init__(self, recordings, window_samples, min_target_std=1e-4, covariate=False):
        self.window_samples = window_samples
        conditions, targets, covariates = [], [], []
        for recording in recordings:
            recording_covariate = None
            if covariate and recording.covariate is not None:
                # Normalised per RECORDING, not per window: how much motion a window
                # holds relative to the rest of the recording is the signal; a
                # per-window z-score would erase it (every window becomes unit
                # variance, still versus sedentary alike).
                spread = recording.covariate.std() + 1e-8
                recording_covariate = (recording.covariate - np.median(recording.covariate)) / spread
            n_windows = len(recording.condition) // window_samples
            for index in range(n_windows):
                lo = index * window_samples
                hi = lo + window_samples
                condition = recording.condition[lo:hi]
                target = recording.target[lo:hi]
                # A flat target window carries no signal to reconstruct and would
                # divide by ~0 during z-scoring, so drop it rather than emit NaNs.
                if target.std() < min_target_std or condition.std() < min_target_std:
                    continue
                conditions.append(_zscore(condition))
                targets.append(_zscore(target))
                if covariate:
                    covariates.append(
                        recording_covariate[lo:hi]
                        if recording_covariate is not None
                        else np.zeros(window_samples, dtype=np.float64)
                    )

        if not conditions:
            raise ValueError(f"no usable windows of {window_samples} samples in {len(recordings)} recordings")
        self.conditions = np.stack(conditions).astype(np.float32)
        if covariate:
            # Channel-stacked condition: (n_windows, 2, window_samples), PPG first.
            self.conditions = np.stack([self.conditions, np.stack(covariates).astype(np.float32)], axis=1)
        self.targets = np.stack(targets).astype(np.float32)

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, index):
        return torch.from_numpy(self.conditions[index]), torch.from_numpy(self.targets[index])


def _zscore(window):
    centred = window - window.mean()
    return centred / (centred.std() + 1e-8)


def build_reconstruction_windows(spec, split_recordings, covariate=False):
    """Build windowed datasets for each subject-wise split of a physio task."""
    window_samples = int(round(spec.window_seconds * spec.target_fs))
    return {
        split: ReconstructionWindows(recordings, window_samples, covariate=covariate)
        for split, recordings in split_recordings.items()
    }
