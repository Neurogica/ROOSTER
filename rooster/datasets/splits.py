"""Chronological train/val/test splitting and train-only standardisation.

This is the part of the benchmark that has to be exactly right or every number
downstream is incomparable to the published literature, so the conventions are
spelled out rather than inferred:

* Splits are **chronological**, never shuffled. ETT uses the fixed 12/4/4-month
  boundaries from the Informer paper; everything else uses a ratio split in time
  order. PEMS uses 60/20/20, which is the STGNN convention rather than the
  70/10/20 used for the Autoformer-family datasets.
* The scaler is fit on the **training split only** and applied to val/test.
* MSE/MAE are reported **in the scaled space**. That is the LTSF convention and
  is why published numbers look like 0.17 rather than something in physical
  units. Deviating makes us incomparable, so `inverse_transform` exists for
  plotting and for physiological metrics, not for the forecasting table.
* Windows never straddle a split boundary. Each split owns a contiguous slice
  and windows are drawn inside it, so a lookback can never peek across the cut.

See docs/02_benchmark_protocol.md for the full contract.
"""

from dataclasses import dataclass

import numpy as np

# ETT uses fixed sample counts rather than ratios: 12 months train, 4 val, 4 test.
# 30 days/month is the convention the original Informer code uses -- not calendar months.
_ETT_HOURS_PER_MONTH = 30 * 24
ETT_HOURLY_BOUNDARIES = (12 * _ETT_HOURS_PER_MONTH, 16 * _ETT_HOURS_PER_MONTH, 20 * _ETT_HOURS_PER_MONTH)
ETT_MINUTE_BOUNDARIES = tuple(4 * b for b in ETT_HOURLY_BOUNDARIES)  # 15-min sampling -> 4x as many rows


@dataclass(frozen=True)
class SplitBoundaries:
    """Row indices delimiting the three chronological splits of a series."""

    train_end: int
    val_end: int
    test_end: int

    def bounds(self, split):
        if split == "train":
            return 0, self.train_end
        if split == "val":
            return self.train_end, self.val_end
        if split == "test":
            return self.val_end, self.test_end
        raise ValueError(f"unknown split {split!r}; expected train/val/test")


def ratio_boundaries(n_rows, train_ratio, val_ratio):
    """Chronological split at the given ratios; the remainder becomes test."""
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    return SplitBoundaries(train_end=train_end, val_end=val_end, test_end=n_rows)


def fixed_boundaries(n_rows, train_end, val_end, test_end):
    """Split at explicit row counts, clipped to the series length (ETT)."""
    return SplitBoundaries(
        train_end=min(train_end, n_rows),
        val_end=min(val_end, n_rows),
        test_end=min(test_end, n_rows),
    )


def windowed_bounds(boundaries, split, lookback):
    """Row range a split's windows may start in.

    Val and test are extended *backwards* by `lookback` rows so the first
    prediction in each split starts exactly at the split boundary -- the
    standard LTSF behaviour. The lookback therefore reads a few rows from the
    preceding split, which is legitimate (those rows are observed history at
    inference time) and is what every published number assumes. Train is not
    extended, since there is nothing before it.
    """
    start, end = boundaries.bounds(split)
    if split != "train":
        start = max(0, start - lookback)
    return start, end


class StandardScaler:
    """Per-variate zero-mean unit-variance scaler, fit on the training split only.

    Deliberately not `sklearn.preprocessing.StandardScaler`: keeping it here
    avoids a dependency, makes the train-only fit impossible to get wrong by
    accident, and lets `inverse_transform` broadcast over a trailing variate
    axis of any rank (needed for `(batch, horizon, variate)` forecast tensors).
    """

    def __init__(self, eps=1e-8):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, train_values):
        values = np.asarray(train_values, dtype=np.float64)
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)
        # A constant variate would otherwise divide by zero and produce NaNs that
        # only surface much later, inside the loss.
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        return self

    def transform(self, values):
        self._check_fitted()
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse_transform(self, values):
        self._check_fitted()
        return np.asarray(values, dtype=np.float64) * self.std + self.mean

    def fit_transform(self, train_values):
        return self.fit(train_values).transform(train_values)

    def _check_fitted(self):
        if self.mean is None:
            raise RuntimeError("StandardScaler used before fit(); fit it on the TRAIN split only")
