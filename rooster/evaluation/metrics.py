"""Forecast metrics: point accuracy and probabilistic quality.

Point metrics (MSE/MAE) are computed **in scaled space**, matching the LTSF
convention -- see docs/02_benchmark_protocol.md.

Probabilistic metrics take an ensemble of sampled trajectories with a leading
sample axis, `(n_samples, batch, horizon, n_variates)`. A deterministic model
is just `n_samples == 1`, so the same code path serves both and the leaderboard
never has to special-case them.

CRPS naming trap, because three different things circulate under similar names
and are routinely conflated in the literature:

* `crps`      -- per-timestep, per-variate, averaged. The univariate GluonTS
                 protocol; this is what TSFlow reports.
* `crps_sum`  -- CRPS of the series summed across variates. The TimeGrad /
                 GluonTS *multivariate* protocol; this is what TimeGrad, CSDI
                 and MG-TSD report. It rewards getting cross-variate
                 correlation right and is NOT comparable to plain CRPS.
* normalised CRPS / WQL -- divided by the sum of ground truth. GIFT-Eval and
                 Chronos-style leaderboards. Not implemented here; add it
                 deliberately if we ever target that benchmark.
"""

import numpy as np

# Nominal coverage levels reported alongside CRPS. A model can have good CRPS and
# badly miscalibrated intervals, and we need to know which we have.
COVERAGE_LEVELS = (0.5, 0.8, 0.9)


def mse(predictions, targets):
    return float(np.mean((np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)) ** 2))


def mae(predictions, targets):
    return float(np.mean(np.abs(np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64))))


def crps_ensemble(samples, targets):
    """Empirical CRPS from an ensemble, averaged over every scalar prediction.

    Uses the energy form
    ``CRPS = E|X - y| - 0.5 * E|X - X'|``
    estimated as
    ``(1/n) sum_i |x_i - y| - (1/(2 n^2)) sum_i sum_j |x_i - x_j|``.

    The `1/(2 n^2)` normalisation (rather than the unbiased `1/(2 n (n-1))`) is
    the "fair CRPS" convention used by GluonTS and therefore by every number we
    would compare against; it is slightly biased for small `n`, which is why
    the sample count belongs in the leaderboard next to the score.

    `samples` is `(n_samples, ...)`; `targets` broadcasts against `samples[0]`.
    """
    samples = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n_samples = samples.shape[0]

    if n_samples == 1:
        # A point forecast's CRPS degenerates to MAE, which is the honest answer
        # rather than a spuriously good score.
        return mae(samples[0], targets)

    absolute_error = np.abs(samples - targets[None, ...]).mean(axis=0)
    # Pairwise spread. Materialising the (n, n, ...) difference is fine for the
    # ensemble sizes we use (<= a few hundred) and keeps the estimator obvious.
    pairwise = np.abs(samples[:, None, ...] - samples[None, :, ...]).sum(axis=(0, 1))
    spread = pairwise / (2.0 * n_samples**2)
    return float(np.mean(absolute_error - spread))


def crps_sum(samples, targets):
    """CRPS of the variate-summed series (the TimeGrad multivariate protocol).

    Expects `(n_samples, batch, horizon, n_variates)` and sums the trailing
    variate axis before scoring, so a model that gets the marginals right but
    the cross-variate correlation wrong is penalised.
    """
    samples = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if samples.ndim != 4:
        raise ValueError(f"crps_sum expects (n_samples, batch, horizon, n_variates); got {samples.shape}")
    return crps_ensemble(samples.sum(axis=-1), targets.sum(axis=-1))


def interval_coverage(samples, targets, levels=COVERAGE_LEVELS):
    """Empirical coverage of central prediction intervals at each nominal level.

    A well-calibrated model returns coverage ≈ the nominal level. Returned as a
    dict keyed by the level so the leaderboard can render `cov@80` columns.
    """
    samples = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if samples.shape[0] < 2:
        return {level: float("nan") for level in levels}

    coverage = {}
    for level in levels:
        tail = (1.0 - level) / 2.0
        lower = np.quantile(samples, tail, axis=0)
        upper = np.quantile(samples, 1.0 - tail, axis=0)
        coverage[level] = float(np.mean((targets >= lower) & (targets <= upper)))
    return coverage


class ForecastMetrics:
    """Streaming accumulator over test batches.

    Exists because materialising a whole test split does not scale: Traffic at
    horizon 720 is 3509 windows x 720 steps x 862 variates, which is ~9 GB in
    float32 for the predictions alone and more again once metrics promote to
    float64. Every metric here is a mean over scalars, so accumulating sums and
    counts per batch is *exactly* equal to the whole-array computation -- this
    is a memory refactor, not an approximation.

    `update` takes `(n_samples, batch, horizon, n_variates)` and
    `(batch, horizon, n_variates)`; `compute` returns the same dict
    `evaluate_forecast` does.
    """

    def __init__(self, levels=COVERAGE_LEVELS):
        self.levels = tuple(levels)
        self.n_samples = None
        self._scalar_count = 0
        self._squared_error = 0.0
        self._absolute_error = 0.0
        # The same errors for the ensemble mean, tracked separately so a model
        # carrying a point head reports BOTH readouts rather than only the one it
        # prefers. See `update`.
        self._squared_error_of_mean = 0.0
        self._absolute_error_of_mean = 0.0
        self._crps = 0.0
        self._summed_count = 0
        self._crps_summed = 0.0
        self._coverage_hits = dict.fromkeys(self.levels, 0)

    def update(self, samples, targets, point=None):
        """`point` is the model's declared point forecast; defaults to the mean.

        A model that trains an estimator specifically for squared error should be
        scored on it, rather than on a Monte-Carlo estimate of the same quantity
        taken from its ensemble. Deterministic models and generative models
        without a point head are unaffected -- the default is exactly the previous
        behaviour.
        """
        samples = np.asarray(samples, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        _check_forecast_shapes(samples, targets)
        if self.n_samples is None:
            self.n_samples = int(samples.shape[0])
        elif self.n_samples != samples.shape[0]:
            raise ValueError(f"ensemble size changed mid-evaluation: {self.n_samples} then {samples.shape[0]}")

        point = samples.mean(axis=0) if point is None else np.asarray(point, dtype=np.float64)
        if point.shape != targets.shape:
            raise ValueError(f"point forecast shape {point.shape} does not match target shape {targets.shape}")
        self._scalar_count += targets.size
        self._squared_error += float(((point - targets) ** 2).sum())
        self._absolute_error += float(np.abs(point - targets).sum())

        # Both readouts, always. The physio side of this project found a 2.1-3.9x
        # difference between a directly supervised readout and the same quantity
        # derived from generated output, and could only find it because both were
        # recorded side by side. Here the derived readout is the ensemble mean.
        # Squared error is minimised by that mean, so the gap should be SMALL --
        # which is the point: it bounds the claim to readouts that are nonlinear
        # functionals of the output, rather than letting it read as universal.
        ensemble_mean = samples.mean(axis=0)
        self._squared_error_of_mean += float(((ensemble_mean - targets) ** 2).sum())
        self._absolute_error_of_mean += float(np.abs(ensemble_mean - targets).sum())
        self._crps += _crps_scalar_sum(samples, targets)

        summed_samples, summed_targets = samples.sum(axis=-1), targets.sum(axis=-1)
        self._summed_count += summed_targets.size
        self._crps_summed += _crps_scalar_sum(summed_samples, summed_targets)

        if samples.shape[0] >= 2:
            for level in self.levels:
                tail = (1.0 - level) / 2.0
                lower = np.quantile(samples, tail, axis=0)
                upper = np.quantile(samples, 1.0 - tail, axis=0)
                self._coverage_hits[level] += int(np.count_nonzero((targets >= lower) & (targets <= upper)))

    def compute(self):
        if self._scalar_count == 0:
            raise RuntimeError("ForecastMetrics.compute() called before any update()")
        metrics = {
            "mse": self._squared_error / self._scalar_count,
            "mae": self._absolute_error / self._scalar_count,
            "crps": self._crps / self._scalar_count,
            "crps_sum": self._crps_summed / self._summed_count,
            "mse_of_mean": self._squared_error_of_mean / self._scalar_count,
            "mae_of_mean": self._absolute_error_of_mean / self._scalar_count,
            "n_samples": self.n_samples,
        }
        deterministic = self.n_samples < 2
        for level in self.levels:
            hits = self._coverage_hits[level]
            metrics[f"coverage_{int(level * 100)}"] = float("nan") if deterministic else hits / self._scalar_count
        return metrics


def _check_forecast_shapes(samples, targets):
    if samples.ndim != targets.ndim + 1:
        raise ValueError(f"expected samples (S,B,H,C) and targets (B,H,C); got {samples.shape} and {targets.shape}")
    if samples.shape[1:] != targets.shape:
        raise ValueError(f"sample shape {samples.shape[1:]} does not match target shape {targets.shape}")


def _crps_scalar_sum(samples, targets):
    """Sum (not mean) of per-scalar CRPS, so it can be accumulated across batches."""
    n_samples = samples.shape[0]
    if n_samples == 1:
        return float(np.abs(samples[0] - targets).sum())
    absolute_error = np.abs(samples - targets[None, ...]).mean(axis=0)
    pairwise = np.abs(samples[:, None, ...] - samples[None, :, ...]).sum(axis=(0, 1))
    return float((absolute_error - pairwise / (2.0 * n_samples**2)).sum())


def evaluate_forecast(samples, targets, levels=COVERAGE_LEVELS):
    """All forecasting metrics for one (model, dataset, horizon) cell.

    `samples`: `(n_samples, batch, horizon, n_variates)`.
    `targets`: `(batch, horizon, n_variates)`.

    The point forecast is the **ensemble mean**, never best-of-N -- reporting
    best-of-N would be cheating and is an easy mistake to make accidentally.

    Whole-array convenience wrapper around `ForecastMetrics`; the harness uses
    the streaming form directly.
    """
    samples = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if samples.ndim != 4 or targets.ndim != 3:
        raise ValueError(f"expected samples (S,B,H,C) and targets (B,H,C); got {samples.shape} and {targets.shape}")

    accumulator = ForecastMetrics(levels)
    accumulator.update(samples, targets)
    return accumulator.compute()
