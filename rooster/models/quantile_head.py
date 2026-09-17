"""Monotone quantile head and pinball loss.

Why this exists, from the measurements in docs/04_error_analysis.md:

* A generative model's point forecast is the **ensemble mean**, a Monte-Carlo
  estimate of the MSE-optimal predictor. Four attempts to make it competitive
  with a directly-supervised deterministic baseline failed. A quantile head
  sidesteps the problem: the median is trained *directly* for absolute error, and
  a mean head *directly* for squared error.
* Our flow model was systematically overconfident -- 80% intervals covering
  0.59-0.72. Pinball loss fits the quantile levels themselves, so calibration is
  in the objective rather than an emergent property of an ODE solver.
* Sampling cost disappears: no ensemble, no integration steps, no discretisation
  error. CRPS follows from the quantile levels in closed form.

**Not a novelty claim.** FlowState (ICML 2026, arXiv:2508.05287) already pairs an
S5 encoder with a quantile head, and it is the nearest competitor to this project
(docs/prior_art.md). The head is machinery; the contribution has to live in the
decomposition.

Relation to ISQF (Park et al., 2022), which GluonTS implements: ISQF fits a
spline quantile *function*, giving a continuous CDF and a closed-form CRPS at any
level. This head is the simpler cousin -- a fixed grid of levels with monotonicity
enforced by construction (a base quantile plus non-negative increments). It buys
the two properties that matter here, non-crossing quantiles and a directly
calibrated objective, without the spline machinery. Move to full ISQF if
interpolating between levels or a closed-form CRPS at arbitrary levels is needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# A grid dense enough to estimate CRPS well and to report the usual intervals.
# Includes 0.5 so the median -- the MAE-optimal estimator -- is a fitted level
# rather than something interpolated.
DEFAULT_QUANTILE_LEVELS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def pinball_loss(predictions, targets, levels):
    """Mean pinball (quantile) loss over a grid of levels.

    `predictions` is `(..., n_levels)` and `targets` `(...)`. For level `q` the
    loss is `max(q * e, (q - 1) * e)` with `e = y - yhat`, which is minimised by
    the true `q`-quantile -- that is the whole point: each output head is pulled
    to its own quantile rather than to a compromise.

    Averaging pinball over a uniform grid of levels approximates CRPS up to a
    factor of two, so this objective is also a consistent CRPS surrogate.
    """
    levels = torch.as_tensor(levels, dtype=predictions.dtype, device=predictions.device)
    errors = targets.unsqueeze(-1) - predictions
    return torch.maximum(levels * errors, (levels - 1.0) * errors).mean()


def crps_from_quantiles(predictions, targets, levels):
    """CRPS estimated from a quantile grid: `2 * mean(pinball)` on a uniform grid.

    Exact in the limit of a dense uniform grid; on a finite grid it is the
    standard approximation and is what quantile-based forecasters report.
    """
    return 2.0 * pinball_loss(predictions, targets, levels)


class MonotoneQuantileHead(nn.Module):
    """Predicts a non-crossing set of quantiles, plus a separate mean.

    Monotonicity is structural, not a penalty: the head emits one base value and
    `n_levels - 1` non-negative increments, so `q_1 <= q_2 <= ...` holds exactly
    for every input, at every point in training. Quantile crossing is otherwise a
    routine failure of multi-output quantile regression and produces negative
    interval widths, which then corrupt coverage and CRPS.

    The mean is a **separate output**, not the median: MSE is minimised by the
    conditional mean and MAE by the conditional median, and on skewed data those
    differ. Both metrics go in the paper, so both estimators are trained.
    """

    def __init__(self, in_features, horizon, levels=DEFAULT_QUANTILE_LEVELS):
        super().__init__()
        self.levels = tuple(levels)
        if sorted(self.levels) != list(self.levels):
            raise ValueError(f"quantile levels must be ascending; got {self.levels}")
        self.median_index = self.levels.index(0.5) if 0.5 in self.levels else len(self.levels) // 2
        self.horizon = horizon

        self.mean_head = nn.Linear(in_features, horizon)
        self.base_head = nn.Linear(in_features, horizon)
        self.increment_head = nn.Linear(in_features, horizon * (len(self.levels) - 1))

    def forward(self, features):
        """`(batch, in_features)` -> `(mean (batch, horizon), quantiles (batch, horizon, n_levels))`."""
        batch = features.shape[0]
        mean = self.mean_head(features)
        base = self.base_head(features).unsqueeze(-1)
        increments = F.softplus(self.increment_head(features)).reshape(batch, self.horizon, len(self.levels) - 1)
        quantiles = torch.cat([base, base + increments.cumsum(dim=-1)], dim=-1)
        return mean, quantiles

    def loss(self, mean, quantiles, targets, mean_weight=1.0, quantile_weight=1.0):
        """Squared error on the mean plus pinball on the quantiles.

        Two terms because the paper reports two metrics: MSE wants the mean, MAE
        wants the median. Weighting them separately keeps either from being
        starved by the other's scale.
        """
        return mean_weight * F.mse_loss(mean, targets) + quantile_weight * pinball_loss(quantiles, targets, self.levels)

    def median(self, quantiles):
        return quantiles[..., self.median_index]

    def interval(self, quantiles, level):
        """Central prediction interval at a nominal `level`, e.g. 0.8.

        Returns `(lower, upper)` from the fitted levels nearest the required
        tails, and raises if the grid cannot express them -- silently widening to
        the nearest available level would report a coverage number for a
        different interval than the one asked for.
        """
        tail = (1.0 - level) / 2.0
        if not (min(self.levels) <= tail and max(self.levels) >= 1.0 - tail):
            raise ValueError(f"level {level} needs quantiles at {tail:.3f} and {1 - tail:.3f}; grid is {self.levels}")
        lower = min(range(len(self.levels)), key=lambda i: abs(self.levels[i] - tail))
        upper = min(range(len(self.levels)), key=lambda i: abs(self.levels[i] - (1.0 - tail)))
        return quantiles[..., lower], quantiles[..., upper]
