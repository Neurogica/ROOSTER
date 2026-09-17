"""Weight each training window by how rare its conditioning statistics are.

The failure this addresses is measured, and it is the same shape on both tasks.

On BIDMC-ECG the trained model's errors are not spread out: the median window is
0.55 bpm and the worst 10% of windows carry 62% of the total error. Broken down by
the reference heart rate:

    < 70 bpm     8.2% of training windows    mean error 7.46 bpm, bias +6.58
    70-85       26.2%                                   1.02        -0.77
    85-100      45.0%                                   1.35        -1.10
    100-200     20.6%                                   0.99        -0.57

The estimator is not at fault -- on ideal beat trains it reads 50 bpm as 50.0 and 60
as 60.0. The model is simply pulled toward the majority band: every well-populated
band is biased slightly low, and the sparse band is biased 6.6 bpm high. That is
regression to the mean under an imbalanced conditional distribution, and forecasting
has the same structure -- rare regimes are dragged toward the common ones.

So the weight is not physiological and not task-specific. It is `1 / density` of the
window's own summary statistics, estimated from the training split by histogram, and
it applies wherever some conditions are rarer than others.

Two things keep this from becoming another tuned constant. The density is estimated
from the data rather than chosen, and the weights are normalised to mean 1 so the
effective learning rate does not change -- turning the weighting on cannot be
confused with turning the step size up.
"""

import numpy as np
import torch

# Clipped because inverse density is unbounded: a single window in an empty bin
# would otherwise dominate the epoch. 10x is well above the 5.5x imbalance measured
# between the sparsest and densest heart-rate bands, so it binds only on outliers.
MAX_WEIGHT = 10.0


class RarityWeights:
    """`1 / density` over a scalar summary of each window, normalised to mean 1."""

    def __init__(self, values, n_bins=20, max_weight=MAX_WEIGHT):
        values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        if values.size < n_bins:
            self.edges, self.weights = None, None
            return
        # UNIFORM-width bins over the observed range. Quantile edges were the first
        # attempt and they are self-defeating: quantile bins hold equal counts by
        # construction, so the density is flat and 1/density is a constant. Measured,
        # every weight came out between 0.99 and 1.01 -- the term was inert.
        #
        # The range is trimmed to the 1st-99th percentile before binning so a single
        # outlier cannot stretch the grid and leave every real window in one bin.
        low, high = np.percentile(values, [1.0, 99.0])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            self.edges, self.weights = None, None
            return
        self.edges = np.linspace(low, high, n_bins + 1)
        self.edges[0], self.edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(values, bins=self.edges)
        density = counts / max(counts.sum(), 1)
        with np.errstate(divide="ignore"):
            weights = np.where(density > 0, 1.0 / np.maximum(density, 1e-12), 0.0)
        weights = np.clip(weights, 0.0, max_weight * weights[weights > 0].min() if (weights > 0).any() else 1.0)
        # Mean 1 over the training distribution, so this reweights without also
        # rescaling the gradient.
        expected = float((density * weights).sum())
        self.weights = weights / expected if expected > 0 else weights

    def __call__(self, values):
        """Per-window weights for a batch of summary values."""
        if self.weights is None:
            return None
        values = np.asarray(values, dtype=np.float64)
        index = np.clip(np.digitize(values, self.edges[1:-1]), 0, len(self.weights) - 1)
        out = self.weights[index]
        out[~np.isfinite(values)] = 1.0  # unresolvable windows are not rare, just unusable
        return torch.tensor(out, dtype=torch.float32)

    def summary(self):
        if self.weights is None:
            return "rarity weighting inactive (too few windows)"
        return f"weights {self.weights.min():.2f}-{self.weights.max():.2f} over {len(self.weights)} uniform bins"
