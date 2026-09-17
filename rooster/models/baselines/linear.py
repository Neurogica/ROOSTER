"""A single shared Linear(lookback -> horizon): DLinear's own ablation."""

import torch.nn as nn

from rooster.models.baselines.registry import register_benchmark_model
from rooster.models.baselines.training import BenchmarkModel


@register_benchmark_model("LinearForecaster")
class LinearForecaster(BenchmarkModel):
    """One shared `Linear(lookback -> horizon)`, no decomposition.

    DLinear's own ablation: if this matches DLinear, the trend/seasonal split is
    contributing nothing on that dataset. Directly relevant to this project,
    whose central claim is about decomposition -- we need to know where
    decomposition helps at all before claiming a better one.
    """

    def __init__(self, lookback, horizon, n_variates):
        super().__init__()
        self.projection = nn.Linear(lookback, horizon)

    def forward(self, x):
        return self.projection(x.permute(0, 2, 1)).permute(0, 2, 1)


# --------------------------------------------------------------------------
# PPG reconstruction
# --------------------------------------------------------------------------
