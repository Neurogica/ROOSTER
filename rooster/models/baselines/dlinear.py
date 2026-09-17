"""DLinear (Zeng et al., AAAI 2023, arXiv:2205.13504)."""

import torch
import torch.nn as nn

from rooster.models.baselines.registry import register_benchmark_model
from rooster.models.baselines.training import BenchmarkModel


class _MovingAverage(nn.Module):
    """Trend extractor: centred moving average with replicated edge padding."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):  # (B, L, C)
        left = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        right = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        padded = torch.cat([left, x, right], dim=1)
        return self.pool(padded.permute(0, 2, 1)).permute(0, 2, 1)


@register_benchmark_model("DLinear")
class DLinear(BenchmarkModel):
    """Decomposition-linear forecaster (Zeng et al., AAAI 2023, arXiv:2205.13504).

    Splits the lookback into moving-average trend and seasonal remainder, maps
    each to the horizon with one shared `Linear(lookback -> horizon)`, and adds
    them back. Channel-independent with shared weights (`individual=False`),
    which is the configuration the published numbers use.

    Implemented here rather than vendored so it runs inside our harness on our
    splits -- the whole point is to check the harness, and a vendored trainer
    would defeat that.
    """

    def __init__(self, lookback, horizon, n_variates, kernel_size=25):
        super().__init__()
        self.decomposition = _MovingAverage(kernel_size)
        self.seasonal = nn.Linear(lookback, horizon)
        self.trend = nn.Linear(lookback, horizon)

    def forward(self, x):  # (B, L, C) -> (B, H, C)
        trend = self.decomposition(x)
        seasonal = x - trend
        seasonal_out = self.seasonal(seasonal.permute(0, 2, 1))
        trend_out = self.trend(trend.permute(0, 2, 1))
        return (seasonal_out + trend_out).permute(0, 2, 1)
