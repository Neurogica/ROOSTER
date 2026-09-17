"""Persistence baseline: hold the last observed value across the horizon."""

from rooster.models.baselines.registry import register_benchmark_model
from rooster.models.baselines.training import BenchmarkModel, _UntrainedMixin


@register_benchmark_model("RepeatLast")
class RepeatLast(_UntrainedMixin, BenchmarkModel):
    """Persistence: the last observed value, held for the whole horizon.

    The floor every forecaster must clear. On standardised data it is also a
    useful scale check -- MSE much above ~1.0 means something is wrong with the
    scaler, not with the model.
    """

    def __init__(self, lookback, horizon, n_variates):
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        return x[:, -1:, :].expand(-1, self.horizon, -1)
