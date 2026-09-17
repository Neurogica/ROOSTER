"""Return the PPG window unchanged as the 'reconstruction'."""

from rooster.models.baselines.registry import register_benchmark_model
from rooster.models.baselines.training import BenchmarkModel, _UntrainedMixin


@register_benchmark_model("CopyInput")
class CopyInput(_UntrainedMixin, BenchmarkModel):
    """Return the (already z-scored) PPG window unchanged as the "reconstruction".

    Not a strawman: PPG genuinely carries the cardiac rhythm, so on PPG->ECG
    this scores a *low* heart-rate error and sets a demanding floor. On
    PPG->respiration it should do much worse. Publishing both is how we show a
    reconstruction model earns its keep rather than rediscovering the pulse.
    """

    def __init__(self, window_samples):
        super().__init__()

    def forward(self, x):
        return x
