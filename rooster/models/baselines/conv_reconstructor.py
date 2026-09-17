"""Dilated 1-D conv encoder/decoder: PPG window -> target window."""

import torch.nn as nn

from rooster.models.baselines.registry import register_benchmark_model
from rooster.models.baselines.training import BenchmarkModel


@register_benchmark_model("ConvReconstructor")
class ConvReconstructor(BenchmarkModel):
    """Dilated 1-D conv encoder/decoder, PPG window -> target window.

    Deliberately small and unremarkable: its job is to be a competent,
    fast-to-train reference that the Flow-SSM variants must beat, not to be
    competitive with RDDM or PENGUIN. Dilations give a wide receptive field
    cheaply, which matters for respiration (a ~0.1-0.6 Hz process needs to see
    tens of seconds).
    """

    def __init__(self, window_samples, channels=64, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        layers = [nn.Conv1d(1, channels, kernel_size=9, padding=4), nn.GELU()]
        for dilation in dilations:
            layers += [
                nn.Conv1d(channels, channels, kernel_size=9, padding=4 * dilation, dilation=dilation),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            ]
        layers.append(nn.Conv1d(channels, 1, kernel_size=9, padding=4))
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # (B, T) -> (B, T)
        return self.network(x.unsqueeze(1)).squeeze(1)
