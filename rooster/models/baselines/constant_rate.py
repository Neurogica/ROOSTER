"""Emit a generic beat train at one fixed rate, ignoring the input entirely.

The control that decides whether a rate result means anything.

`CopyInput` is the floor for tasks where the input already carries the rhythm,
and on the clinical sets it is a demanding one -- 1.53 bpm on BIDMC-ECG. On
PPG-DaLiA it collapses to 18.37 bpm, and PENGUIN's 11.57 bpm looks like a real
win against it. But heart rate is a narrow distribution: most windows of most
subjects sit within a few tens of bpm of each other, so a model that emits a
plausible-looking beat train at a **constant** rate and never looks at the PPG
would also score in the low tens. Without measuring that number, "11.57 bpm"
cannot be distinguished from "learned nothing and guessed the population mean".

So this model does exactly that. It learns one scalar -- the rate that minimises
absolute rate error on the training split, i.e. the training median -- and
renders it as a periodic waveform. Any rate result that does not clearly beat it
is not a result.

Deliberately *not* conditioned on the input in any way: the whole point is to
measure how much of a rate score is available for free.
"""

import numpy as np
import torch

from rooster.models.baselines.registry import register_benchmark_model
from rooster.models.baselines.training import BenchmarkModel

# A beat rendered as a Gaussian bump rather than a delta, because a delta train
# has energy at every harmonic. The width is a fraction of one period, so a breath
# renders broad and a QRS narrow without the renderer being told which it is.
_BEAT_WIDTH_FRACTION = 0.05


@register_benchmark_model("ConstantRate")
class ConstantRate(BenchmarkModel):
    """A periodic beat train at the training-set median rate."""

    is_probabilistic = False

    def __init__(self, window_samples, sample_rate=125.0, target_kind="ECG"):
        super().__init__()
        self.window_samples = window_samples
        self.sample_rate = sample_rate
        self.target_kind = target_kind
        # A buffer, so the fitted rate travels with the checkpoint. Initialised to
        # a physiologically ordinary resting rate; `fit` replaces it.
        self.register_buffer("rate_bpm", torch.tensor(75.0))
        # One unused parameter so the optimiser and checkpoint machinery in
        # `BenchmarkModel.fit` have something to work with. It never affects the
        # output -- see `forward`.
        self.unused = torch.nn.Parameter(torch.zeros(1))

    def fit(self, train_dataset, val_dataset, budget, device, checkpoint_name=None):
        """Set the rate to the training split's median, then stop.

        No gradient step: there is one scalar and its minimiser under absolute
        error is the median, so descending on it would only add noise.
        """
        from rooster.evaluation.physio_metrics import estimate_rate_bpm

        self.to(device)
        rates = []
        for index in range(len(train_dataset)):
            _condition, target = train_dataset[index]
            rate = estimate_rate_bpm(np.asarray(target, dtype=np.float64), self.sample_rate, self.target_kind)
            if np.isfinite(rate):
                rates.append(rate)
        if rates:
            self.rate_bpm.fill_(float(np.median(rates)))
        self.best_val_loss, self.best_step = float("nan"), 0
        return 0.0

    def forward(self, x):
        times = torch.arange(self.window_samples, device=x.device, dtype=torch.float32) / self.sample_rate
        period = 60.0 / self.rate_bpm
        # Distance to the nearest beat, then a Gaussian bump around it.
        phase = torch.remainder(times, period)
        distance = torch.minimum(phase, period - phase)
        beat = torch.exp(-0.5 * (distance / (_BEAT_WIDTH_FRACTION * period)) ** 2)
        beat = (beat - beat.mean()) / beat.std().clamp_min(1e-6)
        # `0 * unused` keeps the parameter in the graph without changing the value,
        # so the shared training and checkpoint code does not special-case this.
        return beat.unsqueeze(0).expand(x.shape[0], -1) + 0.0 * self.unused
