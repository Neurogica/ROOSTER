"""PENGUIN baseline -- the authors' own code, adapted to this harness.

Reference: arXiv:2602.03858 (ICASSP 2026). The model is vendored verbatim in
`models/vendor/penguin_official.py`.

Two details the adapter must get right or it trains a different model:

* the loss is over the **velocity** (`pred_dx_t` vs `dx_t`), read off the module
  after `train_flow`; the waveform `train_flow` returns is only for monitoring
* `sample_rate` sizes the convolutional front end (`kernel_size = sample_rate //
  4`), so passing the wrong rate silently changes the architecture

Protocol differences from the published table are deliberate and recorded here:
the paper splits subjects 6:1:1 and resamples to 128 Hz, while this harness uses
its own 6:2:2 subject-wise split at each task's native rate for **every** model,
because comparability within the harness is what the leaderboard is for. So this
column is a like-for-like in-harness baseline, not a reproduction.
"""

import torch
import torch.nn.functional as F

from rooster.models.baselines.registry import register_benchmark_model
from rooster.models.baselines.training import BenchmarkModel
from rooster.models.vendor.penguin_official import PENGUIN as _PenguinOfficial


@register_benchmark_model("PENGUIN")
class Penguin(BenchmarkModel):
    """The authors' PENGUIN, wrapped for this harness.

    `sample_rate` is passed through because upstream sizes its convolutional
    front end as `kernel_size = sample_rate // 4` -- roughly a quarter-second
    receptive field. Feeding the wrong rate silently changes the architecture,
    so the harness supplies the task's real rate.
    """

    is_probabilistic = True
    n_point_samples = 8

    def __init__(self, window_samples, sample_rate=128, n_step=25, h_dim=16, ssm_block_num=4, max_ensemble_sequences=4_096):
        super().__init__()
        self.window_samples = window_samples
        # Full-length waveforms rather than patch tokens, so the ensemble working
        # set is capped much lower than the patch-based models'.
        self.max_ensemble_sequences = max_ensemble_sequences
        self.model = _PenguinOfficial(n_step=n_step, sample_rate=int(sample_rate), h_dim=h_dim, ssm_block_num=ssm_block_num)

    def compute_loss(self, x, y):
        """Velocity MSE, read off the module after `train_flow`.

        Upstream's `optimize()` computes `mse(self.pred_dx_t, self.dx_t)` -- the
        loss is over the velocity, not over the waveform `train_flow` returns
        (that is only for monitoring). Scoring the returned waveform instead
        would train a different objective.
        """
        self.model.train_flow(x, y)
        return F.mse_loss(self.model.pred_dx_t, self.model.dx_t)

    @torch.no_grad()
    def sample(self, x, n_samples):
        n_windows = x.shape[0]
        per_chunk = max(1, min(n_samples, self.max_ensemble_sequences // max(n_windows, 1)))
        draws = []
        for start in range(0, n_samples, per_chunk):
            width = min(per_chunk, n_samples - start)
            drawn = self.model.sample(x.repeat(width, 1))
            draws.extend(drawn[i * n_windows : (i + 1) * n_windows] for i in range(width))
        return torch.stack(draws, dim=0)

    def forward(self, x):
        return self.sample(x, self.n_point_samples).mean(dim=0)
