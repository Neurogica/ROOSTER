"""DecompSSM baseline -- the authors' own code, adapted to this harness.

Reference: arXiv:2602.05389 (ICASSP 2026). The model itself is vendored verbatim
in `models/vendor/decompssm_official.py`; this file only fits its interface to
`BenchmarkModel`, so it trains under the same budget and splits as everything
else.

DecompSSM expects a TSLib-style `configs` namespace, returns predictions in the
original units (it carries its own non-stationary normalization), and exposes an
auxiliary decomposition loss via `get_auxiliary_loss()` that the caller must add
to the objective -- which upstream's training script does, and so does this one.
"""

import types

import torch.nn.functional as F

from rooster.models.baselines.registry import register_benchmark_model
from rooster.models.baselines.training import BenchmarkModel
from rooster.models.vendor.decompssm_official import Model as _DecompSSMOfficial

# Upstream defaults, read from the repository rather than guessed. Anything not
# listed falls back to the model's own getattr default.
DECOMPSSM_DEFAULTS = {
    "model_type": "s5",
    "d_model": 128,
    "state_size": 64,
    "dropout": 0.1,
    "lambda_reconstruction": 1.0,
    "lambda_orthogonality": 0.02,
    "aux_loss_weight": 0.1,
    "use_channel_interaction": True,
    "channel_interaction_strength": 0.15,
}


@register_benchmark_model("DecompSSM")
class DecompSSM(BenchmarkModel):
    """The authors' DecompSSM, wrapped for this harness.

    Deterministic, so its CRPS degenerates to MAE -- the honest reporting for a
    point forecaster.
    """

    is_probabilistic = False

    def __init__(self, lookback, horizon, n_variates, **overrides):
        super().__init__()
        settings = {**DECOMPSSM_DEFAULTS, **overrides}
        configs = types.SimpleNamespace(seq_len=lookback, pred_len=horizon, enc_in=n_variates, c_out=n_variates, **settings)
        self.model = _DecompSSMOfficial(configs)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y):
        """Forecast MSE plus the model's own auxiliary decomposition loss.

        Upstream adds `get_auxiliary_loss()` to the objective during training;
        omitting it here would train a different model from the published one.
        The auxiliary term is only populated in training mode, so validation
        (which runs under `eval()`) sees the plain MSE -- matching upstream.
        """
        loss = F.mse_loss(self(x), y)
        return loss + self.model.get_auxiliary_loss()
