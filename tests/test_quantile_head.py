"""Tests for the pinball loss and the monotone quantile head.

The properties checked are the ones the head exists to guarantee: quantiles that
never cross, a median that actually minimises absolute error, a mean that
actually minimises squared error, and a CRPS estimate consistent with the
ensemble one already in the harness.
"""

import numpy as np
import pytest
import torch

from rooster.evaluation.metrics import crps_ensemble
from rooster.models.quantile_head import DEFAULT_QUANTILE_LEVELS, MonotoneQuantileHead, crps_from_quantiles, pinball_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_pinball_is_minimised_at_the_true_quantile():
    """The defining property: for level q, the minimiser is the q-quantile."""
    torch.manual_seed(0)
    sample = torch.randn(20000)
    for level in (0.1, 0.5, 0.9):
        truth = torch.quantile(sample, level)
        candidates = truth + torch.tensor([-0.5, -0.2, 0.0, 0.2, 0.5])
        losses = [pinball_loss(sample.unsqueeze(-1) * 0 + c, sample, [level]).item() for c in candidates]
        assert int(np.argmin(losses)) == 2, f"level {level}: minimum not at the true quantile"


def test_pinball_at_the_median_is_half_the_absolute_error():
    predictions = torch.tensor([[1.0], [2.0]])
    targets = torch.tensor([0.0, 5.0])
    expected = 0.5 * torch.tensor([1.0, 3.0]).mean()
    assert pinball_loss(predictions, targets, [0.5]).item() == pytest.approx(expected.item())


def test_pinball_is_asymmetric_in_the_right_direction():
    """A high level must punish under-prediction more than over-prediction."""
    targets = torch.tensor([0.0])
    under = pinball_loss(torch.tensor([[-1.0]]), targets, [0.9])
    over = pinball_loss(torch.tensor([[1.0]]), targets, [0.9])
    assert under > over


def test_quantiles_never_cross():
    """Structural, so it must hold even for untrained weights and wild inputs."""
    torch.manual_seed(0)
    head = MonotoneQuantileHead(16, 8).to(DEVICE)
    features = torch.randn(64, 16, device=DEVICE) * 50.0
    with torch.no_grad():
        _mean, quantiles = head(features)
    differences = quantiles.diff(dim=-1)
    assert (differences >= 0).all(), f"quantile crossing: min gap {differences.min().item()}"


def test_head_shapes_and_median_selection():
    head = MonotoneQuantileHead(16, 8).to(DEVICE)
    mean, quantiles = head(torch.randn(5, 16, device=DEVICE))
    assert mean.shape == (5, 8)
    assert quantiles.shape == (5, 8, len(DEFAULT_QUANTILE_LEVELS))
    assert head.levels[head.median_index] == 0.5
    torch.testing.assert_close(head.median(quantiles), quantiles[..., head.median_index])


def test_training_recovers_the_mean_and_the_median_of_a_skewed_target():
    """On skewed data the mean and the median differ, and each head must find its
    own -- that is why they are separate outputs rather than one."""
    torch.manual_seed(0)
    head = MonotoneQuantileHead(1, 1).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)

    # Exponential targets: mean 1.0, median ln(2) = 0.693.
    features = torch.ones(4096, 1, device=DEVICE)
    targets = torch.distributions.Exponential(1.0).sample((4096, 1)).to(DEVICE)

    for _ in range(600):
        mean, quantiles = head(features)
        loss = head.loss(mean, quantiles, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        mean, quantiles = head(features[:1])
    assert mean.item() == pytest.approx(1.0, abs=0.15)
    assert head.median(quantiles).item() == pytest.approx(0.693, abs=0.15)
    assert mean.item() > head.median(quantiles).item()  # right-skewed, as constructed


def test_crps_from_quantiles_agrees_with_the_ensemble_estimate():
    """The harness scores CRPS from ensembles; a quantile model must land on the
    same number for the same distribution, or the leaderboard is comparing
    apples to oranges."""
    torch.manual_seed(0)
    levels = np.linspace(0.005, 0.995, 200)
    targets = torch.randn(4000)
    # Exact standard-normal quantiles, so both estimators see the same predictive
    # distribution and any disagreement is estimator bias rather than model error.
    normal = torch.distributions.Normal(0.0, 1.0)
    quantiles = normal.icdf(torch.tensor(levels, dtype=torch.float32)).expand(4000, 200)

    from_quantiles = crps_from_quantiles(quantiles, targets, levels).item()
    from_ensemble = crps_ensemble(np.random.default_rng(0).normal(size=(500, 4000)), targets.numpy())
    assert from_quantiles == pytest.approx(from_ensemble, rel=0.05)


def test_interval_returns_the_requested_level_or_refuses():
    head = MonotoneQuantileHead(8, 4)
    _mean, quantiles = head(torch.randn(3, 8))
    lower, upper = head.interval(quantiles, 0.8)
    assert (upper >= lower).all()

    narrow = MonotoneQuantileHead(8, 4, levels=(0.4, 0.5, 0.6))
    _mean, narrow_quantiles = narrow(torch.randn(3, 8))
    with pytest.raises(ValueError, match="needs quantiles at"):
        narrow.interval(narrow_quantiles, 0.9)


def test_ascending_levels_are_required():
    with pytest.raises(ValueError, match="ascending"):
        MonotoneQuantileHead(8, 4, levels=(0.5, 0.1, 0.9))
