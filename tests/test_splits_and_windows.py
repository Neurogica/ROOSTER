"""Tests for the split/scaling contract.

These are the tests that matter most in the whole repo: a silent bug here makes
every leaderboard number wrong *and* plausible-looking. So they check the
properties that would actually be violated by a real mistake -- leakage across
the split boundary, a scaler fit on the wrong rows, windows running off the end
-- rather than just exercising the code path.
"""

import numpy as np
import pytest

from rooster.datasets.forecasting import FORECAST_SPECS, available_datasets, load_forecast_series, resolve_spec
from rooster.datasets.splits import StandardScaler, ratio_boundaries, windowed_bounds
from rooster.datasets.windows import build_forecast_windows


class _FakeSeries:
    """A ramp plus an offset per variate: any leak or mis-scale is visible."""

    def __init__(self, n_timesteps=1000, n_variates=3):
        base = np.arange(n_timesteps, dtype=np.float32)[:, None]
        self.values = base + np.arange(n_variates, dtype=np.float32)[None, :] * 100.0
        self.boundaries = ratio_boundaries(n_timesteps, 0.7, 0.1)
        self.spec = resolve_spec("ETTm2")

    @property
    def n_variates(self):
        return self.values.shape[1]


def test_ratio_boundaries_are_contiguous_and_ordered():
    boundaries = ratio_boundaries(1000, 0.7, 0.1)
    assert (boundaries.train_end, boundaries.val_end, boundaries.test_end) == (700, 800, 1000)
    assert boundaries.bounds("train") == (0, 700)
    assert boundaries.bounds("val") == (700, 800)
    assert boundaries.bounds("test") == (800, 1000)


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="unknown split"):
        ratio_boundaries(100, 0.7, 0.1).bounds("validation")


def test_val_and_test_are_extended_backwards_by_exactly_one_lookback():
    """The first prediction of each split must start at the split boundary."""
    boundaries = ratio_boundaries(1000, 0.7, 0.1)
    assert windowed_bounds(boundaries, "train", 96) == (0, 700)
    assert windowed_bounds(boundaries, "val", 96) == (700 - 96, 800)
    assert windowed_bounds(boundaries, "test", 96) == (800 - 96, 1000)


def test_scaler_is_fit_on_train_rows_only():
    series = _FakeSeries(n_timesteps=1000)
    _splits, scaler = build_forecast_windows(series, lookback=96, horizon=96)

    expected = StandardScaler().fit(series.values[:700])
    np.testing.assert_allclose(scaler.mean, expected.mean, rtol=1e-9)
    np.testing.assert_allclose(scaler.std, expected.std, rtol=1e-9)
    # A scaler fit on the whole series would have a visibly larger mean, since
    # the fixture ramps upward -- this is the actual failure mode being guarded.
    assert np.all(scaler.mean < series.values.mean(axis=0))


def test_scaler_round_trips():
    values = np.random.default_rng(0).normal(size=(500, 4)) * 7.0 + 3.0
    scaler = StandardScaler().fit(values)
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(values)), values, rtol=1e-9, atol=1e-9)


def test_scaler_survives_a_constant_variate():
    values = np.concatenate([np.random.default_rng(0).normal(size=(100, 1)), np.full((100, 1), 5.0)], axis=1)
    transformed = StandardScaler().fit_transform(values)
    assert np.isfinite(transformed).all()


def test_scaler_refuses_to_transform_before_fit():
    with pytest.raises(RuntimeError, match="before fit"):
        StandardScaler().transform(np.zeros((3, 2)))


def test_window_shapes_and_contiguity():
    series = _FakeSeries(n_timesteps=1000, n_variates=3)
    lookback, horizon = 96, 48
    splits, _scaler = build_forecast_windows(series, lookback, horizon)

    x, y = splits["train"][0]
    assert x.shape == (lookback, 3)
    assert y.shape == (horizon, 3)
    # y must be exactly the block following x, with no gap and no overlap.
    # Windows are stored as float32, so the tolerance is set by float32 epsilon
    # on the scaled magnitudes (~1e-7 absolute), not by anything about the split.
    step = float(x[1, 0] - x[0, 0])
    assert float(y[0, 0] - x[-1, 0]) == pytest.approx(step, abs=1e-5)


def test_last_window_stays_in_bounds():
    series = _FakeSeries(n_timesteps=1000)
    splits, _scaler = build_forecast_windows(series, lookback=96, horizon=48)
    for split in ("train", "val", "test"):
        dataset = splits[split]
        x, y = dataset[len(dataset) - 1]
        assert x.shape[0] == 96 and y.shape[0] == 48
        assert np.isfinite(x.numpy()).all() and np.isfinite(y.numpy()).all()


def test_train_windows_never_reach_into_the_validation_split():
    """Train windows must stay strictly inside the training rows."""
    series = _FakeSeries(n_timesteps=1000)
    lookback, horizon = 96, 48
    splits, scaler = build_forecast_windows(series, lookback, horizon)
    train = splits["train"]

    _x, y = train[len(train) - 1]
    # Variate 0 is the raw ramp, so its value *is* its row index. Rounding
    # absorbs the float32 store/scale/unscale round-trip (~1e-5 here) -- the
    # property under test is which row was touched, not its bit pattern.
    last_row = round(float(scaler.inverse_transform(y.numpy())[-1, 0]))
    assert last_row == series.boundaries.train_end - 1


def test_too_short_split_raises_rather_than_yielding_nothing():
    series = _FakeSeries(n_timesteps=200)
    with pytest.raises(ValueError, match="too few for lookback"):
        build_forecast_windows(series, lookback=96, horizon=720)


@pytest.mark.parametrize("name", sorted(FORECAST_SPECS))
def test_every_spec_declares_a_usable_horizon_set(name):
    spec = FORECAST_SPECS[name]
    assert spec.horizons and all(h > 0 for h in spec.horizons)
    assert spec.lookback > 0


def test_ecl_alias_resolves():
    assert resolve_spec("ECL").name == "Electricity"


def test_unknown_dataset_name_is_rejected():
    with pytest.raises(KeyError, match="unknown forecasting dataset"):
        resolve_spec("NotADataset")


@pytest.mark.skipif("ETTm2" not in available_datasets(), reason="ETTm2 not present; see docs/03_datasets.md")
def test_ettm2_loads_with_the_published_shape_and_split():
    series = load_forecast_series("ETTm2")
    assert series.values.shape == (69680, 7)
    # The Informer convention: 12/4/4 months of 15-minute data.
    assert series.boundaries.train_end == 34560
    assert series.boundaries.val_end == 46080
    assert np.isfinite(series.values).all()
