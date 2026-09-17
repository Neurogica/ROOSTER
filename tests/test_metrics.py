"""Tests for the metric implementations.

Checked against closed-form values wherever one exists, because a metric that is
subtly wrong produces a leaderboard that looks fine and ranks models incorrectly.
"""

import numpy as np
import pytest

from rooster.evaluation.leaderboard import RunRecord, latest_per_cell
from rooster.evaluation.metrics import COVERAGE_LEVELS, ForecastMetrics, crps_ensemble, crps_sum, evaluate_forecast, interval_coverage, mae, mse
from rooster.evaluation.physio_metrics import (
    HEART_RATE_BAND_HZ,
    dominant_rate_bpm,
    estimate_rate_bpm,
    evaluate_reconstruction,
    heart_rate_bpm,
    pearson_r,
    rate_error_bpm,
)


def test_point_metrics_on_a_known_case():
    predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
    targets = np.array([[1.0, 0.0], [0.0, 4.0]])
    assert mse(predictions, targets) == pytest.approx((0 + 4 + 9 + 0) / 4)
    assert mae(predictions, targets) == pytest.approx((0 + 2 + 3 + 0) / 4)


def test_single_sample_crps_equals_mae():
    """A point forecast's CRPS must degenerate to MAE, not to something flattering."""
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(1, 8, 12, 3))
    targets = rng.normal(size=(8, 12, 3))
    assert crps_ensemble(samples, targets) == pytest.approx(mae(samples[0], targets))


def test_crps_is_zero_for_a_perfect_deterministic_ensemble():
    targets = np.array([[1.0, 2.0, 3.0]])
    samples = np.repeat(targets[None, ...], 16, axis=0)
    assert crps_ensemble(samples, targets) == pytest.approx(0.0, abs=1e-12)


def test_crps_matches_the_closed_form_on_a_two_point_ensemble():
    """With samples {0, 2} and y = 0:  E|X-y| = 1,  E|X-X'| = 1  ->  CRPS = 0.5."""
    samples = np.array([[0.0], [2.0]])
    targets = np.array([0.0])
    assert crps_ensemble(samples, targets) == pytest.approx(0.5)


def test_crps_prefers_a_calibrated_ensemble_over_an_overconfident_one():
    rng = np.random.default_rng(1)
    targets = rng.normal(size=(64, 8, 2))
    calibrated = targets[None, ...] + rng.normal(scale=1.0, size=(50, *targets.shape))
    overconfident = targets[None, ...] + rng.normal(scale=1.0, size=(50, *targets.shape)) * 0.0 + 2.0
    assert crps_ensemble(calibrated, targets) < crps_ensemble(overconfident, targets)


def test_crps_sum_reacts_to_cross_variate_correlation():
    """Two models with identical marginals but different correlation must score differently."""
    rng = np.random.default_rng(2)
    targets = np.zeros((16, 4, 2))
    shared = rng.normal(size=(64, 16, 4, 1))
    correlated = np.concatenate([shared, shared], axis=-1)  # variates move together -> sums are large
    independent = rng.normal(size=(64, 16, 4, 2))  # sums cancel
    assert crps_sum(correlated, targets) != pytest.approx(crps_sum(independent, targets))


def test_crps_sum_rejects_wrongly_shaped_input():
    with pytest.raises(ValueError, match="expects"):
        crps_sum(np.zeros((4, 8, 12)), np.zeros((8, 12)))


def test_coverage_is_near_nominal_for_a_well_specified_ensemble():
    rng = np.random.default_rng(3)
    targets = rng.normal(size=(4000,))
    samples = rng.normal(size=(400, 4000))
    coverage = interval_coverage(samples, targets)
    for level in COVERAGE_LEVELS:
        assert coverage[level] == pytest.approx(level, abs=0.03)


def test_coverage_is_nan_for_a_deterministic_model():
    coverage = interval_coverage(np.zeros((1, 10)), np.zeros((10,)))
    assert all(np.isnan(value) for value in coverage.values())


def test_evaluate_forecast_uses_the_ensemble_mean_not_best_of_n():
    """One perfect sample among bad ones must NOT produce a perfect score."""
    targets = np.zeros((2, 3, 1))
    samples = np.stack([np.zeros((2, 3, 1)), np.full((2, 3, 1), 10.0)])
    metrics = evaluate_forecast(samples, targets)
    assert metrics["mse"] == pytest.approx(25.0)  # mean of 0 and 10 is 5


def test_evaluate_forecast_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="does not match"):
        evaluate_forecast(np.zeros((3, 4, 5, 2)), np.zeros((4, 5, 3)))


def test_dominant_rate_recovers_a_synthetic_sine():
    fs, seconds, rate_hz = 125.0, 30.0, 1.2  # 72 bpm
    t = np.arange(int(fs * seconds)) / fs
    signal = np.sin(2 * np.pi * rate_hz * t)
    assert dominant_rate_bpm(signal, fs, (0.7, 3.0)) == pytest.approx(rate_hz * 60.0, abs=2.0)


def test_dominant_rate_returns_nan_when_the_window_is_too_short():
    fs = 62.5
    short = np.sin(np.arange(int(fs * 2)) / fs)  # 2 s cannot resolve a 0.1 Hz band
    assert np.isnan(dominant_rate_bpm(short, fs, (0.1, 0.6)))


def test_dominant_rate_returns_nan_on_a_flat_window():
    assert np.isnan(dominant_rate_bpm(np.ones(4000), 125.0, (0.7, 3.0)))


def test_rate_error_is_zero_when_prediction_equals_target():
    fs, t = 125.0, np.arange(3750) / 125.0
    waveforms = np.stack([np.sin(2 * np.pi * rate * t) for rate in (1.0, 1.5, 2.0)])
    result = rate_error_bpm(waveforms, waveforms, fs, "ECG")
    assert result["rate_mae_bpm"] == pytest.approx(0.0, abs=1e-9)
    assert result["rate_unresolved_frac"] == 0.0


def test_unresolved_windows_are_counted_not_silently_dropped():
    fs = 125.0
    flat = np.ones((4, 3750))
    result = rate_error_bpm(flat, flat, fs, "ECG")
    assert result["rate_unresolved_frac"] == 1.0
    assert np.isnan(result["rate_mae_bpm"])


def test_pearson_r_is_one_for_a_scaled_copy():
    rng = np.random.default_rng(4)
    targets = rng.normal(size=(8, 256))
    assert pearson_r(targets * 3.0 + 1.0, targets) == pytest.approx(1.0, abs=1e-9)


def test_evaluate_reconstruction_returns_every_expected_key():
    fs, t = 125.0, np.arange(1000) / 125.0
    targets = np.stack([np.sin(2 * np.pi * 1.2 * t)] * 4)
    samples = targets[None, ...]
    metrics = evaluate_reconstruction(samples, targets, fs, "ECG")
    assert set(metrics) >= {"rmse", "mae", "pearson_r", "rate_mae_bpm", "rate_unresolved_frac", "n_samples"}
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_latest_record_wins_for_a_repeated_cell():
    older = RunRecord("forecast", "ETTm2", "DLinear", 96, 0, {"mse": 1.0}, timestamp="2026-01-01T00:00:00+00:00")
    newer = RunRecord("forecast", "ETTm2", "DLinear", 96, 0, {"mse": 0.5}, timestamp="2026-02-01T00:00:00+00:00")
    kept = latest_per_cell([older, newer])
    assert len(kept) == 1
    assert kept[0].metrics["mse"] == 0.5


def test_distinct_cells_are_all_kept():
    records = [
        RunRecord("forecast", "ETTm2", "DLinear", 96, 0, {"mse": 1.0}),
        RunRecord("forecast", "ETTm2", "DLinear", 192, 0, {"mse": 1.0}),
        RunRecord("forecast", "ETTm2", "RepeatLast", 96, 0, {"mse": 1.0}),
        RunRecord("forecast", "Weather", "DLinear", 96, 0, {"mse": 1.0}),
        RunRecord("forecast", "ETTm2", "DLinear", 96, 1, {"mse": 1.0}),
    ]
    assert len(latest_per_cell(records)) == 5


def test_streaming_accumulator_matches_whole_array_computation():
    """The streaming path is a memory refactor, so it must be EXACTLY equal."""
    rng = np.random.default_rng(5)
    samples = rng.normal(size=(6, 40, 12, 3))
    targets = rng.normal(size=(40, 12, 3))

    whole = evaluate_forecast(samples, targets)

    accumulator = ForecastMetrics()
    for lo in range(0, 40, 7):  # deliberately ragged batches
        accumulator.update(samples[:, lo : lo + 7], targets[lo : lo + 7])
    streamed = accumulator.compute()

    # Coverage is exact too: quantiles are taken over the ensemble axis, which is
    # per-scalar, so batching cannot change them.
    for key, value in whole.items():
        assert streamed[key] == pytest.approx(value, rel=1e-12), key


def test_accumulator_rejects_a_changing_ensemble_size():
    accumulator = ForecastMetrics()
    accumulator.update(np.zeros((4, 2, 3, 1)), np.zeros((2, 3, 1)))
    with pytest.raises(ValueError, match="ensemble size changed"):
        accumulator.update(np.zeros((5, 2, 3, 1)), np.zeros((2, 3, 1)))


def test_accumulator_refuses_to_compute_before_any_update():
    with pytest.raises(RuntimeError, match="before any update"):
        ForecastMetrics().compute()


def test_heart_rate_from_a_synthetic_ecg_like_train_of_qrs_spikes():
    """R-peak detection must recover the true rate from spike-train-like input."""
    fs, seconds, true_bpm = 125.0, 10.0, 72.0
    n = int(fs * seconds)
    signal = np.zeros(n)
    period = int(round(fs * 60.0 / true_bpm))
    for peak in range(period // 2, n, period):
        # A narrow biphasic deflection, the way a QRS complex actually looks.
        signal[peak] = 3.0
        signal[max(peak - 2, 0)] = -1.0
        signal[min(peak + 2, n - 1)] = -1.0
    assert heart_rate_bpm(signal, fs) == pytest.approx(true_bpm, abs=1.0)


def test_spectral_estimator_would_have_failed_on_that_ecg():
    """Why the ECG path is peak-based: QRS energy sits far above the HR band.

    This is the bug the leaderboard's first run exposed -- guarding it so nobody
    'simplifies' the two estimators back into one.
    """
    fs, seconds, true_bpm = 125.0, 10.0, 72.0
    n = int(fs * seconds)
    signal = np.zeros(n)
    period = int(round(fs * 60.0 / true_bpm))
    for peak in range(period // 2, n, period):
        signal[peak] = 3.0
        signal[max(peak - 2, 0)] = -1.0
        signal[min(peak + 2, n - 1)] = -1.0
    signal += 0.5 * np.sin(2 * np.pi * 0.9 * np.arange(n) / fs)  # baseline drift inside the HR band

    spectral = dominant_rate_bpm(signal, fs, HEART_RATE_BAND_HZ)
    assert abs(spectral - true_bpm) > 5.0  # the drift wins
    assert heart_rate_bpm(signal, fs) == pytest.approx(true_bpm, abs=1.0)


def test_heart_rate_is_nan_when_no_beats_are_detectable():
    assert np.isnan(heart_rate_bpm(np.zeros(1250), 125.0))


def test_estimate_rate_dispatches_on_target_kind():
    fs, t = 62.5, np.arange(int(62.5 * 60)) / 62.5
    breathing = np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min
    assert estimate_rate_bpm(breathing, fs, "RESP") == pytest.approx(15.0, abs=1.0)
    with pytest.raises(ValueError, match="unknown target kind"):
        estimate_rate_bpm(breathing, fs, "BP")


def test_budget_label_groups_deterministic_and_generative_models_together():
    """The label is the run's configuration, not a per-model detail.

    Keying on the effective sample count would file DLinear (always 1 sample) and
    FlowSSM (20 samples) under different budgets, so the gate and the ranking
    would never see them as comparable.
    """
    from rooster.evaluation.leaderboard import budget_label

    requested = 20
    assert budget_label(20000, requested, 4096) == budget_label(20000, requested, 4096)
    assert "samples=20" in budget_label(20000, requested, 4096)
    assert budget_label(20000, 20, None).endswith("testwin=all")


def _forecast_cell(model, dataset, mse, mae=None):
    from rooster.evaluation.leaderboard import RunRecord

    return RunRecord("forecast", dataset, model, 96, 0, {"mse": mse, "mae": mae if mae is not None else mse})


def test_promotion_rule_requires_beating_every_baseline():
    from rooster.evaluation.promotion import winning_candidates

    records = []
    for dataset in ["A", "B", "C"]:
        records.append(_forecast_cell("DLinear", dataset, 1.0))
        records.append(_forecast_cell("DecompSSM", dataset, 0.5))
        records.append(_forecast_cell("Cand-good", dataset, 0.4))  # beats both everywhere
        records.append(_forecast_cell("Cand-partial", dataset, 0.7))  # beats DLinear only

    assert winning_candidates(records, "forecast") == ["Cand-good"]


def test_promotion_rule_needs_a_majority_not_a_single_win():
    from rooster.evaluation.promotion import winning_candidates

    records = []
    for dataset, candidate_mse in [("A", 0.4), ("B", 2.0), ("C", 2.0)]:
        records.append(_forecast_cell("DLinear", dataset, 1.0))
        records.append(_forecast_cell("Cand", dataset, candidate_mse))
    assert winning_candidates(records, "forecast") == []


def test_promotion_rule_does_not_count_an_absent_comparison_as_a_win():
    from rooster.evaluation.promotion import winning_candidates

    records = [_forecast_cell("DLinear", "A", 1.0), _forecast_cell("Cand", "B", 0.1)]  # no shared cell
    assert winning_candidates(records, "forecast", min_shared_cells=1) == []


def test_winning_mse_while_losing_mae_is_not_a_win():
    """Both go in the paper, so both have to be won."""
    from rooster.evaluation.promotion import winning_candidates

    records = []
    for dataset in ["A", "B", "C"]:
        records.append(_forecast_cell("DecompSSM", dataset, mse=1.0, mae=1.0))
        records.append(_forecast_cell("Cand-mse-only", dataset, mse=0.5, mae=1.5))
        records.append(_forecast_cell("Cand-both", dataset, mse=0.5, mae=0.9))
    assert winning_candidates(records, "forecast") == ["Cand-both"]


def test_the_required_forecast_metrics_are_mse_and_mae():
    """Pinned deliberately: these are the numbers the paper reports, so a silent
    change of gating metric would change what the project is optimising."""
    from rooster.evaluation.promotion import REQUIRED_METRICS

    assert REQUIRED_METRICS["forecast"] == ("mse", "mae")


def test_both_readouts_are_recorded_and_the_mean_wins_on_squared_error():
    """The forecasting counterpart of the physio rate-readout comparison.

    A model with a point head is scored on that head, but the ensemble mean is
    recorded too, because a claim about readouts needs both numbers side by side.

    The assertion also pins the reason this axis is *not* where a large gap should
    appear: the ensemble mean is the minimiser of squared error, so a point head
    can at best tie it on MSE for the same predictive distribution. That is what
    bounds the project's claim to readouts that are nonlinear functionals of the
    generated output -- a heart rate read off a waveform is one, an MSE-optimal
    point forecast is not.
    """
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(64, 32, 8))
    targets = rng.normal(size=(32, 8))

    accumulator = ForecastMetrics()
    # A deliberately poor point head: the mean shifted away from the optimum.
    accumulator.update(samples, targets, point=samples.mean(axis=0) + 0.5)
    metrics = accumulator.compute()

    assert metrics["mse"] > metrics["mse_of_mean"]
    np.testing.assert_allclose(metrics["mse_of_mean"], ((samples.mean(axis=0) - targets) ** 2).mean())

    # And with no point head the two readouts must coincide exactly, or every
    # deterministic row in the leaderboard would carry a spurious difference.
    plain = ForecastMetrics()
    plain.update(samples, targets)
    plain_metrics = plain.compute()
    assert plain_metrics["mse"] == plain_metrics["mse_of_mean"]
    assert plain_metrics["mae"] == plain_metrics["mae_of_mean"]
