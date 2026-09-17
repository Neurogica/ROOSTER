"""Metrics for PPG-to-vital-sign reconstruction.

Two tiers, and the second is the one that matters:

1. **Waveform fidelity** -- RMSE / MAE / Pearson r on the z-scored waveform.
   Easy to compute, easy to game, and weakly correlated with clinical utility.
2. **Physiological accuracy** -- heart rate error (bpm) for PPG->ECG and
   respiratory rate error (bpm) for PPG->respiration. This is what PENGUIN,
   RDDM, RespDiff and CardioGAN actually report, so it is what our numbers have
   to be expressed in to be comparable at all.

Rate estimation is **per target kind**, because one estimator does not fit both:

* **Respiration** is a smooth, quasi-sinusoidal ~0.1-0.6 Hz process, so the
  dominant frequency of the periodogram inside that band is the right estimate
  and degrades gracefully on a noisy reconstruction.
* **ECG is not.** Its spectral energy is dominated by the QRS complex at
  roughly 10-25 Hz, so the largest bin inside a 0.7-3.0 Hz "heart rate band" is
  usually *not* the heart rate -- it is whatever low-frequency baseline drift
  happens to be present. Using the spectral estimator here produced 28-41 bpm
  errors that measured the estimator, not the model. Heart rate is therefore
  taken from **R-peak intervals**, detected with a Pan-Tompkins-style front end
  (bandpass -> square -> integrate), which is what the PPG->ECG literature uses.

The same estimator is applied to prediction and ground truth, so any residual
bias cancels.
"""

import functools

import numpy as np
import scipy.signal

# Physiologically plausible bands. Outside these, a "detected" rate is an artifact.
HEART_RATE_BAND_HZ = (0.7, 3.0)  # 42-180 bpm
RESPIRATORY_RATE_BAND_HZ = (0.1, 0.6)  # 6-36 breaths/min

# QRS energy band for the Pan-Tompkins front end, and the refractory period that
# stops one QRS complex being counted twice (200 bpm is above any plausible rate).
QRS_BAND_HZ = (5.0, 15.0)
MIN_RR_SECONDS = 0.3
MIN_BEATS_FOR_RATE = 3

_BANDS = {"ECG": HEART_RATE_BAND_HZ, "RESP": RESPIRATORY_RATE_BAND_HZ}


def dominant_rate_bpm(signal, fs, band):
    """Dominant frequency inside `band`, in cycles per minute.

    Returns NaN when the window is too short to resolve the band, rather than
    silently reporting the nearest resolvable bin -- a wrong number is worse
    than a missing one in an aggregate.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"dominant_rate_bpm expects a 1-D window; got shape {signal.shape}")

    low, high = band
    # Need at least one full cycle of the slowest frequency in the band.
    if len(signal) < fs / low:
        return float("nan")

    detrended = signal - signal.mean()
    if detrended.std() < 1e-8:
        return float("nan")

    # nperseg = the whole window: these are single short windows, so Welch
    # averaging would cost more frequency resolution than it buys in variance.
    freqs, power = scipy.signal.periodogram(detrended, fs=fs, window="hann")
    in_band = (freqs >= low) & (freqs <= high)
    if not in_band.any():
        return float("nan")
    return float(freqs[in_band][np.argmax(power[in_band])] * 60.0)


@functools.lru_cache(maxsize=32)
def _qrs_bandpass(low, high):
    """Cached filter design.

    The coefficients depend only on the two normalised band edges, and those take
    a handful of distinct values across the whole project -- one per sample rate.
    Recomputing them per window cost half the estimator's runtime (profiled: 0.334
    s of 0.662 s over 500 calls), and this is called once per training window per
    step by any model supervised on a rate. Same numerics, twice the speed.
    """
    return scipy.signal.butter(2, [low, high], btype="bandpass", output="sos")


def heart_rate_bpm(signal, fs):
    """Heart rate from R-peak intervals, via a Pan-Tompkins-style front end.

    Bandpass to the QRS band, square to make every deflection positive and
    emphasise sharp complexes, smooth over ~100 ms, then pick peaks separated by
    at least a refractory period. The rate is the **median** inter-beat interval,
    not the mean, so a single missed or doubled detection cannot drag it far.

    Returns NaN if fewer than `MIN_BEATS_FOR_RATE` beats are found -- a
    reconstruction too degraded to beat-detect should be reported as
    unresolvable, not scored with a fabricated rate.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"heart_rate_bpm expects a 1-D window; got shape {signal.shape}")

    nyquist = fs / 2.0
    high = min(QRS_BAND_HZ[1], nyquist * 0.95)
    if QRS_BAND_HZ[0] >= high or signal.std() < 1e-8:
        return float("nan")

    coefficients = _qrs_bandpass(QRS_BAND_HZ[0] / nyquist, high / nyquist)
    filtered = scipy.signal.sosfiltfilt(coefficients, signal - signal.mean())
    energy = filtered**2
    window = max(int(round(0.1 * fs)), 1)
    integrated = np.convolve(energy, np.ones(window) / window, mode="same")
    if integrated.max() < 1e-12:
        return float("nan")

    peaks, _ = scipy.signal.find_peaks(
        integrated,
        distance=max(int(round(MIN_RR_SECONDS * fs)), 1),
        height=0.25 * np.percentile(integrated, 99),
    )
    if len(peaks) < MIN_BEATS_FOR_RATE:
        return float("nan")

    median_interval = float(np.median(np.diff(peaks))) / fs
    if median_interval <= 0:
        return float("nan")
    rate = 60.0 / median_interval
    return rate if HEART_RATE_BAND_HZ[0] * 60 <= rate <= HEART_RATE_BAND_HZ[1] * 60 else float("nan")


def hamilton_heart_rate_bpm(signal, fs):
    """Heart rate via a Hamilton-style QRS detector, for the PENGUIN protocol.

    PENGUIN (arXiv:2602.03858) reports HR Error using the Hamilton method over
    8-second windows, so when our numbers share a table with theirs the rate has
    to come from the same family of detector. The Hamilton front end differs
    from the Pan-Tompkins-style one above in two ways that matter on
    reconstructions: the envelope is the rectified *derivative* (not the squared
    signal), and the peak threshold adapts to running estimates of signal and
    noise peak heights rather than being a fixed percentile.

    Same contract as `heart_rate_bpm`: NaN when fewer than MIN_BEATS_FOR_RATE
    beats are found or the median interval implies a rate outside the plausible
    band. Applied to prediction and ground truth alike, so estimator bias
    cancels in the error.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"hamilton_heart_rate_bpm expects a 1-D window; got shape {signal.shape}")

    nyquist = fs / 2.0
    low, high = 8.0, min(16.0, nyquist * 0.95)
    if low >= high or signal.std() < 1e-8:
        return float("nan")

    coefficients = _qrs_bandpass(low / nyquist, high / nyquist)
    filtered = scipy.signal.sosfiltfilt(coefficients, signal - signal.mean())
    envelope = np.abs(np.diff(filtered, prepend=filtered[:1]))
    window = max(int(round(0.08 * fs)), 1)
    envelope = np.convolve(envelope, np.ones(window) / window, mode="same")
    if envelope.max() < 1e-12:
        return float("nan")

    refractory = max(int(round(0.2 * fs)), 1)
    candidates, _ = scipy.signal.find_peaks(envelope, distance=refractory)
    if len(candidates) == 0:
        return float("nan")

    # Adaptive threshold: running averages of accepted QRS peak heights and of
    # rejected (noise) peak heights; a candidate is a beat when it clears
    # noise + 0.3125 * (signal - noise), the Hamilton coefficient.
    signal_level = float(np.percentile(envelope[candidates], 90))
    noise_level = float(np.percentile(envelope[candidates], 10))
    beats = []
    for index in candidates:
        height = envelope[index]
        threshold = noise_level + 0.3125 * (signal_level - noise_level)
        if height >= threshold:
            beats.append(index)
            signal_level = 0.85 * signal_level + 0.15 * height
        else:
            noise_level = 0.85 * noise_level + 0.15 * height
    if len(beats) < MIN_BEATS_FOR_RATE:
        return float("nan")

    median_interval = float(np.median(np.diff(beats))) / fs
    if median_interval <= 0:
        return float("nan")
    rate = 60.0 / median_interval
    return rate if HEART_RATE_BAND_HZ[0] * 60 <= rate <= HEART_RATE_BAND_HZ[1] * 60 else float("nan")


def rate_band_bpm(target_kind):
    """The rate range the metric will accept for this target, in bpm.

    Exported so a model with a bounded rate head reads its output range from the
    same place the metric reads its acceptance range. They diverged once: the head
    was hard-coded to the heart-rate band (30-220 bpm) and used on respiration,
    where the truth is 6-36 breaths/min. The head saturated at its own floor and
    emitted a constant -- 12.50 +/- 0.00 bpm of error on BIDMC-RESP against 1.70
    for the waveform path, which read as "the head does not work" rather than
    "the head cannot express the answer".
    """
    if target_kind not in _BANDS:
        raise ValueError(f"unknown target kind {target_kind!r}; expected one of {sorted(_BANDS)}")
    low, high = _BANDS[target_kind]
    return low * 60.0, high * 60.0


def estimate_rate_bpm(signal, fs, target_kind):
    """Dispatch to the estimator appropriate for the target signal."""
    if target_kind == "ECG":
        return heart_rate_bpm(signal, fs)
    if target_kind == "RESP":
        return dominant_rate_bpm(signal, fs, RESPIRATORY_RATE_BAND_HZ)
    raise ValueError(f"unknown target kind {target_kind!r}; expected one of {sorted(_BANDS)}")


def rate_error_bpm(predictions, targets, fs, target_kind):
    """Mean absolute rate error between predicted and true waveforms, in bpm.

    `predictions` and `targets` are `(batch, n_samples)`. Windows where either
    signal yields no resolvable rate are excluded and counted separately, so a
    model cannot improve its score by producing unreadable output.
    """
    if target_kind not in _BANDS:
        raise ValueError(f"unknown target kind {target_kind!r}; expected one of {sorted(_BANDS)}")

    errors = []
    n_unresolved = 0
    for prediction, target in zip(np.asarray(predictions), np.asarray(targets), strict=True):
        predicted_rate = estimate_rate_bpm(prediction, fs, target_kind)
        true_rate = estimate_rate_bpm(target, fs, target_kind)
        if np.isnan(predicted_rate) or np.isnan(true_rate):
            n_unresolved += 1
            continue
        errors.append(abs(predicted_rate - true_rate))

    return {
        "rate_mae_bpm": float(np.mean(errors)) if errors else float("nan"),
        "rate_unresolved_frac": n_unresolved / max(len(np.asarray(targets)), 1),
    }


def pearson_r(predictions, targets):
    """Mean per-window Pearson correlation.

    Computed per window and then averaged, not pooled: pooling across windows
    lets a model score well purely by matching the between-window variance,
    which is exactly the thing z-scoring already removed.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    predictions = predictions - predictions.mean(axis=-1, keepdims=True)
    targets = targets - targets.mean(axis=-1, keepdims=True)
    numerator = (predictions * targets).sum(axis=-1)
    denominator = np.sqrt((predictions**2).sum(axis=-1) * (targets**2).sum(axis=-1))
    valid = denominator > 1e-12
    if not valid.any():
        return float("nan")
    return float(np.mean(numerator[valid] / denominator[valid]))


def evaluate_reconstruction(samples, targets, fs, target_kind):
    """All reconstruction metrics for one (model, task) cell.

    `samples`: `(n_samples, batch, window)`; `targets`: `(batch, window)`.
    The point estimate is the ensemble mean, matching `evaluate_forecast`.
    """
    samples = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if samples.ndim != 3 or targets.ndim != 2:
        raise ValueError(f"expected samples (S,B,T) and targets (B,T); got {samples.shape} and {targets.shape}")
    if samples.shape[1:] != targets.shape:
        raise ValueError(f"sample shape {samples.shape[1:]} does not match target shape {targets.shape}")

    point = samples.mean(axis=0)
    metrics = {
        "rmse": float(np.sqrt(np.mean((point - targets) ** 2))),
        "mae": float(np.mean(np.abs(point - targets))),
        "n_samples": int(samples.shape[0]),
    }

    # Squared error is minimised by the ensemble mean, so RMSE and MAE are scored
    # on it. Rate and waveform correlation are NOT: they are properties of a
    # realisation, and averaging destroys them whenever the samples agree on the
    # beat rate but disagree on the beat *phase* -- the average of correctly-timed
    # but differently-phased beat trains is flat.
    #
    # This is not hypothetical. Scoring the mean, PENGUIN measured r = +0.008 and
    # 33.7 bpm on DaLiA-ECG with only 0.3% of windows unresolved, i.e. a smooth
    # signal with no cardiac content -- against the 15.64 bpm its own paper
    # reports. The mean was being scored, not the model.
    #
    # So sampled quantities are computed per realisation and then averaged over
    # realisations. For a deterministic model there is one realisation and this
    # reduces exactly to the old behaviour.
    per_sample_r = [pearson_r(sample, targets) for sample in samples]
    metrics["pearson_r"] = float(np.nanmean(per_sample_r))
    metrics["pearson_r_of_mean"] = pearson_r(point, targets)

    per_sample_rate = [rate_error_bpm(sample, targets, fs, target_kind) for sample in samples]
    metrics["rate_mae_bpm"] = float(np.nanmean([m["rate_mae_bpm"] for m in per_sample_rate]))

    # Medoid readout: per window, the ONE sample whose rate sits closest to the
    # ensemble's median rate. Selecting a realisation by self-consistency is more
    # robust than averaging rates across realisations -- measured on trained
    # checkpoints it cut BIDMC-ECG from 1.58 to 1.27 bpm and CapnoBase-ECG from
    # 2.25 to 1.96 at zero training cost. Computed for every probabilistic model,
    # PENGUIN included, so it is a readout everyone gets rather than an advantage
    # we quietly give ourselves.
    def medoid_error(estimator):
        errors = []
        for j in range(targets.shape[0]):
            true_rate = estimator(targets[j], fs)
            if not np.isfinite(true_rate):
                continue
            rates = [estimator(samples[k, j], fs) for k in range(samples.shape[0])]
            rates = [r for r in rates if np.isfinite(r)]
            if not rates:
                continue
            median = np.median(rates)
            errors.append(abs(rates[int(np.argmin([abs(r - median) for r in rates]))] - true_rate))
        return float(np.mean(errors)) if errors else float("nan")

    metrics["rate_mae_bpm_medoid"] = medoid_error(lambda s, f: estimate_rate_bpm(s, f, target_kind))
    # The PENGUIN-protocol read-out, computed alongside rather than instead: HR
    # from the Hamilton-style detector (their reported metric family). For RESP
    # the dominant-frequency estimator above already IS the protocol, so no
    # second number exists to report.
    if target_kind == "ECG":
        metrics["rate_mae_bpm_medoid_hamilton"] = medoid_error(hamilton_heart_rate_bpm)
    metrics["rate_unresolved_frac"] = float(np.mean([m["rate_unresolved_frac"] for m in per_sample_rate]))
    # Kept so the size of the averaging artefact stays visible in the table.
    mean_rate = rate_error_bpm(point, targets, fs, target_kind)
    metrics["rate_mae_bpm_of_mean"] = mean_rate["rate_mae_bpm"]
    metrics["rate_unresolved_frac_of_mean"] = mean_rate["rate_unresolved_frac"]
    return metrics
