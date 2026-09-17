"""A differentiable stand-in for the rate estimator, to be applied to samples.

Why this exists. The metric these tasks turn on is a heart or respiration rate,
which is a **nonlinear functional** of the waveform. A flow-matching model is
trained by regressing a velocity, and no amount of velocity accuracy controls a
nonlinear functional of the integrated result -- `E[f(x)] != f(E[x])`. That gap has
now been measured three separate ways in this project:

* scoring PENGUIN on the ensemble mean read 33.71 bpm; scoring per realisation read
  11.11 bpm, a 3.1x difference produced entirely by averaging before applying `f`;
* in one model with one set of weights, the rate read off the generated waveform
  was 32.36 bpm and the rate from a directly supervised head 8.41 bpm;
* waveform correlation improved while beat timing got worse -- morphology and
  timing moving in opposite directions.

Physics-based Flow Matching (ICLR 2026) names this the Jensen gap and closes it by
**unrolling the learned dynamics during training** so constraints act on the
generated sample rather than on the velocity target. Their constraints are PDEs;
ours is physiological: the generated waveform must beat at the reference rate.

That requires a rate functional that gradients can pass through, and the estimator
the metric uses cannot be one -- it is a scipy Pan-Tompkins pipeline ending in
`find_peaks`. So this module provides two differentiable surrogates that measure
periodicity *without ever estimating a rate*:

* `band_spectrum_loss` compares the two signals' power inside the rate band. No
  peak-picking, so no argmax to differentiate through.
* `envelope_autocorrelation_loss` compares the autocorrelation of the band-limited
  energy envelope over physiologically plausible lags. This is the closest
  differentiable analogue of what the real detector does: Pan-Tompkins bandpasses,
  squares, smooths, and then looks at inter-peak spacing, and the envelope's
  autocorrelation is exactly where that spacing lives.

Neither is a rate estimate, and that is deliberate. A surrogate that estimated a
rate would need a soft-argmax over frequency, whose gradient vanishes away from the
current peak -- the same failure that made a rectangular straight-through estimator
give exactly zero gradient earlier in this project. Matching a distribution over
lags has gradient everywhere.

`tests/test_functional_loss.py` pins the property that makes any of this legitimate:
the surrogate must **order pairs of signals the same way the real estimator does**.
A differentiable loss that does not track the metric would just be a second
objective with no connection to the reported number.
"""

import torch
import torch.nn.functional as F

# The bands the metric accepts, in Hz, mirroring evaluation/physio_metrics.py. Kept
# here as a mapping rather than imported so this module has no dependency on the
# evaluation package; the test asserts the two agree.
RATE_BANDS_HZ = {"ECG": (0.7, 3.0), "RESP": (0.1, 0.6)}


def _band_mask(n_freqs, sample_rate, n_samples, band):
    frequencies = torch.fft.rfftfreq(n_samples, d=1.0 / sample_rate)[:n_freqs]
    low, high = band
    return (frequencies >= low) & (frequencies <= high)


def band_spectrum_loss(prediction, target, sample_rate, target_kind="ECG"):
    """Squared difference of in-band power spectra, normalised per window.

    Compares *where the energy is* inside the rate band rather than which bin is
    largest, so there is no argmax in the path.
    """
    band = RATE_BANDS_HZ[target_kind]
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    predicted_spectrum = torch.fft.rfft(prediction, dim=-1).abs() ** 2
    target_spectrum = torch.fft.rfft(target, dim=-1).abs() ** 2
    mask = _band_mask(predicted_spectrum.shape[-1], sample_rate, prediction.shape[-1], band).to(prediction.device)
    if not mask.any():
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)

    predicted_band = predicted_spectrum[..., mask]
    target_band = target_spectrum[..., mask]
    # Normalised to unit in-band mass: the constraint is on the *shape* of the
    # spectrum, not on amplitude, which the reconstruction loss already handles.
    predicted_band = predicted_band / predicted_band.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    target_band = target_band / target_band.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return F.mse_loss(predicted_band, target_band)


def _energy_envelope(signal, sample_rate, target_kind, smooth_seconds=0.1):
    """Band-limited energy, smoothed -- Pan-Tompkins' front end, differentiably.

    The real detector bandpasses to the QRS band, squares, and box-filters. Here the
    bandpass is done in the frequency domain (no filter design, no `sosfiltfilt`,
    both non-differentiable in this context) and the box filter is a conv1d.
    """
    band = RATE_BANDS_HZ[target_kind]
    # The envelope is taken from content ABOVE the rate band for ECG -- a QRS is a
    # 10-25 Hz event whose *repetition* is the 0.7-3 Hz rate. Filtering to the rate
    # band would throw away the very thing whose spacing carries the rate. For a
    # respiration trace the oscillation itself is in band, so it is kept.
    spectrum = torch.fft.rfft(signal - signal.mean(dim=-1, keepdim=True), dim=-1)
    frequencies = torch.fft.rfftfreq(signal.shape[-1], d=1.0 / sample_rate).to(signal.device)
    if target_kind == "ECG":
        keep = (frequencies >= 5.0) & (frequencies <= min(25.0, 0.45 * sample_rate))
    else:
        keep = (frequencies >= band[0]) & (frequencies <= band[1] * 3.0)
    filtered = torch.fft.irfft(spectrum * keep.to(spectrum.dtype), n=signal.shape[-1], dim=-1)

    energy = filtered**2
    width = max(int(round(smooth_seconds * sample_rate)), 1)
    kernel = torch.ones(1, 1, width, device=signal.device, dtype=energy.dtype) / width
    flat = energy.reshape(-1, 1, energy.shape[-1])
    smoothed = F.conv1d(flat, kernel, padding=width // 2)[..., : energy.shape[-1]]
    return smoothed.reshape(energy.shape)


def envelope_autocorrelation_loss(prediction, target, sample_rate, target_kind="ECG"):
    """Match the autocorrelation of the energy envelope over plausible lags.

    Inter-beat spacing is what the reported rate is computed from, and in the
    envelope that spacing appears as the autocorrelation's structure. Comparing the
    whole normalised autocorrelation over the band's lag range keeps gradient
    everywhere, unlike a soft-argmax over lag which has gradient only near its
    current peak -- the same vanishing-gradient trap that made a rectangular
    straight-through estimator produce exactly zero gradient earlier in this project.
    """
    band = RATE_BANDS_HZ[target_kind]
    min_lag = max(int(round(sample_rate / band[1])), 1)
    max_lag = min(int(round(sample_rate / band[0])), prediction.shape[-1] - 1)
    if max_lag <= min_lag:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)

    losses = []
    for signal, other in ((prediction, target),):
        first = _normalised_autocorrelation(_energy_envelope(signal, sample_rate, target_kind), min_lag, max_lag)
        second = _normalised_autocorrelation(_energy_envelope(other, sample_rate, target_kind), min_lag, max_lag)
        losses.append(F.mse_loss(first, second))
    return sum(losses) / len(losses)


def _normalised_autocorrelation(envelope, min_lag, max_lag):
    """Autocorrelation over `[min_lag, max_lag]`, scaled to unit lag-0 power."""
    centred = envelope - envelope.mean(dim=-1, keepdim=True)
    n_samples = centred.shape[-1]
    # Via FFT, and zero-padded to avoid the circular wrap that would alias a long
    # lag onto a short one.
    spectrum = torch.fft.rfft(centred, n=2 * n_samples, dim=-1)
    correlation = torch.fft.irfft(spectrum * spectrum.conj(), n=2 * n_samples, dim=-1)
    zero_lag = correlation[..., :1].clamp_min(1e-8)
    return (correlation[..., min_lag : max_lag + 1] / zero_lag)


def rate_functional_loss(
    prediction,
    target,
    sample_rate,
    target_kind="ECG",
    spectrum_weight=1.0,
    autocorrelation_weight=1.0,
    alignment_weight=1.0,
):
    """The constraint applied to an unrolled sample: both surrogates, summed.

    All three are kept because they fail differently. The spectrum term is blind to
    phase and so cannot fix a beat train that is right in rate and wrong in
    alignment; the autocorrelation term is built from the envelope and is
    insensitive to polarity and morphology. Together they cover rate and spacing
    without either having to also be a waveform loss -- that is what the
    reconstruction term is for.
    """
    total = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    if spectrum_weight:
        total = total + spectrum_weight * band_spectrum_loss(prediction, target, sample_rate, target_kind)
    if autocorrelation_weight:
        total = total + autocorrelation_weight * envelope_autocorrelation_loss(prediction, target, sample_rate, target_kind)
    if alignment_weight:
        # Without this the other two can be satisfied at r = 0.000 -- measured.
        total = total + alignment_weight * envelope_alignment_loss(prediction, target, sample_rate, target_kind)
    return total


def envelope_alignment_loss(prediction, target, sample_rate, target_kind="ECG"):
    """Match the energy envelopes at lag ZERO -- i.e. beats must land together.

    The term the first version was missing, and its absence was measured. Both
    `band_spectrum_loss` and `envelope_autocorrelation_loss` are phase-blind by
    construction (the tests assert amplitude- and polarity-invariance), so a model
    could satisfy them completely by beating at the right rate in the wrong places.
    It did exactly that: `FlowSSMRecon-u4-w4` reached 7.89 bpm on DaLiA-ECG, better
    than PENGUIN and better than a model that draws no waveform at all, with
    waveform correlation r = 0.000 on every task. Right rate, no alignment, and
    therefore no waveform anybody could use.

    Comparing the envelopes themselves rather than their autocorrelations requires
    the beat *positions* to coincide. Kept separate from a plain waveform MSE
    because it is computed on the band-limited energy, so it asks for beats in the
    right places without also asking for the exact morphology -- that is the
    reconstruction term's job, and conflating them is what made the two objectives
    fight in the first place.
    """
    predicted = _energy_envelope(prediction, sample_rate, target_kind)
    reference = _energy_envelope(target, sample_rate, target_kind)
    # Unit-scaled per window: alignment is about where the energy is, not how much.
    predicted = predicted / predicted.mean(dim=-1, keepdim=True).clamp_min(1e-8)
    reference = reference / reference.mean(dim=-1, keepdim=True).clamp_min(1e-8)
    return F.mse_loss(predicted, reference)
