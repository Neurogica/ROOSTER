"""Builds `UnifiedTokenizer`-format batch dicts for the Flow-SSM variants.

This is the bridge between the plain arrays produced by `forecasting.py` /
`physio.py` and the batch schema `tokenizer.UnifiedTokenizer` consumes:

    value          (B, N, P)  raw samples, patched
    observed_mask  (B, N, P)  1 where a sample is real
    t              (B, N)     continuous timestamp per patch
    dt             (B, N)     sampling interval per patch, in seconds
    channel_id     (B, N)     long
    modality_id    (B, N)     long
    task_id        (B, N)     long

**Time convention.** `t` is measured in units of the *condition window's
duration*, counting from the condition window's start. So the condition always
spans `t in [0, 1)`, and the target then encodes the task geometry directly:

* forecasting   -> target `t in [1, 1 + H/L)`, i.e. strictly after the condition;
* reconstruction -> target `t in [0, 1)`, i.e. aligned with the condition.

This is what the project's design calls "relative-time encoding identifying
whether the target is in the future or aligned with the input". Using raw
seconds instead would put ETTm2 (86 400 s per window) and ECG (8 s per window)
many orders of magnitude apart, which `ContinuousTimeEncoding`'s learnable
frequencies -- initialised near 1.0 -- cannot represent at both scales. `dt`
still carries the physical sampling interval, so the model retains the
information that one is 15-minute data and the other is 125 Hz.

There is no synthetic-signal fallback here on purpose. The repo previously used
a hand-built PPG/ECG proxy because no real physiological data was present; BIDMC
and CapnoBase are now on disk (docs/03_datasets.md), so a proxy would only
invite mistaking it for a result.
"""

import numpy as np
import torch

from rooster.datasets.forecasting import load_forecast_series
from rooster.datasets.physio import load_recordings, subject_split

TASK_FORECAST = 0
TASK_RECONSTRUCT = 1

MODALITY_TIMESERIES = 0
MODALITY_BIOSIGNAL = 1


def _to_patches(windows, patch_len):
    """`(B, T)` -> `(B, T // patch_len, patch_len)`, dropping any ragged tail."""
    windows = np.asarray(windows, dtype=np.float32)
    n_patches = windows.shape[1] // patch_len
    if n_patches == 0:
        raise ValueError(f"window of {windows.shape[1]} samples is shorter than one patch of {patch_len}")
    return windows[:, : n_patches * patch_len].reshape(windows.shape[0], n_patches, patch_len)


def build_batch(windows, patch_len, t_start, t_step, dt_seconds, channel_id, modality_id, task_id, device):
    """Assemble one tokenizer batch dict from `(B, T)` raw windows.

    `t_start` is the first patch's timestamp and `t_step` the spacing between
    consecutive patches, both in condition-window units (see module docstring).
    """
    patches = _to_patches(windows, patch_len)
    batch_size, n_patches, _ = patches.shape
    value = torch.from_numpy(patches).to(device)
    times = t_start + t_step * torch.arange(n_patches, dtype=torch.float32, device=device)

    def full(fill, dtype):
        return torch.full((batch_size, n_patches), fill, dtype=dtype, device=device)

    return {
        "value": value,
        "observed_mask": torch.ones_like(value),
        "t": times.unsqueeze(0).expand(batch_size, -1).contiguous(),
        "dt": full(float(dt_seconds), torch.float32),
        "channel_id": full(int(channel_id), torch.long),
        "modality_id": full(int(modality_id), torch.long),
        "task_id": full(int(task_id), torch.long),
    }


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------

# Seconds per step for each dataset's sampling frequency, so `dt` is physical.
FREQ_SECONDS = {"5min": 300.0, "10min": 600.0, "15min": 900.0, "1h": 3600.0, "1d": 86400.0, "7d": 604800.0}


def load_forecast_channel(dataset_name="ETTm2", variate=-1, root=None):
    """One variate of one LTSF dataset as a 1-D float32 array, plus its `dt`.

    Defaults to the last column, which is the `OT` target in the ETT/TSLib
    layout. Returns `(values, dt_seconds, spec)`.
    """
    series = load_forecast_series(dataset_name, **({"root": root} if root else {}))
    values = np.ascontiguousarray(series.values[:, variate], dtype=np.float32)
    return values, FREQ_SECONDS.get(series.spec.freq, 1.0), series.spec


def make_forecast_batch(
    series, start_indices, condition_len, target_len, patch_len, channel_id=0, modality_id=MODALITY_TIMESERIES, dt_seconds=900.0, device="cpu"
):
    """Condition = a lookback window; target = the next `target_len` steps.

    `series` is a 1-D array (or tensor) of one variate. Returns
    `(condition_batch, target_batch)`.
    """
    series = np.asarray(series.cpu() if torch.is_tensor(series) else series, dtype=np.float32)
    starts = list(start_indices)
    condition = np.stack([series[s : s + condition_len] for s in starts])
    target = np.stack([series[s + condition_len : s + condition_len + target_len] for s in starts])

    n_condition_patches = condition_len // patch_len
    t_step = 1.0 / n_condition_patches  # one condition window spans t in [0, 1)

    condition_batch = build_batch(condition, patch_len, 0.0, t_step, dt_seconds, channel_id, modality_id, TASK_FORECAST, device)
    target_batch = build_batch(target, patch_len, 1.0, t_step, dt_seconds, channel_id, modality_id, TASK_FORECAST, device)
    return condition_batch, target_batch


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------


def load_reconstruction_pairs(task_name="BIDMC-ECG", split="train", root=None):
    """Real PPG/target recordings for one reconstruction task and split.

    Returns `(spec, recordings)`; subject-wise split, never window-wise.
    """
    spec, recordings = load_recordings(task_name, **({"root": root} if root else {}))
    return spec, subject_split(recordings)[split]


def sample_reconstruction_windows(recordings, window_samples, batch_size, generator=None):
    """Draw `batch_size` random aligned `(condition, target)` windows."""
    generator = generator or np.random.default_rng(0)
    usable = [r for r in recordings if len(r.condition) > window_samples]
    if not usable:
        raise ValueError(f"no recording is longer than {window_samples} samples")

    conditions, targets = [], []
    for _ in range(batch_size):
        recording = usable[int(generator.integers(len(usable)))]
        start = int(generator.integers(len(recording.condition) - window_samples))
        conditions.append(recording.condition[start : start + window_samples])
        targets.append(recording.target[start : start + window_samples])
    return np.stack(conditions), np.stack(targets)


def make_reconstruction_batch(
    condition_windows,
    target_windows,
    fs,
    condition_patch_len,
    target_patch_len,
    condition_channel_id=1,
    target_channel_id=2,
    modality_id=MODALITY_BIOSIGNAL,
    device="cpu",
):
    """Condition = PPG; target = a *different channel* over the *same timespan*.

    The only thing distinguishing this from `make_forecast_batch` at the model's
    input is the metadata: the target's `t` overlaps the condition's rather than
    following it, and the channel ids differ. Nothing in the model branches on
    task -- that is the project's locked design decision.
    """
    dt_seconds = 1.0 / fs
    n_condition_patches = condition_windows.shape[1] // condition_patch_len
    n_target_patches = target_windows.shape[1] // target_patch_len
    if n_condition_patches == 0 or n_target_patches == 0:
        raise ValueError("window is shorter than one patch on the condition or target side")

    condition_batch = build_batch(
        condition_windows,
        condition_patch_len,
        0.0,
        1.0 / n_condition_patches,
        dt_seconds,
        condition_channel_id,
        modality_id,
        TASK_RECONSTRUCT,
        device,
    )
    target_batch = build_batch(
        target_windows,
        target_patch_len,
        0.0,  # aligned with the condition, not after it
        1.0 / n_target_patches,
        dt_seconds,
        target_channel_id,
        modality_id,
        TASK_RECONSTRUCT,
        device,
    )
    return condition_batch, target_batch
