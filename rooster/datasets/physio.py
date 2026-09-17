"""Loaders for the PPG-to-vital-sign reconstruction datasets.

BIDMC and CapnoBase are the *light* pair: ~40-50 recordings of 8 minutes each,
so both fit in memory and a full train/eval cycle is cheap enough to run on
every leaderboard refresh. They are also clean, stationary clinical recordings,
and that turned out to matter more than their size -- measured, returning the PPG
unchanged scores 1.53 bpm of heart-rate error on BIDMC-ECG, so the task is nearly
saturated before any model trains (docs/03_datasets.md).

PPG-DaLiA is here for exactly that reason: wrist PPG recorded during daily
activity, where motion corrupts the signal and the task is genuinely hard --
PENGUIN reports 15.64 bpm on it. It is heavier (26 GB, 15 subjects of ~2.6 h) and
needs cross-rate resampling inside each record, 64 Hz wrist BVP against 700 Hz
chest ECG, unlike the clinical pair.

It also ships **expert R-peak positions**, which the light sets do not, and those
are carried through on the `Recording` rather than re-detected. That is what makes
it possible to ask whether a model reproduces beat *timing* or only waveform
shape -- the distinction the measured BIDMC result turns on, where a conv baseline
improved waveform correlation (r -0.00 -> 0.21) while making rate error four times
worse.

Splits are **subject-wise**, never window-wise -- the same recording appearing
in train and test inflates every metric and is the standard reviewer objection
to this literature (docs/02_benchmark_protocol.md section 3).
"""

import glob
import os
from dataclasses import dataclass

import numpy as np
import scipy.signal

DEFAULT_ROOT = os.path.join("data", "physio")

BIDMC_SUBDIR = os.path.join("bidmc", "bidmc-ppg-and-respiration-dataset-1.0.0")
CAPNOBASE_SUBDIR = os.path.join("capnobase", "data", "mat")
DALIA_SUBDIR = os.path.join("ppg_dalia", "PPG_FieldStudy")

# PPG-DaLiA's fixed acquisition rates. Not read from the files: the pickles carry
# no rate metadata, the rates are stated in the dataset paper, and hard-coding
# them makes a silent mismatch impossible.
DALIA_BVP_FS = 64.0
DALIA_ACC_FS = 32.0
DALIA_CHEST_FS = 700.0

# WESAD was recorded by the same group with the same devices (wrist Empatica E4,
# chest RespiBAN), so the rates match DaLiA's -- and like DaLiA's, they are stated
# in the dataset paper rather than stored in the pickles.
WESAD_SUBDIR = os.path.join("wesad", "WESAD")
WESAD_BVP_FS = 64.0
WESAD_ACC_FS = 32.0
WESAD_CHEST_FS = 700.0

WILDPPG_SUBDIR = os.path.join("WildPPG", "data")


@dataclass(frozen=True)
class ReconstructionSpec:
    """One PPG-to-X reconstruction task on one dataset."""

    name: str
    dataset: str  # "BIDMC" | "CapnoBase" | "PPG-DaLiA"
    target: str  # "ECG" | "RESP"
    native_fs: float
    target_fs: float  # rate everything is resampled to before windowing
    window_seconds: float
    notes: str = ""


RECONSTRUCTION_SPECS = {
    # BIDMC: 53 recordings, 8 min, 125 Hz. Impedance respiration + 3 ECG leads.
    # This is the reference set for the PPG->RR comparison against RespDiff and PENGUIN.
    "BIDMC-RESP": ReconstructionSpec("BIDMC-RESP", "BIDMC", "RESP", 125.0, 62.5, 32.0),
    # PENGUIN protocol variant: their RR Error is a Fourier estimate over
    # 60-second windows (arXiv:2602.03858 sec 4.2), and a 32 s window quantizes
    # RR to 1.875 breaths/min where 60 s gives 1.0 -- not comparable. Only rows
    # shared with PENGUIN's table use this spec.
    "BIDMC-RESP-60s": ReconstructionSpec("BIDMC-RESP-60s", "BIDMC", "RESP", 125.0, 62.5, 60.0),
    "BIDMC-ECG": ReconstructionSpec("BIDMC-ECG", "BIDMC", "ECG", 125.0, 125.0, 8.0),
    # CapnoBase: 42 cases, 8 min, 300 Hz, with expert R-peak and breath annotations.
    # CO2 capnography stands in for respiration and is the cleaner RR reference.
    "CapnoBase-RESP": ReconstructionSpec("CapnoBase-RESP", "CapnoBase", "RESP", 300.0, 62.5, 32.0, notes="target is CO2 capnography"),
    "CapnoBase-ECG": ReconstructionSpec("CapnoBase-ECG", "CapnoBase", "ECG", 300.0, 125.0, 8.0),
    # PPG-DaLiA: 15 subjects, ~2.6 h each, wrist BVP during daily activity. The
    # hard one, and the one PENGUIN reports on (15.64 bpm). `native_fs` is the
    # target's rate; the condition arrives at 64 Hz and both are put on the same
    # 125 Hz grid, matching the clinical ECG tasks so numbers stay comparable.
    "DaLiA-ECG": ReconstructionSpec(
        "DaLiA-ECG", "PPG-DaLiA", "ECG", DALIA_CHEST_FS, 125.0, 8.0,
        notes="wrist BVP 64 Hz vs chest ECG 700 Hz; ships R-peaks and activity labels",
    ),
    "DaLiA-RESP": ReconstructionSpec("DaLiA-RESP", "PPG-DaLiA", "RESP", DALIA_CHEST_FS, 62.5, 32.0, notes="chest respiration belt as target"),
    # The two remaining tasks PENGUIN's Table 1 reports that our data covers.
    # Both use the PENGUIN metric-protocol windows directly (60 s RR, 8 s HR)
    # since they exist only for that comparison -- there is no legacy 32 s cell
    # to stay consistent with.
    "WESAD-RESP": ReconstructionSpec("WESAD-RESP", "WESAD", "RESP", WESAD_CHEST_FS, 62.5, 60.0, notes="chest respiration belt; PENGUIN-protocol 60 s RR windows"),
    "WildPPG-ECG": ReconstructionSpec("WildPPG-ECG", "WildPPG", "ECG", 128.0, 125.0, 8.0, notes="wrist PPG vs sternum ECG, both native 128 Hz; resampled to the common 125 Hz ECG grid"),
}

# Respiration is a ~0.1-0.7 Hz process, so a 32 s window at 62.5 Hz sees several
# breaths; ECG needs the sample rate instead, so it gets a short window at full rate.


@dataclass
class Recording:
    """One subject's synchronous condition/target pair, already resampled."""

    subject: str
    condition: np.ndarray  # (n_samples,) PPG
    target: np.ndarray  # (n_samples,) ECG or respiration
    fs: float
    # Expert R-peak sample indices, on the same grid as `target`, where the
    # dataset provides them. Carried rather than re-detected: rate error is the
    # metric that decides these tasks, and a detector applied to the *reference*
    # signal would make the ground truth depend on the detector's own failures.
    rpeaks: np.ndarray = None
    # Per-sample activity code where provided, so results can be split by whether
    # the subject was moving -- which is the entire reason this dataset is hard.
    activity: np.ndarray = None
    # Motion covariate on the same grid as `condition`, where the device records
    # one: for PPG-DaLiA this is wrist accelerometer magnitude. It is a *condition*
    # channel, never a target -- the point is to tell the model which parts of the
    # PPG are artifact, which the stratified error analysis showed it cannot infer
    # from the PPG alone (rate predictions collapse toward the prior exactly in
    # high-motion windows).
    covariate: np.ndarray = None


def _resample_to(signal, source_fs, target_fs):
    if abs(source_fs - target_fs) < 1e-9:
        return np.asarray(signal, dtype=np.float32)
    n_out = int(round(len(signal) * target_fs / source_fs))
    return scipy.signal.resample(np.asarray(signal, dtype=np.float64), n_out).astype(np.float32)


def _load_bidmc(spec, root):
    """Read the WFDB records. `wfdb` is imported lazily so the module stays
    importable (and testable) on a machine without the physio data."""
    import wfdb

    directory = os.path.join(root, BIDMC_SUBDIR)
    headers = sorted(h for h in glob.glob(os.path.join(directory, "bidmc*.hea")) if not h.endswith("n.hea"))
    if not headers:
        raise FileNotFoundError(f"no BIDMC records under {directory}. See docs/03_datasets.md.")

    recordings = []
    for header in headers:
        record = wfdb.rdrecord(header[:-4])
        # Signal names carry a trailing comma in this dataset's headers ("PLETH,").
        names = [n.strip().rstrip(",").upper() for n in record.sig_name]
        ppg = record.p_signal[:, names.index("PLETH")]
        target_name = "RESP" if spec.target == "RESP" else "II"
        target = record.p_signal[:, names.index(target_name)]
        if not (np.isfinite(ppg).all() and np.isfinite(target).all()):
            continue  # a handful of records carry short NaN gaps; drop rather than impute
        recordings.append(
            Recording(
                subject=os.path.basename(header)[:-4],
                condition=_resample_to(ppg, record.fs, spec.target_fs),
                target=_resample_to(target, record.fs, spec.target_fs),
                fs=spec.target_fs,
            )
        )
    return recordings


def _load_capnobase(spec, root):
    """CapnoBase ships MATLAB v7.3 files, so h5py rather than scipy.io.loadmat."""
    import h5py

    directory = os.path.join(root, CAPNOBASE_SUBDIR)
    paths = sorted(glob.glob(os.path.join(directory, "*.mat")))
    if not paths:
        raise FileNotFoundError(f"no CapnoBase records under {directory}. See docs/03_datasets.md.")

    target_key = "co2" if spec.target == "RESP" else "ecg"
    recordings = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            ppg = np.asarray(handle["signal/pleth/y"]).ravel()
            target = np.asarray(handle[f"signal/{target_key}/y"]).ravel()
            ppg_fs = float(np.asarray(handle["param/samplingrate/pleth"]).ravel()[0])
            target_fs = float(np.asarray(handle[f"param/samplingrate/{target_key}"]).ravel()[0])
        if not (np.isfinite(ppg).all() and np.isfinite(target).all()):
            continue
        recordings.append(
            Recording(
                subject=os.path.basename(path).replace(".mat", ""),
                condition=_resample_to(ppg, ppg_fs, spec.target_fs),
                target=_resample_to(target, target_fs, spec.target_fs),
                fs=spec.target_fs,
            )
        )
    return recordings


def _load_dalia(spec, root):
    """Read PPG-DaLiA's per-subject pickles.

    Three things here are not incidental:

    `encoding="latin1"` because these are **Python 2 pickles** and fail to load
    without it (docs/03_datasets.md lists this among the traps).

    The condition and target arrive at different rates -- 64 Hz wrist BVP against
    700 Hz chest signals -- so both are resampled onto the spec's common grid.
    That means the PPG is *upsampled*, which invents no information; it just puts
    the pair on one axis so a window is the same span in both.

    R-peak indices are given on the 700 Hz grid and are rescaled to the target
    grid rather than re-detected. At 125 Hz one sample is 8 ms, which is well
    inside a plausible R-peak localisation error, so this preserves the timing
    reference the rate metrics depend on.
    """
    import pickle

    directory = os.path.join(root, DALIA_SUBDIR)
    paths = sorted(glob.glob(os.path.join(directory, "S*", "S*.pkl")))
    if not paths:
        raise FileNotFoundError(f"no PPG-DaLiA subjects under {directory}. See docs/03_datasets.md.")

    target_key = "Resp" if spec.target == "RESP" else "ECG"
    recordings = []
    for path in paths:
        with open(path, "rb") as handle:
            record = pickle.load(handle, encoding="latin1")

        ppg = np.asarray(record["signal"]["wrist"]["BVP"], dtype=np.float64).ravel()
        target = np.asarray(record["signal"]["chest"][target_key], dtype=np.float64).ravel()
        if not (np.isfinite(ppg).all() and np.isfinite(target).all()):
            continue

        condition = _resample_to(ppg, DALIA_BVP_FS, spec.target_fs)
        resampled_target = _resample_to(target, DALIA_CHEST_FS, spec.target_fs)
        # The two streams cover the same wall-clock span but rounding can leave
        # them a sample apart; trimming to the shorter keeps them index-aligned.
        n = min(len(condition), len(resampled_target))

        rpeaks = np.asarray(record.get("rpeaks", []), dtype=np.float64).ravel()
        scaled = np.round(rpeaks * spec.target_fs / DALIA_CHEST_FS).astype(np.int64)
        scaled = scaled[(scaled >= 0) & (scaled < n)]

        activity = np.asarray(record["activity"], dtype=np.float32).ravel() if "activity" in record else None

        covariate = None
        if "ACC" in record["signal"]["wrist"]:
            acc = np.asarray(record["signal"]["wrist"]["ACC"], dtype=np.float64)
            # Magnitude rather than axes: orientation is arbitrary on a wrist, the
            # amount of motion is what predicts artifact.
            magnitude = np.sqrt((acc.reshape(len(acc), -1) ** 2).sum(axis=1))
            covariate = _resample_to(magnitude, DALIA_ACC_FS, spec.target_fs)

        recordings.append(
            Recording(
                subject=str(record.get("subject", os.path.basename(path)[:-4])),
                condition=condition[:n],
                target=resampled_target[:n],
                fs=spec.target_fs,
                rpeaks=scaled,
                activity=activity,
                covariate=covariate[:n] if covariate is not None else None,
            )
        )
    return recordings


def _load_wesad(spec, root):
    """Read WESAD's per-subject pickles.

    Same acquisition setup and file layout as PPG-DaLiA (same group, same
    devices), so the same three traps apply: Python-2 pickles need
    `encoding="latin1"`, the wrist and chest streams arrive at different rates
    and are resampled onto one grid, and stream lengths can differ by a sample
    after rounding, so both are trimmed to the shorter. Unlike DaLiA there are
    no expert R-peaks to carry.
    """
    import pickle

    directory = os.path.join(root, WESAD_SUBDIR)
    paths = sorted(glob.glob(os.path.join(directory, "S*", "S*.pkl")))
    if not paths:
        raise FileNotFoundError(f"no WESAD subjects under {directory}. See docs/03_datasets.md.")

    target_key = "Resp" if spec.target == "RESP" else "ECG"
    recordings = []
    for path in paths:
        with open(path, "rb") as handle:
            record = pickle.load(handle, encoding="latin1")

        ppg = np.asarray(record["signal"]["wrist"]["BVP"], dtype=np.float64).ravel()
        target = np.asarray(record["signal"]["chest"][target_key], dtype=np.float64).ravel()
        if not (np.isfinite(ppg).all() and np.isfinite(target).all()):
            continue

        condition = _resample_to(ppg, WESAD_BVP_FS, spec.target_fs)
        resampled_target = _resample_to(target, WESAD_CHEST_FS, spec.target_fs)
        n = min(len(condition), len(resampled_target))

        covariate = None
        if "ACC" in record["signal"]["wrist"]:
            acc = np.asarray(record["signal"]["wrist"]["ACC"], dtype=np.float64)
            magnitude = np.sqrt((acc.reshape(len(acc), -1) ** 2).sum(axis=1))
            covariate = _resample_to(magnitude, WESAD_ACC_FS, spec.target_fs)

        recordings.append(
            Recording(
                subject=str(record.get("subject", os.path.basename(path)[:-4])),
                condition=condition[:n],
                target=resampled_target[:n],
                fs=spec.target_fs,
                covariate=covariate[:n] if covariate is not None else None,
            )
        )
    return recordings


def _interpolate_gaps(signal):
    """Fill non-finite samples by linear interpolation between finite neighbours.

    WildPPG is a free-living outdoor recording and carries occasional dropout
    gaps. Dropping a whole 12-hour participant for a gap (the clinical loaders'
    policy) would throw away most of the dataset, and zero-filling would put
    step edges through every gap; interpolation is the least-fabrication option.
    Returns None when less than half the signal is finite.
    """
    signal = np.asarray(signal, dtype=np.float64)
    finite = np.isfinite(signal)
    if finite.all():
        return signal
    if finite.mean() < 0.5:
        return None
    index = np.arange(len(signal))
    signal = signal.copy()
    signal[~finite] = np.interp(index[~finite], index[finite], signal[finite])
    return signal


def _load_wildppg(spec, root):
    """Read WildPPG's per-participant MATLAB files.

    One v5 .mat per participant, with one struct per body site; each signal is a
    {fs, descr, v} struct. The condition is the wrist green-wavelength PPG (the
    channel wearables actually use), the target the sternum lead-I ECG; both are
    native 128 Hz and get resampled to the spec's common ECG grid. Rates are
    read from the file's own `fs` fields and checked, not assumed.
    """
    import scipy.io

    directory = os.path.join(root, WILDPPG_SUBDIR)
    paths = sorted(glob.glob(os.path.join(directory, "WildPPG_Part_*.mat")))
    if not paths:
        raise FileNotFoundError(f"no WildPPG participants under {directory}. See docs/03_datasets.md.")

    recordings = []
    for path in paths:
        record = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        ppg_field = record["wrist"].ppg_g
        ecg_field = record["sternum"].ecg
        ppg = _interpolate_gaps(np.asarray(ppg_field.v).ravel())
        target = _interpolate_gaps(np.asarray(ecg_field.v).ravel())
        if ppg is None or target is None:
            continue

        condition = _resample_to(ppg, float(ppg_field.fs), spec.target_fs)
        resampled_target = _resample_to(target, float(ecg_field.fs), spec.target_fs)
        n = min(len(condition), len(resampled_target))
        recordings.append(
            Recording(
                subject=str(record.get("id", os.path.basename(path)[:-4])),
                condition=condition[:n],
                target=resampled_target[:n],
                fs=spec.target_fs,
            )
        )
    return recordings


_LOADERS = {
    "BIDMC": _load_bidmc,
    "CapnoBase": _load_capnobase,
    "PPG-DaLiA": _load_dalia,
    "WESAD": _load_wesad,
    "WildPPG": _load_wildppg,
}


CACHE_SUBDIR = "_cache"


def _cache_path(name, root):
    # v2: caches written before the covariate field existed lack it entirely, and
    # silently reading one would disable motion conditioning with no error. A new
    # filename forces a one-time rebuild instead.
    return os.path.join(root, CACHE_SUBDIR, f"{name}.v2.npz")


def load_recordings(name, root=DEFAULT_ROOT, use_cache=True):
    """Load every recording for a reconstruction task, resampled to a common rate.

    Cached to disk after the first call. PPG-DaLiA costs 70.7 s to read and
    resample -- 26 GB of Python-2 pickles, then an FFT resample of 700 Hz chest
    signals down to 125 Hz -- and every leaderboard cell paid it again. Across a
    three-seed sweep that is most of the wall clock, with the GPU idle throughout.
    BIDMC loads in 0.2 s and gains nothing, but the cache costs it nothing either.

    The cached arrays are exactly what the loader returned, so a cache hit and a
    cache miss produce identical recordings; delete `data/physio/_cache/` to
    rebuild. `use_cache=False` bypasses it, which is what the equivalence test
    uses.
    """
    if name not in RECONSTRUCTION_SPECS:
        raise KeyError(f"unknown reconstruction task {name!r}; known: {sorted(RECONSTRUCTION_SPECS)}")
    spec = RECONSTRUCTION_SPECS[name]

    path = _cache_path(name, root)
    if use_cache and os.path.exists(path):
        return spec, _read_cache(path, spec)

    recordings = _LOADERS[spec.dataset](spec, root)
    # Sorted subject ids make the split deterministic across machines.
    recordings = sorted(recordings, key=lambda r: r.subject)
    if use_cache:
        _write_cache(path, recordings)
    return spec, recordings


def _write_cache(path, recordings):
    """Written to a temporary file and renamed, so an interrupted write cannot
    leave a half-written cache that a later run would happily read."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {"subjects": np.array([r.subject for r in recordings], dtype=object)}
    for index, recording in enumerate(recordings):
        arrays[f"condition_{index}"] = recording.condition
        arrays[f"target_{index}"] = recording.target
        arrays[f"fs_{index}"] = np.float64(recording.fs)
        if recording.rpeaks is not None:
            arrays[f"rpeaks_{index}"] = recording.rpeaks
        if recording.activity is not None:
            arrays[f"activity_{index}"] = recording.activity
        if recording.covariate is not None:
            arrays[f"covariate_{index}"] = recording.covariate
    # np.savez appends ".npz" when the name lacks it, so the temporary name
    # carries the suffix already and the rename targets the file actually written.
    temporary = f"{path}.tmp{os.getpid()}.npz"
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _read_cache(path, spec):
    with np.load(path, allow_pickle=True) as handle:
        subjects = list(handle["subjects"])
        return [
            Recording(
                subject=str(subject),
                condition=handle[f"condition_{index}"],
                target=handle[f"target_{index}"],
                fs=float(handle[f"fs_{index}"]),
                rpeaks=handle[f"rpeaks_{index}"] if f"rpeaks_{index}" in handle else None,
                activity=handle[f"activity_{index}"] if f"activity_{index}" in handle else None,
                covariate=handle[f"covariate_{index}"] if f"covariate_{index}" in handle else None,
            )
            for index, subject in enumerate(subjects)
        ]


def subject_split(recordings, train_ratio=0.6, val_ratio=0.2):
    """Partition recordings by subject, in sorted-id order.

    Deterministic rather than random: with 42-53 subjects a random split has
    enough variance that two people would get visibly different numbers from
    the same code, which defeats the point of a shared leaderboard.
    """
    n_total = len(recordings)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    return {
        "train": recordings[:n_train],
        "val": recordings[n_train : n_train + n_val],
        "test": recordings[n_train + n_val :],
    }


def available_tasks(root=DEFAULT_ROOT):
    """Reconstruction tasks whose underlying dataset is present on this machine."""
    present = {
        "BIDMC": bool(glob.glob(os.path.join(root, BIDMC_SUBDIR, "bidmc*.hea"))),
        "CapnoBase": bool(glob.glob(os.path.join(root, CAPNOBASE_SUBDIR, "*.mat"))),
        "PPG-DaLiA": bool(glob.glob(os.path.join(root, DALIA_SUBDIR, "S*", "S*.pkl"))),
    }
    return [name for name, spec in RECONSTRUCTION_SPECS.items() if present.get(spec.dataset, False)]
