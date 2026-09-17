"""Loaders for the long-term time-series forecasting (LTSF) benchmark suite.

Every dataset is described by a `ForecastSpec` and reduced to the same shape:
a dense `(n_timesteps, n_variates)` float array plus the split boundaries and
the horizon set the literature evaluates it at. Nothing here trains or scales
-- `windows.py` owns that -- so a spec can be inspected cheaply.

Sources and per-file shapes are recorded in data/forecast/MANIFEST.md.
"""

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rooster.datasets.splits import ETT_HOURLY_BOUNDARIES, ETT_MINUTE_BOUNDARIES, fixed_boundaries, ratio_boundaries

DEFAULT_ROOT = os.path.join("data", "forecast")

# The standard horizon set for the Autoformer-family datasets. ILI and PEMS use
# their own, for the reasons noted on each spec below.
STANDARD_HORIZONS = (96, 192, 336, 720)
ILI_HORIZONS = (24, 36, 48, 60)
PEMS_HORIZONS = (12, 24, 48, 96)


@dataclass(frozen=True)
class ForecastSpec:
    """Everything needed to load and evaluate one forecasting dataset."""

    name: str
    path: str
    loader: str  # "csv_dated" | "headerless_csv" | "pems_npz"
    freq: str
    boundaries: str  # "ett_hourly" | "ett_minute" | "ratio_70_10_20" | "ratio_60_20_20"
    horizons: tuple = STANDARD_HORIZONS
    lookback: int = 96
    notes: str = ""
    aliases: tuple = field(default_factory=tuple)


FORECAST_SPECS = {
    # -- ETT: fixed 12/4/4-month boundaries from the Informer paper, not ratios.
    "ETTh1": ForecastSpec("ETTh1", "ETT-small/ETTh1.csv", "csv_dated", "1h", "ett_hourly"),
    "ETTh2": ForecastSpec("ETTh2", "ETT-small/ETTh2.csv", "csv_dated", "1h", "ett_hourly"),
    "ETTm1": ForecastSpec("ETTm1", "ETT-small/ETTm1.csv", "csv_dated", "15min", "ett_minute"),
    "ETTm2": ForecastSpec("ETTm2", "ETT-small/ETTm2.csv", "csv_dated", "15min", "ett_minute"),
    # -- Autoformer/TimesNet bundle: 70/10/20 chronological ratio split.
    "Weather": ForecastSpec("Weather", "weather/weather.csv", "csv_dated", "10min", "ratio_70_10_20"),
    "Electricity": ForecastSpec("Electricity", "electricity/electricity.csv", "csv_dated", "1h", "ratio_70_10_20", aliases=("ECL",)),
    "Traffic": ForecastSpec("Traffic", "traffic/traffic.csv", "csv_dated", "1h", "ratio_70_10_20"),
    "Exchange": ForecastSpec(
        "Exchange",
        "exchange_rate/exchange_rate.csv",
        "csv_dated",
        "1d",
        "ratio_70_10_20",
        notes="non-ISO date format (1990/1/1 0:00) upstream; pandas infers it correctly",
    ),
    "ILI": ForecastSpec(
        "ILI",
        "illness/national_illness.csv",
        "csv_dated",
        "7d",
        "ratio_70_10_20",
        horizons=ILI_HORIZONS,
        lookback=36,
        notes="only 966 rows; the literature uses lookback 36 and horizons 24/36/48/60",
    ),
    "Solar": ForecastSpec(
        "Solar",
        "Solar/solar_AL.txt",
        "headerless_csv",
        "10min",
        "ratio_70_10_20",
        notes="no date column upstream: 52560 rows = 365 days x 144 ten-minute steps",
    ),
    # -- PEMS: traffic-flow tensors, 60/20/20 split and short horizons (STGNN convention).
    # PEMS07 (883 sensors) is deliberately absent: it alone accounted for ~5 h of a
    # sweep while answering the same cross-variate question as PEMS03/04/08. The
    # file is still on disk if it is ever wanted back.
    "PEMS03": ForecastSpec("PEMS03", "PEMS/PEMS03.npz", "pems_npz", "5min", "ratio_60_20_20", horizons=PEMS_HORIZONS),
    "PEMS04": ForecastSpec("PEMS04", "PEMS/PEMS04.npz", "pems_npz", "5min", "ratio_60_20_20", horizons=PEMS_HORIZONS),
    "PEMS08": ForecastSpec("PEMS08", "PEMS/PEMS08.npz", "pems_npz", "5min", "ratio_60_20_20", horizons=PEMS_HORIZONS),
}

_ALIASES = {alias: spec.name for spec in FORECAST_SPECS.values() for alias in spec.aliases}


def resolve_spec(name):
    """Look up a spec by canonical name or documented alias (e.g. ECL)."""
    canonical = _ALIASES.get(name, name)
    if canonical not in FORECAST_SPECS:
        raise KeyError(f"unknown forecasting dataset {name!r}; known: {sorted(FORECAST_SPECS)}")
    return FORECAST_SPECS[canonical]


def _load_csv_dated(path):
    """Standard TSLib layout: a `date` column followed by the variates."""
    frame = pd.read_csv(path)
    date_columns = [c for c in frame.columns if c.strip().lower() == "date"]
    values = frame.drop(columns=date_columns) if date_columns else frame
    return values.to_numpy(dtype=np.float32)


def _load_headerless_csv(path):
    return np.loadtxt(path, delimiter=",", dtype=np.float32)


def _load_pems_npz(path):
    """PEMS tensors are `(T, N, C)`; only the flow channel is used for forecasting.

    PEMS04/08 carry (flow, occupancy, speed) while PEMS03 carries flow alone.
    Taking channel 0 everywhere gives all four the same `(T, N)` layout and
    matches what the forecasting literature reports on.
    """
    tensor = np.load(path)["data"]
    return np.ascontiguousarray(tensor[..., 0], dtype=np.float32)


_LOADERS = {
    "csv_dated": _load_csv_dated,
    "headerless_csv": _load_headerless_csv,
    "pems_npz": _load_pems_npz,
}

_BOUNDARY_BUILDERS = {
    "ett_hourly": lambda n: fixed_boundaries(n, *ETT_HOURLY_BOUNDARIES),
    "ett_minute": lambda n: fixed_boundaries(n, *ETT_MINUTE_BOUNDARIES),
    "ratio_70_10_20": lambda n: ratio_boundaries(n, 0.7, 0.1),
    "ratio_60_20_20": lambda n: ratio_boundaries(n, 0.6, 0.2),
}


@dataclass
class ForecastSeries:
    """A loaded dataset: dense values plus its split boundaries."""

    spec: ForecastSpec
    values: np.ndarray  # (n_timesteps, n_variates)
    boundaries: object

    @property
    def n_variates(self):
        return self.values.shape[1]

    @property
    def n_timesteps(self):
        return self.values.shape[0]


def load_forecast_series(name, root=DEFAULT_ROOT):
    """Load one LTSF dataset into a `(T, C)` array with its split boundaries."""
    spec = resolve_spec(name)
    path = os.path.join(root, spec.path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{spec.name} not found at {path}. See docs/03_datasets.md for how to obtain it.")

    values = _LOADERS[spec.loader](path)
    if values.ndim != 2:
        raise ValueError(f"{spec.name} loaded with shape {values.shape}; expected 2-D (timesteps, variates)")
    if not np.isfinite(values).all():
        raise ValueError(f"{spec.name} contains non-finite values; the manifest records it as NaN-free, so the file is likely corrupt")

    boundaries = _BOUNDARY_BUILDERS[spec.boundaries](values.shape[0])
    return ForecastSeries(spec=spec, values=values, boundaries=boundaries)


def available_datasets(root=DEFAULT_ROOT):
    """Names of the specs whose files are actually present on this machine."""
    return [name for name, spec in FORECAST_SPECS.items() if os.path.exists(os.path.join(root, spec.path))]
