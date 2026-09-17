"""Data layer for both benchmark tasks.

Named `datasets`, not `data`: the repo-root `data/` directory shadows a
top-level `data` package on `sys.path`, which is what made the previous
`src/data/` package fail to import even when it existed (see
docs/00_project_status.md).

    from rooster.datasets.forecasting import load_forecast_series
    from rooster.datasets.windows import build_forecast_windows
"""

from rooster.datasets.forecasting import (
    FORECAST_SPECS,
    ForecastSeries,
    ForecastSpec,
    available_datasets,
    load_forecast_series,
    resolve_spec,
)
from rooster.datasets.physio import (
    RECONSTRUCTION_SPECS,
    ReconstructionSpec,
    Recording,
    available_tasks,
    load_recordings,
    subject_split,
)
from rooster.datasets.splits import SplitBoundaries, StandardScaler
from rooster.datasets.windows import (
    ForecastWindows,
    ReconstructionWindows,
    build_forecast_windows,
    build_reconstruction_windows,
)

__all__ = [
    "FORECAST_SPECS",
    "RECONSTRUCTION_SPECS",
    "ForecastSeries",
    "ForecastSpec",
    "ForecastWindows",
    "Recording",
    "ReconstructionSpec",
    "ReconstructionWindows",
    "SplitBoundaries",
    "StandardScaler",
    "available_datasets",
    "available_tasks",
    "build_forecast_windows",
    "build_reconstruction_windows",
    "load_forecast_series",
    "load_recordings",
    "resolve_spec",
    "subject_split",
]
