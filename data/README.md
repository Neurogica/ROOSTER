# Data

Nothing here is tracked. Download each dataset from its source and place it as below,
or pass another root with `--data-root`.

## Forecasting (`data/forecast/`)

| Dataset | Source | File |
|---|---|---|
| ETTh1, ETTh2, ETTm1, ETTm2 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) | `ETT-small/ETT*.csv` |
| Weather, Electricity (ECL), Traffic, Exchange | [Time-Series-Library bundle (HuggingFace mirror)](https://huggingface.co/datasets/thuml/Time-Series-Library) | `weather/weather.csv`, `electricity/electricity.csv`, ... |
| PEMS04 | [ASTGNN](https://github.com/guoshnBJTU/ASTGNN) | `PEMS04/PEMS04.npz` |

The protocol follows DecompSSM: lookback 96, horizons 96/192/336/720 (12/24/48/96 on
PEMS04), chronological splits, scaler fit on the training split only, MSE/MAE in scaled
space. See `rooster/datasets/forecasting.py` for the exact file names the loader expects
(`FORECAST_SPECS`).

## PPG-to-vital-sign reconstruction (`data/physio/`)

| Dataset | Source | Used for |
|---|---|---|
| PPG-DaLiA | [UCI](https://archive.ics.uci.edu/dataset/495/ppg+dalia) | PPG -> ECG (wrist, free-living) |
| WildPPG | [ETH Zurich](https://siplab.org/projects/WildPPG) | PPG -> ECG (wrist, outdoor) |
| BIDMC | [PhysioNet](https://physionet.org/content/bidmc/) | PPG -> respiration, PPG -> ECG |
| WESAD | [UCI](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) | PPG -> respiration |
| CapnoBase | [Dataverse](https://borealisdata.ca/dataverse/capnobase) | PPG -> ECG / respiration (matched harness) |

Metric protocol follows PENGUIN: heart rate by Hamilton beat detection over 8 s windows,
respiratory rate by the Fourier dominant frequency over 60 s windows, strictly
subject-wise splits. Loader details and gotchas (Python-2 pickles, MATLAB v7.3 files,
per-signal sampling rates) are documented in `rooster/datasets/physio.py`.
