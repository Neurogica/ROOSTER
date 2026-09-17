"""Benchmark baselines, one model per module.

Importing this package registers every baseline into `BENCHMARK_MODELS`, which is
what `build_model` and the leaderboard runner resolve names against. Adding a
baseline means adding a module here and one line to the import block below --
nothing else in the repo needs to change.

    registry.py   BENCHMARK_MODELS, register_benchmark_model, build_model
    training.py   TrainingBudget, BenchmarkModel, checkpoint save/resume

Baselines:
    repeat_last.py         persistence -- the floor every forecaster must clear
    linear.py              one shared Linear(lookback -> horizon)
    dlinear.py             DLinear (AAAI 2023); validates the harness
    decompssm.py           DecompSSM (ICASSP 2026), our own prior work
    copy_input.py          return the PPG unchanged -- a demanding floor
    constant_rate.py       a fixed-rate beat train -- the control for any rate claim
    direct_rate.py         regress the rate from the PPG -- the constructive arm
    conv_reconstructor.py  small dilated-conv PPG -> vitals reference
    penguin.py             PENGUIN (ICASSP 2026), our own prior work
"""

from rooster.models.baselines.constant_rate import ConstantRate
from rooster.models.baselines.conv_reconstructor import ConvReconstructor
from rooster.models.baselines.copy_input import CopyInput
from rooster.models.baselines.decompssm import DecompSSM
from rooster.models.baselines.dlinear import DLinear
from rooster.models.baselines.linear import LinearForecaster
from rooster.models.baselines.penguin import Penguin
from rooster.models.baselines.registry import BENCHMARK_MODELS, build_model, count_parameters, register_benchmark_model
from rooster.models.baselines.repeat_last import RepeatLast
from rooster.models.baselines.training import BenchmarkModel, TrainingBudget

__all__ = [
    "BENCHMARK_MODELS",
    "BenchmarkModel",
    "ConstantRate",
    "ConvReconstructor",
    "CopyInput",
    "DLinear",
    "DecompSSM",
    "LinearForecaster",
    "Penguin",
    "RepeatLast",
    "TrainingBudget",
    "build_model",
    "count_parameters",
    "register_benchmark_model",
]
