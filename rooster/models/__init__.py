"""Importing this package registers every model configuration used in the paper."""

import rooster.models.configs  # noqa: F401  (the paper's configurations)
import rooster.models.decomp_dict  # noqa: F401  (registers DecompDict)
from rooster.models.aligned_context import BIAS_MODES, RelativeAlignedContext
from rooster.models.baselines import BENCHMARK_MODELS, DecompSSM, Penguin, build_model, register_benchmark_model
from rooster.models.flow_ssm import FlowSSMForecaster, FlowSSMReconstructor

__all__ = ["BENCHMARK_MODELS", "BIAS_MODES", "DecompSSM", "FlowSSMForecaster", "FlowSSMReconstructor", "Penguin", "RelativeAlignedContext", "build_model", "register_benchmark_model"]
