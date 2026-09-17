"""The model configurations reported in the paper, registered under their leaderboard names.

Each entry bakes a set of constructor overrides into a base class; nothing else
changes between a configuration and its base. Names follow the paper:

* ``DecompDict-K3-*``  forecasting host (DecompSSM-derived, K = 3 branches);
  ``-q1``/``-q2`` = auxiliary quantile weight 1/2, ``-align`` = relative aligned
  conditioning, ``-w192``/``-w512`` = model width, ``-paired`` = slow-first branch order.
* ``FlowSSMRecon-*``   reconstruction host (PENGUIN-derived); ``-align`` = ours,
  ``-skiponly`` = PENGUIN's identity skip, plain = per-layer injection only.
* ``*-nobias``/``-bump``/``-free``/``-frozen``  bias-form ablations of the module.
"""

import inspect

from rooster.models.baselines.registry import register_benchmark_model
from rooster.models.decomp_dict import DecompDict
from rooster.models.flow_ssm import FlowSSMReconstructor

_K3 = {"n_atoms": 3, "sparsity_weight": 0.0, "atom_usage_weight": 0.0}

# Table 2 (published protocol): the configuration used on each dataset.
#   ECL      DecompDict-K3-q2-align-w512
#   Weather  DecompDict-K3-q1-align
#   ETTm2    DecompDict-K3-q1-align
#   PEMS04   DecompDict-K3-q1-align-paired-w192
FORECAST_CONFIGS = {
    "DecompDict-K3": {**_K3},
    "DecompDict-K3-q1": {**_K3, "quantile_weight": 1.0},
    "DecompDict-K3-align": {**_K3, "relative_align": True},
    "DecompDict-K3-q1-align": {**_K3, "quantile_weight": 1.0, "relative_align": True},
    "DecompDict-K3-q2-align-w512": {**_K3, "quantile_weight": 2.0, "relative_align": True, "d_model": 512},
    "DecompDict-K3-paired-w192": {**_K3, "slow_first": True, "d_model": 192},
    "DecompDict-K3-q1-paired-w192": {**_K3, "quantile_weight": 1.0, "slow_first": True, "d_model": 192},
    "DecompDict-K3-align-paired-w192": {**_K3, "relative_align": True, "slow_first": True, "d_model": 192},
    "DecompDict-K3-q1-align-paired-w192": {**_K3, "quantile_weight": 1.0, "relative_align": True, "slow_first": True, "d_model": 192},
    # bias-form ablation (Sec. 4.5)
    "DecompDict-K3-q1-align-nobias": {**_K3, "quantile_weight": 1.0, "relative_align": True, "align_bias_mode": "none"},
    "DecompDict-K3-q1-align-bump": {**_K3, "quantile_weight": 1.0, "relative_align": True, "align_bias_mode": "bump"},
    "DecompDict-K3-q1-align-free": {**_K3, "quantile_weight": 1.0, "relative_align": True, "align_bias_mode": "free"},
    "DecompDict-K3-q1-align-frozen": {**_K3, "quantile_weight": 1.0, "relative_align": True, "align_learn_bias": False},
}

# Tables 1, 3, 4: ``FlowSSMRecon`` itself is registered in flow_ssm.py (injection only).
RECONSTRUCTION_CONFIGS = {
    "FlowSSMRecon-align": {"relative_align": True, "condition_skip": False},
    "FlowSSMRecon-skiponly": {"condition_inject": False, "condition_skip": True},
    "FlowSSMRecon-align-nobias": {"relative_align": True, "condition_skip": False, "align_bias_mode": "none"},
}


def _register(name, base_class, overrides):
    def factory(**kwargs):
        return base_class(**{**kwargs, **overrides})

    factory.__name__ = name
    factory.__signature__ = inspect.signature(base_class.__init__)  # lets build_model drop unknown kwargs
    register_benchmark_model(name)(factory)
    return factory


for _name, _overrides in FORECAST_CONFIGS.items():
    _register(_name, DecompDict, _overrides)
for _name, _overrides in RECONSTRUCTION_CONFIGS.items():
    _register(_name, FlowSSMReconstructor, _overrides)
