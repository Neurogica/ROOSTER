"""Model registry for the leaderboard.

Separate from `utils.help_func.MODEL_REGISTRY`, which is keyed to the
tokenizer/flow-variant constructor signature. Benchmark models take plain
tensors and plain window lengths, so they need their own namespace.
"""

import inspect

BENCHMARK_MODELS = {}


def register_benchmark_model(name):
    """Class decorator registering a baseline under a leaderboard-visible name.

    Separate from `utils.help_func.MODEL_REGISTRY`, which is keyed to the
    tokenizer/flow-variant constructor signature. These take plain tensors.
    """

    def decorator(cls):
        BENCHMARK_MODELS[name] = cls
        return cls

    return decorator

def build_model(name, **kwargs):
    """Instantiate a registered baseline by leaderboard name.

    Keyword arguments a model does not accept are dropped rather than raising:
    the harness offers everything it knows about a task (window length, sample
    rate, variate count) and each model takes what it needs. Adding a new
    context field therefore does not require touching every model.
    """
    if name not in BENCHMARK_MODELS:
        raise KeyError(f"unknown benchmark model {name!r}; known: {sorted(BENCHMARK_MODELS)}")
    factory = BENCHMARK_MODELS[name]
    # Candidates (models/candidates.py) register a closure rather than a class, so
    # inspect the constructor for classes and the function itself otherwise.
    target = factory.__init__ if isinstance(factory, type) else factory
    try:
        accepted = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return factory(**kwargs)
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return factory(**kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
