"""Shared training utilities: model registry, seeding, and checkpointing --
mirrors PENGUIN's utils/help_func.py (register_baseline/initialize_model/
fix_seed/load_checkpoint), adapted for UniSignal-Flow's two variants and two
tasks instead of PENGUIN's five baselines and six datasets.

Import direction note: models/*.py import `register_model` from here, so
this module must not import anything from `models` itself (unlike PENGUIN's
utils/__init__.py, which imports from `models` to trigger registration --
that only works there because `register_baseline` lives in a *different*
submodule, utils/help_func.py, than the one doing the importing,
utils/__init__.py). Here, registration is instead triggered by
`models/__init__.py` importing its own submodules -- see that file.
"""

import random

import numpy as np
import torch

MODEL_REGISTRY = {}


def register_model(name):
    """Class decorator: `@register_model("VariantLaplacian")` on a model
    class registers it under that name, so config.yaml's `train.model:
    VariantLaplacian` string is enough for initialize_model() to find it --
    no if/elif chain to extend when a new variant is added."""

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def initialize_model(cfg, device="cpu"):
    """Builds `cfg.train.model` from MODEL_REGISTRY using the shared
    tokenizer vocabulary in cfg.data (patch_len/num_channels/num_modalities/
    num_tasks -- the same vocabulary both tasks draw channel/modality/task
    ids from, see config/data.yaml) plus that model's own hyperparameters
    from cfg.model."""
    model_name = cfg.train.model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"{model_name} is not registered. Known models: {list(MODEL_REGISTRY)}")
    model_cfg = getattr(cfg.model, model_name)
    model = MODEL_REGISTRY[model_name](
        patch_len=cfg.data.patch_len,
        num_channels=cfg.data.num_channels,
        num_modalities=cfg.data.num_modalities,
        num_tasks=cfg.data.num_tasks,
        **model_cfg,
    )
    return model.to(device)


def save_checkpoint(path, model, cfg, step):
    torch.save({"step": step, "state_dict": model.state_dict(), "cfg": cfg}, path)


def load_checkpoint(path, cfg, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = initialize_model(checkpoint.get("cfg", cfg), device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded checkpoint '{path}' (step {checkpoint['step']})")
    return model
