"""The shared training contract: budget, base class, and checkpointing.

Every benchmark model trains through `BenchmarkModel.fit` under a
`TrainingBudget`, so an ablation's arms are guaranteed the same compute. The
checkpointing here is what makes long autonomous sweeps possible.
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TrainingBudget:
    """Matched training budget across every arm of an ablation.

    Held constant on purpose: an ablation where one arm trained longer is not an
    ablation. Expressed in optimiser steps rather than epochs so datasets of
    very different sizes get comparable compute.

    `checkpoint_dir` turns on save/resume. A long sweep gets interrupted -- by a
    session ending, an OOM, a machine reboot -- and without checkpoints every
    interruption costs the whole run from scratch. That happened twice while
    building this, which is why it is here.

    `warmup_steps` and `cosine_decay` matter once runs get long: a constant
    learning rate is fine for 2 000 steps and clearly not for 50 000.

    `patience_fraction` expresses early stopping as a fraction of the budget
    rather than a count of evaluations. Counting evaluations tied the stopping
    rule to `eval_every`, and with patience 3 it truncated exactly the models that
    need the full budget: FlowSSM-point-xvar stopped after ~8 000 of 20 000 steps
    even though its learning curve was still falling monotonically. A fraction of
    the budget is scale-invariant and says what it means -- "stop if a quarter of
    the run has passed with no improvement".

    Mixed precision is deliberately absent: see the note in models/common.py --
    bf16 measured 0.92-1.03x here because the S5 scan cannot be autocast and
    dominates the runtime.
    """

    def __init__(
        self,
        max_steps=2000,
        batch_size=64,
        learning_rate=1e-3,
        patience=3,
        patience_fraction=0.25,
        eval_every=200,
        num_workers=4,
        warmup_steps=0,
        cosine_decay=False,
        checkpoint_dir=None,
        checkpoint_every=0,
        selection="mse",
        epochs=None,
        recipe=None,
        halve_every_steps=0,
    ):
        # `epochs`/`recipe` describe an epoch-based recipe (the DecompSSM paper's
        # TSLib schedule: 10 epochs, LR halved every epoch, patience 3 epochs).
        # Steps per epoch depend on the dataset, so `resolved()` turns this into
        # concrete step counts once the training split is known.
        self.epochs = epochs
        self.recipe = recipe
        self.halve_every_steps = halve_every_steps
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.patience_fraction = patience_fraction
        self.eval_every = eval_every
        self.num_workers = num_workers
        self.warmup_steps = warmup_steps
        self.cosine_decay = cosine_decay
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every or eval_every
        # What early stopping selects on. "mse" (the default) selects on validation
        # forecast MSE; "objective" restores the pre-Aug-3 behaviour of selecting on
        # the model's own training objective, auxiliary terms included. The switch
        # exists because flipping it is the cheapest way to test whether the Aug-3
        # selection change is what moved DecompSSM on Exchange from 0.093 to 0.123.
        self.selection = selection

    @property
    def patience_steps(self):
        """Steps without improvement before stopping, at least one evaluation."""
        return max(int(self.patience_fraction * self.max_steps), self.eval_every)

    def resolved(self, n_train_windows):
        """Concrete step budget for an epoch-based recipe; identity otherwise."""
        if not self.epochs:
            return self
        import copy

        steps_per_epoch = max(n_train_windows // self.batch_size, 1)
        concrete = copy.copy(self)
        concrete.max_steps = self.epochs * steps_per_epoch
        concrete.eval_every = steps_per_epoch
        concrete.checkpoint_every = steps_per_epoch
        # TSLib "type1": the learning rate halves at every epoch boundary.
        concrete.halve_every_steps = steps_per_epoch
        # TSLib early stopping: patience of 3 epochs on validation loss.
        concrete.patience_fraction = min(3.0 / self.epochs, 1.0)
        return concrete

    def learning_rate_at(self, step):
        """Linear warmup then optional cosine decay to 1% of the peak; or the
        TSLib step schedule (halve every epoch) when a recipe sets it."""
        if self.halve_every_steps:
            return self.learning_rate * (0.5 ** (step // self.halve_every_steps))
        if self.warmup_steps and step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps
        if not self.cosine_decay:
            return self.learning_rate
        progress = (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
        return self.learning_rate * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

class BenchmarkModel(nn.Module):
    """Base class: shared supervised training loop and sampling contract.

    `is_probabilistic` tells the harness whether drawing more than one sample
    buys anything. Deterministic models are evaluated with a single sample and
    the ensemble is tiled, so CRPS correctly degenerates to MAE rather than
    being flattered by a fake spread.
    """

    is_probabilistic = False

    def forward(self, x):
        raise NotImplementedError

    def on_optimizer_step(self):
        """Hook for constraints that must hold after every update.

        Default is a no-op. `DictSSM` uses it to project its dictionary atoms back
        onto the unit sphere -- without that the atoms and their coefficients trade
        scale freely and the sparsity threshold stops meaning anything.
        """

    def compute_loss(self, x, y):
        """Training objective. Defaults to MSE on the point prediction.

        Generative models override this: a flow-matching model's objective is a
        velocity regression, not an MSE against `y`, and forcing it through the
        MSE path would train the wrong thing. Validation uses the same function,
        so early stopping always tracks the objective actually being optimised.
        """
        return F.mse_loss(self(x), y)

    def fit(self, train_dataset, val_dataset, budget, device, checkpoint_name=None):
        """Train to the budget, early-stopping on the validation objective.

        Returns the wall-clock training time so the leaderboard can show cost
        next to quality. The evaluated weights are always the best-validation
        ones, never the last -- reporting the last checkpoint is a common and
        silent way to overstate a result.

        When `budget.checkpoint_dir` and `checkpoint_name` are both set, progress
        is saved periodically and an existing checkpoint is resumed from, so a
        long sweep survives interruption. A resumed run that already reached the
        budget returns immediately with the best weights loaded.
        """
        started = time.perf_counter()
        self.to(device)
        budget = budget.resolved(len(train_dataset))
        optimizer = torch.optim.Adam(self.parameters(), lr=budget.learning_rate)

        path = _checkpoint_path(budget, checkpoint_name)
        state = _load_checkpoint(path, self, optimizer, device)
        step, best_loss, best_state, best_step = state
        if step >= budget.max_steps:
            if best_state is not None:
                self.load_state_dict(best_state)
            # The resumed path has to report the same things the trained path
            # does. It did not, and a re-run of a completed Exchange cell came
            # back with no validation loss at all -- which would silently drop
            # exactly the cells a component-count comparison depends on.
            self.best_val_loss, self.best_step = best_loss, best_step
            return time.perf_counter() - started

        train_loader = DataLoader(
            train_dataset,
            batch_size=budget.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=budget.num_workers,
            persistent_workers=budget.num_workers > 0,
        )

        stop = False
        while step < budget.max_steps and not stop:
            for x, y in train_loader:
                if step >= budget.max_steps:
                    break
                self.train()
                for group in optimizer.param_groups:
                    group["lr"] = budget.learning_rate_at(step)
                x, y = x.to(device), y.to(device)
                loss = self.compute_loss(x, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()
                self.on_optimizer_step()
                step += 1

                if step % budget.eval_every == 0 or step == budget.max_steps:
                    if budget.selection == "objective":
                        val_loss = self.evaluate_loss(val_dataset, budget, device)
                    else:
                        val_loss = self.selection_score(val_dataset, budget, device)
                    if val_loss < best_loss - 1e-6:
                        best_loss, best_step = val_loss, step
                        best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                    elif step - best_step >= budget.patience_steps:
                        stop = True

                if path and (step % budget.checkpoint_every == 0 or step >= budget.max_steps or stop):
                    # `stop` is persisted as a completed step count so a resumed
                    # run does not restart a search that early stopping ended.
                    _save_checkpoint(path, self, optimizer, budget.max_steps if stop else step, best_loss, best_state, best_step)
                if stop:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        # Exposed because model *selection* has to be done on validation, never on
        # test. Choosing a component count by test MSE would make the reported
        # number a training metric.
        self.best_val_loss = best_loss
        self.best_step = best_step
        return time.perf_counter() - started

    @torch.no_grad()
    def selection_score(self, dataset, budget, device):
        """What early stopping selects the checkpoint on: validation MSE.

        Deliberately **not** the training objective. For a flow-matching model the
        two are different things, and the objective turned out to be a poor proxy:
        it is computed in per-window normalized space, where the standard-deviation
        floor makes a flat condition window produce a huge normalized target, so a
        handful of windows dominated the number. Measured on ETTm1, FlowSSM's
        validation objective read 1.03 while a point-head variant read 4.39, yet
        the latter had the *better* test MSE -- the signal was tracking scaling
        artefacts rather than forecast quality.

        Selecting on the metric that gets reported is both fairer and simpler: a
        deterministic baseline whose objective already *is* MSE is unaffected,
        while a generative model stops being penalised for optimising something
        else. `evaluate_loss` remains available for diagnostics.
        """
        self.eval()
        loader = DataLoader(dataset, batch_size=budget.batch_size, shuffle=False, num_workers=0)
        total, count = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += float(((self(x) - y) ** 2).mean()) * len(x)
            count += len(x)
        return total / max(count, 1)

    @torch.no_grad()
    def evaluate_loss(self, dataset, budget, device):
        self.eval()
        loader = DataLoader(dataset, batch_size=budget.batch_size, shuffle=False, num_workers=0)
        total, count = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += float(self.compute_loss(x, y)) * len(x)
            count += len(x)
        return total / max(count, 1)

    @torch.no_grad()
    def sample(self, x, n_samples):
        """`(n_samples, batch, ...)` ensemble. Deterministic models tile."""
        prediction = self(x)
        return prediction.unsqueeze(0).expand(n_samples, *prediction.shape)

def _checkpoint_path(budget, checkpoint_name):
    if not budget.checkpoint_dir or not checkpoint_name:
        return None
    os.makedirs(budget.checkpoint_dir, exist_ok=True)
    return os.path.join(budget.checkpoint_dir, f"{checkpoint_name}.pt")


def _save_checkpoint(path, model, optimizer, step, best_loss, best_state, best_step):
    """Write atomically: a checkpoint truncated by a kill is worse than none."""
    payload = {
        "step": step,
        "best_loss": best_loss,
        "best_step": best_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_state": best_state,
    }
    temporary = f"{path}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_checkpoint(path, model, optimizer, device):
    """Restore `(step, best_loss, best_state, best_step)`; fresh start if absent.

    A checkpoint that cannot be loaded is reported and ignored rather than
    crashing the sweep -- losing one cell's history is better than losing the
    run. The catch is deliberately broad: a checkpoint can fail to load as an
    UnpicklingError (truncated by a kill), a RuntimeError (shapes changed since
    it was written), a KeyError (schema changed), and more, and enumerating them
    would mean the resilience feature itself becomes a source of crashes.
    """
    if not path or not os.path.exists(path):
        return 0, float("inf"), None, 0
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        return payload["step"], payload["best_loss"], payload.get("best_state"), payload.get("best_step", 0)
    except Exception as error:  # noqa: BLE001 -- see docstring
        print(f"  [checkpoint] ignoring unusable {path}: {type(error).__name__}: {error}")
        return 0, float("inf"), None, 0

class _UntrainedMixin:
    """Baselines with no parameters: `fit` is a no-op costing zero seconds."""

    def fit(self, train_dataset, val_dataset, budget, device, checkpoint_name=None):
        self.to(device)
        return 0.0


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------
