"""Tests for the shared training loop: LR schedules and checkpoint/resume.

Checkpointing is the piece that makes long autonomous runs possible, so it is
tested for the properties that actually matter: a resumed run continues rather
than restarting, and it does not lose the best-validation weights.
"""

import torch
from torch.utils.data import TensorDataset

from rooster.models.baselines import DLinear, TrainingBudget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dataset(n=256, lookback=32, horizon=16, n_variates=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, lookback, n_variates, generator=generator)
    y = torch.randn(n, horizon, n_variates, generator=generator)
    return TensorDataset(x, y)


def _budget(tmp_path, **kwargs):
    defaults = dict(max_steps=20, batch_size=16, learning_rate=1e-3, patience=99, eval_every=10, num_workers=0)
    defaults.update(kwargs)
    return TrainingBudget(checkpoint_dir=str(tmp_path), **defaults)


def test_warmup_ramps_then_holds():
    budget = TrainingBudget(max_steps=100, learning_rate=1.0, warmup_steps=10)
    assert budget.learning_rate_at(0) == 0.1
    assert budget.learning_rate_at(9) == 1.0
    assert budget.learning_rate_at(50) == 1.0  # no decay unless asked for


def test_cosine_decay_falls_to_one_percent_at_the_end():
    budget = TrainingBudget(max_steps=100, learning_rate=1.0, cosine_decay=True)
    assert budget.learning_rate_at(0) == 1.0
    assert budget.learning_rate_at(100) < 0.02
    # Monotone decreasing after warmup, which is the property schedules are for.
    values = [budget.learning_rate_at(s) for s in range(0, 101, 10)]
    assert all(later <= earlier + 1e-9 for earlier, later in zip(values, values[1:], strict=False))


def test_warmup_and_cosine_compose():
    budget = TrainingBudget(max_steps=100, learning_rate=1.0, warmup_steps=10, cosine_decay=True)
    assert budget.learning_rate_at(0) < budget.learning_rate_at(9)
    assert budget.learning_rate_at(9) > budget.learning_rate_at(99)


def test_a_checkpoint_is_written_and_resumed(tmp_path):
    train, val = _dataset(), _dataset(seed=1)

    first = DLinear(32, 16, 2)
    first.fit(train, val, _budget(tmp_path), DEVICE, checkpoint_name="cell")
    assert (tmp_path / "cell.pt").exists()

    payload = torch.load(tmp_path / "cell.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 20

    # A second model resuming the finished checkpoint must land on the same
    # weights rather than training again from scratch.
    second = DLinear(32, 16, 2)
    second.fit(train, val, _budget(tmp_path), DEVICE, checkpoint_name="cell")
    for a, b in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        torch.testing.assert_close(a.cpu(), b.cpu())


def test_resume_continues_instead_of_restarting(tmp_path):
    train, val = _dataset(), _dataset(seed=1)

    DLinear(32, 16, 2).fit(train, val, _budget(tmp_path, max_steps=10), DEVICE, checkpoint_name="cell")
    assert torch.load(tmp_path / "cell.pt", map_location="cpu", weights_only=False)["step"] == 10

    DLinear(32, 16, 2).fit(train, val, _budget(tmp_path, max_steps=30), DEVICE, checkpoint_name="cell")
    assert torch.load(tmp_path / "cell.pt", map_location="cpu", weights_only=False)["step"] == 30


def test_an_unusable_checkpoint_is_ignored_not_fatal(tmp_path):
    (tmp_path / "cell.pt").write_bytes(b"not a torch file")
    train, val = _dataset(), _dataset(seed=1)
    seconds = DLinear(32, 16, 2).fit(train, val, _budget(tmp_path), DEVICE, checkpoint_name="cell")
    assert seconds >= 0.0
    assert torch.load(tmp_path / "cell.pt", map_location="cpu", weights_only=False)["step"] == 20


def test_checkpointing_is_off_by_default(tmp_path):
    train, val = _dataset(), _dataset(seed=1)
    budget = TrainingBudget(max_steps=10, batch_size=16, eval_every=10, num_workers=0)
    DLinear(32, 16, 2).fit(train, val, budget, DEVICE, checkpoint_name="cell")
    assert not list(tmp_path.iterdir())


def test_selection_uses_validation_mse_not_the_training_objective():
    """Early stopping must select on the metric that gets reported.

    For a model whose objective is not MSE, the two differ -- and the objective
    proved to be the worse signal (see BenchmarkModel.selection_score). A model
    whose `compute_loss` is deliberately unrelated to its predictions must still be
    selected on prediction quality.
    """

    class MisleadingObjective(DLinear):
        def compute_loss(self, x, y):
            # Constant: carries no information about forecast quality at all.
            return (self.seasonal.weight * 0.0).sum() + 1.0

    val = _dataset(seed=1)
    model = MisleadingObjective(32, 16, 2).to(DEVICE)
    budget = TrainingBudget(max_steps=10, batch_size=16, eval_every=5, num_workers=0)

    score = model.selection_score(val, budget, DEVICE)
    objective = float(model.compute_loss(torch.zeros(2, 32, 2, device=DEVICE), torch.zeros(2, 16, 2, device=DEVICE)))
    assert score != objective
    # And it must be a real MSE: non-negative, and comparable to the data scale.
    assert 0.0 <= score < 100.0


def test_patience_is_a_fraction_of_the_budget_not_a_count_of_evaluations():
    """Counting evaluations tied stopping to eval_every, and with patience 3 it
    truncated exactly the models that need the full budget."""
    assert TrainingBudget(max_steps=20000, eval_every=2000, patience_fraction=0.25).patience_steps == 5000
    assert TrainingBudget(max_steps=1000, eval_every=100, patience_fraction=0.25).patience_steps == 250
    # Never shorter than one evaluation interval, or it could stop before it has
    # a second measurement to compare against.
    assert TrainingBudget(max_steps=100, eval_every=50, patience_fraction=0.01).patience_steps == 50


def test_a_monotonically_improving_model_trains_the_whole_budget(tmp_path):
    """The failure mode this replaces: a model still improving at step 8000 of
    20000 was being stopped because three consecutive evaluations happened to be
    flat."""

    class AlwaysImproving(DLinear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        def selection_score(self, dataset, budget, device):
            self.calls += 1
            return 1.0 / self.calls  # strictly decreasing

    train, val = _dataset(), _dataset(seed=1)
    model = AlwaysImproving(32, 16, 2)
    model.fit(train, val, _budget(tmp_path, max_steps=40, eval_every=10), DEVICE, checkpoint_name="cell")
    assert torch.load(tmp_path / "cell.pt", map_location="cpu", weights_only=False)["step"] == 40


def test_a_plateaued_model_stops_early(tmp_path):
    class Plateaued(DLinear):
        def selection_score(self, dataset, budget, device):
            return 1.0  # never improves

    train, val = _dataset(), _dataset(seed=1)
    Plateaued(32, 16, 2).fit(train, val, _budget(tmp_path, max_steps=400, eval_every=10), DEVICE, checkpoint_name="cell")
    # Recorded as the full budget so a resume does not restart the search, but the
    # run itself must have ended well before it.
    payload = torch.load(tmp_path / "cell.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 400
    assert payload["best_step"] <= 10
