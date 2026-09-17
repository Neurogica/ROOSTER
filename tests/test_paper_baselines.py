"""Tests for the vendored DecompSSM and PENGUIN baselines.

The models are the authors' own code (models/vendor/), so these do not re-test
their internals. They check the two things that could silently go wrong on our
side: that the adapters preserve the upstream training objective, and that the
harness feeds them the context they need.
"""

import torch

from rooster.models.baselines import BENCHMARK_MODELS, DecompSSM, Penguin, build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_both_baselines_are_registered():
    assert {"DecompSSM", "PENGUIN"} <= set(BENCHMARK_MODELS)


def test_decompssm_shapes_and_determinism_flag():
    model = DecompSSM(96, 96, 7, d_model=32).to(DEVICE)
    x = torch.randn(4, 96, 7, device=DEVICE)
    assert model(x).shape == (4, 96, 7)
    assert model.is_probabilistic is False


def test_decompssm_objective_includes_the_upstream_auxiliary_loss():
    """Upstream adds get_auxiliary_loss() during training; dropping it would
    train a different model from the published one."""
    model = DecompSSM(96, 96, 3, d_model=32).to(DEVICE)
    x = torch.randn(4, 96, 3, device=DEVICE)
    y = torch.randn(4, 96, 3, device=DEVICE)

    model.train()
    total = model.compute_loss(x, y).item()
    auxiliary = model.model.get_auxiliary_loss().item()
    assert auxiliary > 0.0
    assert total > auxiliary


def test_decompssm_denormalizes_to_the_inputs_units():
    """The model carries its own non-stationary normalization, so a shifted
    input must produce a correspondingly shifted forecast."""
    torch.manual_seed(0)
    model = DecompSSM(96, 96, 2, d_model=32).to(DEVICE).eval()
    base = torch.randn(4, 96, 2, device=DEVICE)
    with torch.no_grad():
        plain = model(base)
        shifted = model(base + 500.0)
    assert (shifted.mean() - plain.mean()).abs() > 100.0


def test_penguin_shapes_and_probabilistic_flag():
    model = Penguin(256, sample_rate=64, n_step=2).to(DEVICE)
    x = torch.randn(3, 256, device=DEVICE)
    samples = model.sample(x, n_samples=4)
    assert samples.shape == (4, 3, 256)
    assert torch.isfinite(samples).all()
    assert model.is_probabilistic is True


def test_penguin_loss_is_the_velocity_mse_not_the_waveform_mse():
    """Upstream's optimize() scores pred_dx_t against dx_t. Scoring the waveform
    train_flow returns instead would optimise a different objective."""
    torch.manual_seed(0)
    model = Penguin(256, sample_rate=64, n_step=2).to(DEVICE)
    x = torch.randn(3, 256, device=DEVICE)
    y = torch.randn(3, 256, device=DEVICE)

    loss = model.compute_loss(x, y)
    expected = torch.nn.functional.mse_loss(model.model.pred_dx_t, model.model.dx_t)
    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)


def test_penguin_starts_as_an_exact_no_op():
    """Upstream zero-initialises the final layer and every adaLN modulation.

    So an untrained PENGUIN predicts exactly zero velocity and `sample` returns
    its own noise unchanged. That is deliberate (the DiT zero-init recipe), and
    it is worth pinning: it means an untrained PENGUIN is not merely bad, it is
    the identity, and any conditioning test has to train first.
    """
    torch.manual_seed(0)
    model = Penguin(256, sample_rate=64, n_step=2).to(DEVICE).eval()
    x = torch.randn(2, 256, device=DEVICE)
    with torch.no_grad():
        velocity = model.model.forward_step(torch.randn(2, 1, 256, device=DEVICE), x.unsqueeze(1), torch.rand(2, 1, device=DEVICE))
    assert torch.count_nonzero(velocity) == 0


def test_penguin_conditions_on_the_ppg_once_trained():
    """Per-timestep additive PPG conditioning is the paper's central design
    choice; if the output ignored the condition, the model would be an
    unconditional generator wearing a baseline's name."""
    torch.manual_seed(0)
    model = Penguin(256, sample_rate=64, n_step=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(4, 256, device=DEVICE)
    y = torch.randn(4, 256, device=DEVICE)
    for _ in range(20):
        loss = model.compute_loss(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    a, b = torch.randn(2, 256, device=DEVICE), torch.randn(2, 256, device=DEVICE)
    torch.manual_seed(5)
    out_a = model.sample(a, n_samples=2)
    torch.manual_seed(5)
    out_b = model.sample(b, n_samples=2)
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_penguin_conv_front_end_tracks_the_task_sample_rate():
    """kernel_size = sample_rate // 4 upstream, so passing the wrong rate
    silently changes the architecture."""
    assert Penguin(512, sample_rate=128).model.pre_conv_ppg[0].kernel_size[0] == 32
    assert Penguin(512, sample_rate=64).model.pre_conv_ppg[0].kernel_size[0] == 16


def test_build_model_drops_kwargs_a_model_does_not_accept():
    """The harness offers every context field it has; models take what they need."""
    model = build_model("ConvReconstructor", window_samples=128, sample_rate=125.0)
    assert model is not None


def test_the_gate_never_compares_across_budgets():
    """Budget is part of cell identity, and the gate has to honour that.

    `RunRecord.key` includes the budget because two sweeps at different step
    counts are not comparable -- that was settled earlier in the project. But
    `scores_by_model` keyed on (dataset, horizon, seed) only, so records from
    different budgets collided in one dict and the surviving value was whichever
    was appended last. A candidate could then be judged against a baseline trained
    for a different number of steps, silently.
    """
    from rooster.evaluation.leaderboard import RunRecord
    from rooster.evaluation.promotion import head_to_head, scores_by_model

    def record(model, budget, mse):
        return RunRecord(
            task="forecast", dataset="ETTm1", model=model, horizon=96, seed=0,
            metrics={"mse": mse, "mae": mse}, lookback=96, protocol="test", budget=budget,
        )

    records = [
        record("Candidate", "steps=20000", 0.30),
        record("Baseline", "steps=20000", 0.40),  # candidate wins at the shared budget
        record("Baseline", "steps=200", 0.20),    # ... and must not be compared to this
    ]
    table = scores_by_model(records, "forecast", "mse")
    assert len(table["Baseline"]) == 2, "the two budgets must occupy separate cells"

    wins, shared = head_to_head(table["Candidate"], table["Baseline"])
    assert (wins, shared) == (1, 1), f"compared across budgets: {wins}/{shared}"
