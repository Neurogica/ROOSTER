"""Tests for DecompDict: DecompSSM with a learned component count.

The point of this model is that it differs from DecompSSM in exactly one way, so
the tests check that -- that the shared machinery is genuinely DecompSSM's, and
that the one thing that changed behaves as claimed.
"""

import pytest
import torch

from rooster.models.decomp_dict import BRANCH_STEP_SPAN, BranchGate, DecompDict, _step_priors
from rooster.models.vendor.decompssm_official import GTSSM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model(n_train_samples=240_000, **kwargs):
    settings = dict(lookback=96, horizon=96, n_variates=7, n_atoms=8, d_model=64, state_size=32)
    settings.update(kwargs)
    model = DecompDict(**settings).to(DEVICE)
    # The harness supplies this; the default `sparsity_weight="mdl"` refuses to
    # invent it, so the tests have to stand in for the harness. 240k is ETTm1's
    # (windows x variates), which keeps the derived coefficient realistic.
    model.n_train_samples = n_train_samples
    return model


def test_the_branches_are_the_vendored_decompssm_branch():
    """If this ever stops being GTSSM, the comparison stops being single-axis."""
    model = _model()
    assert all(isinstance(branch, GTSSM) for branch in model.branches)


def test_the_atom_pool_spans_decompssms_own_frequency_range():
    priors = _step_priors(8)
    assert priors[0][0] == pytest.approx(BRANCH_STEP_SPAN[0])
    assert priors[-1][1] == pytest.approx(BRANCH_STEP_SPAN[1])
    # Contiguous and ascending: no gap in the covered dynamics.
    for (_, high), (low, _) in zip(priors, priors[1:], strict=False):
        assert high == pytest.approx(low)


def test_three_atoms_reproduce_decompssms_own_priors():
    """A pool of three should land on the same span the three named branches use,
    so the ablation reduces to the original at K=3."""
    priors = _step_priors(3)
    assert len(priors) == 3
    assert priors[0][0] == pytest.approx(0.0002)
    assert priors[-1][1] == pytest.approx(0.2)


def test_the_gate_switches_branches_off():
    gate = BranchGate(4, 8, threshold_init=0.999).eval().to(DEVICE)  # above any score
    _usage, open_gates = gate(torch.randn(3, 5, 8, device=DEVICE))
    assert open_gates.sum() == 0


def test_the_gate_threshold_receives_gradient_on_ordinary_inputs():
    """The failure this guards against is silent: with a softplus score against a
    small positive threshold, every score sits outside the straight-through
    bandwidth, the ramp saturates, and the threshold never moves -- so the
    component count would be whatever the initialisation gave, not something
    learned. Ordinary random inputs, not hand-picked ones near the threshold."""
    torch.manual_seed(0)
    gate = BranchGate(4, 8).train().to(DEVICE)
    usage, _open = gate(torch.randn(32, 5, 8, device=DEVICE))
    usage.sum().backward()
    assert gate.threshold_logit.grad is not None
    assert gate.threshold_logit.grad.abs().sum() > 0


def test_score_and_threshold_share_the_same_bounded_range():
    """Which is what keeps the threshold inside the score distribution."""
    gate = BranchGate(4, 8).eval().to(DEVICE)
    usage, _open = gate(torch.randn(64, 5, 8, device=DEVICE) * 10.0)
    assert (usage >= 0).all() and (usage <= 1).all()
    assert (gate.threshold > 0).all() and (gate.threshold < 1).all()


def test_a_descent_step_on_the_sparsity_penalty_closes_gates():
    """The mechanism, tested as an optimiser step rather than a raw gradient sign.

    Asserting the sign directly is easy to get backwards -- descent moves
    *against* the gradient, so a penalty that closes gates produces a negative
    gradient on the threshold logit and a positive change in the threshold. What
    matters is the direction of the update, so that is what is checked.

    Tested as one deterministic step rather than end-to-end: an earlier version
    trained 40 steps on random data and measured noise, reporting counts of 3.21,
    2.25 and 4.21 for penalties of 0, 1 and 5. The benchmark measures the
    end-to-end effect; a unit test should pin the mechanism.
    """
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(16, 96, 7, device=DEVICE)
    before = model.gate.threshold.detach().clone()

    optimizer = torch.optim.SGD(model.gate.parameters(), lr=1.0)
    _embedded, _components, gate, _mean, _std = model._encode(x)
    optimizer.zero_grad()
    gate.sum(dim=-1).mean().backward()
    optimizer.step()

    assert (model.gate.threshold > before).all(), "a sparsity step did not raise the thresholds"


def test_a_closed_gate_reduces_the_measured_count():
    """The reported count must follow the thresholds, or it measures nothing."""
    model = _model().eval()
    x = torch.randn(8, 96, 7, device=DEVICE)
    before = model.cardinality(x)["per_window"]
    with torch.no_grad():
        model.gate.threshold_logit.add_(4.0)
    after = model.cardinality(x)["per_window"]
    assert after < before


def test_a_closed_gate_removes_the_branch_from_the_prediction():
    """A gate that only scaled the loss, without actually removing the component,
    would be reporting a count that means nothing."""
    torch.manual_seed(0)
    model = _model().eval()
    x = torch.randn(4, 96, 7, device=DEVICE)
    with torch.no_grad():
        before = model(x)
        model.gate.threshold_logit.fill_(10.0)  # shut every gate
        after = model(x)
    assert not torch.allclose(before, after)


def test_the_forecast_lands_in_the_inputs_physical_units():
    model = _model().eval()
    base = torch.randn(4, 96, 7, device=DEVICE)
    with torch.no_grad():
        shifted = model(base * 30.0 + 600.0)
        plain = model(base)
    assert 300.0 < shifted.mean() < 900.0
    assert plain.mean().abs() < 100.0


def test_the_objective_keeps_decompssms_auxiliary_pressure():
    """Reconstruction and orthogonality are DecompSSM's; dropping them would make
    this a different model rather than an ablation of one."""
    torch.manual_seed(0)
    x = torch.randn(4, 96, 7, device=DEVICE)
    y = torch.randn(4, 96, 7, device=DEVICE)

    torch.manual_seed(0)
    with_aux = _model(aux_loss_weight=0.1)
    torch.manual_seed(0)
    without_aux = _model(aux_loss_weight=0.0)
    assert with_aux.compute_loss(x, y).item() != pytest.approx(without_aux.compute_loss(x, y).item())


def test_every_parameter_is_reached_by_the_loss():
    model = _model()
    model.compute_loss(torch.randn(4, 96, 7, device=DEVICE), torch.randn(4, 96, 7, device=DEVICE)).backward()
    orphaned = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert orphaned == [], f"parameters never reached by the loss: {orphaned}"


def test_it_is_registered_and_reported_as_deterministic():
    from rooster.models.baselines import BENCHMARK_MODELS, build_model

    assert "DecompDict" in BENCHMARK_MODELS
    assert build_model("DecompDict", lookback=96, horizon=96, n_variates=3).is_probabilistic is False


def test_the_head_does_not_grow_with_the_pool_under_sum_mixing():
    """The property that makes "pool size is not a hyperparameter" claimable.

    Under concat the head takes n_atoms * d_model inputs, so a bigger pool is a
    bigger model and the pool becomes a capacity knob. Under sum it is
    Linear(d_model -> ...) whatever the pool, so a component can only earn its
    place by contributing signal.
    """
    small = _model(n_atoms=4, component_mixing="sum")
    large = _model(n_atoms=64, component_mixing="sum")
    assert small.output_projection[0].in_features == large.output_projection[0].in_features

    concat_small = _model(n_atoms=4, component_mixing="concat")
    concat_large = _model(n_atoms=64, component_mixing="concat")
    assert concat_large.output_projection[0].in_features > concat_small.output_projection[0].in_features


def test_sum_mixing_makes_every_component_reach_the_output():
    """Under sum, zeroing any single component must change the forecast -- if it
    did not, that component would be free and the count would be meaningless."""
    torch.manual_seed(0)
    model = _model(n_atoms=4, component_mixing="sum").eval()
    x = torch.randn(4, 96, 7, device=DEVICE)

    with torch.no_grad():
        embedded, components, _gate, mean, std = model._encode(x)
        full = model.output_projection(model._mix(components))
        for index in range(len(components)):
            dropped = list(components)
            dropped[index] = torch.zeros_like(dropped[index])
            assert not torch.allclose(full, model.output_projection(model._mix(dropped)))


def test_an_unknown_mixing_is_rejected():
    with pytest.raises(ValueError, match="unknown component_mixing"):
        _model(component_mixing="average")


def _batches(n_batches=4, batch_size=32, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        (
            torch.randn(batch_size, 96, 7, generator=generator).to(DEVICE),
            torch.randn(batch_size, 96, 7, generator=generator).to(DEVICE),
        )
        for _ in range(n_batches)
    ]


def _silence_branch(model, index):
    """Force one branch's gate shut, so it contributes exactly zero.

    `gate` is a hard 0/1 outside training, so a strongly negative score makes the
    branch's contribution identically zero rather than merely small -- which is
    what lets the test assert an exact result instead of a tolerance.
    """
    with torch.no_grad():
        model.gate.score.weight[index].zero_()
        model.gate.score.bias[index] = -20.0


def test_selection_takes_no_threshold_to_choose():
    """The whole point: there is no coefficient, so there is no argument for one.

    If a tolerance, weight or threshold ever appears in this signature, the count
    is being chosen by hand again under a new name.
    """
    import inspect

    parameters = set(inspect.signature(DecompDict.select_cardinality).parameters)
    assert parameters == {"self", "batches", "device"}, parameters


def test_a_component_that_contributes_nothing_is_dropped():
    """A silenced branch changes the validation error by exactly zero, so the
    criterion must remove it -- zero is not more than any standard error."""
    torch.manual_seed(0)
    model = _model(n_atoms=4).eval()
    _silence_branch(model, 2)
    result = model.select_cardinality(_batches())
    assert model.active_mask[2].item() == 0.0
    assert result["k_selected"] < 4


def test_selection_keeps_at_least_one_component():
    """Even if nothing looks significant, a decomposition with no parts is not a
    model. The loop stops at one rather than emptying the pool."""
    torch.manual_seed(0)
    model = _model(n_atoms=3).eval()
    for index in range(3):
        _silence_branch(model, index)
    result = model.select_cardinality(_batches())
    assert result["k_selected"] == 1
    assert model.active_mask.sum().item() == 1.0


def test_the_reported_count_is_the_count_the_model_then_uses():
    """`cardinality` has to agree with the selection, or the paper reports one
    number and evaluates another."""
    torch.manual_seed(0)
    model = _model(n_atoms=4).eval()
    _silence_branch(model, 1)
    result = model.select_cardinality(_batches())
    counts = model.cardinality(_batches(n_batches=1)[0][0])
    assert counts["per_dataset"] == result["k_selected"]


def test_the_mask_survives_a_checkpoint_round_trip():
    """It is a buffer for this reason: a reloaded model that forgot its selection
    would evaluate a different model from the one that was selected."""
    torch.manual_seed(0)
    model = _model(n_atoms=4)
    with torch.no_grad():
        model.active_mask[1] = 0.0
    restored = _model(n_atoms=4)
    restored.load_state_dict(model.state_dict())
    assert restored.active_mask.tolist() == [1.0, 0.0, 1.0, 1.0]


def test_the_mask_actually_removes_the_component_from_the_forecast():
    torch.manual_seed(0)
    model = _model(n_atoms=4).eval()
    x = _batches(n_batches=1)[0][0]
    with torch.no_grad():
        before = model(x)
        model.active_mask[0] = 0.0
        after = model(x)
    assert not torch.allclose(before, after)


def test_more_validation_windows_admit_at_least_as_many_components():
    """The adaptivity claim, as a property rather than an anecdote.

    The threshold is the standard error, which falls as 1/sqrt(n), so a larger
    validation set resolves smaller contributions and can only afford more
    components -- never fewer. This is why the selected K varies by dataset
    without anything being tuned per dataset.
    """
    torch.manual_seed(0)
    weights = _model(n_atoms=6).eval().state_dict()

    small = _model(n_atoms=6).eval()
    small.load_state_dict(weights)
    large = _model(n_atoms=6).eval()
    large.load_state_dict(weights)

    few = small.select_cardinality(_batches(n_batches=1, batch_size=8))
    many = large.select_cardinality(_batches(n_batches=8, batch_size=64))
    assert many["n_val_windows"] > few["n_val_windows"]
    assert many["k_selected"] >= few["k_selected"]


def test_the_mdl_coefficient_has_no_free_parameter_and_scales_with_n():
    """Kept as a measured negative result, not as the mechanism.

    BIC is the textbook way to price an extra component and it is genuinely
    knob-free, so it had to be tried. It fails here for a concrete reason: at
    ~43k parameters per branch the penalty dwarfs anything the data can pay, and
    the count collapses to one. The test pins both facts -- that the coefficient
    is determined by (architecture, N) alone, and that it is implausibly large at
    realistic N -- so the negative result cannot be quietly forgotten.
    """
    model = _model(n_atoms=8, sparsity_weight="mdl")
    model.n_train_samples = 240_000
    ettm1 = model.effective_sparsity_weight()
    model.n_train_samples = 2_400_000
    ten_times_more_data = model.effective_sparsity_weight()

    assert ten_times_more_data < ettm1  # more data pays for more components
    assert ettm1 > 0.1  # ... but at ETTm1's size it is far above the 0.01 that already forced K = 1

    model.n_train_samples = None
    with pytest.raises(RuntimeError, match="n_train_samples"):
        model.effective_sparsity_weight()
