"""Tests for the value-space Flow-SSM models.

These guard the properties that motivated moving the flow out of token space
(docs/00_project_status.md section 2): the model must produce a forecast in
physical units, the regression target must not depend on trainable parameters,
and the two tasks must differ only in metadata.
"""

import pytest
import torch

from rooster.models.flow_ssm import FlowSSMForecaster, FlowSSMReconstructor, choose_patch_len

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _forecaster(lookback=32, horizon=16, n_variates=2, **kwargs):
    return FlowSSMForecaster(lookback, horizon, n_variates, d_model=16, n_sampling_steps=3, **kwargs).to(DEVICE)


def test_patch_length_divides_both_window_lengths():
    assert choose_patch_len(96, 96) == 16  # ETT: the preferred size fits
    assert choose_patch_len(36, 24) == 12  # ILI: 16 divides neither
    assert choose_patch_len(96, 12) == 12  # PEMS: horizon is the binding constraint
    assert choose_patch_len(96, 720) == 16
    assert choose_patch_len(7, 5) == 1  # coprime: fall back rather than truncate


@pytest.mark.parametrize(("lookback", "horizon"), [(96, 96), (36, 24), (96, 12), (32, 16)])
def test_no_samples_are_dropped_by_patching(lookback, horizon):
    """A ragged tail would silently remove real data from the evaluation."""
    patch_len = choose_patch_len(lookback, horizon)
    assert lookback % patch_len == 0
    assert horizon % patch_len == 0


def test_forecaster_sample_and_forward_shapes():
    model = _forecaster()
    x = torch.randn(4, 32, 2, device=DEVICE)
    samples = model.sample(x, n_samples=5)
    assert samples.shape == (5, 4, 16, 2)
    assert torch.isfinite(samples).all()
    assert model(x).shape == (4, 16, 2)


def test_reconstructor_sample_shapes():
    model = FlowSSMReconstructor(64, d_model=16, n_sampling_steps=3).to(DEVICE)
    x = torch.randn(4, 64, device=DEVICE)
    samples = model.sample(x, n_samples=3)
    assert samples.shape == (3, 4, 64)
    assert torch.isfinite(samples).all()


def test_the_ensemble_actually_has_spread():
    """A 'probabilistic' model whose samples coincide is a point model in disguise."""
    model = _forecaster()
    x = torch.randn(4, 32, 2, device=DEVICE)
    samples = model.sample(x, n_samples=8)
    assert samples.std(dim=0).mean() > 1e-4


def test_output_is_in_the_condition_windows_physical_units():
    """The whole point of Option A: generation lands in physical units.

    A condition window shifted by +1000 and scaled by 50 must produce samples in
    that same range, because denormalization uses the condition's statistics.
    """
    model = _forecaster()
    base = torch.randn(6, 32, 2, device=DEVICE)
    shifted = base * 50.0 + 1000.0

    samples = model.sample(shifted, n_samples=4)
    assert samples.mean().abs() > 100.0  # nowhere near the normalized space's ~0
    assert 500.0 < samples.mean() < 1500.0
    # And the un-shifted input must NOT land there -- otherwise the offset is
    # coming from the weights rather than from the data.
    assert model.sample(base, n_samples=4).mean().abs() < 100.0


def test_denormalization_is_equivariant_to_an_affine_change_of_input():
    """Scaling and shifting the input must scale and shift the output identically.

    This is the property that makes the RevIN round-trip exact; it fails if the
    statistics are taken from the wrong window or applied in the wrong order.
    """
    model = _forecaster()
    torch.manual_seed(0)
    x = torch.randn(5, 32, 2, device=DEVICE)

    torch.manual_seed(123)
    plain = model.sample(x, n_samples=3)
    torch.manual_seed(123)
    transformed = model.sample(x * 7.0 - 3.0, n_samples=3)

    torch.testing.assert_close(transformed, plain * 7.0 - 3.0, rtol=1e-3, atol=1e-3)


def test_loss_is_finite_and_reaches_every_parameter():
    model = _forecaster()
    x = torch.randn(4, 32, 2, device=DEVICE)
    y = torch.randn(4, 16, 2, device=DEVICE)

    loss = model.compute_loss(x, y)
    assert torch.isfinite(loss)
    loss.backward()

    without_gradient = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    # The tokenizer's value path is only exercised on the condition side, so its
    # patch embedding does receive gradient; nothing should be orphaned.
    assert without_gradient == [], f"parameters never reached by the loss: {without_gradient}"


def test_training_target_does_not_depend_on_model_parameters():
    """x1 must be a fixed function of the data.

    In the token-space variants x1 came from the trainable tokenizer, so it
    drifted every step and needed a stop-gradient. Here, perturbing the weights
    must leave the target untouched -- checked by confirming the loss changes
    only through the prediction, i.e. two models with different weights see the
    same target for the same batch.
    """
    torch.manual_seed(0)
    model = _forecaster()
    x = torch.randn(4, 32, 2, device=DEVICE)
    y = torch.randn(4, 16, 2, device=DEVICE)

    condition_flat, target_flat = model._flatten_inputs(x, y)
    mean, std = model._normalize(condition_flat)
    target_before = model._patch((target_flat - mean) / std, model.n_target_patches).clone()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter))

    condition_flat, target_flat = model._flatten_inputs(x, y)
    mean, std = model._normalize(condition_flat)
    target_after = model._patch((target_flat - mean) / std, model.n_target_patches)
    torch.testing.assert_close(target_before, target_after)


def test_the_two_tasks_differ_only_in_metadata():
    """Forecast targets sit after the condition; reconstruction targets overlap it."""
    forecaster = _forecaster(lookback=32, horizon=32)
    reconstructor = FlowSSMReconstructor(32, d_model=16, n_sampling_steps=3).to(DEVICE)

    forecast_target_t = forecaster._target_meta(2, DEVICE)["t"]
    condition_t = forecaster._condition_meta(2, DEVICE)["t"]
    assert float(forecast_target_t.min()) >= float(condition_t.max())

    recon_target_t = reconstructor._target_meta(2, DEVICE)["t"]
    recon_condition_t = reconstructor._condition_meta(2, DEVICE)["t"]
    torch.testing.assert_close(recon_target_t, recon_condition_t)

    # ... and the channel id changes for reconstruction but not for forecasting.
    assert forecaster._target_meta(2, DEVICE)["channel_id"].unique().tolist() == condition_t.new_zeros(1).long().tolist()
    assert reconstructor._target_meta(2, DEVICE)["channel_id"].unique().tolist() != recon_condition_t.new_zeros(1).long().tolist()


def test_models_are_registered_for_the_leaderboard():
    from rooster.models.baselines import BENCHMARK_MODELS, build_model

    assert {"FlowSSM", "FlowSSMRecon"} <= set(BENCHMARK_MODELS)
    assert build_model("FlowSSM", lookback=32, horizon=16, n_variates=1).is_probabilistic


def test_flow_time_embedding_actually_distinguishes_flow_times():
    """Guard against the collapse that made FlowSSM worse than persistence.

    The sinusoidal ladder spans 1 .. 1/10000, so an input confined to [0, 1]
    leaves every frequency below 1 rad and the embedding becomes nearly constant.
    This asserts the rescaling is present.
    """
    from rooster.models.common import flow_time_embedding

    embeddings = flow_time_embedding(torch.linspace(0, 1, 16), 64)
    similarity = torch.nn.functional.cosine_similarity(embeddings[:1], embeddings[-1:]).item()
    assert similarity < 0.5, f"t=0 and t=1 embeddings are nearly identical (cos={similarity:.3f})"
    assert embeddings.std(dim=0).mean() > 0.2


def test_flow_time_embedding_is_monotone_in_information_not_degenerate():
    """Neighbouring flow times must stay more similar than distant ones."""
    from rooster.models.common import flow_time_embedding

    embeddings = flow_time_embedding(torch.linspace(0, 1, 16), 64)
    near = torch.nn.functional.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
    far = torch.nn.functional.cosine_similarity(embeddings[0:1], embeddings[-1:]).item()
    assert near > far


def test_a_constant_condition_window_does_not_explode_the_target():
    """1.8% of ETTm2's (window, variate) pairs are exactly flat.

    With a 1e-5 std floor those windows amplified the flow-matching target by up
    to 10^5 and drove the loss to 7e8. The floor must keep the normalized target
    at the same order of magnitude as the data.
    """
    model = _forecaster(lookback=32, horizon=16, n_variates=1)
    flat_condition = torch.full((3, 32, 1), 2.5, device=DEVICE)
    target = torch.randn(3, 16, 1, device=DEVICE)

    condition_flat, target_flat = model._flatten_inputs(flat_condition, target)
    mean, std = model._normalize(condition_flat)
    normalized_target = (target_flat - mean) / std

    assert float(std.min()) >= model.min_std
    assert float(normalized_target.abs().max()) < 100.0
    assert torch.isfinite(model.compute_loss(flat_condition, target))


def test_the_flow_matching_loss_stays_order_one_on_standardized_data():
    """A velocity regression between unit-scale endpoints cannot legitimately
    produce a loss in the thousands; if it does, something upstream is
    amplifying the target."""
    torch.manual_seed(0)
    model = _forecaster(lookback=96, horizon=96, n_variates=3)
    x = torch.randn(16, 96, 3, device=DEVICE)
    y = torch.randn(16, 96, 3, device=DEVICE)
    assert model.compute_loss(x, y).item() < 100.0


def test_ensemble_chunking_does_not_change_the_result():
    """The chunk size is a memory knob; it must not affect what is sampled."""
    model = _forecaster(lookback=32, horizon=16, n_variates=2)
    x = torch.randn(6, 32, 2, device=DEVICE)

    torch.manual_seed(7)
    model.max_ensemble_sequences = 10_000_000  # one chunk
    whole = model.sample(x, n_samples=6)
    torch.manual_seed(7)
    model.max_ensemble_sequences = 12  # forces several chunks
    chunked = model.sample(x, n_samples=6)

    assert whole.shape == chunked.shape
    # Chunking changes the order noise is drawn in, so samples are not identical
    # element-wise; the ensemble statistics must still agree.
    assert abs(float(whole.mean()) - float(chunked.mean())) < 0.3
    assert abs(float(whole.std()) - float(chunked.std())) < 0.3


def test_a_single_sample_still_works_when_chunking_is_tight():
    model = _forecaster()
    model.max_ensemble_sequences = 1
    x = torch.randn(3, 32, 2, device=DEVICE)
    assert model.sample(x, n_samples=2).shape == (2, 3, 16, 2)


def test_persistence_source_starts_from_the_last_observed_value():
    """docs/01_ideas.md 3.3: transport from a base forecast, not from noise.

    With the noise scale at zero the starting point must be exactly the last
    observed value, normalized -- that is what makes the degenerate solution the
    base forecast rather than a divergence.
    """
    model = _forecaster(source="persistence", source_noise=0.0)
    x = torch.randn(4, 32, 2, device=DEVICE)
    condition_flat = model._flatten_condition(x)
    mean, std = model._normalize(condition_flat)

    x0 = model._source_sample(condition_flat, mean, std)
    expected = ((condition_flat[:, -1] - mean.squeeze(-1)) / std.squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
    torch.testing.assert_close(x0, expected.expand_as(x0))


def test_noise_source_is_the_default_and_is_not_the_persistence_source():
    model_noise = _forecaster(source="noise")
    model_persist = _forecaster(source="persistence", source_noise=0.0)
    assert model_noise.source == "noise"

    x = torch.randn(4, 32, 2, device=DEVICE) + 10.0
    condition_flat = model_noise._flatten_condition(x)
    mean, std = model_noise._normalize(condition_flat)
    torch.manual_seed(0)
    from_noise = model_noise._source_sample(condition_flat, mean, std)
    from_persistence = model_persist._source_sample(condition_flat, mean, std)
    assert not torch.allclose(from_noise, from_persistence)


def test_an_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="unknown flow source"):
        _forecaster(source="teleport")


def test_source_sample_repeats_match_the_ensemble_layout():
    model = _forecaster(source="persistence", source_noise=0.0)
    x = torch.randn(5, 32, 2, device=DEVICE)
    condition_flat = model._flatten_condition(x)
    mean, std = model._normalize(condition_flat)
    x0 = model._source_sample(condition_flat, mean, std, n_repeats=3)
    assert x0.shape[0] == 3 * condition_flat.shape[0]
    # Each repeat must carry the same per-window base, not a reshuffled one.
    blocks = x0.chunk(3, dim=0)
    torch.testing.assert_close(blocks[0], blocks[1])


def test_cross_variate_context_is_off_by_default():
    assert _forecaster().variate_mixer is None


def test_cross_variate_context_mixes_across_variates_not_across_windows():
    """The fold is index = b * C + c, so the regrouping must recover exactly that.

    If the reshape were wrong the model would average unrelated windows together,
    which would look like it was working while leaking across the batch.
    """
    from rooster.models.flow_ssm import VariateContext

    torch.manual_seed(0)
    mixer = VariateContext(4).to(DEVICE)
    n_windows, n_variates, n_patches = 3, 5, 2
    # Give every (window, variate) a distinguishable constant.
    context = torch.arange(n_windows * n_variates, dtype=torch.float32, device=DEVICE)
    context = context.view(-1, 1, 1).expand(-1, n_patches, 4).contiguous()

    with torch.no_grad():
        mixed = mixer(context, n_variates)
    assert mixed.shape == context.shape

    # Windows must stay separable: the mean injected into window 0 must differ
    # from the one injected into window 2.
    grouped = mixed.reshape(n_windows, n_variates, n_patches, 4)
    assert not torch.allclose(grouped[0].mean(), grouped[2].mean())


def test_cross_variate_candidate_changes_the_forecast():
    """A pathway that is present but inert would be a silent no-op."""
    torch.manual_seed(0)
    plain = _forecaster(n_variates=4)
    torch.manual_seed(0)
    mixed = _forecaster(n_variates=4, cross_variate=True)

    x = torch.randn(6, 32, 4, device=DEVICE)
    torch.manual_seed(1)
    a = plain.sample(x, n_samples=2)
    torch.manual_seed(1)
    b = mixed.sample(x, n_samples=2)
    assert not torch.allclose(a, b, atol=1e-5)



def test_point_head_is_the_declared_point_forecast_not_the_ensemble_mean():
    """With a point head, forward() must return the head, not a sample mean.

    The ensemble mean is only a Monte-Carlo estimate of the MSE-optimal
    predictor; the head is trained for it directly. Measured, that distinction is
    the whole MSE/MAE gap against DecompSSM.
    """
    torch.manual_seed(0)
    model = _forecaster(point_head=True)
    x = torch.randn(6, 32, 2, device=DEVICE)

    with torch.no_grad():
        declared = model(x)
        ensemble_mean = model.sample(x, n_samples=8).mean(dim=0)
    assert declared.shape == (6, 16, 2)
    assert not torch.allclose(declared, ensemble_mean, atol=1e-4)


def test_point_head_is_deterministic_in_eval_mode():
    """A point forecast that changes between calls is not a point forecast.

    Only in eval mode: the condition encoder carries dropout, so in train mode
    the forecast is legitimately stochastic. The harness calls `eval()` before
    scoring, which is what makes the reported number reproducible.
    """
    model = _forecaster(point_head=True).eval()
    x = torch.randn(4, 32, 2, device=DEVICE)
    with torch.no_grad():
        torch.manual_seed(1)
        first = model(x)
        torch.manual_seed(999)
        second = model(x)
    torch.testing.assert_close(first, second)


def test_the_point_head_is_supervised_by_the_loss():
    model = _forecaster(point_head=True)
    x = torch.randn(4, 32, 2, device=DEVICE)
    y = torch.randn(4, 16, 2, device=DEVICE)
    model.compute_loss(x, y).backward()
    assert model.point_head.weight.grad is not None
    assert model.point_head.weight.grad.abs().sum() > 0


def test_point_source_starts_the_flow_from_the_point_forecast():
    model = _forecaster(point_head=True, source="point", source_noise=0.0)
    x = torch.randn(4, 32, 2, device=DEVICE)
    condition_flat = model._flatten_condition(x)
    mean, std = model._normalize(condition_flat)
    with torch.no_grad():
        context = model._build_context(condition_flat, mean, std)
        point = model._point_forecast(context)
        x0 = model._source_sample(condition_flat, mean, std, point=point)
    torch.testing.assert_close(x0, point)


def test_point_source_without_a_head_is_rejected():
    with pytest.raises(ValueError, match="requires point_head"):
        _forecaster(source="point")


def test_point_head_output_lands_in_physical_units():
    model = _forecaster(point_head=True)
    base = torch.randn(6, 32, 2, device=DEVICE)
    with torch.no_grad():
        shifted = model(base * 50.0 + 1000.0)
        plain = model(base)
    assert 500.0 < shifted.mean() < 1500.0
    assert plain.mean().abs() < 100.0


def test_the_flow_cannot_move_the_point_forecast_through_the_source():
    """x0 is detached, so the flow cannot make its own transport easier by
    dragging the point forecast toward the noise it prefers."""
    model = _forecaster(point_head=True, source="point")
    x = torch.randn(4, 32, 2, device=DEVICE)
    condition_flat = model._flatten_condition(x)
    mean, std = model._normalize(condition_flat)
    context = model._build_context(condition_flat, mean, std)
    point = model._point_forecast(context)
    x0 = model._source_sample(condition_flat, mean, std, point=point)
    assert not x0.requires_grad


def test_the_direct_point_term_is_on_by_default_and_ablatable():
    assert _forecaster(point_head=True).direct_point is not None
    assert _forecaster(point_head=True, direct_point=False).direct_point is None
    assert _forecaster().direct_point is None  # no point head at all


def test_the_direct_term_changes_the_point_forecast():
    """A whole-window linear map that made no difference would be dead weight."""
    torch.manual_seed(0)
    with_direct = _forecaster(point_head=True)
    torch.manual_seed(0)
    token_only = _forecaster(point_head=True, direct_point=False)

    x = torch.randn(5, 32, 2, device=DEVICE)
    with torch.no_grad():
        assert not torch.allclose(with_direct.eval()(x), token_only.eval()(x), atol=1e-5)


def test_the_direct_term_is_supervised():
    model = _forecaster(point_head=True)
    x = torch.randn(4, 32, 2, device=DEVICE)
    y = torch.randn(4, 16, 2, device=DEVICE)
    model.compute_loss(x, y).backward()
    assert model.direct_point.weight.grad is not None
    assert model.direct_point.weight.grad.abs().sum() > 0


def test_the_direct_term_maps_the_whole_window_to_the_whole_horizon():
    """Its shape is the property that distinguishes it from the token-wise head:
    every input step can influence every output step."""
    model = _forecaster(lookback=96, horizon=48, point_head=True)
    assert model.direct_point.weight.shape == (48, 96)
