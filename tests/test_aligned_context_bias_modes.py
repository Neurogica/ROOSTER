"""The bias-mode ablation arms must differ ONLY in the bias's form.

Two properties make the ablation table readable: every mode produces the same
shapes through the same forward, and the zero-init modes ("free", and "none" by
construction) start exactly at the content-only attention, so any divergence in
the table is learned, not initial.
"""

import torch

from rooster.models.aligned_context import BIAS_MODES, RelativeAlignedContext
from rooster.models.decomp_dict import DecompDict


def test_every_mode_forwards_with_matching_shapes():
    torch.manual_seed(0)
    condition = torch.randn(2, 12, 32)
    queries = torch.randn(2, 9, 32)
    for mode in BIAS_MODES:
        module = RelativeAlignedContext(32, max_offset=16, bias_mode=mode)
        out, weights = module(condition, queries)
        assert out.shape == (2, 9, 32), mode
        assert weights.shape == (2, module.n_heads, 9, 12), mode
        assert torch.isfinite(out).all(), mode


def test_free_mode_starts_as_content_only_attention():
    torch.manual_seed(0)
    condition = torch.randn(2, 12, 32)
    queries = torch.randn(2, 9, 32)
    none = RelativeAlignedContext(32, max_offset=16, bias_mode="none")
    free = RelativeAlignedContext(32, max_offset=16, bias_mode="free")
    free.load_state_dict(none.state_dict(), strict=False)
    _, w_none = none(condition, queries)
    _, w_free = free(condition, queries)
    assert torch.allclose(w_none, w_free)


def test_frozen_comb_keeps_bias_parameters_out_of_the_gradient():
    module = RelativeAlignedContext(32, max_offset=16, bias_mode="comb", learn_bias=False)
    assert not module.offset_centre.requires_grad
    assert not module.log_period.requires_grad
    assert not module.log_sharpness.requires_grad
    # The attention weights themselves still train.
    assert module.query.weight.requires_grad


def test_diagnostics_attributes_exist_only_where_meaningful():
    comb = RelativeAlignedContext(32, max_offset=16, bias_mode="comb")
    assert hasattr(comb, "offset_centre")
    for mode in ("free", "none"):
        module = RelativeAlignedContext(32, max_offset=16, bias_mode=mode)
        # The harness keys align_* diagnostics on these attributes; a mode with no
        # comb must not present half-meaningful ones.
        assert not hasattr(module, "offset_centre"), mode
        assert not hasattr(module, "log_period"), mode


def test_decomp_dict_threads_the_mode_through():
    model = DecompDict(96, 96, 7, n_atoms=3, relative_align=True, align_bias_mode="bump")
    assert model.relative_align.bias_mode == "bump"
    x = torch.randn(2, 96, 7)
    out = model(x)
    forecast = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(forecast).all()
