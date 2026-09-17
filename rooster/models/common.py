"""Shared building blocks for both flow variants: the condition encoder,
the flow-matching vector field network, the flow-matching training step,
the shared token->waveform decoder, and the Euler-integration sampler used
at generation time.

Both variants' training_step now build condition/target token streams via
UnifiedTokenizer and align condition context onto target timestamps via
TargetTimeAligner (tokenizer.py) -- see each variant module for the wiring.
The target-side embedding used as the flow-matching regression target (x1)
is always built under torch.no_grad(): letting gradients flow into x1 would
let the tokenizer cheat by collapsing the target embedding space to
trivially minimize the flow-matching loss, since nothing else in these
variants (Variant A especially) forces that space to stay non-degenerate.

Forecasting vs. reconstruction: nothing in this module (or in either
variant's training_step/generate) branches on task. Task identity lives
entirely in the batch dicts fed in -- which channel/modality/task ids and
which relative-time convention were used to build them (see
ppg_vitals_data.make_reconstruction_batch vs. real_data_smoke_test.
make_forecast_batch). The same ConditionSSM, VectorFieldNet, and
euler_sample serve both.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from s5 import S5

# NOTE ON MIXED PRECISION, so nobody spends an afternoon rediscovering this:
# `s5-pytorch` implements its associative scan with `vmap`, and
# `aten::_autocast_to_full_precision` has no batching rule, so *any* autocast
# context raises RuntimeError inside the scan. Excluding just the S5 call and
# autocasting everything around it does work, but was measured at 0.92-1.03x --
# i.e. no gain, sometimes a loss, because S5 dominates the runtime. bf16 is
# therefore not wired up. Revisit only if the scan implementation changes.


class SimpleSSMBlock(nn.Module):
    """Real S5 layer (Task 2), ported from DecompSSM's GTSSM
    (DecompSSM/layers/DecompSSM.py:67-287), which wraps the external
    s5-pytorch package this module now depends on. Gated the same way
    GTSSM is: an input-dependent step-size (rescales the SSM's
    discretization step, via a pooled MLP + softplus + clamp) and a
    sigmoid frequency gate multiplying the S5 output elementwise.

    Simplification vs. GTSSM: component-type-specific gate activation
    (Tanh/GELU/ReLU for trend/seasonal/residual) is not wired in here --
    the gate always uses GELU -- to keep this swap a single-purpose,
    reviewable diff. Interface locked: (B, L, d_model) -> (B, L, d_model).
    """

    def __init__(self, d_model, state_size=64, dt_min=0.001, dt_max=0.1, bidir=True, block_count=1, dropout=0.1):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.ssm = S5(
            width=d_model,
            state_width=state_size,
            dt_min=dt_min,
            dt_max=dt_max,
            bidir=bidir,
            block_count=block_count,
            bcInit="dense",
        )
        hidden = max(d_model // 2, 1)
        self.step_proj = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.freq_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x_norm = self.norm_in(x)
        step_scale = self.step_proj(x_norm).mean(dim=1).squeeze(-1)  # (B,)
        step_scale = F.softplus(step_scale).clamp(0.01, 2.0)
        ssm_out = self.ssm(x_norm, step_scale=step_scale)
        gated = ssm_out * self.freq_gate(x_norm)
        return self.norm_out(residual + self.dropout(gated))


class ConditionSSM(nn.Module):
    """Stack of SimpleSSMBlock layers encoding the condition-side token
    stream."""

    def __init__(self, d_model, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([SimpleSSMBlock(d_model) for _ in range(n_layers)])

    def forward(self, condition_tokens):
        x = condition_tokens
        for layer in self.layers:
            x = layer(x)
        return x


# The sinusoidal frequency ladder spans 1 down to 1/10000, so it only resolves an
# input spread over ~10^4. Flow time lives in [0, 1], so it must be rescaled first
# -- the diffusion literature's convention of treating t as a step index in
# [0, 1000] is exactly this rescaling.
#
# Without it, every frequency in the ladder sees an argument below 1 rad, sin is
# ~linear and cos is ~1, and the whole embedding collapses: the cosine similarity
# between the t=0 and t=1 embeddings is 0.967, i.e. the vector field is nearly
# blind to flow time and can only learn a time-averaged velocity. Measured, not
# hypothesised -- it is what made FlowSSM score worse than persistence.
FLOW_TIME_SCALE = 1000.0


def flow_time_embedding(t, dim):
    """Standard sinusoidal embedding of the scalar flow-matching time t in [0,1].

    Distinct from tokenizer.ContinuousTimeEncoding, which encodes the signal's
    own continuous time axis rather than the flow's interpolation parameter.
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
    args = (t * FLOW_TIME_SCALE) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class VectorFieldNet(nn.Module):
    """Predicts flow-matching velocity given the noised target x_t, scalar
    flow time t, and a conditioning context already aligned to the target's
    sequence length. Context is fused additively into x_t (mirrors
    PENGUIN's additive PPG-stream fusion); each SimpleSSMBlock layer is then
    FiLM-modulated by the flow-time embedding (mirrors PENGUIN's adaLN-style
    timestep conditioning)."""

    def __init__(self, d_model, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([SimpleSSMBlock(d_model) for _ in range(n_layers)])
        self.film = nn.ModuleList([nn.Linear(d_model, 2 * d_model) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x_t, t, context):
        x = x_t + context
        t_emb = flow_time_embedding(t, self.d_model)
        for layer, film in zip(self.layers, self.film, strict=True):
            scale, shift = film(t_emb).chunk(2, dim=-1)
            x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            x = layer(x)
        return self.out_proj(x)


class UnpatchDecoder(nn.Module):
    """Inverse of tokenizer.PatchEmbed: projects each d_model token back to a
    patch of patch_len raw samples, then flattens patches into one
    continuous waveform. Shared by both variants (moved here from
    variant_stft_loss.py, Task 3) since it's a generic token->waveform
    transform with no variant-specific logic -- both training-time
    (VariantSTFT's STFT loss) and inference-time (generate(), below) uses
    need the same decoder."""

    def __init__(self, d_model, patch_len):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(d_model, patch_len)

    def forward(self, tokens):
        batch_size, n_patches, _ = tokens.shape
        patches = self.proj(tokens)  # (B, N, patch_len)
        return patches.reshape(batch_size, n_patches * self.patch_len)


def decode_to_waveform(tokens, decoder):
    """(B, N, d_model) -> (B, N * patch_len) continuous waveform."""
    return decoder(tokens)


@torch.no_grad()
def euler_sample(vector_field_fn, shape, context, n_steps=50, device=None, generator=None, x0=None, mode=None):
    """Fixed-step Euler ODE integration of a trained flow-matching vector
    field, from x0 ~ N(0,I) at flow-time t=0 to an estimate of x1 at t=1.
    This is the inference-time counterpart of flow_matching_step's training
    interpolation: training regresses vector_field_fn(x_t, t, context)
    against x1 - x0 along the *straight-line* path from x0 to x1, so
    integrating that learned velocity field forward from a fresh x0 recovers
    an estimate of x1 conditioned on `context`.

    Task-agnostic: nothing here knows or cares whether `context` came from a
    forecasting or reconstruction condition batch -- that distinction lives
    entirely in how `context` was built (see generate() on each variant).
    """
    device = device or context.device
    # `x0` may be supplied when several coupled flows must share one noise draw
    # (see VariantLaplacian.generate: its bands are linear functions of the same
    # target, so independent draws would not sum to a valid sample).
    x = torch.randn(shape, device=device, generator=generator) if x0 is None else x0
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((shape[0],), step * dt, device=device)
        # `mode` is held fixed for the whole trajectory when the field is a
        # mixture. Re-drawing it per step would average the modes back together
        # along the path and give exactly the blur the mixture exists to avoid.
        velocity = vector_field_fn(x, t, context) if mode is None else vector_field_fn(x, t, context, mode=mode)
        x = x + velocity * dt
    return x


@torch.no_grad()
def heun_sample(vector_field_fn, shape, context, n_steps=25, device=None, generator=None, x0=None, mode=None):
    """Heun (2nd-order) integration of the learned velocity field.

    One extra function evaluation per step buys a much smaller discretisation
    error than Euler at the same step count, which matters when the step budget
    is the thing being reported. PENGUIN (arXiv:2602.03858) uses Heun with 25
    steps, so a faithful baseline needs it.

    Cost is 2 NFE per step: `heun_sample(..., n_steps=25)` is 50 evaluations,
    versus `euler_sample(..., n_steps=50)`'s 50. Compare at matched NFE, not at
    matched step count.
    """
    device = device or context.device
    x = torch.randn(shape, device=device, generator=generator) if x0 is None else x0
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((shape[0],), step * dt, device=device)
        t_next = torch.full((shape[0],), (step + 1) * dt, device=device)
        velocity = (vector_field_fn(x, t, context) if mode is None else vector_field_fn(x, t, context, mode=mode))
        predicted = x + velocity * dt
        velocity_next = vector_field_fn(predicted, t_next, context)
        x = x + 0.5 * dt * (velocity + velocity_next)
    return x


def flow_matching_step(vector_field_fn, x1, context, device=None, x0=None, t=None):
    """One conditional-flow-matching training step (rectified-flow / OT-CFM
    style, matching PENGUIN's train_flow): sample t ~ U(0,1) and
    x0 ~ N(0,I), interpolate x_t = (1-t)*x0 + t*x1, and regress
    vector_field_fn(x_t, t, context) against the target velocity x1 - x0.

    Returns (loss, aux); aux carries x_t/t/pred_v/target_v/x0 for callers
    that need them (e.g. reconstructing an x1 estimate for a downstream
    waveform loss).
    """
    device = device or x1.device
    batch_size = x1.shape[0]
    # `t` and `x0` may be supplied so that several coupled flows are trained on
    # the SAME interpolation path -- required whenever their outputs are summed.
    t = torch.rand(batch_size, device=device) if t is None else t
    x0 = torch.randn_like(x1) if x0 is None else x0
    t_broadcast = t.view(batch_size, *([1] * (x1.dim() - 1)))
    x_t = (1 - t_broadcast) * x0 + t_broadcast * x1
    target_v = x1 - x0
    pred_v = vector_field_fn(x_t, t, context)
    loss = F.mse_loss(pred_v, target_v)
    aux = {"x_t": x_t, "t": t_broadcast, "pred_v": pred_v, "target_v": target_v, "x0": x0}
    return loss, aux
