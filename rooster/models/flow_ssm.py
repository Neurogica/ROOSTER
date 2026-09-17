"""Flow-SSM models that actually produce a scoreable forecast.

This is "Option A" from docs/00_project_status.md section 2: flow matching is
run in **RevIN-normalized patch-value space**, not in the tokenizer's LayerNorm'd
embedding space. The consequences are the point of the change:

* The regression target `x1` is a fixed function of the data with **no trainable
  parameters**, so there is no drifting target and no need for the `no_grad`
  workaround that `models/common.py` documents.
* There is an exact inverse back to physical units: undo the patching, then undo
  the normalization with the **condition window's** statistics. That is what
  RevIN is for (Kim et al., ICLR 2022) and what every LTSF model does -- the
  target's own statistics are unavailable at inference, and were never needed.
* The decoder stops being a separate, unsupervised module: the vector field's
  output projection *is* the decoder, trained end to end by the flow-matching
  loss. `VariantLaplacian`'s untrained-decoder bug and `VariantSTFT`'s
  collapse-to-zero STFT loss both become structurally impossible.

The condition path is unchanged from the existing variants -- `UnifiedTokenizer`
-> `ConditionSSM` -> `TargetTimeAligner` -- so the tokenizer, the real S5 layer
and the task-identity-by-metadata design are all preserved. Only the space the
flow lives in has moved.

Nothing here branches on task. `FlowSSMForecaster` and `FlowSSMReconstructor`
differ solely in the metadata they stamp onto their batches: the forecaster puts
the target *after* the condition in relative time and reuses the condition's
channel id; the reconstructor puts it *over* the condition and switches channel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rooster.models.aligned_context import RelativeAlignedContext
from rooster.models.baselines import BenchmarkModel, register_benchmark_model
from rooster.models.common import ConditionSSM, SimpleSSMBlock, euler_sample, flow_matching_step, flow_time_embedding
from rooster.models.quantile_head import DEFAULT_QUANTILE_LEVELS
from rooster.tokenizer import TargetTimeAligner, UnifiedTokenizer

# Where the flow starts. "noise" is textbook conditional flow matching;
# "persistence" transports from the last observed value instead -- see
# FlowSSM._source_sample.
SOURCES = ("noise", "persistence", "point")

# Vocabulary for the metadata the tokenizer consumes. Channel-independent
# forecasting folds the variate axis into the batch, so one channel id suffices
# on that side; reconstruction needs a second to mark the target modality.
CHANNEL_FORECAST, CHANNEL_PPG, CHANNEL_VITAL = 0, 1, 2
MODALITY_TIMESERIES, MODALITY_BIOSIGNAL = 0, 1
TASK_FORECAST, TASK_RECONSTRUCT = 0, 1


def choose_patch_len(*lengths, preferred=16):
    """Largest patch length up to `preferred` that divides every given length.

    Necessary because the benchmark's horizons are not all multiples of 16:
    ILI uses {24, 36, 48, 60} against a lookback of 36, and PEMS uses
    {12, 24, 48, 96}. Silently truncating the ragged tail would drop real
    samples from the evaluation, so the patch size adapts instead.
    """
    upper = min(preferred, *lengths)
    for candidate in range(upper, 0, -1):
        if all(length % candidate == 0 for length in lengths):
            return candidate
    return 1


class VariateContext(nn.Module):
    """Share one global context across the variates of a window.

    Channel-independent forecasting folds the variate axis into the batch, which
    means the model never sees cross-variable structure at all. Measured, that
    is the dominant weakness: on PEMS04 (307 correlated sensors) the reference
    FlowSSM scores MSE 0.60 against DecompSSM's 0.14, and the learning curves
    show both still improving at 20 000 steps -- so the gap is structural, not a
    training budget.

    The mechanism is DecompSSM's Global Context Refinement, which is cheap and
    already proven on this data: average across variates, project, and add back
    as a gated residual.

    The gate starts nearly **closed** (`alpha = -3`, so `sigmoid(alpha) = 0.05`)
    rather than half-open. Measured reason: with a half-open gate the pathway hurt
    the low-variate datasets, where averaging across 7 variates only dilutes the
    signal -- on ILI it took MSE from 2.08 to 2.29 -- while helping the
    high-variate ones, where it produced the best CRPS on both Weather and Solar.
    Starting closed lets each dataset open the gate only as far as its own
    cross-variable structure justifies, instead of paying for it everywhere.
    """

    def __init__(self, d_model, gate_init=-3.0):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.full((1,), gate_init))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, context, n_variates):
        # (B*C, N, D) -> (B, C, N, D); the fold is index = b * C + c, so this
        # reshape recovers the original grouping.
        batch_variate, n_patches, d_model = context.shape
        grouped = context.reshape(-1, n_variates, n_patches, d_model)
        shared = self.projection(grouped.mean(dim=1, keepdim=True))
        mixed = self.norm(grouped + torch.sigmoid(self.alpha) * shared.expand_as(grouped))
        return mixed.reshape(batch_variate, n_patches, d_model)


class ValueVectorField(nn.Module):
    """Velocity field over patch *values* rather than token embeddings.

    Same body as `common.VectorFieldNet` -- additive conditioning plus FiLM on
    the flow time, over `SimpleSSMBlock` (the real S5 layer) -- wrapped in
    `Linear(patch_len -> d_model)` and `Linear(d_model -> patch_len)`. That
    output projection is the decoder, and it is trained by the flow-matching
    loss rather than left to chance.
    """

    def __init__(self, patch_len, d_model, n_layers=2, condition_inject=True, n_modes=1):
        super().__init__()
        self.d_model = d_model
        # K velocity hypotheses plus mixture logits, instead of one velocity.
        #
        # Conditional flow matching regresses E[x1 - x0 | x_t, t, c]. When the
        # plausible x1 fall into distinct modes that expectation points BETWEEN
        # them, and integrating an averaged field lands between modes rather than on
        # one. Both of this project's standing failures are that, measured:
        #
        #   physio       samples agree on the beat rate and disagree on the phase, so
        #                their mean is flat -- PENGUIN reads 33.71 bpm on the ensemble
        #                mean and 11.11 per realisation;
        #   forecasting  the generative model's mean is a worse mean estimator than a
        #                direct regressor -- FlowSSM 0.3578 MSE against DecompSSM's
        #                0.3193 on ETTm1.
        #
        # One cause, so one fix, and it is the same code on both sides. `n_modes=1`
        # is exactly the previous model.
        self.n_modes = n_modes
        self.in_proj = nn.Linear(patch_len, d_model)
        self.layers = nn.ModuleList([SimpleSSMBlock(d_model) for _ in range(n_layers)])
        self.film = nn.ModuleList([nn.Linear(d_model, 2 * d_model) for _ in range(n_layers)])
        # One projection per layer, so the condition is re-read at every depth
        # rather than only at the input. Zero-initialised so an untrained model is
        # exactly the previous architecture and the change cannot be credited to
        # initialisation.
        # A flag rather than always-on, because the ablation has to be runnable
        # under the SAME code as the treatment. Every forecasting record in
        # results.jsonl predates this, so comparing against those would confound the
        # injection with whatever else changed in between.
        self.condition_inject = None
        if condition_inject:
            self.condition_inject = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
            for layer in self.condition_inject:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.out_proj = nn.Linear(d_model, patch_len * n_modes)
        self.mode_logits = nn.Linear(d_model, n_modes) if n_modes > 1 else None
        self.patch_len = patch_len

    def hypotheses(self, x_t, t, context):
        """`(velocities (B, N, K, patch_len), logits (B, N, K) or None)`."""
        features = self._trunk(x_t, t, context)
        raw = self.out_proj(features)
        velocities = raw.reshape(*raw.shape[:-1], self.n_modes, self.patch_len)
        logits = self.mode_logits(features) if self.mode_logits is not None else None
        return velocities, logits

    def forward(self, x_t, t, context, mode=None):
        """Conditioning enters at EVERY layer, not once at the input.

        Adding the context once and then running the stack was measured to leave
        the model with no usable waveform at all: `FlowSSMRecon` scored waveform
        correlation r = 0.000 on every physio task, before any constraint was
        added, while PENGUIN (0.161-0.276) and even a plain conv net (0.141-0.218)
        recovered morphology. PENGUIN runs the PPG as its own stream and injects it
        into the target stream at every block; the single addition here gave the
        velocity field one chance to use the condition and then let it wash out.
        """
        velocities, logits = self.hypotheses(x_t, t, context)
        if self.n_modes == 1:
            return velocities[..., 0, :]
        if mode is None:
            # No commitment asked for: the mixture MEAN. This is the right readout
            # for squared error and the wrong one for anything that has to look like
            # a realisation, which is exactly the distinction the two tasks make.
            weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
            return (velocities * weights).sum(dim=-2)
        # A committed mode, held for the whole trajectory -- see `euler_sample`.
        index = mode.view(-1, *([1] * (velocities.dim() - 2)), 1).expand(
            *velocities.shape[:-2], 1, velocities.shape[-1]
        )
        return velocities.gather(-2, index).squeeze(-2)

    def _trunk(self, x_t, t, context):
        x = self.in_proj(x_t) + context
        t_emb = flow_time_embedding(t, self.d_model)
        for index, (layer, film) in enumerate(zip(self.layers, self.film, strict=True)):
            if self.condition_inject is not None:
                x = x + self.condition_inject[index](context)
            scale, shift = film(t_emb).chunk(2, dim=-1)
            x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            x = layer(x)
        return x


class FlowSSM(BenchmarkModel):
    """Shared backbone: normalize -> patch -> condition SSM -> align -> flow.

    Subclasses supply only the batch metadata (`_condition_meta`,
    `_target_meta`) and the reshaping between the benchmark's tensor layout and
    the `(batch, n_patches, patch_len)` layout used internally.
    """

    is_probabilistic = True

    def __init__(
        self,
        condition_len,
        target_len,
        d_model=64,
        n_condition_layers=2,
        n_vf_layers=2,
        n_sampling_steps=20,
        preferred_patch_len=16,
        dt_seconds=1.0,
        min_std=0.1,
        max_ensemble_sequences=200_000,
        source="noise",
        source_noise=1.0,
        cross_variate=False,
        point_head=False,
        unroll_steps=0,
        functional_weight=0.0,
        sigreg_weight=0.0,
        condition_skip=False,
        relative_align=False,
        align_bias_mode="comb",
        align_learn_bias=True,
        condition_inject=True,
        n_modes=1,
        mode_commitment=True,
        rarity_weight=False,
        multiscale=False,
        functional_target_kind="ECG",
        quantile_head=False,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
        quantile_weight=1.0,
        point_weight=1.0,
        direct_point=True,
        covariate_condition=False,
    ):
        super().__init__()
        if source not in SOURCES:
            raise ValueError(f"unknown flow source {source!r}; expected one of {sorted(SOURCES)}")
        self.source = source
        self.source_noise = source_noise
        self.min_std = min_std
        # Working-set cap for ensemble sampling, in sequences. 200k keeps the
        # peak near 20 GB, comfortable on a 96 GB card alongside a second job.
        self.max_ensemble_sequences = max_ensemble_sequences
        self.condition_len = condition_len
        self.target_len = target_len
        self.patch_len = choose_patch_len(condition_len, target_len, preferred=preferred_patch_len)
        self.n_condition_patches = condition_len // self.patch_len
        self.n_target_patches = target_len // self.patch_len
        self.n_sampling_steps = n_sampling_steps
        # The integrator is swappable so it can be compared at matched NFE
        # (scripts/sampler_sweep.py) without retraining. Euler costs 1 function
        # evaluation per step, Heun 2, so step count alone is not comparable.
        self.sample_fn = euler_sample
        self.dt_seconds = dt_seconds

        self.tokenizer = UnifiedTokenizer(d_model, self.patch_len, num_channels=3, num_modalities=2, num_tasks=2, multiscale=multiscale)
        self.aligner = TargetTimeAligner(d_model)
        self.condition_ssm = ConditionSSM(d_model, n_condition_layers)
        self.vector_field = ValueVectorField(self.patch_len, d_model, n_vf_layers, condition_inject=condition_inject, n_modes=n_modes)
        self.n_modes = n_modes
        self.mode_commitment = mode_commitment
        # Reweight training windows by how rare their conditioning is. The error is
        # concentrated, not spread: on BIDMC-ECG the worst 10% of windows carry 62%
        # of the total error, and the sparse heart-rate band (8.2% of training
        # windows, under 70 bpm) is biased +6.58 bpm while every well-populated band
        # is biased slightly LOW. That is regression to the mean under an imbalanced
        # conditional distribution, and forecasting has the same shape. Fitted from
        # the training split in `fit`; see models/rarity_weight.py.
        self.rarity_weight = rarity_weight
        self._rarity = None
        self.variate_mixer = VariateContext(d_model) if cross_variate else None
        # A directly MSE-supervised point forecast. See `_point_forecast`.
        self.point_head = nn.Linear(d_model, self.patch_len) if point_head else None
        # The forecasting-side test of the readout question, on a NONLINEAR
        # functional. MSE's optimal readout is the mean, so an ensemble mean is
        # already the right estimator and a point head cannot help -- measured,
        # it did not (0.3550 -> 0.3659 on ETTm1). A quantile is not a mean: the
        # q-quantile of an ensemble is a different object from a directly
        # supervised q-quantile, and that is where a readout gap should appear if
        # the principle is real. Same trunk, second readout, so the comparison is
        # about the readout and nothing else.
        self.quantile_levels = tuple(quantile_levels)
        self.quantile_head = (
            nn.Linear(d_model, self.patch_len * len(self.quantile_levels)) if quantile_head else None
        )
        # ... plus, by default, a whole-window linear term. DLinear's lesson is that
        # a single Linear(lookback -> horizon) over the *entire* window is a very
        # strong LTSF predictor, and the token-wise head cannot express it: it sees
        # only one attention-pooled d_model vector per output patch. DecompSSM's
        # head is whole-window too (Linear(96 -> 128) -> ... -> Linear(128 -> 96)),
        # which is the structural difference the measurements pointed at.
        self.direct_point = nn.Linear(condition_len, target_len) if (point_head and direct_point) else None
        # Motion (or other) covariate stream, entering as an additive zero-init
        # projection on the SAME condition tokens. Zero-init makes the model exactly
        # the base model at initialisation, and the shared conditioning mechanism
        # (alignment, injection) is untouched -- the covariate only enriches what a
        # condition token says about its own time span.
        self.covariate_proj = None
        self.wants_covariate = bool(covariate_condition)
        if covariate_condition:
            self.covariate_proj = nn.Linear(self.patch_len, d_model)
            nn.init.zeros_(self.covariate_proj.weight)
            nn.init.zeros_(self.covariate_proj.bias)
        self.quantile_weight = quantile_weight
        self.point_weight = point_weight
        # Closing the Jensen gap. A rate is a nonlinear functional of the waveform,
        # so regressing the velocity does not control it: measured in this project,
        # the same weights gave 32.36 bpm read off the generated waveform and 8.41
        # from a directly supervised head, and scoring PENGUIN on an ensemble mean
        # read 33.71 bpm against 11.11 per realisation. Physics-based Flow Matching
        # (ICLR 2026) closes exactly this by unrolling the learned dynamics during
        # training so constraints act on the generated sample; their constraints are
        # PDEs, ours is that the sample must beat at the reference rate.
        #
        # `unroll_steps=0` disables it and recovers the plain objective, which is the
        # ablation. 1 is the single Euler completion from x_t, which is nearly free.
        self.unroll_steps = unroll_steps
        self.functional_weight = functional_weight
        self.functional_target_kind = functional_target_kind
        # SIGReg (LeJEPA) on the CONDITIONING representation, not on a
        # joint-embedding predictor's output. The degenerate solutions this project
        # keeps finding are collapses of the context: if it only has to carry a rate,
        # it carries only a rate, and the measured symptom was 7.89 bpm with waveform
        # correlation r = 0.000. Enforcing an isotropic-Gaussian context makes rank
        # collapse unavailable, and it adds no coefficient to tune beyond a weight
        # whose zero recovers the plain objective.
        self.sigreg_weight = sigreg_weight
        # Zero-initialised for the same reason as the per-layer injection: with the
        # skip enabled but untrained, the model is bit-identical to the one without.
        self.condition_skip = condition_skip
        self.condition_skip_proj = nn.Linear(d_model, d_model) if condition_skip else None
        if self.condition_skip_proj is not None:
            nn.init.zeros_(self.condition_skip_proj.weight)
            nn.init.zeros_(self.condition_skip_proj.bias)

        # The general form of the skip, and the only part of the conditioning fix
        # that can be shared. `condition_skip` hard-codes the identity alignment and
        # is therefore reconstruction-only; this learns the offset instead, so
        # reconstruction can settle on 0 and forecasting on a seasonal lag from the
        # same module. Verified on a synthetic task: asked to reproduce the condition
        # at lag 0 it attends at -0.08 patches, at lag -12 it attends at -9.97.
        self.relative_align = (
            RelativeAlignedContext(d_model, max_offset=512, bias_mode=align_bias_mode, learn_bias=align_learn_bias)
            if relative_align
            else None
        )
        self.last_alignment = None  # diagnostic: where the model actually looked
        if source == "point" and self.point_head is None:
            raise ValueError("source='point' requires point_head=True: there is no point forecast to start the flow from")

    # -- metadata -----------------------------------------------------------

    def _meta(self, batch_size, n_patches, t_start, channel_id, modality_id, task_id, device):
        t_step = 1.0 / self.n_condition_patches  # one condition window spans t in [0, 1)
        times = t_start + t_step * torch.arange(n_patches, dtype=torch.float32, device=device)

        def full(fill, dtype):
            return torch.full((batch_size, n_patches), fill, dtype=dtype, device=device)

        return {
            "t": times.unsqueeze(0).expand(batch_size, -1).contiguous(),
            "dt": full(float(self.dt_seconds), torch.float32),
            "channel_id": full(channel_id, torch.long),
            "modality_id": full(modality_id, torch.long),
            "task_id": full(task_id, torch.long),
        }

    def _condition_meta(self, batch_size, device):
        raise NotImplementedError

    def _target_meta(self, batch_size, device):
        raise NotImplementedError

    # -- core ---------------------------------------------------------------

    def _normalize(self, condition_flat):
        """Per-(window, channel) statistics taken from the CONDITION window only.

        These are the statistics used to denormalize the generated target, which
        is the whole reason they may not come from the target: at inference the
        target has not been observed.

        The standard-deviation floor is load-bearing, not defensive tidiness.
        **1.8% of ETTm2's (window, variate) pairs are exactly constant** -- long
        flat runs in variates like `LULL` -- so their window std is 0. Dividing
        the target by a floor of 1e-5 amplified it by up to 10^5, and the
        flow-matching loss reached 7e8. Measured, and it is what made FlowSSM
        score worse than persistence.

        The floor is expressed relative to the *globally* standardized scale:
        the harness fits a train-only `StandardScaler` before windowing, so the
        input already has unit variance per variate. A window flatter than
        `min_std` of that carries no scale information worth dividing by, and
        centring alone is the right treatment.
        """
        mean = condition_flat.mean(dim=1, keepdim=True)
        std = condition_flat.std(dim=1, keepdim=True).clamp_min(self.min_std)
        return mean, std

    def _split_covariate(self, flattened):
        """Peel a channel-stacked covariate off the condition, if this model wants one.

        Datasets built with `covariate=True` yield `(B, 2, L)` conditions (signal
        first). Models without a covariate projection never receive such input;
        the guard order matters because the forecaster's raw input is also 3-D.
        """
        if self.covariate_proj is not None and flattened.dim() == 3:
            return flattened[:, 0, :], flattened[:, 1, :]
        return flattened, None

    def _patch(self, flat, n_patches):
        return flat.reshape(flat.shape[0], n_patches, self.patch_len)

    def _build_context(self, condition_flat, mean, std, covariate=None):
        batch_size, device = condition_flat.shape[0], condition_flat.device
        normalized = (condition_flat - mean) / std
        condition_batch = {
            "value": self._patch(normalized, self.n_condition_patches),
            "observed_mask": torch.ones(batch_size, self.n_condition_patches, self.patch_len, device=device),
            **self._condition_meta(batch_size, device),
        }
        condition_tokens, _ = self.tokenizer(condition_batch)
        if covariate is not None:
            condition_tokens = condition_tokens + self.covariate_proj(self._patch(covariate, self.n_condition_patches))
        encoded = self.condition_ssm(condition_tokens)

        target_query = {
            "value": torch.zeros(batch_size, self.n_target_patches, self.patch_len, device=device),
            "observed_mask": torch.ones(batch_size, self.n_target_patches, self.patch_len, device=device),
            **self._target_meta(batch_size, device),
        }
        query_tokens = self.tokenizer.embed_query(target_query)
        context = self.aligner(encoded, query_tokens)

        # A positional skip from the encoded condition, for TIME-ALIGNED tasks only.
        #
        # Measured, the aligner's output is 93% a constant vector: across-patch
        # variation is 7.3% of its overall scale, so the context the velocity field
        # sees is nearly a pooled summary and carries almost no information about
        # *where* the beats are. That is why the waveform correlation was zero
        # before any loss term was involved.
        #
        # For reconstruction the condition and target occupy the same time span --
        # `target_t_start` is 0 and the lengths match -- so patch i of one
        # corresponds to patch i of the other and the correct alignment is the
        # identity. Forecasting has no such correspondence (the target starts after
        # the condition ends), so the skip is off there and this stays a
        # single-axis change rather than a different model per task.
        if self.condition_skip and encoded.shape[1] == context.shape[1]:
            context = context + self.condition_skip_proj(encoded)
        if self.relative_align is not None:
            aligned, weights = self.relative_align(encoded, query_tokens)
            context = context + aligned
            if not self.training:
                self.last_alignment = self.relative_align.dominant_offsets(weights).detach()
        return self._mix_variates(context)

    def _mix_variates(self, context):
        """Hook for sharing context across variates. No-op unless a subclass
        knows how the variate axis was folded into the batch."""
        return context

    def _point_forecast(self, context, normalized_condition=None):
        """Normalized point forecast, `(batch, n_target_patches, patch_len)`.

        Exists because the ensemble mean is only a **Monte-Carlo estimate** of the
        MSE-optimal predictor, while a deterministic baseline minimises MSE
        directly. Measured, that is the gap: FlowSSM's MAE was 1.02-1.24x
        DecompSSM's and its MSE 0.93-1.41x with an identical tail-heaviness index
        -- a better predictive distribution whose mean was a worse point forecast.

        Two additive terms, because a token-wise head alone was not enough (it
        closed only ~1% of an 11% gap on ETTm1):

        * a **whole-window linear map** `Linear(condition_len -> target_len)`, the
          structure that makes DLinear strong on these benchmarks and that
          DecompSSM's head also has;
        * a **token-wise correction** off the SSM context, which is where the
          nonlinear and cross-variate information enters.

        The linear term carries the trend and the correction carries the rest, so
        neither has to do the other's job.
        """
        correction = self.point_head(context)
        if self.direct_point is None or normalized_condition is None:
            return correction
        direct = self.direct_point(normalized_condition)
        return correction + self._patch(direct, self.n_target_patches)

    def _source_sample(self, condition_flat, mean, std, n_repeats=1, point=None):
        """Draw the flow's starting point `x0`.

        `"noise"` is standard conditional flow matching: transport N(0, I) to the
        future. `"persistence"` starts from the last observed value held flat, and
        `"point"` from the learned point forecast -- both are stochastic
        interpolants between two *data* distributions rather than noise-to-data
        (docs/01_ideas.md section 3.3).

        The motivation is measured, not aesthetic: the Recurrent Interpolants
        ablation (arXiv:2409.11684, Table 2) reports a vanilla flow-matching head
        scoring CRPS-sum 0.038 on Traffic and then 64.3 on Wikipedia -- three
        orders of magnitude worse than every other head. Starting from a
        deterministic base means the field only has to model the residual
        uncertainty, so the degenerate solution is the base forecast rather than a
        divergence, and the transport path is far shorter. `"persistence"` tested
        poorly at 5 000 steps; `"point"` starts from a *learned* base instead,
        which is a far stronger anchor.
        """
        shape = (n_repeats * condition_flat.shape[0], self.n_target_patches, self.patch_len)
        noise = torch.randn(shape, device=condition_flat.device)
        if self.source == "noise":
            return noise
        if self.source == "point":
            # Detached: the flow must not be able to move the point forecast to
            # make its own transport easier.
            return point.detach().repeat(n_repeats, 1, 1) + self.source_noise * noise

        # Persistence: the final observed sample, normalized, held across the horizon.
        last = ((condition_flat[:, -1:] - mean) / std).repeat(n_repeats, 1)
        return last.unsqueeze(-1).expand(-1, self.n_target_patches, self.patch_len) + self.source_noise * noise

    def fit(self, train_dataset, val_dataset, budget, device, checkpoint_name=None):
        if self.rarity_weight:
            self._rarity = self._fit_rarity(train_dataset)
            print(f"[rarity] {self._rarity.summary()}", flush=True)
        return super().fit(train_dataset, val_dataset, budget, device, checkpoint_name=checkpoint_name)

    def _fit_rarity(self, train_dataset, max_windows=2000):
        """Density over a scalar summary of each TARGET window.

        The summary is the dominant in-band frequency -- a heart or respiration rate
        for the physio tasks, and the leading periodicity of the horizon for
        forecasting. One statistic, both tasks, and it is the axis the measured
        imbalance actually lies along.
        """
        from rooster.models.rarity_weight import RarityWeights

        stride = max(len(train_dataset) // max_windows, 1)
        values = [self._window_summary(train_dataset[i][1]) for i in range(0, len(train_dataset), stride)]
        return RarityWeights(values)

    def _window_summary(self, target):
        """Dominant in-band frequency of one target window, in cycles per window."""
        import numpy as np

        flat = np.asarray(target, dtype=np.float64).reshape(-1)
        flat = flat - flat.mean()
        if flat.std() < 1e-8:
            return float("nan")
        spectrum = np.abs(np.fft.rfft(flat)) ** 2
        spectrum[0] = 0.0
        return float(np.argmax(spectrum))

    def compute_loss(self, x, y):
        condition_flat, target_flat = self._flatten_inputs(x, y)
        condition_flat, covariate_flat = self._split_covariate(condition_flat)
        mean, std = self._normalize(condition_flat)
        context = self._build_context(condition_flat, mean, std, covariate=covariate_flat)
        # x1 involves no trainable parameters, so unlike the token-space variants
        # there is no moving target and no stop-gradient needed.
        x1 = self._patch((target_flat - mean) / std, self.n_target_patches)

        normalized_condition = (condition_flat - mean) / std
        point = self._point_forecast(context, normalized_condition) if self.point_head is not None else None
        x0 = self._source_sample(condition_flat, mean, std, point=point)
        if self.n_modes > 1:
            loss, aux = self._mixture_flow_step(x1, context, x0)
        else:
            loss, aux = flow_matching_step(self.vector_field, x1, context, x0=x0)
        if point is not None:
            loss = loss + self.point_weight * F.mse_loss(point, x1)
        if self._rarity is not None:
            # Per-window weights on the flow-matching term. Applied here rather than
            # to the batch mean so a rare window's velocity error counts more, which
            # is the whole point; the weights average to 1 over the training
            # distribution so this is not a disguised learning-rate change.
            weights = self._rarity([self._window_summary(w) for w in y.detach().cpu()])
            if weights is not None:
                weights = weights.to(x1.device)
                per_window = (aux["pred_v"] - aux["target_v"]).pow(2).flatten(1).mean(dim=1)
                if per_window.shape[0] == weights.shape[0]:
                    loss = (per_window * weights).mean()
        if self.quantile_head is not None:
            loss = loss + self.quantile_weight * self._quantile_loss(context, x1)
        if self.unroll_steps and self.functional_weight:
            loss = loss + self.functional_weight * self._functional_loss(aux, context, x1, mean, std)
        if self.sigreg_weight:
            from rooster.models.sigreg import sigreg

            loss = loss + self.sigreg_weight * sigreg(context)
        return loss

    def _mixture_flow_step(self, x1, context, x0):
        """Winner-take-all over K velocity hypotheses, plus a loss on the weights.

        Regressing every hypothesis on the same target would reproduce the mean and
        lose the point. Only the closest hypothesis is trained on the sample -- the
        multiple-choice-learning objective -- so the heads specialise on different
        modes instead of all covering the average. The logits are trained by
        cross-entropy against which hypothesis actually won, so at sampling time the
        mixture weights say which modes are plausible.
        """
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device=x1.device)
        x0 = torch.randn_like(x1) if x0 is None else x0
        t_broadcast = t.view(batch_size, *([1] * (x1.dim() - 1)))
        x_t = (1 - t_broadcast) * x0 + t_broadcast * x1
        target = (x1 - x0).unsqueeze(-2)

        velocities, logits = self.vector_field.hypotheses(x_t, t, context)
        per_mode = (velocities - target).pow(2).mean(dim=-1)  # (B, N, K)
        winner = per_mode.argmin(dim=-1)
        loss = per_mode.gather(-1, winner.unsqueeze(-1)).squeeze(-1).mean()
        loss = loss + F.cross_entropy(logits.reshape(-1, self.n_modes), winner.reshape(-1))

        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        aux = {"x_t": x_t, "t": t_broadcast, "pred_v": (velocities * weights).sum(dim=-2),
               "target_v": x1 - x0, "x0": x0}
        return loss, aux

    def _unroll(self, aux, context):
        """Complete the trajectory from `x_t` to `t=1`, differentiably.

        `unroll_steps=1` is the single Euler completion `x_t + (1-t) v`, which the
        training step already computes for monitoring and costs nothing extra. More
        steps re-evaluate the field along the way, which is closer to what the
        sampler actually does at test time and is where the constraint stops being
        a statement about one linear extrapolation.

        Gradients are kept through every evaluation on purpose: the point is for the
        functional constraint to reach the velocity field via the path it will
        actually be integrated along.
        """
        x_t, t = aux["x_t"], aux["t"]
        steps = max(int(self.unroll_steps), 1)
        current, current_t = x_t, t
        for index in range(steps):
            remaining = (1.0 - current_t) / (steps - index)
            flat_t = current_t.reshape(current_t.shape[0])
            velocity = aux["pred_v"] if index == 0 else self.vector_field(current, flat_t, context)
            current = current + remaining * velocity
            current_t = current_t + remaining
        return current

    def _functional_loss(self, aux, context, x1, mean, std):
        """The physiological constraint, applied to the unrolled sample.

        Both the unrolled estimate and the target are returned to the input's own
        units before the constraint is applied: the rate band is defined in Hz, and
        a per-window normalisation does not change a rate but does change the
        amplitudes the surrogate compares.
        """
        from rooster.models.functional_loss import rate_functional_loss

        n_windows = x1.shape[0]
        predicted = self._unroll(aux, context).reshape(n_windows, self.target_len)
        target = x1.reshape(n_windows, self.target_len)
        predicted = predicted * std + mean
        target = target * std + mean
        sample_rate = 1.0 / float(self.dt_seconds) if self.dt_seconds else 1.0
        return rate_functional_loss(predicted, target, sample_rate, self.functional_target_kind)

    def _predicted_quantiles(self, context):
        """`(B, n_patches, patch_len, n_levels)`, monotone across the last axis.

        Built as a base plus cumulative softplus increments so the quantiles
        cannot cross, which is a property of the estimator rather than something
        the loss has to be trusted to enforce.
        """
        raw = self.quantile_head(context)
        shape = raw.shape[:-1] + (self.patch_len, len(self.quantile_levels))
        raw = raw.reshape(shape)
        base = raw[..., :1]
        increments = F.softplus(raw[..., 1:])
        return torch.cat([base, base + increments.cumsum(dim=-1)], dim=-1)

    def _quantile_loss(self, context, x1):
        levels = torch.tensor(self.quantile_levels, device=x1.device, dtype=x1.dtype)
        errors = x1.unsqueeze(-1) - self._predicted_quantiles(context)
        return torch.maximum(levels * errors, (levels - 1.0) * errors).mean()

    @torch.no_grad()
    def sample(self, x, n_samples):
        """Draw `n_samples` trajectories, integrating the ensemble in chunks.

        Samples are tiled into the batch dimension rather than looped over one
        at a time: ODE integration is sequential in flow time, but the samples
        are independent, so one integration of a tiled tensor costs the same
        number of *sequential* steps as one sample.

        The tiling is chunked rather than done in a single tensor, because the
        product is large: Traffic folds 862 variates into the batch, so 64
        windows x 20 samples is 1.1M sequences and peaked at 86 GB (it OOMed on
        a 96 GB card). `max_ensemble_sequences` caps the working set; it changes
        speed and memory only, never the result.
        """
        condition_flat = self._flatten_condition(x)
        condition_flat, covariate_flat = self._split_covariate(condition_flat)
        n_windows = condition_flat.shape[0]
        mean, std = self._normalize(condition_flat)
        context = self._build_context(condition_flat, mean, std, covariate=covariate_flat)

        point = self._point_forecast(context, (condition_flat - mean) / std) if self.point_head is not None else None
        per_chunk = max(1, min(n_samples, self.max_ensemble_sequences // max(n_windows, 1)))
        chunks = []
        for start in range(0, n_samples, per_chunk):
            width = min(per_chunk, n_samples - start)
            # Sampling must start from the same distribution training interpolated
            # from, or the learned field is being integrated from the wrong place.
            x0 = self._source_sample(condition_flat, mean, std, n_repeats=width, point=point)
            repeated_context = context.repeat(width, 1, 1)
            mode = None
            if self.n_modes > 1 and self.mode_commitment:
                # One mode per SAMPLE, drawn from the learned mixture weights at the
                # start of the trajectory and held. This is what makes an ensemble
                # member a coherent realisation rather than a blend: the physio
                # measurements say the samples differ by phase, and a per-step redraw
                # would let a trajectory change phase halfway through.
                with torch.no_grad():
                    _v, logits = self.vector_field.hypotheses(x0, torch.zeros(x0.shape[0], device=x0.device), repeated_context)
                    mode = torch.multinomial(torch.softmax(logits.mean(dim=1), dim=-1), 1).squeeze(-1)
            chunks.append(
                self.sample_fn(
                    self.vector_field, x0.shape, repeated_context, self.n_sampling_steps,
                    condition_flat.device, x0=x0, mode=mode,
                )
            )
        x1_hat = torch.cat(chunks, dim=0)

        flat = x1_hat.reshape(n_samples * n_windows, self.target_len)
        flat = flat * std.repeat(n_samples, 1) + mean.repeat(n_samples, 1)
        draws = [self._unflatten_output(flat[i * n_windows : (i + 1) * n_windows], x) for i in range(n_samples)]
        return torch.stack(draws, dim=0)

    @torch.no_grad()
    def forward(self, x):
        """The model's declared point forecast.

        With a point head, that head's output -- an estimator actually trained for
        squared error. Without one, the mean of a small ensemble, which is only a
        Monte-Carlo estimate of the same quantity. Never best-of-N either way.
        """
        if self.point_head is None:
            return self.sample(x, self.n_point_samples).mean(dim=0)

        condition_flat = self._flatten_condition(x)
        condition_flat, covariate_flat = self._split_covariate(condition_flat)
        mean, std = self._normalize(condition_flat)
        context = self._build_context(condition_flat, mean, std, covariate=covariate_flat)
        point = self._point_forecast(context, (condition_flat - mean) / std)
        flat = point.reshape(condition_flat.shape[0], self.target_len) * std + mean
        return self._unflatten_output(flat, x)

    @torch.no_grad()
    def predict_quantiles(self, x):
        """Quantiles from the head, in the input's units: `(B, ..., n_levels)`.

        The counterpart to reading them off the sampled ensemble. Both are
        available from the same trained weights, which is what makes the
        comparison about the readout.
        """
        condition_flat = self._flatten_condition(x)
        condition_flat, covariate_flat = self._split_covariate(condition_flat)
        mean, std = self._normalize(condition_flat)
        context = self._build_context(condition_flat, mean, std, covariate=covariate_flat)
        quantiles = self._predicted_quantiles(context)  # (B, n_patches, patch_len, n_levels)
        n_windows = condition_flat.shape[0]
        outputs = []
        for index in range(quantiles.shape[-1]):
            flat = quantiles[..., index].reshape(n_windows, self.target_len) * std + mean
            outputs.append(self._unflatten_output(flat, x))
        return torch.stack(outputs, dim=-1)

    n_point_samples = 8

    # -- layout adapters, supplied by subclasses ----------------------------

    def _flatten_inputs(self, x, y):
        raise NotImplementedError

    def _flatten_condition(self, x):
        raise NotImplementedError

    def _unflatten_output(self, flat, x):
        raise NotImplementedError


@register_benchmark_model("FlowSSM")
class FlowSSMForecaster(FlowSSM):
    """Forecasting: `(B, L, C)` lookback -> `(B, H, C)` horizon.

    Channel-independent -- the variate axis is folded into the batch, so one set
    of weights serves every variate, matching the PatchTST/DLinear convention the
    baselines use. The target sits *after* the condition in relative time.
    """

    def __init__(self, lookback, horizon, n_variates, **kwargs):
        super().__init__(condition_len=lookback, target_len=horizon, **kwargs)
        self.n_variates = n_variates

    def _mix_variates(self, context):
        if self.variate_mixer is None:
            return context
        return self.variate_mixer(context, self.n_variates)

    def _condition_meta(self, batch_size, device):
        return self._meta(batch_size, self.n_condition_patches, 0.0, CHANNEL_FORECAST, MODALITY_TIMESERIES, TASK_FORECAST, device)

    def _target_meta(self, batch_size, device):
        # t starts at 1.0: strictly after the condition window. This is the only
        # thing that tells the model it is extrapolating rather than translating.
        return self._meta(batch_size, self.n_target_patches, 1.0, CHANNEL_FORECAST, MODALITY_TIMESERIES, TASK_FORECAST, device)

    def _flatten_condition(self, x):
        return x.permute(0, 2, 1).reshape(-1, self.condition_len)

    def _flatten_inputs(self, x, y):
        return self._flatten_condition(x), y.permute(0, 2, 1).reshape(-1, self.target_len)

    def _unflatten_output(self, flat, x):
        return flat.reshape(x.shape[0], x.shape[2], self.target_len).permute(0, 2, 1)


@register_benchmark_model("FlowSSMRecon")
class FlowSSMReconstructor(FlowSSM):
    """Reconstruction: `(B, T)` PPG -> `(B, T)` target vital sign.

    Identical machinery to the forecaster. The target's relative time *overlaps*
    the condition's rather than following it, and the channel ids differ -- which
    is the entire encoding of "translate this channel" versus "extrapolate this
    one", exactly as the project's task-identity design requires.
    """

    def __init__(self, window_samples, sample_rate=None, target_kind=None, **kwargs):
        # Unlike the other context fields, this one IS used when the unrolled
        # functional constraint is on: a respiration rate lives in 0.1-0.6 Hz and a
        # heart rate in 0.7-3.0 Hz, so the wrong band would constrain the wrong
        # thing entirely.
        if target_kind:
            kwargs.setdefault("functional_target_kind", target_kind)
        # `target_kind` is named and ignored on purpose. build_model drops kwargs a
        # model does not accept, but that filter gives up on any signature with
        # **kwargs -- so this class received every context field the harness knows
        # and forwarded it to a stricter parent, which raised. Naming the fields
        # this model does not use is what keeps that filter working here.
        # The tokenizer embeds the physical sampling interval, so the task's real
        # rate is used rather than a placeholder -- that embedding is the only
        # thing telling the model 125 Hz ECG apart from 15-minute weather data.
        if sample_rate:
            kwargs.setdefault("dt_seconds", 1.0 / float(sample_rate))
        super().__init__(condition_len=window_samples, target_len=window_samples, **kwargs)

    def _condition_meta(self, batch_size, device):
        return self._meta(batch_size, self.n_condition_patches, 0.0, CHANNEL_PPG, MODALITY_BIOSIGNAL, TASK_RECONSTRUCT, device)

    def _target_meta(self, batch_size, device):
        # t starts at 0.0: aligned with the condition, not after it.
        return self._meta(batch_size, self.n_target_patches, 0.0, CHANNEL_VITAL, MODALITY_BIOSIGNAL, TASK_RECONSTRUCT, device)

    def _flatten_condition(self, x):
        return x

    def _flatten_inputs(self, x, y):
        return x, y

    def _unflatten_output(self, flat, x):
        return flat


__all__ = ["FlowSSM", "FlowSSMForecaster", "FlowSSMReconstructor", "ValueVectorField", "choose_patch_len"]
