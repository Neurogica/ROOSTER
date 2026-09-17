"""DecompDict: DecompSSM with a *learned* number of components.

The single-axis ablation the paper needs. Everything is DecompSSM's own code and
DecompSSM's own hyperparameters -- the variate-centric embedding, the `GTSSM`
branch (adaptive step predictor, component gating, frequency gate), the Global
Context Refinement, the concatenated prediction head, the non-stationary
normalization, the auxiliary reconstruction and orthogonality losses. Exactly one
thing changes:

    DecompSSM  : 3 branches, named trend / seasonal / residual, always all three.
    DecompDict : `n_atoms` branches (default 16) with frequency priors spread
                 across the same span, and a **JumpReLU gate that switches
                 branches off**, so the number actually used is learned per
                 window and per dataset.

Building this as a derivative rather than a fresh model was the correction to a
real mistake. A from-scratch dictionary model (`DictSSM`) reached MSE 0.396 on
ETTm1 against DecompSSM's 0.314: it was not losing because a learned dictionary
is worse than a fixed one, it was losing because it was a weaker model in every
other respect at the same time. docs/01_ideas.md section 4 warned about exactly
this -- "factorise; do not swap the decomposition and everything else at once" --
and the project did it anyway. Here the only difference is the cardinality, so a
difference in the table means what it says.

The vendored file is untouched; this imports `GTSSM` from it.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rooster.models.aligned_context import RelativeAlignedContext
from rooster.models.baselines.registry import register_benchmark_model
from rooster.models.baselines.training import BenchmarkModel
from rooster.models.quantile_head import DEFAULT_QUANTILE_LEVELS
from rooster.models.vendor.decompssm_official import GTSSM, decomposition_loss

# DecompSSM assigns its three branches (dt_min, dt_max) of (0.02, 0.2) for trend,
# (0.002, 0.02) for seasonal and (0.0002, 0.002) for residual. The atom pool
# spreads log-uniformly across that same total span, so a pool of three would
# reproduce the original priors and a larger pool interpolates between them --
# the comparison stays about the *number* of components, not their timescales.
BRANCH_STEP_SPAN = (0.0002, 0.2)

# DecompSSM's per-component gate nonlinearities, cycled across the pool so the
# atoms retain the same mix of activations the three named branches had.
BRANCH_TYPES = ("trend", "seasonal", "residual")

# See DecompDict.__init__ for why this choice decides whether the learned count
# is measuring the data or the head.
MIXINGS = ("sum", "concat")


class BranchGate(nn.Module):
    """JumpReLU over per-branch usage: a true L0 on the component count.

    One learnable threshold per branch, applied to a "how strongly is this branch
    used" score, with a straight-through estimator so the threshold can move. A
    branch below its threshold contributes exactly zero and is counted as off.

    JumpReLU rather than L1 because L1 shrinks the components it keeps, so
    coefficient magnitude stops measuring usage and "how many are active" becomes
    a question about an arbitrary cutoff. Here the count is the number of open
    gates.

    **The surrogate gradient is a sigmoid, not a clamped ramp.** The textbook
    rectangular estimator gives gradient only within a bandwidth of the threshold,
    and that silently fails here: scores initialise around 0.5 while the threshold
    starts at 0.3, so every score sits outside the window, the ramp saturates, and
    the threshold receives *exactly zero* gradient -- measured, all eight
    thresholds got 0.0. The component count would then be whatever the
    initialisation happened to give, and calling it "learned" would be false. A
    sigmoid surrogate decays but never reaches zero, so the threshold can always
    move however the score distribution sits.

    Score and threshold are both squashed into (0, 1) for the same reason: it
    keeps them on a comparable scale so `temperature` means the same thing
    regardless of the branch.
    """

    def __init__(self, n_branches, d_model, threshold_init=0.3, temperature=0.1):
        super().__init__()
        self.score = nn.Linear(d_model, n_branches)
        self.threshold_logit = nn.Parameter(torch.full((n_branches,), math.log(threshold_init / (1.0 - threshold_init))))
        self.temperature = temperature

    @property
    def threshold(self):
        return torch.sigmoid(self.threshold_logit)

    def forward(self, shared):
        """`(B, M, d_model)` -> `(usage (B, n_branches), gate (B, n_branches))`."""
        usage = torch.sigmoid(self.score(shared.mean(dim=1)))
        threshold = self.threshold.unsqueeze(0)
        gate = (usage > threshold).to(usage.dtype)
        if self.training:
            soft = torch.sigmoid((usage - threshold) / self.temperature)
            gate = gate.detach() + soft - soft.detach()
        return usage * gate, gate


@register_benchmark_model("DecompDict")
class DecompDict(BenchmarkModel):
    """DecompSSM's architecture with a learned component count."""

    is_probabilistic = False

    def __init__(
        self,
        lookback,
        horizon,
        n_variates,
        n_atoms=16,
        d_model=128,
        state_size=64,
        dropout=0.1,
        sparsity_weight="mdl",
        atom_usage_weight=None,
        lambda_reconstruction=1.0,
        lambda_orthogonality=0.02,
        aux_loss_weight=0.1,
        channel_interaction_strength=0.15,
        component_mixing="sum",
        select_k=False,
        quantile_weight=0.0,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
        gate_off=False,
        slow_first=False,
        relative_align=False,
        align_patch_len=8,
        align_bias_mode="comb",
        align_learn_bias=True,
    ):
        super().__init__()
        # Opt-in rather than default, because it changes what gets evaluated: with
        # it on, the test metrics are those of the pruned model. See
        # `select_cardinality` for why this replaces the training penalty.
        self.selects_cardinality = select_k
        # Diagnostic switch for the high-variate losses (Solar/Electricity/Traffic,
        # same sign at every horizon). DecompSSM has no usage gate at all; here the
        # gate scores are a mean over variates, which on hundreds of channels is a
        # near-constant scalar that still multiplies every component. `gate_off`
        # removes exactly that factor and nothing else.
        self.gate_off = gate_off
        self.n_atoms = n_atoms
        self.horizon = horizon
        self.sparsity_weight = sparsity_weight
        self.atom_usage_weight = atom_usage_weight
        self.n_train_samples = None  # set by the harness; only "mdl" needs it
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_orthogonality = lambda_orthogonality
        self.aux_loss_weight = aux_loss_weight

        self.value_embedding = nn.Linear(lookback, d_model)

        # The shared conditioning mechanism, attached to the model that actually
        # wins forecasting rather than to the one that does not.
        #
        # DecompDict folds the whole lookback into one embedding per variate, so its
        # token axis is the VARIATE axis and there is no time offset for the aligned
        # context to be a function of. This adds the missing axis rather than
        # replacing anything: the lookback is patched per variate, the module scores
        # every (target patch, condition patch) pair by their relative offset, and
        # the pooled result is added back to `embedded`. Everything downstream (the
        # gate, the branches, the head) is untouched.
        #
        # RelativeAlignedContext's own output projection is already zero-init, so at
        # step 0 this model IS the baseline and any difference is attributable to
        # what the module learns. Do NOT stack a second zero-init projection on top:
        # measured, that kills the gradient to both (the outer weight sees an
        # all-zero input, the inner sees an all-zero upstream weight) and only the
        # outer bias ever trains.
        self.relative_align = None
        if relative_align:
            # Prefer the requested patch length, falling back to the largest
            # smaller divisor when the horizon does not divide by it (PEMS04's
            # {12, 24, 48} horizons with the default 8). Runs at the preferred
            # length are bit-for-bit unaffected.
            if lookback % align_patch_len or horizon % align_patch_len:
                for candidate in range(min(align_patch_len, horizon) - 1, 0, -1):
                    if lookback % candidate == 0 and horizon % candidate == 0:
                        align_patch_len = candidate
                        break
            if lookback % align_patch_len or horizon % align_patch_len:
                raise ValueError(f"lookback {lookback} and horizon {horizon} must both divide by align_patch_len {align_patch_len}")
            self.align_patch_len = align_patch_len
            self.n_condition_patches = lookback // align_patch_len
            self.n_target_patches = horizon // align_patch_len
            self.align_patch_embed = nn.Linear(align_patch_len, d_model)
            # Learned queries, one per target patch: the target is not observed at
            # conditioning time, so the query carries position only.
            self.align_target_query = nn.Parameter(torch.randn(self.n_target_patches, d_model) * 0.02)
            self.relative_align = RelativeAlignedContext(
                d_model,
                max_offset=max(lookback, horizon) // align_patch_len + 1,
                bias_mode=align_bias_mode,
                learn_bias=align_learn_bias,
            )
            self.last_alignment = None
        # `_step_priors` returns fast -> slow, and BRANCH_TYPES starts at "trend",
        # so the default pairing gives the trend-type gate (unidirectional) the
        # FASTEST timescale and the residual-type gate the slowest — the exact
        # inverse of DecompSSM, whose trend is (0.02, 0.2) unidirectional and
        # residual (0.0002, 0.002) bidirectional. That breaks the "K = 3
        # reproduces DecompSSM" single-axis claim. `slow_first` restores the
        # vendored pairing; it is a flag rather than a silent fix so every row
        # recorded under the old pairing stays labeled by what it actually ran.
        priors = _step_priors(n_atoms)
        if slow_first:
            priors = priors[::-1]
        self.branches = nn.ModuleList(
            [
                GTSSM(
                    d_model=d_model,
                    state_size=state_size,
                    component_type=BRANCH_TYPES[index % len(BRANCH_TYPES)],
                    enc_in=n_variates,
                    dropout=dropout,
                    dt_min=low,
                    dt_max=high,
                    bidir=index % len(BRANCH_TYPES) != 0,  # trend-like branches stay unidirectional
                    model_type="s5",
                )
                for index, (low, high) in enumerate(priors)
            ]
        )
        self.gate = BranchGate(n_atoms, d_model)
        # Which branches survived selection. A buffer, not a parameter: it is
        # decided on validation after training, and it has to travel with the
        # checkpoint or the reported count would not be the evaluated one.
        self.register_buffer("active_mask", torch.ones(n_atoms))

        # DecompSSM's Global Context Refinement, verbatim in effect.
        self.channel_interaction_strength = nn.Parameter(torch.tensor(channel_interaction_strength))
        self.global_context_proj = nn.Linear(d_model, d_model, bias=False)
        self.channel_norm = nn.LayerNorm(d_model)

        if component_mixing not in MIXINGS:
            raise ValueError(f"unknown component_mixing {component_mixing!r}; expected one of {sorted(MIXINGS)}")
        self.component_mixing = component_mixing

        # How the components reach the head decides whether the learned count
        # means anything.
        #
        # "concat" is DecompSSM's own head, so it is the strict single-axis
        # ablation -- but it takes n_atoms * d_model inputs, which makes the head
        # grow with the pool. Measured, the model then routes everything through
        # ONE branch and the sparsity penalty prunes the rest for free: the count
        # ends up measuring how redundant the branches are given a large head,
        # not how many components the data needs. It also makes the pool size a
        # capacity knob, so "results are invariant to pool size" could never be
        # claimed.
        #
        # "sum" is the dictionary form -- `signal ~= sum_k g_k c_k` -- and the head
        # is Linear(d_model -> horizon) whatever the pool size. A component then
        # earns its place only by contributing signal, and the pool stops being a
        # hyperparameter in any meaningful sense: it just has to be large enough.
        head_input = d_model * n_atoms if component_mixing == "concat" else d_model

        # An auxiliary quantile objective on the SAME features the point head uses.
        # Judged on nothing: the paper's metrics are MSE and MAE, and this exists
        # only because a second objective on a shared trunk has now improved point
        # accuracy three times -- DictSSM (MSE 0.655 at quantile weight 0.1 against
        # 0.396 at 1.0), FlowSSM on ETTm1 (0.3680 -> 0.3561 adding a quantile head),
        # and the physio rate head (PENGUIN's waveform rate 11.11 -> 10.09 bpm).
        # Those were all models well behind the baseline; this puts the same term on
        # the one configuration that is level with DecompSSM.
        self.quantile_levels = tuple(quantile_levels)
        self.quantile_weight = quantile_weight
        self.quantile_projection = (
            nn.Linear(head_input, horizon * len(self.quantile_levels)) if quantile_weight else None
        )
        self.output_projection = nn.Sequential(
            nn.Linear(head_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon, bias=True),
        )

    def effective_sparsity_weight(self):
        """The cost, in loss units, of switching one more component on.

        `sparsity_weight="mdl"` derives this rather than taking it as a knob,
        which is the whole point of the model. **A tuned coefficient is just the
        fixed K wearing a different hat**: 0.01 pinned the count at exactly 1 on
        every pool size measured, removing the penalty gave 3.9, and neither
        number came from the data. Swapping "choose K = 3" for "choose
        lambda = 0.01" moves the arbitrary constant, it does not remove it.

        The derivation is BIC / two-part MDL. Describing a model costs about
        `(k / 2) log N` nats for `k` free parameters and `N` samples, so the
        description cost of one additional component is

            lambda = params_per_component * log(N) / (2 N)

        Both terms are determined: `params_per_component` by the architecture,
        `N` by the dataset under the benchmark protocol. Nothing is left to
        choose. And because lambda falls as `log(N) / N`, a larger dataset
        automatically pays for more components -- which is what "adaptive
        cardinality" has to mean if it means anything.
        """
        if self.sparsity_weight != "mdl":
            return float(self.sparsity_weight)
        if not self.n_train_samples:
            raise RuntimeError(
                "sparsity_weight='mdl' needs n_train_samples (the harness sets it before training); "
                "pass a float instead to use a fixed coefficient"
            )
        params = sum(parameter.numel() for parameter in self.branches[0].parameters())
        return params * math.log(self.n_train_samples) / (2.0 * self.n_train_samples)

    def _refine(self, component):
        """DecompSSM's global_context_refinement_module."""
        context = self.global_context_proj(component.mean(dim=1, keepdim=True))
        strength = torch.sigmoid(self.channel_interaction_strength)
        return self.channel_norm(component + strength * context.expand_as(component))

    def _encode(self, x):
        mean = x.mean(1, keepdim=True).detach()
        centred = x - mean
        std = torch.sqrt(torch.var(centred, dim=1, keepdim=True, unbiased=False) + 1e-5)
        embedded = self.value_embedding((centred / std).transpose(1, 2))  # (B, M, d_model)
        if self.relative_align is not None:
            embedded = embedded + self._aligned_context(centred / std)

        usage, gate = self.gate(embedded)
        if self.gate_off:
            usage = torch.ones_like(usage)
            gate = torch.ones_like(gate)
        # `active_mask` is what `select_cardinality` writes its answer into. It is
        # all ones during training, so it changes nothing there.
        usage = usage * self.active_mask.unsqueeze(0)
        gate = gate * self.active_mask.unsqueeze(0)
        components = [self._refine(branch(embedded)) * usage[:, index].view(-1, 1, 1) for index, branch in enumerate(self.branches)]
        return embedded, components, gate, mean, std

    def _aligned_context(self, normalized):
        """`(B, L, M)` normalised lookback -> `(B, M, d_model)` aligned context.

        Each variate is aligned independently, folded into the batch: which lag a
        variate should attend to is a property of that series, and the branches
        already treat variates independently.
        """
        batch, _, n_variates = normalized.shape
        # (B, L, M) -> (B*M, n_condition_patches, patch_len)
        patched = normalized.permute(0, 2, 1).reshape(batch * n_variates, self.n_condition_patches, self.align_patch_len)
        condition_tokens = self.align_patch_embed(patched)
        queries = self.align_target_query.unsqueeze(0).expand(batch * n_variates, -1, -1)
        aligned, weights = self.relative_align(condition_tokens, queries)
        if not self.training:
            self.last_alignment = weights.detach()
        # One context vector per variate: the branches consume a per-variate token,
        # so the target-patch axis is pooled rather than carried forward.
        return aligned.mean(dim=1).reshape(batch, n_variates, -1)

    def _mix(self, components):
        if self.component_mixing == "concat":
            return torch.cat(components, dim=-1)
        return torch.stack(components, dim=0).sum(dim=0)

    def forward(self, x):
        _embedded, components, _gate, mean, std = self._encode(x)
        output = self.output_projection(self._mix(components)).permute(0, 2, 1)
        return output * std[:, 0, :].unsqueeze(1) + mean[:, 0, :].unsqueeze(1)

    def compute_loss(self, x, y):
        embedded, components, gate, mean, std = self._encode(x)
        output = self.output_projection(self._mix(components)).permute(0, 2, 1)
        prediction = output * std[:, 0, :].unsqueeze(1) + mean[:, 0, :].unsqueeze(1)

        loss = F.mse_loss(prediction, y)

        # DecompSSM's auxiliary loss, over the pool instead of three named parts.
        # Its signature takes exactly three components, so the pool is folded into
        # three groups -- keeping the same reconstruction and orthogonality
        # pressure without inventing a different penalty.
        # A pool smaller than three leaves a group empty, and `sum([])` is the
        # integer 0, which the vendored loss cannot use -- so empty groups are
        # explicit zero tensors. This matters because K < 3 is exactly the region
        # the sweep has to visit to find out whether DecompSSM's three is right.
        empty = torch.zeros_like(embedded)
        grouped = [sum(components[i :: len(BRANCH_TYPES)], empty) for i in range(len(BRANCH_TYPES))]
        auxiliary = decomposition_loss(
            *grouped,
            embedded,
            lambda_reconstruction=self.lambda_reconstruction,
            lambda_orthogonality=self.lambda_orthogonality,
        )
        loss = loss + self.aux_loss_weight * auxiliary

        # The claim: how many components exist is learned. Penalised directly as
        # an L0 count, per window ...
        if self.quantile_projection is not None:
            loss = loss + self.quantile_weight * self._quantile_loss(self._mix(components), y, mean, std)

        weight = self.effective_sparsity_weight()
        loss = loss + weight * gate.sum(dim=-1).mean()
        # ... and as a group penalty on per-branch usage across the batch, which
        # is what lets a whole branch die and the DATASET-level count shrink. It
        # reuses the derived coefficient rather than carrying a second knob.
        usage_weight = weight if self.atom_usage_weight is None else self.atom_usage_weight
        loss = loss + usage_weight * gate.mean(dim=0).clamp_min(1e-8).sqrt().sum()
        return loss

    def _quantile_loss(self, mixed, y, mean, std):
        """Pinball loss on the same mixed components the point head reads.

        Predicted in normalized space and compared against a normalized target, so
        the term does not change scale with the series and the weight means the
        same thing on every dataset.
        """
        levels = torch.tensor(self.quantile_levels, device=y.device, dtype=y.dtype)
        raw = self.quantile_projection(mixed)  # (B, M, horizon * n_levels)
        raw = raw.reshape(raw.shape[0], raw.shape[1], self.horizon, len(self.quantile_levels))
        base = raw[..., :1]
        # Monotone by construction: cumulative softplus increments cannot cross.
        quantiles = torch.cat([base, base + F.softplus(raw[..., 1:]).cumsum(dim=-1)], dim=-1)
        target = ((y - mean[:, 0, :].unsqueeze(1)) / std[:, 0, :].unsqueeze(1)).permute(0, 2, 1)
        errors = target.unsqueeze(-1) - quantiles
        return torch.maximum(levels * errors, (levels - 1.0) * errors).mean()

    @torch.no_grad()
    def _validation_errors(self, batches):
        """Per-window mean squared error under the current `active_mask`.

        One scalar per window, which is what makes the comparison below *paired*:
        the same windows are scored with and without a branch, so the difference
        is measured per window and the between-window variance -- which is far
        larger than the effect being tested -- cancels out.
        """
        errors = []
        for x, y in batches:
            prediction = self.forward(x)
            errors.append((prediction - y).pow(2).flatten(1).mean(dim=1))
        return torch.cat(errors)

    @torch.no_grad()
    def select_cardinality(self, batches, device=None):
        """Choose the component count on validation, with no coefficient to pick.

        This is the answer to the thing that was wrong with every earlier version.
        Penalising the count during training only relocates the arbitrary
        constant: `sparsity_weight = 0.01` pinned K at exactly 1 regardless of
        pool size, `0.0` gave 3.9, and the honest description of both numbers is
        "whatever the coefficient said". A BIC-derived coefficient does not rescue
        it either -- measured on ETTm1, 43k parameters per branch and 240k
        training samples give lambda = 0.69, seventy times the value that already
        collapsed the count to 1, because parameter counting badly overstates the
        effective complexity of a neural branch. Reporting a number that a
        constant chose is not a contribution.

        So no constant. Train with the pool fully open, then ask of each branch
        the only question that matters: **does removing it hurt by more than the
        noise in the measurement?** Concretely, for branch k let `d_i` be the
        per-window increase in validation error when k is switched off. Keep the
        branch if

            mean(d) > standard_error(d) = std(d) / sqrt(n_windows)

        and drop it otherwise. Nothing here is chosen: the threshold is the
        estimator's own uncertainty. It is the one-standard-error rule that CART
        and the lasso literature use for exactly this purpose (Breiman et al.
        1984; Hastie, Tibshirani & Friedman, ESL 7.10), applied to components
        instead of tree size.

        Removal is greedy and backward -- repeatedly drop the *least* useful
        branch while it fails the test -- because components are not independent:
        two branches can be individually redundant while their sum is not, and
        judging them all at once against the full model would delete both. After
        each removal the remaining branches are re-scored against the smaller
        model.

        Because the threshold shrinks as `1 / sqrt(n_windows)`, a larger
        validation set resolves smaller contributions and admits more components.
        That is the sense in which the count adapts to the dataset, and it is a
        property of the criterion rather than something tuned into it.

        `batches` is an iterable of `(x, y)` -- materialised, since it is replayed
        once per candidate removal. Returns the selection trace.
        """
        was_training = self.training
        self.eval()
        if device is not None:
            batches = [(x.to(device), y.to(device)) for x, y in batches]
        else:
            batches = list(batches)

        self.active_mask.fill_(1.0)
        baseline = self._validation_errors(batches)
        n_windows = baseline.numel()
        trace = []

        while self.active_mask.sum() > 1:
            candidates = []
            for index in range(self.n_atoms):
                if self.active_mask[index] == 0:
                    continue
                self.active_mask[index] = 0.0
                difference = self._validation_errors(batches) - baseline
                self.active_mask[index] = 1.0
                mean = float(difference.mean())
                standard_error = float(difference.std(unbiased=True)) / math.sqrt(n_windows)
                candidates.append((mean, standard_error, index))

            # The weakest branch: smallest damage-on-removal relative to noise.
            mean, standard_error, index = min(candidates, key=lambda item: item[0] - item[1])
            if mean > standard_error:
                trace.append({"kept_all_remaining": int(self.active_mask.sum()), "weakest_gain": mean, "se": standard_error})
                break
            self.active_mask[index] = 0.0
            baseline = self._validation_errors(batches)
            trace.append({"dropped": index, "gain": mean, "se": standard_error, "remaining": int(self.active_mask.sum())})

        if was_training:
            self.train()
        return {"k_selected": int(self.active_mask.sum()), "n_atoms": self.n_atoms, "n_val_windows": n_windows, "trace": trace}

    @torch.no_grad()
    def cardinality(self, x):
        """The paper's headline measurement: components used, not components built."""
        _embedded, _components, gate, _mean, _std = self._encode(x)
        return {
            "per_window": float(gate.sum(dim=-1).mean()),
            "per_dataset": int((gate.sum(dim=0) > 0).sum()),
            "n_atoms": self.n_atoms,
        }


def _step_priors(n_atoms):
    """Log-uniform (dt_min, dt_max) pairs spanning DecompSSM's total range.

    Three atoms reproduce DecompSSM's own trend/seasonal/residual priors; more
    atoms interpolate between them. The point is that the pool covers the same
    dynamics DecompSSM covers, so the only thing being compared is how many
    components get used.
    """
    low, high = BRANCH_STEP_SPAN
    edges = [low * (high / low) ** (index / n_atoms) for index in range(n_atoms + 1)]
    return [(edges[index], edges[index + 1]) for index in range(n_atoms)]
