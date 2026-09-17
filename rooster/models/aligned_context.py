"""Condition-to-target alignment as a learned relative offset, shared by both tasks.

The mechanism this replaces, and why. Adding the positionally-matched condition to
the context recovered waveform morphology on the physio side -- correlation went
from 0.001 to 0.109 on BIDMC-ECG and from -0.001 to 0.268 on CapnoBase-RESP, where
per-layer conditioning injection moved nothing at all (every r within +-0.001). But
it was written as an identity map and therefore switched off for forecasting, whose
target starts after the condition ends. That made the one component that works
untransferable, which is no basis for a shared mechanism.

The identity was never the point. What the skip actually did was hand each target
position the condition content at *the position that corresponds to it*. Both tasks
have such a correspondence; they differ only in what it is:

    reconstruction   target patch j  <-  condition patch j          (offset 0)
    forecasting      target patch j  <-  condition patch j - P      (one period back)

So the offset is the thing to learn, not to hard-code. This module scores every
(target, condition) patch pair with a bias that depends only on their relative
offset, so the model can concentrate on offset 0, on a seasonal lag, or on a spread
of both, and which one it picks is a property of the data rather than of a flag.

That also makes the alignment *measurable*: `dominant_offsets` reports where each
target patch actually looks, so "reconstruction learns the identity and forecasting
learns a period" becomes a number rather than an assumption.

Content matters too -- two beats are not interchangeable just because they are one
period apart -- so the logits are content similarity plus the relative bias, and
the bias is what carries the positional prior.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# The ablation axis for the bias itself. "comb" is the proposed form; the other
# three are the forms it was argued to beat, promoted from narrative to measured
# arms so the paper's ablation table can hold numbers rather than reasons:
#   "none"  -- no relative bias at all: content-only cross-attention.
#   "bump"  -- a single aperiodic bump with learnable centre, width and depth
#              per head; the comb minus its periodicity.
#   "free"  -- one free bias per (head, offset), zero-init; the comb minus its
#              parametric structure.
BIAS_MODES = ("comb", "bump", "free", "none")


class RelativeAlignedContext(nn.Module):
    """Cross-attention from target patches to condition patches, with a bias that
    is a learned function of the relative offset between them."""

    def __init__(self, d_model, max_offset=512, n_heads=4, offset_scale=4.0, bias_mode="comb", learn_bias=True):
        super().__init__()
        if bias_mode not in BIAS_MODES:
            raise ValueError(f"unknown bias_mode {bias_mode!r}; expected one of {BIAS_MODES}")
        self.bias_mode = bias_mode
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim * n_heads != d_model:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        # One bias per (head, relative offset). Offsets run over [-max_offset,
        # max_offset]; forecasting needs negative ones (look back), reconstruction
        # needs zero, and a long horizon can need either.
        self.max_offset = max_offset
        # The bias is a PERIODIC comb over offsets: a learnable centre, period and
        # sharpness per head.
        #
        # Three earlier forms failed, each for a diagnosable reason, and the comb is
        # what removes the common cause. Free per-offset entries had to find the
        # right position among 250 candidates and lost to a hard-coded identity skip
        # they should subsume. A single bump peaked at 0 solved translation exactly
        # and trapped extrapolation, learning -5.49 for a true lag of -12; making the
        # bump parametric fixed translation to loss 0.0000 and left extrapolation at
        # +1.08. On real forecasting data the same thing happened: every learned
        # centre stayed within 0.2 patches of zero and the widths collapsed to
        # 0.8-2.2, i.e. the model sharpened instead of moving.
        #
        # The cause is a dead zone. A bump centred at 0 has essentially no mass at
        # offset -12, so there is no gradient pointing there -- sharpening is
        # downhill and moving is not. A comb, `-kappa * (1 - cos(2*pi*(d - phi)/P))`,
        # has peaks at `phi, phi +- P, phi +- 2P, ...` and therefore mass along the
        # whole offset axis, so a period that is slightly wrong is corrected by
        # gradient rather than by search.
        #
        # It also matches the structure of the problem. Periodicity is what makes
        # forecasting possible, and a beat train is periodic by definition, so `P` is
        # the quantity the model should be discovering. Both target solutions remain
        # reachable: `P` longer than the window leaves a single peak at `phi`, which
        # is the identity alignment translation wants, and a short `P` puts mass at
        # a seasonal lag. The learned period is then a read-out -- ETTm1 is 15-minute
        # data, so a daily cycle is 96 steps, which at patch length 8 is 12 patches.
        # The comb parameters double as the bump's (there the "period" is a width):
        # same count, same init, so the comb-vs-bump comparison varies the bias's
        # FORM and nothing else. "free"/"none" do not create them at all, which is
        # what the evaluation harness keys its alignment diagnostics on.
        if bias_mode in ("comb", "bump"):
            self.offset_centre = nn.Parameter(torch.zeros(n_heads))
            # Periods spread geometrically, so some heads start periodic and some start
            # effectively local. Which kind carries the signal is left to training.
            periods = float(max_offset) / (2.0 ** torch.arange(n_heads, dtype=torch.float32))
            self.log_period = nn.Parameter(torch.log(periods))
            self.log_sharpness = nn.Parameter(torch.full((n_heads,), math.log(offset_scale)))
            if not learn_bias:
                # Frozen at init: the geometric prior stays, the data's period is
                # never learned. Isolates "having a comb" from "learning its period".
                self.offset_centre.requires_grad_(False)
                self.log_period.requires_grad_(False)
                self.log_sharpness.requires_grad_(False)
        elif bias_mode == "free":
            # T5-style: one free logit per (head, clamped offset), zero-init so an
            # untrained model matches the no-bias attention exactly.
            self.offset_bias = nn.Parameter(torch.zeros(n_heads, 2 * max_offset + 1))
        self.out = nn.Linear(d_model, d_model)
        # Zero-initialised output, so an untrained model behaves exactly as before
        # and any gain is attributable to training rather than to initialisation.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _bias(self, n_target, n_condition, device):
        if self.bias_mode == "none":
            return torch.zeros(self.n_heads, n_target, n_condition, device=device)
        target_positions = torch.arange(n_target, device=device, dtype=torch.float32).unsqueeze(1)
        condition_positions = torch.arange(n_condition, device=device, dtype=torch.float32).unsqueeze(0)
        offsets = condition_positions - target_positions  # (n_target, n_condition)
        if self.bias_mode == "free":
            index = offsets.long().clamp(-self.max_offset, self.max_offset) + self.max_offset
            return self.offset_bias[:, index.reshape(-1)].reshape(self.n_heads, n_target, n_condition)
        centre = self.offset_centre.view(-1, 1, 1)
        period = self.log_period.exp().clamp_min(1.0).view(-1, 1, 1)
        sharpness = self.log_sharpness.exp().view(-1, 1, 1)
        if self.bias_mode == "bump":
            # Same three parameters, no periodicity: 0 at the centre, -sharpness in
            # the tails, width set by "period". The docstring's dead-zone argument
            # says this form cannot travel to a seasonal lag; this arm measures it.
            deviation = (offsets.unsqueeze(0) - centre) / period
            return -sharpness * (1.0 - torch.exp(-0.5 * deviation.pow(2)))
        phase = 2.0 * math.pi * (offsets.unsqueeze(0) - centre) / period
        return -sharpness * (1.0 - torch.cos(phase))

    def forward(self, condition_tokens, target_queries):
        """`(B, Nc, d)` and `(B, Nt, d)` -> `(B, Nt, d)` aligned condition content."""
        batch, n_target, _ = target_queries.shape
        n_condition = condition_tokens.shape[1]

        def split(x, n):
            return x.reshape(batch, n, self.n_heads, self.head_dim).transpose(1, 2)

        q = split(self.query(target_queries), n_target)
        k = split(self.key(condition_tokens), n_condition)
        v = split(self.value(condition_tokens), n_condition)

        logits = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        logits = logits + self._bias(n_target, n_condition, condition_tokens.device).unsqueeze(0)
        weights = F.softmax(logits, dim=-1)
        gathered = (weights @ v).transpose(1, 2).reshape(batch, n_target, self.d_model)
        return self.out(gathered), weights

    @torch.no_grad()
    def dominant_offsets(self, weights):
        """Mean attended offset per target position, in patches.

        Negative means looking back. Reconstruction should sit near 0; forecasting
        should sit near minus one period. Reported rather than assumed.
        """
        batch, heads, n_target, n_condition = weights.shape
        target_positions = torch.arange(n_target, device=weights.device).unsqueeze(1)
        condition_positions = torch.arange(n_condition, device=weights.device).unsqueeze(0)
        offsets = (condition_positions - target_positions).to(weights.dtype)
        return (weights * offsets).sum(dim=-1).mean(dim=(0, 1))
