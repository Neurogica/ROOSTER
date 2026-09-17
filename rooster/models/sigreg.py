"""SIGReg: force a representation's distribution to be isotropic Gaussian.

From LeJEPA (LeCun & Balestriero) -- the point of which is that enforcing an
isotropic-Gaussian embedding distribution removes collapse *by construction*, so
the predictor tricks, stop-gradients and EMA teachers that earlier JEPA variants
needed to avoid it become unnecessary. It carries guarantees in risk minimisation
and statistical convergence, and Klindt et al. (2026) show encoders satisfying the
LeJEPA objectives are linearly identifiable.

Why it belongs here rather than in a self-supervised pipeline. This project's
generative model keeps finding degenerate solutions that satisfy the objective it
was given while abandoning what the metric wants: a rate-only solution with
waveform correlation r = 0.000, and before that an ensemble mean flat enough to have
no cardiac content at all. Those are collapses of the *conditioning* representation
-- if the context only has to carry a rate, it will only carry a rate. SIGReg makes
that specific failure unavailable, and it does so without adding a coefficient to
choose, which matters in a codebase that has twice had to disown a hand-picked
constant.

Applying SIGReg to the conditioning path of a conditional flow-matching model is
not what LeJEPA is for; there it regularises the embeddings of a joint-embedding
predictor. The object is the same, the place it acts is not.

The statistic is Cramer-von Mises between the empirical CDF of random 1-D
projections and the standard normal. Random projections because a full
multivariate goodness-of-fit test in d_model dimensions is hopeless at these batch
sizes, and any direction being Gaussian is what isotropy means.

**What it does and does not catch, measured.** On 512 samples in 64 dimensions,
against a true isotropic Gaussian scoring 0.00016 -- which is exactly the
theoretical 1/(12n) for a correct sample:

| representation            | score  | vs target |
|---------------------------|--------|-----------|
| isotropic Gaussian        | 0.00016|   1.0x    |
| low-rank, 2 of 64 dims    | 0.00911|  57.4x    |
| low-rank, 16 of 64        | 0.00107|   6.8x    |
| heavy-tailed              | 0.00062|   3.9x    |
| collapsed to a point      | 0.00015|   0.9x    |
| uniform (wrong shape)     | 0.00013|   0.8x    |

Rank collapse -- the failure mode this project actually keeps hitting, where the
conditioning carries a rate and nothing else -- is detected strongly. Collapse to a
single point is **not**, because the global rescaling stretches a point cloud back
to unit variance before the test sees it; and a uniform distribution slips through
because a projection of many independent uniforms is Gaussian by the CLT regardless.
Neither is a reason to rescale differently: per-coordinate standardisation is what
made the first version blind to rank collapse as well, scoring 0.000114 on a
collapsed embedding against 0.000112 on a real Gaussian. Total collapse is already
prevented by the generative loss, which a constant representation cannot satisfy.
"""

import math

import torch


def sigreg(embeddings, n_projections=64, generator=None):
    """Cramer-von Mises distance from isotropic Gaussian, over random directions.

    `embeddings` is `(n, d)` -- flatten any sequence axis into `n` before calling.
    Returns a scalar that is zero only when every projection is standard normal.
    """
    if embeddings.dim() != 2:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    n_samples, dimension = embeddings.shape
    if n_samples < 4:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    # Centred, then scaled by ONE scalar for the whole representation.
    #
    # This is the part that has to be right and was not. Standardising per dimension
    # -- and then again per projection -- destroys exactly the information the test
    # is for: any projection of a low-rank Gaussian is still Gaussian, so after
    # re-standardising, a collapsed representation is indistinguishable from an
    # isotropic one. Measured, the first version scored 0.000114 on a collapsed
    # embedding and 0.000112 on a genuine Gaussian, i.e. it detected nothing.
    #
    # A single global scale keeps anisotropy visible: directions the representation
    # does not use project to near-zero variance, and a unit-variance reference then
    # sees that immediately.
    centred = embeddings - embeddings.mean(dim=0, keepdim=True)
    # `pow(2).mean()` averages over samples AND coordinates, so this sets the mean
    # per-coordinate variance to 1. A projection onto a unit direction then has
    # variance 1 exactly when the covariance is isotropic -- no dimension factor
    # belongs here, and putting one in scaled every projection by sqrt(d) and made
    # even a true Gaussian score 0.058 against a theoretical 1/(12n) ~ 0.00016.
    centred = centred / centred.pow(2).mean().sqrt().clamp_min(1e-6)

    directions = torch.randn(dimension, n_projections, device=embeddings.device, dtype=embeddings.dtype, generator=generator)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    projected = centred @ directions  # (n, n_projections), unit variance iff isotropic

    # Cramer-von Mises: sort, compare the empirical CDF to Phi at the order
    # statistics. Sorting is differentiable through the gather it implies.
    ordered, _indices = torch.sort(projected, dim=0)
    normal_cdf = 0.5 * (1.0 + torch.erf(ordered / math.sqrt(2.0)))
    positions = (torch.arange(n_samples, device=embeddings.device, dtype=embeddings.dtype) + 0.5) / n_samples
    statistic = (normal_cdf - positions.unsqueeze(-1)).pow(2).mean(dim=0)
    return statistic.mean()
