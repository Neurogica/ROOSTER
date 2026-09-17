"""UnifiedTokenizer: converts raw (value, t, dt, channel_id, modality_id,
task_id, observed_mask) patches into fused token embeddings shared by both
the forecasting and reconstruction sides of UniSignal-Flow.

Convention: one forward()/embed_query() call = one channel's window (matches
RevIN's per-window, per-channel normalization). Multi-channel batches are
formed by the caller -- either fold channel into the batch dim, or make
separate calls per channel and concatenate the resulting token streams
along the patch dimension N.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible instance normalization, per (window, channel) call.

    Stateless: normalize() returns the per-call (mean, std) alongside the
    normalized value instead of caching them on the module. A single RevIN
    instance is shared by one UnifiedTokenizer across multiple calls per
    training step (condition-side, then target-side) -- instance-cached
    stats would silently get overwritten by whichever call ran last and
    leak into a denormalize() for the wrong window. Callers that need to
    invert the normalization later must carry the returned stats
    themselves, scoped to the call they came from.
    """

    def __init__(self, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))

    def normalize(self, value, observed_mask):
        mask = observed_mask.float()
        count = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        mean = (value * mask).sum(dim=(1, 2), keepdim=True) / count
        var = ((value - mean) ** 2 * mask).sum(dim=(1, 2), keepdim=True) / count
        std = (var + self.eps).sqrt()
        out = (value - mean) / std
        if self.affine:
            out = out * self.weight + self.bias
        return out, mean, std

    def denormalize(self, value, mean, std):
        out = value
        if self.affine:
            out = (out - self.bias) / self.weight
        return out * std + mean


class PatchEmbed(nn.Module):
    """Shared linear projection from raw patch samples to d_model -- no
    per-domain feature engineering."""

    def __init__(self, patch_len, d_model):
        super().__init__()
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, value):
        return self.proj(value)


class MultiscalePatchEmbed(nn.Module):
    """Convolutional features at several scales, taken BEFORE patch folding.

    Why a linear patch embedding is not enough here, concretely: on the 125 Hz
    physio tasks the auto-chosen patch length is 10 samples and a QRS complex is
    roughly 10 samples wide, so the one event the task turns on lands inside a
    single patch and its sub-patch structure -- the biphasic spike shape that a
    detector keys on -- is flattened by the very first linear map. Every model in
    this project that recovers waveform morphology (PENGUIN r 0.161-0.276, a plain
    conv net 0.141-0.218) sees the sequence through convolutions at several widths
    before any pooling; the two that embed linearly per patch score r = 0.000.
    Capacity is not the difference -- doubling width (571k params) and depth (286k)
    both made things worse -- so the hypothesis is the input representation.

    Kernels at 3, 9 and 27 samples give three dyadic scales: sub-event, event, and
    event-context. The features are computed on the CONTINUOUS sequence (so kernels
    straddle patch boundaries -- the whole point) and only then folded into patches
    and projected. The raw-value projection is kept as a residual so this strictly
    extends PatchEmbed rather than replacing it, and the conv path is
    zero-initialised: an untrained model embeds exactly as the linear version.
    """

    def __init__(self, patch_len, d_model, widths=(3, 9, 27), channels=16):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)
        self.convs = nn.ModuleList(
            [nn.Conv1d(1, channels, kernel_size=w, padding=w // 2) for w in widths]
        )
        self.fuse = nn.Linear(len(widths) * channels * patch_len, d_model)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, value):
        # value: (B, n_patches, patch_len). Unfold to the continuous sequence so
        # the kernels can see across patch boundaries, then refold.
        batch, n_patches, patch_len = value.shape
        sequence = value.reshape(batch, 1, n_patches * patch_len)
        features = torch.cat([conv(sequence)[..., : n_patches * patch_len] for conv in self.convs], dim=1)
        features = features.reshape(batch, -1, n_patches, patch_len).permute(0, 2, 1, 3)
        features = features.reshape(batch, n_patches, -1)
        return self.proj(value) + self.fuse(features)


class ContinuousTimeEncoding(nn.Module):
    """Time2Vec-style encoding over continuous timestamps t (not an integer
    position index): one learnable linear term plus learnable-frequency
    sin/cos pairs."""

    def __init__(self, d_model):
        super().__init__()
        if d_model < 3:
            raise ValueError("d_model must be >= 3 for ContinuousTimeEncoding")
        self.n_freqs = (d_model - 1) // 2
        self.linear_weight = nn.Parameter(torch.randn(1) * 0.02)
        self.linear_bias = nn.Parameter(torch.zeros(1))
        self.freq = nn.Parameter(torch.randn(self.n_freqs) * 0.02 + 1.0)
        self.phase = nn.Parameter(torch.zeros(self.n_freqs))
        out_dim = 1 + 2 * self.n_freqs
        self.out_proj = nn.Linear(out_dim, d_model) if out_dim != d_model else nn.Identity()

    def forward(self, t):
        t = t.unsqueeze(-1)
        linear_term = t * self.linear_weight + self.linear_bias
        angles = t * self.freq + self.phase
        periodic = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        features = torch.cat([linear_term, periodic], dim=-1)
        return self.out_proj(features)


class SamplingRateEmbed(nn.Module):
    """Log-scaled sampling interval (dt) embedding via a small MLP."""

    def __init__(self, d_model, hidden=None):
        super().__init__()
        hidden = hidden or d_model
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, dt):
        log_dt = torch.log1p(dt.clamp_min(0).unsqueeze(-1))
        return self.net(log_dt)


class IdentityEmbeddings(nn.Module):
    """Channel/modality/task identity lookups. The task embedding is a SOFT
    PRIOR ONLY -- task identity is actually encoded upstream via the mask
    pattern (which patches are condition vs. target), not by this embedding.
    """

    def __init__(self, d_model, num_channels, num_modalities, num_tasks, task_embed_scale=0.1):
        super().__init__()
        self.channel_embed = nn.Embedding(num_channels, d_model)
        self.modality_embed = nn.Embedding(num_modalities, d_model)
        self.task_embed = nn.Embedding(num_tasks, d_model)
        self.task_embed_scale = task_embed_scale

    def forward(self, channel_id, modality_id, task_id):
        return self.channel_embed(channel_id) + self.modality_embed(modality_id) + self.task_embed_scale * self.task_embed(task_id)


class TargetTimeAligner(nn.Module):
    """Attention-based interpolation of condition tokens onto target
    timestamps. The same mechanism serves both extrapolation (forecasting:
    target timestamps lie beyond the condition window) and synchronous
    alignment (reconstruction: target timestamps share the condition
    window's span but a different channel) -- no separate code path per
    task.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, condition_tokens, target_query_tokens, key_padding_mask=None):
        aligned, _ = self.attn(
            query=target_query_tokens,
            key=condition_tokens,
            value=condition_tokens,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(aligned + target_query_tokens)


class UnifiedTokenizer(nn.Module):
    """Fuses raw patches into token embeddings: RevIN-normalized value +
    continuous-time encoding + sampling-rate embedding + channel/modality/
    task identity."""

    def __init__(self, d_model, patch_len, num_channels, num_modalities, num_tasks, task_embed_scale=0.1, revin_affine=True, multiscale=False):
        super().__init__()
        self.d_model = d_model
        self.revin = RevIN(affine=revin_affine)
        self.patch_embed = MultiscalePatchEmbed(patch_len, d_model) if multiscale else PatchEmbed(patch_len, d_model)
        self.time_encoding = ContinuousTimeEncoding(d_model)
        self.sampling_rate_embed = SamplingRateEmbed(d_model)
        self.identity_embed = IdentityEmbeddings(d_model, num_channels, num_modalities, num_tasks, task_embed_scale)
        self.norm = nn.LayerNorm(d_model)

    def _meta_embedding(self, batch):
        time_emb = self.time_encoding(batch["t"])
        rate_emb = self.sampling_rate_embed(batch["dt"])
        id_emb = self.identity_embed(batch["channel_id"], batch["modality_id"], batch["task_id"])
        return time_emb + rate_emb + id_emb

    def forward(self, batch):
        """Full tokens (value + metadata), plus this call's RevIN stats
        (mean, std) so the caller can invert the normalization later without
        relying on any state cached on self.revin -- see RevIN's docstring.
        `batch` must include 'value' and 'observed_mask' (B, N, P) in
        addition to the metadata keys ('t', 'dt', 'channel_id',
        'modality_id', 'task_id', each (B, N))."""
        normalized_value, revin_mean, revin_std = self.revin.normalize(batch["value"], batch["observed_mask"])
        value_emb = self.patch_embed(normalized_value)
        tokens = value_emb + self._meta_embedding(batch)
        return self.norm(tokens), (revin_mean, revin_std)

    def embed_query(self, batch):
        """Metadata-only tokens (no value term) -- used to build target-side
        queries for TargetTimeAligner when the target values are the thing
        being generated, not observed."""
        return self.norm(self._meta_embedding(batch))
