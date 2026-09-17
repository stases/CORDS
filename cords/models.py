"""Portable molecular denoising model extracted from the Erwin experiments.

The RFF/FiLM fusion and head derive from ``erwin/models/qm9.py``; ball
attention, SwiGLU, pooling and unpooling derive from ``erwin/models/erwin.py``.
The compact two-level backbone preserves those operations. Its deterministic
PyTorch widest-axis median tree replaces the original Cython tree builder.
Tie ordering differs, rotation and message passing are omitted, and input
sample counts must be powers of two. Historical checkpoints are not compatible.
See docs/provenance.md and THIRD_PARTY.md for source attribution.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MoleculeModelConfig:
    feature_dim: int = 5
    width: int = 32
    heads: int = 4
    ball_size: int = 8
    depth: int = 1
    sigma_coords: float = 0.35
    sigma_log_density: float = 0.2
    sigma_features: float = 0.2


class _RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


@torch.no_grad()
def _tree_order(positions):
    """Batch complete binary trees, splitting each node on its widest axis."""
    batch, samples, _ = positions.shape
    if samples < 2 or samples & (samples - 1):
        raise ValueError("Erwin sample counts must be powers of two (at least 2).")
    indices = torch.arange(samples, device=positions.device).expand(batch, -1)
    width = samples
    while width > 1:
        groups = indices.reshape(batch, -1, width)
        points = positions.gather(1, indices[..., None].expand(-1, -1, 3))
        points = points.reshape(batch, -1, width, 3)
        axes = (points.amax(-2) - points.amin(-2)).argmax(-1)
        projected = points.gather(-1, axes[..., None, None].expand(-1, -1, width, 1))[..., 0]
        ordering = projected.argsort(dim=-1, stable=True)
        indices = groups.gather(-1, ordering).reshape(batch, samples)
        width //= 2
    return indices


def _gather(x, order):
    return x.gather(1, order[..., None].expand(-1, -1, x.shape[-1]))


class _BallBlock(nn.Module):
    def __init__(self, width, heads, ball_size):
        super().__init__()
        self.heads, self.ball_size = heads, ball_size
        self.norm1, self.norm2 = _RMSNorm(width), _RMSNorm(width)
        self.pe = nn.Linear(3, width)
        self.qkv, self.proj = nn.Linear(width, width * 3), nn.Linear(width, width)
        self.distance_scale = nn.Parameter(-1 + 0.01 * torch.randn(1, heads, 1, 1))
        self.w1, self.w2, self.w3 = nn.Linear(width, width * 4), nn.Linear(width, width * 4), nn.Linear(width * 4, width)

    def forward(self, x, positions):
        batch, samples, width = x.shape
        if samples % self.ball_size:
            raise ValueError("Each tree level must be divisible by ball_size.")
        points = positions.detach().reshape(-1, self.ball_size, 3)
        relative = points - points.mean(1, keepdim=True)
        features = self.norm1(x).reshape(-1, self.ball_size, width) + self.pe(relative)
        qkv = self.qkv(features).reshape(-1, self.ball_size, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.distance_scale * torch.cdist(points, points).unsqueeze(1)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        attended = attended.transpose(1, 2).reshape(batch, samples, width)
        x = x + self.proj(attended)
        normalized = self.norm2(x)
        return x + self.w3(F.silu(self.w1(normalized)) * self.w2(normalized))


class _ErwinBackbone(nn.Module):
    """One encoder, one bottleneck and one decoder, with stride-two pooling."""
    def __init__(self, config):
        super().__init__()
        width = config.width
        self.embed = nn.Linear(width, width)
        self.encoder = nn.ModuleList([_BallBlock(width, config.heads, config.ball_size) for _ in range(config.depth)])
        self.bottleneck = nn.ModuleList([_BallBlock(width * 2, config.heads, config.ball_size) for _ in range(config.depth)])
        self.decoder = nn.ModuleList([_BallBlock(width, config.heads, config.ball_size) for _ in range(config.depth)])
        self.pool = nn.Linear(2 * width + 6, width * 2)
        self.unpool = nn.Linear(width * 2 + 6, width * 2)
        self.pool_norm, self.unpool_norm = nn.BatchNorm1d(width * 2), nn.BatchNorm1d(width)

    def forward(self, x, positions):
        batch, samples, width = x.shape
        order = _tree_order(positions)
        x, positions = _gather(self.embed(x), order), _gather(positions, order)
        for block in self.encoder:
            x = block(x, positions)
        skip = x
        with torch.no_grad():
            children = positions.reshape(batch, samples // 2, 2, 3)
            centers = children.mean(-2)
            relative = (children - centers[..., None, :]).reshape(batch, samples // 2, 6)
        pooled = self.pool(torch.cat((x.reshape(batch, samples // 2, 2 * width), relative), -1))
        x = self.pool_norm(pooled.reshape(-1, width * 2)).reshape(batch, samples // 2, width * 2)
        for block in self.bottleneck:
            x = block(x, centers)
        x = skip + self.unpool(torch.cat((x, relative), -1)).reshape(batch, samples, width)
        x = self.unpool_norm(x.reshape(-1, width)).reshape(batch, samples, width)
        for block in self.decoder:
            x = block(x, positions)
        return _gather(x, order.argsort(-1))


class CORDSMoleculeDenoiser(nn.Module):
    """Joint coordinate/log-density/feature EDM denoiser with an Erwin backbone.

    Inputs have shapes ``[B,S,3]``, ``[B,S,1+C]`` and ``sigma[B]``;
    unbatched fields are accepted too. Returned tensors have the input shapes.
    Values contain scaled log count-normalized density, then scaled physical
    feature fields. Normalization is performed by :mod:`cords.train`.
    """
    def __init__(self, config: MoleculeModelConfig | dict | None = None, **kwargs):
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("Pass a config or keyword model options, not both.")
        self.config = config if isinstance(config, MoleculeModelConfig) else MoleculeModelConfig(**(config or kwargs))
        c = self.config
        if c.width < 8 or c.heads < 1 or c.feature_dim < 1 or c.width % 8 or c.width % c.heads or c.ball_size < 2 or c.ball_size & (c.ball_size - 1) or c.depth < 1:
            raise ValueError("width must be divisible by 8 and heads; ball_size must be a power of two and depth positive.")
        if min(c.sigma_coords, c.sigma_log_density, c.sigma_features) <= 0:
            raise ValueError("EDM data scales must be positive.")
        self.frequencies = nn.Parameter(torch.randn(c.width // 4, 3) / 0.15)
        self.register_buffer("sigma_frequencies", torch.randn(16) * 0.2)
        self.feature_embedding = nn.Sequential(nn.Linear(c.feature_dim + 1, c.width // 4), nn.SiLU(), nn.Linear(c.width // 4, c.width // 4))
        self.sigma_embedding = nn.Sequential(nn.Linear(32, c.width // 4), nn.SiLU(), nn.Linear(c.width // 4, c.width // 4))
        self.fusion = nn.Linear(c.width, c.width)
        self.film = nn.Sequential(nn.Linear(1, c.width * 2), nn.SiLU(), nn.Linear(c.width * 2, c.width * 2))
        self.backbone = _ErwinBackbone(c)
        self.head = nn.Sequential(nn.Linear(c.width, c.width), nn.GELU(), nn.Linear(c.width, c.feature_dim + 4))

    def get_config(self):
        return asdict(self.config)

    def forward(self, coordinates, values, sigma):
        unbatched = coordinates.ndim == 2
        if unbatched:
            coordinates, values = coordinates[None], values[None]
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or values.shape != (*coordinates.shape[:2], self.config.feature_dim + 1):
            raise ValueError("Expected coordinates[B,S,3] and values[B,S,1+feature_dim].")
        if coordinates.shape[1] < 2 * self.config.ball_size:
            raise ValueError("Use at least twice ball_size field samples.")
        sigma = torch.as_tensor(sigma, device=coordinates.device, dtype=coordinates.dtype).reshape(-1, 1, 1)
        if sigma.numel() not in (1, coordinates.shape[0]) or not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("sigma must be positive, finite and scalar or one per molecule.")
        data_scale = (self.config.sigma_coords * self.config.sigma_log_density * self.config.sigma_features) ** (1 / 3)
        denominator = sigma.square() + data_scale ** 2
        c_in = denominator.rsqrt()
        pos_in, val_in = coordinates * c_in, values * c_in
        log_sigma = (sigma.log() / 4).expand(*coordinates.shape[:2], 1)
        angles = F.linear(pos_in, self.frequencies)
        pos_embedding = torch.cat((angles.sin(), angles.cos()), -1) * math.sqrt(4 / self.config.width)
        sigma_angles = log_sigma * self.sigma_frequencies
        sigma_embedding = self.sigma_embedding(torch.cat((sigma_angles.sin(), sigma_angles.cos()), -1))
        hidden = self.fusion(torch.cat((pos_embedding, self.feature_embedding(val_in), sigma_embedding), -1))
        gamma, beta = self.film(log_sigma).chunk(2, -1)
        prediction = self.head(self.backbone(gamma * hidden + beta, pos_in))
        # Preserve the original GraphField_QM9.py residual EDM parameterization.
        residual_pos, residual_values = prediction[..., :3], prediction[..., 3:]
        c_skip, c_out = data_scale ** 2 / denominator, sigma * data_scale * c_in
        denoised_pos = c_skip * coordinates + c_out * (pos_in - residual_pos)
        denoised_values = c_skip * values + c_out * (val_in - residual_values)
        if unbatched:
            return denoised_pos[0], denoised_values[0]
        return denoised_pos, denoised_values


def edm_loss(model, clean_coordinates, clean_values, *, generator=None, noisy_batch=None,
             density_weight=4.0, noise_mean=-1.2, noise_std=1.2):
    """Weighted EDM objective and a reusable noisy batch for fixed-input checks."""
    if noisy_batch is None:
        sigma = (torch.randn((clean_coordinates.shape[0],), generator=generator, device=clean_coordinates.device) * noise_std + noise_mean).exp()
        position_noise = torch.randn(clean_coordinates.shape, device=clean_coordinates.device, generator=generator)
        position_noise -= position_noise.mean(1, keepdim=True)
        value_noise = torch.randn(clean_values.shape, device=clean_values.device, generator=generator)
        noisy_batch = (clean_coordinates + position_noise * sigma[:, None, None], clean_values + value_noise * sigma[:, None, None], sigma)
    positions, values, sigma = noisy_batch
    predicted_pos, predicted_values = model(positions, values, sigma)
    c = model.config
    def weighted(predicted, target, scale):
        weight = (sigma.square() + scale ** 2) / (sigma * scale).square()
        return (weight[:, None, None] * (predicted - target).square()).mean()
    coord_loss = weighted(predicted_pos, clean_coordinates, c.sigma_coords)
    density_loss = weighted(predicted_values[..., :1], clean_values[..., :1], c.sigma_log_density)
    feature_loss = weighted(predicted_values[..., 1:], clean_values[..., 1:], c.sigma_features)
    loss = coord_loss + density_weight * density_loss + feature_loss
    return loss, {"coordinates": coord_loss, "log_density": density_loss, "features": feature_loss}, noisy_batch


@torch.no_grad()
def sample_molecular_fields(model, *, num_samples=128, batch_size=1, steps=18,
                            sigma_min=0.002, sigma_max=10.0, rho=7.0, seed=0):
    """Unconditional Karras schedule with deterministic second-order Heun steps."""
    if steps < 2 or not 0 < sigma_min < sigma_max or rho <= 0 or batch_size < 1:
        raise ValueError("Require steps>=2, batch_size>=1, rho>0 and 0<sigma_min<sigma_max.")
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    schedule = torch.linspace(0, 1, steps, device=device, dtype=dtype)
    schedule = (sigma_max ** (1 / rho) + schedule * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    schedule = torch.cat((schedule, schedule.new_zeros(1)))
    coordinates = torch.randn((batch_size, num_samples, 3), device=device, dtype=dtype, generator=generator) * sigma_max
    coordinates -= coordinates.mean(1, keepdim=True)
    values = torch.randn((batch_size, num_samples, model.config.feature_dim + 1), device=device, dtype=dtype, generator=generator) * sigma_max
    was_training = model.training
    model.eval()
    try:
        for current, following in zip(schedule[:-1], schedule[1:]):
            clean_pos, clean_values = model(coordinates, values, current)
            d_pos, d_values = (coordinates - clean_pos) / current, (values - clean_values) / current
            next_pos, next_values = coordinates + (following - current) * d_pos, values + (following - current) * d_values
            if following > 0:
                clean_pos, clean_values = model(next_pos, next_values, following)
                next_pos = coordinates + (following - current) * (d_pos + (next_pos - clean_pos) / following) / 2
                next_values = values + (following - current) * (d_values + (next_values - clean_values) / following) / 2
            coordinates, values = next_pos, next_values
        coordinates -= coordinates.mean(1, keepdim=True)
        if not torch.isfinite(coordinates).all() or not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite molecular sample; inspect model training and noise schedule.")
        return coordinates, values
    finally:
        model.train(was_training)
