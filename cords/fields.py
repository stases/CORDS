"""Single-example fields, with their integration convention made explicit."""

from dataclasses import dataclass, field
import math
from typing import Any

import torch


@dataclass
class FieldSamples:
    """CORDS samples, without the original object positions or count.

    ``coordinates`` is [S, d], ``density`` is [S, 1], and ``features`` is
    [S, C]. For importance samples ``weights = 1 / (S * proposal_density)``.
    Pixel fields instead contain mass per pixel, with unit integration weights.

    ``count_normalized`` is the historical molecular model convention: only
    density is scaled to have sample mean N; feature fields remain physical.
    ``density_scale`` converts stored density back to physical density. It can
    be None for model predictions, in which case reconstruction fits amplitude.
    """

    coordinates: torch.Tensor
    density: torch.Tensor
    features: torch.Tensor
    sigma: float
    weights: torch.Tensor | None = None
    sampling: str = "importance"
    grid_shape: tuple[int, int] | None = None
    density_mode: str = "physical"
    density_scale: float | None = 1.0
    proposal_density: torch.Tensor | None = None

    def __post_init__(self):
        if self.coordinates.ndim != 2 or not self.coordinates.is_floating_point():
            raise ValueError("coordinates must be a floating [samples, dimensions] tensor")
        samples = len(self.coordinates)
        if not samples or self.coordinates.shape[1] < 1:
            raise ValueError("provide at least one sample and spatial dimension")
        if self.density.shape != (samples, 1):
            raise ValueError("density must have shape [samples, 1]")
        if self.features.ndim != 2 or self.features.shape[0] != samples:
            raise ValueError("features must have shape [samples, channels]")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        for name in ("coordinates", "density", "features", "weights", "proposal_density"):
            value = getattr(self, name)
            if value is not None:
                if not value.is_floating_point() or not torch.isfinite(value).all():
                    raise ValueError(f"{name} must contain finite floating values")
                if value.device != self.coordinates.device:
                    raise ValueError("all field tensors must use the same device")
        if (self.density < 0).any():
            raise ValueError("density must be nonnegative")
        for name in ("weights", "proposal_density"):
            value = getattr(self, name)
            if value is not None and (value.shape != (samples,) or (value <= 0).any()):
                raise ValueError(f"{name} must be positive with shape [samples]")
        if self.density_mode not in {"physical", "count_normalized", "pixel_mass"}:
            raise ValueError("unknown density_mode")
        if self.density_scale is not None and (
            not math.isfinite(self.density_scale) or self.density_scale <= 0
        ):
            raise ValueError("density_scale must be positive, finite, or None")
        if self.density_mode == "physical" and self.weights is None:
            raise ValueError("physical fields require integration weights")
        if self.density_mode == "pixel_mass":
            if self.grid_shape is None or math.prod(self.grid_shape) != samples:
                raise ValueError("pixel_mass fields require the matching grid_shape")
            if len(self.grid_shape) != 2 or min(self.grid_shape) <= 0 or self.coordinates.shape[1] != 2:
                raise ValueError("grid_shape must be a positive (height, width) pair")

    def count_estimate(self) -> float:
        """Return the unrounded count using this field's declared convention."""
        density = self.density.detach().double().squeeze(-1)
        if self.density_mode == "count_normalized":
            return float(density.mean())
        if self.density_mode == "pixel_mass":
            return float(density.sum())
        return float((density * self.weights.double()).sum())


@dataclass
class DecodedSet:
    """Recovered object centers and attributes, in arbitrary object order."""

    positions: torch.Tensor
    features: torch.Tensor
    count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
