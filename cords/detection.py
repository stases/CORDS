"""MultiMNIST scenes and the CORDS image-to-field model.

Coordinates are pixel ``(x, y)`` pairs; boxes are ``(x0, y0, x1, y1)``
with exclusive upper bounds. Density is mass per pixel, so its *sum* estimates
object count. The ten class channels and two normalized-size channels are
density-weighted feature masses, not probabilities or raw sizes.

The ConvNeXt/FPN architecture is extracted from ``erwin/FullMultiMNIST.py``
(``LightConvNeXtFPNFields``); see the repository provenance document.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .fields import DecodedSet, FieldSamples


@dataclass
class DetectionScene:
    """A composed MNIST image and its source objects, on CPU."""

    image: Tensor
    positions: Tensor
    features: Tensor
    boxes: Tensor
    labels: Tensor
    source_indices: Tensor

    @property
    def centers(self) -> Tensor:
        """Alias spelling useful when discussing bounding boxes."""
        return self.positions

    @property
    def image_shape(self) -> tuple[int, int]:
        return tuple(self.image.shape[-2:])


@lru_cache(maxsize=4)
def _load_glyphs(glyph_path: str | None = None) -> tuple[np.ndarray, ...]:
    def read(path):
        with np.load(path, allow_pickle=False) as data:
            return tuple(data[key].copy() for key in ("images", "labels", "source_indices"))

    if glyph_path is None:
        images, labels, indices = read(Path(__file__).parent / "data/mnist_glyphs.npz")
    else:
        images, labels, indices = read(glyph_path)
    if images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) == 0:
        raise ValueError("MNIST glyph images must have shape [N, 28, 28], with N > 0")
    if labels.shape != (len(images),) or indices.shape != labels.shape:
        raise ValueError("MNIST glyph labels/source_indices must have shape [N]")
    if images.dtype != np.uint8 or np.any((labels < 0) | (labels > 9)):
        raise ValueError("Expected uint8 MNIST images and integer labels from zero to nine")
    if not np.all(images.reshape(len(images), -1).max(axis=1) > 0):
        raise ValueError("MNIST glyph fixture contains a blank image")
    return images, labels, indices


def make_scene(
    seed: int = 0,
    num_digits: int = 3,
    image_size: int = 128,
    overlap: bool = False,
    glyph_path: str | Path | None = None,
) -> DetectionScene:
    """Compose a reproducible scene from bundled, source-indexed MNIST glyphs.

    The separated layout uses jittered grid cells and never silently drops an
    object. ``overlap=True`` clusters objects near the image center to illustrate
    reconstruction under overlap. No global random state or downloads are used.
    ``glyph_path`` can select another NPZ with the same fixture schema.
    """
    if not isinstance(num_digits, (int, np.integer)) or num_digits < 0:
        raise ValueError("num_digits must be a nonnegative integer")
    if not isinstance(image_size, (int, np.integer)) or image_size < 32:
        raise ValueError("image_size must be an integer >= 32")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    boxes, labels, source_indices = [], [], []
    if num_digits:
        images, glyph_labels, glyph_indices = _load_glyphs(
            None if glyph_path is None else str(Path(glyph_path).resolve())
        )
        cols = math.ceil(math.sqrt(num_digits))
        rows = math.ceil(num_digits / cols)
        x_edges = np.linspace(0, image_size, cols + 1).astype(int)
        y_edges = np.linspace(0, image_size, rows + 1).astype(int)
        cells = rng.permutation(rows * cols)[:num_digits]
        if not overlap and min(np.diff(x_edges).min(), np.diff(y_edges).min()) < 10:
            raise ValueError("Too many separated digits for this image_size")

        for cell in cells:
            source = int(rng.integers(len(images)))
            glyph = Image.fromarray(images[source])
            glyph = glyph.crop(glyph.getbbox())
            side = int(rng.integers(20, 33))
            scale = side / max(glyph.size)
            glyph = glyph.resize(
                tuple(max(1, round(d * scale)) for d in glyph.size),
                resample=Image.Resampling.BILINEAR,
            ).rotate(float(rng.uniform(-20, 20)), resample=Image.Resampling.BICUBIC, expand=True)
            glyph = glyph.point(lambda p: 0 if p < 8 else p)
            glyph = glyph.crop(glyph.getbbox())
            if overlap:
                max_width = max_height = image_size - 8
            else:
                col, row = int(cell % cols), int(cell // cols)
                max_width = int(x_edges[col + 1] - x_edges[col] - 8)
                max_height = int(y_edges[row + 1] - y_edges[row] - 8)
            scale = min(1.0, max_width / glyph.width, max_height / glyph.height)
            if scale < 1:
                glyph = glyph.resize(
                    (max(1, int(glyph.width * scale)), max(1, int(glyph.height * scale))),
                    resample=Image.Resampling.BILINEAR,
                )
                glyph = glyph.crop(glyph.getbbox())
            width, height = glyph.size
            if overlap:
                center = rng.uniform(0.36, 0.64, size=2) * image_size
                left = int(np.clip(round(center[0] - width / 2), 0, image_size - width))
                top = int(np.clip(round(center[1] - height / 2), 0, image_size - height))
            else:
                left = int(rng.integers(x_edges[col] + 4, x_edges[col + 1] - width - 3))
                top = int(rng.integers(y_edges[row] + 4, y_edges[row + 1] - height - 3))
            patch = np.asarray(glyph, dtype=np.float32) / 255.0
            region = canvas[top:top + height, left:left + width]
            region[:] = 1 - (1 - region) * (1 - patch)
            boxes.append((left, top, left + width, top + height))
            labels.append(int(glyph_labels[source]))
            source_indices.append(int(glyph_indices[source]))

    box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    positions = (box_tensor[:, :2] + box_tensor[:, 2:]) / 2
    sizes = (box_tensor[:, 2:] - box_tensor[:, :2]) / image_size
    features = torch.cat((F.one_hot(label_tensor, num_classes=10).float(), sizes), dim=1)
    return DetectionScene(
        image=torch.from_numpy(canvas).unsqueeze(0), positions=positions,
        features=features, boxes=box_tensor, labels=label_tensor,
        source_indices=torch.tensor(source_indices, dtype=torch.long),
    )


class MultiMNISTDataset(Dataset):
    """Deterministic samples derived from seed, epoch, and sample index.

    Call ``set_epoch`` before iterating each training epoch. Create workers after
    that call (DataLoader's default ``persistent_workers=False``), so they see
    the epoch. Validation datasets ignore ``set_epoch``. Distinct worker RNG
    copies cannot duplicate the stream because sampling uses the item index.
    """

    def __init__(
        self, size: int = 1000, seed: int = 0, num_digits: tuple[int, int] = (1, 6),
        image_size: int = 128, overlap: bool = False, training: bool = True,
        glyph_path: str | Path | None = None,
    ):
        if size < 1 or seed < 0 or not 0 <= num_digits[0] <= num_digits[1]:
            raise ValueError("Invalid dataset size, seed, or num_digits range")
        self.size, self.seed, self.num_digits = int(size), int(seed), num_digits
        self.image_size, self.overlap, self.training = image_size, overlap, training
        self.glyph_path, self.epoch = glyph_path, 0

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch) if self.training else 0

    def __getitem__(self, index: int) -> DetectionScene:
        if not 0 <= index < self.size:
            raise IndexError(index)
        sequence = np.random.SeedSequence((self.seed, self.epoch, int(index)))
        scene_seed, count_seed = sequence.generate_state(2, dtype=np.uint64)
        count = int(np.random.default_rng(count_seed).integers(self.num_digits[0], self.num_digits[1] + 1))
        return make_scene(int(scene_seed), count, self.image_size, self.overlap, self.glyph_path)


def scene_to_fields(scene: DetectionScene, sigma: float = 2.0) -> FieldSamples:
    """Encode a scene; every Gaussian has exactly unit mass on the image grid."""
    from .reconstruction import CORDSTransform

    return CORDSTransform(sigma).encode_grid(scene.positions, scene.features, shape=scene.image_shape)


def fields_to_tensor(fields: FieldSamples) -> Tensor:
    """Convert grid samples to ``[density, class masses, size masses]`` maps."""
    if fields.grid_shape is None or len(fields.grid_shape) != 2:
        raise ValueError("Expected fields sampled on a two-dimensional image grid")
    height, width = fields.grid_shape
    values = torch.cat((fields.density.reshape(-1, 1), fields.features), dim=1)
    return values.T.reshape(-1, height, width)


def tensor_to_fields(maps: Tensor, sigma: float = 2.0) -> FieldSamples:
    """Wrap model output as pixel masses for the same decoder as analytic fields.

    No density rescaling, thresholding, suppression, or count information is
    introduced here. Pass a single image's unscaled physical field tensor.
    """
    from .fields import FieldSamples

    if maps.ndim != 3 or maps.shape[0] < 4:
        raise ValueError("Expected [1 + num_classes + 2, height, width] maps")
    if not maps.is_floating_point() or not torch.isfinite(maps).all() or (maps < 0).any():
        raise ValueError("Predicted field maps must be finite, nonnegative floating point values")
    _, height, width = maps.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=maps.device, dtype=maps.dtype),
        torch.arange(width, device=maps.device, dtype=maps.dtype), indexing="ij",
    )
    return FieldSamples(
        coordinates=torch.stack((xx, yy), dim=-1).reshape(-1, 2),
        density=maps[:1].reshape(1, -1).T,
        features=maps[1:].reshape(maps.shape[0] - 1, -1).T,
        sigma=sigma, weights=torch.ones(height * width, device=maps.device, dtype=maps.dtype),
        sampling="grid", grid_shape=(height, width), density_mode="pixel_mass",
    )


# This descriptive alias is convenient at model call sites.
predicted_fields = tensor_to_fields


def decode_boxes(decoded: DecodedSet, image_shape: tuple[int, int]) -> Tensor:
    """Convert recovered normalized sizes and pixel centers to unclipped boxes.

    Keeping boxes unclipped exposes reconstruction errors in diagnostics.
    Only plotting code should clip boxes to the visible image if desired.
    """
    height, width = image_shape
    if decoded.features.shape[-1] < 3 or decoded.positions.shape[-1] != 2:
        raise ValueError("Decoded detection features must include classes and two size channels")
    scale = decoded.features.new_tensor((width, height))
    half_sizes = decoded.features[:, -2:] * scale / 2
    return torch.cat((decoded.positions - half_sizes, decoded.positions + half_sizes), dim=-1)


class _SpatialNorm2d(nn.Module):
    """Historical model's per-channel spatial normalization, with affine terms."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        centered = x - x.mean(dim=(2, 3), keepdim=True)
        normalized = centered * torch.rsqrt(centered.square().mean(dim=(2, 3), keepdim=True) + 1e-6)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class _ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = _SpatialNorm2d(channels)
        self.expand = nn.Conv2d(channels, 4 * channels, 1)
        self.project = nn.Conv2d(4 * channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.project(F.gelu(self.expand(self.norm(self.depthwise(x)))))


def _stage(channels_in: int, channels_out: int, depth: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 2, stride=2), _SpatialNorm2d(channels_out),
        *(_ConvNeXtBlock(channels_out) for _ in range(depth)),
    )


class CORDSDetector(nn.Module):
    """Light ConvNeXt/FPN predicting full-resolution CORDS pixel masses.

    Input: grayscale images ``[B, 1, H, W]`` in [0, 1], H/W >= 16.
    Output: ``[B, 1 + num_classes + 2, H, W]`` physical field masses.
    Class masses sum to density; normalized-size masses lie between zero and
    density. There is no fixed object-query count or suppression stage.

    A small initial density bias gives approximately three objects per image.
    Correcting the bias for input area avoids thousands of inferred objects on
    an untrained model. This normalization is part of the model, so callers
    must not apply the historical experiment's ``rho_rescale`` on decoding.
    """

    def __init__(self, base_channels: int = 32, num_classes: int = 10):
        super().__init__()
        if base_channels < 4 or num_classes < 1:
            raise ValueError("base_channels must be >= 4 and num_classes >= 1")
        self.base_channels, self.num_classes = int(base_channels), int(num_classes)
        base = self.base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, base, 4, stride=4), _SpatialNorm2d(base), _ConvNeXtBlock(base)
        )
        self.stage1, self.stage2 = _stage(base, 2 * base, 2), _stage(2 * base, 4 * base, 3)
        self.lateral0 = nn.Conv2d(base, base, 1)
        self.lateral1 = nn.Conv2d(2 * base, base, 1)
        self.lateral2 = nn.Conv2d(4 * base, base, 1)
        self.smooth = nn.Conv2d(base, base, 3, padding=1)
        self.head = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1), nn.GroupNorm(math.gcd(base, 8), base),
            nn.GELU(), nn.Conv2d(base, 1 + self.num_classes + 2, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[0] = math.log(math.expm1(3.0 / (128 * 128)))
            self.head[-1].bias[-2:] = math.log(0.2 / 0.8)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 1 or min(images.shape[-2:]) < 16:
            raise ValueError("Expected grayscale images [B, 1, H, W] with H,W >= 16")
        c0 = self.stem(images)
        c1 = self.stage1(c0)
        c2 = self.stage2(c1)
        p1 = self.lateral1(c1) + F.interpolate(self.lateral2(c2), size=c1.shape[-2:], mode="nearest")
        p0 = self.lateral0(c0) + F.interpolate(p1, size=c0.shape[-2:], mode="nearest")
        logits = self.head(F.interpolate(self.smooth(p0), size=images.shape[-2:], mode="bilinear", align_corners=False))
        area_correction = math.log(images.shape[-2] * images.shape[-1] / (128 * 128))
        density = F.softplus(logits[:, :1] - area_correction)
        classes = F.softmax(logits[:, 1:1 + self.num_classes], dim=1)
        sizes = torch.sigmoid(logits[:, -2:])
        return torch.cat((density, density * classes, density * sizes), dim=1)
