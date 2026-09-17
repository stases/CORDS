"""Gaussian set encoding and reconstruction derived from Erwin's transforms."""

import math
import torch

from .fields import DecodedSet, FieldSamples
from .gmm import FixedSigmaGMM


def gaussian_basis(coordinates: torch.Tensor, positions: torch.Tensor, sigma: float,
                   *, pixel_mass: bool = False) -> torch.Tensor:
    """Return [samples, objects] Gaussian kernels, optionally grid-normalized."""
    squared = ((coordinates[:, None, :] - positions[None, :, :]) ** 2).sum(-1)
    logits = -squared / (2 * sigma ** 2)
    if pixel_mass:
        return torch.softmax(logits, dim=0)
    return torch.exp(logits) / (math.sqrt(2 * math.pi) * sigma) ** coordinates.shape[1]


class CORDSTransform:
    """A single-example Gaussian transform with explicit sampling conventions.

    ``sigma`` uses the same units as positions (angstroms or pixels). Decoding
    infers count from fields; neither ground-truth count nor centers are inputs.
    ``max_count`` is an explicit resource guard, never a silent count clamp.
    """

    def __init__(self, sigma: float = .2, seed: int = 0, *, max_count: int = 256,
                 ridge: float = 1e-7, refinement_steps: int = 100):
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        if max_count < 1 or ridge < 0 or refinement_steps < 1:
            raise ValueError("invalid count guard, ridge, or refinement_steps")
        self.sigma, self.seed = float(sigma), seed
        self.max_count, self.ridge, self.refinement_steps = max_count, ridge, refinement_steps

    @staticmethod
    def _check_objects(positions, features):
        if positions.ndim != 2 or positions.shape[1] < 1 or not positions.is_floating_point():
            raise ValueError("positions must be a floating [objects, dimensions] tensor")
        if features.ndim != 2 or len(features) != len(positions) or not features.is_floating_point():
            raise ValueError("features must be a floating [objects, channels] tensor")
        if features.device != positions.device:
            raise ValueError("positions and features must use the same device")
        if not torch.isfinite(positions).all() or not torch.isfinite(features).all():
            raise ValueError("positions and features must be finite")

    def evaluate(self, positions, features, coordinates):
        """Evaluate physical density and feature fields at supplied coordinates."""
        self._check_objects(positions, features)
        basis = gaussian_basis(coordinates, positions, self.sigma)
        return basis.sum(-1, keepdim=True), basis @ features.to(basis.dtype)

    def encode(self, positions, features, num_samples=1024, sampling="importance",
               density_mode="physical") -> FieldSamples:
        """Sample q(x)=N⁻¹Σ Kσ(x−rᵢ), retaining q and 1/(S q) quadrature weights.

        In count_normalized mode only density is divided by its sample mean/N;
        feature fields keep physical units, matching the historical model.
        """
        self._check_objects(positions, features)
        if not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if sampling != "importance":
            raise ValueError("encode supports sampling='importance'; use encode_grid for images")
        if density_mode not in {"physical", "count_normalized"}:
            raise ValueError("molecular density_mode must be physical or count_normalized")
        generator = torch.Generator(device=positions.device).manual_seed(self.seed)
        centers = positions if len(positions) else positions.new_zeros((1, positions.shape[1]))
        indices = torch.randint(len(centers), (num_samples,), generator=generator, device=positions.device)
        coordinates = centers[indices] + self.sigma * torch.randn(
            (num_samples, positions.shape[1]), generator=generator,
            device=positions.device, dtype=positions.dtype)
        density, field_features = self.evaluate(positions, features, coordinates)
        proposal = gaussian_basis(coordinates, centers, self.sigma).mean(-1)
        weights = 1 / (num_samples * proposal)
        scale = 1.0
        if density_mode == "count_normalized" and len(positions):
            scale = float(density.detach().mean() / len(positions))
            density = density / scale
        return FieldSamples(coordinates, density, field_features, self.sigma, weights,
                            sampling, density_mode=density_mode, density_scale=scale,
                            proposal_density=proposal)

    def encode_grid(self, positions, features, shape=(64, 64)) -> FieldSamples:
        """Encode XY pixel centers on a row-major (height, width) image grid.

        Each Gaussian sums to one over the finite grid, including boundaries.
        Consequently sum(density) is count at every image resolution.
        """
        self._check_objects(positions, features)
        if len(shape) != 2 or any(not isinstance(x, int) or x < 1 for x in shape):
            raise ValueError("shape must be a positive (height, width) pair")
        if positions.shape[1] != 2:
            raise ValueError("image positions must be XY pairs")
        height, width = shape
        if len(positions) and ((positions < 0).any() or (positions[:, 0] > width - 1).any() or (positions[:, 1] > height - 1).any()):
            raise ValueError("image centers must be inside the pixel grid")
        y, x = torch.meshgrid(torch.arange(height, device=positions.device, dtype=positions.dtype),
                              torch.arange(width, device=positions.device, dtype=positions.dtype), indexing="ij")
        coordinates = torch.stack((x, y), -1).reshape(-1, 2)
        basis = gaussian_basis(coordinates, positions, self.sigma, pixel_mass=True)
        return FieldSamples(coordinates, basis.sum(-1, keepdim=True), basis @ features.to(basis.dtype),
                            self.sigma, torch.ones(height * width, device=positions.device, dtype=positions.dtype),
                            "grid", tuple(shape), "pixel_mass")

    def decode(self, fields: FieldSamples, refine=True) -> DecodedSet:
        """Infer count, fit centers, and recover attributes by weighted Gram solve.

        Refinement enables its own gradient context, so decode is also safe
        inside torch.no_grad(). Reconstruction is not differentiable as a whole.
        """
        fields.__post_init__()  # Validate mutable fields again, including predictions.
        estimate = fields.count_estimate()
        if not math.isfinite(estimate):
            raise ValueError("nonfinite count estimate")
        count = int(math.floor(estimate + .5))
        if count > self.max_count:
            raise ValueError(f"inferred count {count} exceeds explicit max_count={self.max_count}")
        if count > len(fields.coordinates):
            raise ValueError("inferred count exceeds the number of field samples")
        dtype, device = fields.coordinates.dtype, fields.coordinates.device
        diagnostics = {"count_estimate": estimate, "density_mode": fields.density_mode, "refined": bool(refine)}
        if count == 0:
            diagnostics.update(density_relative_error=0.0 if not fields.density.any() else 1.0,
                               feature_relative_error=0.0 if not fields.features.any() else 1.0)
            return DecodedSet(torch.empty((0, fields.coordinates.shape[1]), dtype=dtype, device=device),
                              torch.empty((0, fields.features.shape[1]), dtype=dtype, device=device), 0, diagnostics)
        coordinates = fields.coordinates.detach().double()
        density = fields.density.detach().double()
        feature_fields = fields.features.detach().double()
        pixel_mass = fields.density_mode == "pixel_mass"
        scale_unknown = fields.density_mode == "count_normalized" and fields.density_scale is None
        if fields.density_mode == "count_normalized" and not scale_unknown:
            density = density * fields.density_scale
        weights = None if fields.weights is None else fields.weights.detach().double()
        if pixel_mass:
            initial = self._peak_centers(coordinates, density[:, 0], count, fields.sigma)
        else:
            # Importance correction gives mass in the original integration measure.
            # Unknown model proposals use a sharpened density for peak initialization.
            sample_weight = (density[:, 0] * weights if weights is not None
                             else (density[:, 0] / density.max()).pow(4))
            mixture = FixedSigmaGMM(count, fields.sigma, seed=self.seed).fit(
                coordinates.cpu().numpy(), sample_weight.cpu().numpy())
            initial = torch.as_tensor(mixture.means_, dtype=torch.float64, device=device)
            diagnostics["gmm_converged"] = mixture.converged_
        positions = initial
        amplitude = 1.0
        if refine:
            positions, amplitude = self._refine(coordinates, density, initial, fields.sigma,
                                               pixel_mass, scale_unknown)
        elif scale_unknown:
            initial_density = gaussian_basis(coordinates, positions, fields.sigma).sum(-1, keepdim=True)
            amplitude = float((density * initial_density).sum() / initial_density.square().sum())
        basis = gaussian_basis(coordinates, positions, fields.sigma, pixel_mass=pixel_mass)
        if weights is None:
            if pixel_mass:
                weights = torch.ones(len(coordinates), dtype=torch.float64, device=device)
            else:
                proposal = (basis.sum(-1) / count).clamp_min(torch.finfo(torch.float64).tiny)
                weights = 1 / (len(coordinates) * proposal)
                diagnostics["weights_estimated_from_centers"] = True
        # Normalize quadrature weights by a common scalar for numerical stability.
        # This leaves the regularized projection invariant.
        weights = weights / weights.max()
        gram = basis.T @ (basis * weights[:, None])
        rhs = basis.T @ (feature_fields * weights[:, None])
        penalty = self.ridge * gram.diag().mean()
        regularized = gram + penalty * torch.eye(count, dtype=torch.float64, device=device)
        try:
            features = torch.linalg.solve(regularized, rhs)
        except RuntimeError as error:
            raise RuntimeError("feature projection failed; try a smaller bandwidth or better field sampling") from error
        predicted_density = amplitude * basis.sum(-1, keepdim=True)
        tiny = torch.finfo(torch.float64).tiny
        diagnostics.update(
            initial_positions=initial.to(dtype), density_amplitude=amplitude,
            gram_condition_number=float(torch.linalg.cond(regularized)),
            density_relative_error=float(torch.linalg.vector_norm(predicted_density - density)
                                         / torch.linalg.vector_norm(density).clamp_min(tiny)),
            feature_relative_error=float(torch.linalg.vector_norm(basis @ features - feature_fields)
                                         / torch.linalg.vector_norm(feature_fields).clamp_min(tiny)))
        if not torch.isfinite(positions).all() or not torch.isfinite(features).all():
            raise RuntimeError("nonfinite reconstruction; inspect fields and bandwidth")
        return DecodedSet(positions.to(dtype), features.to(dtype), count, diagnostics)

    @staticmethod
    def _peak_centers(coordinates, density, count, sigma):
        scores = density.clone()
        selected = []
        excluded = torch.zeros(len(coordinates), dtype=torch.bool, device=coordinates.device)
        for _ in range(count):
            index = int(scores.argmax())
            if not torch.isfinite(scores[index]):
                raise ValueError("too many centers to initialize at this image bandwidth")
            center = coordinates[index]
            selected.append(center)
            distance = (coordinates - center).square().sum(-1)
            # Remove one object's mass before selecting the next peak. Merely
            # masking peak pixels can select the shoulder of a boundary kernel
            # twice because finite-grid normalization makes edge peaks taller.
            scores = scores - gaussian_basis(coordinates, center[None], sigma, pixel_mass=True)[:, 0]
            excluded |= distance <= sigma ** 2
            scores[excluded] = -torch.inf
        return torch.stack(selected)

    def _refine(self, coordinates, density, initial, sigma, pixel_mass, learn_amplitude):
        with torch.enable_grad():
            positions = initial.detach().clone().requires_grad_(True)
            basis = gaussian_basis(coordinates, positions, sigma, pixel_mass=pixel_mass)
            initial_amplitude = (density.mean() / basis.sum(-1).mean()).detach() if learn_amplitude else density.new_tensor(1.)
            log_amplitude = initial_amplitude.log().requires_grad_(learn_amplitude)
            parameters = [positions] + ([log_amplitude] if learn_amplitude else [])
            optimizer = torch.optim.LBFGS(parameters, lr=1, max_iter=self.refinement_steps,
                                         tolerance_grad=1e-10, tolerance_change=1e-12,
                                         history_size=20, line_search_fn="strong_wolfe")
            def closure():
                with torch.enable_grad():
                    optimizer.zero_grad(set_to_none=True)
                    prediction = log_amplitude.exp() * gaussian_basis(
                        coordinates, positions, sigma, pixel_mass=pixel_mass).sum(-1, keepdim=True)
                    if pixel_mass:
                        loss = (prediction - density).square().mean() / density.square().mean().clamp_min(1e-20)
                    else:
                        floor = density.max().detach() * 1e-10
                        loss = (prediction.clamp_min(floor).log() - density.clamp_min(floor).log()).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("center refinement produced nonfinite loss")
                    loss.backward()
                    return loss
            optimizer.step(closure)
        return positions.detach(), float(log_amplitude.detach().exp())
