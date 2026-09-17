"""Numerical contracts for the two field measures and inverse transform."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from cords.fields import FieldSamples
from cords.gmm import FixedSigmaGMM
from cords.reconstruction import CORDSTransform


def objects():
    positions = torch.tensor([[0., 0., 0.], [1.2, .1, 0.], [-.2, 1.3, .2], [.1, -.1, 1.4]], dtype=torch.float64)
    return positions, torch.eye(4, dtype=torch.float64)


def match(positions, decoded):
    distance = torch.cdist(positions, decoded.positions)
    indices = distance.argmin(-1)
    assert len(indices.unique()) == len(positions)
    return distance[torch.arange(len(positions)), indices], indices


def test_importance_weights_recover_count_and_record_proposal():
    positions, features = objects()
    transform = CORDSTransform(.2)
    fields = transform.encode(positions, features, 1024)
    assert fields.count_estimate() == pytest.approx(4, abs=1e-12)
    assert torch.allclose(fields.weights * fields.proposal_density * 1024, torch.ones(1024, dtype=torch.float64))
    # Mean density is not a physical count under nonuniform importance sampling.
    assert abs(float(fields.density.mean()) - 4) > .1


def test_roundtrip_without_ground_truth_and_inside_no_grad():
    positions, features = objects()
    transform = CORDSTransform(.2)
    fields = transform.encode(positions, features, 1024)
    with torch.no_grad():
        decoded = transform.decode(fields)
    errors, indices = match(positions, decoded)
    assert decoded.count == 4
    assert errors.max() < 1e-4
    assert torch.allclose(decoded.features[indices], features, atol=1e-5)
    assert decoded.diagnostics["density_relative_error"] < 1e-4


def test_count_normalized_density_keeps_physical_features():
    positions, features = objects()
    transform = CORDSTransform(.2)
    physical = transform.encode(positions, features)
    normalized = transform.encode(positions, features, density_mode="count_normalized")
    assert normalized.count_estimate() == pytest.approx(4)
    assert torch.allclose(physical.features, normalized.features)
    assert torch.allclose(physical.density, normalized.density * normalized.density_scale)
    decoded = transform.decode(normalized)
    errors, indices = match(positions, decoded)
    assert errors.max() < 1e-4
    assert torch.allclose(decoded.features[indices], features, atol=1e-5)


def test_generated_count_normalized_fields_need_no_original_scale_or_proposal():
    positions, features = objects()
    transform = CORDSTransform(.2)
    fields = transform.encode(positions, features, density_mode="count_normalized")
    generated = replace(fields, weights=None, proposal_density=None, density_scale=None)
    decoded = transform.decode(generated)
    errors, indices = match(positions, decoded)
    assert errors.max() < 1e-4
    assert torch.allclose(decoded.features[indices], features, atol=1e-5)
    assert decoded.diagnostics["weights_estimated_from_centers"]


def test_encoding_is_permutation_invariant_at_fixed_coordinates():
    positions, features = objects()
    transform = CORDSTransform(.2)
    coordinates = transform.encode(positions, features).coordinates
    permutation = torch.tensor([3, 0, 2, 1])
    before = transform.evaluate(positions, features, coordinates)
    after = transform.evaluate(positions[permutation], features[permutation], coordinates)
    for first, second in zip(before, after):
        assert torch.allclose(first, second, atol=1e-12)


@pytest.mark.parametrize("resolution", [32, 64])
def test_grid_mass_boundaries_resolution_and_roundtrip(resolution):
    scale = resolution / 32
    positions = torch.tensor([[0., 0.], [15.2, 16.3], [30., 29.]], dtype=torch.float64) * scale
    features = torch.tensor([[1., 0., .2], [0., 1., .3], [1., 0., .4]], dtype=torch.float64)
    transform = CORDSTransform(1.2 * scale)
    fields = transform.encode_grid(positions, features, shape=(resolution, resolution))
    assert fields.count_estimate() == pytest.approx(3, abs=1e-12)
    assert torch.equal(fields.coordinates[1], torch.tensor([1., 0.], dtype=torch.float64))
    assert torch.equal(fields.coordinates[resolution], torch.tensor([0., 1.], dtype=torch.float64))
    decoded = transform.decode(fields)
    errors, indices = match(positions, decoded)
    assert errors.max() < 1e-4
    assert torch.allclose(decoded.features[indices], features, atol=1e-5)


def test_empty_sets_do_not_turn_into_one_object():
    transform = CORDSTransform(.2)
    for fields in [transform.encode(torch.empty(0, 3), torch.empty(0, 5)),
                   transform.encode_grid(torch.empty(0, 2), torch.empty(0, 12), shape=(8, 8))]:
        decoded = transform.decode(fields)
        assert decoded.count == 0
        assert decoded.positions.shape == (0, fields.coordinates.shape[1])
        assert decoded.features.shape == (0, fields.features.shape[1])


def test_safety_guard_raises_instead_of_clamping_count():
    positions, features = objects()
    fields = CORDSTransform(.2).encode(positions, features)
    with pytest.raises(ValueError, match="exceeds explicit max_count"):
        CORDSTransform(.2, max_count=3).decode(fields)


def test_malformed_or_nonfinite_fields_raise():
    with pytest.raises(ValueError, match="integration weights"):
        FieldSamples(torch.zeros(2, 3), torch.ones(2, 1), torch.zeros(2, 4), .2)
    positions, features = objects()
    fields = CORDSTransform(.2).encode(positions, features)
    with pytest.raises(ValueError, match="finite floating"):
        replace(fields, density=fields.density * float("nan"))
    with pytest.raises(ValueError, match="positive with shape"):
        replace(fields, weights=torch.zeros_like(fields.weights))


def test_fixed_sigma_em_recovers_two_weighted_clouds():
    rng = np.random.default_rng(5)
    samples = np.concatenate([rng.normal(-2, .1, (200, 1)), rng.normal(2, .1, (200, 1))])
    fitted = FixedSigmaGMM(2, .1, seed=3).fit(samples, np.r_[np.ones(200), np.full(200, 2.)])
    order = fitted.means_[:, 0].argsort()
    assert np.allclose(fitted.means_[order, 0], [-2, 2], atol=.03)
    assert np.allclose(fitted.weights_[order], [1/3, 2/3], atol=.01)
    assert np.isfinite(fitted.score_samples(samples)).all()
