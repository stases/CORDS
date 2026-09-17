"""Acceptance checks on the actual bundled teaching examples, without plotting."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
import torch

from cords.detection import decode_boxes, make_scene, scene_to_fields
from cords.molecules import load_qm9_examples, reconstruction_metrics
from cords.reconstruction import CORDSTransform


@pytest.mark.parametrize("name,source_index", [("gdb_14", 13), ("gdb_19", 18), ("gdb_826", 825)])
def test_real_qm9_importance_roundtrip(name, source_index):
    examples = {molecule.name: molecule for molecule in load_qm9_examples()}
    assert set(examples) == {"gdb_14", "gdb_19", "gdb_826"}
    molecule = examples[name]
    assert molecule.source_index == source_index
    transform = CORDSTransform(sigma=0.2, seed=0)
    fields = transform.encode(molecule.positions, molecule.features, num_samples=1024)
    # Ground-truth positions/features enter only the forward transform. The
    # decoder receives neither the original object count nor center guesses.
    with torch.no_grad():
        decoded = transform.decode(fields)
    metrics = reconstruction_metrics(molecule, decoded)
    assert metrics["count_correct"], metrics
    assert metrics["type_accuracy"] == 1.0, metrics
    assert metrics["position_rms_angstrom"] < 0.02, metrics


@pytest.mark.parametrize("num_digits", [3, 6, 16])
def test_real_mnist_separated_roundtrip(num_digits):
    scene = make_scene(seed=0, num_digits=num_digits, image_size=128, overlap=False)
    fields = scene_to_fields(scene, sigma=2.0)
    with torch.no_grad():
        decoded = CORDSTransform(sigma=2.0, seed=0).decode(fields)
    assert decoded.count == num_digits
    assert scene.source_indices.shape == (num_digits,)
    assert (scene.source_indices >= 0).all()

    distances = torch.cdist(scene.positions, decoded.positions).numpy()
    true_indices, predicted_indices = linear_sum_assignment(distances)
    np.testing.assert_array_equal(
        scene.labels[true_indices].numpy(),
        decoded.features[predicted_indices, :10].argmax(-1).numpy(),
    )
    true_boxes = scene.boxes[true_indices]
    predicted_boxes = decode_boxes(decoded, scene.image_shape)[predicted_indices]
    intersection_sides = (
        torch.minimum(true_boxes[:, 2:], predicted_boxes[:, 2:])
        - torch.maximum(true_boxes[:, :2], predicted_boxes[:, :2])
    ).clamp_min(0)
    intersection = intersection_sides.prod(-1)
    true_area = (true_boxes[:, 2:] - true_boxes[:, :2]).prod(-1)
    predicted_area = (predicted_boxes[:, 2:] - predicted_boxes[:, :2]).prod(-1)
    iou = intersection / (true_area + predicted_area - intersection)
    assert torch.all(iou > 0.95), iou.tolist()
