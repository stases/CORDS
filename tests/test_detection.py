"""Scene sampling, physical field interfaces, and differentiable prediction."""

import numpy as np
import pytest
import torch

from cords.detection import (
    CORDSDetector, MultiMNISTDataset, decode_boxes, fields_to_tensor,
    make_scene, scene_to_fields, tensor_to_fields,
)
from cords.reconstruction import CORDSTransform


@pytest.fixture
def glyph_path(tmp_path):
    """Small explicit sprites isolate composition checks from fixture packaging."""
    images = np.zeros((10, 28, 28), dtype=np.uint8)
    for label in range(10):
        images[label, 3:25, 4:8 + label] = 255
        images[label, 18:23, 4:24] = 180
    path = tmp_path / "glyphs.npz"
    np.savez(path, images=images, labels=np.arange(10), source_indices=np.arange(100, 110))
    return path


def test_scene_seed_and_box_contract(glyph_path):
    first = make_scene(1, 6, glyph_path=glyph_path)
    repeated = make_scene(1, 6, glyph_path=glyph_path)
    different = make_scene(2, 6, glyph_path=glyph_path)
    assert torch.equal(first.image, repeated.image)
    assert torch.equal(first.boxes, repeated.boxes)
    assert not torch.equal(first.image, different.image)
    assert first.image.shape == (1, 128, 128)
    assert first.image.min() >= 0 and first.image.max() <= 1
    assert first.features.shape == (6, 12)
    torch.testing.assert_close(first.features[:, :10].sum(-1), torch.ones(6))
    assert first.boxes.min() >= 0 and first.boxes.max() <= 128
    for i, box in enumerate(first.boxes):
        for other in first.boxes[i + 1:]:
            intersection = (torch.minimum(box[2:], other[2:]) - torch.maximum(box[:2], other[:2])).clamp_min(0)
            assert intersection.prod() == 0
    assert torch.all((first.source_indices >= 100) & (first.source_indices < 110))


def test_dataset_epoch_and_index_are_reproducible(glyph_path):
    train = MultiMNISTDataset(size=5, glyph_path=glyph_path)
    validation = MultiMNISTDataset(size=5, glyph_path=glyph_path, training=False)
    epoch0 = train[0].image.clone()
    assert not torch.equal(train[0].image, train[1].image)
    train.set_epoch(1)
    assert not torch.equal(epoch0, train[0].image)
    train.set_epoch(0)
    assert torch.equal(epoch0, train[0].image)
    validation.set_epoch(8)
    assert torch.equal(epoch0, validation[0].image)
    with pytest.raises(IndexError):
        train[5]


def _collate_images(scenes):
    return torch.stack([scene.image for scene in scenes])


def test_worker_count_does_not_change_samples(glyph_path):
    dataset = MultiMNISTDataset(size=4, num_digits=(2, 2), glyph_path=glyph_path)
    dataset.set_epoch(3)
    serial = torch.stack([dataset[i].image for i in range(len(dataset))])
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=_collate_images)
    parallel = torch.cat(list(loader))
    assert torch.equal(serial, parallel)


def test_analytic_and_model_field_interfaces_share_decoder(glyph_path):
    scene = make_scene(7, 3, image_size=64, glyph_path=glyph_path)
    analytic = scene_to_fields(scene, sigma=1.8)
    maps = fields_to_tensor(analytic)
    assert maps.shape == (13, 64, 64)
    torch.testing.assert_close(maps[0].sum(), torch.tensor(3.0))
    wrapped = tensor_to_fields(maps, sigma=1.8)
    torch.testing.assert_close(analytic.coordinates, wrapped.coordinates)
    torch.testing.assert_close(analytic.density, wrapped.density)
    torch.testing.assert_close(analytic.features, wrapped.features)
    with torch.no_grad():
        decoded = CORDSTransform(1.8).decode(wrapped)
    assert decoded.count == 3
    predicted_boxes = decode_boxes(decoded, scene.image_shape)
    distance = torch.cdist(decoded.positions, scene.positions)
    matching = distance.argmin(dim=1)
    assert len(matching.unique()) == 3
    assert decoded.features[:, :10].argmax(-1).tolist() == scene.labels[matching].tolist()
    torch.testing.assert_close(predicted_boxes, scene.boxes[matching], atol=0.03, rtol=0)


def test_empty_scene(glyph_path):
    scene = make_scene(num_digits=0, glyph_path=glyph_path)
    assert scene.boxes.shape == (0, 4)
    assert scene.features.shape == (0, 12)
    assert scene.image.count_nonzero() == 0
    assert fields_to_tensor(scene_to_fields(scene)).count_nonzero() == 0


def test_detector_mass_constraints_gradients_and_non_square_output():
    torch.manual_seed(4)
    model = CORDSDetector(base_channels=4)
    output = model(torch.rand(2, 1, 32, 48))
    assert output.shape == (2, 13, 32, 48)
    assert torch.isfinite(output).all() and (output > 0).all()
    torch.testing.assert_close(output[:, 1:11].sum(1), output[:, 0])
    assert torch.all(output[:, -2:] <= output[:, :1])
    counts = output[:, 0].sum(dim=(1, 2))
    assert torch.all((counts > 1) & (counts < 6))
    output.square().mean().backward()
    assert model.stem[0].weight.grad is not None
    assert torch.isfinite(model.stem[0].weight.grad).all()
    assert model.stem[0].weight.grad.abs().sum() > 0


def test_bundled_scene_runs_without_download():
    scene = make_scene(seed=23)
    assert scene.boxes.shape == (3, 4)
    assert scene.source_indices.shape == (3,)
