"""Optional training regression checks on tiny, local fixed batches."""

from pathlib import Path
import tempfile
import unittest

import torch

from cords.models import CORDSMoleculeDenoiser, edm_loss, sample_molecular_fields
from cords.train import load_checkpoint, model_output_to_fields, train, train_overfit


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_erwin_permutation_equivariance_and_gradients(self):
        torch.manual_seed(7)
        model = CORDSMoleculeDenoiser(width=16).eval()
        positions, values = torch.randn(2, 32, 3), torch.randn(2, 32, 6)
        permutation = torch.randperm(32)
        normal = model(positions, values, 0.3)
        permuted = model(positions[:, permutation], values[:, permutation], 0.3)
        for first, second in zip(normal, permuted):
            torch.testing.assert_close(first[:, permutation], second, rtol=1e-5, atol=1e-6)
        loss, _, _ = edm_loss(model, positions, values)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.backbone.encoder[0].qkv.weight.grad.abs().sum()), 0)

    def test_sampler_is_seeded_and_restores_mode(self):
        torch.manual_seed(3)
        model = CORDSMoleculeDenoiser(width=16).train()
        first = sample_molecular_fields(model, num_samples=32, steps=3, seed=11)
        second = sample_molecular_fields(model, num_samples=32, steps=3, seed=11)
        self.assertTrue(model.training)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(first[0].mean(1), torch.zeros(1, 3), atol=1e-6, rtol=0)
        with self.assertRaises(ValueError):
            sample_molecular_fields(model, num_samples=30, steps=3)

    def test_qm9_overfit_reload_and_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            result = train_overfit("qm9", steps=35, output=directory, width=16,
                                   field_samples=64, batch_size=2)
            self.assertLess(result["history"][-1]["eval_loss"], result["history"][0]["eval_loss"])
            model, payload = load_checkpoint(result["checkpoint"])
            with torch.no_grad():
                restored = model(*result["inputs"])
            for a, b in zip(restored, result["after"]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            fields = model_output_to_fields(*restored, result["config"])
            self.assertIsNone(fields[0].density_scale)
            self.assertIsNone(fields[0].proposal_density)
            self.assertEqual(fields[0].density_mode, "count_normalized")
            self.assertEqual(payload["step"], 35)

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            options = {"task": "qm9", "steps": 8, "width": 16, "field_samples": 32, "batch_size": 1}
            first = train({**options, "steps": 4, "output": str(Path(directory) / "part")})
            resumed = train({**options, "resume": first["checkpoint"], "output": str(Path(directory) / "resumed")})
            uninterrupted = train({**options, "output": str(Path(directory) / "whole")})
            for key, value in resumed["model"].state_dict().items():
                torch.testing.assert_close(value, uninterrupted["model"].state_dict()[key], atol=0, rtol=0)
            self.assertEqual(resumed["history"], uninterrupted["history"])

    def test_mnist_overfit_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            result = train_overfit("mnist", steps=25, output=directory,
                                   width=8, batch_size=2, image_size=64)
            self.assertLess(result["history"][-1]["eval_loss"], result["history"][0]["eval_loss"])
            self.assertIn("count", result["history"][-1])
            model, _ = load_checkpoint(result["checkpoint"])
            with torch.no_grad():
                restored = model(result["inputs"])
            torch.testing.assert_close(restored, result["after"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
