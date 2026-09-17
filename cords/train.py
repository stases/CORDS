"""Optional, dependency-light overfit examples and reusable training command.

``python -m cords.train qm9 --steps 100 --output runs/qm9`` uses bundled
examples. Pass ``--full-data --data /path/to/molecules.json`` to cycle through
your own molecular records. MNIST ``--data`` accepts the local glyph archive
format documented in the README. Nothing is downloaded automatically.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from .models import CORDSMoleculeDenoiser, edm_loss


@dataclass
class TrainConfig:
    task: str = "qm9"
    steps: int = 100
    output: str = "runs/qm9"
    data: str | None = None
    device: str = "cpu"
    seed: int = 0
    batch_size: int = 3
    learning_rate: float = 0.001
    width: int = 32
    field_samples: int = 128
    field_sigma: float = 0.35
    coordinate_scale: float = 5.0
    log_density_scale: float = 10.0
    feature_scale: float = 5.0
    density_weight: float = 4.0
    count_weight: float = 1.0
    image_size: int = 96
    image_sigma: float = 2.0
    num_digits: int = 3
    overfit: bool = True
    resume: str | None = None
    threads: int = 2


def model_output_to_fields(coordinates, values, config):
    """Undo molecular model preprocessing; generated fields have no true rho scale.

    Their mean count-normalized density estimates the count. No original
    molecule, proposal density, or atom count is supplied to reconstruction.
    """
    from .fields import FieldSamples
    if isinstance(config, dict):
        config = TrainConfig(**config)
    if coordinates.ndim not in (2, 3) or values.ndim != coordinates.ndim:
        raise ValueError("coordinates and values must both be [S,C] or [B,S,C] tensors")
    if coordinates.shape[:-1] != values.shape[:-1] or coordinates.shape[-1] != 3 or values.shape[-1] != 6:
        raise ValueError("Expected matching sample batches with 3 coordinates and 6 QM9 field channels")
    if coordinates.ndim == 2:
        coordinates, values = coordinates[None], values[None]
    outputs = []
    for pos, val in zip(coordinates.detach(), values.detach()):
        density = (val[:, :1] * config.log_density_scale).exp()
        if not torch.isfinite(density).all():
            raise FloatingPointError("Generated log-density overflowed. Inspect training before reconstruction.")
        outputs.append(FieldSamples(
            coordinates=pos * config.coordinate_scale,
            density=density,
            features=val[:, 1:] * config.feature_scale,
            sigma=config.field_sigma,
            sampling="importance",
            density_mode="count_normalized",
            density_scale=None,
        ))
    return outputs


def _molecule_batch(molecules, config, indices, seed):
    from .reconstruction import CORDSTransform
    coordinates, values = [], []
    for index in indices:
        molecule = molecules[index]
        fields = CORDSTransform(sigma=config.field_sigma, seed=seed + index).encode(
            molecule.positions, molecule.features,
            num_samples=config.field_samples, sampling="importance",
            density_mode="count_normalized",
        )
        pos = fields.coordinates.float() / config.coordinate_scale
        pos = pos - pos.mean(0, keepdim=True)
        val = torch.cat((fields.density.float().clamp_min(1e-12).log() / config.log_density_scale,
                         fields.features.float() / config.feature_scale), -1)
        coordinates.append(pos)
        values.append(val)
    return (torch.stack(coordinates).to(config.device), torch.stack(values).to(config.device))


def detection_loss(prediction, target, *, count_weight=1.0):
    """Scale-balanced dense field loss with an explicit per-image count term."""
    # Normalize each channel family by its target RMS so background does not
    # make tiny mass-valued fields appear to have an already-perfect MSE.
    def relative_mse(start, stop):
        reference = target[:, start:stop]
        return (prediction[:, start:stop] - reference).square().mean() / reference.square().mean().clamp_min(1e-12)
    density = relative_mse(0, 1)
    classes = relative_mse(1, 11)
    sizes = relative_mse(11, 13)
    target_count, predicted_count = target[:, 0].sum((-2, -1)), prediction[:, 0].sum((-2, -1))
    count = ((predicted_count - target_count) / target_count.clamp_min(1)).square().mean()
    return density + classes + sizes + count_weight * count, {"density": density, "classes": classes, "sizes": sizes, "count": count}


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _make_model(config):
    if config.task == "qm9":
        return CORDSMoleculeDenoiser(width=config.width)
    if config.task == "mnist":
        from .detection import CORDSDetector
        return CORDSDetector(base_channels=config.width)
    raise ValueError("task must be 'qm9' or 'mnist'.")


def load_checkpoint(path, *, device="cpu"):
    """Load a CORDS checkpoint with tensor-only deserialization."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "cords-training-v1":
        raise ValueError("Expected a CORDS teaching-release checkpoint.")
    model = _make_model(TrainConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def train(config: TrainConfig | dict):
    """Train on a fixed example batch, or cycle local data when overfit=False.

    ``steps`` is the total optimizer-step target, including resumed steps.
    Resume retains saved training options and data source; only steps, output,
    device and resume location come from the new invocation.
    """
    if isinstance(config, dict):
        config = TrainConfig(**config)
    payload = None
    if config.resume:
        _, payload = load_checkpoint(config.resume, device="cpu")
        saved = payload["config"]
        if saved["task"] != config.task:
            raise ValueError("Checkpoint task and requested task differ.")
        overrides = {name: getattr(config, name) for name in ("steps", "output", "device", "resume")}
        config = TrainConfig(**{**saved, **overrides})
    if config.steps < 1 or config.batch_size < 1 or config.learning_rate <= 0 or config.threads < 1:
        raise ValueError("steps, batch_size, learning_rate and threads must be positive.")
    if min(config.coordinate_scale, config.log_density_scale, config.feature_scale) <= 0:
        raise ValueError("Model preprocessing scales must be positive.")
    if config.density_weight < 0 or config.count_weight < 0:
        raise ValueError("Loss weights must be nonnegative.")
    if not config.overfit and not config.data:
        raise ValueError("Full-data training requires an explicit --data path.")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    generator = torch.Generator(device=device).manual_seed(config.seed + 17)
    model = _make_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    history, start_step = [], 0
    if payload:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        # Optimizer.load_state_dict places moment tensors on their parameter
        # devices and preserves CPU scalar steps for non-capturable AdamW.
        start_step, history = payload["step"], payload["history"]
        if config.steps <= start_step:
            raise ValueError("--steps must exceed the checkpoint's completed step count.")
        if payload["generator_device"] != device.type:
            raise ValueError("Resume on the same device type to preserve the random stream.")
        generator.set_state(payload["generator_state"].cpu())

    model.eval()

    if config.task == "qm9":
        from .molecules import load_qm9
        molecules = load_qm9(config.data)
        if not molecules:
            raise ValueError("The molecule dataset is empty.")
        size = len(molecules)
        count = min(config.batch_size, size)
        fixed_batch = _molecule_batch(molecules, config, range(count), config.seed)
        evaluation_generator = torch.Generator(device=device).manual_seed(config.seed + 991)
        with torch.no_grad():
            _, _, evaluation_noise = edm_loss(model, *fixed_batch, generator=evaluation_generator, density_weight=config.density_weight)
        def batch_for(step):
            if config.overfit:
                return fixed_batch
            indices = [(step * count + i) % size for i in range(count)]
            return _molecule_batch(molecules, config, indices, config.seed + step * size)
        def objective(batch, evaluation=False):
            return edm_loss(model, *batch, generator=generator,
                            noisy_batch=evaluation_noise if evaluation else None,
                            density_weight=config.density_weight)[:2]
        def predict():
            return _cpu(model(*evaluation_noise))
        inputs, targets = _cpu(evaluation_noise), _cpu(fixed_batch)
    else:
        from .detection import MultiMNISTDataset, fields_to_tensor, scene_to_fields
        dataset = MultiMNISTDataset(
            size=max(config.batch_size * 64, config.batch_size), seed=config.seed,
            num_digits=(config.num_digits if config.overfit else 1, config.num_digits), image_size=config.image_size,
            training=True, glyph_path=config.data,
        )
        def make_batch(indices):
            scenes = [dataset[index] for index in indices]
            images = torch.stack([scene.image for scene in scenes]).float().to(device)
            target = torch.stack([fields_to_tensor(scene_to_fields(scene, sigma=config.image_sigma)) for scene in scenes]).float().to(device)
            return images, target
        fixed_batch = make_batch(range(config.batch_size))
        def batch_for(step):
            if config.overfit:
                return fixed_batch
            dataset.set_epoch(step // 64)
            indices = [(step * config.batch_size + i) % len(dataset) for i in range(config.batch_size)]
            return make_batch(indices)
        def objective(batch, evaluation=False):
            return detection_loss(model(batch[0]), batch[1], count_weight=config.count_weight)
        def predict():
            return _cpu(model(fixed_batch[0]))
        inputs, targets = _cpu(fixed_batch[0]), _cpu(fixed_batch[1])

    model.eval()
    with torch.no_grad():
        before = predict()
        initial_loss, _ = objective(fixed_batch, evaluation=True)
    if not history:
        history.append({"step": 0, "eval_loss": float(initial_loss)})
    for step in range(start_step, config.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, components = objective(batch_for(step))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {config.task} loss at step {step + 1}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            evaluation_loss, _ = objective(fixed_batch, evaluation=True)
        history.append({"step": step + 1, "train_loss": float(loss.detach()), "eval_loss": float(evaluation_loss),
                        **{name: float(value.detach()) for name, value in components.items()}})
    with torch.no_grad():
        after = predict()
    output = Path(config.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    state = {"format": "cords-training-v1", "config": asdict(config), "step": config.steps,
             "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
             "generator_state": generator.get_state(), "generator_device": device.type}
    temporary = output / "checkpoint.tmp"
    torch.save(state, temporary)
    temporary.replace(checkpoint)
    (output / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    return {"model": model, "history": history, "before": before, "after": after,
            "inputs": inputs, "targets": targets, "checkpoint": str(checkpoint), "config": asdict(config)}


def train_overfit(task="qm9", *, steps=100, output=None, **kwargs):
    """Notebook-friendly opt-in training, fixed data and fixed evaluation noise."""
    return train(TrainConfig(task=task, steps=steps, output=str(output or f"runs/{task}"), overfit=True, **kwargs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("qm9", "mnist"))
    parser.add_argument("--config", type=Path, help="JSON TrainConfig options; explicit CLI options override them")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output")
    parser.add_argument("--data")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--full-data", action="store_true", help="Use changing batches from the local dataset")
    args = vars(parser.parse_args())
    config_path = args.pop("config")
    full_data = args.pop("full_data")
    options = json.loads(config_path.read_text()) if config_path else {}
    options.update({name: value for name, value in args.items() if value is not None})
    options.setdefault("output", f"runs/{args['task']}")
    if full_data:
        options["overfit"] = False
    result = train(options)
    print(json.dumps({"checkpoint": result["checkpoint"], "initial_eval_loss": result["history"][0]["eval_loss"],
                      "final_eval_loss": result["history"][-1]["eval_loss"]}, indent=2))


if __name__ == "__main__":
    main()
