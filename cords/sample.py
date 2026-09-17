"""Sample learned molecular fields from a local CORDS QM9 checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from .models import sample_molecular_fields
from .train import load_checkpoint, model_output_to_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sigma-max", type=float, default=10.0)
    parser.add_argument("--decode", action="store_true", help="Attempt atom reconstruction and record diagnostics")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    if checkpoint["config"]["task"] != "qm9":
        raise ValueError("cords.sample expects a QM9 denoiser checkpoint.")
    coordinates, values = sample_molecular_fields(model, num_samples=checkpoint["config"]["field_samples"],
                                                  batch_size=args.batch_size, steps=args.steps,
                                                  sigma_max=args.sigma_max, seed=args.seed)
    fields = model_output_to_fields(coordinates, values, checkpoint["config"])
    output = {"format": "cords-molecular-samples-v1", "config": checkpoint["config"],
              "seed": args.seed, "sampler_steps": args.steps, "sigma_max": args.sigma_max,
              "fields": [{"coordinates": field.coordinates.cpu(), "density": field.density.cpu(),
                           "features": field.features.cpu(), "sigma": field.sigma,
                           "density_mode": "count_normalized", "density_scale": None,
                           "sampling": "importance"} for field in fields]}
    if args.decode:
        from .reconstruction import CORDSTransform
        results = []
        for field in fields:
            try:
                decoded = CORDSTransform(sigma=field.sigma).decode(field)
                results.append({"positions": decoded.positions.cpu(), "features": decoded.features.cpu(),
                                "diagnostics": decoded.diagnostics})
            except (ValueError, RuntimeError, FloatingPointError) as error:
                results.append({"error": str(error)})
        output["decoded"] = results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps({"output": str(args.output), "samples": len(fields),
                      "estimated_counts": [float(field.density.mean()) for field in fields]}))


if __name__ == "__main__":
    main()
