"""Small, explicit QM9 records without a dataset-framework dependency."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ELEMENTS = ("H", "C", "N", "O", "F")
ATOMIC_NUMBERS = (1, 6, 7, 8, 9)


@dataclass
class Molecule:
    name: str
    source_index: int
    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    bonds: torch.Tensor
    bond_types: torch.Tensor

    @property
    def features(self) -> torch.Tensor:
        vocabulary = self.atomic_numbers.new_tensor(ATOMIC_NUMBERS)
        out = self.atomic_numbers[:, None] == vocabulary[None, :]
        if not out.any(-1).all():
            raise ValueError("QM9 features support H, C, N, O and F only")
        return out.to(self.positions.dtype)


def load_qm9(path: str | Path | None = None) -> list[Molecule]:
    """Read portable JSON records (Å), or a trusted local QM9 raw cache.

    No download or preprocessing cache is created. The optional .pt adapter reads
    a list of raw QM9 dictionaries; never load .pt files from untrusted sources.
    """
    source = Path(__file__).parent / "data/qm9_examples.json" if path is None else Path(path)
    if str(source).endswith(".pt"):
        rows = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            raise ValueError("Expected the raw QM9 list-of-dictionaries cache")
        records = [record_from_raw(row) for row in rows]
    else:
        with source.open() as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and payload.get("units", "angstrom") != "angstrom":
            raise ValueError("QM9 positions must be in angstrom")
        records = payload["molecules"] if isinstance(payload, dict) else payload
    out = []
    for row in records:
        positions = torch.tensor(row["positions"], dtype=torch.float64)
        z = torch.tensor(row["atomic_numbers"], dtype=torch.long)
        if positions.shape != (len(z), 3) or not torch.isfinite(positions).all():
            raise ValueError("Each molecule needs finite [N,3] coordinates")
        molecule = Molecule(row["name"], int(row.get("source_index", -1)), positions, z,
                            torch.tensor(row.get("bonds", []), dtype=torch.long).reshape(-1, 2),
                            torch.tensor(row.get("bond_types", []), dtype=torch.long))
        _ = molecule.features
        out.append(molecule)
    return out


def load_qm9_examples() -> list[Molecule]:
    return load_qm9()


def record_from_raw(row: dict) -> dict:
    """Preserve original QM9 identifiers and geometry; don't use PyG's 11-D x."""
    edges = torch.as_tensor(row["edge_index"])
    keep = edges[0] < edges[1]
    attr = torch.as_tensor(row["edge_attr"])
    return {"name": row["name"], "source_index": int(row["idx"]),
            "positions": torch.as_tensor(row["pos"]).tolist(),
            "atomic_numbers": torch.as_tensor(row["z"]).tolist(),
            "bonds": edges[:, keep].T.tolist(),
            "bond_types": attr[keep].argmax(-1).tolist()}


def reconstruction_metrics(molecule: Molecule, decoded) -> dict:
    """Permutation matching measures coordinates directly, without alignment."""
    true = molecule.positions.detach().cpu().numpy()
    pred = decoded.positions.detach().cpu().numpy()
    result = {"true_count": len(true), "decoded_count": decoded.count,
              "count_correct": len(true) == decoded.count}
    if not len(pred) or not len(true):
        return {**result, "position_rms_angstrom": float("nan"),
                "max_position_error_angstrom": float("nan"), "type_accuracy": 0.0}
    cost = np.linalg.norm(true[:, None] - pred[None, :], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    errors = cost[rows, cols]
    types = decoded.features.detach().cpu().argmax(-1).numpy()
    gt = molecule.features.argmax(-1).numpy()
    return {**result, "position_rms_angstrom": float(np.sqrt(np.mean(errors**2))),
            "max_position_error_angstrom": float(errors.max()),
            "type_accuracy": float((types[cols] == gt[rows]).sum() / len(true))}
