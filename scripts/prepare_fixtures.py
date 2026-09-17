"""Extract tiny source-indexed fixtures from existing caches; never downloads."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

# Direct script execution starts with scripts/ on sys.path, not the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cords.molecules import record_from_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qm9", required=True, type=Path)
    parser.add_argument("--mnist", required=True, type=Path, help="MNIST raw directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = torch.load(args.qm9, weights_only=False, map_location="cpu")
    by_name = {row["name"]: row for row in raw}
    records = [record_from_raw(by_name[name]) for name in ("gdb_14", "gdb_19", "gdb_826")]
    (args.output / "qm9_examples.json").write_text(json.dumps({
        "dataset": "QM9", "units": "angstrom", "source": "QM9 raw qm9_v3.pt",
        "molecules": records}, indent=2) + "\n")
    images = np.fromfile(args.mnist / "train-images-idx3-ubyte", dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    labels = np.fromfile(args.mnist / "train-labels-idx1-ubyte", dtype=np.uint8, offset=8)
    selected = np.concatenate([np.flatnonzero(labels == label)[:10] for label in range(10)])
    np.savez_compressed(args.output / "mnist_glyphs.npz", images=images[selected], labels=labels[selected],
                        source_indices=selected, source_split=np.array("train"))
    print(f"Extracted {len(records)} QM9 molecules and {len(selected)} original MNIST glyphs")


if __name__ == "__main__":
    main()
