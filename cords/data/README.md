# Bundled examples

These small fixtures make the default notebooks independent of downloads and
machine-specific dataset paths. They are examples, not benchmark splits.

## QM9

`qm9_examples.json` retains original QM9 geometries, explicit hydrogens, original
GDB names, zero-based source identifiers, and bond annotations. Positions are in
angstroms. The five element channels are rebuilt from atomic numbers (H, C, N,
O, F); the original PyG `x` array contains eleven different features and is not
treated as five-channel atom-type data.

| Name | Original source index | Atoms | Elements |
| --- | ---: | ---: | --- |
| gdb_14 | 13 | 9 | H, C, O |
| gdb_19 | 18 | 9 | H, C, N, O |
| gdb_826 | 825 | 7 | H, C, F |

Source: the existing PyG-compatible raw `qm9_v3.pt` cache. Filtered dataset row
numbers differ from original source indices, so extraction selects original
names. No conformers were regenerated. Bond types follow the source one-hot
ordering: single, double, triple, aromatic; the roundtrip recovers atoms/types,
not bonds.

Dataset reference: Ramakrishnan et al., *Quantum chemistry structures and
properties of 134 kilo molecules*, Scientific Data 1, 140022 (2014),
[doi:10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22).

## MNIST

`mnist_glyphs.npz` contains the first ten glyphs of each digit class from the
original MNIST training split (100 glyphs total). It stores uint8 `[N,28,28]`
images, labels, original zero-based indices and `source_split="train"`.
The composition code generates images and their source annotations from these
glyphs. This tiny set is for illustration and deliberate overfitting; it cannot
support an independent benchmark evaluation.

Dataset reference: LeCun et al., *Gradient-Based Learning Applied to Document
Recognition*, Proceedings of the IEEE 86(11), 2278–2324 (1998),
[doi:10.1109/5.726791](https://doi.org/10.1109/5.726791).

To reproduce fixture extraction from existing caches:

```bash
python scripts/prepare_fixtures.py --qm9 /path/to/qm9_v3.pt \
  --mnist /path/to/MNIST/raw --output cords/data
```

This command never downloads data. Only load trusted local `.pt` files.
