# Validation

Validated on 2026-09-17 using Python 3.11, PyTorch 2.5.0, NumPy 2.1.3,
SciPy 1.15.2 and CPU execution with two Torch threads. Training tests also
passed on the existing PyTorch 2.3 CPU environment. No dataset or model weights
were downloaded, and no full experiment or GPU benchmark was run.

## Automated checks

`python -m pytest -q`: **29 tests passed** (approximately 5 seconds).

- Gaussian encoding and proposal-weighted integration; density-only model
  normalization and unknown-amplitude predicted fields.
- Empty sets, malformed fields, explicit excessive-count guards, and
  reconstruction inside `torch.no_grad()`.
- Fixed-sigma GMM fitting, feature projection, permutation-invariant fields,
  finite image boundaries and native-resolution mass.
- Deterministic scene generation, distinct epochs, and serial versus two-worker
  data loading.
- Real bundled QM9 roundtrips and separated MNIST scenes with 3, 6 and 16 digits.
- Model gradients, permutation-equivariant molecular predictions, seeded Heun
  sampling, decreasing fixed-evaluation loss, exact checkpoint reload and exact
  resumed-versus-uninterrupted CPU training.

The three QM9 fixtures were additionally checked at seeds 0, 7 and 42 with
1,024 samples and sigma 0.2 Å. All recovered counts and atom types exactly;
positional RMS errors were below 2e-7 Å. This verifies those fixtures, not
arbitrary molecules or every bandwidth/sample-count setting.

## Notebook execution

Both committed notebooks were executed from clean kernels starting in
`notebooks/` using `scripts/execute_notebooks.py --write`, with no errors and
training disabled:

| Notebook | CPU wall time |
| --- | ---: |
| Molecular roundtrips | 6.8 s |
| MNIST detection | 8.9 s |

Static plots were visually inspected. Notebook controls and Plotly outputs are
included alongside static equivalents. Kernel execution verifies the notebook
code; browser-specific interactive rendering is left to the Jupyter frontend.

Temporary copies were also run with `RUN_TRAINING=True`: the full molecular
notebook completed in 9.3 s (80 optimizer steps), and the full MNIST notebook in
17.4 s (120 optimizer steps). Both sections produced learning curves and
before/after field predictions and reloaded their checkpoints. These are
intentional fixed-batch overfits, not generative or generalization results.
The fixed evaluation losses decreased from 12.15 to 5.61 for QM9 and from
2.97 to 0.30 for MNIST in those notebook executions.

## Running from the checkout

The repository runs directly from its local `cords/` directory, with dependencies
listed in `requirements.txt`. Direct imports and bundled fixture loading were
checked from the repository root without setting `PYTHONPATH` or installing
CORDS. All 29 tests also passed with this layout. Notebook setup resolves the
repository root when Jupyter starts in `notebooks/`, and training checkpoints
are written under the repository's `runs/` directory.

The molecular architecture and tree operations use PyTorch; running them requires
no separate Erwin checkout or compiled extension. The citation file validates
against the CFF 1.2.0 schema.

## Numerical limitations shown explicitly

The overlapping MNIST example (seed 7, sixteen digits, sigma 7 pixels) keeps
count correct but obtains approximately 0.858 matched IoU and 87.5% class
accuracy, despite density relative error around 0.00035. This demonstrates the
conditioning and local-optimization limits of practical decoding.

Unconditional sampling from a tiny overfit model can produce implausible counts
or overflow. These failures are reported; generated molecules are not claimed
to be chemically valid. Historical W&B metrics are provenance only.
