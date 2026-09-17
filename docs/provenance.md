# Implementation provenance

This is a corrected, minimal teaching implementation of CORDS. It preserves the
set/field construction and selected experimental model families. The included
presets are small learning demonstrations, not paper reproduction recipes.

## Evidence inspected

The local paper was `CORDS___ICLR_2026.pdf` (38 pages; SHA256
`34c878c91c88909d80c2ce0973351185766473495df8f08cd1087ecbe330ca8c`).
Erwin's checkout HEAD was `93cf23f685104dce091adc677995c8579b2822a9`.
Most CORDS experiment files were untracked, so that commit alone does not
identify their content. Selection used source inspection, job launch commands,
W&B configurations, retained code artifacts, and numerical roundtrip probes.

| Component | Source within the Erwin workspace | Release implementation |
| --- | --- | --- |
| Gaussian fields and inverse | `utils/rebuilt_transform_old.py`, `utils/rebuilt_transform.py` | `cords.reconstruction`, `cords.fields` |
| Fixed-width Gaussian mixtures | `utils/custom_gmm.py` | `cords.gmm.FixedSigmaGMM` |
| Molecular denoising | `GraphField_QM9.py`, `models/qm9.py`, standard Erwin blocks | `cords.models.CORDSMoleculeDenoiser` |
| MNIST field model | `FullMultiMNIST.py`, later shared fields branch in `YOLOMNIST.py` | `cords.detection.CORDSDetector` |

Source SHA256 values:

```text
utils/custom_gmm.py             5988eff5e7ecdebcd446fe725b6b7559916bd3d1beb8ac9a86ef49db537e3769
utils/rebuilt_transform_old.py  89b5f06f2bb0a2853ecfdf19a6cd6ba8610010e6fd5458041f77ae269bba8e0b
utils/rebuilt_transform.py      f74b23aefda4fa85a8f9e9fdf68553c83bc3b401a7c9ee39de00705f32022aa7
GraphField_QM9.py               b1f5b96ca4b5793dd1fa44d54ff4fabb5963e1e1215799b837aad54405fb8b24
FullMultiMNIST.py               23caa431d526174925da1e22d52e2cd910a6a4d9e19339478c221abd384889b3
YOLOMNIST.py                    9621f2ac3af19e931a29fda16a0800a2413dec962f687af3adb6b3021c97d776
```

## Molecular lineage

The GEOM-Drugs investigation served as a reconstruction reference; GEOM support
is not part of this release. The latest substantial inspected run,
[`tin/GeomDrugs‑EDM/u5u6uv4f`](https://wandb.ai/tin/GeomDrugs%E2%80%91EDM/runs/u5u6uv4f),
was created on 2025-08-04 on Snellius. Its `fieldformer_EDM.py` main script is
retained as artifact `source_df61d30e:v5`, file
`script_20250804_142203_df61d30e.py`. This artifact does not snapshot the
transform/GMM dependencies. The run was marked crashed but reached epoch 480
and 904,249 steps; filtering to finished runs would have lost this evidence.
No model checkpoint was logged there.

The local latest transform had accumulated detection-specific density filtering,
a changed decode return signature, and an arbitrary atom cap. The extracted
molecular path instead retains fixed-width mixture fitting, density refinement,
and joint feature projection. A real `gdb_14` probe of the historical assembly
recovered nine atoms/types with positional RMS error approximately 6e-6 Å.

Intentional simplifications/corrections in the release:

- The public transform defaults to physical density with explicit proposal and
  integration weights. The optional molecular model retains density-only
  count normalization: feature fields stay physical.
- Fixed-sigma EM uses deterministic importance-corrected weights rather than
  stochastic density-sharpened resampling. Five greedy k-means++ restarts avoid
  a dependency on sklearn; there is one CPU NumPy implementation.
- Count comes from field mass. No truth count, truth centers, arbitrary 35/100
  atom clamp, or BIC count search is used in the teaching path.
- Refinement locally enables gradients inside `no_grad`. Known physical density
  uses unit-amplitude kernels; generated count-normalized density fits a common
  unknown amplitude. Features are solved against the physical Gaussian basis.
- The Erwin-derived model keeps RFF/FiLM field fusion, ball attention, hierarchical
  pooling and unpooling. A pure PyTorch median tree replaces the compiled builder;
  tree ties/padding and compact widths can differ from historical runs. Existing
  Erwin checkpoints are not promised to load into this model.
- Training uses explicit JSON options, ordinary PyTorch loops and compact
  checkpointing. It does not port Lightning, flash attention, SH experiments,
  PyG message passing, SWA sweeps, or chemistry benchmark evaluators.

## MNIST lineage

| Date | Run / program | Epoch | Final val mAP50:95 | Count MAE |
| --- | --- | ---: | ---: | ---: |
| 2025-09-21 | [8z4jds68](https://wandb.ai/tin/MultiMNIST-simple/runs/8z4jds68), `FullMultiMNIST.py` | 353 | 0.57820 | 0.36612 |
| 2025-09-23 | [poa9zj10](https://wandb.ai/tin/MultiMNIST-simple/runs/poa9zj10), `YOLOMNIST.py` | 193 | 0.57010 | 0.82507 |
| 2025-09-23 | [8mj8n7xy](https://wandb.ai/tin/MultiMNIST-simple/runs/8mj8n7xy), `YOLOMNIST.py` | 25 | 0.53937 | 52.97422 |

These are final recorded summaries, not a search for the best historical epoch.
The configurations differ, so this is provenance rather than a comparison table.
No source or model checkpoint artifacts were attached to these inspected runs.

The latest `FIELDSMultiMNIST.job` launched `YOLOMNIST.py --model
fields_convnext_fpn`, despite the filename. It selected `decode_upsample=2`.
Read-only probes showed the old decoder multiplied density mass by four when
upsampling; this explains why latest-file selection is inappropriate here.
The usable runs instead used native-resolution counting. The selected model
family is shared between the September sources.

The teaching detector retains the ConvNeXt/FPN field head, now exposed under
CORDS names. Changes include:

- Finite-grid unit-mass kernels, including near image boundaries.
- One density-fit/Gram-projection decoder for analytic and predicted fields,
  replacing the experiment's approximate peak/size-ratio/NMS decoder. This is
  a deliberate pedagogical difference, not a benchmark-compatible decoder.
- Native-resolution mass integration and explicit excessive-count errors.
- Seed/epoch/index-based dataset randomness, fixing duplicate worker streams.
- Grayscale input, compact default width, and a calibrated initial density bias.
- Dense, scale-balanced field regression with an active normalized count loss.
  The old launcher's count coefficient was inactive when weak supervision was off.
- Removal of baseline detectors, dead refinement flags, and misleading per-class
  implementations of a purported per-image output cap.

The notebooks distinguish invertibility of the representation from numerical
decoding and from learning. No pretrained result or paper metric is claimed by
the optional overfit demonstrations.
