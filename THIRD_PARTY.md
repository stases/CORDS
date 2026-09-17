# Research and implementation attribution

CORDS is described by Tin Hadži Veljković, Erik Bekkers, Michael Tiemann and
Jan-Willem van de Meent in *Continuous Representations of Discrete Structures*,
ICLR 2026. See [the paper](https://arxiv.org/abs/2601.21583).

The molecular model's tree attention, pooling and unpooling derive from
**Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems**,
by Maksim Zhdanov, Max Welling and Jan-Willem van de Meent (ICML 2025):
[repository](https://github.com/maxxxzdn/erwin),
[paper](https://arxiv.org/abs/2502.17019).
CORDS public names do not change that architectural attribution.

Fieldformer fusion, the custom fixed-sigma Gaussian mixture, CORDS transforms,
and the compact ConvNeXt/FPN field detector were extracted and revised from the
CORDS experiment code developed in the Erwin workspace. The detector uses
ConvNeXt-style blocks; its source was the experiment implementation, not a
pretrained torchvision model.

The denoising preconditioning and Karras/Heun schedule follow the EDM family
(*Elucidating the Design Space of Diffusion-Based Generative Models*, Karras
et al., 2022), with the coordinate/density/feature conventions of the CORDS
experiments. This repository uses its own compact PyTorch training loop.

Dataset references and exact bundled example identities are in
[the data README](cords/data/README.md). Source hashes, historical run
evidence, and intentional implementation changes are in
[provenance](docs/provenance.md).
