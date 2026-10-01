# PolyMicros: Bootstrapping a Foundation Model for Polycrystalline Material Structure

**Michael Buzzy, Andreas Robertson, Peng Chen, Surya R. Kalidindi**

[![Paper](https://img.shields.io/badge/Paper-Acta%20Materialia-blue)](https://www.sciencedirect.com/science/article/abs/pii/S135964542600697X)
[![arXiv](https://img.shields.io/badge/arXiv-2506.11055-b31b1b)](https://arxiv.org/abs/2506.11055)
[![Data & Weights](https://img.shields.io/badge/Data%20%26%20Weights-Google%20Drive-green)](https://drive.google.com/drive/folders/1bbl5pD1XcSC_U9KAhuZufGlXAWgq6ipQ?usp=sharing)

This repository contains the official implementation of **PolyMicros**, the first foundation model for polycrystalline material microstructure.

- **Full paper:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S135964542600697X)
- **Preprint:** [arXiv:2506.11055](https://arxiv.org/abs/2506.11055). The preprint omits some results that appear in the published version.
- **Dataset and pretrained model weights:** [Google Drive](https://drive.google.com/drive/folders/1bbl5pD1XcSC_U9KAhuZufGlXAWgq6ipQ?usp=sharing)

## Abstract

Foundation models for materials science could change how materials with tailored properties are discovered, manufactured, and designed. So far, though, they have succeeded mainly for material classes where repositories of millions of samples can be curated, such as atomistic structures. For many structural and functional materials, such as mesoscale-structured metal alloys, building datasets of that size is too costly, and only a handful of examples are available. To address this, we introduce a machine learning approach for learning from hyper-sparse, complex spatial data in scientific domains. Our main contribution is a physics-driven data augmentation scheme. It uses an ensemble of local generative models, trained on as few as five experimental observations, and coordinates them with a new diversity curation strategy to generate a large, physically diverse dataset. We use this framework to build PolyMicros, the first foundation model for polycrystalline materials, a structural material class that matters across many industrial and scientific applications. We show what PolyMicros can do by solving several long-standing problems in accelerating 3D experimental microscopy zero-shot. Our models and datasets are openly available to the community.

## Method Overview

The PolyMicros pipeline has three stages:

1. **Multi-output Gaussian random field (MOGRF) sampling.** Statistically diverse initial microstructures are sampled from MOGRFs. Their auto- and cross-correlations come from a multi-output spectral mixture (MOSM) kernel.
2. **Local refinement.** An ensemble of local generative (diffusion) models, each trained on a few experimental observations, refines the MOGRF samples into physically realistic polycrystalline microstructures. A diversity curation strategy coordinates the ensemble.
3. **Foundation model training.** A 3D score-based diffusion model (EDM formulation) is trained on the curated synthetic dataset. Conditioning at inference time then lets it solve downstream tasks zero-shot.

Crystallographic orientations are represented with real-valued generalized spherical harmonics (ROGSH) for cubic (432) crystal symmetry.

## Repository Structure

### Diffusion models

| File | Description |
|---|---|
| `diffusion/` | PyTorch Lightning module for EDM-style diffusion models |
| `DMRunner.py` | Script for training diffusion models and running inference with them |
| `Unet.py` | DDPM-style U-Net adapted to 3D volumetric data |
| `UnetP.py` | 3D U-Net with periodic boundary conditions |
| `config.json` | Example model and training configuration |

### Multi-output Gaussian random fields

| File | Description |
|---|---|
| `MOSMKernel.py` | Generates auto- and cross-correlations with the MOSM kernel |
| `GRFSampler.py` | Samples from a MOGRF given a covariance |

### Workflows

| File | Description |
|---|---|
| `PolyMicros.py` | Dataset generation pipeline: MOGRF sampling and local refinement |
| `CaseStudies.py` | Conditional generation case studies: super-resolution and dimensionality expansion (2D to 3D) |
| `superresolution.py` | Super-resolution with the PolyMicros foundation model |
| `periodic_padding.py`, `code_periodicpadding.py` | Converting experimental microstructures into periodic microstructures that still look experimental, including post-processing and segmentation |

### Utilities

| File | Description |
|---|---|
| `ROGSHUtils.py` | Conversion between ROGSH coefficients and Euler angles |
| `GSHTrees/` | Precomputed lookup tables (KD-trees) for the ROGSH to Euler conversion |
| `HDFPartialLoader.py` | Data loader that streams batches from HDF5 files, for datasets larger than available memory |
| `CUBEPlot.py` | Matplotlib plotting for 3D volumes and orthogonal two-point statistics |
| `MKFigs.py` | Scripts that reproduce the figures in the paper |

## Requirements

The code is written in Python and mainly depends on:

- PyTorch, PyTorch Lightning, torchmetrics, torchvision
- NumPy, SciPy, scikit-learn, scikit-image
- h5py, einops, rich, tqdm
- matplotlib, seaborn
- [orix](https://orix.readthedocs.io/), for crystallographic orientation handling

## Data and Pretrained Models

The PolyMicros datasets and the trained model weights are available on [Google Drive](https://drive.google.com/drive/folders/1bbl5pD1XcSC_U9KAhuZufGlXAWgq6ipQ?usp=sharing). Datasets are stored in HDF5 format. Model weights are provided as PyTorch Lightning checkpoints (`.ckpt`), each with its JSON configuration file. Place the downloaded checkpoint and configuration in `model_checkpoints/` to use the examples below unchanged.

## Usage

Run all scripts from the repository root. Several modules load resources from relative paths, such as `./GSHTrees/`.

### Training a diffusion model

`DMRunner.py` trains and samples the 3D diffusion models. Model and training hyperparameters are read from a JSON configuration file (see `config.json`). Training data is read from an HDF5 file, from one or more groups or datasets given with `--prefix`:

```bash
python DMRunner.py train -c config.json -d path/to/dataset.h5 -p <hdf5_group>
```

Each run creates a time-stamped directory under `experiments/`. It holds a copy of the configuration and the model checkpoints (`logs/version_0/checkpoints/`). To resume from or fine-tune an existing checkpoint, pass it with `-m path/to/checkpoint.ckpt`.

If each sample is stored as a separate HDF5 dataset under the prefix group, add `--HDFPartial`. This streams one batch at a time instead of loading the whole dataset into memory.

### Unconditional sampling

```bash
python DMRunner.py sample --dir experiments/<run_id> -e 64 -s 300 --nsamples 4
```

`--dir` loads the configuration and the latest checkpoint from a training run. You can also give them explicitly with `-c` and `-m`. `-e` sets the edge length of the generated volume in voxels, and `-s` sets the number of diffusion steps. Samples are saved as a NumPy array, together with PNG renderings of each volume.

To partially denoise an existing microstructure instead of starting from noise, pass an initialization and the number of steps to skip:

```bash
python DMRunner.py sample --dir experiments/<run_id> -e 64 -s 300 \
    -i path/to/initial.h5 -p <hdf5_dataset> --skip 75
```

| Argument | Description |
|---|---|
| `mode` | `train`, `sample`, or `plot` |
| `-c, --config` | Path to the JSON configuration file |
| `-d, --dataset` | Path to the HDF5 training dataset |
| `-m, --model` | Path to a model checkpoint (`.ckpt`) |
| `--dir` | Experiment directory; overrides `--config` and `--model` |
| `-p, --prefix` | HDF5 group(s) or dataset(s) to load |
| `-e, --edge_length` | Edge length of sampled volumes (voxels) |
| `-s, --steps` | Number of diffusion sampling steps |
| `-i, --inital` | HDF5 file containing an initialization for sampling |
| `--skip` | Number of initial diffusion steps to skip when an initialization is given |
| `--nsamples` | Number of samples to generate |
| `--HDFPartial` | Stream training data from HDF5 one batch at a time |

### Loading the pretrained model in Python

Microstructures are represented as volumes with 3 channels of ROGSH coefficients in channels-first layout `(N, 3, X, Y, Z)`. The example below loads the pretrained model and draws an unconditional sample:

```python
import json, argparse
import numpy as np
from diffusion import Diffusion
from UnetP import UnetND          # periodic U-Net; use Unet.UnetND for non-periodic
from ROGSHUtils import ROGSH432

with open("model_checkpoints/config.json") as f:
    config = argparse.Namespace(**json.load(f))

unet = UnetND(config.udim, dim_mults=tuple(config.channels), channels=config.in_channels)
unet.sample_size = 64             # spatial edge length of generated volumes
unet.in_channels = config.in_channels
unet.dims = config.dims

model = Diffusion.load_from_checkpoint(
    "model_checkpoints/PolyMicros.ckpt", model=unet, bins_max=config.sample_steps
).cuda().eval()

sample = model.sample(num_samples=1, steps=300)   # (1, 3, 64, 64, 64), ROGSH

# Convert to Euler angles (radians) using the nearest point on a discrete orientation grid
GSH = ROGSH432()
euler = GSH.Rogsh2Euler(np.moveaxis(sample[0].cpu().numpy(), 0, -1))
```

To convert experimental Euler-angle data of shape `(X, Y, Z, 3)` into the model's representation, use `GSH.ROGSH(euler)`.

### Conditional generation

`model.sample` accepts a conditioning function, `cond_fn`. At each denoising step this function projects the current estimate onto the known data. For a masked (inpainting-style) constraint, `masked_cond_fn` in `CaseStudies.py` sets every voxel with a known value (non-`NaN` in the mask) to that value. All the case studies in the paper use this mechanism:

| Task | Script |
|---|---|
| Super-resolution of serial-section data | `CaseStudies.py`, `superresolution.py` |
| Dimensionality expansion (2D sections to 3D volumes) | `CaseStudies.py` |
| Periodic padding of experimental microstructures | `periodic_padding.py`, `code_periodicpadding.py` |

These scripts are configured by editing the parameters in their `main()` functions (model checkpoint, input data, number of steps, padding width) rather than through command-line arguments. Run them directly, for example `python superresolution.py`.

### Synthetic dataset generation

`PolyMicros.py` reproduces the data augmentation pipeline from the paper. It samples MOGRF microstructures with the MOSM kernel, then refines them with the ensemble of local diffusion models. The output is written to an HDF5 file. Set the paths to the local model checkpoints at the top of the script before running it.

## Known Limitations

This is research code, released to support reproducibility of the published results. Please note:

- Some scripts contain hard-coded file paths (for example in `diffusion/diffusion.py` and the dataset loading routines). You will need to change these to match your local environment.
- Different modules use different tensor layout conventions (channels-first or channels-last).
- Dataset schemas are not fully consistent across files.

## Citation

If you use this code or the dataset in your research, please cite:

```bibtex
@article{buzzy2026polymicros,
  title   = {PolyMicros: Bootstrapping a Foundation Model for Polycrystalline Material Structure},
  author  = {Buzzy, Michael and Robertson, Andreas and Chen, Peng and Kalidindi, Surya R.},
  journal = {Acta Materialia},
  year    = {2026},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S135964542600697X}
}
```

## Contact

For questions about the code or data, please contact Michael Buzzy or open an issue on this repository.
