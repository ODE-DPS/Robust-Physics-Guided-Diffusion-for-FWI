# Robust Physics-Guided Diffusion for Full-Waveform Inversion

This repository contains scripts for Robust Physics-Guided Diffusion for Full-Waveform Inversion experiments.

The workflow combines:
- A pretrained diffusion model (UNet + DDPM scheduler from the local folder)
- Forward wave simulation using Deepwave
- Data-consistency guidance losses (MSE, cumulative-sum error, or W2-style loss)
- Optional regularization terms (range constraint, TV)

## Requirements

- Python 3.11+
- CUDA-enabled GPU recommended
- Core libraries (from `pyproject.toml`):
  - torch
  - diffusers
  - deepwave
  - matplotlib
  - tqdm
  - accelerate

## Setup

Install dependencies with your preferred tool.

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Using uv (if available):

```bash
uv sync
```

## Quick Start

1. Edit experiment parameters in `configs/***.yaml` (such as `configs/FlatFault-B.yaml`).
2. Obtain the model from the link [model](https://drive.google.com/drive/folders/145M3lmS7PA1cqz9UDam1NSC3BfM7hjPC?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto), and then extract it to the Model folder
3. Run the main script:

```bash
python sample.py --config ./configs/sample-config-FlatFault-B.yaml
```

## Main Script Behavior

`sample.py` will:
- Load model components from `Model/unet` and `Model/scheduler`
- Load a test velocity sample from `test_datasets/` based on `ex_num`
- Run DDPM sampling with optional guidance and regularization
- Save outputs into a timestamped folder under `experiments/`

Typical outputs include:
- Reconstructed model images (`real.png`, `pred.png`, `diff.png`)
- Error and loss curves
- Waveform plots
- Final predicted tensor (`data/pred.pt`)
- A copy of the used config file

## Configuration (sample_config.yaml)

Key fields include:
- `loss_type`: `mse`, or `w2`
- `rho`: guidance step size
- `shot_num`: number of shots used in simulation
- `k`, `sigma`, `seed`: weighting/noise/randomness controls
- `normalize`, `adap_along`: guidance behavior switches
- `ex_num`: index of test sample in `test_datasets/`

## Repository Layout

- `sample.py`: main single-run experiment script
- `configs/`: YAML configuration files
- `Model/`: local pretrained diffusion model artifacts
- `test_datasets/`: velocity model test samples (`.npy`)
- `experiments/`: generated experiment outputs

