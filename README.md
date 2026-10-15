# pegs-sam3d-initialisation

Code and setup for the Master's thesis **Initialising Physically Embodied Gaussian Splatting using 3D Reconstruction**, Queensland University of Technology, 2026.

The project investigates whether single-image 3D reconstruction with SAM 3D Objects can improve object initialisation in Physically Embodied Gaussian Splatting (PEGS).

## Status

Preliminary. A feasibility result has been obtained: SAM 3D Objects produces a plausible 3D reconstruction from a single PEGS dataset image and mask. No geometric error has yet been measured, and the reconstruction has not yet been registered into PEGS's metric frame or converted into PEGS particles.

## Hardware

| Item | Value |
|---|---|
| Provider | RunPod, Secure Cloud, CA-MTL-1 |
| GPU | 1x NVIDIA A40, 48 GB VRAM (sm_86) |
| CPU / RAM | 9 vCPU / 50 GB |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Persistent storage | 100 GB network volume mounted at `/workspace` |

## Upstream components

Neither is redistributed here; both are installed from source.

- PEGS: https://github.com/rai-opensource/embodied_gaussians
- SAM 3D Objects: https://github.com/facebookresearch/sam-3d-objects (gated weights)
- PEGS dataset: https://huggingface.co/datasets/anon211/embodied_gaussians

## Layout

```
setup/      environment bootstrap scripts
scripts/    data preparation and inference
docs/       environment specification and experiment log
results/    figures and measured metrics
```

## Not included

Model checkpoints, the PEGS dataset, and large `.ply` outputs are excluded by `.gitignore`. Checkpoints are gated and may not be redistributed; the dataset is published separately at the link above.

## Author

Omar Kassab, Master of Robotics and Artificial Intelligence, Queensland University of Technology.
