# Federated Learning Crop Recognition (Optimized)

Based on [PFLlib](https://github.com/TsingZ0/PFLlib), optimized for crop image classification with FedProxV2 / CropV2.

## What's New

### V2 (FedProxV2) - 3 new files, zero modifications to original code

| File | Description |
|------|-------------|
| `system/main_v2.py` | Entry point with pretrained weight support |
| `system/flcore/clients/clientprox_v2.py` | Optimized FedProx client |
| `system/flcore/servers/serverprox_v2.py` | Optimized FedProx server |

### V3 (CropV2) - Full optimization suite for higher accuracy

| File | Description |
|------|-------------|
| `system/main_crop.py` | CropV2 entry point with enhanced data augmentation |
| `system/flcore/clients/client_crop_v2.py` | CropV2 client (full optimization) |
| `system/flcore/servers/server_crop_v2.py` | CropV2 server (dynamic mu + warmup) |

### Optimizations Summary

| Optimization | FedProxV2 | CropV2 | Effect |
|---|:---:|:---:|---|
| Label Smoothing (0.1) | ✓ | ✓ | Prevents overconfident predictions |
| Cosine Annealing LR | ✓ | ✓ | Smoother convergence than MultiStepLR |
| Warmup LR | ✓ | ✓ | Avoids large gradients early in training |
| Momentum + Weight Decay | ✓ | ✓ | Faster convergence + regularization |
| Gradient Clipping (5.0) | ✓ | ✓ | Prevents gradient explosion |
| Dynamic mu Decay | | ✓ | High mu early → suppress drift; low mu late → release personalization |
| ColorJitter Augmentation | | ✓ | Better generalization on color images |
| RandomErasing | | ✓ | Robustness to occlusion |
| Best Accuracy Tracking | ✓ | ✓ | Monitors peak performance during training |
| Pretrained Weight Loading | ✓ | ✓ | Transfer learning from ImageNet |

## Quick Start

### Original FedProx
```bash
python main.py -algo FedProx -model ResNet18 -data Cifar10 -ncl 10 -nc 20 -gr 100 -ls 5 -lbs 64 -lr 0.01 -mu 0.05 -eg 5 -dev cuda
```

### FedProxV2 (label smoothing + CosineAnnealing + gradient clipping)
```bash
python main_v2.py -algo FedProxV2 -model ResNet18 -data Cifar10 -ncl 10 -nc 20 -gr 100 -ls 5 -lbs 64 -lr 0.01 -mu 0.05 -eg 5 -dev cuda
```

### CropV2 (full optimization + dynamic mu + ColorJitter + RandomErasing)
```bash
python main_crop.py -algo CropV2 -model ResNet18 -data Cifar10 -ncl 10 -nc 20 -gr 100 -ls 5 -lbs 64 -lr 0.01 -mu 0.05 -eg 5 -dev cuda
```

### Optional: Pretrained Weights
1. Download: https://download.pytorch.org/models/resnet18-f37072fd.pth
2. Place as `system/resnet18_imagenet.pth`
3. Run with `-pp resnet18_imagenet.pth`

## Dataset Generation

```bash
cd dataset
python generate_Cifar10.py noniid balance Dirichlet
python generate_Crop.py
```

## Project Structure

```
system/
  main.py                # Original entry
  main_v2.py             # FedProxV2 entry
  main_crop.py           # CropV2 entry (NEW)
  config.py              # Configuration
  flcore/
    clients/
      clientprox.py      # Original FedProx client
      clientprox_v2.py   # FedProxV2 client
      client_crop_v2.py  # CropV2 client (NEW)
    servers/
      serverprox.py      # Original FedProx server
      serverprox_v2.py   # FedProxV2 server
      server_crop_v2.py  # CropV2 server (NEW)
    trainmodel/          # Model definitions
    optimizers/          # FL optimizers
  utils/                 # Data & metric utilities
  fl_app.py              # PyQt5 visualization app
dataset/                 # Dataset generation scripts
```
