# Federated Learning Crop Recognition (Optimized)

Based on [PFLlib](https://github.com/TsingZ0/PFLlib), optimized for crop image classification with FedProxV2.

## What's New (V2)

3 new files, **zero modifications** to original code:

| File | Description |
|------|-------------|
| `system/main1_v2.py` | New entry point with pretrained weight support |
| `system/flcore/clients/clientprox_v2.py` | Optimized FedProx client |
| `system/flcore/servers/serverprox_v2.py` | Optimized FedProx server |

### Optimizations

1. **Label Smoothing** (`label_smoothing=0.1`) - Prevents overconfident predictions
2. **Cosine Annealing LR** + **Warmup** - Smoother convergence than ExponentialLR
3. **Dynamic mu decay** - High mu early (suppress drift), low mu late (release personalization)
4. **Pretrained weight loading** - Auto-detect `resnet18_imagenet.pth` in `system/` directory

## Quick Start

### Original version
```bash
python main1.py -algo FedProx -model ResNet18 -data Cifar10 -ncl 10 -nc 20 -gr 100 -ls 5 -lbs 64 -lr 0.01 -mu 0.05 -eg 5 -dev cuda
```

### Optimized version (just change 2 words)
```bash
python main1_v2.py -algo FedProxV2 -model ResNet18 -data Cifar10 -ncl 10 -nc 20 -gr 100 -ls 5 -lbs 64 -lr 0.01 -mu 0.05 -eg 5 -dev cuda
```

### Optional: Pretrained Weights
1. Download: https://download.pytorch.org/models/resnet18-f37072fd.pth
2. Place as `system/resnet18_imagenet.pth`
3. Run as usual - auto-detected

## Dataset Generation

```bash
cd dataset
python generate_Cifar10.py noniid balance Dirichlet
python generate_Crop.py
```

## Project Structure

```
system/
  main1.py              # Original entry
  main1_v2.py           # Optimized entry (NEW)
  config.py             # Configuration
  flcore/
    clients/
      clientprox.py     # Original FedProx client
      clientprox_v2.py  # Optimized FedProx client (NEW)
    servers/
      serverprox.py     # Original FedProx server
      serverprox_v2.py  # Optimized FedProx server (NEW)
    trainmodel/         # Model definitions
    optimizers/         # FL optimizers
  utils/                # Data & metric utilities
dataset/                # Dataset generation scripts
```
