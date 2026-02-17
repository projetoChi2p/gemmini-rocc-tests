# ViT Model Runner - Build Configuration Guide

## Overview

This project now supports dynamic model and dataset selection at build time through Makefile variables. You no longer need to manually edit `input_weights.h` to switch between models and datasets.

## Quick Start

### Basic Usage

```bash
# Build and run with default configuration (minivit + sat6)
make -f RunViT.mk run

# Build and run with specific model and dataset
make -f RunViT.mk run MODEL=minivit DATASET=sat6
make -f RunViT.mk run MODEL=deitvit DATASET=mnist
make -f RunViT.mk run MODEL=deitcnn DATASET=cifar10
```

### Available Options

**Models:**
- `minivit` - Mini Vision Transformer (default)
- `deitvit` - DeiT Vision Transformer
- `deitcnn` - DeiT with CNN Embedding

**Datasets:**
- `sat6` - SAT-6 Satellite Dataset (default)
- `emnist` - Extended MNIST
- `mnist` - MNIST Digits
- `cifar10` - CIFAR-10
- `cifar100` - CIFAR-100

### Output

Results are automatically saved to `outputs/MODEL_DATASET_report.txt`

Example: `outputs/minivit_sat6_report.txt`

## How It Works

### 1. Makefile Configuration (`RunViT.mk`)
- Sets `MODEL` and `DATASET` variables
- Calls `build.sh` with these environment variables
- Redirects output to the appropriate report file

### 2. Build Script (`build.sh`)
- Accepts `MODEL` and `DATASET` from environment
- Passes them as compiler flags (`-DMODEL=... -DDATASET=...`)
- Forces rebuild when configuration changes

### 3. Header Configuration (`include/input_weights.h`)
- Uses preprocessor macros to dynamically select the correct parameter file
- Constructs include path: `includes/{MODEL}_{DATASET}_[quant_]params.h`
- Automatically handles quantized vs non-quantized builds

### 4. Main Program (`vit_merged.c`)
- Prints build configuration at runtime
- Helps verify correct model/dataset selection

## Commands

### Build Only
```bash
make -f RunViT.mk build MODEL=minivit DATASET=sat6
```

### Clean Build Artifacts
```bash
# Clean ViT binaries only
make -f RunViT.mk clean

# Clean everything including outputs
make -f RunViT.mk clean-all
```

### Help
```bash
make -f RunViT.mk help
```

## Examples

### Run multiple configurations
```bash
# MiniViT on different datasets
make -f RunViT.mk run MODEL=minivit DATASET=sat6
make -f RunViT.mk run MODEL=minivit DATASET=mnist
make -f RunViT.mk run MODEL=minivit DATASET=cifar10

# Different models on MNIST
make -f RunViT.mk run MODEL=minivit DATASET=mnist
make -f RunViT.mk run MODEL=deitvit DATASET=mnist
make -f RunViT.mk run MODEL=deitcnn DATASET=mnist
```

### Batch testing
```bash
#!/bin/bash
# test_all.sh - Run all model/dataset combinations

MODELS="minivit deitvit deitcnn"
DATASETS="mnist cifar10"

for model in $MODELS; do
    for dataset in $DATASETS; do
        echo "Testing $model on $dataset..."
        make -f RunViT.mk run MODEL=$model DATASET=$dataset
    done
done

echo "All results saved to outputs/"
```

## Directory Structure

```
gemmini-rocc-tests/
├── RunViT.mk                 # Main Makefile wrapper (NEW)
├── build.sh                  # Updated to accept MODEL/DATASET
├── outputs/                  # Generated reports (NEW)
│   ├── minivit_sat6_report.txt
│   ├── deitvit_mnist_report.txt
│   └── ...
└── ViTRuns/
    ├── vit_merged.c          # Updated to print config
    ├── Makefile              # Updated to pass EXTRA_CFLAGS
    └── include/
        ├── input_weights.h   # Dynamic include selection (NEW)
        └── includes/
            ├── minivit_sat6_quant_params.h
            ├── deitvit_mnist_quant_params.h
            └── ...
```

## Troubleshooting

### Build not picking up new MODEL/DATASET
The build system automatically cleans previous binaries when switching configurations. If you still see old behavior:
```bash
make -f RunViT.mk clean
make -f RunViT.mk build MODEL=<model> DATASET=<dataset>
```

### Parameter file not found
Ensure the parameter file exists with the correct naming convention:
- Quantized: `includes/{model}_{dataset}_quant_params.h`
- Non-quantized: `includes/{model}_{dataset}_params.h`

### Wrong output displayed
Check the runtime output for "Build Config:" line to verify MODEL and DATASET values.

## Advanced Usage

### Custom configuration
You can still override individual parameters by modifying the parameter header files in `ViTRuns/include/includes/`.

### Integration with CI/CD
```yaml
# Example GitHub Actions workflow
- name: Test ViT Models
  run: |
    for model in minivit deitvit; do
      for dataset in mnist cifar10; do
        make -f RunViT.mk run MODEL=$model DATASET=$dataset
      done
    done
```

## Migration from Old System

**Before (manual editing):**
```c
// Had to manually uncomment the desired include
#include "includes/minivit_sat6_quant_params.h"  // ✓ Active
// #include "includes/deitvit_mnist_quant_params.h"  // ✗ Commented
```

**Now (automatic):**
```bash
# Just specify at build time
make -f RunViT.mk run MODEL=minivit DATASET=sat6
make -f RunViT.mk run MODEL=deitvit DATASET=mnist
```

## Notes

- The system defaults to `minivit` and `sat6` if MODEL and DATASET are not specified
- Each build configuration is independent - outputs are saved separately
- The build system automatically detects BAREMETAL vs Linux builds
- Quantization is determined by `ELEM_T_IS_FLOAT` (unchanged from original system)
