# DHGNN Minimal Public Release

This repository contains a minimal, runnable implementation of DHGNN for remote
sensing scene classification from precomputed patch-grid features.

The public release intentionally excludes datasets, pretrained backbone weights,
feature caches, trained weight files, logs, and multi-seed experiment outputs. Any image
encoder can be used outside this repository as long as it exports features in
the format described below.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Smoke Test

Generate a tiny synthetic feature dataset:

```bash
python scripts/make_toy_features.py --output data/toy_features.npz
```

Train DHGNN on CPU:

```bash
python scripts/train_features.py \
  --data data/toy_features.npz \
  --output-dir runs/toy \
  --epochs 2 \
  --batch-size 8 \
  --hidden-dim 32 \
  --device cpu
```

The training script writes:

- `config.json`
- `dataset_info.json`
- `history.json`
- `history.csv`
- `test_metrics.json`
- `test_predictions.json`

## Public API

```python
import torch
from dhgnn import DHGNNClassifier

model = DHGNNClassifier(
    num_classes=3,
    feature_dim=32,
    hidden_dim=64,
    num_levels=3,
    graph_depth=1,
    adjacency_mode="dynamic",
    dynamic_topk=4,
    use_edge_state=True,
)

features = torch.randn(2, 8, 8, 32)
logits = model(features)
```

`forward(features)` expects precomputed patch-grid features with shape
`[batch_size, height, width, channels]`.

## Feature File Format

`scripts/train_features.py` reads a compressed NumPy `.npz` file with:

- `features`: `float32` array with shape `[N, H, W, C]`
- `labels`: `int64` array with shape `[N]`

Optional arrays:

- `class_names`: string array with class names
- `paths`: string array with source image or feature identifiers

For `num_levels=3`, `H` and `W` must be divisible by 4 because DHGNN performs
two 2x hierarchical aggregations. The toy data uses an `8x8` feature grid.

## Using Real AID or NWPU-RESISC45 Features

1. Download and prepare the dataset according to its official terms.
2. Run any image encoder, such as a frozen ViT/DINO-style backbone, outside this
   repository.
3. Export patch features as `[N, H, W, C]` features and labels as integer class
   indices.
4. Save them to `.npz` with the format above.
5. Train with `scripts/train_features.py`.

This minimal release does not include online DINO or DINOv3 feature extraction
code. Keeping feature extraction separate avoids publishing local paths,
large model artifacts, and dataset-specific experiment state.

## Method Defaults

The runnable defaults match the compact DHGNN configuration:

- hierarchical levels: `3`
- spatial relation mode: `dynamic`
- dynamic Top-K: `4`
- edge state: enabled
- readout: mean pooling over three hierarchy levels

For smaller feature grids, set `--num-levels 1` or `--num-levels 2`.
