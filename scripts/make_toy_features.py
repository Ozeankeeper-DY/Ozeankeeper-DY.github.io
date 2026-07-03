import argparse
from pathlib import Path

import numpy as np


def make_toy_features(
    num_samples: int,
    num_classes: int,
    grid_size: int,
    feature_dim: int,
    seed: int,
):
    if num_samples < num_classes * 2:
        raise ValueError("num_samples must provide at least two samples per class.")
    if grid_size < 4 or grid_size % 4 != 0:
        raise ValueError("grid_size must be at least 4 and divisible by 4.")
    if feature_dim < num_classes:
        raise ValueError("feature_dim must be >= num_classes for the toy generator.")

    rng = np.random.default_rng(seed)
    labels = np.arange(num_samples, dtype=np.int64) % num_classes
    rng.shuffle(labels)

    y = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    spatial_basis = np.stack(
        [
            np.ones_like(xx),
            xx,
            yy,
            xx * yy,
            xx**2 - yy**2,
        ],
        axis=-1,
    )

    class_proto = rng.normal(loc=0.0, scale=0.5, size=(num_classes, feature_dim)).astype(np.float32)
    for class_index in range(num_classes):
        class_proto[class_index, class_index] += 2.0

    features = np.empty((num_samples, grid_size, grid_size, feature_dim), dtype=np.float32)
    for sample_index, label in enumerate(labels):
        noise = rng.normal(
            loc=0.0,
            scale=0.15,
            size=(grid_size, grid_size, feature_dim),
        ).astype(np.float32)
        feature = class_proto[label].reshape(1, 1, feature_dim) + noise
        for basis_index in range(min(spatial_basis.shape[-1], feature_dim)):
            feature[..., basis_index] += spatial_basis[..., basis_index] * (0.2 + 0.1 * label)
        features[sample_index] = feature

    class_names = np.array([f"class_{index}" for index in range(num_classes)])
    paths = np.array([f"toy/class_{label}/sample_{index:04d}.npy" for index, label in enumerate(labels)])
    return features, labels, class_names, paths


def main():
    parser = argparse.ArgumentParser(description="Create a tiny synthetic feature dataset for DHGNN smoke tests.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=72)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features, labels, class_names, paths = make_toy_features(
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        grid_size=args.grid_size,
        feature_dim=args.feature_dim,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        labels=labels,
        class_names=class_names,
        paths=paths,
    )
    print(f"Saved {args.output}")
    print(f"features={features.shape} labels={labels.shape} classes={len(class_names)}")


if __name__ == "__main__":
    main()
