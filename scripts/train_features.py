import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dhgnn import DHGNNClassifier


@dataclass
class TrainConfig:
    data: str
    output_dir: str
    train_ratio: float = 0.8
    batch_size: int = 8
    epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    num_levels: int = 3
    graph_depth: int = 1
    readout_type: str = "mean"
    adjacency_mode: str = "dynamic"
    dynamic_topk: int = 4
    use_edge_state: bool = True
    include_input_readout: bool = False
    label_smoothing: float = 0.0
    seed: int = 42
    device: str = "cpu"


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, indices: List[int]):
        self.features = features
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        return {
            "features": torch.from_numpy(self.features[index]).float(),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "index": index,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_feature_npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "features" not in data or "labels" not in data:
            raise ValueError("Input .npz must contain 'features' and 'labels'.")
        features = data["features"]
        labels = data["labels"]
        class_names = data["class_names"] if "class_names" in data else None
        paths = data["paths"] if "paths" in data else None

    if features.ndim != 4:
        raise ValueError(f"features must have shape [N,H,W,C], got {features.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {labels.shape}.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same first dimension.")
    if features.shape[0] < 2:
        raise ValueError("Dataset must contain at least two samples.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be integer class indices.")

    features = features.astype(np.float32, copy=False)
    labels = labels.astype(np.int64, copy=False)
    if labels.min() < 0:
        raise ValueError("labels must be non-negative.")

    if class_names is None:
        num_classes = int(labels.max()) + 1
        class_names = np.array([f"class_{index}" for index in range(num_classes)])
    else:
        class_names = class_names.astype(str)
        num_classes = len(class_names)
    if int(labels.max()) >= num_classes:
        raise ValueError("labels contain an index outside class_names.")

    if paths is not None:
        paths = paths.astype(str)
        if len(paths) != len(labels):
            raise ValueError("paths must have the same length as labels.")

    return features, labels, class_names.tolist(), None if paths is None else paths.tolist()


def stratified_split(labels: np.ndarray, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    rng = random.Random(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []

    for class_index in sorted(set(int(label) for label in labels)):
        class_indices = [index for index, label in enumerate(labels) if int(label) == class_index]
        rng.shuffle(class_indices)
        if len(class_indices) < 2:
            raise ValueError(
                f"Class {class_index} has {len(class_indices)} sample(s); at least 2 are required."
            )
        split = int(round(len(class_indices) * train_ratio))
        split = min(max(split, 1), len(class_indices) - 1)
        train_indices.extend(class_indices[:split])
        test_indices.extend(class_indices[split:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def collate_fn(batch):
    features = torch.stack([item["features"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    indices = [item["index"] for item in batch]
    return {"features": features, "labels": labels, "indices": indices}


def compute_accuracy(labels, preds) -> float:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    return float((labels == preds).mean())


def compute_macro_f1(labels, preds, num_classes: int) -> float:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    f1_values = []
    for class_index in range(num_classes):
        tp = np.logical_and(labels == class_index, preds == class_index).sum()
        fp = np.logical_and(labels != class_index, preds == class_index).sum()
        fn = np.logical_and(labels == class_index, preds != class_index).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append(float(2 * precision * recall / (precision + recall)))
    return float(np.mean(f1_values))


def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    preds_all = []
    labels_all = []

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        preds_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())

    return total_loss / len(loader.dataset), compute_accuracy(labels_all, preds_all)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    preds_all = []
    labels_all = []
    indices_all = []

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * features.size(0)
        preds_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
        indices_all.extend(batch["indices"])

    avg_loss = total_loss / len(loader.dataset)
    accuracy = compute_accuracy(labels_all, preds_all)
    macro_f1 = compute_macro_f1(labels_all, preds_all, num_classes=num_classes)
    return avg_loss, accuracy, macro_f1, preds_all, labels_all, indices_all


def main():
    parser = argparse.ArgumentParser(description="Train DHGNN on precomputed patch-grid features.")
    parser.add_argument("--data", type=Path, required=True, help=".npz with features [N,H,W,C] and labels [N].")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-levels", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--graph-depth", type=int, default=1)
    parser.add_argument("--readout-type", type=str, default="mean", choices=["mean", "weighted"])
    parser.add_argument("--adjacency-mode", type=str, default="dynamic", choices=["fixed", "dynamic"])
    parser.add_argument("--dynamic-topk", type=int, default=4)
    parser.add_argument("--use-edge-state", action="store_true")
    parser.add_argument("--no-use-edge-state", dest="use_edge_state", action="store_false")
    parser.set_defaults(use_edge_state=True)
    parser.add_argument("--include-input-readout", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = TrainConfig(
        data=str(args.data),
        output_dir=str(args.output_dir),
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_levels=args.num_levels,
        graph_depth=args.graph_depth,
        readout_type=args.readout_type,
        adjacency_mode=args.adjacency_mode,
        dynamic_topk=args.dynamic_topk,
        use_edge_state=args.use_edge_state,
        include_input_readout=args.include_input_readout,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        device=args.device,
    )

    if cfg.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {cfg.epochs}")
    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {cfg.batch_size}")
    if cfg.graph_depth <= 0:
        raise ValueError(f"graph_depth must be positive, got {cfg.graph_depth}")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    set_seed(cfg.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features, labels, class_names, paths = load_feature_npz(args.data)
    train_indices, test_indices = stratified_split(labels, cfg.train_ratio, cfg.seed)
    train_dataset = FeatureDataset(features, labels, train_indices)
    test_dataset = FeatureDataset(features, labels, test_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = torch.device(cfg.device)
    model = DHGNNClassifier(
        num_classes=len(class_names),
        feature_dim=int(features.shape[-1]),
        hidden_dim=cfg.hidden_dim,
        num_levels=cfg.num_levels,
        graph_depth=cfg.graph_depth,
        readout_type=cfg.readout_type,
        adjacency_mode=cfg.adjacency_mode,
        dynamic_topk=cfg.dynamic_topk,
        use_edge_state=cfg.use_edge_state,
        include_input_readout=cfg.include_input_readout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    save_json(asdict(cfg), args.output_dir / "config.json")
    save_json(
        {
            "features_shape": list(features.shape),
            "num_classes": len(class_names),
            "class_names": class_names,
            "num_train": len(train_dataset),
            "num_test": len(test_dataset),
            "has_paths": paths is not None,
        },
        args.output_dir / "dataset_info.json",
    )

    history = []
    start_time = time.time()
    final_eval = None
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, test_mf1, preds, eval_labels, eval_indices = evaluate(
            model,
            test_loader,
            criterion,
            device,
            num_classes=len(class_names),
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_mf1": test_mf1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        final_eval = (preds, eval_labels, eval_indices, record)
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_mf1={test_mf1:.4f}"
        )

        save_json(history, args.output_dir / "history.json")
        with (args.output_dir / "history.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            writer.writeheader()
            writer.writerows(history)

    preds, eval_labels, eval_indices, final_record = final_eval
    elapsed_seconds = time.time() - start_time
    prediction_rows = []
    for pred, label, index in zip(preds, eval_labels, eval_indices):
        row = {
            "index": int(index),
            "pred": int(pred),
            "label": int(label),
            "pred_class_name": class_names[int(pred)],
            "label_class_name": class_names[int(label)],
        }
        if paths is not None:
            row["path"] = paths[int(index)]
        prediction_rows.append(row)

    save_json(
        {
            "test_loss": final_record["test_loss"],
            "test_acc": final_record["test_acc"],
            "test_mf1": final_record["test_mf1"],
            "elapsed_seconds": elapsed_seconds,
            "evaluated_epoch": cfg.epochs,
        },
        args.output_dir / "test_metrics.json",
    )
    save_json({"predictions": prediction_rows}, args.output_dir / "test_predictions.json")


if __name__ == "__main__":
    main()
