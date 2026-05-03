"""Train IEEE-24 graph classification baselines with MLOps instrumentation.

This script covers the next implementation step after data validation:
- GCN baseline
- GINe baseline
- optional TransformerConv baseline
- MLflow logging when available
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from powergraph_data import PowerGraphDataset, SplitConfig, make_dataloaders
from powergraph_models import GCN_Graph, GINe_Graph, Transformer_Graph, GPS_Graph


@dataclass
class TrainConfig:
    architecture: str = "GINe"
    hidden_dim: int = 32
    num_layers: int = 3
    pooling: str = "max"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    focal_gamma: float = 2.0
    alpha_cap: float = 10.0
    dropout: float = 0.3
    epochs: int = 200
    patience: int = 30
    batch_size: int = 16
    heads: int = 4
    grid: str = "ieee24"
    task: str = "binary"
    seed: int = 23


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.architecture == "GCN":
        return GCN_Graph(
            num_node_features=3,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_classes=2,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
        )
    if cfg.architecture == "GINe":
        return GINe_Graph(
            num_node_features=3,
            num_edge_features=5,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_classes=2,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
        )
    if cfg.architecture == "Transformer":
        return Transformer_Graph(
            num_node_features=3,
            num_edge_features=5,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_classes=2,
            heads=cfg.heads,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
        )
    if cfg.architecture == "GPS":
        return GPS_Graph(
            num_node_features=3,
            num_edge_features=5,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_classes=2,
            heads=cfg.heads,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
            pe_dim=8,
        )
    raise ValueError(f"Unknown architecture: {cfg.architecture}")


def get_class_weights(train_dataset, alpha_cap: float = 10.0) -> torch.Tensor:
    labels = [int(train_dataset[i].y.item()) for i in range(len(train_dataset))]
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    alpha = len(labels) / (2.0 * counts)
    max_ratio = float(alpha.max() / alpha.min())
    if max_ratio > alpha_cap:
        alpha = np.clip(alpha, alpha.min(), alpha.min() * alpha_cap)
    return torch.tensor(alpha, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = criterion(logits, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    for data in loader:
        data = data.to(device)
        logits = model(data)
        loss = criterion(logits, data.y.view(-1))
        total_loss += float(loss.item()) * data.num_graphs
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        labels = data.y.view(-1).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    labels_arr = np.asarray(all_labels)
    preds_arr = np.asarray(all_preds)
    probs_arr = np.asarray(all_probs)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "bal_acc": balanced_accuracy_score(labels_arr, preds_arr),
        "macro_f1": f1_score(labels_arr, preds_arr, average="macro"),
        "auroc": 0.0,
        "pr_auc": 0.0,
        "preds": preds_arr,
        "labels": labels_arr,
        "probs": probs_arr,
    }
    try:
        metrics["auroc"] = roc_auc_score(labels_arr, probs_arr)
        metrics["pr_auc"] = average_precision_score(labels_arr, probs_arr)
    except ValueError:
        pass
    return metrics


def maybe_import_mlflow():
    try:
        import mlflow  # type: ignore
    except Exception:
        mlflow = None
    return mlflow


def run_training(cfg: TrainConfig, enable_mlops: bool = True) -> Dict[str, object]:
    seed_everything(cfg.seed)

    print(f"Loading dataset: grid={cfg.grid}, task={cfg.task} ...")

    dataset_root = os.path.join("dataset_cascades", cfg.grid, cfg.grid)
    dataset = PowerGraphDataset(root=dataset_root, grid_name=cfg.grid, task=cfg.task)
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = make_dataloaders(
        dataset,
        batch_size=cfg.batch_size,
        split_cfg=SplitConfig(random_state=cfg.seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    print(f"Dataset loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"Device: {device} | Architecture: {cfg.architecture} | Starting training...")

    class_weights = get_class_weights(train_dataset, alpha_cap=cfg.alpha_cap).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=cfg.focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    mlflow = maybe_import_mlflow()
    output_dir = os.path.join("artifacts", "training", f"{cfg.grid}_{cfg.architecture.lower()}")
    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, "best_model.pt")
    history_path = os.path.join(output_dir, "history.json")

    best_val_bal_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    history: List[Dict[str, float]] = []

    if enable_mlops and mlflow is not None:
        mlflow.set_experiment(f"powergraph-{cfg.grid}")
        mlflow.start_run(run_name=f"{cfg.architecture.lower()}_baseline")
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("num_train", len(train_dataset))
        mlflow.log_param("num_val", len(val_dataset))
        mlflow.log_param("num_test", len(test_dataset))
        mlflow.log_param("class_weights", class_weights.detach().cpu().tolist())

    try:
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_metrics["bal_acc"])

            epoch_row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_bal_acc": float(val_metrics["bal_acc"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_auroc": float(val_metrics["auroc"]),
                "val_pr_auc": float(val_metrics["pr_auc"]),
            }
            history.append(epoch_row)

            if epoch == 1 or epoch % 10 == 0 or patience_counter >= cfg.patience - 1:
                print(
                    f"Epoch {epoch:3d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_bal_acc={val_metrics['bal_acc']:.4f} | "
                    f"val_f1={val_metrics['macro_f1']:.4f} | "
                    f"patience={patience_counter}/{cfg.patience}"
                )

            if enable_mlops and mlflow is not None:
                mlflow.log_metrics({k: v for k, v in epoch_row.items() if k != "epoch"}, step=epoch)

            if val_metrics["bal_acc"] > best_val_bal_acc:
                best_val_bal_acc = float(val_metrics["bal_acc"])
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                break
    finally:
        if enable_mlops and mlflow is not None and mlflow.active_run() is not None:
            mlflow.log_param("best_epoch", best_epoch)
            mlflow.log_artifact(best_path)
            mlflow.end_run()

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if enable_mlops and mlflow is not None:
        mlflow.set_experiment(f"powergraph-{cfg.grid}")
        with mlflow.start_run(run_name=f"{cfg.architecture.lower()}_test_summary"):
            mlflow.log_metrics(
                {
                    "test_loss": float(test_metrics["loss"]),
                    "test_bal_acc": float(test_metrics["bal_acc"]),
                    "test_macro_f1": float(test_metrics["macro_f1"]),
                    "test_auroc": float(test_metrics["auroc"]),
                    "test_pr_auc": float(test_metrics["pr_auc"]),
                }
            )
            mlflow.log_artifact(history_path)

    print(f"Architecture: {cfg.architecture}")
    print(f"Best val balanced accuracy: {best_val_bal_acc:.4f} @ epoch {best_epoch}")
    print(f"Test balanced accuracy: {test_metrics['bal_acc']:.4f}")
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    return {
        "config": asdict(cfg),
        "best_val_bal_acc": best_val_bal_acc,
        "best_epoch": best_epoch,
        "test_metrics": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in test_metrics.items()
            if k not in {"preds", "labels", "probs"}
        },
        "history_path": history_path,
        "best_model_path": best_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IEEE-24 PowerGraph baselines")
    parser.add_argument("--architecture", choices=["GCN", "GINe", "Transformer", "GPS"], default="GINe")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--pooling", type=str, default="max", choices=["max", "sum", "mean_max", "mean_max_sum"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--alpha-cap", type=float, default=10.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--grid", type=str, default="ieee24")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pooling=args.pooling,
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        alpha_cap=args.alpha_cap,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        heads=args.heads,
        grid=args.grid,
        seed=args.seed,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
