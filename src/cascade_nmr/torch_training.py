from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    mean: float
    std: float
    batch_size: int = 64
    epochs: int = 250
    lr: float = 7.5e-4
    lr_decay_every: int = 35
    lr_decay_factor: float = 0.96
    reduce_lr_factor: float = 0.85
    reduce_lr_patience: int = 6
    min_lr: float = 1e-4
    early_stop_patience: int = 15
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


@dataclass
class GPRConfig:
    mean: float
    std: float
    batch_size: int = 64
    epochs: int = 150
    lr: float = 2.5e-4
    lr_decay_every: int = 50
    lr_decay_factor: float = 0.96
    reduce_lr_factor: float = 0.85
    reduce_lr_patience: int = 10
    min_lr: float = 7.5e-5
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    inducing_points: Path = Path("inducing_index_points_250.npy")
    backbone_weights: Optional[Path] = None
    freeze_backbone: bool = True


def _prepare_pickle_context(base_dir: Path) -> None:
    os.environ.setdefault("NFP_NO_KERAS", "1")
    modules_path = base_dir / "modules"
    if str(modules_path) not in sys.path:
        sys.path.insert(0, str(modules_path))

    def atomic_number_tokenizer(atom):
        return atom.GetAtomicNum()

    def Mol_iter(df):
        for _, row in df.iterrows():
            yield row["Mol"], row["Atomic_Indices"]

    def _compute_stacked_offsets(sizes, repeats):
        return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)

    def ragged_const(inp_arr):
        raise RuntimeError("ragged_const should not be used in torch training.")

    main_module = sys.modules.get("__main__")
    if main_module is None:
        return

    for name, value in (
        ("atomic_number_tokenizer", atomic_number_tokenizer),
        ("Mol_iter", Mol_iter),
        ("_compute_stacked_offsets", _compute_stacked_offsets),
        ("ragged_const", ragged_const),
    ):
        setattr(main_module, name, value)


def load_processed_inputs(data_dir: Path) -> dict:
    _prepare_pickle_context(data_dir.parent)
    with open(data_dir / "processed_inputs.p", "rb") as handle:
        return pickle.load(handle)  # noqa: S301


def _ensure_torch_model_path() -> None:
    model_dir = Path(__file__).resolve().parent / "Predict_SMILES_FF"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))


def load_targets(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_pickle(data_dir / "train.pkl.gz")
    valid = pd.read_pickle(data_dir / "valid.pkl.gz")
    test = pd.read_pickle(data_dir / "test.pkl.gz")
    return train.Shifts.values, valid.Shifts.values, test.Shifts.values


def normalize_targets(targets: Sequence[np.ndarray], mean: float, std: float) -> List[np.ndarray]:
    normalized = []
    for values in targets:
        arr = np.asarray(values, dtype=np.float32)
        normalized.append((arr - mean) / std)
    return normalized


class GraphDataset(Dataset):
    def __init__(self, inputs: Sequence[dict], targets: Optional[Sequence[np.ndarray]] = None) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        if self.targets is None:
            return self.inputs[idx]
        return self.inputs[idx], self.targets[idx]


def _collate_graphs(batch: Sequence[dict]) -> dict:
    node_attributes = []
    node_coordinates = []
    edge_indices = []
    atom_index = []
    node_offset = 0
    target_offset = 0

    for graph in batch:
        n_atom = int(graph["n_atom"])
        n_pro = int(graph["n_pro"])

        node_attributes.append(graph["node_attributes"])
        node_coordinates.append(graph["node_coordinates"])

        edges = graph["edge_indices"].astype(np.int64, copy=False) + node_offset
        edge_indices.append(edges)

        atom_idx = graph["atom_index"].astype(np.int64, copy=True)
        mask = atom_idx >= 0
        atom_idx[mask] += target_offset
        atom_index.append(atom_idx)

        node_offset += n_atom
        target_offset += n_pro

    return {
        "node_attributes": np.concatenate(node_attributes, axis=0),
        "node_coordinates": np.concatenate(node_coordinates, axis=0),
        "edge_indices": np.concatenate(edge_indices, axis=0),
        "atom_index": np.concatenate(atom_index, axis=0),
        "num_targets": target_offset,
    }


def collate_supervised(batch: Sequence[Tuple[dict, np.ndarray]]):
    graphs, targets = zip(*batch)
    batch_graph = _collate_graphs(graphs)
    batch_targets = np.concatenate([np.asarray(t) for t in targets], axis=0).reshape(-1, 1)
    return batch_graph, batch_targets


def collate_inputs(batch: Sequence[dict]) -> dict:
    return _collate_graphs(batch)


def _to_torch_batch(batch: dict, device: str, dtype: torch.dtype) -> dict:
    return {
        "node_attributes": torch.as_tensor(batch["node_attributes"], device=device, dtype=torch.long),
        "node_coordinates": torch.as_tensor(batch["node_coordinates"], device=device, dtype=dtype),
        "edge_indices": torch.as_tensor(batch["edge_indices"], device=device, dtype=torch.long),
        "atom_index": torch.as_tensor(batch["atom_index"], device=device, dtype=torch.long),
        "num_targets": batch["num_targets"],
    }


def _log_csv(path: Path, header: Sequence[str], row: Sequence) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    dtype: torch.dtype,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    total_loss = 0.0

    for batch_graphs, batch_targets in loader:
        batch = _to_torch_batch(batch_graphs, device, dtype)
        targets = torch.as_tensor(batch_targets, device=device, dtype=dtype)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        preds = model(batch)
        loss = torch.mean(torch.abs(preds - targets))

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * targets.shape[0]
        total_abs += torch.sum(torch.abs(preds - targets)).item()
        total_sq += torch.sum((preds - targets) ** 2).item()
        total_count += targets.numel()

    mae = total_abs / max(total_count, 1)
    mse = total_sq / max(total_count, 1)
    rmse = float(np.sqrt(mse))
    avg_loss = total_loss / max(total_count, 1)
    return avg_loss, mae, rmse


def train_supervised(
    data_dir: Path,
    output_dir: Path,
    config: TrainConfig,
    num_workers: int = 0,
) -> None:
    _ensure_torch_model_path()
    from torch_model import PaiNNConfig, PaiNNModel

    inputs = load_processed_inputs(data_dir)
    y_train, y_valid, _ = load_targets(data_dir)
    y_train = normalize_targets(y_train, config.mean, config.std)
    y_valid = normalize_targets(y_valid, config.mean, config.std)

    train_dataset = GraphDataset(inputs["inputs_train"], y_train)
    valid_dataset = GraphDataset(inputs["inputs_valid"], y_valid)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_supervised,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_supervised,
    )

    model = PaiNNModel(PaiNNConfig()).to(config.device, dtype=config.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.reduce_lr_factor,
        patience=config.reduce_lr_patience,
        min_lr=config.min_lr,
    )

    best_val = float("inf")
    patience = 0
    log_path = output_dir / "log_painn.csv"
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        if config.lr_decay_every and epoch % config.lr_decay_every == 0:
            for group in optimizer.param_groups:
                group["lr"] *= config.lr_decay_factor

        train_loss, train_mae, train_rmse = _run_epoch(
            model, train_loader, config.device, config.dtype, optimizer
        )
        val_loss, val_mae, val_rmse = _run_epoch(
            model, valid_loader, config.device, config.dtype, None
        )

        lr_plateau.step(val_loss)

        _log_csv(
            log_path,
            ["epoch", "loss", "mae", "rmse", "val_loss", "val_mae", "val_rmse", "lr"],
            [
                epoch,
                round(train_loss, 8),
                round(train_mae, 8),
                round(train_rmse, 8),
                round(val_loss, 8),
                round(val_mae, 8),
                round(val_rmse, 8),
                optimizer.param_groups[0]["lr"],
            ],
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                break


def _gpr_val_mae(
    model: nn.Module,
    likelihood: nn.Module,
    loader: DataLoader,
    device: str,
    dtype: torch.dtype,
) -> float:
    model.eval()
    likelihood.eval()
    total_abs = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_graphs, batch_targets in loader:
            batch = _to_torch_batch(batch_graphs, device, dtype)
            targets = torch.as_tensor(batch_targets, device=device, dtype=dtype).squeeze(-1)
            preds = likelihood(model(batch))
            mean = preds.mean
            total_abs += torch.sum(torch.abs(mean - targets)).item()
            total_count += targets.numel()
    return total_abs / max(total_count, 1)


def _gpr_epoch_loss(
    model: nn.Module,
    mll: nn.Module,
    loader: DataLoader,
    device: str,
    dtype: torch.dtype,
) -> float:
    model.eval()
    if hasattr(model, "likelihood"):
        model.likelihood.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_graphs, batch_targets in loader:
            batch = _to_torch_batch(batch_graphs, device, dtype)
            targets = torch.as_tensor(batch_targets, device=device, dtype=dtype).squeeze(-1)
            output = model(batch)
            loss = -mll(output, targets)
            total_loss += loss.item() * targets.shape[0]
            total_count += targets.shape[0]
    return total_loss / max(total_count, 1)


def train_gpr(
    data_dir: Path,
    output_dir: Path,
    config: GPRConfig,
    num_workers: int = 0,
) -> None:
    import gpytorch
    _ensure_torch_model_path()
    from torch_model import PaiNNConfig, PaiNNGPModel

    inputs = load_processed_inputs(data_dir)
    y_train, y_valid, _ = load_targets(data_dir)
    y_train = normalize_targets(y_train, config.mean, config.std)
    y_valid = normalize_targets(y_valid, config.mean, config.std)

    train_dataset = GraphDataset(inputs["inputs_train"], y_train)
    valid_dataset = GraphDataset(inputs["inputs_valid"], y_valid)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_supervised,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_supervised,
    )

    inducing_points = np.load(config.inducing_points)
    model = PaiNNGPModel(PaiNNConfig(), inducing_points).to(config.device, dtype=config.dtype)

    if config.backbone_weights:
        state_dict = torch.load(config.backbone_weights, map_location=config.device)
        model.backbone.load_state_dict(state_dict, strict=False)

    if config.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.reduce_lr_factor,
        patience=config.reduce_lr_patience,
        min_lr=config.min_lr,
    )

    train_size = len(y_train)
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp, num_data=train_size)

    log_path = output_dir / "log_painn.csv"
    val_mae_path = output_dir / "val_mae.csv"
    best_path = output_dir / "best_model.pt"
    best_val_mae_path = output_dir / "best_model_val_mae.pt"
    best_train = float("inf")
    best_val_mae = float("inf")

    for epoch in range(1, config.epochs + 1):
        if config.lr_decay_every and optimizer.param_groups[0]["lr"] > config.min_lr:
            if epoch % config.lr_decay_every == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= config.lr_decay_factor

        model.train()
        model.likelihood.train()

        total_loss = 0.0
        total_count = 0
        for batch_graphs, batch_targets in train_loader:
            batch = _to_torch_batch(batch_graphs, config.device, config.dtype)
            targets = torch.as_tensor(batch_targets, device=config.device, dtype=config.dtype).squeeze(-1)

            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = -mll(output, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.shape[0]
            total_count += targets.shape[0]

        train_loss = total_loss / max(total_count, 1)
        val_loss = _gpr_epoch_loss(model, mll, valid_loader, config.device, config.dtype)
        val_mae = _gpr_val_mae(model, model.likelihood, valid_loader, config.device, config.dtype)

        lr_plateau.step(val_mae)

        _log_csv(
            log_path,
            ["epoch", "loss", "val_loss", "val_mae", "lr"],
            [
                epoch,
                round(train_loss, 8),
                round(val_loss, 8),
                round(val_mae, 8),
                optimizer.param_groups[0]["lr"],
            ],
        )
        _log_csv(val_mae_path, ["epoch", "mean_absolute_error"], [epoch, round(val_mae, 8)])

        if train_loss < best_train:
            best_train = train_loss
            torch.save(model.state_dict(), best_path)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_val_mae_path)


def parse_common_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CASCADE-2.0 models with PyTorch.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def update_config_from_args(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.lr = args.lr
    config.device = args.device
    return config


def update_gpr_config_from_args(config: GPRConfig, args: argparse.Namespace) -> GPRConfig:
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.lr = args.lr
    config.device = args.device
    return config
