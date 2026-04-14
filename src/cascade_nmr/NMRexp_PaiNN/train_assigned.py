from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset


MODELS_DIR = Path(__file__).resolve().parents[1]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from torch_training import _collate_graphs, _to_torch_batch  # noqa: E402

PAINN_DIR = MODELS_DIR / "Predict_SMILES_FF"
if str(PAINN_DIR) not in sys.path:
    sys.path.insert(0, str(PAINN_DIR))

from torch_model import PaiNNConfig, PaiNNModel  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import _build_edge_indices_from_coords  # noqa: E402


def _compute_stacked_offsets(sizes, repeats):
    return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)


def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()


def Mol_iter(df):
    for _, row in df.iterrows():
        yield row["Mol"], row["atom_index"]


def ragged_const(inp_arr):
    raise RuntimeError("ragged_const is no longer used; switch to torch batching.")


def load_atom_tokenizer_map(preprocessor_path: Path) -> tuple[dict[int, int], int]:
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    os.environ.setdefault("NFP_NO_KERAS", "1")
    preproc_dir = preprocessor_path.parent.resolve()
    modules_dir = preproc_dir / "modules"
    for path in (preproc_dir, modules_dir):
        if path.exists():
            p = str(path)
            if p not in sys.path:
                sys.path.insert(0, p)

    main_module = sys.modules.get("__main__")
    if main_module is None or not isinstance(main_module, types.ModuleType):
        main_module = types.ModuleType("__main__")
        sys.modules["__main__"] = main_module
    for name, value in (
        ("atomic_number_tokenizer", atomic_number_tokenizer),
        ("Mol_iter", Mol_iter),
        ("_compute_stacked_offsets", _compute_stacked_offsets),
        ("ragged_const", ragged_const),
    ):
        setattr(main_module, name, value)

    with preprocessor_path.open("rb") as fh:
        bundle = pickle.load(fh)  # noqa: S301
    preprocessor = bundle.get("preprocessor")
    if preprocessor is None:
        raise KeyError(f"'preprocessor' key not found in {preprocessor_path}")

    tokenizer = getattr(preprocessor, "atom_tokenizer", None)
    tok_data = getattr(tokenizer, "_data", None)
    if not isinstance(tok_data, dict):
        raise TypeError("Could not read atom tokenizer mapping from preprocessor bundle.")

    mapping = {int(k): int(v) for k, v in tok_data.items() if isinstance(k, int)}
    unk = int(tok_data.get("unk", 1))
    if not mapping:
        raise ValueError("Atom tokenizer mapping is empty in preprocessor bundle.")
    return mapping, unk


def canonical_target(target: str) -> str:
    t = target.strip().lower().replace(" ", "")
    if t in {"13c", "c13", "c"}:
        return "13C"
    if t in {"1h", "h1", "h"}:
        return "1H"
    raise ValueError(f"Unknown target: {target}")


def target_atomic_num(target: str) -> int:
    return 6 if canonical_target(target) == "13C" else 1


def _as_float(val) -> float:
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            raise ValueError("Empty target list")
        return float(np.mean(val))
    return float(val)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_csv(path: Path, header: Sequence[str], row: Sequence) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def _load_entries(entries_path: Path) -> List[dict]:
    with entries_path.open("rb") as handle:
        entries = pickle.load(handle)  # noqa: S301
    if not isinstance(entries, list):
        raise TypeError(f"Expected list in entries file: {entries_path}")
    return entries


def _load_splits(splits_path: Path) -> Tuple[List[int], List[int], List[int]]:
    with splits_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    splits = payload.get("splits", payload)
    if not isinstance(splits, dict):
        raise TypeError(f"Invalid splits format in {splits_path}")
    train_idx = list(splits.get("train", []))
    val_idx = list(splits.get("val", []))
    test_idx = list(splits.get("test", []))
    return train_idx, val_idx, test_idx


def _infer_default_entries_path(data_dir: Path, target: str) -> Path:
    t = canonical_target(target).lower()
    return data_dir / f"dft8k_dft_{t}_entries.pkl"


def _infer_default_splits_path(data_dir: Path, target: str) -> Path:
    t = canonical_target(target).lower()
    return data_dir / f"dft8k_dft_{t}_splits.json"


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 50
    lr: float = 2e-4
    reduce_lr_factor: float = 0.85
    reduce_lr_patience: int = 6
    min_lr: float = 1e-5
    early_stop_patience: int = 10
    grad_clip: float = 5.0
    normalize_targets: bool = False
    mean_ppm: float = 0.0
    std_ppm: float = 1.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class AssignedPaiNNDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[dict],
        indices: Sequence[int],
        *,
        target: str,
        n_neighbors: int,
        cutoff: float,
        atom_token_map: dict[int, int] | None,
        atom_token_unk: int,
    ) -> None:
        self.samples: List[Tuple[dict, np.ndarray, str]] = []
        target_num = target_atomic_num(target)

        for idx in indices:
            entry = entries[int(idx)]
            mol_block = entry.get("mol_block")
            shift_dict = entry.get("exp_shift_dict", {})
            if not mol_block or not shift_dict:
                continue

            try:
                mol = Chem.MolFromMolBlock(mol_block, sanitize=True, removeHs=False)
            except Exception:
                mol = None
            if mol is None or mol.GetNumConformers() == 0:
                continue

            conf = mol.GetConformer()
            n_atom = int(mol.GetNumAtoms())
            atom_numbers = np.array(
                [int(atom.GetAtomicNum()) for atom in mol.GetAtoms()],
                dtype=np.int64,
            )
            coords = np.array(conf.GetPositions(), dtype=np.float32)
            if coords.shape != (n_atom, 3):
                continue

            target_atoms = []
            target_vals = []
            for atom_idx_raw, shift_raw in shift_dict.items():
                atom_idx = int(atom_idx_raw)
                if atom_idx < 0 or atom_idx >= n_atom:
                    continue
                if int(atom_numbers[atom_idx]) != target_num:
                    continue
                try:
                    shift = _as_float(shift_raw)
                except Exception:
                    continue
                if not np.isfinite(shift):
                    continue
                target_atoms.append(atom_idx)
                target_vals.append(float(shift))

            if not target_atoms:
                continue

            order = np.argsort(np.asarray(target_atoms, dtype=np.int64))
            target_atoms_arr = np.asarray(target_atoms, dtype=np.int64)[order]
            target_vals_arr = np.asarray(target_vals, dtype=np.float32)[order]

            atom_index = np.full(n_atom, -1, dtype=np.int64)
            atom_index[target_atoms_arr] = np.arange(target_atoms_arr.shape[0], dtype=np.int64)

            edge_indices = _build_edge_indices_from_coords(
                coords,
                n_neighbors=n_neighbors,
                cutoff=cutoff,
            )

            node_attributes = (
                np.array(
                    [atom_token_map.get(int(z), atom_token_unk) for z in atom_numbers],
                    dtype=np.int64,
                )
                if atom_token_map is not None
                else atom_numbers
            )
            graph = {
                "n_atom": n_atom,
                "n_pro": int(target_vals_arr.shape[0]),
                "node_attributes": node_attributes,
                "node_coordinates": coords,
                "edge_indices": edge_indices,
                "atom_index": atom_index,
            }
            smiles = str(entry.get("smiles", ""))
            self.samples.append((graph, target_vals_arr, smiles))

        if not self.samples:
            raise RuntimeError("AssignedPaiNNDataset is empty after filtering.")

        self.all_targets_ppm = np.concatenate([sample[1] for sample in self.samples], axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


AssignedPaperDataset = AssignedPaiNNDataset


def collate_assigned(batch: Sequence[Tuple[dict, np.ndarray, str]]):
    graphs = [item[0] for item in batch]
    targets_ppm = [item[1] for item in batch]
    smiles = [item[2] for item in batch]
    batch_graph = _collate_graphs(graphs)
    batch_targets = np.concatenate(targets_ppm, axis=0).reshape(-1, 1).astype(np.float32)
    return batch_graph, batch_targets, smiles


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    config: TrainConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mae_ppm = 0.0
    total_count = 0

    for batch_graph, batch_targets_ppm, _ in loader:
        batch_torch = _to_torch_batch(batch_graph, config.device, config.dtype)
        targets_ppm = torch.as_tensor(batch_targets_ppm, device=config.device, dtype=config.dtype)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred_ppm = model(batch_torch)
        if pred_ppm.shape != targets_ppm.shape:
            raise RuntimeError(
                f"Shape mismatch: pred={tuple(pred_ppm.shape)} targets={tuple(targets_ppm.shape)}"
            )

        if config.normalize_targets and config.std_ppm > 0:
            pred_loss = (pred_ppm - config.mean_ppm) / config.std_ppm
            targets_loss = (targets_ppm - config.mean_ppm) / config.std_ppm
        else:
            pred_loss = pred_ppm
            targets_loss = targets_ppm

        loss = torch.mean(torch.abs(pred_loss - targets_loss))
        mae_ppm = torch.mean(torch.abs(pred_ppm - targets_ppm))

        if is_train:
            loss.backward()
            if config.grad_clip and config.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    continue
            grads_finite = True
            for param in model.parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    grads_finite = False
                    break
            if not grads_finite:
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        count = int(targets_ppm.shape[0])
        total_loss += float(loss.item()) * count
        total_mae_ppm += float(mae_ppm.item()) * count
        total_count += count

    if total_count == 0:
        return float("inf"), float("inf")
    return total_loss / total_count, total_mae_ppm / total_count


def save_checkpoint(
    checkpoint_path: Path,
    *,
    epoch: int,
    model: PaiNNModel,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    best_epoch: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: PaiNNModel,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[int, float, int]:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch = int(ckpt.get("epoch", 0))
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    best_epoch = int(ckpt.get("best_epoch", 0))
    return epoch, best_val_loss, best_epoch


def _write_model_metadata(
    output_dir: Path,
    *,
    target: str,
    normalize_targets: bool,
    mean_ppm: float,
    std_ppm: float,
    train_target_mean_ppm: float,
    train_target_std_ppm: float,
    entries_path: Path,
    splits_path: Path,
    train_size: int,
    val_size: int,
    test_size: int,
) -> None:
    output_scale = float(std_ppm if normalize_targets else 1.0)
    output_bias = float(mean_ppm if normalize_targets else 0.0)
    metadata = {
        "model_family": "CASCADE_PaiNN",
        "target": str(target),
        "output_units": "normalized" if normalize_targets else "ppm",
        "output_scale": output_scale,
        "output_bias": output_bias,
        "normalize_targets": bool(normalize_targets),
        "target_mean_ppm": float(train_target_mean_ppm),
        "target_std_ppm": float(train_target_std_ppm),
        "entries_path": str(entries_path),
        "splits_path": str(splits_path),
        "train_molecules": int(train_size),
        "val_molecules": int(val_size),
        "test_molecules": int(test_size),
    }
    path = output_dir / "model_metadata.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def train_assigned(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = _load_entries(args.entries_path)
    train_idx, val_idx, test_idx = _load_splits(args.splits_path)

    if not val_idx and test_idx:
        split_point = max(1, len(test_idx) // 2)
        val_idx = test_idx[:split_point]
        test_idx = test_idx[split_point:]
        print(
            f"[warn] splits had no val set; reused test for val/test split: val={len(val_idx)} test={len(test_idx)}"
        )
    if not test_idx and val_idx:
        split_point = max(1, len(val_idx) // 2)
        test_idx = val_idx[split_point:]
        val_idx = val_idx[:split_point]
        if not test_idx and val_idx:
            test_idx = val_idx[-1:]
            val_idx = val_idx[:-1]
        print(
            f"[warn] splits had no test set; split val into val/test: val={len(val_idx)} test={len(test_idx)}"
        )
    if not val_idx or not test_idx:
        raise RuntimeError(
            f"Need non-empty val and test splits after adjustment, got val={len(val_idx)} test={len(test_idx)}"
        )

    if args.train_size and args.train_size > 0:
        train_idx = train_idx[: int(args.train_size)]
    if args.max_val_mols and args.max_val_mols > 0:
        val_idx = val_idx[: int(args.max_val_mols)]
    if args.max_test_mols and args.max_test_mols > 0:
        test_idx = test_idx[: int(args.max_test_mols)]

    atom_token_map = None
    atom_token_unk = 1
    if not args.no_atom_tokenizer_map:
        atom_token_map, atom_token_unk = load_atom_tokenizer_map(args.atom_tokenizer_preprocessor)

    train_set = AssignedPaiNNDataset(
        entries,
        train_idx,
        target=args.target,
        n_neighbors=args.n_neighbors,
        cutoff=args.cutoff,
        atom_token_map=atom_token_map,
        atom_token_unk=atom_token_unk,
    )
    val_set = AssignedPaiNNDataset(
        entries,
        val_idx,
        target=args.target,
        n_neighbors=args.n_neighbors,
        cutoff=args.cutoff,
        atom_token_map=atom_token_map,
        atom_token_unk=atom_token_unk,
    )
    test_set = AssignedPaiNNDataset(
        entries,
        test_idx,
        target=args.target,
        n_neighbors=args.n_neighbors,
        cutoff=args.cutoff,
        atom_token_map=atom_token_map,
        atom_token_unk=atom_token_unk,
    )

    train_mean = float(np.mean(train_set.all_targets_ppm))
    train_std = float(np.std(train_set.all_targets_ppm))
    if not np.isfinite(train_std) or train_std <= 0:
        train_std = 1.0

    mean_ppm = float(args.mean) if args.mean is not None else train_mean
    std_ppm = float(args.std) if args.std is not None else train_std
    if std_ppm <= 0:
        std_ppm = 1.0

    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        reduce_lr_factor=args.reduce_lr_factor,
        reduce_lr_patience=args.reduce_lr_patience,
        min_lr=args.min_lr,
        early_stop_patience=args.early_stop_patience,
        grad_clip=args.grad_clip,
        normalize_targets=bool(args.normalize_targets),
        mean_ppm=mean_ppm,
        std_ppm=std_ppm,
        device=args.device,
        dtype=torch.float32,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": str(config.device).startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_assigned,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_assigned,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_assigned,
        **loader_kwargs,
    )

    model = PaiNNModel(
        PaiNNConfig(
            input_dim=args.input_dim,
            units=args.units,
            num_radial=args.num_radial,
            cutoff=args.cutoff,
            depth=args.depth,
        )
    ).to(config.device, dtype=config.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.reduce_lr_factor,
        patience=config.reduce_lr_patience,
        min_lr=config.min_lr,
    )

    best_model_path = output_dir / "best_model.pt"
    checkpoint_path = output_dir / "checkpoint.pt"
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    log_path = output_dir / "log_painn_assigned.csv"
    summary_path = output_dir / "summary_metrics.json"

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 0
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume requested but checkpoint missing: {checkpoint_path}")
        last_epoch, best_val_loss, best_epoch = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config.device,
        )
        start_epoch = int(last_epoch) + 1
        print(
            f"resumed checkpoint={checkpoint_path} start_epoch={start_epoch} "
            f"best_val_loss={best_val_loss:.6f}@{best_epoch}"
        )

    print(
        "dataset_summary "
        f"train={len(train_set)} val={len(val_set)} test={len(test_set)} "
        f"normalize_targets={config.normalize_targets} mean_ppm={config.mean_ppm:.6f} std_ppm={config.std_ppm:.6f}"
    )

    for epoch in range(start_epoch, config.epochs + 1):
        train_loss, train_mae_ppm = run_epoch(model, train_loader, config, optimizer)
        val_loss, val_mae_ppm = run_epoch(model, val_loader, config, None)
        lr_plateau.step(val_loss)

        _log_csv(
            log_path,
            ["epoch", "loss", "mae_ppm", "val_loss", "val_mae_ppm", "lr"],
            [
                int(epoch),
                round(float(train_loss), 8),
                round(float(train_mae_ppm), 8),
                round(float(val_loss), 8),
                round(float(val_mae_ppm), 8),
                float(optimizer.param_groups[0]["lr"]),
            ],
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} train_mae_ppm={train_mae_ppm:.6f} "
            f"val_loss={val_loss:.6f} val_mae_ppm={val_mae_ppm:.6f} lr={optimizer.param_groups[0]['lr']:.6g}"
        )

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            torch.save(model.state_dict(), best_model_path)
            save_checkpoint(
                best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
            )
            patience = 0
            should_stop = False
        else:
            patience += 1
            should_stop = patience >= config.early_stop_patience

        save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
        )
        if should_stop:
            print(f"early_stop epoch={epoch} best_val_loss={best_val_loss:.6f}@{best_epoch}")
            break

    if best_model_path.exists():
        best_state = torch.load(str(best_model_path), map_location=config.device)
        model.load_state_dict(best_state, strict=True)

    test_loss, test_mae_ppm = run_epoch(model, test_loader, config, None)
    summary = {
        "target": args.target,
        "entries_path": str(args.entries_path),
        "splits_path": str(args.splits_path),
        "train_molecules": int(len(train_set)),
        "val_molecules": int(len(val_set)),
        "test_molecules": int(len(test_set)),
        "normalize_targets": bool(config.normalize_targets),
        "mean_ppm": float(config.mean_ppm),
        "std_ppm": float(config.std_ppm),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_mae_ppm": float(test_mae_ppm),
        "best_model_path": str(best_model_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))

    _write_model_metadata(
        output_dir,
        target=args.target,
        normalize_targets=config.normalize_targets,
        mean_ppm=config.mean_ppm,
        std_ppm=config.std_ppm,
        train_target_mean_ppm=train_mean,
        train_target_std_ppm=train_std,
        entries_path=args.entries_path,
        splits_path=args.splits_path,
        train_size=len(train_set),
        val_size=len(val_set),
        test_size=len(test_set),
    )


def main() -> None:
    data_root = os.environ.get("CASCADE_DATA_ROOT", "./data")
    default_data_dir = Path(data_root) / "original_cascade_data" / "DFT8K"

    parser = argparse.ArgumentParser(
        description="Train CASCADE PaiNN on assigned DFT8K labels (ppm-native by default)."
    )
    parser.add_argument("--target", choices=["13C", "1H"], default="13C")
    parser.add_argument("--data_dir", "--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--entries_path", "--entries-path", type=Path, default=None)
    parser.add_argument("--splits_path", "--splits-path", type=Path, default=None)
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=27407)
    parser.add_argument("--train_size", "--train-size", type=int, default=0)
    parser.add_argument("--max_val_mols", "--max-val-mols", type=int, default=0)
    parser.add_argument("--max_test_mols", "--max-test-mols", type=int, default=0)

    parser.add_argument("--batch_size", "--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--reduce_lr_factor", "--reduce-lr-factor", type=float, default=0.85)
    parser.add_argument("--reduce_lr_patience", "--reduce-lr-patience", type=int, default=6)
    parser.add_argument("--min_lr", "--min-lr", type=float, default=1e-5)
    parser.add_argument("--early_stop_patience", "--early-stop-patience", type=int, default=10)
    parser.add_argument("--grad_clip", "--grad-clip", type=float, default=5.0)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--normalize_targets", "--normalize-targets", action="store_true", default=False)
    parser.add_argument("--mean", type=float, default=None)
    parser.add_argument("--std", type=float, default=None)

    parser.add_argument("--input_dim", "--input-dim", type=int, default=256)
    parser.add_argument("--units", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_radial", "--num-radial", type=int, default=20)
    parser.add_argument("--n_neighbors", "--n-neighbors", type=int, default=20)
    parser.add_argument("--cutoff", type=float, default=5.0)

    parser.add_argument(
        "--atom_tokenizer_preprocessor",
        "--atom-tokenizer-preprocessor",
        type=Path,
        default=PAINN_DIR / "preprocessor_orig.p",
    )
    parser.add_argument("--no_atom_tokenizer_map", "--no-atom-tokenizer-map", action="store_true", default=False)

    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    args.target = canonical_target(args.target)
    if args.entries_path is None:
        args.entries_path = _infer_default_entries_path(args.data_dir, args.target)
    if args.splits_path is None:
        args.splits_path = _infer_default_splits_path(args.data_dir, args.target)
    if not args.entries_path.exists():
        raise FileNotFoundError(f"Entries file not found: {args.entries_path}")
    if not args.splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {args.splits_path}")
    train_assigned(args)


if __name__ == "__main__":
    main()
