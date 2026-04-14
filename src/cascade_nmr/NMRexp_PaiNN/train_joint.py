from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader


MODELS_DIR = Path(__file__).resolve().parents[1]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from NMRexp_PaiNN.dataset import (  # noqa: E402
    NMRexpEnsembleShardDataset,
    NMRexpShardDataset,
    collate_unassigned,
    normalize_solvent_key,
    normalize_solvent_name,
)
from NMRexp_PaiNN.train_assigned import (  # noqa: E402
    AssignedPaperDataset,
    canonical_target,
    collate_assigned,
    load_atom_tokenizer_map,
)
from nmrexp.matching import match_indices_dp_batch  # noqa: E402
from torch_training import _to_torch_batch  # noqa: E402

PAINN_DIR = MODELS_DIR / "Predict_SMILES_FF"
if str(PAINN_DIR) not in sys.path:
    sys.path.insert(0, str(PAINN_DIR))

from torch_model import PaiNNConfig, PaiNNModel  # noqa: E402


_INTEG_EXPAND_FULL_TARGET_KEY = "integration_expand_full_target_total"
_INTEG_EXPAND_FULL_OFFSET_KEY = "integration_expand_full_offset"
_INTEG_EXPAND_FULL_PEAK_IDX_KEY = "integration_expand_full_peak_idx"
_ENSEMBLE_SHARD_RE = re.compile(r"nmrexp_ensemble_shard_(\d+)\.npz$")


@dataclass
class UnassignedConfig:
    mean: float
    std: float
    normalize: bool
    dummy_cost: float
    matching_workers: int
    hungarian_solver: str
    integration_matching_mode: str
    max_abs_pred: float
    grad_clip: float
    amp: bool
    amp_dtype: torch.dtype
    tf32: bool
    device: str
    dtype: torch.dtype


@dataclass
class AssignedConfig:
    normalize_targets: bool
    norm_mean_ppm: float
    norm_std_ppm: float
    device: str
    dtype: torch.dtype


@dataclass
class JointStats:
    train_target_loss: float
    train_target_mae: float
    train_assigned_loss: float
    train_assigned_mae: float
    train_total_loss: float
    train_target_matches: int
    train_assigned_atoms: int
    train_steps: int
    skipped_empty: int
    skipped_pred_nonfinite: int
    skipped_pred_extreme: int
    skipped_loss_nonfinite: int
    skipped_grad_nonfinite: int
    skipped_assigned_nonfinite: int


@dataclass
class UnassignedEvalStats:
    loss: float
    mae: float
    matches: int
    skipped_empty: int
    skipped_pred_nonfinite: int
    skipped_pred_extreme: int
    skipped_loss_nonfinite: int


def _log_csv(path: Path, header: Sequence[str], row: Sequence[object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def _load_entries(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Entries file not found: {path}")
    with path.open("rb") as handle:
        payload = json.load(handle) if path.suffix == ".json" else None
    if payload is not None:
        raise TypeError("Entries must be pickle list[dict], JSON entries are unsupported.")

    import pickle

    with path.open("rb") as handle:
        entries = pickle.load(handle)  # noqa: S301
    if not isinstance(entries, list):
        raise TypeError(f"Entries payload must be list[dict], got {type(entries)}")
    return entries


def _load_splits(path: Path) -> tuple[list[int], list[int], list[int]]:
    if not path.exists():
        raise FileNotFoundError(f"Splits file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Splits payload must be dict, got {type(payload)}")
    splits = payload.get("splits", payload)
    if not isinstance(splits, dict):
        raise TypeError(f"'splits' must be dict, got {type(splits)}")

    train_raw = splits.get("train")
    val_raw = splits.get("val")
    test_raw = splits.get("test")
    if train_raw is None and "train_order" in payload:
        train_raw = payload["train_order"]
    if train_raw is None or val_raw is None or test_raw is None:
        raise KeyError("Splits must contain train/val/test lists.")

    return [int(v) for v in train_raw], [int(v) for v in val_raw], [int(v) for v in test_raw]


def _validate_indices(indices: Sequence[int], n_entries: int, split_name: str) -> None:
    for idx in indices:
        if idx < 0 or idx >= n_entries:
            raise IndexError(
                f"Split index out of bounds in {split_name}: {idx} (entries={n_entries})"
            )


def _parse_solvent_csv(raw: str) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for token in str(raw).split(","):
        normalized = normalize_solvent_name(token)
        if not normalized:
            continue
        key = normalize_solvent_key(normalized)
        if key == "unknown" and token.strip() == "":
            continue
        out.append(normalized)
    return out


def _resolve_metadata_path(raw_path: str, metadata_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        if path.exists():
            return path.resolve()
        raw = str(path)
        candidate_roots: list[Path] = [Path.cwd().resolve()]
        candidate_roots.extend(parent.resolve() for parent in metadata_path.resolve().parents)
        seen: set[Path] = set()
        for root in candidate_roots:
            if root in seen:
                continue
            seen.add(root)
            marker = f"/{root.name}/"
            if marker not in raw:
                continue
            suffix = raw.split(marker, 1)[1]
            candidate = (root / suffix).resolve()
            if candidate.exists():
                return candidate
        return path

    from_meta = (metadata_path.parent / path).resolve()
    if from_meta.exists():
        return from_meta
    return (Path.cwd() / path).resolve()


def _infer_split_and_shard_index(shard_path: Path) -> tuple[str, int]:
    split = shard_path.parent.name.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Could not infer split from path: {shard_path}")
    match = _ENSEMBLE_SHARD_RE.search(shard_path.name)
    if not match:
        raise ValueError(f"Could not infer shard index from name: {shard_path.name}")
    return split, int(match.group(1))


class EnsembleSolventResolver:
    def __init__(self, ensemble_root: str) -> None:
        self.ensemble_root = Path(ensemble_root).resolve()
        metadata_path = self.ensemble_root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing ensemble metadata: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        source_root_raw = metadata.get("source_shards_root")
        selected = metadata.get("source_split_selected_shards")
        if not source_root_raw or not isinstance(selected, dict):
            raise ValueError(
                "Ensemble metadata missing source_shards_root and/or source_split_selected_shards."
            )

        self.source_root = _resolve_metadata_path(str(source_root_raw), metadata_path)
        self.selected = {
            str(split).lower(): [str(name) for name in names]
            for split, names in selected.items()
            if isinstance(names, list)
        }
        self._cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def _load_source_arrays(self, split: str, source_name: str) -> tuple[np.ndarray, np.ndarray]:
        key = (split, source_name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        source_path = self.source_root / split / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Source shard not found for solvent resolution: {source_path}")

        with np.load(source_path, allow_pickle=True) as data:
            if "solvent" in data.files:
                solvents = np.array([normalize_solvent_name(v) for v in data["solvent"].tolist()], dtype=object)
            else:
                solvents = np.array(["unknown"] * int(data["n_node"].shape[0]), dtype=object)

            if "smi" in data.files:
                smiles = np.array([str(v) for v in data["smi"].tolist()], dtype=object)
            elif "smiles" in data.files:
                smiles = np.array([str(v) for v in data["smiles"].tolist()], dtype=object)
            else:
                smiles = np.array([""] * int(solvents.shape[0]), dtype=object)

        self._cache[key] = (solvents, smiles)
        return solvents, smiles

    def row_solvents_for_ensemble_shard(self, shard_path: Path) -> np.ndarray:
        split, shard_idx = _infer_split_and_shard_index(shard_path)
        split_selected = self.selected.get(split)
        if split_selected is None or shard_idx >= len(split_selected):
            raise IndexError(f"source_split_selected_shards missing split={split} shard_idx={shard_idx}")

        source_name = split_selected[shard_idx]
        source_solvents, source_smiles = self._load_source_arrays(split, source_name)

        with np.load(shard_path, allow_pickle=True) as data:
            if "row_id" not in data.files:
                raise KeyError(f"Ensemble shard missing row_id: {shard_path}")
            row_id = data["row_id"].astype(np.int64)
            ensemble_smiles = (
                np.array([str(v) for v in data["smiles"].tolist()], dtype=object)
                if "smiles" in data.files
                else np.array([""] * int(row_id.shape[0]), dtype=object)
            )

        if row_id.size == 0:
            return np.array([], dtype=object)

        if int(row_id.min()) >= 0 and int(row_id.max()) < int(source_solvents.shape[0]):
            return source_solvents[row_id]

        smile_to_indices: dict[str, list[int]] = {}
        for idx, smi in enumerate(source_smiles.tolist()):
            smile_to_indices.setdefault(smi, []).append(int(idx))

        out = np.empty((int(ensemble_smiles.shape[0]),), dtype=object)
        used_per_smiles: dict[str, int] = {}
        for idx, smi in enumerate(ensemble_smiles.tolist()):
            indices = smile_to_indices.get(smi)
            if not indices:
                out[idx] = "unknown"
                continue
            offset = int(used_per_smiles.get(smi, 0))
            if offset >= len(indices):
                offset = len(indices) - 1
            src_idx = int(indices[offset])
            used_per_smiles[smi] = int(offset + 1)
            out[idx] = source_solvents[src_idx]
        return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_shard_paths(shards_dir: str, pattern: str) -> list[str]:
    if not shards_dir:
        return []
    return sorted(glob.glob(os.path.join(shards_dir, pattern)))


def split_shards(
    shard_paths: list[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    paths = list(shard_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    n_total = len(paths)
    n_val = int(round(n_total * val_fraction))
    n_test = int(round(n_total * test_fraction))
    train_paths = paths[: n_total - n_val - n_test]
    val_paths = paths[n_total - n_val - n_test : n_total - n_test]
    test_paths = paths[n_total - n_test :]
    return train_paths, val_paths, test_paths


def resolve_shard_splits(args) -> tuple[list[str], list[str], list[str]]:
    use_ensembles = bool(args.ensemble_shards_dir)
    shards_dir = args.ensemble_shards_dir if use_ensembles else args.shards_dir
    pattern = args.ensemble_shard_glob if use_ensembles else args.shard_glob
    if not pattern:
        if use_ensembles:
            pattern = "nmrexp_ensemble_shard_*.npz"
        else:
            pattern = f"nmrexp_{args.target}_{args.graph_tag}_shard_*.npz"

    if not shards_dir:
        raise ValueError("No shard directory set. Provide --shards-dir or --ensemble-shards-dir.")

    train_dir = os.path.join(shards_dir, "train")
    if os.path.isdir(train_dir):
        train_paths = get_shard_paths(train_dir, pattern)
        val_paths = get_shard_paths(os.path.join(shards_dir, "val"), pattern)
        test_paths = get_shard_paths(os.path.join(shards_dir, "test"), pattern)
        if not train_paths:
            raise RuntimeError(f"No training shards found in {train_dir}")
        return train_paths, val_paths, test_paths

    shard_paths = get_shard_paths(shards_dir, pattern)
    if not shard_paths:
        raise RuntimeError(f"No shard files found in {shards_dir} matching {pattern}")
    return split_shards(shard_paths, float(args.val_fraction), float(args.test_fraction), int(args.seed))


def _ensemble_shards_have_solvent(shard_paths: Sequence[str]) -> bool:
    if not shard_paths:
        return False
    probe = Path(shard_paths[0])
    if not probe.exists():
        raise FileNotFoundError(f"Ensemble shard not found: {probe}")
    with np.load(probe, allow_pickle=False) as data:
        return "solvent" in data.files


def load_metadata(shards_dir: str) -> Optional[dict]:
    metadata_path = Path(shards_dir) / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def _limit_shard_paths(paths: list[str], max_shards: int, *, label: str) -> list[str]:
    if int(max_shards) <= 0 or len(paths) <= int(max_shards):
        return paths
    limited = paths[: int(max_shards)]
    print(f"subset_shards {label} using_first={len(limited)}/{len(paths)}")
    return limited


def _valid_integration_array(
    integrations: Sequence[float | None] | None,
    *,
    expected_len: int,
) -> np.ndarray | None:
    if integrations is None or int(expected_len) <= 0:
        return None
    if len(integrations) != int(expected_len):
        return None

    values = np.empty((int(expected_len),), dtype=np.float64)
    for idx, raw in enumerate(integrations):
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value) or value <= 0:
            return None
        values[idx] = value
    return values


def _integration_repeats_from_values(
    integration_values: np.ndarray,
    *,
    target_total: int,
) -> np.ndarray | None:
    n_obs = int(integration_values.shape[0])
    target_total = int(target_total)
    if n_obs <= 0 or target_total <= 0:
        return None

    total = float(np.sum(integration_values))
    if not np.isfinite(total) or total <= 0:
        return None

    scaled = integration_values * (float(target_total) / total)
    repeats = np.floor(scaled).astype(np.int64, copy=False)
    frac = scaled - np.floor(scaled)

    if n_obs <= target_total:
        repeats = np.maximum(repeats, 1)
        min_allowed = 1
    else:
        min_allowed = 0

    current = int(repeats.sum())
    if current < target_total:
        need = target_total - current
        order = np.argsort(-frac)
        for pos in range(need):
            repeats[int(order[pos % n_obs])] += 1
    elif current > target_total:
        remove = current - target_total
        order = np.argsort(frac)
        idx = 0
        max_iters = n_obs * 8
        while remove > 0 and idx < max_iters:
            j = int(order[idx % n_obs])
            if repeats[j] > min_allowed:
                repeats[j] -= 1
                remove -= 1
            idx += 1
        if remove > 0:
            for j in order:
                if remove <= 0:
                    break
                can_take = int(max(repeats[int(j)], 0))
                if can_take <= 0:
                    continue
                take = min(remove, can_take)
                repeats[int(j)] -= take
                remove -= take

    repeats = np.maximum(repeats, 0)
    if int(repeats.sum()) <= 0:
        return None
    return repeats


def _expand_obs_peaks_by_integration(
    obs_raw: torch.Tensor,
    integrations: Sequence[float | None] | None,
    *,
    target_total: int,
    precomputed: tuple[int, np.ndarray] | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    n_obs = int(obs_raw.shape[0])
    peak_index = np.arange(n_obs, dtype=np.int64)

    if precomputed is not None:
        precomputed_target_total, precomputed_peak_idx = precomputed
        if int(precomputed_target_total) == int(target_total):
            peak_index = np.asarray(precomputed_peak_idx, dtype=np.int64).reshape(-1)
            if (
                int(peak_index.shape[0]) > 0
                and np.all(peak_index >= 0)
                and np.all(peak_index < n_obs)
            ):
                peak_index_t = torch.as_tensor(peak_index, device=obs_raw.device, dtype=torch.long)
                expanded = obs_raw.index_select(0, peak_index_t)
                if int(expanded.shape[0]) > 0:
                    return expanded, peak_index

    integ_vals = _valid_integration_array(integrations, expected_len=n_obs)
    if integ_vals is None:
        return obs_raw, peak_index

    repeats = _integration_repeats_from_values(integ_vals, target_total=int(target_total))
    if repeats is None:
        return obs_raw, peak_index

    keep = repeats > 0
    if not np.any(keep):
        return obs_raw, peak_index

    peak_index = np.repeat(np.arange(n_obs, dtype=np.int64)[keep], repeats[keep])
    if peak_index.size == 0:
        return obs_raw, np.arange(n_obs, dtype=np.int64)

    peak_index_t = torch.as_tensor(peak_index, device=obs_raw.device, dtype=torch.long)
    expanded = obs_raw.index_select(0, peak_index_t)
    return expanded, peak_index


def _match_indices_scipy(pred_vals: np.ndarray, obs_vals: np.ndarray, dummy_cost: float) -> tuple[list[int], list[int]]:
    n_pred = int(pred_vals.shape[0])
    n_obs = int(obs_vals.shape[0])
    n = max(n_pred, n_obs)
    if n_pred <= 0 or n_obs <= 0:
        return [], []

    cost = np.full((n, n), float(dummy_cost), dtype=np.float64)
    cost[:n_pred, :n_obs] = np.abs(pred_vals.reshape(-1, 1) - obs_vals.reshape(1, -1))

    rows, cols = linear_sum_assignment(cost)
    out_rows: list[int] = []
    out_cols: list[int] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r < n_pred and c < n_obs:
            out_rows.append(int(r))
            out_cols.append(int(c))
    return out_rows, out_cols


def compute_batch_loss_hungarian(
    pred_mean_flat: torch.Tensor,
    batch_graph: dict,
    n_atoms: np.ndarray,
    n_pros: np.ndarray,
    peaks: Sequence[Sequence[float]],
    integrations: Sequence[Sequence[float] | None],
    config: UnassignedConfig,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    atom_index = batch_graph["atom_index"]
    expand_precomputed = batch_graph.get("integration_expand_precomputed")
    if not isinstance(expand_precomputed, list) or len(expand_precomputed) != len(peaks):
        expand_precomputed = [None] * len(peaks)

    total_loss = pred_mean_flat.new_tensor(0.0)
    total_mae = pred_mean_flat.new_tensor(0.0)
    total_matches = 0

    atom_offset = 0
    pred_offset = 0
    prepared: list[dict] = []

    for idx, n_pro in enumerate(n_pros.tolist()):
        n_atom = int(n_atoms[idx])
        pred_offset_start = pred_offset
        pred_chunk = pred_mean_flat[pred_offset_start : pred_offset_start + n_pro]
        pred_offset += n_pro

        atom_index_slice = atom_index[atom_offset : atom_offset + n_atom]
        atom_offset += n_atom

        if n_pro == 0 or not peaks[idx]:
            continue

        local_atom_index = atom_index_slice.copy()
        mask = local_atom_index >= 0
        if not np.any(mask):
            continue
        local_atom_index[mask] = local_atom_index[mask] - int(pred_offset_start)
        valid = mask & (local_atom_index < n_pro)
        if not np.any(valid):
            continue

        obs_base_np = np.asarray(peaks[idx], dtype=np.float64).reshape(-1)
        if obs_base_np.size <= 0:
            continue
        obs_raw = torch.as_tensor(obs_base_np, device=pred_mean_flat.device, dtype=pred_mean_flat.dtype)
        obs_to_peak_idx = np.arange(int(obs_raw.shape[0]), dtype=np.int64)

        if config.integration_matching_mode == "expanded":
            obs_raw, obs_to_peak_idx = _expand_obs_peaks_by_integration(
                obs_raw,
                integrations[idx],
                target_total=int(pred_chunk.shape[0]),
                precomputed=expand_precomputed[idx],
            )
            if int(obs_raw.shape[0]) <= 0:
                continue

        prepared.append(
            {
                "pred_raw": pred_chunk,
                "obs_raw": obs_raw,
                "obs_base_np": obs_base_np,
                "integrations": integrations[idx],
                "obs_to_peak_idx": obs_to_peak_idx,
                "pred_offset_start": int(pred_offset_start),
                "pred_np": pred_chunk.detach().float().cpu().numpy(),
                "obs_np": obs_raw.detach().float().cpu().numpy(),
            }
        )

    if not prepared:
        return None, None, 0

    if config.hungarian_solver == "dp":
        if match_indices_dp_batch is None:
            raise RuntimeError("Hungarian solver 'dp' requested but nmrexp.matching backend unavailable.")
        match_results = match_indices_dp_batch(
            [item["pred_np"] for item in prepared],
            [item["obs_np"] for item in prepared],
            dummy_cost=float(config.dummy_cost),
            n_threads=int(config.matching_workers),
        )
    else:
        match_results = [
            _match_indices_scipy(item["pred_np"], item["obs_np"], float(config.dummy_cost))
            for item in prepared
        ]

    for item, (rows, cols) in zip(prepared, match_results):
        if not rows:
            continue

        pred_match_raw = item["pred_raw"][rows]
        obs_match_raw = item["obs_raw"][cols]

        obs_match = obs_match_raw
        if config.normalize and config.std > 0:
            obs_match = (obs_match_raw - config.mean) / config.std
            pred_for_loss = (pred_match_raw - config.mean) / config.std
        else:
            pred_for_loss = pred_match_raw

        weights = None
        match_count = len(rows)
        if config.integration_matching_mode == "weighted" and item["integrations"] is not None:
            integ_vals = item["integrations"]
            if all(val is not None for val in integ_vals):
                peak_index = np.asarray(item["obs_to_peak_idx"], dtype=np.int64)
                mapped = [integ_vals[int(peak_index[c])] for c in cols]
                weights = torch.as_tensor(mapped, device=pred_mean_flat.device, dtype=pred_mean_flat.dtype)
                weight_sum = weights.sum().clamp_min(1.0)
                match_count = int(weight_sum.item())

        if weights is not None:
            weight_sum = weights.sum().clamp_min(1.0)
            loss = (torch.abs(pred_for_loss - obs_match) * weights).sum() / weight_sum
            mae = (torch.abs(pred_match_raw - obs_match_raw) * weights).sum() / weight_sum
        else:
            loss = torch.mean(torch.abs(pred_for_loss - obs_match))
            mae = torch.mean(torch.abs(pred_match_raw - obs_match_raw))

        total_loss += loss * match_count
        total_mae += mae * match_count
        total_matches += int(match_count)

    if total_matches <= 0:
        return None, None, 0
    return total_loss / total_matches, total_mae / total_matches, total_matches


def _build_target_solvent_ids(
    *,
    solvents: Sequence[str],
    n_pros: np.ndarray,
    solvent_to_id: dict[str, int],
    unknown_id: int,
    device: str,
) -> Optional[torch.Tensor]:
    out: list[int] = []
    for solvent, n_pro in zip(solvents, n_pros.tolist()):
        sid = solvent_to_id.get(normalize_solvent_key(solvent), int(unknown_id))
        for _ in range(int(n_pro)):
            out.append(int(sid))
    if not out:
        return None
    return torch.as_tensor(out, device=device, dtype=torch.long)


def _unpack_unassigned_batch(batch):
    if len(batch) == 9:
        return batch
    if len(batch) == 8:
        (
            batch_graph,
            n_atoms,
            n_pros,
            symm,
            h_counts,
            peaks,
            integrations,
            smiles,
        ) = batch
        solvents = ["unknown"] * len(smiles)
        return batch_graph, n_atoms, n_pros, symm, h_counts, peaks, integrations, smiles, solvents
    raise RuntimeError(f"Unexpected unassigned batch tuple length: {len(batch)}")


def _compute_unassigned_batch_loss(
    *,
    model: torch.nn.Module,
    batch_graph: dict,
    n_atoms: np.ndarray,
    n_pros: np.ndarray,
    peaks: Sequence[Sequence[float]],
    integrations: Sequence[Sequence[float] | None],
    solvents: Sequence[str],
    config: UnassignedConfig,
    solvent_to_id: dict[str, int],
    unknown_solvent_id: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, str]:
    batch_torch = _to_torch_batch(batch_graph, config.device, config.dtype)
    solvent_ids = _build_target_solvent_ids(
        solvents=solvents,
        n_pros=n_pros,
        solvent_to_id=solvent_to_id,
        unknown_id=unknown_solvent_id,
        device=str(config.device),
    )
    pred = model(batch_torch, solvent_ids=solvent_ids)

    if pred.ndim == 1:
        pred_mean = pred.view(-1)
    elif pred.ndim == 2 and int(pred.shape[1]) == 1:
        pred_mean = pred[:, 0].contiguous().view(-1)
    else:
        raise RuntimeError(f"Joint trainer expects deterministic output shape [N,1], got {tuple(pred.shape)}")

    if not torch.isfinite(pred_mean).all():
        return None, None, 0, "pred_nonfinite"

    if config.max_abs_pred > 0 and float(torch.max(torch.abs(pred_mean)).item()) > float(config.max_abs_pred):
        return None, None, 0, "pred_extreme"

    loss, mae, matches = compute_batch_loss_hungarian(
        pred_mean,
        batch_graph,
        n_atoms,
        n_pros,
        peaks,
        integrations,
        config,
    )
    if loss is None or matches <= 0:
        return None, None, 0, "empty"

    if not torch.isfinite(loss) or (mae is not None and not torch.isfinite(mae)):
        return None, None, 0, "loss_nonfinite"

    return loss, mae, int(matches), ""


def _run_unassigned_loader(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    config: UnassignedConfig,
    solvent_to_id: dict[str, int],
    unknown_solvent_id: int,
) -> UnassignedEvalStats:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_matches = 0
    skipped_empty = 0
    skipped_pred_nonfinite = 0
    skipped_pred_extreme = 0
    skipped_loss_nonfinite = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                skipped_empty += 1
                continue

            (
                batch_graph,
                n_atoms,
                n_pros,
                _symm,
                _h_counts,
                peaks,
                integrations,
                _smiles,
                solvents,
            ) = _unpack_unassigned_batch(batch)

            amp_enabled = bool(config.amp and str(config.device).startswith("cuda"))
            amp_ctx = torch.cuda.amp.autocast(dtype=config.amp_dtype) if amp_enabled else nullcontext()
            with amp_ctx:
                loss, mae, matches, reason = _compute_unassigned_batch_loss(
                    model=model,
                    batch_graph=batch_graph,
                    n_atoms=n_atoms,
                    n_pros=n_pros,
                    peaks=peaks,
                    integrations=integrations,
                    solvents=solvents,
                    config=config,
                    solvent_to_id=solvent_to_id,
                    unknown_solvent_id=unknown_solvent_id,
                )

            if matches <= 0 or loss is None or mae is None:
                if reason == "pred_nonfinite":
                    skipped_pred_nonfinite += 1
                elif reason == "pred_extreme":
                    skipped_pred_extreme += 1
                elif reason == "loss_nonfinite":
                    skipped_loss_nonfinite += 1
                else:
                    skipped_empty += 1
                continue

            total_loss += float(loss.item()) * int(matches)
            total_mae += float(mae.item()) * int(matches)
            total_matches += int(matches)

    if total_matches > 0:
        loss = total_loss / total_matches
        mae = total_mae / total_matches
    else:
        loss = float("inf")
        mae = float("inf")

    return UnassignedEvalStats(
        loss=float(loss),
        mae=float(mae),
        matches=int(total_matches),
        skipped_empty=int(skipped_empty),
        skipped_pred_nonfinite=int(skipped_pred_nonfinite),
        skipped_pred_extreme=int(skipped_pred_extreme),
        skipped_loss_nonfinite=int(skipped_loss_nonfinite),
    )


def _build_unassigned_dataset(
    *,
    args: argparse.Namespace,
    shard_path: Path,
    epoch: int,
    train_mode: bool,
    shard_offset: int,
    row_solvents: Sequence[object] | None,
) -> torch.utils.data.Dataset:
    if args.ensemble_shards_dir:
        dataset = NMRexpEnsembleShardDataset(
            shard_path,
            target=args.target,
            sampling=args.ensemble_conformer_sampling if train_mode else args.ensemble_eval_sampling,
            temperature=float(args.ensemble_temperature),
            boltzmann_alpha=float(args.ensemble_boltzmann_alpha),
            n_neighbors=int(args.n_neighbors),
            cutoff=float(args.cutoff),
            seed=int(args.seed + shard_offset),
            atom_token_map=args.atom_tokenizer_map_dict,
            atom_token_unk=int(args.atom_tokenizer_unk),
            disable_fragment_filter=bool(args.disable_fragment_filter),
            min_conformer_distance=float(args.min_conformer_distance),
            drop_smiles=args.drop_smiles_set,
            row_solvents=row_solvents,
            solvent_filter=args.target_solvents_filter,
        )
        dataset.set_epoch(int(epoch))
        return dataset

    return NMRexpShardDataset(
        shard_path,
        solvent_filter=args.target_solvents_filter,
    )


def _build_unassigned_loader(
    *,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "collate_fn": collate_unassigned,
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if int(prefetch_factor) > 0:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def _build_assigned_loader(
    *,
    dataset: AssignedPaperDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "collate_fn": collate_assigned,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if int(prefetch_factor) > 0:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def _next_assigned_batch(assigned_iter, assigned_loader: DataLoader):
    try:
        batch = next(assigned_iter)
    except StopIteration:
        assigned_iter = iter(assigned_loader)
        batch = next(assigned_iter)
    return batch, assigned_iter


def _compute_assigned_loss(
    *,
    model: torch.nn.Module,
    batch_graph: dict,
    batch_targets_ppm: np.ndarray,
    config: AssignedConfig,
    assigned_solvent_id: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    batch_torch = _to_torch_batch(batch_graph, config.device, config.dtype)
    targets_ppm = torch.as_tensor(batch_targets_ppm, device=config.device, dtype=config.dtype).view(-1)

    solvent_ids = None
    if assigned_solvent_id is not None:
        solvent_ids = torch.full(
            (int(targets_ppm.shape[0]),),
            int(assigned_solvent_id),
            device=config.device,
            dtype=torch.long,
        )

    pred = model(batch_torch, solvent_ids=solvent_ids)
    if pred.ndim == 2 and int(pred.shape[1]) == 1:
        pred_mean = pred[:, 0].contiguous().view(-1)
    elif pred.ndim == 1:
        pred_mean = pred.view(-1)
    else:
        raise RuntimeError(f"Assigned loss expects deterministic output [N,1], got {tuple(pred.shape)}")

    if pred_mean.shape != targets_ppm.shape:
        raise RuntimeError(
            f"Assigned shape mismatch: pred={tuple(pred_mean.shape)} targets={tuple(targets_ppm.shape)}"
        )

    if config.normalize_targets and config.norm_std_ppm > 0:
        pred_loss = (pred_mean - config.norm_mean_ppm) / config.norm_std_ppm
        target_loss = (targets_ppm - config.norm_mean_ppm) / config.norm_std_ppm
    else:
        pred_loss = pred_mean
        target_loss = targets_ppm

    loss = torch.mean(torch.abs(pred_loss - target_loss))
    mae_ppm = torch.mean(torch.abs(pred_mean - targets_ppm))
    return loss, mae_ppm, int(targets_ppm.shape[0])


def _evaluate_assigned_loader(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    config: AssignedConfig,
    assigned_solvent_id: Optional[int],
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_graph, batch_targets_ppm, _smiles in loader:
            loss, mae, count = _compute_assigned_loss(
                model=model,
                batch_graph=batch_graph,
                batch_targets_ppm=batch_targets_ppm,
                config=config,
                assigned_solvent_id=assigned_solvent_id,
            )
            if count <= 0:
                continue
            if not torch.isfinite(loss) or not torch.isfinite(mae):
                continue
            total_loss += float(loss.item()) * int(count)
            total_mae += float(mae.item()) * int(count)
            total_count += int(count)

    if total_count <= 0:
        return float("inf"), float("inf")
    return total_loss / total_count, total_mae / total_count


def _save_joint_checkpoint(
    *,
    checkpoint_path: Path,
    epoch: int,
    model: PaiNNModel,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    best_monitor: float,
    best_epoch: int,
    patience_counter: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_monitor": float(best_monitor),
        "best_epoch": int(best_epoch),
        "patience_counter": int(patience_counter),
    }
    torch.save(payload, checkpoint_path)


def _load_joint_checkpoint(
    *,
    checkpoint_path: Path,
    model: PaiNNModel,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: str,
) -> tuple[int, float, int, int]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be dict, got {type(payload)}")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    return (
        int(payload.get("epoch", 0)),
        float(payload.get("best_monitor", float("inf"))),
        int(payload.get("best_epoch", 0)),
        int(payload.get("patience_counter", 0)),
    )


def _load_warm_start_state(path: Path, device: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Warm-start weights not found: {path}")
    payload = torch.load(str(path), map_location=device)
    if isinstance(payload, dict) and "model_state" in payload and isinstance(payload["model_state"], dict):
        state = payload["model_state"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise TypeError(f"Warm-start state must be dict, got {type(state)}")
    return state


def _apply_warm_start(model: PaiNNModel, state: dict, strict: bool) -> None:
    if strict:
        model.load_state_dict(state, strict=True)
        return

    model_state = model.state_dict()
    filtered = {}
    for key, value in state.items():
        if key not in model_state:
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            continue
        filtered[key] = value
    if not filtered:
        raise RuntimeError("Nonstrict warm-start found no compatible parameters.")
    result = model.load_state_dict(filtered, strict=False)
    print(
        "warm_start_nonstrict "
        f"loaded={len(filtered)} missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"
    )


def _resolve_monitor_weights(args: argparse.Namespace) -> tuple[float, float]:
    monitor_target_weight = (
        float(args.monitor_target_weight)
        if args.monitor_target_weight is not None
        else float(args.target_loss_weight)
    )
    monitor_assigned_weight = (
        float(args.monitor_assigned_weight)
        if args.monitor_assigned_weight is not None
        else (float(args.assigned_loss_weight) if int(args.assigned_step_ratio) > 0 else 0.0)
    )
    return monitor_target_weight, monitor_assigned_weight


def _train_joint_epoch(
    *,
    model: torch.nn.Module,
    args: argparse.Namespace,
    epoch: int,
    train_paths: list[str],
    resolver: Optional[EnsembleSolventResolver],
    unassigned_cfg: UnassignedConfig,
    assigned_cfg: AssignedConfig,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    assigned_loader: DataLoader,
    assigned_solvent_id: Optional[int],
) -> JointStats:
    model.train()

    paths = list(train_paths)
    if bool(args.shuffle_shards):
        random.shuffle(paths)

    assigned_iter = iter(assigned_loader)

    total_t_loss = 0.0
    total_t_mae = 0.0
    total_t_matches = 0
    total_a_loss = 0.0
    total_a_mae = 0.0
    total_a_atoms = 0
    total_joint_loss = 0.0
    n_steps = 0

    skipped_empty = 0
    skipped_pred_nonfinite = 0
    skipped_pred_extreme = 0
    skipped_loss_nonfinite = 0
    skipped_grad_nonfinite = 0
    skipped_assigned_nonfinite = 0

    for shard_idx, shard_path_str in enumerate(paths):
        shard_path = Path(shard_path_str)
        row_solvents = None
        if resolver is not None:
            row_solvents = resolver.row_solvents_for_ensemble_shard(shard_path)

        dataset = _build_unassigned_dataset(
            args=args,
            shard_path=shard_path,
            epoch=epoch,
            train_mode=True,
            shard_offset=shard_idx,
            row_solvents=row_solvents,
        )
        loader = _build_unassigned_loader(
            dataset=dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor),
        )

        shard_steps = 0
        shard_start = time.time()

        for batch in loader:
            if batch is None:
                skipped_empty += 1
                continue

            (
                batch_graph,
                n_atoms,
                n_pros,
                _symm,
                _h_counts,
                peaks,
                integrations,
                _smiles,
                solvents,
            ) = _unpack_unassigned_batch(batch)

            optimizer.zero_grad(set_to_none=True)

            amp_enabled = bool(unassigned_cfg.amp and str(unassigned_cfg.device).startswith("cuda"))
            amp_ctx = torch.cuda.amp.autocast(dtype=unassigned_cfg.amp_dtype) if amp_enabled else nullcontext()
            with amp_ctx:
                t_loss, t_mae, t_matches, t_reason = _compute_unassigned_batch_loss(
                    model=model,
                    batch_graph=batch_graph,
                    n_atoms=n_atoms,
                    n_pros=n_pros,
                    peaks=peaks,
                    integrations=integrations,
                    solvents=solvents,
                    config=unassigned_cfg,
                    solvent_to_id=args.solvent_to_id,
                    unknown_solvent_id=args.solvent_unknown_id,
                )
                if t_loss is None or t_matches <= 0:
                    if t_reason == "pred_nonfinite":
                        skipped_pred_nonfinite += 1
                    elif t_reason == "pred_extreme":
                        skipped_pred_extreme += 1
                    elif t_reason == "loss_nonfinite":
                        skipped_loss_nonfinite += 1
                    else:
                        skipped_empty += 1
                    continue

                joint_loss = float(args.target_loss_weight) * t_loss

                step_a_loss = 0.0
                step_a_mae = 0.0
                step_a_atoms = 0
                step_a_loss_numer = None

                if float(args.assigned_loss_weight) > 0 and int(args.assigned_step_ratio) > 0:
                    for _ in range(int(args.assigned_step_ratio)):
                        assigned_batch, assigned_iter = _next_assigned_batch(assigned_iter, assigned_loader)
                        batch_graph_a, batch_targets_ppm, _ = assigned_batch
                        a_loss, a_mae, a_count = _compute_assigned_loss(
                            model=model,
                            batch_graph=batch_graph_a,
                            batch_targets_ppm=batch_targets_ppm,
                            config=assigned_cfg,
                            assigned_solvent_id=assigned_solvent_id,
                        )
                        if a_count <= 0:
                            continue
                        if not torch.isfinite(a_loss) or not torch.isfinite(a_mae):
                            skipped_assigned_nonfinite += 1
                            continue
                        weighted_a_loss = a_loss * float(int(a_count))
                        if step_a_loss_numer is None:
                            step_a_loss_numer = weighted_a_loss
                        else:
                            step_a_loss_numer = step_a_loss_numer + weighted_a_loss
                        step_a_loss += float(a_loss.item()) * int(a_count)
                        step_a_mae += float(a_mae.item()) * int(a_count)
                        step_a_atoms += int(a_count)

                if step_a_loss_numer is not None and step_a_atoms > 0:
                    joint_loss = joint_loss + float(args.assigned_loss_weight) * (
                        step_a_loss_numer / float(step_a_atoms)
                    )

            if not torch.isfinite(joint_loss):
                skipped_loss_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler is not None:
                scaler.scale(joint_loss).backward()
            else:
                joint_loss.backward()

            if float(unassigned_cfg.grad_clip) > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(unassigned_cfg.grad_clip))
                if not torch.isfinite(grad_norm):
                    skipped_grad_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.update()
                    continue

            grads_finite = True
            for param in model.parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    grads_finite = False
                    break
            if not grads_finite:
                skipped_grad_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
                continue

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            total_t_loss += float(t_loss.item()) * int(t_matches)
            total_t_mae += float(t_mae.item()) * int(t_matches)
            total_t_matches += int(t_matches)

            total_a_loss += float(step_a_loss)
            total_a_mae += float(step_a_mae)
            total_a_atoms += int(step_a_atoms)

            total_joint_loss += float(joint_loss.item())
            n_steps += 1
            shard_steps += 1

        if bool(args.log_shard_progress):
            print(
                f"epoch={epoch} train_shard={shard_idx+1}/{len(paths)} "
                f"steps={shard_steps} sec={time.time() - shard_start:.1f}"
            )

    avg_t_loss = total_t_loss / total_t_matches if total_t_matches > 0 else float("inf")
    avg_t_mae = total_t_mae / total_t_matches if total_t_matches > 0 else float("inf")
    avg_a_loss = total_a_loss / total_a_atoms if total_a_atoms > 0 else float("inf")
    avg_a_mae = total_a_mae / total_a_atoms if total_a_atoms > 0 else float("inf")
    avg_joint = total_joint_loss / n_steps if n_steps > 0 else float("inf")

    return JointStats(
        train_target_loss=float(avg_t_loss),
        train_target_mae=float(avg_t_mae),
        train_assigned_loss=float(avg_a_loss),
        train_assigned_mae=float(avg_a_mae),
        train_total_loss=float(avg_joint),
        train_target_matches=int(total_t_matches),
        train_assigned_atoms=int(total_a_atoms),
        train_steps=int(n_steps),
        skipped_empty=int(skipped_empty),
        skipped_pred_nonfinite=int(skipped_pred_nonfinite),
        skipped_pred_extreme=int(skipped_pred_extreme),
        skipped_loss_nonfinite=int(skipped_loss_nonfinite),
        skipped_grad_nonfinite=int(skipped_grad_nonfinite),
        skipped_assigned_nonfinite=int(skipped_assigned_nonfinite),
    )


def _evaluate_unassigned_splits(
    *,
    model: torch.nn.Module,
    args: argparse.Namespace,
    epoch: int,
    paths: list[str],
    resolver: Optional[EnsembleSolventResolver],
    config: UnassignedConfig,
    batch_size: int,
) -> UnassignedEvalStats:
    if not paths:
        return UnassignedEvalStats(
            loss=float("inf"),
            mae=float("inf"),
            matches=0,
            skipped_empty=0,
            skipped_pred_nonfinite=0,
            skipped_pred_extreme=0,
            skipped_loss_nonfinite=0,
        )

    total_loss = 0.0
    total_mae = 0.0
    total_matches = 0
    skipped_empty = 0
    skipped_pred_nonfinite = 0
    skipped_pred_extreme = 0
    skipped_loss_nonfinite = 0

    for shard_idx, shard_path_str in enumerate(paths):
        shard_path = Path(shard_path_str)
        row_solvents = None
        if resolver is not None:
            row_solvents = resolver.row_solvents_for_ensemble_shard(shard_path)

        dataset = _build_unassigned_dataset(
            args=args,
            shard_path=shard_path,
            epoch=epoch,
            train_mode=False,
            shard_offset=shard_idx,
            row_solvents=row_solvents,
        )
        loader = _build_unassigned_loader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor),
        )

        stats = _run_unassigned_loader(
            model=model,
            loader=loader,
            config=config,
            solvent_to_id=args.solvent_to_id,
            unknown_solvent_id=args.solvent_unknown_id,
        )

        skipped_empty += int(stats.skipped_empty)
        skipped_pred_nonfinite += int(stats.skipped_pred_nonfinite)
        skipped_pred_extreme += int(stats.skipped_pred_extreme)
        skipped_loss_nonfinite += int(stats.skipped_loss_nonfinite)

        if stats.matches > 0 and np.isfinite(stats.loss) and np.isfinite(stats.mae):
            total_loss += float(stats.loss) * int(stats.matches)
            total_mae += float(stats.mae) * int(stats.matches)
            total_matches += int(stats.matches)

    if total_matches > 0:
        loss = total_loss / total_matches
        mae = total_mae / total_matches
    else:
        loss = float("inf")
        mae = float("inf")

    return UnassignedEvalStats(
        loss=float(loss),
        mae=float(mae),
        matches=int(total_matches),
        skipped_empty=int(skipped_empty),
        skipped_pred_nonfinite=int(skipped_pred_nonfinite),
        skipped_pred_extreme=int(skipped_pred_extreme),
        skipped_loss_nonfinite=int(skipped_loss_nonfinite),
    )


def train_joint(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint_path if args.checkpoint_path is not None else output_dir / "checkpoint.pt"
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    best_model_path = output_dir / "best_model.pt"
    log_path = output_dir / "log_painn_joint.csv"
    summary_path = output_dir / "summary_metrics.json"

    train_paths, val_paths, test_paths = resolve_shard_splits(args)
    train_paths = _limit_shard_paths(train_paths, int(args.target_max_train_shards), label="target_train")
    val_paths = _limit_shard_paths(val_paths, int(args.target_max_val_shards), label="target_val")
    test_paths = _limit_shard_paths(test_paths, int(args.target_max_test_shards), label="target_test")

    if not train_paths:
        raise RuntimeError("No target unassigned training shards found.")
    if not val_paths:
        raise RuntimeError("No target unassigned validation shards found.")

    metadata_root = str(args.ensemble_shards_dir or args.shards_dir)
    metadata = load_metadata(metadata_root)
    if metadata:
        if args.mean == 0.0 and metadata.get("mean") is not None:
            args.mean = float(metadata["mean"])
        if args.std == 1.0 and metadata.get("std") is not None:
            args.std = float(metadata["std"])
        if args.ensemble_shards_dir:
            if metadata.get("cutoff") is not None and args.cutoff == 5.0:
                args.cutoff = float(metadata["cutoff"])
            if metadata.get("n_neighbors") is not None and args.n_neighbors == 20:
                args.n_neighbors = int(metadata["n_neighbors"])

    if args.no_atom_tokenizer_map:
        args.atom_tokenizer_map_dict = None
        args.atom_tokenizer_unk = 1
    else:
        token_map, unk = load_atom_tokenizer_map(args.atom_tokenizer_preprocessor)
        args.atom_tokenizer_map_dict = token_map
        args.atom_tokenizer_unk = unk

    args.drop_smiles_set = set()
    if args.drop_smiles_file:
        if not args.drop_smiles_file.exists():
            raise FileNotFoundError(f"Drop-smiles file not found: {args.drop_smiles_file}")
        with args.drop_smiles_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                args.drop_smiles_set.add(s)

    args.target_solvents_list = _parse_solvent_csv(args.target_solvents)
    args.solvent_vocab_list = _parse_solvent_csv(args.solvent_vocab)
    args.target_solvents_filter = {normalize_solvent_key(s) for s in args.target_solvents_list}

    solvent_metadata_required = bool(args.target_solvents_filter) or bool(args.solvent_vocab_list)
    if not solvent_metadata_required:
        solvent_metadata_required = bool(args.solvent_use_bias) or int(args.solvent_adapter_hidden_dim) > 0
    if not solvent_metadata_required and str(args.trainable_mode) in {
        "solvent_only",
        "solvent_plus_head",
        "solvent_plus_lastblock",
    }:
        solvent_metadata_required = True

    resolver = None
    if args.ensemble_shards_dir and solvent_metadata_required:
        solvent_probe_paths = list(train_paths) + list(val_paths) + list(test_paths)
        if not _ensemble_shards_have_solvent(solvent_probe_paths):
            raise RuntimeError(
                "Solvent-aware training requires ensemble shards that include a 'solvent' field. "
                "Rebuild ensemble shards using this repo's build_conformer_ensembles.py."
            )

    entries = _load_entries(args.exp22k_entries_path)
    train_idx, val_idx, test_idx = _load_splits(args.exp22k_splits_path)

    if int(args.exp22k_train_size) > 0:
        train_idx = train_idx[: int(args.exp22k_train_size)]
    if int(args.exp22k_max_val_mols) > 0:
        val_idx = val_idx[: int(args.exp22k_max_val_mols)]
    if int(args.exp22k_max_test_mols) > 0:
        test_idx = test_idx[: int(args.exp22k_max_test_mols)]

    if not train_idx or not val_idx or not test_idx:
        raise RuntimeError(
            "Assigned splits must be non-empty after truncation: "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )

    _validate_indices(train_idx, len(entries), "train")
    _validate_indices(val_idx, len(entries), "val")
    _validate_indices(test_idx, len(entries), "test")

    assigned_atom_map, assigned_atom_unk = load_atom_tokenizer_map(args.atom_tokenizer_preprocessor)

    train_set_a = AssignedPaperDataset(
        entries=entries,
        indices=train_idx,
        target=args.target,
        n_neighbors=int(args.n_neighbors),
        cutoff=float(args.cutoff),
        atom_token_map=assigned_atom_map,
        atom_token_unk=assigned_atom_unk,
    )
    val_set_a = AssignedPaperDataset(
        entries=entries,
        indices=val_idx,
        target=args.target,
        n_neighbors=int(args.n_neighbors),
        cutoff=float(args.cutoff),
        atom_token_map=assigned_atom_map,
        atom_token_unk=assigned_atom_unk,
    )
    test_set_a = AssignedPaperDataset(
        entries=entries,
        indices=test_idx,
        target=args.target,
        n_neighbors=int(args.n_neighbors),
        cutoff=float(args.cutoff),
        atom_token_map=assigned_atom_map,
        atom_token_unk=assigned_atom_unk,
    )

    train_mean_a = float(np.mean(train_set_a.all_targets_ppm))
    train_std_a = float(np.std(train_set_a.all_targets_ppm))
    if not np.isfinite(train_mean_a) or not np.isfinite(train_std_a):
        raise RuntimeError("Assigned train target mean/std are non-finite.")

    assigned_mean = float(args.assigned_mean) if args.assigned_mean is not None else float(train_mean_a)
    assigned_std = float(args.assigned_std) if args.assigned_std is not None else float(train_std_a)
    if bool(args.assigned_normalize_targets) and assigned_std <= 0:
        raise RuntimeError("Assigned normalization enabled but std <= 0.")

    device = str(args.device)
    use_cuda = device.startswith("cuda")
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        if bool(args.tf32):
            torch.set_float32_matmul_precision("high")

    amp_dtype = torch.bfloat16 if str(args.amp_dtype).lower() == "bf16" else torch.float16
    if use_cuda and bool(args.amp) and amp_dtype == torch.bfloat16:
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if not bf16_ok:
            amp_dtype = torch.float16

    unassigned_cfg = UnassignedConfig(
        mean=float(args.mean),
        std=float(args.std),
        normalize=bool(args.normalize),
        dummy_cost=float(args.dummy_cost),
        matching_workers=max(1, int(args.matching_workers)),
        hungarian_solver=str(args.hungarian_solver),
        integration_matching_mode=str(args.integration_matching_mode),
        max_abs_pred=float(args.max_abs_pred),
        grad_clip=float(args.grad_clip),
        amp=bool(args.amp),
        amp_dtype=amp_dtype,
        tf32=bool(args.tf32),
        device=device,
        dtype=torch.float32,
    )
    assigned_cfg = AssignedConfig(
        normalize_targets=bool(args.assigned_normalize_targets),
        norm_mean_ppm=float(assigned_mean),
        norm_std_ppm=float(assigned_std if assigned_std > 0 else 1.0),
        device=device,
        dtype=torch.float32,
    )

    scaler = None
    if use_cuda and bool(args.amp) and amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler()

    solvent_vocab_keys = ["unknown"]
    solvent_vocab_seen = {"unknown"}
    for name in args.solvent_vocab_list + args.target_solvents_list + [args.assigned_anchor_solvent]:
        key = normalize_solvent_key(name)
        if key not in solvent_vocab_seen:
            solvent_vocab_seen.add(key)
            solvent_vocab_keys.append(key)

    args.solvent_to_id = {key: idx for idx, key in enumerate(solvent_vocab_keys)}
    args.solvent_unknown_id = int(args.solvent_to_id["unknown"])
    assigned_solvent_id = args.solvent_to_id.get(
        normalize_solvent_key(args.assigned_anchor_solvent),
        args.solvent_unknown_id,
    )

    solvent_modules_enabled = bool(args.solvent_use_bias or int(args.solvent_adapter_hidden_dim) > 0)
    solvent_vocab_size = len(args.solvent_to_id) if solvent_modules_enabled else 0

    model = PaiNNModel(
        PaiNNConfig(
            input_dim=int(args.input_dim),
            units=int(args.units),
            num_radial=int(args.num_radial),
            cutoff=float(args.cutoff),
            depth=int(args.depth),
            output_dim=1,
            solvent_vocab_size=int(solvent_vocab_size),
            solvent_emb_dim=int(args.solvent_emb_dim),
            solvent_use_bias=bool(args.solvent_use_bias),
            solvent_adapter_hidden_dim=int(args.solvent_adapter_hidden_dim),
            solvent_adapter_dropout=float(args.solvent_adapter_dropout),
        )
    ).to(device, dtype=torch.float32)

    if args.init_from and args.resume:
        raise ValueError("Use either --resume or --init-from, not both.")

    if args.init_from:
        state = _load_warm_start_state(Path(args.init_from), device=device)
        _apply_warm_start(model, state=state, strict=bool(args.init_strict))
        print(f"warm_start weights={args.init_from} strict={bool(args.init_strict)}")

    if args.trainable_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
    elif args.trainable_mode in {"solvent_only", "solvent_plus_head", "solvent_plus_lastblock"}:
        if not solvent_modules_enabled:
            raise ValueError(
                f"trainable_mode={args.trainable_mode} requires solvent modules (--solvent-use-bias or adapter)."
            )
        last_block_idx = max(len(model.convs) - 1, 0)
        for name, param in model.named_parameters():
            trainable = name.startswith("solvent_")
            if args.trainable_mode in {"solvent_plus_head", "solvent_plus_lastblock"} and name.startswith("mlp."):
                trainable = True
            if args.trainable_mode == "solvent_plus_lastblock":
                if name.startswith(f"convs.{last_block_idx}.") or name.startswith(f"updates.{last_block_idx}."):
                    trainable = True
            param.requires_grad = trainable
    else:
        raise ValueError(f"Unknown trainable mode: {args.trainable_mode}")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected.")

    trainable_count = int(sum(int(p.numel()) for p in trainable_params))
    total_count = int(sum(int(p.numel()) for p in model.parameters()))

    optimizer = torch.optim.Adam(trainable_params, lr=float(args.lr))
    lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(args.reduce_lr_factor),
        patience=int(args.reduce_lr_patience),
        min_lr=float(args.min_lr),
    )

    start_epoch = 1
    best_monitor = float("inf")
    best_epoch = 0
    patience_counter = 0
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume requested but checkpoint not found: {checkpoint_path}")
        last_epoch, best_monitor, best_epoch, patience_counter = _load_joint_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        start_epoch = int(last_epoch) + 1

    monitor_target_weight, monitor_assigned_weight = _resolve_monitor_weights(args)

    assigned_train_loader = _build_assigned_loader(
        dataset=train_set_a,
        batch_size=int(args.assigned_batch_size),
        shuffle=True,
        num_workers=int(args.assigned_num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.assigned_persistent_workers),
        prefetch_factor=int(args.assigned_prefetch_factor),
    )
    assigned_val_loader = _build_assigned_loader(
        dataset=val_set_a,
        batch_size=int(args.assigned_batch_size),
        shuffle=False,
        num_workers=int(args.assigned_num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.assigned_persistent_workers),
        prefetch_factor=int(args.assigned_prefetch_factor),
    )
    assigned_test_loader = _build_assigned_loader(
        dataset=test_set_a,
        batch_size=int(args.assigned_batch_size),
        shuffle=False,
        num_workers=int(args.assigned_num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.assigned_persistent_workers),
        prefetch_factor=int(args.assigned_prefetch_factor),
    )

    print(
        "dataset_summary "
        f"target_unassigned_shards train={len(train_paths)} val={len(val_paths)} test={len(test_paths)} "
        f"assigned train={len(train_set_a)} val={len(val_set_a)} test={len(test_set_a)}"
    )
    print(
        "runtime "
        f"device={device} amp={bool(args.amp)} "
        f"amp_dtype={'bf16' if amp_dtype == torch.bfloat16 else 'fp16'} tf32={bool(args.tf32)} "
        f"batch_target={int(args.batch_size)} batch_assigned={int(args.assigned_batch_size)} "
        f"target_weight={float(args.target_loss_weight):.4f} "
        f"assigned_weight={float(args.assigned_loss_weight):.4f} assigned_step_ratio={int(args.assigned_step_ratio)} "
        f"integration_matching_mode={str(args.integration_matching_mode)} "
        f"hungarian_solver={str(args.hungarian_solver)} "
        f"monitor_target_weight={monitor_target_weight:.4f} "
        f"monitor_assigned_weight={monitor_assigned_weight:.4f} "
        f"trainable_mode={args.trainable_mode} trainable_params={trainable_count}/{total_count}"
    )

    log_header = [
        "epoch",
        "train_target_loss",
        "train_target_mae",
        "train_assigned_loss",
        "train_assigned_mae",
        "train_total_loss",
        "val_target_loss",
        "val_target_mae",
        "val_assigned_loss",
        "val_assigned_mae",
        "monitor",
        "lr",
        "train_steps",
        "train_target_matches",
        "train_assigned_atoms",
        "skipped_empty",
        "skipped_pred_nonfinite",
        "skipped_pred_extreme",
        "skipped_loss_nonfinite",
        "skipped_grad_nonfinite",
        "skipped_assigned_nonfinite",
    ]

    for epoch in range(start_epoch, int(args.epochs) + 1):
        epoch_start = time.time()

        stats = _train_joint_epoch(
            model=model,
            args=args,
            epoch=epoch,
            train_paths=train_paths,
            resolver=resolver,
            unassigned_cfg=unassigned_cfg,
            assigned_cfg=assigned_cfg,
            optimizer=optimizer,
            scaler=scaler,
            assigned_loader=assigned_train_loader,
            assigned_solvent_id=assigned_solvent_id,
        )

        val_target_stats = _evaluate_unassigned_splits(
            model=model,
            args=args,
            epoch=epoch,
            paths=val_paths,
            resolver=resolver,
            config=unassigned_cfg,
            batch_size=int(args.batch_size),
        )

        val_a_loss, val_a_mae = _evaluate_assigned_loader(
            model=model,
            loader=assigned_val_loader,
            config=assigned_cfg,
            assigned_solvent_id=assigned_solvent_id,
        )

        monitor = 0.0
        terms = (
            (monitor_target_weight, float(val_target_stats.loss)),
            (monitor_assigned_weight, float(val_a_loss)),
        )
        for weight, loss_value in terms:
            if weight <= 0:
                continue
            if not np.isfinite(loss_value):
                monitor = float("inf")
                break
            monitor += weight * loss_value

        lr_plateau.step(monitor)

        _log_csv(
            log_path,
            log_header,
            [
                int(epoch),
                float(stats.train_target_loss),
                float(stats.train_target_mae),
                float(stats.train_assigned_loss),
                float(stats.train_assigned_mae),
                float(stats.train_total_loss),
                float(val_target_stats.loss),
                float(val_target_stats.mae),
                float(val_a_loss),
                float(val_a_mae),
                float(monitor),
                float(optimizer.param_groups[0]["lr"]),
                int(stats.train_steps),
                int(stats.train_target_matches),
                int(stats.train_assigned_atoms),
                int(stats.skipped_empty),
                int(stats.skipped_pred_nonfinite),
                int(stats.skipped_pred_extreme),
                int(stats.skipped_loss_nonfinite),
                int(stats.skipped_grad_nonfinite),
                int(stats.skipped_assigned_nonfinite),
            ],
        )

        improved = np.isfinite(monitor) and monitor < best_monitor
        if not best_model_path.exists():
            improved = True
        if improved:
            best_monitor = float(monitor)
            best_epoch = int(epoch)
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            _save_joint_checkpoint(
                checkpoint_path=best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_monitor=best_monitor,
                best_epoch=best_epoch,
                patience_counter=patience_counter,
            )
        else:
            patience_counter += 1

        _save_joint_checkpoint(
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            best_monitor=best_monitor,
            best_epoch=best_epoch,
            patience_counter=patience_counter,
        )

        print(
            f"epoch={epoch:03d} "
            f"train_t_loss={stats.train_target_loss:.6f} train_t_mae={stats.train_target_mae:.6f} "
            f"train_a_loss={stats.train_assigned_loss:.6f} train_a_mae={stats.train_assigned_mae:.6f} "
            f"train_total={stats.train_total_loss:.6f} "
            f"val_t_loss={val_target_stats.loss:.6f} val_t_mae={val_target_stats.mae:.6f} "
            f"val_a_loss={val_a_loss:.6f} val_a_mae={val_a_mae:.6f} "
            f"monitor={monitor:.6f} lr={optimizer.param_groups[0]['lr']:.6g} "
            f"epoch_sec={time.time() - epoch_start:.1f}"
        )

        if int(args.early_stop_patience) > 0 and patience_counter >= int(args.early_stop_patience):
            print(
                f"early_stop epoch={epoch} best_monitor={best_monitor:.6f}@{best_epoch} "
                f"patience={patience_counter}/{args.early_stop_patience}"
            )
            break

    if best_model_path.exists():
        best_state = torch.load(str(best_model_path), map_location=device)
        if not isinstance(best_state, dict):
            raise TypeError(f"Best model payload invalid: {type(best_state)}")
        model.load_state_dict(best_state, strict=True)
    else:
        raise FileNotFoundError(f"Best model not found: {best_model_path}")

    test_target_stats = _evaluate_unassigned_splits(
        model=model,
        args=args,
        epoch=0,
        paths=test_paths,
        resolver=resolver,
        config=unassigned_cfg,
        batch_size=int(args.batch_size),
    )
    test_a_loss, test_a_mae = _evaluate_assigned_loader(
        model=model,
        loader=assigned_test_loader,
        config=assigned_cfg,
        assigned_solvent_id=assigned_solvent_id,
    )

    summary = {
        "mode": "joint",
        "target": args.target,
        "target_ensemble_shards_dir": str(args.ensemble_shards_dir or ""),
        "target_shards_dir": str(args.shards_dir or ""),
        "target_solvents": sorted(args.target_solvents_filter),
        "solvent_vocab": solvent_vocab_keys,
        "assigned_anchor_solvent": normalize_solvent_key(args.assigned_anchor_solvent),
        "trainable_mode": str(args.trainable_mode),
        "solvent_modules_enabled": bool(solvent_modules_enabled),
        "solvent_use_bias": bool(args.solvent_use_bias),
        "solvent_emb_dim": int(args.solvent_emb_dim),
        "solvent_adapter_hidden_dim": int(args.solvent_adapter_hidden_dim),
        "solvent_adapter_dropout": float(args.solvent_adapter_dropout),
        "trainable_parameter_count": int(trainable_count),
        "total_parameter_count": int(total_count),
        "exp22k_entries_path": str(args.exp22k_entries_path),
        "exp22k_splits_path": str(args.exp22k_splits_path),
        "target_loss_weight": float(args.target_loss_weight),
        "assigned_loss_weight": float(args.assigned_loss_weight),
        "assigned_step_ratio": int(args.assigned_step_ratio),
        "integration_matching_mode": str(args.integration_matching_mode),
        "hungarian_solver": str(args.hungarian_solver),
        "output_dim": 1,
        "monitor_target_weight": float(monitor_target_weight),
        "monitor_assigned_weight": float(monitor_assigned_weight),
        "best_epoch": int(best_epoch),
        "best_monitor": float(best_monitor),
        "test_target_unassigned_loss": float(test_target_stats.loss),
        "test_target_unassigned_mae": float(test_target_stats.mae),
        "test_assigned_loss": float(test_a_loss),
        "test_assigned_mae": float(test_a_mae),
        "best_model_path": str(best_model_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "log_csv": str(log_path),
        "train_target_unassigned_shards": int(len(train_paths)),
        "val_target_unassigned_shards": int(len(val_paths)),
        "test_target_unassigned_shards": int(len(test_paths)),
        "train_assigned_molecules": int(len(train_set_a)),
        "val_assigned_molecules": int(len(val_set_a)),
        "test_assigned_molecules": int(len(test_set_a)),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def _resolve_exp22k_defaults(args: argparse.Namespace) -> None:
    if args.exp22k_entries_path is None:
        if args.target == "1H":
            args.exp22k_entries_path = Path(args.exp22k_data_root) / "dft8k_dft_1h_entries.pkl"
        else:
            args.exp22k_entries_path = Path(args.exp22k_data_root) / "exp22k_ff_13c_entries.pkl"
    if args.exp22k_splits_path is None:
        if args.target == "1H":
            args.exp22k_splits_path = (
                Path(args.exp22k_data_root)
                / "dft8k_dft_1h_splits_nmrexp_test_optimized_target16000_eligible_20260215.json"
            )
        else:
            args.exp22k_splits_path = (
                Path(args.exp22k_data_root)
                / "exp22k_ff_13c_splits_nmrexp_test_optimized_target16000_eligible_20260214.json"
            )


def build_parser() -> argparse.ArgumentParser:
    data_root = Path(os.environ.get("CASCADE_DATA_ROOT", "./data"))

    parser = argparse.ArgumentParser(
        description="CASCADE joint trainer for manuscript reproduction (deterministic output_dim=1)."
    )
    parser.add_argument("--target", choices=["13C", "1H"], default="13C")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ensemble-shards-dir", type=str, required=True)
    parser.add_argument("--target-max-train-shards", type=int, default=0)
    parser.add_argument("--target-max-val-shards", type=int, default=0)
    parser.add_argument("--target-max-test-shards", type=int, default=0)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=27407)
    parser.add_argument("--target-solvents", type=str, default="")
    parser.add_argument("--solvent-vocab", type=str, default="")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--assigned-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.7)
    parser.add_argument("--reduce-lr-patience", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--assigned-loss-weight", type=float, default=0.25)
    parser.add_argument("--assigned-step-ratio", type=int, default=1)

    parser.add_argument("--dummy-cost", type=float, default=1000.0)
    parser.add_argument("--hungarian-solver", choices=["scipy", "dp"], default="dp")
    parser.add_argument("--matching-workers", type=int, default=8)
    parser.add_argument(
        "--integration-matching-mode",
        choices=["none", "weighted", "expanded"],
        default="expanded",
    )
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument(
        "--max-abs-pred",
        type=float,
        default=0.0,
        help="Skip unassigned batches whose predicted |ppm| exceeds this threshold (<=0 disables).",
    )

    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--units", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-radial", type=int, default=20)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--cutoff", type=float, default=5.0)

    parser.add_argument(
        "--trainable-mode",
        choices=["full", "solvent_only", "solvent_plus_head", "solvent_plus_lastblock"],
        default="full",
    )
    parser.add_argument("--solvent-emb-dim", type=int, default=16)
    parser.add_argument("--solvent-adapter-hidden-dim", type=int, default=0)
    parser.add_argument("--solvent-adapter-dropout", type=float, default=0.0)
    parser.add_argument("--solvent-use-bias", action="store_true", default=False)
    parser.add_argument("--assigned-anchor-solvent", type=str, default="CDCl3")

    parser.add_argument("--ensemble-conformer-sampling", choices=["boltzmann", "uniform", "lowest"], default="boltzmann")
    parser.add_argument("--ensemble-eval-sampling", choices=["boltzmann", "uniform", "lowest"], default="lowest")
    parser.add_argument("--ensemble-temperature", type=float, default=298.15)
    parser.add_argument("--ensemble-boltzmann-alpha", type=float, default=1.0)
    parser.add_argument("--min-conformer-distance", type=float, default=0.7)
    parser.add_argument("--disable-fragment-filter", action="store_true", default=False)
    parser.add_argument("--atom-tokenizer-preprocessor", type=Path, default=PAINN_DIR / "preprocessor_orig.p")

    parser.add_argument("--exp22k-data-root", type=Path, default=None)
    parser.add_argument("--exp22k-entries-path", type=Path, default=None)
    parser.add_argument("--exp22k-splits-path", type=Path, default=None)
    parser.add_argument("--exp22k-train-size", type=int, default=0)
    parser.add_argument("--exp22k-max-val-mols", type=int, default=0)
    parser.add_argument("--exp22k-max-test-mols", type=int, default=0)
    parser.add_argument("--assigned-normalize-targets", action="store_true", default=False)
    parser.add_argument("--assigned-mean", type=float, default=None)
    parser.add_argument("--assigned-std", type=float, default=None)

    parser.add_argument("--init-from", type=str, default="")
    parser.add_argument("--init-strict", action="store_true", default=True)
    parser.add_argument("--init-nonstrict", dest="init_strict", action="store_false")
    parser.add_argument("--resume", action="store_true", default=False)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--assigned-num-workers", type=int, default=4)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.set_defaults(exp22k_data_root=data_root / "Exp22K_FF")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Keep internal training behavior fixed to manuscript defaults.
    args.shards_dir = ""
    args.graph_tag = "painn"
    args.shard_glob = ""
    args.ensemble_shard_glob = ""
    args.val_fraction = 0.05
    args.test_fraction = 0.05
    args.shuffle_shards = True
    args.monitor_target_weight = None
    args.monitor_assigned_weight = None
    args.no_atom_tokenizer_map = False
    args.drop_smiles_file = None
    args.pin_memory = True
    args.persistent_workers = False
    args.assigned_persistent_workers = False
    args.prefetch_factor = 2
    args.assigned_prefetch_factor = 2
    args.log_shard_progress = False
    args.tf32 = True

    args.target = canonical_target(args.target)
    if not args.shards_dir and not args.ensemble_shards_dir:
        raise ValueError("Provide --ensemble-shards-dir.")

    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")
    if int(args.assigned_batch_size) <= 0:
        raise ValueError("--assigned-batch-size must be > 0")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be > 0")
    if float(args.lr) <= 0:
        raise ValueError("--lr must be > 0")
    if float(args.reduce_lr_factor) <= 0 or float(args.reduce_lr_factor) >= 1:
        raise ValueError("--reduce-lr-factor must be in (0,1)")
    if int(args.reduce_lr_patience) < 0:
        raise ValueError("--reduce-lr-patience must be >= 0")
    if float(args.min_lr) <= 0:
        raise ValueError("--min-lr must be > 0")
    if int(args.early_stop_patience) <= 0:
        raise ValueError("--early-stop-patience must be > 0")
    if float(args.grad_clip) < 0:
        raise ValueError("--grad-clip must be >= 0")

    if int(args.assigned_step_ratio) < 0:
        raise ValueError("--assigned-step-ratio must be >= 0")

    for name in ("target_max_train_shards", "target_max_val_shards", "target_max_test_shards"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")

    if float(args.val_fraction) < 0 or float(args.test_fraction) < 0:
        raise ValueError("--val-fraction and --test-fraction must be >= 0")

    if int(args.matching_workers) <= 0:
        raise ValueError("--matching-workers must be > 0")

    if str(args.integration_matching_mode).lower() not in {"none", "weighted", "expanded"}:
        raise ValueError("--integration-matching-mode must be one of: none, weighted, expanded")
    args.integration_matching_mode = str(args.integration_matching_mode).lower()

    if float(args.std) <= 0 and bool(args.normalize):
        raise ValueError("--std must be > 0 when --normalize is enabled")

    if float(args.target_loss_weight) < 0 or float(args.assigned_loss_weight) < 0:
        raise ValueError("--target-loss-weight and --assigned-loss-weight must be >= 0")
    if float(args.target_loss_weight) == 0 and float(args.assigned_loss_weight) == 0:
        raise ValueError("At least one of target/assigned loss weights must be > 0")

    if int(args.input_dim) <= 0 or int(args.units) <= 0 or int(args.depth) <= 0 or int(args.num_radial) <= 0:
        raise ValueError("Model dimensions must be > 0")
    if int(args.n_neighbors) <= 0:
        raise ValueError("--n-neighbors must be > 0")
    if float(args.cutoff) <= 0:
        raise ValueError("--cutoff must be > 0")

    if float(args.solvent_adapter_dropout) < 0:
        raise ValueError("--solvent-adapter-dropout must be >= 0")
    if int(args.solvent_adapter_hidden_dim) > 0 and int(args.solvent_emb_dim) <= 0:
        raise ValueError("--solvent-adapter-hidden-dim requires --solvent-emb-dim > 0")

    if int(args.num_workers) < 0 or int(args.assigned_num_workers) < 0:
        raise ValueError("--num-workers and --assigned-num-workers must be >= 0")
    if int(args.prefetch_factor) < 0 or int(args.assigned_prefetch_factor) < 0:
        raise ValueError("--prefetch-factor and --assigned-prefetch-factor must be >= 0")

    if int(args.exp22k_train_size) < 0 or int(args.exp22k_max_val_mols) < 0 or int(args.exp22k_max_test_mols) < 0:
        raise ValueError("Assigned split truncation args must be >= 0")

    args.assigned_anchor_solvent = normalize_solvent_name(args.assigned_anchor_solvent)

    if args.exp22k_data_root is None:
        data_root = Path(os.environ.get("CASCADE_DATA_ROOT", "./data"))
        if args.target == "1H":
            args.exp22k_data_root = data_root / "original_cascade_data" / "DFT8K"
        else:
            args.exp22k_data_root = data_root / "Exp22K_FF"

    _resolve_exp22k_defaults(args)

    if args.exp22k_entries_path is None or args.exp22k_splits_path is None:
        raise RuntimeError("Failed to resolve assigned entries/splits paths.")

    if not Path(args.exp22k_entries_path).exists():
        raise FileNotFoundError(f"Assigned entries not found: {args.exp22k_entries_path}")
    if not Path(args.exp22k_splits_path).exists():
        raise FileNotFoundError(f"Assigned splits not found: {args.exp22k_splits_path}")

    if args.init_from:
        args.init_from = str(args.init_from).strip()

    train_joint(args)


if __name__ == "__main__":
    main()
