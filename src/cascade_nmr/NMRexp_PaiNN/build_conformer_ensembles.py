from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import AllChem


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    class tqdm:
        def __init__(self, total=None, desc=None, unit=None):
            self.total = total
            self.desc = desc or ""
            self.unit = unit or "it"
            self.count = 0
            self.last_print = 0

        def update(self, n=1):
            self.count += n
            if self.count - self.last_print < 1000 and self.total:
                return
            self.last_print = self.count
            if self.total:
                pct = 100.0 * self.count / self.total
                msg = f"\r{self.desc} {self.count}/{self.total} {self.unit} ({pct:5.1f}%)"
            else:
                msg = f"\r{self.desc} {self.count} {self.unit}"
            sys.stderr.write(msg)
            sys.stderr.flush()

        def close(self):
            sys.stderr.write("\n")
            sys.stderr.flush()


KCAL_PER_MOL_KT = 0.0019872041
_INTEG_EXPAND_FULL_TARGET_KEY = "integration_expand_full_target_total"
_INTEG_EXPAND_FULL_OFFSET_KEY = "integration_expand_full_offset"
_INTEG_EXPAND_FULL_PEAK_IDX_KEY = "integration_expand_full_peak_idx"
_INTEG_EXPAND_FRAGMENT_TARGET_KEY = "integration_expand_fragment_target_total"
_INTEG_EXPAND_FRAGMENT_OFFSET_KEY = "integration_expand_fragment_offset"
_INTEG_EXPAND_FRAGMENT_PEAK_IDX_KEY = "integration_expand_fragment_peak_idx"


def parse_integration(raw):
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.endswith("H"):
        raw = raw[:-1]
    try:
        return float(raw)
    except ValueError:
        return None


def _valid_integration_array(integrations, expected_len):
    if integrations is None or int(expected_len) <= 0:
        return None
    if len(integrations) != int(expected_len):
        return None
    values = np.empty((int(expected_len),), dtype=np.float64)
    for idx, raw in enumerate(integrations):
        if raw is None:
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val) or val <= 0:
            return None
        values[idx] = val
    return values


def _integration_repeats_from_values(integration_values, target_total):
    n_obs = int(integration_values.shape[0])
    target_total = int(target_total)
    if n_obs <= 0 or target_total <= 0:
        return None
    total = float(np.sum(integration_values))
    if not np.isfinite(total) or total <= 0:
        return None

    scaled = integration_values * (float(target_total) / total)
    repeats = np.floor(scaled).astype(np.int64, copy=False)
    floor_vals = np.floor(scaled)
    frac = scaled - floor_vals

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


def _precompute_expanded_peak_index(peaks, integrations, target_total):
    n_obs = int(len(peaks))
    if n_obs <= 0 or int(target_total) <= 0:
        return None
    integ_vals = _valid_integration_array(integrations, expected_len=n_obs)
    if integ_vals is None:
        return None
    repeats = _integration_repeats_from_values(integ_vals, target_total=int(target_total))
    if repeats is None:
        return None
    keep = repeats > 0
    if not np.any(keep):
        return None
    peak_idx = np.repeat(np.arange(n_obs, dtype=np.int32)[keep], repeats[keep])
    if int(peak_idx.shape[0]) <= 0:
        return None
    return peak_idx.astype(np.int32, copy=False)


def _pack_ragged_int32(rows):
    n_row = int(len(rows))
    if n_row == 0:
        return np.zeros((1,), dtype=np.int64), np.zeros((0,), dtype=np.int32)
    lengths = np.empty((n_row,), dtype=np.int64)
    normalized = []
    for idx, row in enumerate(rows):
        arr = np.asarray(row, dtype=np.int32).reshape(-1)
        lengths[idx] = int(arr.shape[0])
        normalized.append(arr)
    offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)]).astype(np.int64, copy=False)
    if int(offsets[-1]) <= 0:
        flat = np.zeros((0,), dtype=np.int32)
    else:
        flat = np.concatenate(normalized, axis=0).astype(np.int32, copy=False)
    return offsets, flat


def normalize_nucleus(nmr_type):
    if not isinstance(nmr_type, str):
        return None
    t = nmr_type.strip()
    if t.startswith("1H"):
        return "1H"
    if t.startswith("13C"):
        return "13C"
    if t.startswith("19F"):
        return "19F"
    if t.startswith("31P") or t == "P NMR" or t.endswith("31P NMR"):
        return "31P"
    if t.startswith("11B") or t.startswith("10B") or t == "B NMR":
        return "11B"
    if t.startswith("29Si") or t == "Si NMR":
        return "29Si"
    return t


def parse_processed(nmr_type, processed):
    if not isinstance(processed, str):
        return None, None
    try:
        parsed = ast.literal_eval(processed)
    except Exception:
        return None, None
    if not isinstance(parsed, (list, tuple)):
        return None, None

    if isinstance(nmr_type, str) and nmr_type.startswith("1H"):
        shifts = []
        integrations = []
        for item in parsed:
            if not isinstance(item, (list, tuple)) or len(item) != 5:
                continue
            _, _, integ, start, end = item
            try:
                shift = (float(start) + float(end)) / 2.0
            except Exception:
                continue
            shifts.append(shift)
            integrations.append(parse_integration(integ))
        return shifts, integrations

    shifts = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            continue
        try:
            shifts.append(float(item[0]))
        except Exception:
            continue
    return shifts, None


def _iter_nmrexp_batches(parquet_path, columns, row_groups=None, batch_size=5000):
    parquet_file = pq.ParquetFile(parquet_path)
    if row_groups is None:
        row_groups = list(range(parquet_file.num_row_groups))
    for row_group in row_groups:
        for batch in parquet_file.iter_batches(
            row_groups=[row_group],
            columns=columns,
            batch_size=batch_size,
        ):
            yield batch.to_pandas()


def load_manifest(manifest_path):
    manifest = {}
    splits = set()
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in manifest {manifest_path} at line {line_num}"
                ) from exc
            row_id = record.get("row_id")
            if row_id is None:
                continue
            split = record.get("split") or "all"
            manifest[int(row_id)] = split
            splits.add(split)
    if not splits:
        splits.add("all")
    return manifest, splits


def order_splits(splits):
    preferred = ["train", "val", "test"]
    ordered = [name for name in preferred if name in splits]
    for name in sorted(splits):
        if name not in ordered:
            ordered.append(name)
    return ordered


def stable_seed(base_seed: int, row_id: int) -> int:
    token = f"{base_seed}|{row_id}"
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    # RDKit expects a 32-bit signed integer seed.
    return int(digest[:8], 16) % 0x7FFFFFFF


def embed_conformers(mol, params, num_confs, max_embed_tries):
    conf_ids = []
    for _ in range(max_embed_tries):
        mol.RemoveAllConformers()
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
        if conf_ids:
            break
    return conf_ids


def optimize_conformers(mol, conf_ids, max_iters, allow_uff, num_threads):
    del num_threads  # API compatibility: per-conformer optimization uses single-threaded calls.
    energies = []
    statuses = []
    if AllChem.MMFFHasAllMoleculeParams(mol):
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        for cid in conf_ids:
            status = AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters, confId=int(cid))
            ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=int(cid))
            energy = ff.CalcEnergy() if ff is not None else float("nan")
            statuses.append(int(status))
            energies.append(float(energy))
    elif allow_uff:
        for cid in conf_ids:
            status = AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters, confId=int(cid))
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(cid))
            energy = ff.CalcEnergy() if ff is not None else float("nan")
            statuses.append(int(status))
            energies.append(float(energy))
    else:
        return None, None, "mmff_missing"

    return np.array(energies, dtype=np.float64), np.array(statuses, dtype=np.int32), None


def boltzmann_weights(rel_energies, temperature, alpha):
    if rel_energies.size == 0:
        return rel_energies
    if temperature <= 0:
        return np.full_like(rel_energies, 1.0 / rel_energies.size, dtype=np.float64)
    kT = KCAL_PER_MOL_KT * temperature
    beta = 1.0 / (alpha * kT)
    weights = np.exp(-rel_energies * beta)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return np.full_like(rel_energies, 1.0 / rel_energies.size, dtype=np.float64)
    return weights / total


def _select_largest_fragment_atom_ids_from_fragments(atom_numbers, frags):
    if len(frags) <= 1:
        return None
    best_atom_ids = None
    best_score = None
    for frag_idx, atom_ids in enumerate(frags):
        atom_ids_arr = np.asarray(atom_ids, dtype=np.int64)
        frag_nums = atom_numbers[atom_ids_arr]
        n_carbons = int(np.sum(frag_nums == 6))
        n_heavy = int(np.sum(frag_nums > 1))
        n_atoms = int(atom_ids_arr.shape[0])
        has_carbon = int(n_carbons > 0)
        score = (has_carbon, n_heavy, n_atoms, n_carbons, -frag_idx)
        if best_score is None or score > best_score:
            best_score = score
            best_atom_ids = atom_ids_arr
    if best_atom_ids is None or best_atom_ids.shape[0] == atom_numbers.shape[0]:
        return None
    return np.sort(best_atom_ids)


def _build_edge_indices_from_coords(coords, n_neighbors, cutoff):
    n_atom = int(coords.shape[0])
    if n_atom <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    if n_atom == 1:
        return np.array([[0, 0]], dtype=np.int32)

    diffs = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    edge_indices = []
    for i in range(n_atom):
        row = dist[i]
        order = np.argsort(row)
        neighbors = []
        for nb in order[1:]:
            if cutoff > 0 and row[nb] >= cutoff:
                break
            neighbors.append(int(nb))
            if n_neighbors > 0 and len(neighbors) >= n_neighbors:
                break
        if not neighbors:
            neighbors = [i]
        for nb in neighbors:
            edge_indices.append([i, nb])
    return np.asarray(edge_indices, dtype=np.int32)


def _precompute_edge_indices(coords, n_atom, n_conf, n_neighbors, cutoff):
    n_edge_per_conf = np.zeros((int(n_conf),), dtype=np.int64)
    edge_chunks = []
    for conf_idx in range(int(n_conf)):
        start = conf_idx * int(n_atom)
        end = start + int(n_atom)
        conf_coords = coords[start:end]
        edges = _build_edge_indices_from_coords(conf_coords, n_neighbors=n_neighbors, cutoff=cutoff)
        n_edge_per_conf[conf_idx] = int(edges.shape[0])
        edge_chunks.append(edges)
    edge_indices = (
        np.concatenate(edge_chunks, axis=0).astype(np.int32, copy=False)
        if edge_chunks
        else np.zeros((0, 2), dtype=np.int32)
    )
    return n_edge_per_conf, edge_indices


def update_running_stats(stats, values):
    for value in values:
        x = float(value)
        stats["count"] += 1
        delta = x - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = x - stats["mean"]
        stats["m2"] += delta * delta2


def finalize_stats(stats):
    if stats["count"] <= 1:
        return stats["mean"], 1.0
    variance = stats["m2"] / (stats["count"] - 1)
    std = float(np.sqrt(max(variance, 1e-12)))
    return float(stats["mean"]), std


def init_buffer():
    return {
        "n_atom": [],
        "n_conf": [],
        "atom_numbers": [],
        "coords": [],
        "n_edge_per_conf": [],
        "edge_indices": [],
        "fragment_n_atom": [],
        "fragment_atom_ids": [],
        "rel_energies": [],
        "weights": [],
        "symm": [],
        "h_counts": [],
        "peaks": [],
        "integrations": [],
        _INTEG_EXPAND_FULL_TARGET_KEY: [],
        _INTEG_EXPAND_FULL_PEAK_IDX_KEY: [],
        _INTEG_EXPAND_FRAGMENT_TARGET_KEY: [],
        _INTEG_EXPAND_FRAGMENT_PEAK_IDX_KEY: [],
        "smiles": [],
        "solvent": [],
        "row_id": [],
    }


def write_shard(output_dir, shard_idx, buffer, coord_dtype, store_weights, precompute_edges):
    if not buffer["n_atom"]:
        return None

    n_atom = np.array(buffer["n_atom"], dtype=np.int64)
    n_conf = np.array(buffer["n_conf"], dtype=np.int64)
    conf_sizes = (n_atom * n_conf).astype(np.int64)

    atom_numbers = np.concatenate(buffer["atom_numbers"], axis=0).astype(np.int64)
    coords = np.vstack(buffer["coords"]).astype(coord_dtype)
    fragment_n_atom = np.array(buffer["fragment_n_atom"], dtype=np.int64)
    fragment_atom_ids = np.concatenate(buffer["fragment_atom_ids"], axis=0).astype(np.int64)
    rel_energies = np.concatenate(buffer["rel_energies"], axis=0).astype(np.float32)
    if len(buffer[_INTEG_EXPAND_FULL_TARGET_KEY]) != int(n_atom.shape[0]):
        raise RuntimeError("integration_expand_full_target_total row count mismatch while writing shard.")
    if len(buffer[_INTEG_EXPAND_FRAGMENT_TARGET_KEY]) != int(n_atom.shape[0]):
        raise RuntimeError("integration_expand_fragment_target_total row count mismatch while writing shard.")
    full_offsets, full_peak_idx = _pack_ragged_int32(buffer[_INTEG_EXPAND_FULL_PEAK_IDX_KEY])
    fragment_offsets, fragment_peak_idx = _pack_ragged_int32(buffer[_INTEG_EXPAND_FRAGMENT_PEAK_IDX_KEY])

    mol_atom_offset = np.concatenate([[0], np.cumsum(n_atom)])
    mol_conf_offset = np.concatenate([[0], np.cumsum(n_conf)])
    mol_coord_offset = np.concatenate([[0], np.cumsum(conf_sizes)])

    out = {
        "n_atom": n_atom,
        "n_conf": n_conf,
        "atom_numbers": atom_numbers,
        "coords": coords,
        "rel_energies": rel_energies,
        "symm": np.concatenate(buffer["symm"], axis=0).astype(np.int64),
        "h_counts": np.concatenate(buffer["h_counts"], axis=0).astype(np.int64),
        "peaks": np.array(buffer["peaks"], dtype=object),
        "integrations": np.array(buffer["integrations"], dtype=object),
        _INTEG_EXPAND_FULL_TARGET_KEY: np.asarray(buffer[_INTEG_EXPAND_FULL_TARGET_KEY], dtype=np.int32),
        _INTEG_EXPAND_FULL_OFFSET_KEY: full_offsets,
        _INTEG_EXPAND_FULL_PEAK_IDX_KEY: full_peak_idx,
        _INTEG_EXPAND_FRAGMENT_TARGET_KEY: np.asarray(buffer[_INTEG_EXPAND_FRAGMENT_TARGET_KEY], dtype=np.int32),
        _INTEG_EXPAND_FRAGMENT_OFFSET_KEY: fragment_offsets,
        _INTEG_EXPAND_FRAGMENT_PEAK_IDX_KEY: fragment_peak_idx,
        "mol_atom_offset": mol_atom_offset,
        "mol_conf_offset": mol_conf_offset,
        "mol_coord_offset": mol_coord_offset,
        "smiles": np.array(buffer["smiles"], dtype=object),
        "solvent": np.array(buffer["solvent"], dtype=object),
        "row_id": np.array(buffer["row_id"], dtype=np.int64),
        "fragment_n_atom": fragment_n_atom,
        "fragment_atom_ids": fragment_atom_ids,
    }
    if precompute_edges:
        out["n_edge_per_conf"] = np.concatenate(buffer["n_edge_per_conf"], axis=0).astype(np.int64)
        out["edge_indices"] = np.concatenate(buffer["edge_indices"], axis=0).astype(np.int32)
    if store_weights:
        weights = np.concatenate(buffer["weights"], axis=0).astype(np.float32)
        out["weights"] = weights

    shard_path = os.path.join(output_dir, f"nmrexp_ensemble_shard_{shard_idx:05d}.npz")
    np.savez_compressed(shard_path, **out)
    return shard_path


def _parse_shard_index(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("nmrexp_ensemble_shard_"):
        return None
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return None


def find_resume_state(split_dirs: dict[str, str]) -> tuple[dict[str, int], int]:
    shard_indices = {}
    max_row_id = -1
    for split_name, split_dir in split_dirs.items():
        shard_dir = Path(split_dir)
        shard_paths = list(shard_dir.glob("nmrexp_ensemble_shard_*.npz"))
        if not shard_paths:
            shard_indices[split_name] = 0
            continue

        shard_indices[split_name] = max(
            idx for idx in (_parse_shard_index(path) for path in shard_paths) if idx is not None
        ) + 1

        last_idx = shard_indices[split_name] - 1
        last_path = shard_dir / f"nmrexp_ensemble_shard_{last_idx:05d}.npz"
        if last_path.exists():
            try:
                data = np.load(last_path, allow_pickle=True)
                if "row_id" in data:
                    max_row_id = max(max_row_id, int(np.max(data["row_id"])))
            except Exception:
                pass

    return shard_indices, max_row_id + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build conformer ensemble shards for NMRexp molecules."
    )
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--target", choices=["13C", "1H", "all"], default="13C")
    parser.add_argument("--require_peaks", action="store_true", default=False)
    parser.add_argument("--min_shift", type=float, default=None)
    parser.add_argument("--max_shift", type=float, default=None)
    parser.add_argument("--row_groups", type=str, default="")
    parser.add_argument("--batch_read_size", type=int, default=2000)
    parser.add_argument("--shard_size", type=int, default=500)
    parser.add_argument("--max_mols", type=int, default=0)
    parser.add_argument("--max_atoms", type=int, default=0)
    parser.add_argument("--num_confs", type=int, default=20)
    parser.add_argument("--max_confs", type=int, default=20)
    parser.add_argument("--min_confs", type=int, default=1)
    parser.add_argument("--prune_rms", type=float, default=0.5)
    parser.add_argument("--max_embed_tries", type=int, default=5)
    parser.add_argument("--embed_seed", type=int, default=0xF00D)
    parser.add_argument("--embed_threads", type=int, default=1)
    parser.add_argument("--mmff_max_iters", type=int, default=200)
    parser.add_argument("--allow_uff", action="store_true", default=False)
    parser.add_argument("--energy_window", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--boltzmann_alpha", type=float, default=1.0)
    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--precompute-edges", action="store_true", default=True)
    parser.add_argument("--no-precompute-edges", action="store_false", dest="precompute_edges")
    parser.add_argument(
        "--coord_dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument("--store_weights", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume_from", type=int, default=0)
    args = parser.parse_args()

    if args.num_confs <= 0:
        raise ValueError("num_confs must be positive.")
    if args.max_confs < 0:
        raise ValueError("max_confs must be >= 0.")
    if args.min_confs < 1:
        raise ValueError("min_confs must be >= 1.")
    if args.max_confs and args.max_confs < args.min_confs:
        raise ValueError("max_confs must be >= min_confs when provided.")
    if args.target == "13C":
        target_atomic_num = 6
    elif args.target == "1H":
        target_atomic_num = 1
    else:
        target_atomic_num = None

    coord_dtype = np.float16 if args.coord_dtype == "float16" else np.float32

    manifest = None
    manifest_splits = None
    if args.manifest_path:
        manifest, manifest_splits = load_manifest(args.manifest_path)

    shift_ranges = {"13C": (-5.0, 250.0), "1H": (-1.0, 15.0)}
    if args.min_shift is not None or args.max_shift is not None:
        low = args.min_shift if args.min_shift is not None else -1e9
        high = args.max_shift if args.max_shift is not None else 1e9
    else:
        low, high = shift_ranges.get(args.target, (-1e9, 1e9))

    row_groups = None
    if args.row_groups:
        row_groups = [int(x) for x in args.row_groups.split(",") if x.strip()]

    if manifest_splits:
        if manifest_splits == {"all"}:
            split_dirs = {"all": args.output_dir}
        else:
            split_dirs = {
                split: os.path.join(args.output_dir, split)
                for split in order_splits(manifest_splits)
            }
        for path in split_dirs.values():
            os.makedirs(path, exist_ok=True)
        buffers = {split: init_buffer() for split in split_dirs}
        shard_indices = {split: 0 for split in split_dirs}
        split_counts = {split: 0 for split in split_dirs}
    else:
        split_dirs = {"all": args.output_dir}
        os.makedirs(args.output_dir, exist_ok=True)
        buffers = {"all": init_buffer()}
        shard_indices = {"all": 0}
        split_counts = {"all": 0}

    resume_from = max(args.resume_from, 0)
    if args.resume:
        shard_indices, computed_resume = find_resume_state(split_dirs)
        if resume_from == 0 and computed_resume > 0:
            resume_from = computed_resume

    need_peaks = bool(args.require_peaks or args.target != "all")

    columns = ["SMILES", "NMR_solvent"]
    if manifest is None or args.target != "all" or need_peaks:
        columns.extend(["NMR_type", "NMR_processed"])

    total_rows = None
    try:
        parquet_file = pq.ParquetFile(args.parquet_path)
        if row_groups is None:
            total_rows = parquet_file.metadata.num_rows
        else:
            total_rows = sum(
                parquet_file.metadata.row_group(idx).num_rows for idx in row_groups
            )
    except Exception:
        total_rows = None

    params = AllChem.ETKDGv3()
    params.numThreads = args.embed_threads
    if args.prune_rms > 0:
        params.pruneRmsThresh = args.prune_rms

    skip_counts = {
        "rows_seen": 0,
        "manifest_skip": 0,
        "wrong_nucleus": 0,
        "parse_failed": 0,
        "no_shifts": 0,
        "all_shifts_filtered": 0,
        "invalid_smiles": 0,
        "too_many_atoms": 0,
        "no_targets": 0,
        "embed_failed": 0,
        "mmff_missing": 0,
        "no_confs": 0,
        "resume_skip": 0,
    }
    stats = {"count": 0, "mean": 0.0, "m2": 0.0}

    total = 0
    row_offset = 0
    pbar = tqdm(total=total_rows, desc="Reading rows", unit="rows")
    try:
        for df in _iter_nmrexp_batches(
            args.parquet_path, columns, row_groups=row_groups, batch_size=args.batch_read_size
        ):
            row_ids = np.arange(row_offset, row_offset + len(df))
            row_offset += len(df)
            df = df.assign(row_id=row_ids)

            processed = 0
            for row in df.itertuples(index=False):
                processed += 1
                skip_counts["rows_seen"] += 1

                if resume_from and int(row.row_id) < resume_from:
                    skip_counts["resume_skip"] += 1
                    continue

                if manifest is not None:
                    split_name = manifest.get(int(row.row_id))
                    if split_name is None:
                        skip_counts["manifest_skip"] += 1
                        continue
                else:
                    split_name = "all"

                if args.target != "all":
                    nucleus = normalize_nucleus(getattr(row, "NMR_type", None))
                    if nucleus != args.target:
                        skip_counts["wrong_nucleus"] += 1
                        continue

                filtered = None
                filtered_integ = None
                if need_peaks:
                    shifts, integrations = parse_processed(
                        getattr(row, "NMR_type", None),
                        getattr(row, "NMR_processed", None),
                    )
                    if shifts is None:
                        skip_counts["parse_failed"] += 1
                        continue
                    if not shifts:
                        skip_counts["no_shifts"] += 1
                        continue
                    filtered = [shift for shift in shifts if low <= shift <= high]
                    if not filtered:
                        skip_counts["all_shifts_filtered"] += 1
                        continue
                    filtered_integ = []
                    if integrations:
                        for idx_shift, shift in enumerate(shifts):
                            if low <= shift <= high:
                                filtered_integ.append(integrations[idx_shift])
                    if not filtered_integ:
                        filtered_integ = None

                mol = Chem.MolFromSmiles(row.SMILES)
                if mol is None:
                    skip_counts["invalid_smiles"] += 1
                    continue
                if args.max_atoms and mol.GetNumAtoms() > args.max_atoms:
                    skip_counts["too_many_atoms"] += 1
                    continue

                mol = Chem.AddHs(mol)
                params.randomSeed = stable_seed(args.embed_seed, int(row.row_id))
                conf_ids = embed_conformers(
                    mol, params, args.num_confs, args.max_embed_tries
                )
                if not conf_ids:
                    skip_counts["embed_failed"] += 1
                    continue

                energies, statuses, fail_reason = optimize_conformers(
                    mol,
                    conf_ids,
                    args.mmff_max_iters,
                    args.allow_uff,
                    args.embed_threads,
                )
                if energies is None:
                    skip_counts["mmff_missing"] += 1
                    continue

                valid = np.isfinite(energies) & (statuses >= 0)
                if not np.any(valid):
                    skip_counts["no_confs"] += 1
                    continue

                conf_ids = [cid for cid, keep in zip(conf_ids, valid) if keep]
                energies = energies[valid]

                min_energy = float(np.min(energies))
                rel_energies = energies - min_energy

                order = np.argsort(rel_energies)
                rel_energies = rel_energies[order]
                conf_ids = [conf_ids[idx] for idx in order]

                if args.energy_window > 0:
                    keep_mask = rel_energies <= args.energy_window
                    rel_energies = rel_energies[keep_mask]
                    conf_ids = [cid for cid, keep in zip(conf_ids, keep_mask) if keep]

                if args.max_confs and len(conf_ids) > args.max_confs:
                    rel_energies = rel_energies[: args.max_confs]
                    conf_ids = conf_ids[: args.max_confs]

                if len(conf_ids) < args.min_confs:
                    skip_counts["no_confs"] += 1
                    continue

                coords = []
                for cid in conf_ids:
                    pos = mol.GetConformer(int(cid)).GetPositions()
                    coords.append(pos.astype(np.float32, copy=False))
                coords = np.stack(coords, axis=0)
                n_conf = coords.shape[0]
                n_atom = coords.shape[1]
                coords = coords.reshape(n_conf * n_atom, 3)

                atom_numbers = np.array(
                    [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64
                )
                if target_atomic_num is not None and not np.any(atom_numbers == target_atomic_num):
                    skip_counts["no_targets"] += 1
                    continue
                symm = np.array(Chem.CanonicalRankAtoms(mol, breakTies=False), dtype=np.int64)
                h_counts = np.array(
                    [atom.GetTotalNumHs(includeNeighbors=True) for atom in mol.GetAtoms()],
                    dtype=np.int64,
                )
                frag_atom_ids = _select_largest_fragment_atom_ids_from_fragments(
                    atom_numbers,
                    Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False),
                )
                if frag_atom_ids is None:
                    frag_n_atom = 0
                    frag_atom_ids_arr = np.zeros((0,), dtype=np.int64)
                else:
                    frag_n_atom = int(frag_atom_ids.shape[0])
                    frag_atom_ids_arr = frag_atom_ids.astype(np.int64, copy=False)

                if args.precompute_edges:
                    n_edge_per_conf, edge_indices = _precompute_edge_indices(
                        coords,
                        n_atom=n_atom,
                        n_conf=n_conf,
                        n_neighbors=int(args.n_neighbors),
                        cutoff=float(args.cutoff),
                    )
                    buffers[split_name]["n_edge_per_conf"].append(n_edge_per_conf)
                    buffers[split_name]["edge_indices"].append(edge_indices)

                buffers[split_name]["n_atom"].append(int(n_atom))
                buffers[split_name]["n_conf"].append(int(n_conf))
                buffers[split_name]["atom_numbers"].append(atom_numbers)
                buffers[split_name]["coords"].append(coords)
                buffers[split_name]["fragment_n_atom"].append(frag_n_atom)
                buffers[split_name]["fragment_atom_ids"].append(frag_atom_ids_arr)
                buffers[split_name]["rel_energies"].append(rel_energies.astype(np.float32))
                buffers[split_name]["symm"].append(symm)
                buffers[split_name]["h_counts"].append(h_counts)
                peaks_row = filtered if filtered is not None else []
                buffers[split_name]["peaks"].append(peaks_row)
                buffers[split_name]["integrations"].append(filtered_integ)
                full_target_total = int(np.sum(atom_numbers == int(target_atomic_num))) if target_atomic_num is not None else 0
                full_peak_idx = _precompute_expanded_peak_index(
                    peaks_row,
                    filtered_integ,
                    target_total=full_target_total,
                )
                if full_peak_idx is None:
                    buffers[split_name][_INTEG_EXPAND_FULL_TARGET_KEY].append(-1)
                    buffers[split_name][_INTEG_EXPAND_FULL_PEAK_IDX_KEY].append(np.zeros((0,), dtype=np.int32))
                else:
                    buffers[split_name][_INTEG_EXPAND_FULL_TARGET_KEY].append(int(full_target_total))
                    buffers[split_name][_INTEG_EXPAND_FULL_PEAK_IDX_KEY].append(full_peak_idx)
                fragment_target_total = -1
                fragment_peak_idx = None
                if frag_atom_ids is not None and target_atomic_num is not None:
                    fragment_target_total = int(np.sum(atom_numbers[frag_atom_ids_arr] == int(target_atomic_num)))
                    fragment_peak_idx = _precompute_expanded_peak_index(
                        peaks_row,
                        filtered_integ,
                        target_total=fragment_target_total,
                    )
                if fragment_peak_idx is None:
                    buffers[split_name][_INTEG_EXPAND_FRAGMENT_TARGET_KEY].append(-1)
                    buffers[split_name][_INTEG_EXPAND_FRAGMENT_PEAK_IDX_KEY].append(np.zeros((0,), dtype=np.int32))
                else:
                    buffers[split_name][_INTEG_EXPAND_FRAGMENT_TARGET_KEY].append(int(fragment_target_total))
                    buffers[split_name][_INTEG_EXPAND_FRAGMENT_PEAK_IDX_KEY].append(fragment_peak_idx)
                buffers[split_name]["smiles"].append(row.SMILES)
                solvent = getattr(row, "NMR_solvent", None)
                if not isinstance(solvent, str) or not solvent.strip():
                    solvent = "unknown"
                else:
                    solvent = solvent.strip()
                buffers[split_name]["solvent"].append(solvent)
                buffers[split_name]["row_id"].append(int(row.row_id))

                if filtered:
                    update_running_stats(stats, filtered)

                if args.store_weights:
                    weights = boltzmann_weights(
                        rel_energies,
                        temperature=args.temperature,
                        alpha=args.boltzmann_alpha,
                    )
                    buffers[split_name]["weights"].append(weights.astype(np.float32))

                split_counts[split_name] += 1
                total += 1

                if args.shard_size and len(buffers[split_name]["n_atom"]) >= args.shard_size:
                    write_shard(
                        split_dirs[split_name],
                        shard_indices[split_name],
                        buffers[split_name],
                        coord_dtype,
                        args.store_weights,
                        args.precompute_edges,
                    )
                    shard_indices[split_name] += 1
                    buffers[split_name] = init_buffer()

                if args.max_mols and total >= args.max_mols:
                    break

            pbar.update(processed)
            if args.max_mols and total >= args.max_mols:
                break
    finally:
        pbar.close()

    for split_name, buffer in buffers.items():
        write_shard(
            split_dirs[split_name],
            shard_indices[split_name],
            buffer,
            coord_dtype,
            args.store_weights,
            args.precompute_edges,
        )

    peak_mean, peak_std = finalize_stats(stats)
    metadata = {
        "target": args.target,
        "mean": peak_mean,
        "std": peak_std,
        "num_confs": args.num_confs,
        "max_confs": args.max_confs,
        "min_confs": args.min_confs,
        "prune_rms": args.prune_rms,
        "max_embed_tries": args.max_embed_tries,
        "mmff_max_iters": args.mmff_max_iters,
        "allow_uff": args.allow_uff,
        "energy_window": args.energy_window,
        "temperature": args.temperature,
        "boltzmann_alpha": args.boltzmann_alpha,
        "n_neighbors": args.n_neighbors,
        "cutoff": args.cutoff,
        "precomputed_edge_indices": bool(args.precompute_edges),
        "precomputed_integration_expansion": True,
        "precomputed_integration_expansion_keys": {
            "full_target_total": _INTEG_EXPAND_FULL_TARGET_KEY,
            "full_offset": _INTEG_EXPAND_FULL_OFFSET_KEY,
            "full_peak_idx": _INTEG_EXPAND_FULL_PEAK_IDX_KEY,
            "fragment_target_total": _INTEG_EXPAND_FRAGMENT_TARGET_KEY,
            "fragment_offset": _INTEG_EXPAND_FRAGMENT_OFFSET_KEY,
            "fragment_peak_idx": _INTEG_EXPAND_FRAGMENT_PEAK_IDX_KEY,
        },
        "precomputed_fragment_atom_ids": True,
        "precomputed_solvent": True,
        "coord_dtype": args.coord_dtype,
        "store_weights": args.store_weights,
        "require_peaks": args.require_peaks,
        "total_molecules": total,
        "row_groups": args.row_groups,
    }
    if args.manifest_path:
        metadata["manifest_path"] = args.manifest_path
        metadata["manifest_splits"] = order_splits(manifest_splits)

    metadata_path = Path(args.output_dir) / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    if manifest_splits and manifest_splits != {"all"}:
        summary = ", ".join(
            f"{split}={count}" for split, count in split_counts.items()
        )
        print(f"Wrote {total} molecules to {args.output_dir} ({summary})")
    else:
        print(f"Wrote {total} molecules to {args.output_dir}")
    print(f"Saved metadata to {metadata_path}")
    print("Skip summary:")
    print(f"rows_seen={skip_counts['rows_seen']} kept={total}")
    for key in [
        "manifest_skip",
        "wrong_nucleus",
        "parse_failed",
        "no_shifts",
        "all_shifts_filtered",
        "invalid_smiles",
        "too_many_atoms",
        "no_targets",
        "embed_failed",
        "mmff_missing",
        "no_confs",
        "resume_skip",
    ]:
        print(f"{key}={skip_counts[key]}")


if __name__ == "__main__":
    main()
