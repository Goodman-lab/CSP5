from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem


MODELS_DIR = Path(__file__).resolve().parents[1]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from torch_training import _collate_graphs  # noqa: E402


KCAL_PER_MOL_KT = 0.0019872041

_INTEG_EXPAND_FULL_TARGET_KEY = "integration_expand_full_target_total"
_INTEG_EXPAND_FULL_OFFSET_KEY = "integration_expand_full_offset"
_INTEG_EXPAND_FULL_PEAK_IDX_KEY = "integration_expand_full_peak_idx"
_INTEG_EXPAND_FRAGMENT_TARGET_KEY = "integration_expand_fragment_target_total"
_INTEG_EXPAND_FRAGMENT_OFFSET_KEY = "integration_expand_fragment_offset"
_INTEG_EXPAND_FRAGMENT_PEAK_IDX_KEY = "integration_expand_fragment_peak_idx"


def normalize_solvent_name(raw: object) -> str:
    if not isinstance(raw, str):
        return "unknown"
    s = raw.strip()
    return s if s else "unknown"


def normalize_solvent_key(raw: object) -> str:
    return normalize_solvent_name(raw).lower()


def _normalize_solvent_filter(solvent_filter: set[str] | None) -> set[str]:
    if not solvent_filter:
        return set()
    return {normalize_solvent_key(s) for s in solvent_filter}


def _load_ragged_peak_index(
    data: np.lib.npyio.NpzFile,
    *,
    target_key: str,
    offset_key: str,
    peak_idx_key: str,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if target_key not in data.files or offset_key not in data.files or peak_idx_key not in data.files:
        return None
    targets = np.asarray(data[target_key], dtype=np.int32).reshape(-1)
    offsets = np.asarray(data[offset_key], dtype=np.int64).reshape(-1)
    peak_idx = np.asarray(data[peak_idx_key], dtype=np.int32).reshape(-1)
    if int(targets.shape[0]) != int(n_rows):
        return None
    if int(offsets.shape[0]) != int(n_rows) + 1:
        return None
    if int(offsets[0]) != 0:
        return None
    if not np.all(offsets[1:] >= offsets[:-1]):
        return None
    if int(offsets[-1]) != int(peak_idx.shape[0]):
        return None
    return targets, offsets, peak_idx


def _select_largest_fragment_atom_ids(
    smiles: str,
    atom_numbers: np.ndarray,
) -> np.ndarray | None:
    if "." not in smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if mol is None:
        return None

    if mol.GetNumAtoms() != int(atom_numbers.shape[0]):
        return None

    rdkit_atom_numbers = np.fromiter(
        (int(atom.GetAtomicNum()) for atom in mol.GetAtoms()),
        dtype=np.int64,
    )
    if not np.array_equal(rdkit_atom_numbers, atom_numbers):
        return None

    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    return _select_largest_fragment_atom_ids_from_fragments(
        rdkit_atom_numbers, frags
    )


def _select_largest_fragment_atom_ids_from_fragments(
    atom_numbers: np.ndarray,
    frags,
) -> np.ndarray | None:
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


class NMRexpShardDataset(Dataset):
    def __init__(
        self,
        shard_path: Path,
        *,
        solvent_filter: set[str] | None = None,
    ) -> None:
        data = np.load(shard_path, allow_pickle=True)
        self.n_atom = data["n_atom"].astype(np.int64)
        self.n_edge = data["n_edge"].astype(np.int64)
        self.n_pro = data["n_pro"].astype(np.int64)
        self.node_attributes = data["node_attributes"].astype(np.int64)
        self.node_coordinates = data["node_coordinates"].astype(np.float32)
        self.edge_indices = data["edge_indices"].astype(np.int64)
        self.atom_index = data["atom_index"].astype(np.int64)
        self.symm = data["symm"].astype(np.int64)
        self.h_counts = data["h_counts"].astype(np.int64)
        self.peaks = data["peaks"].tolist()
        self.integrations = data["integrations"].tolist()
        self.smiles = data["smiles"].tolist()
        if "solvent" in data.files:
            self.solvent = [normalize_solvent_name(v) for v in data["solvent"].tolist()]
        else:
            self.solvent = ["unknown"] * int(self.n_atom.shape[0])
        self.solvent_filter = _normalize_solvent_filter(solvent_filter)

        self.atom_csum = np.concatenate([[0], np.cumsum(self.n_atom)])
        self.edge_csum = np.concatenate([[0], np.cumsum(self.n_edge)])

    def __len__(self) -> int:
        return int(self.n_atom.shape[0])

    def __getitem__(self, idx: int):
        atom_start = int(self.atom_csum[idx])
        atom_end = int(self.atom_csum[idx + 1])
        edge_start = int(self.edge_csum[idx])
        edge_end = int(self.edge_csum[idx + 1])

        graph = {
            "n_atom": int(self.n_atom[idx]),
            "n_pro": int(self.n_pro[idx]),
            "node_attributes": self.node_attributes[atom_start:atom_end],
            "node_coordinates": self.node_coordinates[atom_start:atom_end],
            "edge_indices": self.edge_indices[edge_start:edge_end],
            "atom_index": self.atom_index[atom_start:atom_end],
        }

        symm = self.symm[atom_start:atom_end]
        h_counts = self.h_counts[atom_start:atom_end]
        peaks = self.peaks[idx]
        integrations = self.integrations[idx]
        smiles = self.smiles[idx]
        solvent = self.solvent[idx]
        if self.solvent_filter and normalize_solvent_key(solvent) not in self.solvent_filter:
            return None
        return graph, symm, h_counts, peaks, integrations, smiles, solvent


def _build_edge_indices_from_coords(
    coords: np.ndarray,
    n_neighbors: int,
    cutoff: float,
) -> np.ndarray:
    n_atom = int(coords.shape[0])
    if n_atom <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    if n_atom == 1:
        return np.array([[0, 0]], dtype=np.int64)

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
    return np.asarray(edge_indices, dtype=np.int64)


def _ensure_node_coverage_edges(edge_indices: np.ndarray, n_atom: int) -> np.ndarray:
    if n_atom <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    if edge_indices.size == 0:
        nodes = np.arange(n_atom, dtype=np.int64)
        return np.stack([nodes, nodes], axis=1)

    edges = np.asarray(edge_indices, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        nodes = np.arange(n_atom, dtype=np.int64)
        return np.stack([nodes, nodes], axis=1)

    valid = (
        (edges[:, 0] >= 0)
        & (edges[:, 0] < n_atom)
        & (edges[:, 1] >= 0)
        & (edges[:, 1] < n_atom)
    )
    edges = edges[valid]
    if edges.size == 0:
        nodes = np.arange(n_atom, dtype=np.int64)
        return np.stack([nodes, nodes], axis=1)

    has_src = np.zeros((n_atom,), dtype=bool)
    has_src[edges[:, 0]] = True
    missing = np.flatnonzero(~has_src)
    if missing.size > 0:
        self_edges = np.stack([missing, missing], axis=1).astype(np.int64, copy=False)
        edges = np.concatenate([edges, self_edges], axis=0)
    return edges


def _remap_edge_indices_to_fragment(
    edge_indices: np.ndarray,
    frag_atom_ids: np.ndarray,
    full_n_atom: int,
) -> np.ndarray:
    n_frag = int(frag_atom_ids.shape[0])
    if n_frag <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    remap = np.full((int(full_n_atom),), -1, dtype=np.int64)
    remap[frag_atom_ids] = np.arange(n_frag, dtype=np.int64)

    edges = np.asarray(edge_indices, dtype=np.int64)
    if edges.size == 0:
        return _ensure_node_coverage_edges(edges, n_frag)

    valid = (
        (edges[:, 0] >= 0)
        & (edges[:, 0] < full_n_atom)
        & (edges[:, 1] >= 0)
        & (edges[:, 1] < full_n_atom)
    )
    edges = edges[valid]
    if edges.size == 0:
        return _ensure_node_coverage_edges(edges, n_frag)

    src_new = remap[edges[:, 0]]
    dst_new = remap[edges[:, 1]]
    keep = (src_new >= 0) & (dst_new >= 0)
    if not np.any(keep):
        return _ensure_node_coverage_edges(np.zeros((0, 2), dtype=np.int64), n_frag)

    remapped = np.stack([src_new[keep], dst_new[keep]], axis=1)
    return _ensure_node_coverage_edges(remapped, n_frag)


def _boltzmann_weights(rel_energies: np.ndarray, temperature: float, alpha: float) -> np.ndarray:
    if rel_energies.size == 0:
        return np.array([], dtype=np.float64)
    if temperature <= 0 or alpha <= 0:
        return np.full(rel_energies.shape, 1.0 / float(rel_energies.size), dtype=np.float64)
    kT = KCAL_PER_MOL_KT * temperature
    beta = 1.0 / (alpha * kT)
    weights = np.exp(-rel_energies * beta)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return np.full(rel_energies.shape, 1.0 / float(rel_energies.size), dtype=np.float64)
    return weights / total


class NMRexpEnsembleShardDataset(Dataset):
    def __init__(
        self,
        shard_path: Path,
        *,
        target: str,
        sampling: str = "boltzmann",
        temperature: float = 298.15,
        boltzmann_alpha: float = 1.0,
        n_neighbors: int = 20,
        cutoff: float = 5.0,
        seed: int = 0,
        atom_token_map: dict[int, int] | None = None,
        atom_token_unk: int = 1,
        disable_fragment_filter: bool = False,
        min_conformer_distance: float = 0.0,
        drop_smiles: set[str] | None = None,
        row_solvents: Sequence[object] | None = None,
        solvent_filter: set[str] | None = None,
    ) -> None:
        data = np.load(shard_path, allow_pickle=True)
        self.n_atom = data["n_atom"].astype(np.int64)
        self.n_conf = data["n_conf"].astype(np.int64)
        self.atom_numbers = data["atom_numbers"].astype(np.int64)
        self.coords = data["coords"].astype(np.float32)
        self.rel_energies = data["rel_energies"].astype(np.float32)
        self.symm = data["symm"].astype(np.int64)
        self.h_counts = data["h_counts"].astype(np.int64)
        self.peaks = data["peaks"].tolist()
        self.integrations = data["integrations"].tolist()
        self.smiles = data["smiles"].tolist()

        self.weights = data["weights"].astype(np.float32) if "weights" in data.files else None
        self.solvent = (
            [normalize_solvent_name(v) for v in data["solvent"].tolist()]
            if "solvent" in data.files
            else None
        )
        self.row_id = data["row_id"].astype(np.int64) if "row_id" in data.files else None
        self.n_edge_per_conf = data["n_edge_per_conf"].astype(np.int64) if "n_edge_per_conf" in data.files else None
        self.edge_indices = data["edge_indices"].astype(np.int64) if "edge_indices" in data.files else None
        self.fragment_n_atom = data["fragment_n_atom"].astype(np.int64) if "fragment_n_atom" in data.files else None
        self.fragment_atom_ids = data["fragment_atom_ids"].astype(np.int64) if "fragment_atom_ids" in data.files else None
        self.integration_expand_full = _load_ragged_peak_index(
            data,
            target_key=_INTEG_EXPAND_FULL_TARGET_KEY,
            offset_key=_INTEG_EXPAND_FULL_OFFSET_KEY,
            peak_idx_key=_INTEG_EXPAND_FULL_PEAK_IDX_KEY,
            n_rows=int(self.n_atom.shape[0]),
        )
        self.integration_expand_fragment = _load_ragged_peak_index(
            data,
            target_key=_INTEG_EXPAND_FRAGMENT_TARGET_KEY,
            offset_key=_INTEG_EXPAND_FRAGMENT_OFFSET_KEY,
            peak_idx_key=_INTEG_EXPAND_FRAGMENT_PEAK_IDX_KEY,
            n_rows=int(self.n_atom.shape[0]),
        )

        self.atom_csum = np.concatenate([[0], np.cumsum(self.n_atom)])
        self.conf_csum = np.concatenate([[0], np.cumsum(self.n_conf)])
        self.coord_csum = np.concatenate([[0], np.cumsum(self.n_atom * self.n_conf)])
        self.edge_csum = None
        if self.n_edge_per_conf is not None and self.edge_indices is not None:
            expected_conf = int(np.sum(self.n_conf))
            if int(self.n_edge_per_conf.shape[0]) == expected_conf:
                self.edge_csum = np.concatenate([[0], np.cumsum(self.n_edge_per_conf)])
            else:
                self.n_edge_per_conf = None
                self.edge_indices = None
        self.fragment_csum = None
        if self.fragment_n_atom is not None and self.fragment_atom_ids is not None:
            if int(self.fragment_n_atom.shape[0]) == int(self.n_atom.shape[0]):
                self.fragment_csum = np.concatenate([[0], np.cumsum(self.fragment_n_atom)])
            else:
                self.fragment_n_atom = None
                self.fragment_atom_ids = None

        self.target_atomic_num = 6 if str(target).upper() == "13C" else 1
        self.sampling = str(sampling).lower()
        self.temperature = float(temperature)
        self.boltzmann_alpha = float(boltzmann_alpha)
        self.n_neighbors = int(n_neighbors)
        self.cutoff = float(cutoff)
        self.seed = int(seed)
        self.epoch = 0
        self.atom_token_map = atom_token_map
        self.atom_token_unk = int(atom_token_unk)
        self.disable_fragment_filter = bool(disable_fragment_filter)
        self.min_conformer_distance = float(min_conformer_distance)
        self.drop_smiles = set(drop_smiles) if drop_smiles else set()
        self._fragment_atom_ids_cache: dict[int, np.ndarray | None] = {}
        self.solvent_filter = _normalize_solvent_filter(solvent_filter)
        if row_solvents is None:
            self.row_solvents = None
        else:
            arr = np.asarray(row_solvents, dtype=object)
            if int(arr.shape[0]) != int(self.n_atom.shape[0]):
                raise ValueError(
                    "row_solvents length mismatch: "
                    f"{int(arr.shape[0])} != {int(self.n_atom.shape[0])}"
                )
            self.row_solvents = [normalize_solvent_name(v) for v in arr.tolist()]
        if self.solvent is not None and int(len(self.solvent)) != int(self.n_atom.shape[0]):
            raise ValueError(
                "solvent length mismatch: "
                f"{int(len(self.solvent))} != {int(self.n_atom.shape[0])}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.n_atom.shape[0])

    def _integration_expand_for_row(
        self,
        idx: int,
        *,
        use_fragment: bool,
    ) -> tuple[int, np.ndarray] | None:
        source = self.integration_expand_fragment if use_fragment else self.integration_expand_full
        if source is None:
            return None
        targets, offsets, peak_idx = source
        target_total = int(targets[int(idx)])
        if target_total <= 0:
            return None
        start = int(offsets[int(idx)])
        end = int(offsets[int(idx) + 1])
        if end <= start:
            return None
        idx_slice = peak_idx[start:end]
        if int(idx_slice.shape[0]) <= 0:
            return None
        return target_total, idx_slice

    def _sample_conf_idx(self, idx: int) -> int:
        n_conf = int(self.n_conf[idx])
        if n_conf <= 1 or self.sampling == "lowest":
            return 0
        if self.sampling == "uniform":
            rng = np.random.default_rng(self.seed + self.epoch * 1000003 + int(idx))
            return int(rng.integers(0, n_conf))

        conf_start = int(self.conf_csum[idx])
        conf_end = int(self.conf_csum[idx + 1])
        if self.weights is not None:
            probs = np.asarray(self.weights[conf_start:conf_end], dtype=np.float64)
        else:
            probs = _boltzmann_weights(
                np.asarray(self.rel_energies[conf_start:conf_end], dtype=np.float64),
                temperature=self.temperature,
                alpha=self.boltzmann_alpha,
            )
        if probs.size != n_conf:
            return 0
        probs = np.where(np.isfinite(probs), probs, 0.0)
        s = float(np.sum(probs))
        if s <= 0:
            return 0
        probs = probs / s
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + int(idx))
        return int(rng.choice(n_conf, p=probs))

    def __getitem__(self, idx: int):
        n_atom = int(self.n_atom[idx])
        full_n_atom = int(n_atom)
        conf_idx = self._sample_conf_idx(idx)

        atom_start = int(self.atom_csum[idx])
        atom_end = int(self.atom_csum[idx + 1])
        coord_start = int(self.coord_csum[idx]) + conf_idx * n_atom
        coord_end = coord_start + n_atom

        atom_numbers = self.atom_numbers[atom_start:atom_end]
        coords = self.coords[coord_start:coord_end]
        precomputed_edges = None
        if self.edge_csum is not None and self.edge_indices is not None:
            conf_global_idx = int(self.conf_csum[idx]) + int(conf_idx)
            if 0 <= conf_global_idx < int(self.n_edge_per_conf.shape[0]):
                edge_start = int(self.edge_csum[conf_global_idx])
                edge_end = int(self.edge_csum[conf_global_idx + 1])
                precomputed_edges = self.edge_indices[edge_start:edge_end]

        smiles = self.smiles[idx]
        if self.row_solvents is not None:
            solvent = self.row_solvents[idx]
        elif self.solvent is not None:
            solvent = self.solvent[idx]
        else:
            solvent = "unknown"
        if self.solvent_filter and normalize_solvent_key(solvent) not in self.solvent_filter:
            return None
        if self.drop_smiles and smiles in self.drop_smiles:
            return None
        frag_atom_ids = None
        if not self.disable_fragment_filter:
            if (
                self.fragment_csum is not None
                and self.fragment_n_atom is not None
                and self.fragment_atom_ids is not None
            ):
                n_frag = int(self.fragment_n_atom[idx])
                if 0 < n_frag < int(atom_numbers.shape[0]):
                    frag_start = int(self.fragment_csum[idx])
                    frag_end = int(self.fragment_csum[idx + 1])
                    frag_atom_ids = self.fragment_atom_ids[frag_start:frag_end]
            else:
                if int(idx) in self._fragment_atom_ids_cache:
                    frag_atom_ids = self._fragment_atom_ids_cache[int(idx)]
                else:
                    frag_atom_ids = _select_largest_fragment_atom_ids(str(smiles), atom_numbers)
                    self._fragment_atom_ids_cache[int(idx)] = frag_atom_ids

        if frag_atom_ids is not None:
            atom_numbers = atom_numbers[frag_atom_ids]
            coords = coords[frag_atom_ids]
            n_atom = int(atom_numbers.shape[0])

        target_atoms = np.flatnonzero(atom_numbers == self.target_atomic_num).astype(np.int64)
        n_pro = int(target_atoms.shape[0])

        atom_index = np.full(n_atom, -1, dtype=np.int64)
        atom_index[target_atoms] = np.arange(n_pro, dtype=np.int64)
        if precomputed_edges is not None:
            if frag_atom_ids is not None:
                edge_indices = _remap_edge_indices_to_fragment(
                    precomputed_edges,
                    frag_atom_ids=frag_atom_ids,
                    full_n_atom=full_n_atom,
                )
            else:
                edge_indices = _ensure_node_coverage_edges(precomputed_edges, n_atom)
        else:
            edge_indices = _build_edge_indices_from_coords(
                coords,
                n_neighbors=self.n_neighbors,
                cutoff=self.cutoff,
            )

        # Guard against malformed conformers (e.g., near-overlapping atoms),
        # which can cause PaiNN radial features to explode numerically.
        if self.min_conformer_distance > 0 and int(n_atom) > 1:
            edges = np.asarray(edge_indices, dtype=np.int64)
            if edges.ndim == 2 and edges.shape[1] == 2 and edges.size > 0:
                nonself = edges[:, 0] != edges[:, 1]
                if np.any(nonself):
                    e = edges[nonself]
                    diffs = coords[e[:, 0]] - coords[e[:, 1]]
                    d2 = np.sum(diffs * diffs, axis=1)
                    if d2.size > 0 and float(np.min(d2)) < float(self.min_conformer_distance ** 2):
                        return None

        graph = {
            "n_atom": int(n_atom),
            "n_pro": int(n_pro),
            "node_attributes": (
                np.array(
                    [self.atom_token_map.get(int(z), self.atom_token_unk) for z in atom_numbers],
                    dtype=np.int64,
                )
                if self.atom_token_map is not None
                else atom_numbers
            ),
            "node_coordinates": coords,
            "edge_indices": edge_indices,
            "atom_index": atom_index,
        }
        if frag_atom_ids is not None:
            graph["integration_expand_precomputed"] = (
                self._integration_expand_for_row(int(idx), use_fragment=True)
                or self._integration_expand_for_row(int(idx), use_fragment=False)
            )
        else:
            graph["integration_expand_precomputed"] = self._integration_expand_for_row(
                int(idx),
                use_fragment=False,
            )
        symm = self.symm[atom_start:atom_end]
        h_counts = self.h_counts[atom_start:atom_end]
        if frag_atom_ids is not None:
            symm = symm[frag_atom_ids]
            h_counts = h_counts[frag_atom_ids]
        peaks = self.peaks[idx]
        integrations = self.integrations[idx]
        return graph, symm, h_counts, peaks, integrations, smiles, solvent


def collate_unassigned(
    batch: Sequence[Tuple[dict, np.ndarray, np.ndarray, Sequence[float], Sequence[float], str, str]]
):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    graphs = [item[0] for item in batch]
    symm_list = [item[1] for item in batch]
    h_counts_list = [item[2] for item in batch]
    peaks = [item[3] for item in batch]
    integrations = [item[4] for item in batch]
    smiles = [item[5] for item in batch]
    solvents = [item[6] for item in batch]

    batch_graph = _collate_graphs(graphs)
    batch_graph["integration_expand_precomputed"] = [
        graph.get("integration_expand_precomputed") for graph in graphs
    ]
    n_atoms = np.array([graph["n_atom"] for graph in graphs], dtype=np.int64)
    n_pros = np.array([graph["n_pro"] for graph in graphs], dtype=np.int64)
    symm = np.concatenate(symm_list, axis=0)
    h_counts = np.concatenate(h_counts_list, axis=0)

    return batch_graph, n_atoms, n_pros, symm, h_counts, peaks, integrations, smiles, solvents
