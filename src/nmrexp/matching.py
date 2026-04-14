"""Capacity-aware matching helpers used by CSP5.

Implements a lightweight substitute for nmrexp.matching so this repo is
self-contained. The matching expands each symmetry environment according to
its capacity (env_counts) and applies a split penalty for additional matches
within the same environment.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple
import ctypes
import os

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:  # pragma: no cover
    linear_sum_assignment = None
    _IMPORT_ERR = exc
else:
    _IMPORT_ERR = None


def env_counts_from_symm(symm: Iterable[int]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for val in symm:
        key = int(val)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _require_scipy():
    if linear_sum_assignment is None:
        raise RuntimeError(
            "scipy is required for matching but is not available: %s" % (_IMPORT_ERR,)
        )


_DP_CPP_FUNC = None
_DP_CPP_BATCH_FUNC = None
_DP_CPP_ERR = None


def _load_dp_cpp():
    global _DP_CPP_FUNC, _DP_CPP_BATCH_FUNC, _DP_CPP_ERR
    if _DP_CPP_FUNC is not None or _DP_CPP_ERR is not None:
        return _DP_CPP_FUNC
    lib_path = os.environ.get("CASCADE_DP_LIB")
    if not lib_path:
        lib_path = os.path.join(os.path.dirname(__file__), "libmatching_dp.so")
    if not os.path.exists(lib_path):
        _DP_CPP_ERR = FileNotFoundError(lib_path)
        return None
    try:
        lib = ctypes.CDLL(lib_path)
        func = lib.nmrexp_match_indices_dp
        func.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        func.restype = ctypes.c_int
        try:
            batch_func = lib.nmrexp_match_indices_dp_batch
        except AttributeError:
            batch_func = None
        if batch_func is not None:
            batch_func.argtypes = [
                ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
            ]
            batch_func.restype = ctypes.c_int
        _DP_CPP_BATCH_FUNC = batch_func
        _DP_CPP_FUNC = func
        return func
    except Exception as exc:  # pragma: no cover
        _DP_CPP_ERR = exc
        return None


def _match_indices_dp_py(
    pred_vals: Sequence[float],
    obs_vals: Sequence[float],
    *,
    dummy_cost: float,
    row_penalties: Sequence[float] | None = None,
) -> Tuple[List[int], List[int]]:
    pred_vals = np.asarray(pred_vals, dtype=float).reshape(-1)
    obs_vals = np.asarray(obs_vals, dtype=float).reshape(-1)
    if pred_vals.size == 0 or obs_vals.size == 0:
        return [], []

    if row_penalties is not None:
        row_penalties = np.asarray(row_penalties, dtype=float).reshape(-1)
        if row_penalties.shape[0] != pred_vals.shape[0]:
            raise ValueError("row_penalties must align with pred_vals")
    else:
        row_penalties = None

    pred_order = np.argsort(pred_vals, kind="mergesort")
    obs_order = np.argsort(obs_vals, kind="mergesort")
    pred_sorted = pred_vals[pred_order]
    obs_sorted = obs_vals[obs_order]
    penalty_sorted = row_penalties[pred_order] if row_penalties is not None else None

    n_pred = pred_sorted.shape[0]
    n_obs = obs_sorted.shape[0]
    dp = np.full((n_pred + 1, n_obs + 1), np.inf, dtype=float)
    back = np.full((n_pred + 1, n_obs + 1), -1, dtype=np.int8)
    dp[0, 0] = 0.0

    allow_skip_pred = n_pred > n_obs
    allow_skip_obs = n_obs > n_pred
    if allow_skip_pred:
        for i in range(1, n_pred + 1):
            dp[i, 0] = dp[i - 1, 0] + float(dummy_cost)
            back[i, 0] = 1
    if allow_skip_obs:
        for j in range(1, n_obs + 1):
            dp[0, j] = dp[0, j - 1] + float(dummy_cost)
            back[0, j] = 2

    for i in range(1, n_pred + 1):
        pred_val = float(pred_sorted[i - 1])
        penalty = float(penalty_sorted[i - 1]) if penalty_sorted is not None else 0.0
        for j in range(1, n_obs + 1):
            best_cost = dp[i - 1, j - 1] + abs(pred_val - float(obs_sorted[j - 1])) + penalty
            best_move = 0
            if allow_skip_pred:
                skip_pred = dp[i - 1, j] + float(dummy_cost)
                if skip_pred < best_cost:
                    best_cost = skip_pred
                    best_move = 1
            if allow_skip_obs:
                skip_obs = dp[i, j - 1] + float(dummy_cost)
                if skip_obs < best_cost:
                    best_cost = skip_obs
                    best_move = 2
            dp[i, j] = best_cost
            back[i, j] = best_move

    rows: List[int] = []
    cols: List[int] = []
    i, j = n_pred, n_obs
    while i > 0 or j > 0:
        move = back[i, j]
        if move == 0:
            rows.append(int(pred_order[i - 1]))
            cols.append(int(obs_order[j - 1]))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    rows.reverse()
    cols.reverse()
    return rows, cols


def match_indices_dp(
    pred_vals: Sequence[float],
    obs_vals: Sequence[float],
    *,
    dummy_cost: float,
    row_penalties: Sequence[float] | None = None,
) -> Tuple[List[int], List[int]]:
    use_cpp = os.environ.get("CASCADE_DP_USE_CPP", "1") != "0"
    if use_cpp:
        func = _load_dp_cpp()
        if func is not None:
            pred_arr = np.asarray(pred_vals, dtype=np.float64).reshape(-1)
            obs_arr = np.asarray(obs_vals, dtype=np.float64).reshape(-1)
            n_pred = int(pred_arr.shape[0])
            n_obs = int(obs_arr.shape[0])
            if n_pred == 0 or n_obs == 0:
                return [], []
            penalties_arr = None
            penalties_ptr = None
            if row_penalties is not None:
                penalties_arr = np.asarray(row_penalties, dtype=np.float64).reshape(-1)
                if penalties_arr.shape[0] != n_pred:
                    raise ValueError("row_penalties must align with pred_vals")
                penalties_ptr = penalties_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            out_size = int(min(n_pred, n_obs))
            out_rows = np.empty(out_size, dtype=np.int32)
            out_cols = np.empty(out_size, dtype=np.int32)
            count = func(
                pred_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                n_pred,
                obs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                n_obs,
                float(dummy_cost),
                penalties_ptr,
                out_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                out_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                out_size,
            )
            if count <= 0:
                return [], []
            return out_rows[:count].tolist(), out_cols[:count].tolist()

    return _match_indices_dp_py(
        pred_vals,
        obs_vals,
        dummy_cost=dummy_cost,
        row_penalties=row_penalties,
    )


def match_indices_dp_batch(
    pred_vals_list: Sequence[Sequence[float]],
    obs_vals_list: Sequence[Sequence[float]],
    *,
    dummy_cost: float,
    row_penalties_list: Sequence[Sequence[float] | None] | None = None,
    n_threads: int = 0,
) -> List[Tuple[List[int], List[int]]]:
    use_cpp = os.environ.get("CASCADE_DP_USE_CPP", "1") != "0"
    if not use_cpp:
        return [
            match_indices_dp(pred, obs, dummy_cost=dummy_cost, row_penalties=penalties)
            for pred, obs, penalties in zip(
                pred_vals_list,
                obs_vals_list,
                row_penalties_list or [None] * len(pred_vals_list),
            )
        ]

    func = _load_dp_cpp()
    batch_func = _DP_CPP_BATCH_FUNC
    if func is None or batch_func is None:
        return [
            match_indices_dp(pred, obs, dummy_cost=dummy_cost, row_penalties=penalties)
            for pred, obs, penalties in zip(
                pred_vals_list,
                obs_vals_list,
                row_penalties_list or [None] * len(pred_vals_list),
            )
        ]

    batch_size = len(pred_vals_list)
    if batch_size == 0:
        return []

    if row_penalties_list is None:
        row_penalties_list = [None] * batch_size

    pred_arrays = []
    obs_arrays = []
    penalties_arrays = []
    n_pred = np.zeros(batch_size, dtype=np.int32)
    n_obs = np.zeros(batch_size, dtype=np.int32)
    out_caps = np.zeros(batch_size, dtype=np.int32)
    out_rows_arrays = []
    out_cols_arrays = []
    out_counts = np.zeros(batch_size, dtype=np.int32)

    for idx, (pred_vals, obs_vals, penalties) in enumerate(
        zip(pred_vals_list, obs_vals_list, row_penalties_list)
    ):
        pred_arr = np.asarray(pred_vals, dtype=np.float64).reshape(-1)
        obs_arr = np.asarray(obs_vals, dtype=np.float64).reshape(-1)
        pred_arrays.append(pred_arr)
        obs_arrays.append(obs_arr)
        n_pred[idx] = int(pred_arr.shape[0])
        n_obs[idx] = int(obs_arr.shape[0])
        out_cap = int(min(n_pred[idx], n_obs[idx]))
        out_caps[idx] = out_cap
        out_rows_arrays.append(np.empty(out_cap, dtype=np.int32))
        out_cols_arrays.append(np.empty(out_cap, dtype=np.int32))

        if penalties is None:
            penalties_arrays.append(None)
        else:
            penalties_arr = np.asarray(penalties, dtype=np.float64).reshape(-1)
            if penalties_arr.shape[0] != pred_arr.shape[0]:
                raise ValueError("row_penalties must align with pred_vals")
            penalties_arrays.append(penalties_arr)

    pred_ptrs = (ctypes.POINTER(ctypes.c_double) * batch_size)()
    obs_ptrs = (ctypes.POINTER(ctypes.c_double) * batch_size)()
    penalty_ptrs = (ctypes.POINTER(ctypes.c_double) * batch_size)()
    out_rows_ptrs = (ctypes.POINTER(ctypes.c_int) * batch_size)()
    out_cols_ptrs = (ctypes.POINTER(ctypes.c_int) * batch_size)()

    for i in range(batch_size):
        pred_ptrs[i] = pred_arrays[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        obs_ptrs[i] = obs_arrays[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if penalties_arrays[i] is None:
            penalty_ptrs[i] = ctypes.cast(None, ctypes.POINTER(ctypes.c_double))
        else:
            penalty_ptrs[i] = penalties_arrays[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_rows_ptrs[i] = out_rows_arrays[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        out_cols_ptrs[i] = out_cols_arrays[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    dummy_costs = np.full(batch_size, float(dummy_cost), dtype=np.float64)

    batch_func(
        pred_ptrs,
        n_pred.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        obs_ptrs,
        n_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dummy_costs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        penalty_ptrs,
        out_rows_ptrs,
        out_cols_ptrs,
        out_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_caps.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(batch_size),
        ctypes.c_int(int(n_threads or 0)),
    )

    results: List[Tuple[List[int], List[int]]] = []
    for i in range(batch_size):
        count = int(out_counts[i])
        if count <= 0:
            results.append(([], []))
        else:
            results.append(
                (
                    out_rows_arrays[i][:count].tolist(),
                    out_cols_arrays[i][:count].tolist(),
                )
            )
    return results


def match_indices_dp_batch_packed(
    pred_vals_list: Sequence[Sequence[float]],
    obs_vals_list: Sequence[Sequence[float]],
    *,
    dummy_cost: float,
    row_penalties_list: Sequence[Sequence[float] | None] | None = None,
    n_threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_size = len(pred_vals_list)
    if batch_size == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((1,), dtype=np.int64),
        )

    use_cpp = os.environ.get("CASCADE_DP_USE_CPP", "1") != "0"
    if row_penalties_list is None:
        row_penalties_list = [None] * batch_size

    # If C++ backend is unavailable, reuse existing Python/C++ mixed path and
    # pack the results once to avoid per-call list conversions downstream.
    if not use_cpp:
        results = match_indices_dp_batch(
            pred_vals_list,
            obs_vals_list,
            dummy_cost=dummy_cost,
            row_penalties_list=row_penalties_list,
            n_threads=n_threads,
        )
        offsets = np.zeros((batch_size + 1,), dtype=np.int64)
        counts = np.asarray([len(rows) for rows, _ in results], dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        n_pair = int(offsets[-1])
        rows_flat = np.empty((n_pair,), dtype=np.int32)
        cols_flat = np.empty((n_pair,), dtype=np.int32)
        cursor = 0
        for rows, cols in results:
            n = len(rows)
            if n <= 0:
                continue
            rows_flat[cursor : cursor + n] = np.asarray(rows, dtype=np.int32)
            cols_flat[cursor : cursor + n] = np.asarray(cols, dtype=np.int32)
            cursor += n
        return rows_flat, cols_flat, offsets

    func = _load_dp_cpp()
    batch_func = _DP_CPP_BATCH_FUNC
    if func is None or batch_func is None:
        results = match_indices_dp_batch(
            pred_vals_list,
            obs_vals_list,
            dummy_cost=dummy_cost,
            row_penalties_list=row_penalties_list,
            n_threads=n_threads,
        )
        offsets = np.zeros((batch_size + 1,), dtype=np.int64)
        counts = np.asarray([len(rows) for rows, _ in results], dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        n_pair = int(offsets[-1])
        rows_flat = np.empty((n_pair,), dtype=np.int32)
        cols_flat = np.empty((n_pair,), dtype=np.int32)
        cursor = 0
        for rows, cols in results:
            n = len(rows)
            if n <= 0:
                continue
            rows_flat[cursor : cursor + n] = np.asarray(rows, dtype=np.int32)
            cols_flat[cursor : cursor + n] = np.asarray(cols, dtype=np.int32)
            cursor += n
        return rows_flat, cols_flat, offsets

    pred_arrays = []
    obs_arrays = []
    penalties_arrays = []
    n_pred = np.zeros(batch_size, dtype=np.int32)
    n_obs = np.zeros(batch_size, dtype=np.int32)
    out_caps = np.zeros(batch_size, dtype=np.int32)
    out_rows_arrays = []
    out_cols_arrays = []
    out_counts = np.zeros(batch_size, dtype=np.int32)

    for idx, (pred_vals, obs_vals, penalties) in enumerate(
        zip(pred_vals_list, obs_vals_list, row_penalties_list)
    ):
        pred_arr = np.asarray(pred_vals, dtype=np.float64).reshape(-1)
        obs_arr = np.asarray(obs_vals, dtype=np.float64).reshape(-1)
        pred_arrays.append(pred_arr)
        obs_arrays.append(obs_arr)
        n_pred[idx] = int(pred_arr.shape[0])
        n_obs[idx] = int(obs_arr.shape[0])
        out_cap = int(min(n_pred[idx], n_obs[idx]))
        out_caps[idx] = out_cap
        out_rows_arrays.append(np.empty(out_cap, dtype=np.int32))
        out_cols_arrays.append(np.empty(out_cap, dtype=np.int32))

        if penalties is None:
            penalties_arrays.append(None)
        else:
            penalties_arr = np.asarray(penalties, dtype=np.float64).reshape(-1)
            if penalties_arr.shape[0] != pred_arr.shape[0]:
                raise ValueError("row_penalties must align with pred_vals")
            penalties_arrays.append(penalties_arr)

    pred_ptrs = (ctypes.POINTER(ctypes.c_double) * batch_size)()
    obs_ptrs = (ctypes.POINTER(ctypes.c_double) * batch_size)()
    penalty_ptrs = (ctypes.POINTER(ctypes.c_double) * batch_size)()
    out_rows_ptrs = (ctypes.POINTER(ctypes.c_int) * batch_size)()
    out_cols_ptrs = (ctypes.POINTER(ctypes.c_int) * batch_size)()

    for i in range(batch_size):
        pred_ptrs[i] = pred_arrays[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        obs_ptrs[i] = obs_arrays[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if penalties_arrays[i] is None:
            penalty_ptrs[i] = ctypes.cast(None, ctypes.POINTER(ctypes.c_double))
        else:
            penalty_ptrs[i] = penalties_arrays[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_rows_ptrs[i] = out_rows_arrays[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        out_cols_ptrs[i] = out_cols_arrays[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    dummy_costs = np.full(batch_size, float(dummy_cost), dtype=np.float64)

    batch_func(
        pred_ptrs,
        n_pred.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        obs_ptrs,
        n_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dummy_costs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        penalty_ptrs,
        out_rows_ptrs,
        out_cols_ptrs,
        out_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_caps.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(batch_size),
        ctypes.c_int(int(n_threads or 0)),
    )

    out_counts = np.clip(out_counts.astype(np.int64, copy=False), 0, out_caps.astype(np.int64, copy=False))
    offsets = np.zeros((batch_size + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(out_counts, dtype=np.int64)
    n_pair = int(offsets[-1])
    rows_flat = np.empty((n_pair,), dtype=np.int32)
    cols_flat = np.empty((n_pair,), dtype=np.int32)
    cursor = 0
    for i in range(batch_size):
        count = int(out_counts[i])
        if count <= 0:
            continue
        rows_flat[cursor : cursor + count] = out_rows_arrays[i][:count]
        cols_flat[cursor : cursor + count] = out_cols_arrays[i][:count]
        cursor += count
    return rows_flat, cols_flat, offsets


def match_indices_with_capacity(
    pred_vals: Sequence[float],
    obs_vals: Sequence[float],
    *,
    dummy_cost: float,
    pred_env_ids: Sequence[int] | None,
    env_counts: Dict[int, int] | None,
    split_penalty: float = 0.0,
    solver: str = "scipy",
    pairwise_cost: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> Tuple[List[int], List[int]]:

    pred_vals = np.asarray(pred_vals, dtype=float).reshape(-1)
    obs_vals = np.asarray(obs_vals, dtype=float).reshape(-1)
    pairwise_cost_arr: np.ndarray | None = None
    if pairwise_cost is not None:
        pairwise_cost_arr = np.asarray(pairwise_cost, dtype=float)
        expected_shape = (int(pred_vals.shape[0]), int(obs_vals.shape[0]))
        if pairwise_cost_arr.shape != expected_shape:
            raise ValueError(
                "pairwise_cost must have shape [n_pred, n_obs], "
                f"got {pairwise_cost_arr.shape} vs expected {expected_shape}"
            )

    if not np.isfinite(pred_vals).all() or not np.isfinite(obs_vals).all():
        pred_mask = np.isfinite(pred_vals)
        obs_mask = np.isfinite(obs_vals)
        if pred_env_ids is not None:
            pred_env_ids = np.asarray(pred_env_ids, dtype=int).reshape(-1)
            if pred_env_ids.shape[0] != pred_vals.shape[0]:
                raise ValueError("pred_env_ids must align with pred_vals")
            pred_env_ids = pred_env_ids[pred_mask]
        if pairwise_cost_arr is not None:
            pairwise_cost_arr = pairwise_cost_arr[np.ix_(pred_mask, obs_mask)]
        pred_vals = pred_vals[pred_mask]
        obs_vals = obs_vals[obs_mask]

    if pred_vals.size == 0 or obs_vals.size == 0:
        return [], []

    solver = str(solver or "scipy").lower()

    if pred_env_ids is None or env_counts is None:
        # Fallback to standard Hungarian matching.
        if solver == "dp" and pairwise_cost_arr is None:
            return match_indices_dp(pred_vals, obs_vals, dummy_cost=dummy_cost)
        _require_scipy()
        if pairwise_cost_arr is None:
            cost = np.abs(pred_vals[:, None] - obs_vals[None, :])
        else:
            cost = pairwise_cost_arr.copy()
        n_pred, n_obs = cost.shape
        if n_pred > n_obs:
            pad = np.full((n_pred, n_pred - n_obs), float(dummy_cost))
            cost = np.concatenate([cost, pad], axis=1)
        elif n_obs > n_pred:
            pad = np.full((n_obs - n_pred, n_obs), float(dummy_cost))
            cost = np.concatenate([cost, pad], axis=0)
        row_ind, col_ind = linear_sum_assignment(cost)
        rows = []
        cols = []
        for r, c in zip(row_ind, col_ind):
            if r < pred_vals.size and c < obs_vals.size:
                rows.append(int(r))
                cols.append(int(c))
        return rows, cols

    pred_env_ids = np.asarray(pred_env_ids, dtype=int).reshape(-1)
    if pred_env_ids.shape[0] != pred_vals.shape[0]:
        raise ValueError("pred_env_ids must align with pred_vals")

    expanded_vals: List[float] = []
    expanded_idx: List[int] = []
    penalties: List[float] = []

    for i, env_id in enumerate(pred_env_ids.tolist()):
        capacity = int(env_counts.get(int(env_id), 1)) if env_counts else 1
        capacity = max(1, capacity)
        for copy_idx in range(capacity):
            expanded_vals.append(float(pred_vals[i]))
            expanded_idx.append(int(i))
            penalties.append(float(split_penalty) if copy_idx > 0 else 0.0)

    expanded_vals_arr = np.asarray(expanded_vals, dtype=float)
    penalties_arr = np.asarray(penalties, dtype=float)

    n_pred = expanded_vals_arr.shape[0]
    n_obs = obs_vals.shape[0]

    if solver == "dp" and pairwise_cost_arr is None:
        rows_exp, cols = match_indices_dp(
            expanded_vals_arr,
            obs_vals,
            dummy_cost=dummy_cost,
            row_penalties=penalties_arr,
        )
    else:
        _require_scipy()
        if pairwise_cost_arr is None:
            cost = np.abs(expanded_vals_arr[:, None] - obs_vals[None, :])
        else:
            cost = pairwise_cost_arr[np.asarray(expanded_idx, dtype=np.int64), :].copy()
        if penalties_arr.size:
            cost = cost + penalties_arr[:, None]
        if n_pred > n_obs:
            pad = np.full((n_pred, n_pred - n_obs), float(dummy_cost))
            cost = np.concatenate([cost, pad], axis=1)
        elif n_obs > n_pred:
            pad = np.full((n_obs - n_pred, n_obs), float(dummy_cost))
            cost = np.concatenate([cost, pad], axis=0)
        row_ind, col_ind = linear_sum_assignment(cost)
        rows_exp = []
        cols = []
        for r, c in zip(row_ind, col_ind):
            if r < n_pred and c < n_obs:
                rows_exp.append(int(r))
                cols.append(int(c))

    rows: List[int] = []
    out_cols: List[int] = []
    for r, c in zip(rows_exp, cols):
        if r < n_pred and c < n_obs:
            rows.append(expanded_idx[int(r)])
            out_cols.append(int(c))
    return rows, out_cols
