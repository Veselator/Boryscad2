"""Helpers shared by library modules."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..engine.values import EvalError, RangeValue, is_number, normalize_number, to_float, to_int


def vec(value: Any, what: str = "argument", dtype=float) -> np.ndarray:
    if isinstance(value, RangeValue):
        value = value.to_array()
    if is_number(value):
        return np.array([value], dtype=dtype)
    if isinstance(value, (list, tuple)):
        value = np.array(value)
    if not isinstance(value, np.ndarray):
        raise EvalError(f"{what} must be a vector")
    if value.ndim == 2 and 1 in value.shape:
        value = value.reshape(-1)
    if value.ndim != 1:
        raise EvalError(f"{what} must be a vector, got a matrix")
    try:
        return value.astype(dtype)
    except (TypeError, ValueError):
        raise EvalError(f"{what} must contain numbers") from None


def mat(value: Any, what: str = "argument", dtype=float) -> np.ndarray:
    if isinstance(value, RangeValue):
        value = value.to_array()
    if is_number(value):
        return np.array([[value]], dtype=dtype)
    if not isinstance(value, np.ndarray):
        raise EvalError(f"{what} must be a matrix")
    if value.ndim == 1:
        value = value.reshape(1, -1) if what.endswith("row") else value.reshape(-1, 1)
    try:
        return value.astype(dtype)
    except (TypeError, ValueError):
        raise EvalError(f"{what} must contain numbers") from None


def strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        return [str(normalize_number(v)) for v in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise EvalError("expected a vector of strings")


def origin_of(ctx) -> int:
    return ctx.origin if ctx is not None else 0


def clean(value: float, tol: float = 1e-9) -> float:
    """Round tiny numeric noise: 2.9999999999 → 3."""
    r = round(value)
    if abs(value - r) < tol:
        return float(r)
    return value


def clean_array(arr: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    out = np.array(arr, dtype=float)
    rounded = np.round(out)
    mask = np.abs(out - rounded) < tol
    out[mask] = rounded[mask]
    out[np.abs(out) < tol] = 0.0
    return out


def as_int_if_whole(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "f" and np.all(np.isfinite(arr)) and np.allclose(arr, np.round(arr)):
        return np.round(arr).astype(np.int64)
    return arr


__all__: Sequence[str] = ["vec", "mat", "strings", "origin_of", "clean", "clean_array", "as_int_if_whole",
                          "to_float", "to_int", "EvalError"]
