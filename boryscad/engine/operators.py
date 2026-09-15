"""Arithmetic semantics of the language (Mathcad-like conventions on top of numpy)."""
from __future__ import annotations

import math
import operator
from typing import Any

import numpy as np

from .values import EvalError, RangeValue, is_number, normalize_number


def _arr(value: Any) -> Any:
    if isinstance(value, RangeValue):
        return value.to_array()
    if isinstance(value, (list, tuple)):
        return np.array(value)
    return value


def _check_numeric(value: Any, op: str) -> None:
    if isinstance(value, str):
        if op == "+":
            return
        raise EvalError(f"operator '{op}' cannot be applied to a string")
    if isinstance(value, np.ndarray) and value.dtype == object:
        return
    if not (is_number(value) or isinstance(value, np.ndarray)):
        raise EvalError(f"operator '{op}' cannot be applied to {type(value).__name__}")


def _result(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return normalize_number(value.item())
        return value
    return normalize_number(value)


def add(a: Any, b: Any) -> Any:
    a, b = _arr(a), _arr(b)
    if isinstance(a, str) and isinstance(b, str):
        return a + b
    _check_numeric(a, "+"); _check_numeric(b, "+")
    _check_shapes(a, b, "+")
    return _result(np.add(a, b) if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) else a + b)


def sub(a: Any, b: Any) -> Any:
    a, b = _arr(a), _arr(b)
    _check_numeric(a, "-"); _check_numeric(b, "-")
    _check_shapes(a, b, "-")
    return _result(np.subtract(a, b) if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) else a - b)


def _check_shapes(a: Any, b: Any, op: str) -> None:
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape != b.shape:
        if a.ndim == 1 and b.ndim == 2 and 1 in b.shape and b.size == a.size:
            return
        if b.ndim == 1 and a.ndim == 2 and 1 in a.shape and a.size == b.size:
            return
        raise EvalError(f"dimension mismatch for '{op}': {_shape(a)} and {_shape(b)}")


def _shape(a: np.ndarray) -> str:
    return "×".join(str(s) for s in a.shape) if a.ndim > 1 else f"vector[{a.shape[0]}]"


def mul(a: Any, b: Any) -> Any:
    a, b = _arr(a), _arr(b)
    _check_numeric(a, "*"); _check_numeric(b, "*")
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.ndim == 1 and b.ndim == 1:
            if a.shape != b.shape:
                raise EvalError(f"dot product needs vectors of equal length: {a.size} and {b.size}")
            return _result(np.dot(a, b))
        try:
            if a.ndim == 2 and b.ndim == 2:
                return _result(a @ b)
            if a.ndim == 2 and b.ndim == 1:
                return _result(a @ b)
            if a.ndim == 1 and b.ndim == 2:
                if b.shape[0] == 1:
                    return _result(np.outer(a, b[0]))
                return _result(a @ b)
        except ValueError:
            raise EvalError(f"matrix dimensions do not match for product: {_shape(a)} and {_shape(b)}") from None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return _result(np.multiply(a, b))
    return _result(a * b)


def div(a: Any, b: Any) -> Any:
    a, b = _arr(a), _arr(b)
    _check_numeric(a, "/"); _check_numeric(b, "/")
    if isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[0] == b.shape[1] and not (isinstance(a, np.ndarray) and a.ndim == 2 and a.shape != b.shape):
        # a / M  ==  a * M^-1   (Mathcad convention)
        return mul(a, _inverse(b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        with np.errstate(divide="ignore", invalid="ignore"):
            return _result(np.true_divide(a, b))
    if b == 0:
        if a == 0:
            raise EvalError("0/0 is undefined")
        raise EvalError("division by zero")
    if isinstance(a, int) and isinstance(b, int) and a % b == 0:
        return a // b
    return _result(a / b)


def _inverse(m: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(m.astype(float) if m.dtype != complex else m)
    except np.linalg.LinAlgError:
        raise EvalError("matrix is singular") from None


def power(a: Any, b: Any) -> Any:
    a, b = _arr(a), _arr(b)
    _check_numeric(a, "^"); _check_numeric(b, "^")
    if isinstance(a, np.ndarray) and a.ndim == 2 and is_number(b):
        if a.shape[0] != a.shape[1]:
            raise EvalError("matrix power requires a square matrix")
        n = float(b.real if isinstance(b, complex) else b)
        if not n.is_integer():
            raise EvalError("matrix power requires an integer exponent")
        n = int(n)
        if n < 0:
            return _result(np.linalg.matrix_power(_inverse(a), -n))
        return _result(np.linalg.matrix_power(a, n))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return elem_power(a, b)
    try:
        if isinstance(a, int) and isinstance(b, int):
            if b >= 0:
                if abs(b) > 10000:
                    return float(a) ** b
                return a ** b
            return a ** float(b)
        if isinstance(a, (int, float)) and a < 0 and isinstance(b, float) and not b.is_integer():
            return _result(complex(a) ** b)
        return _result(a ** b)
    except ZeroDivisionError:
        raise EvalError("zero cannot be raised to a negative power") from None
    except OverflowError:
        return math.inf


def elem_mul(a: Any, b: Any) -> Any:
    return _result(np.multiply(_arr(a), _arr(b)))


def elem_div(a: Any, b: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        return _result(np.true_divide(_arr(a), _arr(b)))


def elem_power(a: Any, b: Any) -> Any:
    a, b = _arr(a), _arr(b)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        aa = np.asarray(a)
        bb = np.asarray(b)
        if np.issubdtype(aa.dtype, np.integer) and np.issubdtype(bb.dtype, np.integer) and np.all(bb >= 0):
            return _result(np.power(aa, bb))
        res = np.power(aa.astype(float) if not np.iscomplexobj(aa) else aa, bb)
        if np.any(np.isnan(res)) and not np.iscomplexobj(aa):
            res = np.power(aa.astype(complex), bb)
        return _result(res)


def neg(a: Any) -> Any:
    a = _arr(a)
    _check_numeric(a, "-")
    return _result(-a)


_COMPARE = {
    "==": operator.eq, "!=": operator.ne, "<": operator.lt,
    "<=": operator.le, ">": operator.gt, ">=": operator.ge,
}


def compare(op: str, a: Any, b: Any) -> Any:
    a, b = _arr(a), _arr(b)
    fn = _COMPARE[op]
    if isinstance(a, str) or isinstance(b, str):
        if op in ("==", "!="):
            return int(fn(a, b))
        if isinstance(a, str) and isinstance(b, str):
            return int(fn(a, b))
        raise EvalError("cannot compare a string with a number")
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if op in ("==", "!=") and isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape != b.shape:
            return int(op == "!=")
        return _result(fn(a, b).astype(np.int64))
    if isinstance(a, complex) or isinstance(b, complex):
        if op in ("==", "!="):
            return int(fn(a, b))
        raise EvalError("complex numbers cannot be ordered")
    if isinstance(a, float) or isinstance(b, float):
        if op == "==":
            return int(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15))
        if op == "!=":
            return int(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15))
    return int(fn(a, b))


BINARY = {
    "+": add, "-": sub, "*": mul, "/": div, "^": power,
    ".*": elem_mul, "./": elem_div, ".^": elem_power,
}
