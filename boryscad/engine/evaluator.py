"""Tree-walking evaluator for expressions and programs."""
from __future__ import annotations

import itertools
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..language import ast as A
from . import operators as ops
from .environment import Environment
from .registry import FunctionRegistry, FunctionSpec
from .values import (BuiltinRef, EvalError, Record, RangeValue, SeriesGroup, Table, UserFunction, is_number,
                     normalize_number, to_float, to_int)


class ArrayConditionError(EvalError):
    """A condition evaluated to an array; the caller may retry element-wise."""


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class Cancelled(EvalError):
    pass


@dataclass
class SolveState:
    """Constraints collected after a ``given`` statement."""

    active: bool = False
    constraints: list[A.Compare] = field(default_factory=list)
    env_at_given: Optional[Environment] = None


@dataclass
class CallContext:
    """Passed to lazy builtins: gives controlled access to the evaluator."""

    evaluator: "Evaluator"
    env: Environment
    span: A.Span = None

    def eval(self, node: A.Expr, bindings: dict[str, Any] | None = None) -> Any:
        env = self.env.child(bindings) if bindings else self.env
        return self.evaluator.eval(node, env)

    def eval_vectorized(self, node: A.Expr, bindings: dict[str, Any]) -> Any:
        """Evaluate with array bindings where * / ^ are element-wise (Mathcad's vectorize operator)."""
        previous = self.evaluator.elementwise
        self.evaluator.elementwise = True
        try:
            return self.eval(node, bindings)
        finally:
            self.evaluator.elementwise = previous

    def eval_kwargs(self, kwargs: tuple[A.Keyword, ...]) -> dict[str, Any]:
        return {k.name: self.evaluator.eval(k.value, self.env) for k in kwargs}

    def compile(self, node: A.Expr, names: list[str]) -> Callable[..., Any]:
        """Return a Python callable evaluating ``node`` with ``names`` bound to its arguments."""
        evaluator, env = self.evaluator, self.env

        def fn(*values):
            local = Environment(env, dict(zip(names, values)))
            return evaluator.eval(node, local)

        return fn

    @property
    def registry(self) -> FunctionRegistry:
        return self.evaluator.registry

    @property
    def solve_state(self) -> SolveState:
        return self.evaluator.solve_state

    @property
    def origin(self) -> int:
        return self.evaluator.origin(self.env)

    def lookup(self, name: str, default: Any = None) -> Any:
        return self.env.get(name, default)


class Evaluator:
    MAX_ITERATIONS = 50_000_000

    def __init__(self, registry: FunctionRegistry, cancel_event: threading.Event | None = None):
        self.registry = registry
        self.solve_state = SolveState()
        self.cancel_event = cancel_event
        self.elementwise = False  # vectorized evaluation (plots): * / ^ act element by element
        self._steps = 0
        self._dispatch: dict[type, Callable[[Any, Environment], Any]] = {
            A.Number: self._number,
            A.Imag: lambda n, e: n.value,
            A.Infinity: lambda n, e: math.inf,
            A.String: lambda n, e: n.value,
            A.Var: self._var,
            A.Paren: lambda n, e: self.eval(n.body, e),
            A.BinOp: self._binop,
            A.UnOp: self._unop,
            A.Compare: self._compare,
            A.RangeExpr: self._range,
            A.Call: self._call,
            A.Index: self._index,
            A.Member: self._member,
            A.Factorial: self._factorial,
            A.Abs: self._abs,
            A.Sqrt: self._sqrt,
            A.MatrixLit: self._matrix,
            A.IfExpr: self._if_expr,
            A.Program: self._program,
        }

    # ------------------------------------------------------------------ core
    def eval(self, node: A.Expr, env: Environment) -> Any:
        handler = self._dispatch.get(type(node))
        if handler is None:
            raise EvalError(f"cannot evaluate {type(node).__name__}", node.span)
        try:
            return handler(node, env)
        except EvalError as exc:
            if exc.span is None:
                exc.span = node.span
            raise
        except (_Return, _Break, _Continue):
            raise
        except RecursionError:
            raise EvalError("recursion is too deep", node.span) from None
        except ZeroDivisionError:
            raise EvalError("division by zero", node.span) from None
        except OverflowError:
            raise EvalError("numeric overflow", node.span) from None
        except (TypeError, ValueError, IndexError, KeyError, np.linalg.LinAlgError) as exc:
            raise EvalError(str(exc) or type(exc).__name__, node.span) from None

    def origin(self, env: Environment) -> int:
        value = env.get("ORIGIN", 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def tick(self) -> None:
        self._steps += 1
        if self._steps & 0x3FF == 0 and self.cancel_event is not None and self.cancel_event.is_set():
            raise Cancelled("calculation interrupted")

    # ------------------------------------------------------------- leaves
    @staticmethod
    def _number(node: A.Number, env: Environment) -> Any:
        return node.value

    def _var(self, node: A.Var, env: Environment) -> Any:
        try:
            return env.lookup(node.name)
        except KeyError:
            pass
        constants = self.registry._constants
        if node.name in constants:
            return constants[node.name]
        if node.name in self.registry:
            return BuiltinRef(node.name)
        raise EvalError(f"'{node.name}' is not defined", node.span)

    # ---------------------------------------------------------- operators
    def _binop(self, node: A.BinOp, env: Environment) -> Any:
        op = node.op
        if op in ("and", "or"):
            left = self.eval(node.left, env)
            if is_number(left):
                if op == "and" and not left:
                    return 0
                if op == "or" and left:
                    return 1
                right = self.eval(node.right, env)
                if is_number(right):
                    return int(bool(right))
                return np.asarray(right).astype(bool).astype(np.int64)
            right = self.eval(node.right, env)
            fn = np.logical_and if op == "and" else np.logical_or
            return fn(np.asarray(ops._arr(left)), np.asarray(ops._arr(right))).astype(np.int64)
        left = self.eval(node.left, env)
        if op == "^" and isinstance(node.right, A.Var) and node.right.name == "T" and isinstance(left, np.ndarray) \
                and not env.has("T"):
            return left.reshape(1, -1) if left.ndim == 1 else left.T
        right = self.eval(node.right, env)
        if self.elementwise and op in ("*", "/", "^"):
            op = "." + op
        fn = ops.BINARY.get(op)
        if fn is None:
            raise EvalError(f"unknown operator '{op}'", node.span)
        return fn(left, right)

    def _unop(self, node: A.UnOp, env: Environment) -> Any:
        value = self.eval(node.operand, env)
        if node.op == "-":
            return ops.neg(value)
        if node.op == "not":
            if is_number(value):
                return int(not value)
            return np.logical_not(np.asarray(ops._arr(value))).astype(np.int64)
        raise EvalError(f"unknown operator '{node.op}'", node.span)

    def _compare(self, node: A.Compare, env: Environment) -> Any:
        values = [self.eval(o, env) for o in node.operands]
        result: Any = 1
        for op, a, b in zip(node.ops, values, values[1:]):
            r = ops.compare(op, a, b)
            if isinstance(result, np.ndarray) or isinstance(r, np.ndarray):
                result = np.logical_and(result, r).astype(np.int64)
            else:
                result = int(bool(result) and bool(r))
        return result

    def _range(self, node: A.RangeExpr, env: Environment) -> RangeValue:
        start = normalize_number(self.eval(node.start, env))
        stop = normalize_number(self.eval(node.stop, env))
        for v, what in ((start, "range start"), (stop, "range end")):
            to_float(v, what)
        if node.second is not None:
            second = normalize_number(self.eval(node.second, env))
            step = ops.sub(second, start)
        elif node.step is not None:
            step = normalize_number(self.eval(node.step, env))
        else:
            step = 1 if stop >= start else -1
        if isinstance(step, float) and step.is_integer() and isinstance(start, int):
            step = int(step)
        return RangeValue(start, stop, step)

    # -------------------------------------------------------------- calls
    def _call(self, node: A.Call, env: Environment) -> Any:
        self.tick()
        target = env.get(node.name, None) if env.has(node.name) else None
        if isinstance(target, UserFunction):
            if node.kwargs:
                raise EvalError(f"user function '{node.name}' does not accept keyword arguments", node.span)
            args = [self.eval(a, env) for a in node.args]
            return self.call_function(target, args, node.span)
        if isinstance(target, BuiltinRef):
            spec = self.registry.get(target.name)
        elif target is not None:
            raise EvalError(f"'{node.name}' is not a function", node.span)
        else:
            spec = self.registry.get(node.name)
        if spec is None:
            raise EvalError(f"unknown function '{node.name}'", node.span)
        return self.call_builtin(spec, node, env)

    def call_builtin(self, spec: FunctionSpec, node: A.Call, env: Environment) -> Any:
        if spec.lazy:
            ctx = CallContext(self, env, node.span)
            return self._wrap_call(spec, lambda: spec.impl(ctx, node.args, node.kwargs), node.span)
        args = [self._arg_value(self.eval(a, env)) for a in node.args]
        kwargs = {k.name: self._arg_value(self.eval(k.value, env)) for k in node.kwargs}
        return self._wrap_call(spec, lambda: spec.impl(*args, **kwargs), node.span)

    @staticmethod
    def _arg_value(value: Any) -> Any:
        if isinstance(value, RangeValue):
            return value.to_array()
        return value

    def _wrap_call(self, spec: FunctionSpec, thunk: Callable[[], Any], span: A.Span) -> Any:
        try:
            return _normalize_result(thunk())
        except EvalError:
            raise
        except TypeError as exc:
            msg = str(exc)
            if "argument" in msg or "positional" in msg or "keyword" in msg:
                raise EvalError(f"wrong arguments for {spec.signature}", span) from None
            raise EvalError(f"{spec.name}: {msg}", span) from None
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, IndexError, KeyError) as exc:
            raise EvalError(f"{spec.name}: {exc}", span) from None

    def call_value(self, fn: Any, args: list[Any], span: A.Span = None) -> Any:
        """Call a function value (user function or builtin reference)."""
        if isinstance(fn, UserFunction):
            return self.call_function(fn, args, span)
        if isinstance(fn, BuiltinRef):
            spec = self.registry.get(fn.name)
            if spec is None or spec.lazy:
                raise EvalError(f"'{fn.name}' cannot be used as a function value", span)
            return self._wrap_call(spec, lambda: spec.impl(*args), span)
        if callable(fn):
            return fn(*args)
        raise EvalError("value is not a function", span)

    def call_function(self, fn: UserFunction, args: list[Any], span: A.Span = None) -> Any:
        if len(args) != len(fn.params):
            raise EvalError(f"{fn.name} expects {len(fn.params)} argument(s), got {len(args)}", span)
        args = [a.to_array() if isinstance(a, RangeValue) else a for a in args]
        local = Environment(fn.closure, dict(zip(fn.params, args)))
        try:
            return self._run_body(fn.body, local)
        except ArrayConditionError:
            if not any(isinstance(a, np.ndarray) for a in args):
                raise
            return self._elementwise(fn, args, span)

    def _run_body(self, body: A.Expr, env: Environment) -> Any:
        try:
            return self.eval(body, env)
        except _Return as ret:
            return ret.value

    def _elementwise(self, fn: UserFunction, args: list[Any], span: A.Span) -> Any:
        arrays = [np.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
        shape = np.broadcast_shapes(*[a.shape for a in arrays if isinstance(a, np.ndarray)])
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape):
            call_args = [np.broadcast_to(a, shape)[idx] if isinstance(a, np.ndarray) else a for a in arrays]
            out[idx] = self.call_function(fn, [normalize_number(v) for v in call_args], span)
        try:
            return out.astype(float) if not any(isinstance(v, complex) for v in out.flat) else out.astype(complex)
        except (TypeError, ValueError):
            return out

    # ---------------------------------------------------------- structure
    def _index(self, node: A.Index, env: Environment) -> Any:
        target = self.eval(node.target, env)
        if isinstance(target, RangeValue):
            target = target.to_array()
        indices = [self.eval(i, env) for i in node.indices]
        origin = self.origin(env)
        if isinstance(target, (list, SeriesGroup)):
            return target[to_int(indices[0], "index") - origin]
        if isinstance(target, str):
            return target[to_int(indices[0], "index") - origin]
        if isinstance(target, Table):
            r = to_int(indices[0], "row") - origin
            if len(indices) == 1:
                return np.array(target.rows[r], dtype=object)
            return target.rows[r][to_int(indices[1], "column") - origin]
        if not isinstance(target, np.ndarray):
            raise EvalError("only vectors, matrices and strings can be indexed", node.span)
        if len(indices) > target.ndim:
            raise EvalError(f"too many indices for {'vector' if target.ndim == 1 else 'matrix'}", node.span)
        py_idx = tuple(_to_index(v, origin, target.shape[d]) for d, v in enumerate(indices))
        result = target[py_idx]
        return normalize_number(result) if not isinstance(result, np.ndarray) else result

    def _member(self, node: A.Member, env: Environment) -> Any:
        target = self.eval(node.target, env)
        if isinstance(target, Record):
            return target.get(node.name)
        if isinstance(target, Table):
            if node.name in target.headers:
                return target.column(node.name)
            raise EvalError(f"table has no column '{node.name}'", node.span)
        if isinstance(target, np.ndarray) and node.name == "T":
            return target.reshape(1, -1) if target.ndim == 1 else target.T
        raise EvalError(f"value has no field '{node.name}'", node.span)

    def _factorial(self, node: A.Factorial, env: Environment) -> Any:
        value = self.eval(node.operand, env)
        if isinstance(value, (np.ndarray, RangeValue)):
            from scipy.special import factorial
            return factorial(ops._arr(value), exact=False)
        n = to_float(value, "factorial argument")
        if n < 0:
            raise EvalError("factorial of a negative number")
        if n.is_integer() and n <= 170:
            return math.factorial(int(n))
        return math.gamma(n + 1)

    def _abs(self, node: A.Abs, env: Environment) -> Any:
        value = ops._arr(self.eval(node.operand, env))
        if isinstance(value, np.ndarray):
            if value.ndim == 2 and value.shape[0] == value.shape[1]:
                return normalize_number(np.linalg.det(value))
            if value.ndim == 1:
                return normalize_number(np.linalg.norm(value))
            return np.abs(value)
        if isinstance(value, str):
            return len(value)
        return normalize_number(abs(value))

    def _sqrt(self, node: A.Sqrt, env: Environment) -> Any:
        value = ops._arr(self.eval(node.operand, env))
        return _normalize_result(np.emath.sqrt(value))

    def _matrix(self, node: A.MatrixLit, env: Environment) -> Any:
        if not node.rows:
            return np.array([], dtype=float)
        rows = [[self.eval(cell, env) for cell in row] for row in node.rows]
        rows = [[ops._arr(c) for c in row] for row in rows]
        has_block = any(isinstance(c, np.ndarray) for row in rows for c in row)
        if has_block:
            if len(rows) > 1 and all(len(r) == 1 for r in rows) and all(
                    isinstance(r[0], np.ndarray) and r[0].ndim == 1 for r in rows):
                return np.concatenate([r[0] for r in rows])
            blocks = [[c if isinstance(c, np.ndarray) and c.ndim == 2 else
                       (c.reshape(-1, 1) if isinstance(c, np.ndarray) else np.array([[c]])) for c in row] for row in rows]
            try:
                return np.block(blocks)
            except ValueError as exc:
                raise EvalError(f"cannot assemble block matrix: {exc}", node.span) from None
        flat = [c for row in rows for c in row]
        dtype = _common_dtype(flat)
        arr = np.array(rows, dtype=dtype)
        if arr.shape[0] == 1 or arr.shape[1] == 1:
            return arr.reshape(-1)
        return arr

    def _if_expr(self, node: A.IfExpr, env: Environment) -> Any:
        cond = self.eval(node.cond, env)
        if isinstance(cond, np.ndarray):
            return np.where(cond.astype(bool), ops._arr(self.eval(node.then, env)), ops._arr(self.eval(node.otherwise, env)))
        return self.eval(node.then if truthy(cond) else node.otherwise, env)

    # ------------------------------------------------------------ programs
    def _program(self, node: A.Program, env: Environment) -> Any:
        local = env.child()
        return self._exec_block(node, local)

    def _exec_block(self, node: A.Program, env: Environment) -> Any:
        result: Any = None
        for stmt in node.body:
            result = self._exec_pstmt(stmt, env)
        return result

    def _exec_pstmt(self, stmt: A.PStmt, env: Environment) -> Any:
        self.tick()
        if isinstance(stmt, A.PExpr):
            return self.eval(stmt.expr, env)
        if isinstance(stmt, A.LocalAssign):
            value = self.eval(stmt.value, env)
            if isinstance(stmt.target, A.Var):
                env.define(stmt.target.name, value)
            else:
                self._assign_index_local(stmt.target, value, env)
            return value
        if isinstance(stmt, A.IfStmt):
            cond = self.eval(stmt.cond, env)
            if truthy(cond):
                return self._exec_block(stmt.body, env)
            if isinstance(stmt.orelse, A.Program):
                return self._exec_block(stmt.orelse, env)
            if isinstance(stmt.orelse, A.IfStmt):
                return self._exec_pstmt(stmt.orelse, env)
            return None
        if isinstance(stmt, A.ForStmt):
            iterable = self.eval(stmt.iterable, env)
            result = None
            for value in iterate(iterable):
                env.define(stmt.var, value)
                try:
                    result = self._exec_block(stmt.body, env)
                except _Break:
                    break
                except _Continue:
                    continue
            return result
        if isinstance(stmt, A.WhileStmt):
            result = None
            count = 0
            while truthy(self.eval(stmt.cond, env)):
                count += 1
                if count > self.MAX_ITERATIONS:
                    raise EvalError("while loop exceeded the iteration limit", stmt.span)
                try:
                    result = self._exec_block(stmt.body, env)
                except _Break:
                    break
                except _Continue:
                    continue
            return result
        if isinstance(stmt, A.ReturnStmt):
            raise _Return(self.eval(stmt.value, env))
        if isinstance(stmt, A.BreakStmt):
            raise _Break()
        if isinstance(stmt, A.ContinueStmt):
            raise _Continue()
        raise EvalError(f"unsupported statement {type(stmt).__name__}", stmt.span)

    def _assign_index_local(self, target: A.Index, value: Any, env: Environment) -> None:
        base = A.strip_parens(target.target)
        if not isinstance(base, A.Var):
            raise EvalError("only named arrays can be assigned by index", target.span)
        name = base.name
        current = env.vars.get(name) if name in env.vars else None
        if current is None and env.has(name):
            outer = env.lookup(name)
            current = outer.copy() if isinstance(outer, np.ndarray) else None
        indices = [to_int(self.eval(i, env), "index") - self.origin(env) for i in target.indices]
        env.define(name, set_element(current, indices, value))


# --------------------------------------------------------------------- helpers
def truthy(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return bool(value.reshape(-1)[0])
        raise ArrayConditionError("condition is a vector; a scalar condition is required")
    if isinstance(value, RangeValue):
        raise ArrayConditionError("condition is a range; a scalar condition is required")
    if isinstance(value, str):
        return bool(value)
    if value is None:
        return False
    return bool(value)


def iterate(value: Any):
    if isinstance(value, RangeValue):
        return iter(value)
    if isinstance(value, np.ndarray):
        return (normalize_number(v) if not isinstance(v, np.ndarray) else v for v in value)
    if isinstance(value, (list, tuple, SeriesGroup)):
        return iter(value)
    if isinstance(value, str):
        return iter(value)
    if is_number(value):
        return iter([value])
    raise EvalError("value is not iterable")


def _to_index(value: Any, origin: int, size: int) -> Any:
    if isinstance(value, RangeValue):
        value = value.to_array()
    if isinstance(value, np.ndarray):
        idx = value.astype(np.int64) - origin
        if np.any(idx < 0) or np.any(idx >= size):
            raise EvalError(f"index out of range (ORIGIN = {origin}, size = {size})")
        return idx
    i = to_int(value, "index") - origin
    if i < 0 or i >= size:
        raise EvalError(f"index {i + origin} is out of range (ORIGIN = {origin}, size = {size})")
    return i


def _common_dtype(values: list[Any]):
    if any(isinstance(v, str) or not (is_number(v)) for v in values):
        return object
    if any(isinstance(v, (complex, np.complexfloating)) for v in values):
        return complex
    if all(isinstance(v, (int, np.integer, bool, np.bool_)) for v in values):
        return np.int64
    return float


def set_element(array: Any, indices: list[int], value: Any) -> np.ndarray:
    """Return ``array`` with ``array[indices] = value``, growing it with zeros like Mathcad."""
    if any(i < 0 for i in indices):
        raise EvalError("negative index in assignment")
    value = normalize_number(value)
    if len(indices) == 1:
        arr = array if isinstance(array, np.ndarray) and array.ndim == 1 else np.zeros(0)
        if isinstance(array, np.ndarray) and array.ndim == 2:
            raise EvalError("matrix element assignment needs two indices")
        needed = indices[0] + 1
        dtype = _merge_dtype(arr.dtype, value)
        if needed > arr.size:
            grown = np.zeros(needed, dtype=dtype)
            grown[:arr.size] = arr
            arr = grown
        else:
            arr = arr.astype(dtype, copy=True)
        arr[indices[0]] = value
        return arr
    if len(indices) == 2:
        arr = array if isinstance(array, np.ndarray) and array.ndim == 2 else np.zeros((0, 0))
        rows = max(arr.shape[0], indices[0] + 1)
        cols = max(arr.shape[1], indices[1] + 1)
        dtype = _merge_dtype(arr.dtype, value)
        grown = np.zeros((rows, cols), dtype=dtype)
        grown[:arr.shape[0], :arr.shape[1]] = arr
        grown[indices[0], indices[1]] = value
        return grown
    raise EvalError("at most two indices are supported")


def _merge_dtype(dtype, value):
    if dtype == object or isinstance(value, (str, np.ndarray)) or not is_number(value):
        return object
    if isinstance(value, complex) or np.issubdtype(dtype, np.complexfloating):
        return complex
    if isinstance(value, float) or np.issubdtype(dtype, np.floating):
        return float
    return np.int64 if dtype != float else float


def _normalize_result(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return normalize_number(value.item())
        if np.iscomplexobj(value) and value.dtype != object and np.allclose(value.imag, 0):
            return value.real
        return value
    if isinstance(value, (list, tuple)) and not isinstance(value, SeriesGroup):
        if value and all(is_number(v) for v in value):
            return np.array(value)
        return SeriesGroup(value) if value and all(hasattr(v, "series_kind") for v in value) else value
    return normalize_number(value)


def cartesian_ranges(names: list[str], values: list[Any]):
    """Iterate over the cartesian product of range values."""
    lists = [list(iterate(v)) for v in values]
    for combo in itertools.product(*lists):
        yield dict(zip(names, combo))
