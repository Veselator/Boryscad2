"""Executes whole worksheets: region by region, statement by statement."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np
import sympy as sp

from ..core.document import Document, MathRegion, PlotRegion, Region, TextRegion
from ..language import ParseError, ast as A, parse_cached, parse_expr
from .environment import Environment
from .evaluator import Cancelled, Evaluator, SolveState, cartesian_ranges, set_element
from .formatting import NumberFormat, format_value
from .registry import FunctionRegistry, default_registry
from .symbolic import SymbolicTranslator, sympy_to_source, sympy_to_value
from .values import EvalError, RangeValue, UserFunction, normalize_number, to_int


# ------------------------------------------------------------------ results
@dataclass
class StatementResult:
    stmt: A.Stmt
    kind: str
    value: Any = None
    symbolic_ast: Optional[A.Expr] = None
    symbolic_text: str = ""
    error: Optional[EvalError] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class RegionResult:
    region_id: str
    statements: list[StatementResult] = field(default_factory=list)
    parse_error: Optional[ParseError] = None
    plot: Any = None                 # PlotData for plot regions
    text: Optional[str] = None       # interpolated text for text regions
    error: Optional[str] = None      # region level error message
    elapsed: float = 0.0

    @property
    def has_error(self) -> bool:
        return bool(self.parse_error or self.error or any(s.error for s in self.statements))

    def messages(self) -> list[str]:
        out = []
        if self.parse_error:
            out.append(f"Syntax error: {self.parse_error}")
        if self.error:
            out.append(self.error)
        out.extend(str(s.error) for s in self.statements if s.error)
        return out


@dataclass
class DocumentResult:
    regions: dict[str, RegionResult]
    env: Environment
    elapsed: float
    cancelled: bool = False

    def errors(self) -> list[tuple[str, str]]:
        return [(rid, msg) for rid, r in self.regions.items() for msg in r.messages()]


# ------------------------------------------------------------------ session
class Session:
    """Mutable evaluation state shared by all regions of one calculation."""

    def __init__(self, registry: FunctionRegistry, fmt: NumberFormat, cancel_event: threading.Event | None = None):
        self.registry = registry
        self.fmt = fmt
        self.evaluator = Evaluator(registry, cancel_event)
        self.env = Environment(None, {"ORIGIN": 0, "TOL": 1e-6, "CTOL": 1e-6})

    @property
    def solve_state(self) -> SolveState:
        return self.evaluator.solve_state


class RegionEvaluator(Protocol):
    def evaluate(self, region: Region, session: Session) -> RegionResult: ...


class MathRegionEvaluator:
    def evaluate(self, region: MathRegion, session: Session) -> RegionResult:  # type: ignore[override]
        result = RegionResult(region.id)
        try:
            statements = parse_cached(region.source)
        except ParseError as exc:
            result.parse_error = exc
            return result
        for stmt in statements:
            result.statements.append(execute_statement(stmt, session))
        return result


class TextRegionEvaluator:
    """Interpolates ``{{expr}}`` placeholders with evaluated values."""

    PATTERN = re.compile(r"\{\{(.+?)\}\}")

    def evaluate(self, region: TextRegion, session: Session) -> RegionResult:  # type: ignore[override]
        result = RegionResult(region.id)
        if "{{" not in region.text:
            return result

        def repl(match: re.Match) -> str:
            try:
                value = session.evaluator.eval(parse_expr(match.group(1)), session.env)
                return format_value(value, session.fmt)
            except (ParseError, EvalError) as exc:
                result.error = f"{match.group(1)}: {exc}"
                return f"⚠{match.group(1)}"

        result.text = self.PATTERN.sub(repl, region.text)
        return result


class PlotRegionEvaluator:
    def evaluate(self, region: PlotRegion, session: Session) -> RegionResult:  # type: ignore[override]
        from ..library.plotting import PlotData, collect_series
        result = RegionResult(region.id)
        data = PlotData()
        for index, source in enumerate(region.series):
            if not source.strip():
                continue
            try:
                statements = parse_cached(source)
                for stmt in statements:
                    if isinstance(stmt, (A.ExprStmt, A.Evaluate)):
                        value = session.evaluator.eval(stmt.expr, session.env)
                        data.series.extend(collect_series(value, source, index))
                    else:
                        execute_statement(stmt, session)
            except (ParseError, EvalError) as exc:
                data.errors.append(f"{source}: {exc}")
        for attr in ("xlim", "ylim", "zlim"):
            text = getattr(region, attr)
            if text.strip():
                try:
                    value = session.evaluator.eval(parse_expr(text), session.env)
                    if isinstance(value, RangeValue):
                        setattr(data, attr, (float(value.start), float(value.stop)))
                    elif isinstance(value, np.ndarray) and value.size == 2:
                        setattr(data, attr, (float(value[0]), float(value[1])))
                    else:
                        data.errors.append(f"{attr}: expected a range like 0..10")
                except (ParseError, EvalError) as exc:
                    data.errors.append(f"{attr}: {exc}")
        try:
            from ..library.autorange import resolve_plot
            resolve_plot(data)
        except Exception as exc:  # automatic bounds must never break a plot
            data.errors.append(f"auto range: {exc}")
        result.plot = data
        if data.errors:
            result.error = "; ".join(data.errors)
        return result


class NullEvaluator:
    def evaluate(self, region: Region, session: Session) -> RegionResult:
        return RegionResult(region.id)


class WorksheetEngine:
    """Facade that evaluates documents.  Region evaluators are pluggable."""

    def __init__(self, registry: FunctionRegistry | None = None):
        self.registry = registry or default_registry()
        self.region_evaluators: dict[str, RegionEvaluator] = {
            "math": MathRegionEvaluator(),
            "text": TextRegionEvaluator(),
            "plot": PlotRegionEvaluator(),
        }
        self._null = NullEvaluator()

    def register_region_evaluator(self, kind: str, evaluator: RegionEvaluator) -> None:
        self.region_evaluators[kind] = evaluator

    def create_session(self, fmt: NumberFormat | None = None, cancel_event: threading.Event | None = None) -> Session:
        return Session(self.registry, fmt or NumberFormat(), cancel_event)

    def run(self, document: Document, cancel_event: threading.Event | None = None) -> DocumentResult:
        fmt = NumberFormat(precision=document.settings.precision, exp_threshold=document.settings.exp_threshold)
        session = self.create_session(fmt, cancel_event)
        started = time.perf_counter()
        results: dict[str, RegionResult] = {}
        cancelled = False
        for region in document.ordered():
            t0 = time.perf_counter()
            handler = self.region_evaluators.get(region.kind, self._null)
            try:
                res = handler.evaluate(region, session)
            except Cancelled:
                cancelled = True
                break
            except Exception as exc:  # defensive: one broken region must not stop the sheet
                res = RegionResult(region.id, error=f"internal error: {exc!r}")
            res.elapsed = time.perf_counter() - t0
            results[region.id] = res
        return DocumentResult(results, session.env, time.perf_counter() - started, cancelled)

    def run_source(self, source: str) -> tuple[list[StatementResult], Session]:
        session = self.create_session()
        statements = parse_cached(source)
        return [execute_statement(s, session) for s in statements], session


# ---------------------------------------------------------------- statements
def execute_statement(stmt: A.Stmt, session: Session) -> StatementResult:
    ev, env = session.evaluator, session.env
    try:
        if isinstance(stmt, A.Assign):
            value = ev.eval(stmt.value, env)
            env.define(stmt.name, value)
            return StatementResult(stmt, "assign", value)
        if isinstance(stmt, A.FuncDef):
            fn = UserFunction(stmt.name, stmt.params, stmt.body, env)
            env.define(stmt.name, fn)
            return StatementResult(stmt, "funcdef", fn)
        if isinstance(stmt, A.IndexAssign):
            value = _index_assign(stmt, session)
            return StatementResult(stmt, "assign", value)
        if isinstance(stmt, A.Destructure):
            value = ev.eval(stmt.value, env)
            items = _as_sequence(value)
            if len(items) != len(stmt.names):
                raise EvalError(f"cannot unpack {len(items)} value(s) into {len(stmt.names)} name(s)", stmt.span)
            for name, item in zip(stmt.names, items):
                env.define(name, normalize_number(item))
            return StatementResult(stmt, "assign", value)
        if isinstance(stmt, A.Evaluate):
            value = ev.eval(stmt.expr, env)
            return StatementResult(stmt, "evaluate", value)
        if isinstance(stmt, A.Symbolic):
            return _symbolic(stmt, session)
        if isinstance(stmt, A.Given):
            ev.solve_state = SolveState(True, [], env)
            return StatementResult(stmt, "given")
        if isinstance(stmt, A.ExprStmt):
            expr = stmt.expr
            if isinstance(expr, A.Compare) and ev.solve_state.active:
                ev.solve_state.constraints.append(expr)
                return StatementResult(stmt, "constraint")
            value = ev.eval(expr, env)
            return StatementResult(stmt, "expr", value)
        raise EvalError(f"unsupported statement {type(stmt).__name__}", stmt.span)
    except Cancelled:
        raise
    except EvalError as exc:
        if exc.span is None:
            exc.span = stmt.span
        return StatementResult(stmt, _kind_of(stmt), error=exc)
    except RecursionError:
        return StatementResult(stmt, _kind_of(stmt), error=EvalError("recursion is too deep", stmt.span))
    except Exception as exc:  # library failures are reported, not raised
        return StatementResult(stmt, _kind_of(stmt), error=EvalError(f"{type(exc).__name__}: {exc}", stmt.span))


def _kind_of(stmt: A.Stmt) -> str:
    return {A.Assign: "assign", A.FuncDef: "funcdef", A.IndexAssign: "assign", A.Destructure: "assign",
            A.Evaluate: "evaluate", A.Symbolic: "symbolic", A.Given: "given"}.get(type(stmt), "expr")


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return list(value.reshape(-1)) if value.ndim == 1 or 1 in value.shape else list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    raise EvalError("the right-hand side must be a vector to unpack")


def _index_assign(stmt: A.IndexAssign, session: Session) -> Any:
    ev, env = session.evaluator, session.env
    origin = ev.origin(env)
    index_values = [ev.eval(i, env) for i in stmt.indices]
    current = env.get(stmt.name) if env.has(stmt.name) else None
    if isinstance(current, np.ndarray):
        current = current.copy()
    else:
        current = None
    range_names, range_values = [], []
    for node, value in zip(stmt.indices, index_values):
        if isinstance(value, RangeValue):
            if not isinstance(node, A.Var):
                raise EvalError("range indices must be range variables", stmt.span)
            range_names.append(node.name)
            range_values.append(value)
    if not range_names:
        value = ev.eval(stmt.value, env)
        indices = [to_int(v, "index") - origin for v in index_values]
        current = set_element(current, indices, value)
    else:
        for binding in cartesian_ranges(range_names, range_values):
            local = env.child(binding)
            indices = [to_int(ev.eval(i, local), "index") - origin for i in stmt.indices]
            current = set_element(current, indices, ev.eval(stmt.value, local))
    env.define(stmt.name, current)
    return current


def _symbolic(stmt: A.Symbolic, session: Session) -> StatementResult:
    ev, env = session.evaluator, session.env
    keep: set[str] = set()
    for cmd in stmt.commands:
        for arg in cmd.args:
            if isinstance(arg, A.Var):
                keep.add(arg.name)
        for kw in cmd.kwargs:
            keep.add(kw.name)
    translator = SymbolicTranslator(env, keep=keep, evaluator=ev)
    expr = translator.to_sympy(stmt.expr)
    if hasattr(expr, "doit"):
        expr = expr.doit()
    for cmd in stmt.commands:
        command = session.registry.get_symbolic(cmd.name)
        if command is None:
            raise EvalError(f"unknown symbolic keyword '{cmd.name}'. Known: {', '.join(session.registry.symbolic_names())}",
                            cmd.span)
        args = [translator.to_sympy(a) if not isinstance(a, A.String) else a.value for a in cmd.args]
        kwargs = {k.name: translator.to_sympy(k.value) for k in cmd.kwargs}
        kwargs = {translator.symbol(k): v for k, v in kwargs.items()}
        expr = command.impl(expr, *args, **({"substitutions": kwargs} if kwargs else {}))
    text = sympy_to_source(expr)
    try:
        ast_result = parse_expr(text)
    except ParseError:
        ast_result = A.String(text)
    value = None
    try:
        value = sympy_to_value(expr)
    except (EvalError, TypeError, ValueError, AttributeError):
        value = None
    return StatementResult(stmt, "symbolic", value, ast_result, text)


def sympy_number(expr: sp.Basic) -> Any:
    return sympy_to_value(expr)
