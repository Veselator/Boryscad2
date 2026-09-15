"""Solve blocks (given … find / maximize / minimize / minerr) and root finding.

Solving uses a chain of strategies (Strategy + Chain of Responsibility):
exact linear programming, exact KKT enumeration for small polynomial
problems and robust multistart numeric optimisation.
"""
from __future__ import annotations

import itertools
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np
import sympy as sp
from scipy import optimize

from ..engine.symbolic import SymbolicTranslator, sympy_to_value
from ..engine.values import BuiltinRef, EvalError, UserFunction, is_number, normalize_number
from ..language import ast as A
from ._util import clean, to_float, to_int, vec

FEAS_TOL = 1e-7


# ------------------------------------------------------------------ problem
@dataclass
class SymbolicForm:
    symbols: list[sp.Symbol]
    objective: Optional[sp.Expr]
    eqs: list[sp.Expr]           # expr == 0
    ineqs: list[sp.Expr]         # expr >= 0


@dataclass
class Problem:
    names: list[str]
    guess: np.ndarray
    objective: Optional[Callable[[np.ndarray], float]] = None
    sense: str = "min"
    eqs: list[Callable[[np.ndarray], float]] = field(default_factory=list)
    ineqs: list[Callable[[np.ndarray], float]] = field(default_factory=list)
    strict: list[bool] = field(default_factory=list)
    symbolic: Optional[SymbolicForm] = None
    integer: bool = False
    tol: float = 1e-6

    @property
    def n(self) -> int:
        return len(self.names)

    def value(self, x: np.ndarray) -> float:
        v = self.objective(x) if self.objective else 0.0
        return v

    def violation(self, x: np.ndarray) -> float:
        worst = 0.0
        for g in self.eqs:
            worst = max(worst, abs(g(x)))
        for h in self.ineqs:
            worst = max(worst, -h(x))
        return worst

    def feasible(self, x: np.ndarray, tol: float | None = None) -> bool:
        return self.violation(x) <= (tol if tol is not None else max(FEAS_TOL, self.tol * 10))


@dataclass
class Solution:
    x: np.ndarray
    value: float
    method: str
    exact: bool = False


class Strategy(Protocol):
    name: str

    def solve(self, problem: Problem) -> Optional[Solution]: ...


# --------------------------------------------------------------- strategies
class LinearProgrammingStrategy:
    name = "linear programming (HiGHS)"

    def solve(self, problem: Problem) -> Optional[Solution]:
        s = problem.symbolic
        if s is None or s.objective is None:
            return None
        try:
            if not all(_degree(e, s.symbols) <= 1 for e in [s.objective, *s.eqs, *s.ineqs]):
                return None
        except (sp.PolynomialError, sp.GeneratorsNeeded, TypeError, ValueError):
            return None
        n = len(s.symbols)
        c = np.array([float(sp.diff(s.objective, v)) for v in s.symbols])
        if problem.sense == "max":
            c = -c
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for e in s.ineqs:  # e >= 0  →  -grad·x <= const
            row = [float(sp.diff(e, v)) for v in s.symbols]
            const = float(e.subs({v: 0 for v in s.symbols}))
            A_ub.append([-x for x in row])
            b_ub.append(const)
        for e in s.eqs:
            row = [float(sp.diff(e, v)) for v in s.symbols]
            const = float(e.subs({v: 0 for v in s.symbols}))
            A_eq.append(row)
            b_eq.append(-const)
        res = optimize.linprog(
            c, A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq) if A_eq else None, b_eq=np.array(b_eq) if b_eq else None,
            bounds=[(None, None)] * n, method="highs",
            integrality=np.ones(n) if problem.integer else None,
        )
        if res.status == 2:
            raise EvalError("the constraints are inconsistent (no feasible solution)")
        if res.status == 3:
            raise EvalError("the objective is unbounded on the feasible region")
        if not res.success:
            return None
        x = np.array([clean(v, 1e-9) for v in res.x])
        return Solution(x, problem.value(x), self.name, exact=True)


class KKTStrategy:
    """Enumerates active constraint sets and solves the Lagrange conditions exactly."""

    name = "Lagrange / KKT (exact)"
    max_vars = 3
    max_ineqs = 7
    timeout = 4.0

    def solve(self, problem: Problem) -> Optional[Solution]:
        s = problem.symbolic
        if s is None or s.objective is None or problem.integer:
            return None
        if len(s.symbols) > self.max_vars or len(s.ineqs) > self.max_ineqs:
            return None
        try:
            if not all(_degree(e, s.symbols) <= 2 for e in [s.objective, *s.eqs, *s.ineqs]):
                return None
        except (sp.PolynomialError, sp.GeneratorsNeeded, TypeError, ValueError):
            return None
        box: dict[str, Any] = {}

        def work():
            try:
                box["candidates"] = self._candidates(s)
            except Exception as exc:  # sympy may fail on degenerate systems
                box["error"] = exc

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive() or "candidates" not in box:
            return None
        best: Optional[Solution] = None
        for point in box["candidates"]:
            x = np.array([float(point[v]) for v in s.symbols])
            if not problem.feasible(x, 1e-8):
                continue
            val = problem.value(x)
            better = best is None or (val > best.value + 1e-12 if problem.sense == "max" else val < best.value - 1e-12)
            if better:
                best = Solution(x, val, self.name, exact=True)
        return best

    @staticmethod
    def _candidates(s: SymbolicForm) -> list[dict]:
        out = []
        m = len(s.ineqs)
        for k in range(0, min(m, len(s.symbols) + len(s.eqs) + 1) + 1):
            for active in itertools.combinations(range(m), k):
                cons = list(s.eqs) + [s.ineqs[i] for i in active]
                if len(cons) > len(s.symbols) + len(s.eqs) + 1 and len(cons) > len(s.symbols):
                    continue
                lambdas = sp.symbols(f"_l0:{len(cons)}")
                lag = s.objective - sum(l * g for l, g in zip(lambdas, cons))
                system = [sp.diff(lag, v) for v in s.symbols] + cons
                unknowns = list(s.symbols) + list(lambdas)
                try:
                    sols = sp.solve(system, unknowns, dict=True)
                except (NotImplementedError, ValueError, TypeError):
                    continue
                for sol in sols:
                    if not all(v in sol for v in s.symbols):
                        continue
                    vals = {}
                    ok = True
                    for v in s.symbols:
                        num = complex(sp.N(sol[v]))
                        if abs(num.imag) > 1e-9:
                            ok = False
                            break
                        vals[v] = num.real
                    if ok:
                        out.append(vals)
        return out


class NumericOptimizationStrategy:
    name = "SLSQP multistart"
    starts = 40

    def solve(self, problem: Problem) -> Optional[Solution]:
        if problem.objective is None:
            return None
        sign = -1.0 if problem.sense == "max" else 1.0
        cons = [{"type": "eq", "fun": g} for g in problem.eqs] + [{"type": "ineq", "fun": h} for h in problem.ineqs]

        def f(x):
            v = problem.objective(x)
            return sign * v

        best: Optional[Solution] = None
        rng = np.random.default_rng(12345)
        for start in _start_points(problem.guess, self.starts, rng):
            try:
                with np.errstate(all="ignore"):
                    res = optimize.minimize(f, start, method="SLSQP", constraints=cons,
                                            options={"maxiter": 500, "ftol": 1e-12})
            except (ValueError, ArithmeticError, EvalError):
                continue
            x = res.x
            if not np.all(np.isfinite(x)) or not problem.feasible(x):
                continue
            val = problem.value(x)
            if abs(val) > 1e12:
                raise EvalError("the objective appears to be unbounded")
            if best is None or sign * val < sign * best.value - 1e-10:
                best = Solution(x, val, self.name)
        if best is not None:
            best.x = np.array([clean(v, 1e-7) for v in best.x])
            best.value = problem.value(best.x)
        return best


class EquationSystemStrategy:
    name = "root finding"
    starts = 40

    def solve(self, problem: Problem) -> Optional[Solution]:
        rng = np.random.default_rng(2024)
        n = problem.n
        square = len(problem.eqs) == n and not problem.ineqs
        best_x, best_res = None, math.inf
        for start in _start_points(problem.guess, self.starts, rng):
            try:
                with np.errstate(all="ignore"):
                    if square:
                        res = optimize.root(lambda x: [g(x) for g in problem.eqs], start, method="hybr")
                        x = res.x
                    else:
                        obj = lambda x: sum(g(x) ** 2 for g in problem.eqs)
                        cons = [{"type": "ineq", "fun": h} for h in problem.ineqs]
                        res = optimize.minimize(obj, start, method="SLSQP", constraints=cons,
                                                options={"maxiter": 800, "ftol": 1e-16})
                        x = res.x
            except (ValueError, ArithmeticError, EvalError):
                continue
            if not np.all(np.isfinite(x)):
                continue
            viol = problem.violation(x)
            if viol < best_res:
                best_x, best_res = x, viol
            if viol <= problem.tol:
                break
        if best_x is None:
            return None
        x = np.array([clean(v, 1e-8) for v in best_x])
        return Solution(x, best_res, self.name)


def _start_points(guess: np.ndarray, count: int, rng):
    yield guess.astype(float)
    scale = np.maximum(1.0, np.abs(guess))
    for i in range(count - 1):
        factor = 10.0 ** rng.uniform(-1, 1.5)
        yield guess + rng.normal(0, 1, guess.size) * scale * factor


def _degree(expr: sp.Expr, symbols: list[sp.Symbol]) -> int:
    expr = sp.expand(expr)
    if not expr.free_symbols & set(symbols):
        return 0
    return sp.Poly(expr, *symbols).total_degree()


# ------------------------------------------------------------- problem build
def _unknowns(args: tuple[A.Expr, ...]) -> list[str]:
    names = []
    for a in args:
        a = A.strip_parens(a)
        if not isinstance(a, A.Var):
            raise EvalError("unknowns must be variable names")
        names.append(a.name)
    if not names:
        raise EvalError("list the unknowns, e.g. find(x, y)")
    return names


def _guess(ctx, names: list[str]) -> np.ndarray:
    values = []
    for n in names:
        v = ctx.env.get(n) if ctx.env.has(n) else None
        if v is not None and is_number(v) and not isinstance(v, complex):
            values.append(float(v))
        else:
            values.append(1.0)
    return np.array(values)


def _vector_fn(ctx, node: A.Expr, names: list[str]) -> Callable[[np.ndarray], float]:
    fn = ctx.compile(node, names)

    def wrapped(x: np.ndarray) -> float:
        v = fn(*[float(t) for t in x])
        if isinstance(v, np.ndarray):
            if v.size != 1:
                raise EvalError("constraint or objective must be scalar")
            v = v.reshape(-1)[0]
        if isinstance(v, complex):
            if abs(v.imag) > 1e-9:
                return math.nan
            v = v.real
        return float(v)

    return wrapped


def build_problem(ctx, objective: Optional[A.Expr], names: list[str], sense: str, integer: bool = False) -> Problem:
    problem = Problem(names, _guess(ctx, names), sense=sense, integer=integer,
                      tol=float(ctx.env.get("TOL", 1e-6) or 1e-6))
    constraint_nodes: list[tuple[str, A.Expr]] = []  # ("eq"|"ineq", expr) meaning expr ==0 / >=0
    for comp in ctx.solve_state.constraints:
        for op, lhs, rhs in zip(comp.ops, comp.operands, comp.operands[1:]):
            if op == "==":
                constraint_nodes.append(("eq", A.BinOp("-", lhs, rhs)))
            elif op in ("<=", "<"):
                constraint_nodes.append(("ineq", A.BinOp("-", rhs, lhs)))
            elif op in (">=", ">"):
                constraint_nodes.append(("ineq", A.BinOp("-", lhs, rhs)))
            else:
                raise EvalError(f"operator {op} cannot be used as a constraint")
    for kind, node in constraint_nodes:
        target = problem.eqs if kind == "eq" else problem.ineqs
        target.append(_vector_fn(ctx, node, names))
    if objective is not None:
        problem.objective = _vector_fn(ctx, objective, names)
    # symbolic form (best effort)
    try:
        tr = SymbolicTranslator(ctx.env, keep=names, evaluator=ctx.evaluator)
        symbols = [tr.symbol(n) for n in names]
        sym_obj = sp.expand(tr.to_sympy(objective)) if objective is not None else None
        eqs = [sp.expand(tr.to_sympy(n)) for k, n in constraint_nodes if k == "eq"]
        ineqs = [sp.expand(tr.to_sympy(n)) for k, n in constraint_nodes if k == "ineq"]
        exprs = ([sym_obj] if sym_obj is not None else []) + eqs + ineqs
        if all(isinstance(e, sp.Expr) and e.free_symbols <= set(symbols) for e in exprs):
            problem.symbolic = SymbolicForm(symbols, sym_obj, eqs, ineqs)
    except Exception:
        problem.symbolic = None
    return problem


def _objective_node(ctx, node: A.Expr, names: list[str]) -> A.Expr:
    bare = A.strip_parens(node)
    if isinstance(bare, A.Var):
        value = ctx.env.get(bare.name) if ctx.env.has(bare.name) else None
        if isinstance(value, UserFunction):
            if len(value.params) != len(names):
                raise EvalError(f"{bare.name} has {len(value.params)} parameter(s) but {len(names)} unknown(s) are listed")
            return A.Call(bare.name, tuple(A.Var(n) for n in names))
        if value is None and ctx.registry.get(bare.name) is not None:
            return A.Call(bare.name, tuple(A.Var(n) for n in names))
    return node


def _result(x: np.ndarray) -> Any:
    x = np.array([normalize_number(clean(float(v), 1e-10)) for v in x])
    if x.size == 1:
        v = float(x[0])
        return int(v) if v.is_integer() and abs(v) < 1e15 else v
    if np.allclose(x, np.round(x), rtol=0, atol=1e-10) and np.all(np.abs(x) < 1e15):
        return np.round(x).astype(np.int64)
    return x


def _optimize(ctx, args, kwargs, sense: str):
    if len(args) < 2:
        raise EvalError(f"{'maximize' if sense == 'max' else 'minimize'}(f, x, y, …)")
    names = _unknowns(args[1:])
    opts = ctx.eval_kwargs(kwargs)
    integer = bool(opts.get("integer", 0))
    objective = _objective_node(ctx, args[0], names)
    problem = build_problem(ctx, objective, names, sense, integer)
    candidates: list[Solution] = []
    for strategy in (LinearProgrammingStrategy(), KKTStrategy(), NumericOptimizationStrategy()):
        if integer and not isinstance(strategy, LinearProgrammingStrategy):
            continue
        sol = strategy.solve(problem)
        if sol is not None:
            candidates.append(sol)
            if isinstance(strategy, LinearProgrammingStrategy):
                break
    if not candidates:
        if integer:
            raise EvalError("integer=1 is supported for linear problems only")
        raise EvalError("no feasible solution was found (check constraints and initial guesses)")
    sign = 1 if sense == "max" else -1
    best = candidates[0]
    for c in candidates[1:]:
        if sign * c.value > sign * best.value + 1e-7 * max(1.0, abs(best.value)):
            best = c
    return _result(best.x)


def register(r):
    cat = "Solving"

    @r.function(lazy=True, category=cat, signature="find(x, y, …)",
                doc="solve the equations/inequalities of the solve block (after 'given')", aliases=("Find",))
    def find(ctx, args, kwargs):
        names = _unknowns(args)
        if not ctx.solve_state.active:
            raise EvalError("find must follow a 'given' block")
        problem = build_problem(ctx, None, names, "min")
        if not problem.eqs and not problem.ineqs:
            raise EvalError("the solve block has no constraints")
        sol = EquationSystemStrategy().solve(problem)
        if sol is None or sol.value > max(problem.tol, 1e-6):
            raise EvalError("no solution was found; try other initial guesses")
        return _result(sol.x)

    @r.function(lazy=True, category=cat, signature="minerr(x, y, …)",
                doc="approximate solution minimizing the constraint error", aliases=("Minerr",))
    def minerr(ctx, args, kwargs):
        names = _unknowns(args)
        problem = build_problem(ctx, None, names, "min")
        sol = EquationSystemStrategy().solve(problem)
        if sol is None:
            raise EvalError("minerr failed")
        return _result(sol.x)

    @r.function(lazy=True, category=cat, signature="maximize(f, x, y, …, integer=0)",
                doc="maximum of f subject to the constraints of the current solve block", aliases=("Maximize",))
    def maximize(ctx, args, kwargs):
        return _optimize(ctx, args, kwargs, "max")

    @r.function(lazy=True, category=cat, signature="minimize(f, x, y, …, integer=0)",
                doc="minimum of f subject to the constraints of the current solve block", aliases=("Minimize",))
    def minimize(ctx, args, kwargs):
        return _optimize(ctx, args, kwargs, "min")

    @r.function(lazy=True, category=cat, signature="root(f(x), x [, a, b])",
                doc="root of f near the current x, or bracketed in [a, b]")
    def root(ctx, args, kwargs):
        if len(args) not in (2, 4):
            raise EvalError("root(f(x), x) or root(f(x), x, a, b)")
        name = _unknowns(args[1:2])[0]
        f = ctx.compile(args[0], [name])
        fr = lambda x: to_float(f(x), "function value")
        if len(args) == 4:
            a, b = to_float(ctx.eval(args[2])), to_float(ctx.eval(args[3]))
            if fr(a) * fr(b) > 0:
                raise EvalError("f(a) and f(b) must have opposite signs")
            return clean(optimize.brentq(fr, a, b, xtol=1e-14), 1e-12)
        guess = _guess(ctx, [name])[0]
        try:
            x = optimize.newton(fr, guess, tol=1e-12, maxiter=200)
        except (RuntimeError, ArithmeticError):
            raise EvalError("root did not converge; try a different guess") from None
        return clean(float(x), 1e-12)

    @r.function(category=cat, signature="polyroots(v)", doc="roots of the polynomial v₀ + v₁x + v₂x² + …")
    def polyroots(v):
        coeffs = vec(v, "coefficients", dtype=complex)
        roots = np.roots(coeffs[::-1])
        roots = np.sort_complex(roots)
        return roots.real if np.allclose(roots.imag, 0) else roots

    @r.function(lazy=True, category=cat, signature="solve(equation, x)", doc="exact symbolic solutions as a vector")
    def solve(ctx, args, kwargs):
        names = _unknowns(args[1:])
        tr = SymbolicTranslator(ctx.env, keep=names, evaluator=ctx.evaluator)
        expr = tr.to_sympy(args[0])
        symbols = [tr.symbol(n) for n in names]
        sols = sp.solve(expr, symbols, dict=True)
        if not sols:
            raise EvalError("no solution")
        if len(symbols) == 1:
            values = [sympy_to_value(s[symbols[0]]) for s in sols if symbols[0] in s]
            if len(values) == 1:
                return values[0]
            return np.array(values)
        return np.array([[sympy_to_value(s.get(v, v)) for v in symbols] for s in sols]).squeeze()

    @r.function(lazy=True, category=cat, signature="fmin(f(x), x, a, b)", doc="minimum point of f on [a, b]")
    def fmin(ctx, args, kwargs):
        name = _unknowns(args[1:2])[0]
        f = ctx.compile(args[0], [name])
        a, b = to_float(ctx.eval(args[2])), to_float(ctx.eval(args[3]))
        res = optimize.minimize_scalar(lambda x: to_float(f(x)), bounds=(a, b), method="bounded",
                                       options={"xatol": 1e-12})
        return clean(float(res.x), 1e-9)

    @r.function(lazy=True, category=cat, signature="fmax(f(x), x, a, b)", doc="maximum point of f on [a, b]")
    def fmax(ctx, args, kwargs):
        name = _unknowns(args[1:2])[0]
        f = ctx.compile(args[0], [name])
        a, b = to_float(ctx.eval(args[2])), to_float(ctx.eval(args[3]))
        res = optimize.minimize_scalar(lambda x: -to_float(f(x)), bounds=(a, b), method="bounded",
                                       options={"xatol": 1e-12})
        return clean(float(res.x), 1e-9)
