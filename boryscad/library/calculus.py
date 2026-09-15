"""Calculus operators: derivatives, integrals, sums, products, limits, ODEs, interpolation."""
from __future__ import annotations

import math

import numpy as np
import sympy as sp
from scipy import integrate, interpolate

from ..engine.symbolic import SymbolicTranslator, sympy_to_value
from ..engine.values import EvalError, RangeValue, UserFunction, BuiltinRef, is_number, normalize_number
from ..language import ast as A
from ._util import to_float, to_int, vec


def _var(node: A.Expr) -> str:
    node = A.strip_parens(node)
    if not isinstance(node, A.Var):
        raise EvalError("the variable of the operator must be a name", node.span)
    return node.name


def register(r):
    cat = "Calculus"

    @r.function(lazy=True, category=cat, signature="deriv(f(x), x, n=1)",
                doc="n-th derivative with respect to x, evaluated at the current x", aliases=("diff",))
    def deriv(ctx, args, kwargs):
        if len(args) < 2:
            raise EvalError("deriv(expression, variable [, order])")
        name = _var(args[1])
        order = to_int(ctx.eval(args[2]), "order") if len(args) > 2 else 1
        at = ctx.env.get(name) if ctx.env.has(name) else None
        if at is None:
            raise EvalError(f"define {name} before taking the derivative (the point where it is evaluated)")
        try:
            tr = SymbolicTranslator(ctx.env, keep=[name], evaluator=ctx.evaluator)
            expr = sp.diff(tr.to_sympy(args[0]), tr.symbol(name), order)
            fn = sp.lambdify(tr.symbol(name), expr, modules=["numpy"])
            value = fn(at.to_array() if isinstance(at, RangeValue) else at)
            if isinstance(value, np.ndarray) or is_number(value):
                if np.all(np.isfinite(np.asarray(value, dtype=complex))):
                    return normalize_number(value) if not isinstance(value, np.ndarray) else value
        except (EvalError, TypeError, ValueError, NotImplementedError, NameError, AttributeError, ZeroDivisionError):
            pass
        f = ctx.compile(args[0], [name])
        return _numeric_derivative(f, at, order)

    @r.function(lazy=True, category=cat, signature="integral(f(x), x, a, b)", doc="definite integral from a to b",
                aliases=("integrate",))
    def integral(ctx, args, kwargs):
        if len(args) != 4:
            raise EvalError("integral(expression, variable, lower, upper)")
        name = _var(args[1])
        a, b = to_float(ctx.eval(args[2]), "lower limit"), to_float(ctx.eval(args[3]), "upper limit")
        f = ctx.compile(args[0], [name])

        def real_f(x):
            v = f(x)
            if isinstance(v, complex):
                return v.real
            return float(v)

        value, err = integrate.quad(real_f, a, b, limit=200)
        return value

    @r.function(lazy=True, category=cat, signature="sum(expr, i, a, b) | sum(v)", doc="summation Σ", aliases=("Σ",))
    def sum_(ctx, args, kwargs):
        return _accumulate(ctx, args, np.add, 0, "sum")

    @r.function(lazy=True, category=cat, signature="prod(expr, i, a, b) | prod(v)", doc="product Π", aliases=("Π",))
    def prod(ctx, args, kwargs):
        return _accumulate(ctx, args, np.multiply, 1, "prod")

    @r.function(lazy=True, category=cat, signature="limit(f(x), x, a, dir=\"+-\")", doc="limit (symbolic)")
    def limit(ctx, args, kwargs):
        name = _var(args[1])
        tr = SymbolicTranslator(ctx.env, keep=[name], evaluator=ctx.evaluator)
        expr = tr.to_sympy(args[0])
        point = tr.to_sympy(args[2])
        direction = str(ctx.eval_kwargs(kwargs).get("dir", "+-"))
        return sympy_to_value(sp.limit(expr, tr.symbol(name), point, dir=direction))

    r.function("rkfixed", lazy=True, category=cat, signature="rkfixed(y0, t0, t1, n, D)",
               doc="solve y' = D(t, y) on [t0, t1] with n steps; returns matrix [t, y…]")(_rkfixed)

    @r.function(category=cat, signature="linterp(vx, vy, x)", doc="linear interpolation")
    def linterp(vx, vy, x):
        return np.interp(x, vec(vx), vec(vy))

    @r.function(category=cat, signature="cspline_interp(vx, vy, x)", doc="cubic spline interpolation", aliases=("spline",))
    def cspline_interp(vx, vy, x):
        return interpolate.CubicSpline(vec(vx), vec(vy))(x)

    @r.function(category=cat, signature="polyfit(vx, vy, n)", doc="least squares polynomial coefficients (ascending)")
    def polyfit(vx, vy, n):
        return np.polynomial.polynomial.polyfit(vec(vx), vec(vy), to_int(n))

    @r.function(category=cat, signature="polyval(c, x)", doc="evaluate polynomial with ascending coefficients c")
    def polyval(c, x):
        return np.polynomial.polynomial.polyval(x, vec(c))

    @r.function(category=cat, signature="trapz(vy, vx)", doc="trapezoidal integration of samples")
    def trapz(vy, vx):
        return float(np.trapezoid(vec(vy), vec(vx)))

    @r.function(category=cat, signature="gradient(f, x0)", doc="numeric gradient of f(x1, x2, …) at the point x0",
                lazy=True)
    def gradient(ctx, args, kwargs):
        fn = ctx.eval(args[0])
        point = vec(ctx.eval(args[1]), "point")
        h = 1e-6
        grad = []
        for i in range(point.size):
            p1, p2 = point.copy(), point.copy()
            p1[i] += h
            p2[i] -= h
            grad.append((to_float(ctx.evaluator.call_value(fn, list(p1))) - to_float(ctx.evaluator.call_value(fn, list(p2)))) / (2 * h))
        return np.array(grad)


def _accumulate(ctx, args, op, start, name):
    if len(args) == 1:
        value = ctx.eval(args[0])
        if isinstance(value, RangeValue):
            value = value.to_array()
        if isinstance(value, np.ndarray):
            return normalize_number(np.sum(value) if op is np.add else np.prod(value))
        return value
    var = _var(args[1])
    if len(args) == 2:
        rng = ctx.eval(args[1])
        if not isinstance(rng, (RangeValue, np.ndarray)):
            raise EvalError(f"{name}(expr, i) needs i to be a range variable")
        values = list(rng) if isinstance(rng, RangeValue) else list(rng.reshape(-1))
    elif len(args) == 4:
        lo, hi = ctx.eval(args[2]), ctx.eval(args[3])
        lo_i, hi_i = to_float(lo, "lower limit"), to_float(hi, "upper limit")
        if math.isinf(hi_i) or math.isinf(lo_i):
            tr = SymbolicTranslator(ctx.env, keep=[var], evaluator=ctx.evaluator)
            fn = sp.summation if op is np.add else sp.product
            return sympy_to_value(fn(tr.to_sympy(args[0]), (tr.symbol(var), tr.to_sympy(args[2]), tr.to_sympy(args[3]))))
        values = range(int(round(lo_i)), int(round(hi_i)) + 1)
    else:
        raise EvalError(f"{name}(expression, variable, from, to) or {name}(vector)")
    total = start
    for v in values:
        ctx.evaluator.tick()
        total = op(total, ctx.eval(args[0], {var: normalize_number(v)}))
    return normalize_number(total)


def _numeric_derivative(f, x, order):
    x = to_float(x, "point")
    h = 1e-3 * max(1.0, abs(x)) if order > 1 else 1e-5 * max(1.0, abs(x))
    if order == 1:
        d1 = (f(x + h) - f(x - h)) / (2 * h)
        d2 = (f(x + h / 2) - f(x - h / 2)) / h
        return (4 * d2 - d1) / 3
    if order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2
    coeffs = [(-1) ** k * math.comb(order, k) for k in range(order + 1)]
    return sum(c * f(x + (order / 2 - k) * h) for k, c in enumerate(coeffs)) / h ** order


def _rkfixed(ctx, args, kwargs):
    if len(args) != 5:
        raise EvalError("rkfixed(y0, t0, t1, npoints, D)")
    y0 = np.atleast_1d(np.asarray(ctx.eval(args[0]), dtype=float))
    t0, t1 = to_float(ctx.eval(args[1])), to_float(ctx.eval(args[2]))
    n = to_int(ctx.eval(args[3]))
    D = ctx.eval(args[4])
    if not isinstance(D, (UserFunction, BuiltinRef)):
        raise EvalError("D must be a function D(t, y)")
    scalar = y0.size == 1 and not isinstance(ctx.eval(args[0]), np.ndarray)

    def rhs(t, y):
        arg = float(y[0]) if scalar else y
        return np.atleast_1d(np.asarray(ctx.evaluator.call_value(D, [t, arg]), dtype=float))

    ts = np.linspace(t0, t1, n + 1)
    sol = integrate.solve_ivp(rhs, (t0, t1), y0, t_eval=ts, method="RK45", rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise EvalError(f"ODE solver failed: {sol.message}")
    return np.column_stack([sol.t, sol.y.T])
