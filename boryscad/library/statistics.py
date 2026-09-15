"""Statistics and random numbers."""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..engine.values import EvalError, normalize_number
from ._util import to_float, to_int, vec

_rng = np.random.default_rng()


def _data(*values) -> np.ndarray:
    items = []
    for v in values:
        if isinstance(v, np.ndarray):
            items.extend(v.reshape(-1).tolist())
        else:
            items.append(v)
    if not items:
        raise EvalError("no data")
    return np.asarray(items, dtype=float)


def register(r):
    cat = "Statistics & random"

    @r.function(category=cat, signature="mean(v, …)", doc="arithmetic mean")
    def mean(*v):
        return normalize_number(np.mean(_data(*v)))

    @r.function(category=cat, signature="median(v, …)", doc="median")
    def median(*v):
        return normalize_number(np.median(_data(*v)))

    @r.function(category=cat, signature="var(v)", doc="population variance")
    def var(*v):
        return normalize_number(np.var(_data(*v)))

    @r.function(category=cat, signature="Var(v)", doc="sample variance")
    def Var(*v):
        return normalize_number(np.var(_data(*v), ddof=1))

    @r.function(category=cat, signature="stdev(v)", doc="population standard deviation")
    def stdev(*v):
        return normalize_number(np.std(_data(*v)))

    @r.function(category=cat, signature="Stdev(v)", doc="sample standard deviation")
    def Stdev(*v):
        return normalize_number(np.std(_data(*v), ddof=1))

    @r.function(category=cat, signature="corr(x, y)", doc="Pearson correlation")
    def corr(x, y):
        return normalize_number(np.corrcoef(vec(x), vec(y))[0, 1])

    @r.function(category=cat, signature="seed(n)", doc="reset the random generator (reproducible results)")
    def seed(n):
        global _rng
        _rng = np.random.default_rng(to_int(n))
        return to_int(n)

    @r.function(category=cat, signature="rnd(x)", doc="uniform random number in [0, x)")
    def rnd(x=1):
        return float(_rng.random() * to_float(x))

    @r.function(category=cat, signature="rand()", doc="uniform random number in [0, 1) (like Excel RAND)", aliases=("RAND",))
    def rand():
        return float(_rng.random())

    @r.function(category=cat, signature="runif(n, a, b)", doc="vector of n uniform random numbers")
    def runif(n, a=0, b=1):
        return _rng.uniform(to_float(a), to_float(b), to_int(n))

    @r.function(category=cat, signature="rnorm(n, μ, σ)", doc="vector of n normal random numbers")
    def rnorm(n, mu=0, sigma=1):
        return _rng.normal(to_float(mu), to_float(sigma), to_int(n))

    @r.function(category=cat, signature="randint(a, b, m=0, n=0)", doc="random integer(s) in [a, b]")
    def randint(a, b, m=0, n=0):
        a, b, m, n = to_int(a), to_int(b), to_int(m), to_int(n)
        if m == 0:
            return int(_rng.integers(a, b + 1))
        return _rng.integers(a, b + 1, size=(m, n) if n else m)

    @r.function(category=cat, signature="randmatrix(m, n, max)", doc="m×n matrix of ROUND(RAND()·max)")
    def randmatrix(m, n, mx):
        return np.round(_rng.random((to_int(m), to_int(n))) * to_float(mx)).astype(np.int64)

    @r.function(category=cat, signature="dnorm(x, μ, σ)", doc="normal probability density")
    def dnorm(x, mu=0, sigma=1):
        return stats.norm.pdf(x, to_float(mu), to_float(sigma))

    @r.function(category=cat, signature="pnorm(x, μ, σ)", doc="normal cumulative distribution")
    def pnorm(x, mu=0, sigma=1):
        return stats.norm.cdf(x, to_float(mu), to_float(sigma))

    @r.function(category=cat, signature="qnorm(p, μ, σ)", doc="normal quantile")
    def qnorm(p, mu=0, sigma=1):
        return stats.norm.ppf(p, to_float(mu), to_float(sigma))

    @r.function(category=cat, signature="hist(v, bins)", doc="histogram counts")
    def hist(v, bins=10):
        counts, _ = np.histogram(vec(v), bins=to_int(bins))
        return counts
