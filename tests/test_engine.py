import numpy as np
import pytest

from boryscad.engine import WorksheetEngine
from boryscad.language import ParseError, parse

engine = WorksheetEngine()


def run(src):
    results, session = engine.run_source(src)
    errors = [r.error for r in results if r.error]
    assert not errors, errors
    return [r for r in results if r.kind in ("evaluate", "symbolic")], session


def value(src):
    evaluated, _ = run(src)
    return evaluated[-1].value


# ------------------------------------------------------------------ language
@pytest.mark.parametrize("src", [
    "x := 5", "f(x, y) := x^2 + y", "A := [1, 2; 3, 4]", "v[i] := i^2", "[a, b] := find(a, b)", "x := 0, 0.1 .. 1",
    "r := 0..10 step 2", "(x + 1)^2 -> expand", "given", "0 <= x <= 5", "x ≔ π·2", "g(n) := {\n s <- 0\n return s\n}",
    "line(x, sin(x), label=\"s\")", "x := 1; y := 2",
])
def test_parse_ok(src):
    assert parse(src)


@pytest.mark.parametrize("src", ["x := ", "f(1) := 2", "(a + b", "[1, 2; 3]"])
def test_parse_errors(src):
    with pytest.raises(ParseError):
        parse(src)


# -------------------------------------------------------------------- basics
def test_arithmetic_and_functions():
    assert value("a := 3\nb := 4\nsqrt(a^2 + b^2) =") == 5
    assert value("f(t) := t^2 - 3\nf(4) =") == 13
    assert value("sum(k^2, k, 1, 10) =") == 385
    assert value("integral(sin(t), t, 0, π) =") == pytest.approx(2)
    assert value("x := 2\nderiv(x^3, x) =") == pytest.approx(12)


def test_matrices():
    assert np.allclose(value("A := [1, 2; 3, 4]\nA^-1 ="), [[-2, 1], [1.5, -0.5]])
    assert value("A := [1, 2; 3, 4]\n|A| =") == pytest.approx(-2)
    assert np.allclose(value("A := [1, 2; 3, 4]\nA*[1; 1] ="), [3, 7])


def test_ranges_and_programs():
    assert list(value("i := 0..4\nv[i] := i^2\nv =")) == [0, 1, 4, 9, 16]
    assert value("g(n) := {\n s <- 0\n for j in 1..n {\n  s <- s + j\n }\n return s\n}\ng(100) =") == 5050
    assert list(value("p(z) := { if z > 0 { return 1 } else { return -1 } }\np(-1..1) =")) == [-1, -1, 1]


def test_symbolic():
    evaluated, _ = run("t^2 - 5*t + 6 -> solve(t)")
    assert evaluated[0].symbolic_text == "[2; 3]"
    evaluated, _ = run("(x + 1)^2 -> expand")
    assert evaluated[0].symbolic_text == "x^2 + 2*x + 1"


def test_solve_block_find():
    x = value("x := 1\ny := 1\ngiven\nx^2 + y^2 == 4\nx - y == 1\nfind(x, y) =")
    assert x[0] ** 2 + x[1] ** 2 == pytest.approx(4)
    assert x[0] - x[1] == pytest.approx(1)


# ------------------------------------------------------------ lab tasks
def test_lab3_nonlinear():
    src = ("N := 5\nk := 2\nf(x1, x2) := x1^2 - k*N*x1 + x2^2\nx1 := 1\nx2 := 1\ngiven\nx1^2 + x2^2 <= N^2\n"
           "x2 == N*x1\nx1 >= 0\nx2 >= 0\nminimize(f, x1, x2) =\nmaximize(f, x1, x2) =")
    evaluated, _ = run(src)
    pmin, pmax = evaluated[0].value, evaluated[1].value
    assert pmin[0] == pytest.approx(10 / 52, abs=1e-6)
    assert pmax[0] == pytest.approx(5 / np.sqrt(26), abs=1e-6)


def test_lab1_graphical():
    r = value("c := [10, 12]\nA := [5, 5; 2.5, 4]\nb := [1700, 1600]\nlp_graphical(c, A, b).F =")
    assert r == pytest.approx(4080)


def test_lab2_simplex_matches_highs():
    ev, _ = run("seed(3)\nZ := randmatrix(5, 5, 10)\nc := [3, 5, 4, 2, 6]\nb := [100, 150, 120, 200, 80]\n"
                "simplex(c, Z, b).F =\nlinprog(c, Z, b).F =")
    assert ev[0].value == pytest.approx(ev[1].value)


def test_lab4_transport():
    ev, _ = run("C := [4, 6, 8; 5, 3, 7]\na := [50, 60]\nb := [30, 40, 40]\nt := transport_all(C, a, b)\nt.Zopt =\n"
                "transport(C, a, b, method=\"nw\").Z =")
    assert ev[0].value <= ev[1].value
    assert ev[0].value == 30 * 4 + 40 * 3 + 20 * 8 + 20 * 7 or ev[0].value <= 540


def test_lab5_routing():
    ev, _ = run("E := [0, 1, 2; 1, 2, 3; 0, 2, 4; 2, 3, 1]\nmst(E).length =\nshortest_paths(E) =\n"
                "sweep([10, 20, 30], [5, 5, 5], 10).count =")
    assert ev[0].value == 6
    assert ev[1].value[0, 3] == 5
    assert ev[2].value == 2


def test_lab6_location():
    ev, _ = run("X := [0, 4]\nY := [0, 0]\nQ := [1, 3]\ncentroid(X, Y, Q) =")
    assert np.allclose(ev[0].value, [3, 0])


def test_lab7_gomory():
    ev, _ = run("g := gomory([3, 1], [4, 11; 1, 0], [44, 5])\ng.F =\ng.x =")
    assert ev[0].value == 17
    assert list(ev[1].value) == [5, 2]


def test_lab8_dijkstra():
    ev, _ = run("E := [1, 2, 4; 2, 3, 2; 1, 3, 7; 3, 4, 1]\ndijkstra(E, 1, 4).cost =")
    assert ev[0].value == 7


def test_lab9_allocation():
    ev, _ = run("F := [9, 18, 24, 38, 50; 11, 19, 30, 44, 59; 16, 32, 40, 57, 70; 13, 27, 44, 69, 73]\n"
                "allocate(F, unit=20).F =\nallocate_graph(F, unit=20).F =")
    assert ev[0].value == ev[1].value == 85


# ------------------------------------------------------------ automatic plot bounds
def _plot(series, **fields):
    from boryscad.core.document import Document, PlotRegion
    region = PlotRegion(series=series, **fields)
    doc = Document(regions=[region])
    return engine.run(doc).regions[region.id].plot


def test_autorange_fplot_features():
    data = _plot(["fplot((x - 50)^2 - 100, x)"])
    x = data.series[0].x
    assert 30 < x.min() < 40 and 60 < x.max() < 70


def test_autorange_respects_domain_and_limits():
    data = _plot(["fplot(ln(x), x)"])
    assert data.series[0].x.min() >= 0
    data = _plot(["fplot(sin(x), x)"], xlim="0..3")
    assert data.series[0].x.min() == 0 and data.series[0].x.max() == 3


def test_autorange_sets_and_poles():
    data = _plot(["implicit(u^2 + v^2 == 9, u, v)"])
    X = data.series[0].X
    assert -4.5 < X.min() < -3 and 3 < X.max() < 4.5
    data = _plot(["fplot(tan(x), x)"])
    assert data.ylim is not None and data.ylim[1] < 1000
