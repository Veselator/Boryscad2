"""Graphs and routing: Dijkstra, minimum spanning tree, distance matrix, sweep method, TSP."""
from __future__ import annotations

import heapq
import itertools
import math
from typing import Any

import numpy as np

from ..engine.values import DText, EvalError, Record, SeriesGroup, Table, normalize_number
from ._util import clean, strings, to_float, to_int, vec


def _node(v: Any) -> Any:
    v = normalize_number(v)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, (int, str)):
        return v
    return str(v)


def edges_from_value(E: Any) -> list[tuple[Any, Any, float]]:
    if isinstance(E, Record) and "edges" in E.fields:
        E = E.fields["edges"]
    arr = np.asarray(E, dtype=object)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise EvalError("edges must be a matrix with rows [from, to, weight]")
    edges = []
    for u, v, w in arr:
        edges.append((_node(u), _node(v), to_float(normalize_number(w), "edge weight")))
    return edges


def _sort_nodes(nodes):
    return sorted(nodes, key=lambda n: (isinstance(n, str), n))


def _adjacency(edges, directed=False):
    adj: dict[Any, list[tuple[Any, float]]] = {}
    for u, v, w in edges:
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, [])
        if not directed:
            adj[v].append((u, w))
    return adj


def _name(node, names: dict) -> str:
    return names.get(node, str(node))


def _names_dict(nodes, names, origin=0):
    if names is None:
        return {}
    labels = strings(names)
    out = {}
    for n in nodes:
        if isinstance(n, int) and 0 <= n - origin < len(labels):
            out[n] = labels[n - origin]
    return out


def dijkstra_core(edges, start, directed=False):
    adj = _adjacency(edges, directed)
    if start not in adj:
        raise EvalError(f"node {start!r} is not in the graph")
    dist = {n: math.inf for n in adj}
    prev: dict[Any, Any] = {n: None for n in adj}
    dist[start] = 0.0
    visited = []
    heap = [(0.0, 0, start)]
    counter = itertools.count(1)
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.append(u)
        for v, w in adj[u]:
            if d + w < dist[v] - 1e-12:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], next(counter), v))
    return dist, prev, visited


def register(r):
    cat = "Operations research · Graphs"

    @r.function(category=cat, signature="dijkstra(E, start, finish, directed=0, names=)",
                doc="shortest path (Dijkstra); E rows are [from, to, weight]")
    def dijkstra(E, start, finish, directed=0, names=None):
        edges = edges_from_value(E)
        start, finish = _node(start), _node(finish)
        dist, prev, order = dijkstra_core(edges, start, bool(directed))
        if finish not in dist:
            raise EvalError(f"node {finish!r} is not in the graph")
        if math.isinf(dist[finish]):
            raise EvalError("the finish node cannot be reached")
        path = [finish]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()
        nd = _names_dict(dist.keys(), names)
        rows = [[k + 1, _name(n, nd), clean(dist[n]), "" if prev[n] is None else _name(prev[n], nd)]
                for k, n in enumerate(order)]
        path_idx = {n for n in path}
        table = Table(["step", "node", "distance", "previous"], rows, "Permanent labels (order of fixing)",
                      {(k, 1) for k, n in enumerate(order) if n in path_idx})
        route = " → ".join(_name(n, nd) for n in path)
        return Record("Dijkstra's algorithm", {
            "path": np.array(path, dtype=object if any(isinstance(p, str) for p in path) else np.int64),
            "cost": clean(dist[finish], 1e-9), "route": route,
            "distances": np.array([clean(dist[n]) for n in _sort_nodes(dist)])},
            [table.display()], ["route", "cost"])

    @r.function(category=cat, signature="mst(E, start=, names=)",
                doc="minimum spanning tree (Kruskal): edges, total length, ties and service order")
    def mst(E, start=None, names=None):
        edges = edges_from_value(E)
        nodes = _sort_nodes({u for u, v, w in edges} | {v for u, v, w in edges})
        parent = {n: n for n in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        nd = _names_dict(nodes, names)
        chosen, rows = [], []
        ordered = sorted(edges, key=lambda e: e[2])
        for k, (u, v, w) in enumerate(ordered):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                chosen.append((u, v, w))
                action = "added"
            else:
                action = "skipped (cycle)"
            rows.append([k + 1, f"{_name(u, nd)} – {_name(v, nd)}", w, action])
            if len(chosen) == len(nodes) - 1:
                break
        if len(chosen) != len(nodes) - 1:
            raise EvalError("the graph is not connected")
        total = clean(sum(w for _, _, w in chosen), 1e-9)
        chosen_set = {(u, v) for u, v, _ in chosen} | {(v, u) for u, v, _ in chosen}
        ties = []
        for u, v, w in edges:
            if (u, v) in chosen_set:
                continue
            same = [c for c in chosen if abs(c[2] - w) < 1e-9]
            if same:
                # replacing is valid when the tree path between u and v contains an edge of equal weight
                path = _tree_path(chosen, u, v)
                if any(abs(pw - w) < 1e-9 for pw in path):
                    ties.append(f"{_name(u, nd)} – {_name(v, nd)} ({w:g})")
        root = _node(start) if start is not None else (nodes[0])
        order = _preorder(chosen, root)
        details = [Table(["step", "edge", "length", "action"], rows, "Kruskal's algorithm").display(),
                   DText("Service order (depth-first from " + _name(root, nd) + "): " +
                         " – ".join(_name(n, nd) for n in order), "note")]
        if ties:
            details.append(DText("Alternative trees of equal length are possible with: " + ", ".join(ties), "note"))
        else:
            details.append(DText("The minimum spanning tree is unique.", "note"))
        tree = np.array([[u, v, w] for u, v, w in chosen], dtype=object)
        return Record("Minimum spanning tree", {"edges": tree, "length": total, "order": np.array(order, dtype=object),
                                                "ties": len(ties)}, details, ["length"])

    @r.function(category=cat, signature="shortest_paths(E, names=)",
                doc="matrix of shortest distances between all nodes (Floyd–Warshall)", aliases=("distance_matrix",))
    def shortest_paths(E, names=None, directed=0):
        edges = edges_from_value(E)
        nodes = _sort_nodes({u for u, v, w in edges} | {v for u, v, w in edges})
        idx = {n: k for k, n in enumerate(nodes)}
        n = len(nodes)
        D = np.full((n, n), math.inf)
        np.fill_diagonal(D, 0)
        for u, v, w in edges:
            D[idx[u], idx[v]] = min(D[idx[u], idx[v]], w)
            if not directed:
                D[idx[v], idx[u]] = min(D[idx[v], idx[u]], w)
        for k in range(n):
            D = np.minimum(D, D[:, [k]] + D[[k], :])
        return np.round(D, 9)

    @r.function(category=cat, signature="route_count(Q, q)", doc="number of ring routes N = ⌈ΣQ / q⌉")
    def route_count(Q, q):
        return int(math.ceil(float(np.sum(vec(Q))) / to_float(q, "capacity") - 1e-12))

    @r.function(category=cat, signature='sweep(L, Q, q, direction="ccw", names=)',
                doc="sweep method: order consumers by polar angle of (L, Q) and split into routes by capacity q")
    def sweep(L, Q, q, direction="ccw", names=None):
        from .plotting import ArrowSeries, LineSeries, ScatterSeries
        L, Q = vec(L, "L"), vec(Q, "Q")
        if L.size != Q.size:
            raise EvalError("L and Q must have the same length")
        cap = to_float(q, "capacity")
        labels = strings(names) if names is not None else [_roman(i + 1) for i in range(L.size)]
        angles = np.degrees(np.arctan2(Q, L))
        order = sorted(range(L.size), key=lambda i: (angles[i], L[i]))
        if str(direction).lower() in ("cw", "clockwise"):
            order = sorted(range(L.size), key=lambda i: (-angles[i], -L[i]))
        routes, current, load = [], [], 0.0
        route_of = {}
        for i in order:
            if Q[i] > cap + 1e-9:
                raise EvalError(f"consumer {labels[i]} needs {Q[i]:g} > capacity {cap:g}")
            if load + Q[i] > cap + 1e-9:
                routes.append((current, load))
                current, load = [], 0.0
            current.append(i)
            load += Q[i]
            route_of[i] = len(routes) + 1
        if current:
            routes.append((current, load))
        rows = [[k + 1, labels[i], L[i], Q[i], round(float(angles[i]), 2), route_of[i]] for k, i in enumerate(order)]
        route_texts = [" – ".join(labels[i] for i in reversed(rt)) for rt, _ in routes]
        details = [Table(["sweep order", "consumer", "L", "Q", "angle, °", "route"], rows, "Sweep method").display()]
        for k, ((rt, ld), txt) in enumerate(zip(routes, route_texts)):
            details.append(DText(f"Route {k + 1}: depot – {txt} – depot, load {ld:g} of {cap:g}", "note"))
        plot = SeriesGroup()
        palette = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#8338ec", "#fb5607"]
        for k, (rt, _) in enumerate(routes):
            color = palette[k % len(palette)]
            for i in rt:
                plot.append(LineSeries(color=color, width=1.0, style=":", x=np.array([0, L[i]]), y=np.array([0, Q[i]])))
            plot.append(ScatterSeries(label=f"route {k + 1}", color=color, size=60, x=L[rt], y=Q[rt],
                                      annotations=[labels[i] for i in rt]))
        reach = float(max(np.max(L), np.max(Q)))
        plot.append(ArrowSeries(label="vector A", color="#222222", x0=0, y0=0, dx=reach * 0.9, dy=0.0))
        return Record("Sweep method", {"order": np.array([labels[i] for i in order], dtype=object),
                                       "routes": np.array(route_texts, dtype=object),
                                       "loads": np.array([ld for _, ld in routes]),
                                       "angles": np.round(angles, 4), "plot": plot, "count": len(routes)},
                      details, ["count", "routes"])

    @r.function(category=cat, signature="tsp(D, nodes=)",
                doc="shortest closed tour through the nodes (exact, Held–Karp); first node is the depot", lazy=True)
    def tsp(ctx, args, kwargs):
        D = np.asarray(ctx.eval(args[0]), dtype=float)
        opts = ctx.eval_kwargs(kwargs)
        origin = ctx.origin
        if "nodes" in opts:
            nodes = [to_int(v) - origin for v in np.asarray(opts["nodes"]).reshape(-1)]
        elif len(args) > 1:
            nodes = [to_int(v) - origin for v in np.asarray(ctx.eval(args[1])).reshape(-1)]
        else:
            nodes = list(range(D.shape[0]))
        if len(nodes) > 16:
            raise EvalError("tsp: at most 16 nodes are supported by the exact method")
        sub = D[np.ix_(nodes, nodes)]
        tour, length = held_karp(sub)
        route = [nodes[i] + origin for i in tour]
        return Record("Travelling salesman tour", {"route": np.array(route), "length": clean(length, 1e-9)},
                      [DText(" → ".join(str(v) for v in route), "note")], ["route", "length"])

    @r.function(category=cat, signature="route_length(D, route)", doc="length of a route through a distance matrix",
                lazy=True)
    def route_length(ctx, args, kwargs):
        D = np.asarray(ctx.eval(args[0]), dtype=float)
        route = [to_int(v) - ctx.origin for v in np.asarray(ctx.eval(args[1])).reshape(-1)]
        return clean(float(sum(D[a, b] for a, b in zip(route, route[1:]))), 1e-9)

    @r.function(category=cat, signature="edges(D)", doc="edge list [i, j, d] from a symmetric distance matrix",
                lazy=True, aliases=("edgelist",))
    def edges(ctx, args, kwargs):
        D = np.asarray(ctx.eval(args[0]), dtype=float)
        o = ctx.origin
        rows = [[i + o, j + o, D[i, j]] for i in range(D.shape[0]) for j in range(i + 1, D.shape[1])
                if D[i, j] > 0 and np.isfinite(D[i, j])]
        return np.array(rows)


def held_karp(D: np.ndarray) -> tuple[list[int], float]:
    n = D.shape[0]
    if n == 1:
        return [0, 0], 0.0
    if n == 2:
        return [0, 1, 0], float(D[0, 1] + D[1, 0])
    C: dict[tuple[int, int], tuple[float, int]] = {}
    for k in range(1, n):
        C[(1 << k, k)] = (D[0, k], 0)
    for size in range(2, n):
        for subset in itertools.combinations(range(1, n), size):
            bits = sum(1 << b for b in subset)
            for k in subset:
                prev = bits & ~(1 << k)
                C[(bits, k)] = min((C[(prev, m)][0] + D[m, k], m) for m in subset if m != k)
    bits = (1 << n) - 2
    length, k = min((C[(bits, k)][0] + D[k, 0], k) for k in range(1, n))
    path = []
    while k != 0:
        path.append(k)
        _, prev = C[(bits, k)]
        bits &= ~(1 << k)
        k = prev
    return [0] + list(reversed(path)) + [0], float(length)


def _tree_path(tree, a, b):
    adj: dict[Any, list[tuple[Any, float]]] = {}
    for u, v, w in tree:
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
    stack = [(a, None, [])]
    while stack:
        node, parent, weights = stack.pop()
        if node == b:
            return weights
        for nxt, w in adj.get(node, []):
            if nxt != parent:
                stack.append((nxt, node, weights + [w]))
    return []


def _preorder(tree, root):
    adj: dict[Any, list[tuple[float, Any]]] = {}
    for u, v, w in tree:
        adj.setdefault(u, []).append((w, v))
        adj.setdefault(v, []).append((w, u))
    order, seen = [], set()

    def visit(n):
        seen.add(n)
        order.append(n)
        for w, nxt in sorted(adj.get(n, []), key=lambda t: (t[0], str(t[1]))):
            if nxt not in seen:
                visit(nxt)

    visit(root)
    return order


def _roman(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out
