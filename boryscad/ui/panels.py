"""Side panels: math palette, function browser, variables and messages."""
from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QGridLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QScrollArea,
                               QTextBrowser, QToolBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..engine.formatting import NumberFormat, format_value
from ..engine.registry import FunctionRegistry
from ..engine.values import type_name

PALETTE: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Арифметика", [
        ("≔", " := ", "определение"), ("=", " =", "вычислить"), ("→", " -> ", "символьно"), ("x²", "^2", "степень"),
        ("xⁿ", "^", "степень"), ("√", "sqrt()", "корень"), ("ⁿ√", "nthroot(, n)", "корень n-й степени"),
        ("|x|", "abs()", "модуль"), ("n!", "!", "факториал"), ("·", "*", "умножение"), ("÷", "/", "дробь"),
        ("( )", "()", "скобки"), ("π", "π", "пи"), ("e", "e", "число e"), ("∞", "∞", "бесконечность"),
        ("i", "1i", "мнимая единица"), ("≤", " <= ", "меньше или равно"), ("≥", " >= ", "больше или равно"),
        ("≠", " != ", "не равно"), ("==", " == ", "уравнение"),
    ]),
    ("Анализ", [
        ("d/dx", "deriv(f(x), x)", "производная"), ("dⁿ/dxⁿ", "deriv(f(x), x, 2)", "n-я производная"),
        ("∫", "integral(f(x), x, a, b)", "определённый интеграл"), ("Σ", "sum(i^2, i, 1, n)", "сумма"),
        ("Π", "prod(i, i, 1, n)", "произведение"), ("lim", "limit(sin(x)/x, x, 0)", "предел"),
        ("a..b", "0..10", "диапазон"), ("step", "0..1 step 0.1", "диапазон с шагом"),
        ("root", "root(f(x), x)", "корень уравнения"), ("poly", "polyroots([c0, c1, c2])", "корни многочлена"),
        ("ODE", "rkfixed(y0, t0, t1, n, D)", "ОДУ Рунге–Кутта"),
    ]),
    ("Матрицы", [
        ("[ ]", "[1, 2; 3, 4]", "матрица"), ("[;]", "[1; 2; 3]", "вектор"), ("Mᵀ", "^T", "транспонирование"),
        ("M⁻¹", "^-1", "обратная"), ("|M|", "det()", "определитель"), ("Mᵢⱼ", "[i, j]", "элемент"),
        ("E", "identity(3)", "единичная"), ("Ax=b", "lsolve(A, b)", "решить СЛАУ"), ("λ", "eigenvals(M)", "собственные числа"),
        ("rows", "rows(M)", "число строк"), ("cols", "cols(M)", "число столбцов"),
    ]),
    ("Решение", [
        ("Given", "given", "начало блока решения"), ("Find", "find(x, y)", "найти решение"),
        ("Max", "maximize(f, x, y)", "максимум"), ("Min", "minimize(f, x, y)", "минимум"),
        ("MinErr", "minerr(x, y)", "приближённое решение"), ("solve", " -> solve(x)", "символьное решение"),
        ("expand", " -> expand", "раскрыть"), ("factor", " -> factor", "разложить"), ("simplify", " -> simplify", "упростить"),
    ]),
    ("Программы", [
        ("{ }", "{\n  \n}", "программа"), ("←", " <- ", "локальное присваивание"), ("for", "for i in 1..n {\n  \n}", "цикл for"),
        ("while", "while cond {\n  \n}", "цикл while"), ("if", "if cond {\n  \n} else {\n  \n}", "условие"),
        ("if()", "if(cond, a, b)", "условное выражение"), ("return", "return ", "возврат"),
    ]),
    ("Греческие", [(g, g, g) for g in "αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"]),
    ("Исследование операций", [
        ("ЛП", "linprog(c, A, b)", "линейное программирование"), ("граф.", "lp_graphical(c, A, b)", "графический метод"),
        ("симплекс", "simplex(c, A, b)", "симплекс-таблицы"), ("транспорт", "transport_all(C, a, b)", "9 методов + потенциалы"),
        ("Гомори", "gomory(c, A, b)", "целочисленное ЛП"), ("Дейкстра", "dijkstra(E, s, t)", "кратчайший путь"),
        ("остов", "mst(E)", "минимальное дерево"), ("Свір", "sweep(L, Q, q)", "метод Свира"),
        ("TSP", "tsp(D)", "задача коммивояжёра"), ("склад", "warehouse(X, Y, Q)", "положение склада"),
        ("ДП", "allocate(F, unit=20)", "распределение ресурсов"),
    ]),
    ("Графики", [
        ("f(x)", "fplot(sin(x), x, 0, 10)", "график функции"), ("линия", "line(x, y)", "ломаная"),
        ("точки", "scatter(x, y)", "точки"), ("контур", "contour(f(x, y), x, a, b, y, c, d)", "линии уровня"),
        ("неявн.", "implicit(x^2 + y^2 == 1, x, -2, 2, y, -2, 2)", "неявная кривая"),
        ("область", "region(x + y <= 5, x, 0, 6, y, 0, 6)", "область"), ("3D", "surface(f(x, y), x, a, b, y, c, d)", "поверхность"),
        ("граф", "graph(E, path=d.path)", "сеть"),
    ]),
]


class PalettePanel(QWidget):
    insert = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        content.setObjectName("PaletteContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        for title, entries in PALETTE:
            header = QPushButton("▾  " + title)
            header.setObjectName("PaletteHeader")
            header.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            header.setCheckable(True)
            header.setChecked(True)
            page = QWidget()
            grid = QGridLayout(page)
            grid.setSpacing(4)
            grid.setContentsMargins(0, 2, 0, 8)
            cols = 3 if title in ("Исследование операций", "Графики", "Решение", "Анализ") else 5
            for k, (label, snippet, tip) in enumerate(entries):
                button = QPushButton(label)
                button.setObjectName("Palette")
                button.setToolTip(f"{tip}\n{snippet}")
                button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                button.clicked.connect(lambda _=False, s=snippet: self.insert.emit(s))
                grid.addWidget(button, k // cols, k % cols)

            def toggle(checked, p=page, h=header, t=title):
                p.setVisible(checked)
                h.setText(("▾  " if checked else "▸  ") + t)
            header.toggled.connect(toggle)
            layout.addWidget(header)
            layout.addWidget(page)
        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)


class FunctionsPanel(QWidget):
    insert = Signal(str)

    def __init__(self, registry: FunctionRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск функции…")
        self.search.textChanged.connect(self._fill)
        layout.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemDoubleClicked.connect(self._activate)
        self.tree.currentItemChanged.connect(self._show)
        layout.addWidget(self.tree, 3)
        self.doc = QTextBrowser()
        self.doc.setMaximumHeight(120)
        layout.addWidget(self.doc, 1)
        self._fill("")

    def _fill(self, text: str) -> None:
        self.tree.clear()
        text = text.lower().strip()
        for category, specs in self.registry.categories().items():
            parent = None
            for spec in specs:
                if text and text not in spec.name.lower() and text not in spec.doc.lower():
                    continue
                if parent is None:
                    parent = QTreeWidgetItem([category])
                    parent.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    self.tree.addTopLevelItem(parent)
                item = QTreeWidgetItem([spec.signature])
                item.setData(0, Qt.ItemDataRole.UserRole, spec.name)
                item.setToolTip(0, spec.doc)
                parent.addChild(item)
            if parent is not None:
                parent.setExpanded(bool(text))

    def _show(self, item, _prev=None) -> None:
        if item is None or item.data(0, Qt.ItemDataRole.UserRole) is None:
            return
        spec = self.registry.get(item.data(0, Qt.ItemDataRole.UserRole))
        if spec:
            self.doc.setHtml(f"<b>{spec.signature}</b><br/>{spec.doc}<br/><i>{spec.category}</i>")

    def _activate(self, item, column) -> None:
        name = item.data(0, Qt.ItemDataRole.UserRole)
        if name:
            spec = self.registry.get(name)
            sig = spec.signature.split(" |")[0] if spec else name + "()"
            self.insert.emit(sig)


class VariablesPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Фильтр…")
        self.filter.textChanged.connect(lambda _: self.refresh(self._env, self._fmt))
        layout.addWidget(self.filter)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Имя", "Тип", "Значение"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setColumnWidth(0, 90)
        self.tree.setColumnWidth(1, 80)
        layout.addWidget(self.tree)
        self._env = None
        self._fmt = NumberFormat()

    def refresh(self, env, fmt: NumberFormat) -> None:
        self._env, self._fmt = env, fmt
        self.tree.clear()
        if env is None:
            return
        needle = self.filter.text().lower()
        for name, value in sorted(env.vars.items(), key=lambda kv: kv[0].lower()):
            if needle and needle not in name.lower():
                continue
            text = format_value(value, fmt, max_items=8)
            short = text.replace("\n", " ")
            if len(short) > 80:
                short = short[:77] + "…"
            item = QTreeWidgetItem([name, type_name(value), short])
            item.setToolTip(2, text[:4000])
            self.tree.addTopLevelItem(item)


class MessagesPanel(QListWidget):
    region_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemClicked.connect(lambda it: self.region_selected.emit(it.data(Qt.ItemDataRole.UserRole)))

    def show_errors(self, errors: list[tuple[str, str]]) -> None:
        self.clear()
        for rid, message in errors:
            item = QListWidgetItem(f"⚠  {message}")
            item.setData(Qt.ItemDataRole.UserRole, rid)
            self.addItem(item)
        if not errors:
            item = QListWidgetItem("✓  Ошибок нет")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.addItem(item)
