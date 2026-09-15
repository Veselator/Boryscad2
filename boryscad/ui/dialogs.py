"""Dialogs: plot properties, settings / theme editor, syntax help."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFontComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton,
                               QSpinBox, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ..core.document import PlotRegion
from ..core.theme import Theme, available_themes, save_user_theme
from ..render.qt_backend import qcolor
from .settings import AppSettings


class PlotDialog(QDialog):
    preview_requested = Signal(object)

    def __init__(self, region: PlotRegion, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Свойства графика")
        self.resize(640, 560)
        self.region = PlotRegion.from_dict(region.to_dict())
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.title = QLineEdit(region.title)
        self.xlabel = QLineEdit(region.xlabel)
        self.ylabel = QLineEdit(region.ylabel)
        self.zlabel = QLineEdit(region.zlabel)
        self.xlim = QLineEdit(region.xlim)
        self.xlim.setPlaceholderText("авто, например 0..10")
        self.ylim = QLineEdit(region.ylim)
        self.ylim.setPlaceholderText("авто")
        self.mode = QComboBox()
        self.mode.addItems(["2d", "3d", "polar"])
        self.mode.setCurrentText(region.mode)
        self.xscale = QComboBox()
        self.xscale.addItems(["linear", "log"])
        self.xscale.setCurrentText(region.xscale)
        self.yscale = QComboBox()
        self.yscale.addItems(["linear", "log"])
        self.yscale.setCurrentText(region.yscale)
        form.addRow("Заголовок", self.title)
        axes = QHBoxLayout()
        axes.addWidget(self.xlabel)
        axes.addWidget(self.ylabel)
        axes.addWidget(self.zlabel)
        form.addRow("Подписи осей x / y / z", axes)
        lims = QHBoxLayout()
        lims.addWidget(self.xlim)
        lims.addWidget(self.ylim)
        form.addRow("Пределы x / y", lims)
        kinds = QHBoxLayout()
        kinds.addWidget(self.mode)
        kinds.addWidget(QLabel("шкала x"))
        kinds.addWidget(self.xscale)
        kinds.addWidget(QLabel("y"))
        kinds.addWidget(self.yscale)
        form.addRow("Тип", kinds)
        flags = QHBoxLayout()
        self.grid = QCheckBox("сетка")
        self.grid.setChecked(region.grid)
        self.legend = QCheckBox("легенда")
        self.legend.setChecked(region.legend)
        self.equal = QCheckBox("равный масштаб")
        self.equal.setChecked(region.equal)
        for w in (self.grid, self.legend, self.equal):
            flags.addWidget(w)
        flags.addStretch()
        form.addRow("", flags)
        size = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(120, 2400)
        self.width_spin.setValue(int(region.width))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(100, 2400)
        self.height_spin.setValue(int(region.height))
        size.addWidget(self.width_spin)
        size.addWidget(QLabel("×"))
        size.addWidget(self.height_spin)
        size.addStretch()
        form.addRow("Размер, px", size)
        layout.addLayout(form)
        layout.addWidget(QLabel("Серии — по одному выражению в строке:"))
        self.series = QPlainTextEdit("\n".join(region.series))
        mono = QFont("Cascadia Mono")
        mono.setPointSize(10)
        self.series.setFont(mono)
        layout.addWidget(self.series, 1)
        hint = QLabel("fplot(sin(x), x, 0, 10, label=\"sin\") · line(x, y) · scatter(x, y) · contour(f(x,y), x, a, b, y, c, d)"
                      " · implicit(eq, x, a, b, y, c, d) · region(cond, …) · surface(…) · graph(E, path=p) · lp_graph(c, A, b)")
        hint.setWordWrap(True)
        hint.setObjectName("Muted")
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        apply = buttons.addButton("Применить", QDialogButtonBox.ButtonRole.ApplyRole)
        apply.clicked.connect(lambda: self.preview_requested.emit(self.result_region()))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_region(self) -> PlotRegion:
        r = self.region
        r.title, r.xlabel, r.ylabel, r.zlabel = self.title.text(), self.xlabel.text(), self.ylabel.text(), self.zlabel.text()
        r.xlim, r.ylim = self.xlim.text().strip(), self.ylim.text().strip()
        r.mode, r.xscale, r.yscale = self.mode.currentText(), self.xscale.currentText(), self.yscale.currentText()
        r.grid, r.legend, r.equal = self.grid.isChecked(), self.legend.isChecked(), self.equal.isChecked()
        r.width, r.height = float(self.width_spin.value()), float(self.height_spin.value())
        r.series = [s.strip() for s in self.series.toPlainText().split("\n") if s.strip()]
        return r


COLOR_KEYS = [
    ("accent", "Акцент"), ("window", "Окно"), ("sidebar", "Боковые панели"), ("surface", "Поверхности"),
    ("text", "Текст интерфейса"), ("desk", "Фон рабочего стола"), ("page", "Лист"), ("page_grid", "Сетка листа"),
    ("math_variable", "Переменные"), ("math_number", "Числа"), ("math_operator", "Операторы"),
    ("math_function", "Функции"), ("math_keyword", "Ключевые слова"), ("math_result", "Результаты"),
    ("math_result_bg", "Фон результатов"), ("math_error", "Ошибки"), ("text_body", "Текст документа"),
    ("heading", "Заголовки"), ("plot_bg", "Фон графиков"),
]


class SettingsDialog(QDialog):
    applied = Signal(object, object)   # (AppSettings, Theme)

    def __init__(self, settings: AppSettings, theme: Theme, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки и стиль")
        self.resize(760, 600)
        self.settings = replace(settings)
        self.themes = available_themes()
        self.theme = theme.copy()
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # ---- appearance
        page = QWidget()
        grid = QHBoxLayout(page)
        left = QVBoxLayout()
        left.addWidget(QLabel("Темы"))
        self.theme_list = QListWidget()
        for name, t in self.themes.items():
            item = QListWidgetItem(("🌙  " if t.dark else "☀  ") + name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.theme_list.addItem(item)
            if name == theme.name:
                self.theme_list.setCurrentItem(item)
        self.theme_list.currentItemChanged.connect(self._pick_theme)
        left.addWidget(self.theme_list)
        row = QHBoxLayout()
        imp = QPushButton("Импорт…")
        imp.clicked.connect(self._import)
        exp = QPushButton("Экспорт…")
        exp.clicked.connect(self._export)
        row.addWidget(imp)
        row.addWidget(exp)
        left.addLayout(row)
        grid.addLayout(left, 1)
        right = QVBoxLayout()
        colors = QGroupBox("Цвета")
        cgrid = QGridLayout(colors)
        self.color_buttons: dict[str, QPushButton] = {}
        for k, (key, label) in enumerate(COLOR_KEYS):
            btn = QPushButton()
            btn.setFixedSize(46, 22)
            btn.clicked.connect(lambda _=False, kk=key: self._choose_color(kk))
            self.color_buttons[key] = btn
            cgrid.addWidget(QLabel(label), k // 2, (k % 2) * 2)
            cgrid.addWidget(btn, k // 2, (k % 2) * 2 + 1)
        right.addWidget(colors)
        name_row = QHBoxLayout()
        self.theme_name = QLineEdit(self.theme.name)
        save = QPushButton("Сохранить как тему")
        save.clicked.connect(self._save_theme)
        name_row.addWidget(QLabel("Имя темы"))
        name_row.addWidget(self.theme_name)
        name_row.addWidget(save)
        right.addLayout(name_row)
        grid.addLayout(right, 2)
        tabs.addTab(page, "Оформление")

        # ---- fonts
        fonts = QWidget()
        form = QFormLayout(fonts)
        self.math_font = QFontComboBox()
        self.math_font.setCurrentFont(QFont(settings.math_font or theme.font("math")))
        self.math_size = QDoubleSpinBox()
        self.math_size.setRange(8, 48)
        self.math_size.setValue(settings.math_size or float(theme.font("math_size", 17)))
        self.ui_font = QFontComboBox()
        self.ui_font.setCurrentFont(QFont(settings.ui_font or theme.font("ui")))
        self.ui_size = QSpinBox()
        self.ui_size.setRange(7, 20)
        self.ui_size.setValue(settings.ui_size or int(theme.font("ui_size", 10)))
        self.text_font = QFontComboBox()
        self.text_font.setCurrentFont(QFont(theme.font("text")))
        form.addRow("Шрифт формул", self.math_font)
        form.addRow("Размер формул, px", self.math_size)
        form.addRow("Шрифт текста документа", self.text_font)
        form.addRow("Шрифт интерфейса", self.ui_font)
        form.addRow("Размер интерфейса, pt", self.ui_size)
        tabs.addTab(fonts, "Шрифты")

        # ---- calculation
        calc = QWidget()
        form = QFormLayout(calc)
        self.precision = QSpinBox()
        self.precision.setRange(0, 15)
        self.precision.setValue(settings.precision)
        self.exp = QSpinBox()
        self.exp.setRange(1, 15)
        self.exp.setValue(settings.exp_threshold)
        self.auto = QCheckBox("Автоматический пересчёт")
        self.auto.setChecked(settings.auto_calc)
        self.page_width = QSpinBox()
        self.page_width.setRange(400, 3000)
        self.page_width.setValue(int(settings.page_width))
        self.grid_cb = QCheckBox("Показывать сетку")
        self.grid_cb.setChecked(settings.show_grid)
        self.snap = QCheckBox("Привязка к сетке")
        self.snap.setChecked(settings.snap)
        form.addRow("Знаков после запятой", self.precision)
        form.addRow("Порог экспоненциальной записи", self.exp)
        form.addRow("", self.auto)
        form.addRow("Ширина листа, px", self.page_width)
        form.addRow("", self.grid_cb)
        form.addRow("", self.snap)
        tabs.addTab(calc, "Вычисления и лист")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
                                   | QDialogButtonBox.StandardButton.Apply)
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply)
        layout.addWidget(buttons)
        self._refresh_colors()

    def _pick_theme(self, item, _prev=None) -> None:
        if item is None:
            return
        self.theme = self.themes[item.data(Qt.ItemDataRole.UserRole)].copy()
        self.theme_name.setText(self.theme.name)
        self.math_font.setCurrentFont(QFont(self.theme.font("math")))
        self.math_size.setValue(float(self.theme.font("math_size", 17)))
        self.text_font.setCurrentFont(QFont(self.theme.font("text")))
        self._refresh_colors()
        self._apply()

    def _refresh_colors(self) -> None:
        for key, btn in self.color_buttons.items():
            c = qcolor(self.theme.color(key))
            btn.setStyleSheet(f"background: rgba({c.red()},{c.green()},{c.blue()},{c.alpha()}); border-radius: 5px;"
                              f" border: 1px solid #888;")

    def _choose_color(self, key: str) -> None:
        current = qcolor(self.theme.color(key))
        color = QColorDialog.getColor(current, self, "Цвет", QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if color.isValid():
            self.theme.colors[key] = f"#{color.red():02x}{color.green():02x}{color.blue():02x}" + (
                f"{color.alpha():02x}" if color.alpha() < 255 else "")
            self._refresh_colors()
            self._apply()

    def _collect(self) -> tuple[AppSettings, Theme]:
        s = self.settings
        t = self.theme
        t.fonts["math"] = self.math_font.currentFont().family()
        t.fonts["math_size"] = self.math_size.value()
        t.fonts["text"] = self.text_font.currentFont().family()
        s.theme = t.name
        s.math_font, s.math_size = "", 0.0
        s.ui_font, s.ui_size = self.ui_font.currentFont().family(), self.ui_size.value()
        s.precision, s.exp_threshold = self.precision.value(), self.exp.value()
        s.auto_calc = self.auto.isChecked()
        s.page_width = float(self.page_width.value())
        s.show_grid, s.snap = self.grid_cb.isChecked(), self.snap.isChecked()
        return s, t

    def _apply(self) -> None:
        s, t = self._collect()
        self.applied.emit(replace(s), t.copy())

    def _ok(self) -> None:
        self._apply()
        self.accept()

    def _save_theme(self) -> None:
        name = self.theme_name.text().strip() or "Custom"
        self.theme.name = name
        path = save_user_theme(self._collect()[1])
        self.themes = available_themes()
        item = QListWidgetItem(("🌙  " if self.theme.dark else "☀  ") + name)
        item.setData(Qt.ItemDataRole.UserRole, name)
        existing = [self.theme_list.item(i) for i in range(self.theme_list.count())
                    if self.theme_list.item(i).data(Qt.ItemDataRole.UserRole) == name]
        if not existing:
            self.theme_list.blockSignals(True)
            self.theme_list.addItem(item)
            self.theme_list.setCurrentItem(item)
            self.theme_list.blockSignals(False)
        self._apply()
        QMessageBox.information(self, "Тема сохранена", f"Тема «{name}» сохранена:\n{path}")

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Импорт темы", "", "Тема Boryscad (*.json)")
        if not path:
            return
        try:
            theme = Theme.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Ошибка", str(exc))
            return
        self.theme = theme
        save_user_theme(theme)
        self.themes = available_themes()
        self.theme_name.setText(theme.name)
        self._refresh_colors()
        self._apply()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт темы", f"{self.theme.name}.json", "Тема Boryscad (*.json)")
        if path:
            Path(path).write_text(json.dumps(self._collect()[1].to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


SYNTAX_HELP = """
<h2>Язык формул Boryscad</h2>
<table cellpadding="4">
<tr><td><code>x := 5</code></td><td>определение переменной (≔)</td></tr>
<tr><td><code>f(x, y) := x^2 + y</code></td><td>определение функции</td></tr>
<tr><td><code>f(2, 3) =</code></td><td>вычисление (результат справа)</td></tr>
<tr><td><code>(x + 1)^2 -&gt; expand</code></td><td>символьное вычисление: expand, factor, simplify, solve(x), diff(x), integrate(x), series(x, 0, 5), float(6), substitute(x = 2)</td></tr>
<tr><td><code>i := 0..10</code>, <code>t := 0..1 step 0.1</code>, <code>t := 0, 0.1 .. 1</code></td><td>диапазоны</td></tr>
<tr><td><code>v[i] := i^2</code></td><td>векторизованное присваивание по диапазону (ORIGIN := 1 — нумерация с 1)</td></tr>
<tr><td><code>A := [1, 2; 3, 4]</code></td><td>матрица; <code>A^T</code>, <code>A^-1</code>, <code>|A|</code>, <code>A[0, 1]</code></td></tr>
<tr><td><code>given</code> … <code>find(x, y)</code></td><td>блок решения: уравнения <code>==</code> (или <code>=</code>) и неравенства <code>&lt;=</code>, <code>&gt;=</code></td></tr>
<tr><td><code>maximize(f, x, y)</code>, <code>minimize(f, x, y)</code></td><td>экстремум с ограничениями блока given (ЛП — точно, НЛП — Лагранж/KKT + численно)</td></tr>
<tr><td><code>deriv(f(x), x)</code>, <code>integral(f(x), x, a, b)</code>, <code>sum(k^2, k, 1, n)</code></td><td>анализ (рисуются как d/dx, ∫, Σ)</td></tr>
<tr><td><code>g(n) := { s &lt;- 0; for k in 1..n { s &lt;- s + k }; s }</code></td><td>программы: for, while, if/else, return, break</td></tr>
<tr><td><code>r := simplex(c, A, b)</code>, <code>r.x =</code></td><td>функции возвращают записи с полями</td></tr>
<tr><td><code>// комментарий</code></td><td>комментарий</td></tr>
</table>
<h3>Работа с листом</h3>
<p>Щёлкните по пустому месту листа и начинайте печатать — появится математический регион. Символ <b>"</b> создаёт текстовый регион (Markdown: **жирный**, *курсив*, # заголовок, {{x}} — значение переменной).
Enter — завершить, Shift+Enter — новая строка, Esc — выйти. Двойной щелчок — редактировать. Регионы перетаскиваются мышью, график и изображение масштабируются за угол.
Вычисление идёт сверху вниз и слева направо, как в Mathcad.</p>
<h3>Терминал</h3>
<p><code>boryscad render файл.bmath -o отчёт.pdf</code> · <code>boryscad run файл.bmath</code> · <code>boryscad formula "x := √(a^2+b^2)" -o f.png</code> · <code>boryscad functions</code></p>
"""


class HelpDialog(QDialog):
    def __init__(self, parent=None, html: str = SYNTAX_HELP, title: str = "Справка"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 620)
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(html)
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
