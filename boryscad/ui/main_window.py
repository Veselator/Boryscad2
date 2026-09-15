"""Main application window."""
from __future__ import annotations

import base64
from importlib import resources
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QGuiApplication, QImage, QKeySequence
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QLabel, QMainWindow, QMenu, QMessageBox,
                               QSlider, QTabWidget, QToolBar, QToolButton, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..core.bmath import image_region, image_size
from ..core.document import Document, ImageRegion, MathRegion, PageBreakRegion, PlotRegion, TextRegion
from ..core.io import load_document, save_document
from ..core.theme import Theme, load_theme
from ..engine.formatting import NumberFormat
from ..engine.worksheet import DocumentResult, WorksheetEngine
from ..render.regions import RenderContext
from .calc import CalculationService
from .controller import DocumentController
from .dialogs import HelpDialog, PlotDialog, SettingsDialog
from .panels import FunctionsPanel, MessagesPanel, PalettePanel, VariablesPanel
from .settings import AppSettings
from .style import build_qss, icon
from .worksheet import WorksheetScene, WorksheetView

FILE_FILTER = "Документы Boryscad (*.bcad *.bmath);;Worksheet (*.bcad);;Текстовый worksheet (*.bmath)"


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self.theme: Theme = self._effective_theme(load_theme(settings.theme))
        self.engine = WorksheetEngine()
        self.controller = DocumentController(Document())
        self.results: DocumentResult | None = None
        self.ctx = RenderContext(self.theme, NumberFormat(settings.precision, settings.exp_threshold), self.engine.registry,
                                 settings.page_width, 48.0, self._scale())
        self.scene = WorksheetScene(self.controller, self.ctx, self)
        self.scene.snap = settings.snap
        self.scene.grid_step = settings.grid_step
        self.scene.show_grid = settings.show_grid
        self.view = WorksheetView(self.scene)
        self.calc = CalculationService(self.engine, self)

        central = QWidget()
        central.setObjectName("Central")
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self.setCentralWidget(central)

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_docks()
        self._build_statusbar()
        self._connect()
        self.apply_theme(self.theme)
        self.view.set_zoom(settings.zoom or 1.0)
        self.resize(1440, 900)
        if settings.geometry:
            self.restoreGeometry(QByteArray.fromBase64(settings.geometry.encode()))
        if settings.window_state:
            self.restoreState(QByteArray.fromBase64(settings.window_state.encode()))
        self.new_document(welcome=True)

    # ----------------------------------------------------------------- setup
    def _scale(self) -> float:
        screen = QGuiApplication.primaryScreen()
        return max(1.0, (screen.devicePixelRatio() if screen else 1.0)) * 1.5

    def _effective_theme(self, theme: Theme) -> Theme:
        t = theme.copy()
        if self.settings.math_font:
            t.fonts["math"] = self.settings.math_font
        if self.settings.math_size:
            t.fonts["math_size"] = self.settings.math_size
        return t

    def _act(self, text: str, slot, shortcut=None, icon_name: str | None = None, tip: str = "", checkable=False) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        action.setData(icon_name)
        return action

    def _build_actions(self) -> None:
        a = self._act
        self.a_new = a("Новый", lambda: self.new_document(), QKeySequence.StandardKey.New, "new", "Новый документ")
        self.a_open = a("Открыть…", self.open_dialog, QKeySequence.StandardKey.Open, "open", "Открыть документ")
        self.a_save = a("Сохранить", self.save, QKeySequence.StandardKey.Save, "save", "Сохранить")
        self.a_save_as = a("Сохранить как…", self.save_as, "Ctrl+Shift+S")
        self.a_export = a("Экспорт…", self.export_dialog, "Ctrl+E", "export", "Экспорт в PDF, HTML, PNG, SVG")
        self.a_quit = a("Выход", self.close, QKeySequence.StandardKey.Quit)
        self.a_undo = self.controller.undo_stack.createUndoAction(self, "Отменить")
        self.a_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.a_undo.setData("undo")
        self.a_redo = self.controller.undo_stack.createRedoAction(self, "Повторить")
        self.a_redo.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")])
        self.a_redo.setData("redo")
        self.a_cut = a("Вырезать", lambda: self.copy(cut=True), QKeySequence.StandardKey.Cut)
        self.a_copy = a("Копировать", lambda: self.copy(), QKeySequence.StandardKey.Copy, None,
                        "Копировать регионы (в другие программы — как изображение)")
        self.a_paste = a("Вставить", self.paste, QKeySequence.StandardKey.Paste)
        self.a_duplicate = a("Дублировать", self.duplicate, "Ctrl+D", None, "Дублировать выделенные регионы")
        self.a_delete = a("Удалить", self.delete_selected, None, "delete", "Удалить выделенные регионы")
        self.a_select_all = a("Выделить всё", self.select_all, QKeySequence.StandardKey.SelectAll)
        self.a_math = a("Формула", lambda: self.insert_region("math"), "Ctrl+M", "math", "Математический регион")
        self.a_text = a("Текст", lambda: self.insert_region("text"), "Ctrl+T", "text", "Текстовый регион (Markdown)")
        self.a_heading = a("Заголовок", lambda: self.insert_region("text", style="h2"), "Ctrl+H")
        self.a_plot = a("График", self.insert_plot, "Ctrl+G", "plot", "Вставить график")
        self.a_image = a("Изображение…", self.insert_image, "Ctrl+I", "image", "Вставить изображение")
        self.a_given = a("Блок решения", self.insert_given, None, "given", "Given … find/maximize/minimize")
        self.a_pagebreak = a("Разрыв страницы", lambda: self.insert_region("pagebreak"), None, "pagebreak")
        self.a_calc = a("Пересчитать", lambda: self.recalculate(True), "F9", "calc", "Пересчитать документ (F9)")
        self.a_auto = a("Автопересчёт", self.toggle_auto, None, None, "", checkable=True)
        self.a_auto.setChecked(self.settings.auto_calc)
        self.a_interrupt = a("Прервать вычисление", lambda: self.calc.interrupt(), "Ctrl+Break")
        self.a_separate = a("Раздвинуть регионы", self.scene.separate_regions, "Ctrl+Shift+R", "align")
        self.a_align_left = a("Выровнять по левому краю", lambda: self.scene.align("left"))
        self.a_align_top = a("Выровнять по верху", lambda: self.scene.align("top"))
        self.a_zoom_in = a("Увеличить", lambda: self.view.set_zoom(self.view.zoom * 1.15), QKeySequence.StandardKey.ZoomIn, "zoomin")
        self.a_zoom_out = a("Уменьшить", lambda: self.view.set_zoom(self.view.zoom / 1.15), QKeySequence.StandardKey.ZoomOut, "zoomout")
        self.a_zoom_reset = a("Масштаб 100%", lambda: self.view.set_zoom(1.0), "Ctrl+0")
        self.a_settings = a("Настройки и стиль…", self.open_settings, "Ctrl+,", "settings", "Темы, цвета, шрифты")
        self.a_help = a("Синтаксис", lambda: HelpDialog(self).exec(), "F1", "help")
        self.a_functions_help = a("Список функций", self.show_function_reference)
        self.a_about = a("О программе", self.about)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("Файл")
        m.addActions([self.a_new, self.a_open])
        self.recent_menu = m.addMenu("Недавние")
        self.examples_menu = m.addMenu("Примеры (лабораторные)")
        m.addSeparator()
        m.addActions([self.a_save, self.a_save_as, self.a_export])
        m.addSeparator()
        m.addAction(self.a_quit)
        self._fill_examples()
        self._fill_recent()
        m = mb.addMenu("Правка")
        m.addActions([self.a_undo, self.a_redo])
        m.addSeparator()
        m.addActions([self.a_cut, self.a_copy, self.a_paste, self.a_duplicate, self.a_delete, self.a_select_all])
        m.addSeparator()
        m.addActions([self.a_separate, self.a_align_left, self.a_align_top])
        m = mb.addMenu("Вставка")
        m.addActions([self.a_math, self.a_text, self.a_heading, self.a_plot, self.a_image, self.a_given, self.a_pagebreak])
        m = mb.addMenu("Вычисления")
        m.addActions([self.a_calc, self.a_auto, self.a_interrupt])
        self.view_menu = mb.addMenu("Вид")
        self.view_menu.addActions([self.a_zoom_in, self.a_zoom_out, self.a_zoom_reset])
        self.view_menu.addSeparator()
        self.theme_menu = self.view_menu.addMenu("Тема")
        self._fill_theme_menu()
        self.view_menu.addAction(self.a_settings)
        self.view_menu.addSeparator()
        m = mb.addMenu("Справка")
        m.addActions([self.a_help, self.a_functions_help, self.a_about])

    def _build_toolbar(self) -> None:
        tb = QToolBar("Главная")
        tb.setObjectName("MainToolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(tb)
        for act in (self.a_new, self.a_open, self.a_save, self.a_export):
            tb.addAction(act)
        tb.addSeparator()
        tb.addActions([self.a_undo, self.a_redo])
        tb.addSeparator()
        tb2 = tb
        for act in (self.a_math, self.a_text, self.a_plot, self.a_image, self.a_given):
            tb2.addAction(act)
        for act in (self.a_math, self.a_text, self.a_plot):
            btn = tb2.widgetForAction(act)
            if isinstance(btn, QToolButton):
                btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addSeparator()
        tb.addAction(self.a_separate)
        tb.addAction(self.a_delete)
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding, spacer.sizePolicy().verticalPolicy().Preferred)
        tb.addWidget(spacer)
        tb.addAction(self.a_calc)
        calc_btn = tb.widgetForAction(self.a_calc)
        if isinstance(calc_btn, QToolButton):
            calc_btn.setObjectName("Accent")
            calc_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.addAction(self.a_settings)
        self.toolbar = tb

    def _build_docks(self) -> None:
        self.palette = PalettePanel()
        self.functions = FunctionsPanel(self.engine.registry)
        left = QTabWidget()
        left.addTab(self.palette, "Палитра")
        left.addTab(self.functions, "Функции")
        self.left_dock = QDockWidget("Инструменты", self)
        self.left_dock.setObjectName("ToolsDock")
        self.left_dock.setWidget(left)
        self.left_dock.setMinimumWidth(270)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.left_dock)

        self.variables = VariablesPanel()
        self.messages = MessagesPanel()
        right = QTabWidget()
        right.addTab(self.variables, "Переменные")
        right.addTab(self.messages, "Сообщения")
        self.right_tabs = right
        self.right_dock = QDockWidget("Обозреватель", self)
        self.right_dock.setObjectName("InspectorDock")
        self.right_dock.setWidget(right)
        self.right_dock.setMinimumWidth(290)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_dock)
        self.view_menu.addAction(self.left_dock.toggleViewAction())
        self.view_menu.addAction(self.right_dock.toggleViewAction())

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.status_calc = QLabel("Готово")
        self.status_msg = QLabel("")
        self.zoom_label = QLabel("100%")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(30, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(140)
        self.zoom_slider.valueChanged.connect(lambda v: self.view.set_zoom(v / 100))
        sb.addWidget(self.status_calc)
        sb.addWidget(self.status_msg, 1)
        sb.addPermanentWidget(self.zoom_slider)
        sb.addPermanentWidget(self.zoom_label)

    def _connect(self) -> None:
        self.controller.changed.connect(self._on_changed)
        self.controller.restored.connect(self._on_restored)
        self.controller.modified_changed.connect(lambda _: self._update_title())
        self.calc.started.connect(lambda: self.status_calc.setText("⏳ Вычисление…"))
        self.calc.finished.connect(self._on_results)
        self.calc.failed.connect(lambda msg: self.status_calc.setText("⚠ Ошибка вычисления") or print(msg))
        self.palette.insert.connect(self.scene.insert_at_cursor)
        self.functions.insert.connect(self.scene.insert_at_cursor)
        self.messages.region_selected.connect(self.select_region)
        self.scene.edit_plot_requested.connect(self.edit_plot)
        self.scene.edit_image_requested.connect(self.edit_image)
        self.scene.status.connect(self.status_msg.setText)
        self.scene.context_menu_requested.connect(self.show_context_menu)
        self.scene.ensure_visible.connect(lambda rect: self.view.ensureVisible(rect, 40, 40))
        self.view.zoom_changed.connect(self._on_zoom)
        self.view.files_dropped.connect(self._on_files_dropped)

    # --------------------------------------------------------------- theme
    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        ui_font = self.settings.ui_font or theme.font("ui", "Segoe UI")
        ui_size = self.settings.ui_size or int(theme.font("ui_size", 10))
        QApplication.instance().setStyleSheet(build_qss(theme, ui_font, ui_size))
        self.ctx.theme = theme
        self.ctx._builder = None
        self.ctx.page_width = self.settings.page_width
        self.ctx.fmt = NumberFormat(self.settings.precision, self.settings.exp_threshold)
        for action in self.findChildren(QAction):
            name = action.data()
            if isinstance(name, str) and name:
                action.setIcon(icon(name, theme, "accent_text" if action is self.a_calc else "text"))
        self.scene.relayout_all()
        self.scene.update_scene_rect()

    def _fill_theme_menu(self) -> None:
        from ..core.theme import available_themes
        self.theme_menu.clear()
        for name in available_themes():
            act = self.theme_menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(name == self.theme.name)
            act.triggered.connect(lambda _=False, n=name: self.set_theme(n))

    def set_theme(self, name: str) -> None:
        self.settings.theme = name
        self.apply_theme(self._effective_theme(load_theme(name)))
        self._fill_theme_menu()
        self.recalculate(True)

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self.theme, self)
        dialog.applied.connect(self._apply_settings)
        dialog.exec()
        self._fill_theme_menu()

    def _apply_settings(self, settings: AppSettings, theme: Theme) -> None:
        recent, geometry = self.settings.recent, self.settings.geometry
        self.settings = settings
        self.settings.recent, self.settings.geometry = recent, geometry
        self.scene.snap, self.scene.show_grid = settings.snap, settings.show_grid
        self.a_auto.setChecked(settings.auto_calc)
        self.controller.document.settings.precision = settings.precision
        self.controller.document.settings.exp_threshold = settings.exp_threshold
        self.controller.document.settings.page_width = settings.page_width
        self.apply_theme(theme)
        self.settings.save()
        self.recalculate(True)

    # ---------------------------------------------------------- documents
    def new_document(self, welcome: bool = False) -> None:
        if not welcome and not self._confirm_discard():
            return
        doc = Document()
        doc.settings.precision = self.settings.precision
        doc.settings.page_width = self.settings.page_width
        if welcome:
            doc.regions = [
                TextRegion(x=48, y=40, text="Добро пожаловать в Boryscad", style="h1"),
                TextRegion(x=48, y=112, text="Щёлкните по листу и начинайте печатать формулу. Символ **\"** создаёт текст. "
                                            "Enter — готово, Shift+Enter — новая строка. Примеры лабораторных — в меню *Файл → Примеры*."),
                MathRegion(x=48, y=190, source="a := 3\nb := 4\nc := sqrt(a^2 + b^2)\nc ="),
                MathRegion(x=300, y=190, source="f(x) := x^3 - 6*x^2 + 11*x - 6\nf(x) -> factor\nintegral(f(x), x, 0, 3) ="),
                MathRegion(x=48, y=330, source="A := [2, 1; 1, 3]\nA^-1 =\n|A| ="),
                PlotRegion(x=360, y=330, series=["fplot(f(t), t, 0, 4, label=\"f(t)\")", "hline(0, color=\"gray\", width=1)"],
                           title="f(x) = (x−1)(x−2)(x−3)", width=440, height=280),
            ]
        self.controller.load(doc, None)
        self._update_title()

    def open_dialog(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Открыть", str(Path.home()), FILE_FILTER)
        if path:
            self.open_file(path)

    def open_file(self, path: str, as_template: bool = False) -> None:
        try:
            doc = load_document(path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка открытия", f"{path}\n\n{exc}")
            return
        if doc.flow_layout:
            self._materialize_layout(doc)
        self.controller.load(doc, None if as_template else path)
        if not as_template:
            self.settings.add_recent(path)
            self._fill_recent()
        self._update_title()
        self.recalculate(True)

    def _materialize_layout(self, doc: Document) -> None:
        from ..render.document import DocumentRenderer
        renderer = DocumentRenderer(self.theme, self.engine, self._scale())
        results = renderer.calculate(doc)
        doc.settings.page_width = max(doc.settings.page_width, self.settings.page_width)
        renderer.layout(doc, results)
        doc.flow_layout = False
        doc.rows = []
        self.results = results

    def save(self) -> bool:
        if not self.controller.path or self.controller.path.lower().endswith(".bmath") and self._has_images():
            return self.save_as()
        return self._save_to(self.controller.path)

    def _has_images(self) -> bool:
        return any(isinstance(r, ImageRegion) and not r.source_path for r in self.controller.document.regions)

    def save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить как", self.controller.path or str(Path.home() / "worksheet.bcad"),
                                              FILE_FILTER)
        if not path:
            return False
        if not Path(path).suffix:
            path += ".bcad"
        return self._save_to(path)

    def _save_to(self, path: str) -> bool:
        self.scene.end_all_edits()
        try:
            save_document(self.controller.document, path)
        except OSError as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))
            return False
        self.controller.path = path
        self.controller.undo_stack.setClean()
        self.settings.add_recent(path)
        self._fill_recent()
        self._update_title()
        self.status_msg.setText(f"Сохранено: {path}")
        return True

    def export_dialog(self) -> None:
        self.scene.end_all_edits()
        base = Path(self.controller.path).with_suffix("") if self.controller.path else Path.home() / "worksheet"
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт", str(base) + ".pdf",
                                              "PDF (*.pdf);;HTML (*.html);;PNG (*.png);;SVG (*.svg)")
        if not path:
            return
        from ..render.document import DocumentRenderer
        renderer = DocumentRenderer(self.theme, self.engine, scale=2.0)
        doc = Document.from_dict(self.controller.document.to_dict())
        try:
            results = self.results if self.results is not None else renderer.calculate(doc)
            renderer.export(doc, path, results)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))
            return
        self.status_msg.setText(f"Экспортировано: {path}")

    def _confirm_discard(self) -> bool:
        if self.controller.undo_stack.isClean():
            return True
        answer = QMessageBox.question(self, APP_NAME, "Сохранить изменения?",
                                      QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                                      | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        for path in self.settings.recent:
            act = self.recent_menu.addAction(path)
            act.triggered.connect(lambda _=False, p=path: self._confirm_discard() and self.open_file(p))
        self.recent_menu.setEnabled(bool(self.settings.recent))

    def _fill_examples(self) -> None:
        self.examples_menu.clear()
        folder = resources.files("boryscad.examples")
        for entry in sorted(folder.iterdir(), key=lambda e: e.name):
            if entry.name.endswith(".bmath"):
                act = self.examples_menu.addAction(entry.name.replace(".bmath", "").replace("_", " "))
                act.triggered.connect(lambda _=False, p=str(entry): self._confirm_discard() and self.open_file(p, as_template=True))

    def _update_title(self) -> None:
        name = Path(self.controller.path).name if self.controller.path else (self.controller.document.title or "Без имени")
        mark = "" if self.controller.undo_stack.isClean() else " •"
        self.setWindowTitle(f"{name}{mark} — {APP_NAME}")

    # ----------------------------------------------------------- inserts
    def insert_region(self, kind: str, **fields) -> None:
        pos = self.scene.cursor_pos()
        if kind == "pagebreak":
            self.scene.create_region("pagebreak", pos, edit=False)
        else:
            self.scene.create_region(kind, pos, "", **fields)
        self.view.setFocus()

    def insert_given(self) -> None:
        pos = self.scene.cursor_pos()
        self.scene.create_region("math", pos, "x := 1\ny := 1\ngiven\nx^2 + y^2 == 4\nx - y == 1\nfind(x, y) =")

    def insert_plot(self) -> None:
        pos = self.scene.cursor_pos()
        region = PlotRegion(x=pos.x(), y=pos.y(), series=["fplot(sin(x), x, -2*π, 2*π, label=\"sin x\")"], title="")
        dialog = PlotDialog(region, self)
        if dialog.exec():
            self.controller.add_region(dialog.result_region(), "Insert plot")
            self.scene.sync()

    def insert_image(self, pos: QPointF | None = None) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Изображение", str(Path.home()),
                                              "Изображения (*.png *.jpg *.jpeg *.bmp *.gif *.svg *.webp)")
        if path:
            self._add_image_file(path, pos or self.scene.cursor_pos())

    def _add_image_file(self, path: str, pos: QPointF) -> None:
        region = image_region(Path(path))
        region.x, region.y = pos.x(), pos.y()
        if isinstance(region, ImageRegion) and region.mime == "image/svg+xml":
            image = QImage(path)
            region = self._image_from_qimage(image, pos) if not image.isNull() else region
        self.controller.add_region(region, "Insert image")
        self.scene.sync()

    def _image_from_qimage(self, image: QImage, pos: QPointF) -> ImageRegion:
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "PNG")
        data = base64.b64encode(bytes(buffer.data())).decode("ascii")
        scale = min(1.0, 640 / max(1, image.width()))
        return ImageRegion(x=pos.x(), y=pos.y(), data=data, mime="image/png", width=image.width() * scale,
                           height=image.height() * scale)

    def _active_editor(self):
        item = self.scene.editing_item()
        return getattr(item, "editor", None) if item is not None else None

    def copy(self, cut: bool = False) -> None:
        editor = self._active_editor()
        if editor is not None:
            editor.cut() if cut else editor.copy()
            return
        if self.scene.copy_selection(cut=cut):
            n = len(self.scene.selected_regions()) if not cut else 0
            self.status_msg.setText("Вырезано" if cut else f"Скопировано регионов: {n} (также как изображение)")

    def duplicate(self) -> None:
        editor = self._active_editor()
        if editor is not None:  # Ctrl+D inside the editor duplicates the current line
            cursor = editor.textCursor()
            if cursor.hasSelection():
                text = cursor.selectedText().replace(" ", "\n")
                cursor.setPosition(cursor.selectionEnd())
                cursor.insertText(text)
            else:
                line = cursor.block().text()
                cursor.movePosition(cursor.MoveOperation.EndOfBlock)
                cursor.insertText("\n" + line)
            editor.setTextCursor(cursor)
            return
        if not self.scene.duplicate_selection():
            self.status_msg.setText("Выделите регионы для дублирования")

    def show_context_menu(self, screen_pos) -> None:
        menu = QMenu(self)
        has_selection = bool(self.scene.selected_regions())
        for act in (self.a_cut, self.a_copy, self.a_duplicate, self.a_delete):
            act.setEnabled(has_selection)
            menu.addAction(act)
        menu.addAction(self.a_paste)
        menu.addSeparator()
        menu.addActions([self.a_math, self.a_text, self.a_plot, self.a_image])
        menu.addSeparator()
        menu.addAction(self.a_separate)
        menu.exec(screen_pos)
        for act in (self.a_cut, self.a_copy, self.a_duplicate, self.a_delete):
            act.setEnabled(True)

    def paste(self) -> None:
        editor = self._active_editor()
        if editor is not None:
            editor.paste()
            return
        mime = QApplication.clipboard().mimeData()
        if not self.scene.paste() and mime.hasImage():
            image = QApplication.clipboard().image()
            if not image.isNull():
                self.controller.add_region(self._image_from_qimage(image, self.scene.cursor_pos()), "Paste image")
                self.scene.sync()

    def _on_files_dropped(self, files: list, pos: QPointF) -> None:
        for k, f in enumerate(files):
            suffix = Path(f).suffix.lower()
            if suffix in (".bcad", ".bmath"):
                if self._confirm_discard():
                    self.open_file(f)
                return
            self._add_image_file(f, QPointF(pos.x() + 20 * k, pos.y() + 20 * k))

    def delete_selected(self) -> None:
        ids = [i.region.id for i in self.scene.selectedItems() if hasattr(i, "region")]
        self.controller.remove_regions(ids)

    def select_all(self) -> None:
        item = self.scene.editing_item()
        if item is not None and getattr(item, "editor", None) is not None:
            item.editor.selectAll()
            return
        for i in self.scene.items_by_id.values():
            i.setSelected(True)

    def select_region(self, region_id: str) -> None:
        item = self.scene.items_by_id.get(region_id)
        if item is None:
            return
        self.scene.clearSelection()
        item.setSelected(True)
        self.view.ensureVisible(item, 60, 60)

    def edit_plot(self, region_id: str) -> None:
        region = self.controller.document.find(region_id)
        if not isinstance(region, PlotRegion):
            return
        dialog = PlotDialog(region, self)

        def apply(new_region: PlotRegion):
            values = {k: v for k, v in new_region.to_dict().items() if k not in ("id", "kind", "x", "y")}
            self.controller.set_fields(region_id, "Edit plot", **values)
        dialog.preview_requested.connect(apply)
        if dialog.exec():
            apply(dialog.result_region())

    def edit_image(self, region_id: str) -> None:
        region = self.controller.document.find(region_id)
        if not isinstance(region, ImageRegion):
            return
        from PySide6.QtWidgets import QInputDialog
        caption, ok = QInputDialog.getText(self, "Подпись изображения", "Подпись:", text=region.caption)
        if ok:
            self.controller.set_fields(region_id, "Caption", caption=caption)

    # --------------------------------------------------------- calculation
    def toggle_auto(self) -> None:
        self.settings.auto_calc = self.a_auto.isChecked()
        if self.settings.auto_calc:
            self.recalculate(True)

    def _on_changed(self) -> None:
        self.scene.sync()
        self._update_title()
        if self.settings.auto_calc:
            self.recalculate()

    def _on_restored(self) -> None:
        self.scene.rebuild()
        if self.results is not None:
            self.scene.apply_results(self.results.regions)
        if not self.controller.undo_stack.canUndo():  # freshly loaded document: show its top-left corner
            QTimer.singleShot(0, self._scroll_home)

    def _scroll_home(self) -> None:
        for bar in (self.view.horizontalScrollBar(), self.view.verticalScrollBar()):
            bar.setValue(bar.minimum())

    def recalculate(self, immediate: bool = False) -> None:
        self.calc.request(self.controller.document, self.theme, self.ctx.scale, immediate)

    def _on_results(self, result: DocumentResult) -> None:
        self.results = result
        self.scene.apply_results(result.regions)
        errors = result.errors()
        status = f"✓ Вычислено за {result.elapsed * 1000:.0f} мс" if not errors else \
            f"⚠ {len(errors)} ошибк{'а' if len(errors) == 1 else 'и'} · {result.elapsed * 1000:.0f} мс"
        self.status_calc.setText(status)
        self.variables.refresh(result.env, self.ctx.fmt)
        self.messages.show_errors(errors)
        words = set(self.engine.registry.names()) | set(result.env.vars) | {"given", "for", "while", "return"}
        self.scene.words = sorted(words)

    def _on_zoom(self, value: float) -> None:
        self.zoom_label.setText(f"{value * 100:.0f}%")
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(value * 100))
        self.zoom_slider.blockSignals(False)
        self.settings.zoom = value

    # ---------------------------------------------------------------- misc
    def show_function_reference(self) -> None:
        rows = []
        for category, specs in self.engine.registry.categories().items():
            rows.append(f"<h3>{category}</h3><table cellpadding='3'>")
            rows += [f"<tr><td><code>{s.signature}</code></td><td>{s.doc}</td></tr>" for s in specs]
            rows.append("</table>")
        rows.append("<h3>Символьные ключевые слова (expr -&gt; …)</h3><p>" + ", ".join(self.engine.registry.symbolic_names()) + "</p>")
        HelpDialog(self, "".join(rows), "Функции").exec()

    def about(self) -> None:
        QMessageBox.about(self, APP_NAME, f"<h2>{APP_NAME} {__version__}</h2><p>Математический редактор в стиле Mathcad: "
                                          "формулы, символьные и численные вычисления, блоки решения, графики, "
                                          "исследование операций.</p><p>PySide6 · Lark · SymPy · NumPy/SciPy · Matplotlib</p>")

    def closeEvent(self, event: QCloseEvent) -> None:
        self.scene.end_all_edits()
        if not self._confirm_discard():
            event.ignore()
            return
        self.calc.interrupt()
        self.settings.geometry = bytes(self.saveGeometry().toBase64()).decode()
        self.settings.window_state = bytes(self.saveState().toBase64()).decode()
        self.settings.save()
        event.accept()
