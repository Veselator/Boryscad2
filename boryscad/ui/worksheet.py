"""The worksheet canvas: free placement of regions like in Mathcad."""
from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import QMimeData, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QKeyEvent, QKeySequence, QPainter, QPainterPath, QPen, QTransform
from PySide6.QtWidgets import (QApplication, QGraphicsItem, QGraphicsObject, QGraphicsProxyWidget, QGraphicsScene,
                               QGraphicsSceneMouseEvent, QGraphicsView, QStyleOptionGraphicsItem)

from ..core.document import (Document, ImageRegion, MathRegion, PageBreakRegion, PlotRegion, Region, RegionFactory,
                             TextRegion)
from ..engine.worksheet import RegionResult
from ..render.qt_backend import qcolor
from ..render.regions import RegionLayout, RenderContext, layout_region
from .controller import DocumentController
from .editor import RegionEditor

MIME = "application/x-boryscad-regions"
HANDLE = 12.0


class RegionItem(QGraphicsObject):
    def __init__(self, region: Region, scene: "WorksheetScene"):
        super().__init__()
        self.region = region
        self.ws = scene
        self.result: Optional[RegionResult] = None
        self.layout: Optional[RegionLayout] = None
        self._hover = False
        self._press_pos: Optional[QPointF] = None
        self._resizing = False
        self._resize_origin = (0.0, 0.0, QPointF())
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        self.setAcceptHoverEvents(True)
        self.setPos(region.x, region.y)
        self.relayout()

    # ------------------------------------------------------------ layout
    def relayout(self) -> None:
        self.prepareGeometryChange()
        try:
            self.layout = layout_region(self.region, self.result, self.ws.ctx)
        except Exception as exc:  # never crash painting
            from ..render.regions import BoxLayout
            self.layout = BoxLayout(self.ws.ctx.builder.text(f"render error: {exc}", 0, "math_error"))
        self.update()

    def content_size(self) -> tuple[float, float]:
        return (self.layout.width, self.layout.height) if self.layout else (40.0, 24.0)

    def boundingRect(self) -> QRectF:
        w, h = self.content_size()
        return QRectF(-4, -4, max(w, 24) + 8, max(h, 18) + 8)

    def resizable(self) -> bool:
        return False

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        theme = self.ws.ctx.theme
        rect = self.boundingRect().adjusted(1, 1, -1, -1)
        if self.isSelected():
            painter.setPen(QPen(qcolor(theme.color("accent")), 1.2))
            painter.setBrush(qcolor(theme.color("selection")))
            painter.drawRoundedRect(rect, 6, 6)
        elif self._hover:
            painter.setPen(QPen(qcolor(theme.color("border")), 1.0, Qt.PenStyle.DashLine))
            painter.setBrush(qcolor(theme.color("hover")))
            painter.drawRoundedRect(rect, 6, 6)
        if self.result is not None and self.result.has_error and not self.isSelected():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(qcolor(theme.color("math_error")))
            painter.drawRoundedRect(QRectF(-4, -4, 3, rect.height()), 1.5, 1.5)
        self.paint_content(painter)
        if self.resizable() and (self.isSelected() or self._hover):
            w, h = self.content_size()
            painter.setPen(QPen(qcolor(theme.color("accent")), 2))
            for k in (4, 8):
                painter.drawLine(QPointF(w - k, h + 2), QPointF(w + 2, h - k))

    def paint_content(self, painter: QPainter) -> None:
        if self.layout:
            self.layout.paint(painter, 0, 0)

    # ------------------------------------------------------------ events
    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.unsetCursor()
        self.update()

    def hoverMoveEvent(self, event):
        if self.resizable() and self._in_handle(event.pos()):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor if self.isSelected() else Qt.CursorShape.ArrowCursor)

    def _in_handle(self, pos: QPointF) -> bool:
        w, h = self.content_size()
        return w - HANDLE <= pos.x() <= w + 6 and h - HANDLE <= pos.y() <= h + 6

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        self.ws.hide_cursor()
        if event.button() == Qt.MouseButton.LeftButton and self.resizable() and self._in_handle(event.pos()):
            self._resizing = True
            self._resize_origin = (self.region.width, self.region.height, event.scenePos())
            event.accept()
            return
        self._press_pos = self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if self._resizing:
            w0, h0, p0 = self._resize_origin
            delta = event.scenePos() - p0
            self.apply_resize(max(80.0, w0 + delta.x()), max(60.0, h0 + delta.y()), preview=True)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if self._resizing:
            self._resizing = False
            w0, h0, _ = self._resize_origin
            new_w, new_h = self.region.width, self.region.height
            self.region.width, self.region.height = w0, h0
            self.ws.controller.set_fields(self.region.id, "Resize", width=round(new_w), height=round(new_h))
            return
        super().mouseReleaseEvent(event)
        self.ws.commit_moves()

    def apply_resize(self, w: float, h: float, preview: bool) -> None:
        self.region.width, self.region.height = w, h
        self.relayout()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            p: QPointF = value
            step = self.ws.grid_step if self.ws.snap else 1
            x = max(0.0, round(p.x() / step) * step)
            y = max(0.0, round(p.y() / step) * step)
            return QPointF(x, y)
        return super().itemChange(change, value)

    def begin_edit(self, initial: Optional[str] = None) -> None:
        pass

    def is_editing(self) -> bool:
        return False

    def end_edit(self, commit: bool = True) -> None:
        pass


class EditableItem(RegionItem):
    """Region edited as source text with a live preview below the editor."""

    math_editor = True

    def __init__(self, region, scene):
        self.proxy: Optional[QGraphicsProxyWidget] = None
        self.editor: Optional[RegionEditor] = None
        self.preview: Optional[RegionLayout] = None
        self._ending = False
        super().__init__(region, scene)

    def source(self) -> str:
        raise NotImplementedError

    def source_field(self) -> str:
        raise NotImplementedError

    def preview_region(self, text: str) -> Region:
        raise NotImplementedError

    def is_editing(self) -> bool:
        return self.proxy is not None

    def begin_edit(self, initial: Optional[str] = None) -> None:
        if self.proxy is not None:
            self.editor.setFocus()
            return
        self.ws.end_all_edits(except_item=self)
        editor = RegionEditor(self.ws.ctx.theme, self.ws.completion_words(), self.ws.function_names(), self.math_editor)
        editor.setPlainText(self.source() if initial is None else initial)
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        self.editor = editor
        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(editor)
        self.proxy.setZValue(10)
        editor.textChanged.connect(self._on_text_changed)
        editor.committed.connect(lambda: QTimer.singleShot(0, lambda: self.end_edit(True)))
        self.setZValue(5)
        self._on_text_changed()
        editor.setFocus(Qt.FocusReason.MouseFocusReason)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

    def _editor_size(self) -> tuple[float, float]:
        fm = self.editor.fontMetrics()
        lines = self.editor.toPlainText().split("\n")
        if not self.math_editor:
            width = self.region.width if self.region.width > 0 else max(
                300.0, self.ws.ctx.page_width - self.region.x - self.ws.ctx.margin)
            visual_lines = max(len(lines), int(self.editor.document().size().height()))
            return width, fm.lineSpacing() * visual_lines + 14
        width = max([180.0] + [fm.horizontalAdvance(l) + 40.0 for l in lines])
        return width, fm.lineSpacing() * len(lines) + 14

    def _on_text_changed(self) -> None:
        if self.editor is None:
            return
        w, h = self._editor_size()
        self.editor.setFixedSize(int(w), int(h))
        if self.proxy is not None:
            self.proxy.setMinimumSize(w, h)
            self.proxy.setMaximumSize(w, h)
            self.proxy.setGeometry(QRectF(0, 0, w, h))
        self.prepareGeometryChange()
        try:
            self.preview = layout_region(self.preview_region(self.editor.toPlainText()), None, self.ws.ctx)
        except Exception:
            self.preview = None
        self.update()

    def content_size(self) -> tuple[float, float]:
        if self.proxy is not None and self.editor is not None:
            w, h = self._editor_size()
            pw, ph = (self.preview.width, self.preview.height) if self.preview else (0.0, 0.0)
            return max(w, pw), h + 6 + ph
        return super().content_size()

    def paint_content(self, painter: QPainter) -> None:
        if self.proxy is not None:
            if self.preview is not None:
                _, h = self._editor_size()
                painter.setOpacity(0.92)
                self.preview.paint(painter, 0, h + 6)
                painter.setOpacity(1.0)
            return
        super().paint_content(painter)

    def end_edit(self, commit: bool = True) -> None:
        if self.proxy is None or self._ending:
            return
        self._ending = True
        text = self.editor.toPlainText()
        proxy, self.proxy = self.proxy, None
        editor, self.editor = self.editor, None
        editor.blockSignals(True)
        editor.hide()
        proxy.hide()
        proxy.setParentItem(None)
        if proxy.scene() is not None:
            proxy.scene().removeItem(proxy)
        proxy.deleteLater()
        self._dead_proxies = getattr(self, "_dead_proxies", [])[-4:] + [proxy]
        self.preview = None
        self.setZValue(0)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self._ending = False
        if not commit:
            self.relayout()
            return
        if not text.strip():
            self.ws.controller.remove_regions([self.region.id])
            return
        if text != self.source():
            self.ws.controller.set_fields(self.region.id, "Edit", **{self.source_field(): text})
        self.relayout()

    def mouseDoubleClickEvent(self, event):
        self.begin_edit()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F2) and not self.is_editing():
            self.begin_edit()
            return
        super().keyPressEvent(event)


class MathItem(EditableItem):
    def source(self) -> str:
        return self.region.source

    def source_field(self) -> str:
        return "source"

    def preview_region(self, text: str) -> Region:
        return MathRegion(source=text)


class TextItem(EditableItem):
    math_editor = False

    def source(self) -> str:
        return self.region.text

    def source_field(self) -> str:
        return "text"

    def preview_region(self, text: str) -> Region:
        return TextRegion(text=text, style=self.region.style, format=self.region.format, width=self.region.width,
                          x=self.region.x)


class PlotItem(RegionItem):
    def resizable(self) -> bool:
        return True

    def content_size(self):
        return self.region.width, self.region.height

    def boundingRect(self) -> QRectF:
        return QRectF(-4, -4, self.region.width + 8, (self.layout.height if self.layout else self.region.height) + 8)

    def apply_resize(self, w, h, preview):
        self.prepareGeometryChange()
        self.region.width, self.region.height = w, h
        self.update()

    def paint_content(self, painter):
        if self._resizing and self.layout is not None:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(QRectF(0, 0, self.region.width, self.region.height), self.layout.image)
            painter.restore()
            return
        super().paint_content(painter)

    def mouseDoubleClickEvent(self, event):
        self.ws.edit_plot_requested.emit(self.region.id)


class ImageItem(RegionItem):
    def resizable(self) -> bool:
        return True

    def apply_resize(self, w, h, preview):
        ratio = self._resize_origin[1] / max(1.0, self._resize_origin[0])
        self.region.width, self.region.height = w, w * ratio
        self.relayout()

    def mouseDoubleClickEvent(self, event):
        self.ws.edit_image_requested.emit(self.region.id)


class PageBreakItem(RegionItem):
    pass


ITEM_TYPES = {"math": MathItem, "text": TextItem, "plot": PlotItem, "image": ImageItem, "pagebreak": PageBreakItem}


class InsertionCursor(QGraphicsItem):
    def __init__(self, color: str):
        super().__init__()
        self.color = color
        self.setZValue(100)

    def boundingRect(self) -> QRectF:
        return QRectF(-8, -8, 16, 16)

    def paint(self, painter, option, widget=None):
        pen = QPen(qcolor(self.color), 1.6)
        painter.setPen(pen)
        painter.drawLine(QPointF(-7, 0), QPointF(7, 0))
        painter.drawLine(QPointF(0, -7), QPointF(0, 7))


class WorksheetScene(QGraphicsScene):
    edit_plot_requested = Signal(str)
    edit_image_requested = Signal(str)
    status = Signal(str)
    ensure_visible = Signal(QRectF)
    context_menu_requested = Signal(object)

    def __init__(self, controller: DocumentController, ctx: RenderContext, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.ctx = ctx
        self.items_by_id: dict[str, RegionItem] = {}
        self.results: dict[str, RegionResult] = {}
        self.snap = True
        self.grid_step = 8
        self.show_grid = True
        self.words: list[str] = []
        self._functions: set[str] = set(ctx.registry.names()) if ctx.registry else set()
        self.cursor = InsertionCursor(ctx.theme.color("cursor"))
        self.addItem(self.cursor)
        self.cursor.hide()
        self._move_start: dict[str, tuple[float, float]] = {}
        self.selectionChanged.connect(self._remember_positions)
        self.rebuild()

    # ---------------------------------------------------------- model sync
    def rebuild(self) -> None:
        for item in list(self.items_by_id.values()):
            if item.is_editing():
                item.end_edit(False)
            self.discard_item(item)
        self.items_by_id.clear()
        for region in self.controller.document.regions:
            self._add_item(region)
        self.update_scene_rect()

    def discard_item(self, item: QGraphicsItem) -> None:
        item.setSelected(False)
        item.clearFocus()
        item.hide()
        if item.scene() is self:
            self.removeItem(item)
        self._graveyard = (getattr(self, '_graveyard', []) + [item])[-50:]

    def _add_item(self, region: Region) -> RegionItem:
        item_cls = ITEM_TYPES.get(region.kind, RegionItem)
        item = item_cls(region, self)
        item.result = self.results.get(region.id)
        item.relayout()
        self.addItem(item)
        self.items_by_id[region.id] = item
        return item

    def sync(self) -> None:
        """Bring items in line with the document after in-place modifications."""
        doc = self.controller.document
        ids = {r.id for r in doc.regions}
        for rid in [rid for rid in self.items_by_id if rid not in ids]:
            item = self.items_by_id.pop(rid)
            if item.is_editing():
                item.end_edit(False)
            self.discard_item(item)
        for region in doc.regions:
            item = self.items_by_id.get(region.id)
            if item is None:
                self._add_item(region)
            else:
                if item.region is not region:
                    item.region = region
                if not item.is_editing():
                    item.setPos(region.x, region.y)
                    item.relayout()
        self.update_scene_rect()

    def apply_results(self, results: dict[str, RegionResult]) -> None:
        self.results = results
        for rid, item in self.items_by_id.items():
            item.result = results.get(rid)
            if not item.is_editing():
                item.relayout()
        self.update_scene_rect()

    def relayout_all(self) -> None:
        for item in self.items_by_id.values():
            item.relayout()
        self.cursor.color = self.ctx.theme.color("cursor")
        self.update()

    def update_scene_rect(self) -> None:
        bottom = max([900.0] + [i.pos().y() + i.boundingRect().height() + 400 for i in self.items_by_id.values()])
        right = max([self.ctx.page_width] + [i.pos().x() + i.boundingRect().width() + 60 for i in self.items_by_id.values()])
        self.setSceneRect(QRectF(-40, -40, right + 80, bottom + 40))
        self.update()

    # ----------------------------------------------------------- editing
    def completion_words(self) -> list[str]:
        return self.words

    def function_names(self) -> set[str]:
        return self._functions

    def end_all_edits(self, except_item: Optional[RegionItem] = None) -> None:
        for item in list(self.items_by_id.values()):
            if item is not except_item and item.is_editing():
                item.end_edit(True)

    def editing_item(self) -> Optional[RegionItem]:
        return next((i for i in self.items_by_id.values() if i.is_editing()), None)

    def insert_at_cursor(self, snippet: str) -> None:
        item = self.editing_item()
        if item is not None and getattr(item, "editor", None) is not None:
            item.editor.insertPlainText(snippet)
            item.editor.setFocus()
            return
        self.create_region("math", self.cursor_pos(), snippet)

    def cursor_pos(self) -> QPointF:
        if self.cursor.isVisible():
            return self.cursor.pos()
        items = list(self.items_by_id.values())
        y = max([self.ctx.margin] + [i.pos().y() + i.boundingRect().height() + 16 for i in items])
        return QPointF(self.ctx.margin, self._snap(y))

    def _snap(self, v: float) -> float:
        return round(v / self.grid_step) * self.grid_step if self.snap else v

    def create_region(self, kind: str, pos: QPointF, initial: str = "", edit: bool = True, **fields) -> Region:
        data = {"kind": kind, "x": self._snap(pos.x()), "y": self._snap(pos.y())}
        data.update(fields)
        region = RegionFactory.create(data)
        if kind in ("math", "text") and edit:
            # do not create an undo entry for an empty region: it is added when the edit is committed
            self.controller.document.add(region)
            item = self._add_item(region)
            item.setSelected(True)
            self.hide_cursor()
            item.begin_edit(initial)
            original_end = item.end_edit

            def first_end(commit=True, _item=item, _region=region):
                text = _item.editor.toPlainText() if _item.editor is not None else ""
                _item.end_edit = original_end
                self.controller.document.remove(_region.id)
                original_end(False)
                self.discard_item(_item)
                self.items_by_id.pop(_region.id, None)
                if commit and text.strip():
                    setattr(_region, "source" if kind == "math" else "text", text)
                    self.controller.add_region(_region, "Insert " + kind)
                    self.sync()
            item.end_edit = first_end  # type: ignore[assignment]
            return region
        self.controller.add_region(region, "Insert " + kind)
        self.sync()
        return region

    def hide_cursor(self) -> None:
        self.cursor.hide()

    # ------------------------------------------------------------ moves
    def _remember_positions(self) -> None:
        self._move_start = {i.region.id: (i.region.x, i.region.y) for i in self.selectedItems() if isinstance(i, RegionItem)}

    def commit_moves(self) -> None:
        moved = {}
        before = {}
        for item in self.selectedItems():
            if isinstance(item, RegionItem):
                p = item.pos()
                if (item.region.x, item.region.y) != (p.x(), p.y()):
                    moved[item.region.id] = (p.x(), p.y())
                    before[item.region.id] = (item.region.x, item.region.y)
        if moved:
            self.controller.move_regions(moved, before)
            self.update_scene_rect()

    def separate_regions(self) -> None:
        items = sorted(self.items_by_id.values(), key=lambda i: (i.pos().y(), i.pos().x()))
        placed: list[QRectF] = []
        positions = {}
        for item in items:
            rect = QRectF(item.pos().x(), item.pos().y(), *item.content_size())
            changed = True
            while changed:
                changed = False
                for other in placed:
                    if rect.adjusted(-2, -2, 2, 2).intersects(other):
                        rect.moveTop(self._snap(other.bottom() + 12))
                        changed = True
            placed.append(rect)
            if abs(rect.top() - item.pos().y()) > 0.5:
                positions[item.region.id] = (rect.left(), rect.top())

        def mutate(doc: Document):
            for region in doc.regions:
                if region.id in positions:
                    region.x, region.y = positions[region.id]
        if positions:
            self.controller.apply("Separate regions", mutate)
            self.sync()

    def align(self, mode: str) -> None:
        items = [i for i in self.selectedItems() if isinstance(i, RegionItem)]
        if len(items) < 2:
            return
        if mode == "left":
            x = min(i.pos().x() for i in items)
            values = {i.region.id: (x, i.pos().y()) for i in items}
        else:
            y = min(i.pos().y() for i in items)
            values = {i.region.id: (i.pos().x(), y) for i in items}

        def mutate(doc):
            for region in doc.regions:
                if region.id in values:
                    region.x, region.y = values[region.id]
        self.controller.apply("Align", mutate)
        self.sync()

    # ------------------------------------------------------------ events
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        t = self.ctx.theme
        painter.fillRect(rect, qcolor(t.color("desk")))
        page = QRectF(-24, -24, self.ctx.page_width + 48, max(self.sceneRect().bottom(), rect.bottom()) + 24)
        shadow = page.translated(0, 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor(t.color("page_shadow")))
        painter.drawRoundedRect(shadow, 12, 12)
        painter.setBrush(qcolor(t.color("page")))
        painter.drawRoundedRect(page, 12, 12)
        scale = painter.worldTransform().m11()
        if self.show_grid and scale >= 0.6:
            step = self.grid_step * 3
            left = max(page.left() + 12, rect.left() - rect.left() % step)
            top = max(page.top() + 12, rect.top() - rect.top() % step)
            painter.setPen(QPen(qcolor(t.color("page_grid")), 1.6 / scale))
            x = left - left % step
            points = []
            while x < min(rect.right(), page.right() - 8):
                y = top - top % step
                while y < min(rect.bottom(), page.bottom() - 8):
                    points.append(QPointF(x, y))
                    y += step
                x += step
            if points:
                painter.drawPoints(points)
        # right margin guide
        painter.setPen(QPen(qcolor(t.color("page_grid")), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(self.ctx.page_width - self.ctx.margin, page.top() + 8),
                         QPointF(self.ctx.page_width - self.ctx.margin, page.bottom() - 8))

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        item = self.itemAt(event.scenePos(), QTransform())
        if item is None or item is self.cursor:
            self.end_all_edits()
            if event.button() == Qt.MouseButton.LeftButton:
                p = event.scenePos()
                self.cursor.setPos(self._snap(max(0, p.x())), self._snap(max(0, p.y())))
                self.cursor.show()
                self.status.emit(f"cursor at {self.cursor.pos().x():.0f}, {self.cursor.pos().y():.0f} — start typing "
                                 "to create a math region, type \" for text")
        super().mousePressEvent(event)

    def contextMenuEvent(self, event) -> None:
        item = self.itemAt(event.scenePos(), QTransform())
        while item is not None and not isinstance(item, RegionItem):
            item = item.parentItem()
        if isinstance(item, RegionItem) and item.is_editing():
            super().contextMenuEvent(event)
            return
        if isinstance(item, RegionItem) and not item.isSelected():
            self.clearSelection()
            item.setSelected(True)
        if item is None:
            p = event.scenePos()
            self.cursor.setPos(self._snap(max(0, p.x())), self._snap(max(0, p.y())))
            self.cursor.show()
        self.context_menu_requested.emit(event.screenPos())
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        editing = self.editing_item()
        if editing is not None:
            super().keyPressEvent(event)
            return
        selected = [i for i in self.selectedItems() if isinstance(i, RegionItem)]
        key = event.key()
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and selected:
            self.controller.remove_regions([i.region.id for i in selected])
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down) and selected:
            dx = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1}.get(key, 0) * self.grid_step
            dy = {Qt.Key.Key_Up: -1, Qt.Key.Key_Down: 1}.get(key, 0) * self.grid_step
            self._remember_positions()
            for i in selected:
                i.setPos(i.pos() + QPointF(dx, dy))
            self.commit_moves()
            return
        if len(selected) == 1 and key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F2):
            item = selected[0]
            if isinstance(item, EditableItem):
                item.begin_edit()
            elif isinstance(item, PlotItem):
                self.edit_plot_requested.emit(item.region.id)
            return
        text = event.text()
        if text and text.isprintable() and not (event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)):
            pos = self.cursor_pos()
            if text == '"':
                self.create_region("text", pos, "")
            else:
                self.create_region("math", pos, text)
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------- clipboard
    def selected_regions(self) -> list[Region]:
        items = [i for i in self.selectedItems() if isinstance(i, RegionItem)]
        items.sort(key=lambda i: (i.pos().y(), i.pos().x()))
        return [i.region for i in items]

    def _selection_rect(self, regions: list[Region]) -> QRectF:
        rect = QRectF()
        for r in regions:
            item = self.items_by_id.get(r.id)
            if item is not None:
                w, h = item.content_size()
                rect = rect.united(QRectF(item.pos().x(), item.pos().y(), w, h))
        return rect

    def selection_image(self, regions: list[Region], scale: float = 2.0) -> QImage:
        """Picture of the selected regions (pasted into Word, Telegram, …)."""
        rect = self._selection_rect(regions).adjusted(-10, -10, 10, 10)
        image = QImage(max(1, int(rect.width() * scale)), max(1, int(rect.height() * scale)),
                       QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(qcolor(self.ctx.theme.color("page")))
        painter = QPainter(image)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
                               | QPainter.RenderHint.SmoothPixmapTransform)
        painter.scale(scale, scale)
        for r in regions:
            item = self.items_by_id.get(r.id)
            if item is not None and item.layout is not None:
                item.layout.paint(painter, item.pos().x() - rect.left(), item.pos().y() - rect.top())
        painter.end()
        return image

    def copy_selection(self, cut: bool = False) -> bool:
        regions = self.selected_regions()
        if not regions:
            return False
        mime = QMimeData()
        mime.setData(MIME, json.dumps([r.to_dict() for r in regions]).encode("utf-8"))
        mime.setText("\n\n".join(getattr(r, "source", None) or getattr(r, "text", "") for r in regions))
        try:
            mime.setImageData(self.selection_image(regions))
        except Exception:
            pass
        QApplication.clipboard().setMimeData(mime)
        self._paste_count = 0
        if cut:
            self.controller.remove_regions([r.id for r in regions])
        return True

    def _insert_copies(self, data: list[dict], origin: QPointF, text: str) -> list[str]:
        min_x = min(d["x"] for d in data)
        min_y = min(d["y"] for d in data)
        new = []
        for d in data:
            d = dict(d)
            d.pop("id", None)
            d["x"] = self._snap(max(0.0, origin.x() + d["x"] - min_x))
            d["y"] = self._snap(max(0.0, origin.y() + d["y"] - min_y))
            new.append(RegionFactory.create(d))
        self.controller.apply(text, lambda doc: doc.regions.extend(new))
        self.sync()
        self.clearSelection()
        for r in new:
            item = self.items_by_id.get(r.id)
            if item is not None:
                item.setSelected(True)
        if new:
            self.ensure_visible.emit(self._selection_rect(new))
        return [r.id for r in new]

    def paste(self) -> bool:
        mime = QApplication.clipboard().mimeData()
        if mime.hasFormat(MIME):
            data = json.loads(bytes(mime.data(MIME)).decode("utf-8"))
            if not data:
                return False
            stack = self.controller.undo_stack
            if self.cursor.isVisible():
                # at the red insertion cursor, like Mathcad
                origin = self.cursor.pos()
                self.hide_cursor()
                self._insert_copies(data, origin, "Paste")
                return True
            # no cursor: below the previous paste (or below the copied regions), making room if needed
            last = getattr(self, "_last_paste_rect", None)
            ids = [d.get("id") for d in data]
            source_rect = self._selection_rect([r for r in self.controller.document.regions if r.id in ids])
            anchor = last if (last is not None and getattr(self, "_paste_count", 0) > 0) else source_rect
            if anchor is None or anchor.isNull():
                origin = self.cursor_pos()
            else:
                origin = QPointF(min(d["x"] for d in data), anchor.bottom() + 16)
            stack.beginMacro("Paste")
            try:
                new_ids = self._insert_copies(data, origin, "Paste")
                self.separate_regions()
            finally:
                stack.endMacro()
            self._paste_count = getattr(self, "_paste_count", 0) + 1
            new_regions = [r for r in self.controller.document.regions if r.id in set(new_ids)]
            self._last_paste_rect = self._selection_rect(new_regions)
            for rid in new_ids:
                item = self.items_by_id.get(rid)
                if item is not None:
                    item.setSelected(True)
            return True
        if mime.hasImage():
            return False  # handled by the main window (needs image encoding)
        if mime.hasText() and mime.text().strip():
            pos = self.cursor_pos()
            self.hide_cursor()
            self.controller.add_region(MathRegion(x=self._snap(pos.x()), y=self._snap(pos.y()), source=mime.text().strip()),
                                       "Paste")
            self.sync()
            return True
        return False

    def duplicate_selection(self) -> bool:
        regions = self.selected_regions()
        if not regions:
            return False
        rect = self._selection_rect(regions)
        data = [r.to_dict() for r in regions]
        # place the copy directly below the selection; one undo step for insert + making room
        stack = self.controller.undo_stack
        stack.beginMacro("Duplicate")
        try:
            new_ids = self._insert_copies(data, QPointF(rect.left(), rect.bottom() + 16), "Duplicate")
            self.separate_regions()
        finally:
            stack.endMacro()
        for rid in new_ids:
            item = self.items_by_id.get(rid)
            if item is not None:
                item.setSelected(True)
        return True


class WorksheetView(QGraphicsView):
    zoom_changed = Signal(float)
    files_dropped = Signal(list, QPointF)

    def __init__(self, scene: WorksheetScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
                            | QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setAcceptDrops(True)
        self._zoom = 1.0

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_zoom(self, value: float) -> None:
        value = max(0.3, min(4.0, value))
        self.setTransform(QTransform.fromScale(value, value))
        self._zoom = value
        self.zoom_changed.emit(value)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
            self.set_zoom(self._zoom * factor)
            return
        super().wheelEvent(event)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            files = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            self.files_dropped.emit(files, self.mapToScene(event.position().toPoint()))
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
