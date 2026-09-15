"""Document controller: every modification goes through here (undo/redo + change notification)."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from ..core.document import Document, Region


class SnapshotCommand(QUndoCommand):
    def __init__(self, controller: "DocumentController", text: str, before: dict, after: dict):
        super().__init__(text)
        self.controller, self.before, self.after = controller, before, after
        self._first = True

    def redo(self) -> None:
        if self._first:
            self._first = False
            return
        self.controller.restore(self.after)

    def undo(self) -> None:
        self.controller.restore(self.before)


class DocumentController(QObject):
    changed = Signal()            # content changed → recalculate
    restored = Signal()           # whole document replaced (undo/redo/open) → rebuild the view
    modified_changed = Signal(bool)

    def __init__(self, document: Document | None = None, parent=None):
        super().__init__(parent)
        self.document = document or Document()
        self.undo_stack = QUndoStack(self)
        self.undo_stack.cleanChanged.connect(lambda clean: self.modified_changed.emit(not clean))
        self.path: str | None = None

    # ------------------------------------------------------------ generic
    def apply(self, text: str, mutate: Callable[[Document], None]) -> None:
        before = self.document.to_dict()
        mutate(self.document)
        after = self.document.to_dict()
        if before == after:
            return
        self.undo_stack.push(SnapshotCommand(self, text, before, after))
        self.changed.emit()

    def restore(self, data: dict) -> None:
        self.document = Document.from_dict(data)
        self.restored.emit()
        self.changed.emit()

    def load(self, document: Document, path: str | None = None) -> None:
        self.document = document
        self.path = path
        self.undo_stack.clear()
        self.undo_stack.setClean()
        self.restored.emit()
        self.changed.emit()

    # ----------------------------------------------------------- helpers
    def add_region(self, region: Region, text: str = "Insert region") -> None:
        self.apply(text, lambda d: d.add(region))

    def remove_regions(self, ids: list[str]) -> None:
        if ids:
            self.apply("Delete", lambda d: setattr(d, "regions", [r for r in d.regions if r.id not in set(ids)]))

    def move_regions(self, positions: dict[str, tuple[float, float]], before: dict[str, tuple[float, float]]) -> None:
        """Items were already moved interactively; record the move for undo."""
        snapshot_before = self.document.to_dict()
        for r in snapshot_before["regions"]:
            if r["id"] in before:
                r["x"], r["y"] = before[r["id"]]
        for region in self.document.regions:
            if region.id in positions:
                region.x, region.y = positions[region.id]
        after = self.document.to_dict()
        if snapshot_before != after:
            self.undo_stack.push(SnapshotCommand(self, "Move", snapshot_before, after))
            self.changed.emit()

    def set_fields(self, region_id: str, text: str = "Edit", **values) -> None:
        def mutate(doc: Document):
            region = doc.find(region_id)
            if region is not None:
                for k, v in values.items():
                    setattr(region, k, v)
        self.apply(text, mutate)
