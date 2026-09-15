"""Background calculation service: debounced, cancellable, never blocks the UI."""
from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

from ..core.document import Document, PlotRegion
from ..core.theme import Theme
from ..engine.worksheet import DocumentResult, WorksheetEngine


class _Signals(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)


class _Job(QRunnable):
    def __init__(self, generation: int, snapshot: dict, engine: WorksheetEngine, cancel: threading.Event,
                 theme: Theme, scale: float, signals: _Signals):
        super().__init__()
        self.generation, self.snapshot, self.engine = generation, snapshot, engine
        self.cancel, self.theme, self.scale, self.signals = cancel, theme, scale, signals

    def run(self) -> None:
        try:
            doc = Document.from_dict(self.snapshot)
            result = self.engine.run(doc, self.cancel)
            if self.cancel.is_set():
                return
            # pre-render plots here so the GUI thread only blits images
            from ..render.regions import prerender_plot
            for region in doc.regions:
                if isinstance(region, PlotRegion) and self.cancel.is_set() is False:
                    res = result.regions.get(region.id)
                    if res is not None and res.plot is not None:
                        prerender_plot(region, res.plot, self.theme, self.scale)
            self.signals.finished.emit(self.generation, result)
        except Exception:
            self.signals.failed.emit(self.generation, traceback.format_exc())


class CalculationService(QObject):
    started = Signal()
    finished = Signal(object)      # DocumentResult
    failed = Signal(str)

    def __init__(self, engine: WorksheetEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.generation = 0
        self._cancel: threading.Event | None = None
        self._pending: tuple[dict, Theme, float] | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._launch)
        self._signals = _Signals()
        self._signals.finished.connect(self._on_finished)
        self._signals.failed.connect(self._on_failed)
        self.busy = False

    def request(self, document: Document, theme: Theme, scale: float, immediate: bool = False) -> None:
        self._pending = (document.to_dict(), theme, scale)
        if self._cancel is not None:
            self._cancel.set()
        if immediate:
            self._timer.stop()
            self._launch()
        else:
            self._timer.start()

    def _launch(self) -> None:
        if self._pending is None:
            return
        snapshot, theme, scale = self._pending
        self._pending = None
        self.generation += 1
        self._cancel = threading.Event()
        self.busy = True
        self.started.emit()
        self.pool.start(_Job(self.generation, snapshot, self.engine, self._cancel, theme, scale, self._signals))

    def interrupt(self) -> None:
        if self._cancel is not None:
            self._cancel.set()

    def _on_finished(self, generation: int, result: DocumentResult) -> None:
        if generation != self.generation:
            return
        self.busy = False
        self.finished.emit(result)

    def _on_failed(self, generation: int, message: str) -> None:
        if generation != self.generation:
            return
        self.busy = False
        self.failed.emit(message)
