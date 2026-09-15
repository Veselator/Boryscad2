"""GUI entry point."""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .. import APP_NAME


def run_gui(file: str | None = None) -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setStyle("Fusion")
    from .main_window import MainWindow
    from .settings import AppSettings
    window = MainWindow(AppSettings.load())
    window.show()
    if file:
        window.open_file(file)
    code = app.exec()
    # Settings are saved in closeEvent; skip interpreter teardown of Qt objects (avoids destructor-order issues).
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(code)
