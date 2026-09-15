"""Rendering: formula layout, plots, region views and exporters."""
from __future__ import annotations

import os
import sys


def ensure_qt_app(headless: bool = True):
    """Create a QGuiApplication if none exists (needed for fonts and painting)."""
    # On Windows the native platform plugin works without showing windows and provides the system fonts;
    # elsewhere use the offscreen plugin (with the system font directory) when there is no display.
    if headless and "QT_QPA_PLATFORM" not in os.environ and sys.platform != "win32" and not os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        for candidate in ("/usr/share/fonts", "/System/Library/Fonts"):
            if os.path.isdir(candidate):
                os.environ.setdefault("QT_QPA_FONTDIR", candidate)
                break
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication.instance()
    if app is None:
        app = QGuiApplication(sys.argv[:1])
    return app
