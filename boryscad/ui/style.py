"""Qt style sheet generated from a theme, and a small SVG icon set."""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from ..core.theme import Theme
from ..render.qt_backend import css_rgba, qcolor


def build_qss(t: Theme, ui_font: str, ui_size: int) -> str:
    c = {k: css_rgba(v) for k, v in t.colors.items()}
    r = t.radius
    return f"""
* {{ font-family: '{ui_font}'; font-size: {ui_size}pt; color: {c['text']}; }}
QMainWindow, QDialog {{ background: {c['window']}; }}
QWidget#Central {{ background: {c['desk']}; }}
QToolTip {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 6px; border-radius: 6px; }}
QMenuBar {{ background: {c['window']}; padding: 2px 6px; }}
QMenuBar::item {{ padding: 6px 10px; border-radius: 6px; background: transparent; }}
QMenuBar::item:selected {{ background: {c['hover']}; }}
QMenu {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {r}px; padding: 6px; }}
QMenu::item {{ padding: 6px 28px 6px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c['selection']}; }}
QMenu::separator {{ height: 1px; background: {c['border']}; margin: 5px 8px; }}
QToolBar {{ background: {c['window']}; border: none; border-bottom: 1px solid {c['border']}; padding: 4px 8px; spacing: 2px; }}
QToolBar::separator {{ width: 1px; background: {c['border']}; margin: 6px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 5px 7px; }}
QToolButton:hover {{ background: {c['hover']}; border-color: {c['border']}; }}
QToolButton:pressed, QToolButton:checked {{ background: {c['selection']}; border-color: {c['accent']}; }}
QToolButton#Accent {{ background: {c['accent']}; color: {c['accent_text']}; font-weight: 600; padding: 5px 12px; }}
QToolButton#Accent:hover {{ border-color: {c['accent_text']}; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {c['sidebar']}; padding: 8px 10px; text-align: left; font-weight: 600; border-bottom: 1px solid {c['border']}; }}
QDockWidget > QWidget {{ background: {c['sidebar']}; }}
QTabWidget::pane {{ border: none; background: {c['sidebar']}; }}
QTabBar::tab {{ background: transparent; padding: 7px 12px; border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ border-bottom: 2px solid {c['accent']}; color: {c['text']}; }}
QTabBar::tab:!selected {{ color: {c['text_muted']}; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QFontComboBox {{
    background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 7px; padding: 5px 8px;
    selection-background-color: {c['selection']}; selection-color: {c['text']}; }}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {c['surface']}; border: 1px solid {c['border']}; selection-background-color: {c['selection']}; }}
QPushButton {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 7px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {c['accent']}; }}
QPushButton:default, QPushButton#Primary {{ background: {c['accent']}; color: {c['accent_text']}; border-color: {c['accent']}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {c['border']}; background: {c['surface']}; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']}; }}
QTreeWidget, QListWidget, QTableWidget {{ background: {c['sidebar']}; border: none; alternate-background-color: {c['surface_alt']}; }}
QTreeWidget::item, QListWidget::item {{ padding: 4px 2px; border-radius: 5px; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: {c['selection']}; color: {c['text']}; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: {c['hover']}; }}
QHeaderView::section {{ background: {c['sidebar']}; border: none; border-bottom: 1px solid {c['border']}; padding: 5px; color: {c['text_muted']}; }}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {c['text_muted']}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {c['border']}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QStatusBar {{ background: {c['window']}; border-top: 1px solid {c['border']}; color: {c['text_muted']}; }}
QStatusBar QLabel {{ color: {c['text_muted']}; padding: 0 8px; }}
QSlider::groove:horizontal {{ height: 4px; background: {c['border']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; border-radius: 7px; background: {c['accent']}; }}
QToolBox::tab {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 7px; padding: 6px 10px; font-weight: 600; }}
QToolBox::tab:selected {{ border-color: {c['accent']}; }}
QGroupBox {{ border: 1px solid {c['border']}; border-radius: {r}px; margin-top: 14px; padding: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {c['text_muted']}; }}
QLabel#Muted {{ color: {c['text_muted']}; }}
QLabel#Title {{ font-size: {ui_size + 6}pt; font-weight: 700; }}
QPushButton#Palette {{ padding: 4px; min-width: 34px; min-height: 28px; font-family: 'Cambria Math'; font-size: {ui_size + 2}pt; }}
QSplitter::handle {{ background: {c['border']}; }}
QWidget#PaletteContent, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {c['sidebar']}; }}
QPushButton#PaletteHeader {{ background: transparent; border: none; text-align: left; padding: 6px 2px; font-weight: 600; color: {c['text_muted']}; }}
QPushButton#PaletteHeader:hover {{ color: {c['text']}; }}
QPushButton#Palette:hover {{ background: {c['selection']}; }}
"""


# 24×24 stroke icons (lucide-like, hand made)
ICONS = {
    "new": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/><path d="M12 12v6M9 15h6"/>',
    "open": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "save": '<path d="M5 3h11l5 5v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M7 3v6h8V3"/><rect x="7" y="13" width="10" height="8"/>',
    "export": '<path d="M12 3v12"/><path d="M7 8l5-5 5 5"/><path d="M5 15v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"/>',
    "undo": '<path d="M9 14L4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/>',
    "redo": '<path d="M15 14l5-5-5-5"/><path d="M20 9H9a5 5 0 0 0 0 10h3"/>',
    "math": '<path d="M4 7h7M7.5 3.5v7"/><path d="M14 5l6 6M20 5l-6 6"/><path d="M4 17h7"/><path d="M14 15h6M14 19h6"/>',
    "text": '<path d="M4 6V4h16v2"/><path d="M12 4v16"/><path d="M9 20h6"/>',
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 17l-5-5-9 8"/>',
    "plot": '<path d="M3 3v18h18"/><path d="M6 15l4-5 4 3 6-7"/>',
    "given": '<path d="M9 4C6 4 6 7 6 9s-2 3-2 3 2 1 2 3 0 5 3 5"/><path d="M11 9h8M11 15h8"/>',
    "calc": '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    "theme": '<circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 0 0 18z" fill="currentColor"/>',
    "zoomin": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M11 8v6M8 11h6"/>',
    "zoomout": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M8 11h6"/>',
    "delete": '<path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M6 6l1 14h10l1-14"/>',
    "pagebreak": '<path d="M4 4v5h16V4"/><path d="M4 20v-5h16v5"/><path d="M2 12h3M8 12h3M13 12h3M19 12h3"/>',
    "align": '<path d="M4 4v16"/><rect x="7" y="6" width="12" height="4" rx="1"/><rect x="7" y="14" width="8" height="4" rx="1"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "function": '<path d="M9 21c2 0 2-2 2.5-5L13 7c.5-3 1-4 3-4"/><path d="M8 11h8"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5z"/><path d="M4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5"/>',
    "variables": '<path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5a2 2 0 0 0 2 2h1"/><path d="M16 3h1a2 2 0 0 1 2 2v5a2 2 0 0 0 2 2 2 2 0 0 0-2 2v5a2 2 0 0 1-2 2h-1"/>',
}


@lru_cache(maxsize=256)
def _icon_image(name: str, color: str, size: int) -> QImage:
    body = ICONS.get(name, ICONS["help"]).replace("currentColor", color)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
           f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return image


def icon(name: str, theme: Theme, color_key: str = "text") -> QIcon:
    c = qcolor(theme.color(color_key))
    color = f"#{c.red():02x}{c.green():02x}{c.blue():02x}"
    result = QIcon()
    for size in (16, 20, 24, 32, 48):
        result.addPixmap(QPixmap.fromImage(_icon_image(name, color, size)))
    return result
