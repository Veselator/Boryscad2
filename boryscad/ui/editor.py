"""Code editor used inside math and text regions: highlighting and completion."""
from __future__ import annotations

import re

from PySide6.QtCore import QStringListModel, Qt, Signal
from PySide6.QtGui import QFont, QKeyEvent, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QCompleter, QFrame, QPlainTextEdit

from ..core.theme import Theme
from ..render.qt_backend import qcolor

KEYWORDS = {"given", "for", "in", "while", "if", "else", "return", "break", "continue", "and", "or", "not", "step"}


class MathHighlighter(QSyntaxHighlighter):
    TOKEN = re.compile(r'(?P<comment>//.*)|(?P<string>"[^"\n]*")|(?P<number>\b\d+(\.\d+)?([eE][+-]?\d+)?i?\b)|'
                       r'(?P<name>[^\W\d]\w*)|(?P<op>:=|<-|->|==|<=|>=|!=|[-+*/^=<>|!.,;\[\](){}√≤≥≠→←≔·])')

    def __init__(self, document, theme: Theme, functions: set[str]):
        super().__init__(document)
        self.functions = functions
        self.set_theme(theme)

    def set_theme(self, theme: Theme) -> None:
        def fmt(key, bold=False, italic=False):
            f = QTextCharFormat()
            f.setForeground(qcolor(theme.color(key)))
            if bold:
                f.setFontWeight(QFont.Weight.Bold)
            f.setFontItalic(italic)
            return f
        self.formats = {
            "comment": fmt("text_muted", italic=True), "string": fmt("math_string"), "number": fmt("math_number"),
            "keyword": fmt("math_keyword", bold=True), "function": fmt("math_function"), "name": fmt("math_variable"),
            "op": fmt("math_operator"),
        }
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:
        for m in self.TOKEN.finditer(text):
            kind = m.lastgroup
            if kind == "name":
                word = m.group("name")
                kind = "keyword" if word in KEYWORDS else "function" if word in self.functions else "name"
            if kind in self.formats:
                self.setFormat(m.start(), m.end() - m.start(), self.formats[kind])


class RegionEditor(QPlainTextEdit):
    committed = Signal()
    cancelled = Signal()
    navigate = Signal(str)

    def __init__(self, theme: Theme, words: list[str], functions: set[str], math: bool = True, parent=None):
        super().__init__(parent)
        self.math = math
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap if math else QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTabChangesFocus(False)
        self.highlighter = MathHighlighter(self.document(), theme, functions) if math else None
        self.completer = QCompleter(sorted(set(words)), self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseSensitive)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.completer.setWidget(self)
        self.completer.activated.connect(self._insert_completion)
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        font = QFont(theme.font("mono", "Consolas") if self.math else theme.font("text", "Segoe UI"))
        font.setPixelSize(int(float(theme.font("mono_size" if self.math else "text_size", 11)) * 1.33))
        self.setFont(font)
        bg = theme.color("surface")
        self.setStyleSheet(
            f"QPlainTextEdit {{ background: {_css(bg)}; color: {_css(theme.color('text'))}; border: 1.5px solid "
            f"{_css(theme.color('accent'))}; border-radius: 6px; padding: 2px 4px; }}")
        if self.highlighter:
            self.highlighter.set_theme(theme)

    def set_words(self, words: list[str]) -> None:
        self.completer.setModel(QStringListModel(sorted(set(words)), self.completer))

    # --------------------------------------------------------------- keys
    def keyPressEvent(self, event: QKeyEvent) -> None:
        popup = self.completer.popup()
        if popup.isVisible() and event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Tab,
                                                 Qt.Key.Key_Escape, Qt.Key.Key_Backtab):
            event.ignore()
            if event.key() == Qt.Key.Key_Escape:
                popup.hide()
                event.accept()
            elif event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Return, Qt.Key.Key_Enter):
                index = popup.currentIndex()
                if index.isValid():
                    self._insert_completion(index.data())
                popup.hide()
                event.accept()
            return
        key = event.key()
        mods = event.modifiers()
        if key == Qt.Key.Key_Escape:
            self.committed.emit()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
            shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
            if not self.math:
                if ctrl:
                    self.committed.emit()
                else:
                    self.insertPlainText("\n")
                return
            text = self.toPlainText()
            open_braces = text.count("{") - text.count("}")
            if shift or ctrl or open_braces > 0:
                line = self.textCursor().block().text()
                indent = re.match(r"\s*", line).group(0)
                extra = "  " if line.rstrip().endswith("{") else ""
                self.insertPlainText("\n" + indent + extra)
                return
            self.committed.emit()
            return
        if key == Qt.Key.Key_Tab:
            self.insertPlainText("  ")
            return
        if key == Qt.Key.Key_BraceRight:
            cursor = self.textCursor()
            line = cursor.block().text()
            if line.strip() == "" and len(line) >= 2:
                for _ in range(2):
                    cursor.deletePreviousChar()
        super().keyPressEvent(event)
        self._maybe_complete(event)

    def _word_under_cursor(self) -> str:
        cursor = self.textCursor()
        text = cursor.block().text()[:cursor.positionInBlock()]
        m = re.search(r"[^\W\d]\w*$", text)
        return m.group(0) if m else ""

    def _maybe_complete(self, event: QKeyEvent) -> None:
        if not self.math:
            return
        prefix = self._word_under_cursor()
        if len(prefix) < 2 or not event.text() or not (event.text().isalnum() or event.text() == "_"):
            self.completer.popup().hide()
            return
        if prefix != self.completer.completionPrefix():
            self.completer.setCompletionPrefix(prefix)
            self.completer.popup().setCurrentIndex(self.completer.completionModel().index(0, 0))
        if self.completer.completionCount() == 0 or (self.completer.completionCount() == 1 and
                                                     self.completer.currentCompletion() == prefix):
            self.completer.popup().hide()
            return
        rect = self.cursorRect()
        rect.setWidth(260)
        self.completer.complete(rect)

    def _insert_completion(self, completion: str) -> None:
        prefix = self._word_under_cursor()
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Left, QTextCursor.MoveMode.KeepAnchor, len(prefix))
        cursor.insertText(completion)
        self.setTextCursor(cursor)

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        if self.completer.popup().isVisible():
            return
        if event.reason() in (Qt.FocusReason.PopupFocusReason, Qt.FocusReason.ActiveWindowFocusReason):
            return
        self.committed.emit()


def _css(color: str) -> str:
    c = qcolor(color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{c.alpha()})"
