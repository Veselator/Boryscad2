"""Loading and saving documents (.bcad JSON and .bmath text)."""
from __future__ import annotations

import json
from pathlib import Path

from .bmath import read_bmath, write_bmath
from .document import Document

NATIVE_EXT = ".bcad"
TEXT_EXT = ".bmath"


def load_document(path: str | Path) -> Document:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == TEXT_EXT or not text.lstrip().startswith("{"):
        doc = read_bmath(text, path.parent)
        if doc.title == "Untitled":
            doc.title = path.stem
        return doc
    return Document.from_dict(json.loads(text))


def save_document(doc: Document, path: str | Path) -> None:
    path = Path(path)
    if path.suffix.lower() == TEXT_EXT:
        path.write_text(write_bmath(doc), encoding="utf-8")
    else:
        path.write_text(json.dumps(doc.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
