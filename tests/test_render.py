from pathlib import Path

import pytest

from boryscad.core.bmath import read_bmath, write_bmath
from boryscad.core.document import Document
from boryscad.core.io import load_document
from boryscad.core.theme import available_themes, load_theme

EXAMPLES = sorted((Path(__file__).parents[1] / "boryscad" / "examples").glob("*.bmath"))


def test_themes_load():
    themes = available_themes()
    assert {"Midnight", "Paper"} <= set(themes)


def test_bmath_roundtrip():
    doc = read_bmath("# Title\n> text\n\nx := 1\nx =\n\n@plot title=\"t\"\nfplot(x, x, 0, 1)\n@end\n")
    kinds = [r.kind for r in doc.regions]
    assert kinds == ["text", "text", "math", "plot"]
    again = read_bmath(write_bmath(doc))
    assert [r.kind for r in again.regions] == kinds
    assert Document.from_dict(doc.to_dict()).to_dict() == doc.to_dict()


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_examples_calculate_and_render(path, tmp_path):
    from boryscad.render import ensure_qt_app
    from boryscad.render.document import DocumentRenderer
    ensure_qt_app()
    doc = load_document(path)
    renderer = DocumentRenderer(load_theme("Paper"), scale=1.0)
    results = renderer.calculate(doc)
    assert not results.errors(), results.errors()
    out = renderer.export(doc, tmp_path / (path.stem + ".html"), results)
    assert out.stat().st_size > 1000
