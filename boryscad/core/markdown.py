"""Tiny Markdown → HTML converter for text regions (no external dependency)."""
from __future__ import annotations

import html
import re

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_CODE = re.compile(r"`([^`]+)`")
_SUB = re.compile(r"~(\S+?)~")
_SUP = re.compile(r"\^(\S+?)\^")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def inline(text: str) -> str:
    out = html.escape(text, quote=False)
    out = _CODE.sub(r"<code>\1</code>", out)
    out = _BOLD.sub(r"<b>\1</b>", out)
    out = _ITALIC.sub(r"<i>\1</i>", out)
    out = _SUB.sub(r"<sub>\1</sub>", out)
    out = _SUP.sub(r"<sup>\1</sup>", out)
    out = _LINK.sub(r'<a href="\2">\1</a>', out)
    return out


def to_html(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    parts: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    list_tag = "ul"

    def flush_paragraph():
        if paragraph:
            parts.append("<p>" + "<br/>".join(inline(l) for l in paragraph) + "</p>")
            paragraph.clear()

    def flush_list():
        if list_items:
            parts.append(f"<{list_tag}>" + "".join(f"<li>{inline(i)}</li>" for i in list_items) + f"</{list_tag}>")
            list_items.clear()

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        heading = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        bullet = re.match(r"^[-*•]\s+(.*)$", stripped)
        numbered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if not stripped:
            flush_paragraph()
            flush_list()
        elif heading:
            flush_paragraph()
            flush_list()
            level = len(heading.group(1))
            parts.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
        elif bullet or numbered:
            flush_paragraph()
            tag = "ul" if bullet else "ol"
            if list_items and tag != list_tag:
                flush_list()
            list_tag = tag
            list_items.append((bullet or numbered).group(1))
        else:
            flush_list()
            paragraph.append(stripped)
    flush_paragraph()
    flush_list()
    return "".join(parts)


def to_plain(text: str) -> str:
    out = _BOLD.sub(r"\1", text)
    out = _ITALIC.sub(r"\1", out)
    out = _CODE.sub(r"\1", out)
    return out
