"""Themes: colors and fonts of the application, formulas and plots (JSON files)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

DEFAULT_THEME = "Midnight"


@dataclass
class Theme:
    name: str
    dark: bool
    colors: dict[str, str]
    fonts: dict[str, Any]
    plot_cycle: list[str] = field(default_factory=list)
    radius: int = 8
    path: str = ""

    def color(self, key: str, fallback: str = "#ff00ff") -> str:
        return self.colors.get(key, fallback)

    def font(self, key: str, fallback: Any = None) -> Any:
        return self.fonts.get(key, fallback)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dark": self.dark, "colors": dict(self.colors), "fonts": dict(self.fonts),
                "plot_cycle": list(self.plot_cycle), "radius": self.radius}

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "") -> "Theme":
        base = _builtin_defaults()
        colors = dict(base["colors"])
        colors.update(data.get("colors", {}))
        fonts = dict(base["fonts"])
        fonts.update(data.get("fonts", {}))
        return cls(data.get("name", "Custom"), bool(data.get("dark", False)), colors, fonts,
                   list(data.get("plot_cycle", base["plot_cycle"])), int(data.get("radius", 8)), path)

    def copy(self, **changes) -> "Theme":
        data = self.to_dict()
        data.update(changes)
        return Theme.from_dict(data, self.path)


def _builtin_defaults() -> dict[str, Any]:
    return {
        "colors": {
            "window": "#f4f5f8", "sidebar": "#eceef3", "surface": "#ffffff", "surface_alt": "#f0f2f6",
            "border": "#d9dde5", "text": "#1d2230", "text_muted": "#6b7385", "accent": "#4f5bd5",
            "accent_text": "#ffffff", "page": "#ffffff", "page_grid": "#e6e9ef", "page_shadow": "#00000022",
            "desk": "#e4e7ee", "selection": "#4f5bd533", "hover": "#0000000a", "cursor": "#e5364f",
            "math_text": "#1d2230", "math_variable": "#1d2230", "math_number": "#1d2230",
            "math_operator": "#1d2230", "math_function": "#1d2230", "math_keyword": "#6d3fd1",
            "math_string": "#2b8a3e", "math_result": "#0b6e4f", "math_result_bg": "#0b6e4f10",
            "math_error": "#d6336c", "math_constraint": "#1d2230", "math_symbolic": "#1a5fb4",
            "text_body": "#1d2230", "heading": "#11151f", "table_header": "#eef0f6", "table_border": "#c9ceda",
            "table_highlight": "#ffe8a3", "plot_bg": "#ffffff", "plot_fg": "#2a3040", "plot_grid": "#e3e6ee",
        },
        "fonts": {
            "ui": "Segoe UI", "ui_size": 10, "math": "Cambria Math", "math_size": 17,
            "text": "Segoe UI", "text_size": 11, "mono": "Cascadia Mono", "mono_size": 11,
        },
        "plot_cycle": ["#4f5bd5", "#e4572e", "#17bebb", "#ffc914", "#76b041", "#8e6c8a", "#2e282a", "#f25f5c"],
    }


def themes_dir() -> Path:
    return Path.home() / ".boryscad" / "themes"


def available_themes() -> dict[str, Theme]:
    themes: dict[str, Theme] = {}
    package = resources.files("boryscad.themes")
    for entry in package.iterdir():
        if entry.name.endswith(".json"):
            data = json.loads(entry.read_text(encoding="utf-8"))
            themes[data["name"]] = Theme.from_dict(data, f"builtin:{entry.name}")
    user = themes_dir()
    if user.is_dir():
        for path in sorted(user.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                themes[data["name"]] = Theme.from_dict(data, str(path))
            except (OSError, ValueError, KeyError):
                continue
    return themes


def load_theme(name: str | None) -> Theme:
    themes = available_themes()
    if name and name in themes:
        return themes[name]
    if name and Path(name).is_file():
        return Theme.from_dict(json.loads(Path(name).read_text(encoding="utf-8")), name)
    return themes.get(DEFAULT_THEME) or next(iter(themes.values()))


def save_user_theme(theme: Theme) -> Path:
    directory = themes_dir()
    directory.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in theme.name).strip() or "custom"
    path = directory / f"{safe}.json"
    path.write_text(json.dumps(theme.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
