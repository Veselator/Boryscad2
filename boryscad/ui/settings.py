"""Persistent application settings (~/.boryscad/settings.json)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def settings_path() -> Path:
    return Path.home() / ".boryscad" / "settings.json"


@dataclass
class AppSettings:
    theme: str = "Midnight"
    math_font: str = ""            # empty: use the theme font
    math_size: float = 0.0         # 0: use the theme size
    ui_font: str = ""
    ui_size: int = 0
    precision: int = 4
    exp_threshold: int = 6
    page_width: float = 860.0
    show_grid: bool = True
    snap: bool = True
    grid_step: int = 8
    auto_calc: bool = True
    recent: list[str] = field(default_factory=list)
    geometry: str = ""
    window_state: str = ""
    zoom: float = 1.0

    @classmethod
    def load(cls) -> "AppSettings":
        path = settings_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def save(self) -> None:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    def add_recent(self, file: str) -> None:
        file = str(Path(file).resolve())
        self.recent = [file] + [f for f in self.recent if f != file]
        self.recent = self.recent[:12]
