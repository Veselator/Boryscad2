"""Worksheet document model (independent of GUI and rendering)."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar, Type


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Region:
    kind: ClassVar[str] = "region"
    id: str = field(default_factory=new_id)
    x: float = 40.0
    y: float = 40.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Region":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def clone(self, **changes) -> "Region":
        data = self.to_dict()
        data.update(changes)
        data["id"] = changes.get("id", new_id())
        return type(self).from_dict(data)


@dataclass
class MathRegion(Region):
    kind: ClassVar[str] = "math"
    source: str = ""


@dataclass
class TextRegion(Region):
    kind: ClassVar[str] = "text"
    text: str = ""
    format: str = "markdown"      # markdown | html
    style: str = "body"           # body | h1 | h2 | h3 | note
    width: float = 0.0            # 0 = automatic (page width)


@dataclass
class ImageRegion(Region):
    kind: ClassVar[str] = "image"
    data: str = ""                # base64 encoded bytes
    mime: str = "image/png"
    width: float = 320.0
    height: float = 240.0
    caption: str = ""
    source_path: str = ""


@dataclass
class PlotRegion(Region):
    kind: ClassVar[str] = "plot"
    series: list[str] = field(default_factory=list)
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""
    xlim: str = ""                # expression like "0..10"
    ylim: str = ""
    zlim: str = ""
    grid: bool = True
    legend: bool = True
    equal: bool = False
    mode: str = "2d"              # 2d | 3d | polar
    xscale: str = "linear"
    yscale: str = "linear"
    width: float = 520.0
    height: float = 360.0


@dataclass
class PageBreakRegion(Region):
    kind: ClassVar[str] = "pagebreak"


class RegionFactory:
    """Maps ``kind`` strings to region classes; new region types register here."""

    _classes: dict[str, Type[Region]] = {}

    @classmethod
    def register(cls, region_cls: Type[Region]) -> Type[Region]:
        cls._classes[region_cls.kind] = region_cls
        return region_cls

    @classmethod
    def create(cls, data: dict[str, Any]) -> Region:
        kind = data.get("kind")
        region_cls = cls._classes.get(kind)
        if region_cls is None:
            raise ValueError(f"unknown region kind: {kind!r}")
        return region_cls.from_dict(data)

    @classmethod
    def kinds(cls) -> list[str]:
        return list(cls._classes)


for _cls in (MathRegion, TextRegion, ImageRegion, PlotRegion, PageBreakRegion):
    RegionFactory.register(_cls)


@dataclass
class DocumentSettings:
    page_width: float = 860.0
    margin: float = 48.0
    precision: int = 4
    exp_threshold: int = 6
    theme: str = ""               # empty: use the application theme


@dataclass
class Document:
    regions: list[Region] = field(default_factory=list)
    title: str = "Untitled"
    author: str = ""
    settings: DocumentSettings = field(default_factory=DocumentSettings)
    flow_layout: bool = False     # True: positions are computed (e.g. documents loaded from .bmath)
    rows: list[list[str]] = field(default_factory=list)  # ids placed side by side in flow layout

    def add(self, region: Region) -> Region:
        self.regions.append(region)
        return region

    def remove(self, region_id: str) -> None:
        self.regions = [r for r in self.regions if r.id != region_id]

    def find(self, region_id: str) -> Region | None:
        return next((r for r in self.regions if r.id == region_id), None)

    def ordered(self) -> list[Region]:
        """Calculation order: top to bottom, then left to right (Mathcad semantics)."""
        if self.flow_layout:
            return list(self.regions)
        return sorted(self.regions, key=lambda r: (round(r.y / 4), r.x))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "boryscad",
            "version": 2,
            "title": self.title,
            "author": self.author,
            "settings": asdict(self.settings),
            "flow_layout": self.flow_layout,
            "rows": self.rows,
            "regions": [r.to_dict() for r in self.regions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        settings_names = {f.name for f in fields(DocumentSettings)}
        settings = DocumentSettings(**{k: v for k, v in data.get("settings", {}).items() if k in settings_names})
        return cls(
            regions=[RegionFactory.create(r) for r in data.get("regions", [])],
            title=data.get("title", "Untitled"),
            author=data.get("author", ""),
            settings=settings,
            flow_layout=data.get("flow_layout", False),
            rows=data.get("rows", []),
        )
