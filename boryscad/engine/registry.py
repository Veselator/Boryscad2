"""Function registry — the extension point of the engine (Open/Closed principle).

Library modules register builtins through :func:`FunctionRegistry.function`;
new functionality is added by writing a new module with a ``register``
function, never by editing the evaluator.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

log = logging.getLogger(__name__)


@dataclass
class FunctionSpec:
    name: str
    impl: Callable[..., Any]
    lazy: bool = False            # receives (ctx, args_ast, kwargs_ast) instead of values
    sympy: Optional[Callable[..., Any]] = None  # symbolic counterpart
    category: str = "General"
    signature: str = ""
    doc: str = ""
    aliases: tuple[str, ...] = ()
    upright: bool = True          # rendered in upright font
    vectorize: bool = False       # element-wise application on arrays

    def help_line(self) -> str:
        return f"{self.signature or self.name + '(…)'} — {self.doc}"


@dataclass
class SymbolicCommand:
    name: str
    impl: Callable[..., Any]  # (sympy_expr, *args, **kwargs) -> sympy_expr
    doc: str = ""


@dataclass
class FunctionRegistry:
    _functions: dict[str, FunctionSpec] = field(default_factory=dict)
    _sym_commands: dict[str, SymbolicCommand] = field(default_factory=dict)
    _constants: dict[str, Any] = field(default_factory=dict)
    _constant_docs: dict[str, str] = field(default_factory=dict)

    # -------------------------------------------------------------- functions
    def add(self, spec: FunctionSpec) -> FunctionSpec:
        self._functions[spec.name] = spec
        for alias in spec.aliases:
            self._functions[alias] = spec
        return spec

    def function(self, name: str | None = None, *, lazy: bool = False, sympy: Callable | None = None,
                 category: str = "General", signature: str = "", doc: str = "", aliases: Iterable[str] = (),
                 upright: bool = True, vectorize: bool = False):
        """Decorator registering a builtin function."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            fname = name or fn.__name__.rstrip("_")
            sig = signature
            if not sig:
                try:
                    params = [p for p in inspect.signature(fn).parameters.values()]
                    if lazy:
                        params = params[1:]
                    sig = f"{fname}({', '.join(p.name for p in params if p.kind != p.VAR_KEYWORD)})"
                except (TypeError, ValueError):
                    sig = f"{fname}(…)"
            self.add(FunctionSpec(fname, fn, lazy, sympy, category, sig, doc or (fn.__doc__ or "").strip().split("\n")[0],
                                  tuple(aliases), upright, vectorize))
            return fn

        return deco

    def get(self, name: str) -> Optional[FunctionSpec]:
        return self._functions.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._functions

    def names(self) -> list[str]:
        return sorted(self._functions)

    def specs(self) -> list[FunctionSpec]:
        unique = {id(s): s for s in self._functions.values()}
        return sorted(unique.values(), key=lambda s: (s.category, s.name))

    def categories(self) -> dict[str, list[FunctionSpec]]:
        result: dict[str, list[FunctionSpec]] = {}
        for spec in self.specs():
            result.setdefault(spec.category, []).append(spec)
        return result

    # --------------------------------------------------------------- symbolic
    def symbolic_command(self, name: str, doc: str = ""):
        def deco(fn):
            self._sym_commands[name] = SymbolicCommand(name, fn, doc)
            return fn

        return deco

    def get_symbolic(self, name: str) -> Optional[SymbolicCommand]:
        return self._sym_commands.get(name)

    def symbolic_names(self) -> list[str]:
        return sorted(self._sym_commands)

    # -------------------------------------------------------------- constants
    def constant(self, name: str, value: Any, doc: str = "") -> None:
        self._constants[name] = value
        self._constant_docs[name] = doc

    def constants(self) -> dict[str, Any]:
        return dict(self._constants)

    def constant_docs(self) -> dict[str, str]:
        return dict(self._constant_docs)


BUILTIN_MODULES = (
    "boryscad.library.elementary",
    "boryscad.library.linalg",
    "boryscad.library.calculus",
    "boryscad.library.statistics",
    "boryscad.library.strings",
    "boryscad.library.symbolic",
    "boryscad.library.solving",
    "boryscad.library.plotting",
    "boryscad.library.optimization",
    "boryscad.library.transport",
    "boryscad.library.graphs",
    "boryscad.library.integer",
    "boryscad.library.dynamic",
    "boryscad.library.location",
)

_default: Optional[FunctionRegistry] = None


def plugin_dir() -> Path:
    return Path.home() / ".boryscad" / "plugins"


def load_plugins(registry: FunctionRegistry, directory: Path | None = None) -> list[str]:
    """Load user plugins: every ``*.py`` file exposing ``register(registry)``."""
    loaded = []
    directory = directory or plugin_dir()
    if not directory.is_dir():
        return loaded
    for path in sorted(directory.glob("*.py")):
        try:
            spec = importlib.util.spec_from_file_location(f"boryscad_plugin_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            module.register(registry)
            loaded.append(path.stem)
        except Exception:  # pragma: no cover - plugin errors must not break the app
            log.exception("failed to load plugin %s", path)
    return loaded


def default_registry() -> FunctionRegistry:
    global _default
    if _default is None:
        registry = FunctionRegistry()
        for module_name in BUILTIN_MODULES:
            importlib.import_module(module_name).register(registry)
        load_plugins(registry)
        _default = registry
    return _default
