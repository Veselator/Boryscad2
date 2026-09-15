"""Lexical environments (scopes) for worksheet and program evaluation."""
from __future__ import annotations

from typing import Any, Iterator, Optional


class Environment:
    __slots__ = ("vars", "parent")

    def __init__(self, parent: Optional["Environment"] = None, values: dict[str, Any] | None = None):
        self.vars: dict[str, Any] = dict(values or {})
        self.parent = parent

    def lookup(self, name: str) -> Any:
        env: Optional[Environment] = self
        while env is not None:
            if name in env.vars:
                return env.vars[name]
            env = env.parent
        raise KeyError(name)

    def has(self, name: str) -> bool:
        env: Optional[Environment] = self
        while env is not None:
            if name in env.vars:
                return True
            env = env.parent
        return False

    def define(self, name: str, value: Any) -> None:
        self.vars[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self.lookup(name)
        except KeyError:
            return default

    def child(self, values: dict[str, Any] | None = None) -> "Environment":
        return Environment(self, values)

    def items(self) -> Iterator[tuple[str, Any]]:
        seen: set[str] = set()
        env: Optional[Environment] = self
        while env is not None:
            for k, v in env.vars.items():
                if k not in seen:
                    seen.add(k)
                    yield k, v
            env = env.parent

    def snapshot(self) -> "Environment":
        """Shallow copy of the whole chain flattened into one scope."""
        return Environment(None, dict(reversed(list(self.items()))))
