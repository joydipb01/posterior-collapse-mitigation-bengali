"""YAML config loading with inheritance and ``--set`` overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A nested dict with dotted-path access."""

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set_path(self, path: str, value: Any) -> None:
        keys = path.split(".")
        node: Any = self
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config, resolving an optional ``inherit:`` parent."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    parent = raw.pop("inherit", None)
    if parent:
        base = load_config((path.parent / parent).resolve())
        raw = deep_merge(base.to_dict(), raw)

    cfg = Config(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"bad override {item!r}; expected key=value")
        key, _, value = item.partition("=")
        cfg.set_path(key.strip(), yaml.safe_load(value))
    return cfg
