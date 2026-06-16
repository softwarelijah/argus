"""YAML config loading with light dotted-key access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports attribute and dotted-key access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a :class:`Config`."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return Config(data)
