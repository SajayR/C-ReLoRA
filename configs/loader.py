"""Tiny YAML config loader with dotlist override support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _set_nested(config: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def load_config(path_or_name: str, overrides: Iterable[str] | None = None) -> Dict[str, Any]:
    path = Path(path_or_name)
    if not path.exists():
        candidate = Path("configs") / f"{path_or_name}.yaml"
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Could not locate config: {path_or_name}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be key=value, got: {item}")
            key, raw_value = item.split("=", 1)
            _set_nested(config, key, _coerce_value(raw_value))

    return config


__all__ = ["load_config"]
