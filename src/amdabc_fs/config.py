"""Load a YAML config; ``inherit:`` merges a parent file (deep merge)."""
from __future__ import annotations

from pathlib import Path

import yaml


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load(path.parent / parent), cfg)
    return cfg


def seeds(n: int) -> list[int]:
    return list(range(n))
