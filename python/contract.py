"""Python side of the Go <-> Python contract.

Reads the SAME contracts/task.spec.schema.json the Go server reads. Defaults
and required fields are never hardcoded here.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULT_SCHEMA = Path(__file__).resolve().parent.parent / "contracts" / "task.spec.schema.json"


def load_schema(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_SCHEMA
    return json.loads(p.read_text(encoding="utf-8"))


def normalize(spec: dict, schema: dict) -> dict:
    """Return a validated copy with schema defaults applied."""
    node = copy.deepcopy(spec)
    _apply_defaults(node, schema)
    _validate(node, schema, "$")
    return node


def _apply_defaults(node: dict, sch: dict) -> None:
    for name, prop in sch.get("properties", {}).items():
        if name not in node and "default" in prop:
            node[name] = prop["default"]
        if prop.get("properties") and isinstance(node.get(name), dict):
            _apply_defaults(node[name], prop)


def _validate(node: dict, sch: dict, path: str) -> None:
    for name in sch.get("required", []):
        if name not in node:
            raise ValueError(f"{path}: missing required field {name!r}")
    for name, prop in sch.get("properties", {}).items():
        if name not in node:
            continue
        value = node[name]
        kind = prop.get("type")
        if kind == "object" or prop.get("properties"):
            if not isinstance(value, dict):
                raise ValueError(f"{path}.{name}: expected object")
            _validate(value, prop, f"{path}.{name}")
        if kind == "string" and not isinstance(value, str):
            raise ValueError(f"{path}.{name}: expected string, got {type(value).__name__}")
        if kind == "integer" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ValueError(f"{path}.{name}: expected integer, got {value!r}")
        choices = prop.get("enum")
        if choices and value not in choices:
            raise ValueError(f"{path}.{name}: {value!r} not in {choices}")
