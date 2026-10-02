"""Core service surface for NpcMind.

The frozen baseline only reports process health. Later work adds the real
capabilities described in README.md behind this module; keep the public
surface here backward compatible.
"""

from __future__ import annotations

import copy
from typing import Any

from . import __version__

STATUSES = ("SUCCESS", "FAILURE", "RUNNING")
_COMPOSITE_TYPES = ("sequence", "selector")
_CONDITION_OPS = ("exists", "equals", "not_equals")
_ACTION_OPS = ("set", "delete", "status")


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _json_equal(a: Any, b: Any) -> bool:
    """Type-sensitive equality over JSON values (true != 1, 1 == 1.0)."""
    ta, tb = _json_type(a), _json_type(b)
    if ta != tb:
        return False
    if ta == "array":
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if ta == "object":
        return set(a) == set(b) and all(_json_equal(a[k], b[k]) for k in a)
    return a == b


def _validate_blackboard(blackboard: Any) -> None:
    if not isinstance(blackboard, dict):
        raise ValueError("blackboard must be an object")
    for key in blackboard:
        if not isinstance(key, str) or not key:
            raise ValueError(f"invalid blackboard key {key!r}: keys must be non-empty strings")


def _validate_node(node: Any, path: str, seen_ids: set[str]) -> None:
    if not isinstance(node, dict):
        raise ValueError(f"node at {path} must be an object")
    node_id = node.get("id")
    if not isinstance(node_id, str) or not node_id:
        raise ValueError(f"node at {path}: id must be a non-empty string")
    where = f"node {node_id!r}"
    if node_id in seen_ids:
        raise ValueError(f"duplicate id {node_id!r} at {path}")
    seen_ids.add(node_id)

    node_type = node.get("type")
    if node_type in _COMPOSITE_TYPES:
        children = node.get("children")
        if not isinstance(children, list):
            raise ValueError(f"{where}: children must be a list")
        for index, child in enumerate(children):
            _validate_node(child, f"{path}.children[{index}]", seen_ids)
    elif node_type == "condition":
        key = node.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{where}: key must be a non-empty string")
        op = node.get("op")
        if op not in _CONDITION_OPS:
            raise ValueError(f"{where}: unknown condition op {op!r}")
        if op in ("equals", "not_equals") and "value" not in node:
            raise ValueError(f"{where}: missing value for {op}")
    elif node_type == "action":
        op = node.get("op")
        if op not in _ACTION_OPS:
            raise ValueError(f"{where}: unknown action op {op!r}")
        if op in ("set", "delete"):
            key = node.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError(f"{where}: key must be a non-empty string")
        if op == "set" and "value" not in node:
            raise ValueError(f"{where}: missing value for set")
        if op == "status" and node.get("status") not in STATUSES:
            raise ValueError(f"{where}: status must be one of {', '.join(STATUSES)}")
    else:
        raise ValueError(f"{where}: unknown type {node_type!r}")


def _eval_condition(node: dict, blackboard: dict) -> str:
    key = node["key"]
    op = node["op"]
    if op == "exists":
        return "SUCCESS" if key in blackboard else "FAILURE"
    if key not in blackboard:
        return "FAILURE"
    equal = _json_equal(blackboard[key], node["value"])
    if op == "equals":
        return "SUCCESS" if equal else "FAILURE"
    return "FAILURE" if equal else "SUCCESS"


def _run_action(node: dict, blackboard: dict) -> str:
    op = node["op"]
    if op == "set":
        blackboard[node["key"]] = copy.deepcopy(node["value"])
        return "SUCCESS"
    if op == "delete":
        blackboard.pop(node["key"], None)
        return "SUCCESS"
    return node["status"]


def _tick(node: dict, blackboard: dict, trace: list[dict]) -> str:
    node_type = node["type"]
    if node_type == "sequence":
        status = "SUCCESS"
        for child in node["children"]:
            status = _tick(child, blackboard, trace)
            if status != "SUCCESS":
                break
    elif node_type == "selector":
        status = "FAILURE"
        for child in node["children"]:
            status = _tick(child, blackboard, trace)
            if status != "FAILURE":
                break
    elif node_type == "condition":
        status = _eval_condition(node, blackboard)
    else:
        status = _run_action(node, blackboard)
    trace.append({"id": node["id"], "type": node_type, "status": status})
    return status


class Service:
    """NpcMind service: health reporting and single-tick behavior trees."""

    name = "npcmind"
    version = __version__

    def health(self) -> dict[str, str]:
        return {"status": "ok", "service": self.name, "version": self.version}

    def evaluate_behavior(self, request: dict) -> dict:
        """Run one tick of the given behavior tree against the blackboard.

        Raises ValueError for any structural or field error in the request;
        no partial result is produced in that case.
        """
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        if "tree" not in request:
            raise ValueError("missing tree")
        tree = request["tree"]
        blackboard = request.get("blackboard", {})
        _validate_blackboard(blackboard)
        _validate_node(tree, "root", set())

        working = copy.deepcopy(blackboard)
        trace: list[dict] = []
        status = _tick(tree, working, trace)
        return {"status": status, "blackboard": working, "trace": trace}
