"""Core service surface for NpcMind.

Provides process health reporting plus a stateless behavior-tree evaluator.
Each ``evaluate_behavior`` call executes exactly one tick of the supplied
tree against the supplied blackboard; no state is kept between calls.
"""

from __future__ import annotations

from typing import Any

from . import __version__

STATUSES = ("SUCCESS", "FAILURE", "RUNNING")
_COMPOSITE_TYPES = ("sequence", "selector")
_CONDITION_OPS = ("exists", "equals", "not_equals")
_ACTION_OPS = ("set", "delete", "status")


class TreeError(ValueError):
    """Raised when a behavior-tree request fails structural validation."""


def _is_valid_key(key: Any) -> bool:
    return isinstance(key, str) and key != ""


def _json_equal(a: Any, b: Any) -> bool:
    """JSON type-sensitive equality: booleans never equal numbers."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def _validate_node(node: Any, path: str, seen_ids: set[str]) -> None:
    if not isinstance(node, dict):
        raise TreeError(f"node at {path} must be an object")
    node_id = node.get("id")
    if not isinstance(node_id, str) or node_id == "":
        raise TreeError(f"node at {path} requires a non-empty string id")
    if node_id in seen_ids:
        raise TreeError(f"duplicate node id {node_id!r} at {path}")
    seen_ids.add(node_id)
    where = f"node {node_id!r}"

    node_type = node.get("type")
    if node_type in _COMPOSITE_TYPES:
        children = node.get("children", [])
        if not isinstance(children, list):
            raise TreeError(f"{where}: children must be a list")
        for index, child in enumerate(children):
            _validate_node(child, f"{path}.children[{index}]", seen_ids)
    elif node_type == "condition":
        op = node.get("op")
        if op not in _CONDITION_OPS:
            raise TreeError(f"{where}: unknown condition op {op!r}")
        if not _is_valid_key(node.get("key")):
            raise TreeError(f"{where}: condition requires a non-empty string key")
        if op in ("equals", "not_equals") and "value" not in node:
            raise TreeError(f"{where}: condition op {op!r} requires a value")
    elif node_type == "action":
        op = node.get("op")
        if op not in _ACTION_OPS:
            raise TreeError(f"{where}: unknown action op {op!r}")
        if op in ("set", "delete") and not _is_valid_key(node.get("key")):
            raise TreeError(f"{where}: action op {op!r} requires a non-empty string key")
        if op == "set" and "value" not in node:
            raise TreeError(f"{where}: action op 'set' requires a value")
        if op == "status" and node.get("status") not in STATUSES:
            raise TreeError(f"{where}: illegal status {node.get('status')!r}")
    else:
        raise TreeError(f"{where}: unknown node type {node_type!r}")


def _run_condition(node: dict, blackboard: dict) -> str:
    op = node["op"]
    key = node["key"]
    if op == "exists":
        ok = key in blackboard
    elif key not in blackboard:
        ok = False
    elif op == "equals":
        ok = _json_equal(blackboard[key], node["value"])
    else:  # not_equals
        ok = not _json_equal(blackboard[key], node["value"])
    return "SUCCESS" if ok else "FAILURE"


def _run_action(node: dict, blackboard: dict) -> str:
    op = node["op"]
    if op == "set":
        blackboard[node["key"]] = node["value"]
        return "SUCCESS"
    if op == "delete":
        blackboard.pop(node["key"], None)
        return "SUCCESS"
    return node["status"]  # op == "status"


def _tick(node: dict, blackboard: dict, trace: list) -> str:
    node_type = node["type"]
    if node_type == "sequence":
        status = "SUCCESS"
        for child in node.get("children", []):
            status = _tick(child, blackboard, trace)
            if status != "SUCCESS":
                break
    elif node_type == "selector":
        status = "FAILURE"
        for child in node.get("children", []):
            status = _tick(child, blackboard, trace)
            if status != "FAILURE":
                break
    elif node_type == "condition":
        status = _run_condition(node, blackboard)
    else:  # action
        status = _run_action(node, blackboard)
    trace.append({"id": node["id"], "type": node_type, "status": status})
    return status


class Service:
    """Health reporting plus stateless behavior-tree evaluation."""

    name = "npcmind"
    version = __version__

    def health(self) -> dict[str, str]:
        return {"status": "ok", "service": self.name, "version": self.version}

    def evaluate_behavior(self, request: dict) -> dict:
        """Execute one tick of ``request['tree']`` against the blackboard.

        Raises ValueError (TreeError) when the request structure, the tree,
        or the blackboard is invalid; no partial result is produced.
        """
        if not isinstance(request, dict):
            raise TreeError("request must be a JSON object")
        if "tree" not in request:
            raise TreeError("request is missing 'tree'")
        tree = request["tree"]
        blackboard = request.get("blackboard", {})
        if not isinstance(blackboard, dict):
            raise TreeError("blackboard must be an object")
        for key in blackboard:
            if not _is_valid_key(key):
                raise TreeError(f"blackboard key {key!r} must be a non-empty string")

        _validate_node(tree, "root", set())

        board = dict(blackboard)
        trace: list = []
        status = _tick(tree, board, trace)
        return {"status": status, "blackboard": board, "trace": trace}
