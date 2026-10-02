"""Core service surface for NpcMind.

Provides process health reporting, a stateless behavior-tree evaluator, and
a single-step finite-state-machine driver. Each ``evaluate_behavior`` call
executes exactly one tick of the supplied tree against the supplied
blackboard; each ``step_state_machine`` call advances a machine by exactly
one event. No state is kept between calls.
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


class MachineError(ValueError):
    """Raised when a state-machine request fails structural validation."""


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


def _validate_sm_condition(condition: Any, where: str) -> None:
    if not isinstance(condition, dict):
        raise MachineError(f"{where}: condition must be an object")
    op = condition.get("op")
    if op not in _CONDITION_OPS:
        raise MachineError(f"{where}: unknown condition op {op!r}")
    if not _is_valid_key(condition.get("key")):
        raise MachineError(f"{where}: condition requires a non-empty string key")
    if op in ("equals", "not_equals") and "value" not in condition:
        raise MachineError(f"{where}: condition op {op!r} requires a value")


def _validate_sm_action(action: Any, where: str) -> None:
    if not isinstance(action, dict):
        raise MachineError(f"{where}: action must be an object")
    op = action.get("op")
    if op not in ("set", "delete"):
        raise MachineError(f"{where}: unknown action op {op!r}")
    if not _is_valid_key(action.get("key")):
        raise MachineError(f"{where}: action op {op!r} requires a non-empty string key")
    if op == "set" and "value" not in action:
        raise MachineError(f"{where}: action op 'set' requires a value")


def _validate_machine(machine: Any) -> set[str]:
    """Validate a machine definition fully; return the declared state ids."""
    if not isinstance(machine, dict):
        raise MachineError("machine must be an object")
    states = machine.get("states")
    if not isinstance(states, list) or not states:
        raise MachineError("machine requires a non-empty list of states")
    state_ids: set[str] = set()
    for index, state in enumerate(states):
        state_id = state.get("id") if isinstance(state, dict) else state
        if not _is_valid_key(state_id):
            raise MachineError(f"state at states[{index}] requires a non-empty string id")
        if state_id in state_ids:
            raise MachineError(f"duplicate state id {state_id!r}")
        state_ids.add(state_id)

    initial = machine.get("initial")
    if not isinstance(initial, str) or initial not in state_ids:
        raise MachineError(f"initial references unknown state {initial!r}")

    transitions = machine.get("transitions")
    if not isinstance(transitions, list):
        raise MachineError("machine requires a list of transitions")
    transition_ids: set[str] = set()
    for index, transition in enumerate(transitions):
        where = f"transition at transitions[{index}]"
        if not isinstance(transition, dict):
            raise MachineError(f"{where} must be an object")
        transition_id = transition.get("id")
        if not _is_valid_key(transition_id):
            raise MachineError(f"{where} requires a non-empty string id")
        if transition_id in transition_ids:
            raise MachineError(f"duplicate transition id {transition_id!r}")
        transition_ids.add(transition_id)
        where = f"transition {transition_id!r}"
        for endpoint in ("from", "to"):
            target = transition.get(endpoint)
            if not isinstance(target, str) or target not in state_ids:
                raise MachineError(f"{where}: {endpoint} references unknown state {target!r}")
        if not isinstance(transition.get("event"), str):
            raise MachineError(f"{where}: event must be a string")
        if "condition" in transition:
            _validate_sm_condition(transition["condition"], where)
        actions = transition.get("actions", [])
        if not isinstance(actions, list):
            raise MachineError(f"{where}: actions must be a list")
        for action_index, action in enumerate(actions):
            _validate_sm_action(action, f"{where}.actions[{action_index}]")
    return state_ids


def _sm_condition_holds(condition: dict | None, blackboard: dict) -> bool:
    if condition is None:
        return True
    op = condition["op"]
    key = condition["key"]
    if op == "exists":
        return key in blackboard
    if key not in blackboard:
        return False
    if op == "equals":
        return _json_equal(blackboard[key], condition["value"])
    return not _json_equal(blackboard[key], condition["value"])  # not_equals


class Service:
    """Health reporting, stateless behavior-tree evaluation, FSM stepping."""

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

    def step_state_machine(self, request: dict) -> dict:
        """Advance ``request['machine']`` by exactly one ``request['event']``.

        The machine and the request are validated in full before any action
        runs. The first transition (in declaration order) whose ``from``
        equals the current state, whose ``event`` matches exactly, and whose
        condition holds is taken; its actions then run in order. Raises
        ValueError (MachineError) on any structural problem; a step that
        matches no transition is not an error.
        """
        if not isinstance(request, dict):
            raise MachineError("request must be a JSON object")
        if "machine" not in request:
            raise MachineError("request is missing 'machine'")
        machine = request["machine"]
        event = request.get("event")
        if not isinstance(event, str):
            raise MachineError("event must be a string")
        blackboard = request.get("blackboard", {})
        if not isinstance(blackboard, dict):
            raise MachineError("blackboard must be an object")
        for key in blackboard:
            if not _is_valid_key(key):
                raise MachineError(f"blackboard key {key!r} must be a non-empty string")

        state_ids = _validate_machine(machine)

        current = request.get("current_state", machine["initial"])
        if not isinstance(current, str) or current not in state_ids:
            raise MachineError(f"current_state references unknown state {current!r}")

        board = dict(blackboard)
        trace: list = []
        matched = None
        for transition in machine["transitions"]:
            if transition["from"] != current or transition["event"] != event:
                continue
            holds = _sm_condition_holds(transition.get("condition"), board)
            trace.append({"id": transition["id"], "condition": holds})
            if holds:
                matched = transition
                break

        if matched is None:
            return {
                "previous_state": current,
                "state": current,
                "transition": None,
                "blackboard": board,
                "trace": trace,
            }

        for action in matched.get("actions", []):
            if action["op"] == "set":
                board[action["key"]] = action["value"]
            else:  # delete
                board.pop(action["key"], None)
        return {
            "previous_state": current,
            "state": matched["to"],
            "transition": matched["id"],
            "blackboard": board,
            "trace": trace,
        }
