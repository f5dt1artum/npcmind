"""Core service surface for NpcMind.

Provides process health reporting, a stateless behavior-tree evaluator, and
a single-step finite-state-machine advance. Each ``evaluate_behavior`` call
executes exactly one tick of the supplied tree against the supplied
blackboard; each ``step_state_machine`` call processes exactly one event and
returns the deterministic next state. No state is kept between calls.
"""

from __future__ import annotations

from typing import Any

from . import __version__

STATUSES = ("SUCCESS", "FAILURE", "RUNNING")
_COMPOSITE_TYPES = ("sequence", "selector")
_CONDITION_OPS = ("exists", "equals", "not_equals")
_ACTION_OPS = ("set", "delete", "status")
_MACHINE_ACTION_OPS = ("set", "delete")


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


def _validate_machine_condition(condition: Any, where: str) -> None:
    if not isinstance(condition, dict):
        raise MachineError(f"{where}: condition must be an object")
    op = condition.get("op")
    if op not in _CONDITION_OPS:
        raise MachineError(f"{where}: unknown condition op {op!r}")
    if not _is_valid_key(condition.get("key")):
        raise MachineError(f"{where}: condition requires a non-empty string key")
    if op in ("equals", "not_equals") and "value" not in condition:
        raise MachineError(f"{where}: condition op {op!r} requires a value")


def _validate_machine_action(action: Any, where: str) -> None:
    if not isinstance(action, dict):
        raise MachineError(f"{where}: action must be an object")
    op = action.get("op")
    if op not in _MACHINE_ACTION_OPS:
        raise MachineError(f"{where}: unknown action op {op!r}")
    if not _is_valid_key(action.get("key")):
        raise MachineError(f"{where}: action op {op!r} requires a non-empty string key")
    if op == "set" and "value" not in action:
        raise MachineError(f"{where}: action op 'set' requires a value")


def _validate_machine(machine: Any) -> tuple[set[str], list]:
    """Validate the machine definition; return (state ids, transitions)."""
    if not isinstance(machine, dict):
        raise MachineError("machine must be an object")
    states = machine.get("states")
    if not isinstance(states, list) or not states:
        raise MachineError("machine.states must be a non-empty list")
    state_ids: set[str] = set()
    for index, state in enumerate(states):
        where = f"states[{index}]"
        if not isinstance(state, dict):
            raise MachineError(f"{where} must be an object")
        state_id = state.get("id")
        if not _is_valid_key(state_id):
            raise MachineError(f"{where} requires a non-empty string id")
        if state_id in state_ids:
            raise MachineError(f"duplicate state id {state_id!r}")
        state_ids.add(state_id)

    initial = machine.get("initial")
    if not isinstance(initial, str) or initial not in state_ids:
        raise MachineError(f"initial {initial!r} does not reference a declared state")

    transitions = machine.get("transitions")
    if not isinstance(transitions, list):
        raise MachineError("machine.transitions must be a list")
    transition_ids: set[str] = set()
    for index, transition in enumerate(transitions):
        where = f"transitions[{index}]"
        if not isinstance(transition, dict):
            raise MachineError(f"{where} must be an object")
        transition_id = transition.get("id")
        if not _is_valid_key(transition_id):
            raise MachineError(f"{where} requires a non-empty string id")
        if transition_id in transition_ids:
            raise MachineError(f"duplicate transition id {transition_id!r}")
        transition_ids.add(transition_id)
        where = f"transition {transition_id!r}"
        for field in ("from", "to"):
            ref = transition.get(field)
            if not isinstance(ref, str) or ref not in state_ids:
                raise MachineError(f"{where}: {field} {ref!r} does not reference a declared state")
        if not _is_valid_key(transition.get("event")):
            raise MachineError(f"{where}: event must be a non-empty string")
        if "condition" in transition:
            _validate_machine_condition(transition["condition"], where)
        actions = transition.get("actions", [])
        if not isinstance(actions, list):
            raise MachineError(f"{where}: actions must be a list")
        for action_index, action in enumerate(actions):
            _validate_machine_action(action, f"{where}.actions[{action_index}]")
    return state_ids, transitions


def _condition_holds(condition: dict, blackboard: dict) -> bool:
    op = condition["op"]
    key = condition["key"]
    if op == "exists":
        return key in blackboard
    if key not in blackboard:
        return False
    if op == "equals":
        return _json_equal(blackboard[key], condition["value"])
    return not _json_equal(blackboard[key], condition["value"])  # not_equals


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
    """Health reporting, stateless behavior-tree evaluation, and FSM stepping."""

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
        """Advance ``request['machine']`` by exactly one event.

        The caller carries the state explicitly: ``current_state`` defaults
        to the machine's ``initial`` state and the returned ``state`` is the
        deterministic next state. The machine and the request are fully
        validated before any action runs; raises ValueError (MachineError)
        on any structural problem. A call that matches no transition is not
        an error: state and blackboard are returned unchanged and
        ``transition`` is None.
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

        state_ids, transitions = _validate_machine(machine)

        current = request.get("current_state", machine["initial"])
        if not isinstance(current, str) or current not in state_ids:
            raise MachineError(f"current_state {current!r} does not reference a declared state")

        board = dict(blackboard)
        trace: list = []
        hit: dict | None = None
        for transition in transitions:
            if transition["from"] != current or transition["event"] != event:
                continue
            condition = transition.get("condition")
            holds = True if condition is None else _condition_holds(condition, board)
            trace.append({"id": transition["id"], "condition": holds})
            if holds:
                hit = transition
                break

        if hit is None:
            new_state = current
        else:
            for action in hit.get("actions", []):
                if action["op"] == "set":
                    board[action["key"]] = action["value"]
                else:  # delete
                    board.pop(action["key"], None)
            new_state = hit["to"]

        return {
            "previous_state": current,
            "state": new_state,
            "transition": hit["id"] if hit is not None else None,
            "blackboard": board,
            "trace": trace,
        }
