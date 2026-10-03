"""Core service surface for NpcMind.

Provides process health reporting, a stateless behavior-tree evaluator, a
single-step finite-state-machine driver, one-shot goal-oriented action
planning, and stateless 2D grid navigation. Each ``evaluate_behavior`` call
executes exactly one tick of the supplied tree against the supplied
blackboard; each ``step_state_machine`` call advances a machine by exactly
one event; each ``plan_goap`` call searches for a minimum-cost action
sequence from the supplied world state to the supplied goal; each
``find_path`` call searches a one-shot grid request for a minimum-cost
orthogonal route; each ``select_utility`` call scores the supplied options
against the supplied context once and picks the best; each
``update_perception`` call merges one batch of observations into the
supplied memory snapshot and ages records out by retention. No state is
kept between calls.
"""

from __future__ import annotations

import heapq
import math
from copy import deepcopy
from fractions import Fraction
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


class GoapError(ValueError):
    """Raised when a GOAP planning request fails validation."""


class NavigationError(ValueError):
    """Raised when a navigation request fails validation."""


class UtilityError(ValueError):
    """Raised when a utility-selection request fails validation."""


class PerceptionError(ValueError):
    """Raised when a perception-memory request fails validation."""


_UTILITY_CURVES = ("linear", "inverse")


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validate_utility_consideration(consideration: Any, where: str) -> None:
    if not isinstance(consideration, dict):
        raise UtilityError(f"{where} must be an object")
    if not _is_valid_key(consideration.get("key")):
        raise UtilityError(f"{where} requires a non-empty string key")
    low = consideration.get("min")
    high = consideration.get("max")
    if not _is_finite_number(low):
        raise UtilityError(f"{where}: min must be a finite number")
    if not _is_finite_number(high):
        raise UtilityError(f"{where}: max must be a finite number")
    if not low < high:
        raise UtilityError(f"{where}: min must be less than max")
    curve = consideration.get("curve")
    if curve not in _UTILITY_CURVES:
        raise UtilityError(f"{where}: unknown curve {curve!r}")
    weight = consideration.get("weight", 1)
    if not _is_finite_number(weight) or weight <= 0:
        raise UtilityError(f"{where}: weight must be a positive finite number")


_NAV_NEIGHBORS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _validate_nav_grid(grid: Any) -> tuple[int, int]:
    """Validate a non-empty rectangular grid of null/positive-int cells.

    Returns ``(width, height)``.
    """
    if not isinstance(grid, list) or not grid:
        raise NavigationError("grid must be a non-empty rectangular 2D array")
    first = grid[0]
    if not isinstance(first, list) or not first:
        raise NavigationError("grid must be a non-empty rectangular 2D array")
    width = len(first)
    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            raise NavigationError("grid must be a non-empty rectangular 2D array")
        for cell in row:
            if cell is None:
                continue
            if isinstance(cell, bool) or not isinstance(cell, int) or cell <= 0:
                raise NavigationError("grid cells must be null or positive integers")
    return width, len(grid)


def _validate_nav_coordinate(value: Any, where: str) -> tuple[int, int]:
    """Validate ``{"x": int, "y": int}``; booleans are not integers."""
    if not isinstance(value, dict):
        raise NavigationError(f"{where} must be an object")
    if "x" not in value or "y" not in value:
        raise NavigationError(f"{where} requires integer fields x and y")
    x = value["x"]
    y = value["y"]
    if isinstance(x, bool) or not isinstance(x, int):
        raise NavigationError(f"{where}.x must be an integer")
    if isinstance(y, bool) or not isinstance(y, int):
        raise NavigationError(f"{where}.y must be an integer")
    return x, y


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


def _validate_json_value(value: Any, where: str) -> None:
    """Ensure ``value`` is a legal JSON value with no non-finite numbers."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GoapError(f"{where}: non-finite number is not a legal JSON value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise GoapError(f"{where}: object key {key!r} must be a string")
            _validate_json_value(item, f"{where}.{key}")
        return
    raise GoapError(f"{where}: {type(value).__name__} is not a legal JSON value")


def _validate_goap_state(state: dict, where: str) -> None:
    for key in state:
        if not _is_valid_key(key):
            raise GoapError(f"{where} key {key!r} must be a non-empty string")
    for key, value in state.items():
        _validate_json_value(value, f"{where}[{key!r}]")


def _freeze_value(value: Any) -> tuple:
    """Hashable canonical form honouring _json_equal (bool != number)."""
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", Fraction(value))
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, list):
        return ("list", tuple(_freeze_value(item) for item in value))
    return ("dict", tuple(sorted((key, _freeze_value(item)) for key, item in value.items())))


def _freeze_state(state: dict) -> tuple:
    return tuple(sorted((key, _freeze_value(value)) for key, value in state.items()))


def _goap_conditions_met(conditions: dict, state: dict) -> bool:
    return all(key in state and _json_equal(state[key], value) for key, value in conditions.items())


def _validate_perception_json(value: Any, where: str) -> None:
    """Ensure ``value`` is a legal JSON value with no non-finite numbers."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PerceptionError(f"{where}: non-finite number is not a legal JSON value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_perception_json(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PerceptionError(f"{where}: object key {key!r} must be a string")
            _validate_perception_json(item, f"{where}.{key}")
        return
    raise PerceptionError(f"{where}: {type(value).__name__} is not a legal JSON value")


def _validate_perception_entity(entity: Any, where: str, now: float, *, with_last_seen: bool) -> dict:
    """Validate one memory record or observation fully; return a parsed copy."""
    if not isinstance(entity, dict):
        raise PerceptionError(f"{where} must be an object")
    entity_id = entity.get("id")
    if not _is_valid_key(entity_id):
        raise PerceptionError(f"{where} requires a non-empty string id")
    if "kind" not in entity:
        raise PerceptionError(f"{where} is missing 'kind'")
    kind = entity["kind"]
    if not _is_valid_key(kind):
        raise PerceptionError(f"{where}: kind must be a non-empty string")
    last_seen = None
    if with_last_seen:
        if "last_seen" not in entity:
            raise PerceptionError(f"{where} is missing 'last_seen'")
        last_seen = entity["last_seen"]
        if not _is_finite_number(last_seen) or last_seen < 0:
            raise PerceptionError(f"{where}: last_seen must be a non-negative finite number")
        if last_seen > now:
            raise PerceptionError(f"{where}: last_seen must not be in the future")
    if "confidence" not in entity:
        raise PerceptionError(f"{where} is missing 'confidence'")
    confidence = entity["confidence"]
    if not _is_finite_number(confidence) or not 0 <= confidence <= 1:
        raise PerceptionError(f"{where}: confidence must be a finite number between 0 and 1")
    if "position" not in entity:
        raise PerceptionError(f"{where} is missing 'position'")
    position = entity["position"]
    if not isinstance(position, dict) or "x" not in position or "y" not in position:
        raise PerceptionError(f"{where}: position must be an object with finite fields x and y")
    if not _is_finite_number(position["x"]) or not _is_finite_number(position["y"]):
        raise PerceptionError(f"{where}: position.x and position.y must be finite numbers")
    if "data" in entity:
        data = entity["data"]
        if not isinstance(data, dict):
            raise PerceptionError(f"{where}: data must be an object")
        _validate_perception_json(data, f"{where}.data")
    parsed = {
        "id": entity_id,
        "kind": kind,
        "confidence": confidence,
        "position": {"x": position["x"], "y": position["y"]},
        "data": deepcopy(entity["data"]) if "data" in entity else {},
    }
    if with_last_seen:
        parsed["last_seen"] = last_seen
    return parsed


class Service:
    """Health, behavior-tree evaluation, FSM stepping, GOAP, navigation, utility, perception."""
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

    def plan_goap(self, request: dict) -> dict:
        """Find a minimum-cost action sequence from ``world`` to ``goal``.

        The request is validated in full before any search happens. Actions
        may only fire when every precondition key exists in the current
        state with an equal value; their effects then overwrite the matching
        keys while all other state entries are kept. The goal is satisfied
        once every goal key exists with an equal value (extra state keys are
        allowed). The returned plan minimises total cost; ties are broken
        by comparing the sequences of action positions in the input list
        lexicographically and choosing the smallest. Raises ValueError
        (GoapError) on any invalid input; the request is never mutated.
        """
        if not isinstance(request, dict):
            raise GoapError("request must be a JSON object")
        world = request.get("world")
        if not isinstance(world, dict):
            raise GoapError("world must be an object")
        goal = request.get("goal")
        if not isinstance(goal, dict):
            raise GoapError("goal must be an object")
        actions = request.get("actions")
        if not isinstance(actions, list):
            raise GoapError("actions must be a list")
        _validate_goap_state(world, "world")
        _validate_goap_state(goal, "goal")

        parsed_actions: list[tuple[str, int, dict, dict]] = []
        seen_ids: set[str] = set()
        for index, action in enumerate(actions):
            where = f"action at actions[{index}]"
            if not isinstance(action, dict):
                raise GoapError(f"{where} must be an object")
            action_id = action.get("id")
            if not _is_valid_key(action_id):
                raise GoapError(f"{where} requires a non-empty string id")
            if action_id in seen_ids:
                raise GoapError(f"duplicate action id {action_id!r}")
            seen_ids.add(action_id)
            where = f"action {action_id!r}"
            cost = action.get("cost", 1)
            if isinstance(cost, bool) or not isinstance(cost, int) or cost <= 0:
                raise GoapError(f"{where}: cost must be a positive integer")
            preconditions = action.get("preconditions", {})
            if not isinstance(preconditions, dict):
                raise GoapError(f"{where}: preconditions must be an object")
            effects = action.get("effects", {})
            if not isinstance(effects, dict):
                raise GoapError(f"{where}: effects must be an object")
            _validate_goap_state(preconditions, f"{where} preconditions")
            _validate_goap_state(effects, f"{where} effects")
            parsed_actions.append((action_id, cost, preconditions, effects))

        start = deepcopy(world)
        # Uniform-cost search; the priority (cost, position sequence) pops
        # the cheapest plan first and breaks cost ties lexicographically by
        # the actions' positions in the input list.
        heap: list[tuple[int, tuple[int, ...], int, tuple, dict]] = []
        counter = 0
        start_key = _freeze_state(start)
        best: dict[tuple, tuple[int, tuple[int, ...]]] = {start_key: (0, ())}
        heapq.heappush(heap, (0, (), counter, start_key, start))
        while heap:
            cost, positions, _, state_key, state = heapq.heappop(heap)
            if best.get(state_key) != (cost, positions):
                continue  # stale entry superseded by a better path
            if _goap_conditions_met(goal, state):
                return {
                    "status": "SUCCESS",
                    "plan": [parsed_actions[i][0] for i in positions],
                    "cost": cost,
                    "final_world": state,
                }
            for index, (_, action_cost, preconditions, effects) in enumerate(parsed_actions):
                if not _goap_conditions_met(preconditions, state):
                    continue
                next_state = dict(state)
                for key, value in effects.items():
                    next_state[key] = deepcopy(value)
                next_cost = cost + action_cost
                next_positions = positions + (index,)
                next_key = _freeze_state(next_state)
                known = best.get(next_key)
                if known is not None and known <= (next_cost, next_positions):
                    continue
                best[next_key] = (next_cost, next_positions)
                counter += 1
                heapq.heappush(heap, (next_cost, next_positions, counter, next_key, next_state))
        return {"status": "UNREACHABLE", "plan": [], "cost": None, "final_world": start}

    def find_path(self, request: Any) -> dict:
        """Find the minimum-cost orthogonal path from ``start`` to ``goal``.

        The request is validated in full before any search happens. The grid
        is a non-empty rectangular array whose cells are either ``None``
        (impassable) or a positive integer giving the cost of entering that
        cell; coordinates use integer ``x``/``y`` fields with the origin at
        the top-left corner and movement restricted to the four orthogonal
        neighbours. The start cell itself costs nothing. The returned route
        minimises total cost; ties are broken by converting each route to its
        full sequence of ``(y, x)`` coordinates and choosing the
        lexicographically smallest. Raises ValueError (NavigationError) on
        any invalid input; the request is never mutated and no grid or search
        state is retained between calls.
        """
        if not isinstance(request, dict):
            raise NavigationError("request must be a JSON object")
        if "grid" not in request:
            raise NavigationError("request is missing 'grid'")
        if "start" not in request:
            raise NavigationError("request is missing 'start'")
        if "goal" not in request:
            raise NavigationError("request is missing 'goal'")
        width, height = _validate_nav_grid(request["grid"])
        start = _validate_nav_coordinate(request["start"], "start")
        goal = _validate_nav_coordinate(request["goal"], "goal")
        grid = request["grid"]
        for x, y, where in ((start[0], start[1], "start"), (goal[0], goal[1], "goal")):
            if not (0 <= x < width and 0 <= y < height):
                raise NavigationError(f"{where} is outside the grid")
            if grid[y][x] is None:
                raise NavigationError(f"{where} must be on a passable cell")

        if start == goal:
            return {
                "status": "SUCCESS",
                "path": [{"x": start[0], "y": start[1]}],
                "cost": 0,
            }

        # Dijkstra over (cost, node sequence); the sequence key is a tuple of
        # (y, x) coordinates, so equal-cost routes tie-break lexicographically
        # by y then x as required.
        sx, sy = start
        gx, gy = goal
        best: dict[tuple[int, int], tuple[int, tuple]] = {(sx, sy): (0, ())}
        heap: list[tuple[int, tuple, int, int, int]] = [(0, (), 0, sx, sy)]
        counter = 0
        while heap:
            cost, sequence, _, x, y = heapq.heappop(heap)
            if best.get((x, y)) != (cost, sequence):
                continue  # stale entry superseded by a better route
            if (x, y) == (gx, gy):
                return {
                    "status": "SUCCESS",
                    "path": [{"x": px, "y": py} for py, px in ((sy, sx),) + sequence],
                    "cost": cost,
                }
            for dx, dy in _NAV_NEIGHBORS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                enter_cost = grid[ny][nx]
                if enter_cost is None:
                    continue
                next_cost = cost + enter_cost
                next_sequence = sequence + ((ny, nx),)
                known = best.get((nx, ny))
                if known is not None and known <= (next_cost, next_sequence):
                    continue
                best[(nx, ny)] = (next_cost, next_sequence)
                counter += 1
                heapq.heappush(heap, (next_cost, next_sequence, counter, nx, ny))
        return {"status": "UNREACHABLE", "path": [], "cost": None}

    def select_utility(self, request: Any) -> dict:
        """Score each enabled option and select the highest-scoring one.

        The request is validated in full before any scoring happens. Every
        option requires a unique non-empty string ``id``; ``enabled``
        (default ``True``), ``base`` (default ``1``, a non-negative finite
        number) and ``considerations`` (default ``[]``) may be omitted. Each
        consideration names a ``context`` key, a finite ``min``/``max`` range
        with ``min < max``, a ``curve`` of ``linear`` or ``inverse``, and an
        optional positive finite ``weight`` (default ``1``). The context value
        is normalised to ``(value - min) / (max - min)`` clamped to ``[0, 1]``;
        ``linear`` uses that value and ``inverse`` uses one minus it. An
        enabled option's score is its ``base`` times the weight-averaged
        responses (just ``base`` when it has no considerations). The highest
        score wins; ties go to the option appearing first in ``options``.
        Disabled options are validated structurally but never scored and
        never require their context keys to exist. When every option is
        disabled the result is ``NO_SELECTION`` with null ``selected`` and
        ``score`` — not an error. Raises ValueError (UtilityError) on any
        invalid input; the request is never mutated and no state is kept
        between calls.
        """
        if not isinstance(request, dict):
            raise UtilityError("request must be a JSON object")
        context = request.get("context")
        if not isinstance(context, dict):
            raise UtilityError("context must be an object")
        for key, value in context.items():
            if not _is_valid_key(key):
                raise UtilityError(f"context key {key!r} must be a non-empty string")
            if not _is_finite_number(value):
                raise UtilityError(f"context[{key!r}] must be a finite number")
        options = request.get("options")
        if not isinstance(options, list) or not options:
            raise UtilityError("options must be a non-empty list")

        parsed_options: list[tuple[str, bool, float, list]] = []
        seen_ids: set[str] = set()
        for index, option in enumerate(options):
            where = f"option at options[{index}]"
            if not isinstance(option, dict):
                raise UtilityError(f"{where} must be an object")
            option_id = option.get("id")
            if not _is_valid_key(option_id):
                raise UtilityError(f"{where} requires a non-empty string id")
            if option_id in seen_ids:
                raise UtilityError(f"duplicate option id {option_id!r}")
            seen_ids.add(option_id)
            where = f"option {option_id!r}"
            enabled = option.get("enabled", True)
            if not isinstance(enabled, bool):
                raise UtilityError(f"{where}: enabled must be a boolean")
            base = option.get("base", 1)
            if not _is_finite_number(base) or base < 0:
                raise UtilityError(f"{where}: base must be a non-negative finite number")
            considerations = option.get("considerations", [])
            if not isinstance(considerations, list):
                raise UtilityError(f"{where}: considerations must be a list")
            for c_index, consideration in enumerate(considerations):
                _validate_utility_consideration(
                    consideration, f"{where} consideration at considerations[{c_index}]"
                )
            if enabled:
                for consideration in considerations:
                    key = consideration["key"]
                    if key not in context:
                        raise UtilityError(f"{where}: context is missing key {key!r}")
            parsed_options.append((option_id, enabled, base, considerations))

        details: list[dict] = []
        selected: str | None = None
        selected_score: float | None = None
        for option_id, enabled, base, considerations in parsed_options:
            if not enabled:
                details.append(
                    {"id": option_id, "enabled": False, "score": None, "considerations": []}
                )
                continue
            responses: list[dict] = []
            weighted_sum = 0.0
            weight_total = 0.0
            for consideration in considerations:
                key = consideration["key"]
                value = context[key]
                low = consideration["min"]
                high = consideration["max"]
                normalised = (value - low) / (high - low)
                normalised = min(max(normalised, 0.0), 1.0)
                response = normalised if consideration["curve"] == "linear" else 1.0 - normalised
                weight = consideration.get("weight", 1)
                weighted_sum += response * weight
                weight_total += weight
                responses.append(
                    {"key": key, "value": value, "response": response, "weight": weight}
                )
            score = base * (weighted_sum / weight_total) if responses else base
            details.append(
                {"id": option_id, "enabled": True, "score": score, "considerations": responses}
            )
            if selected_score is None or score > selected_score:
                selected = option_id
                selected_score = score

        if selected is None:
            return {"status": "NO_SELECTION", "selected": None, "score": None, "options": details}
        return {"status": "SELECTED", "selected": selected, "score": selected_score, "options": details}

    def update_perception(self, request: Any) -> dict:
        """Merge one batch of observations into a snapshot of memory.

        The request is validated in full before anything is merged: ``now``
        is a non-negative finite number, ``retention`` a positive finite
        number, and ``memory``/``observations`` are lists of entities with
        unique non-empty string ``id``, non-empty string ``kind``,
        ``confidence`` in ``[0, 1]`` and a finite ``x``/``y`` position;
        memory entities additionally carry a non-negative ``last_seen`` no
        later than ``now``, and either may carry an optional JSON-object
        ``data`` (omitted means ``{}``). Re-observed entities have their
        kind/confidence/position/data replaced by the observation and their
        ``last_seen`` set to ``now`` while keeping their position in the
        memory order; new entities append in observation order. Unobserved
        records with ``now - last_seen >= retention`` are forgotten; the
        rest keep every field. Raises ValueError (PerceptionError) on any
        invalid input; the request is never mutated and no state is kept
        between calls.
        """
        if not isinstance(request, dict):
            raise PerceptionError("request must be a JSON object")
        if "now" not in request:
            raise PerceptionError("request is missing 'now'")
        now = request["now"]
        if not _is_finite_number(now) or now < 0:
            raise PerceptionError("now must be a non-negative finite number")
        if "retention" not in request:
            raise PerceptionError("request is missing 'retention'")
        retention = request["retention"]
        if not _is_finite_number(retention) or retention <= 0:
            raise PerceptionError("retention must be a positive finite number")
        memory = request.get("memory", [])
        if not isinstance(memory, list):
            raise PerceptionError("memory must be a list")
        observations = request.get("observations", [])
        if not isinstance(observations, list):
            raise PerceptionError("observations must be a list")

        records: list[dict] = []
        memory_ids: set[str] = set()
        for index, entity in enumerate(memory):
            parsed = _validate_perception_entity(
                entity, f"memory[{index}]", now, with_last_seen=True
            )
            if parsed["id"] in memory_ids:
                raise PerceptionError(f"duplicate memory id {parsed['id']!r}")
            memory_ids.add(parsed["id"])
            records.append(parsed)

        parsed_observations: list[dict] = []
        observation_ids: set[str] = set()
        for index, entity in enumerate(observations):
            parsed = _validate_perception_entity(
                entity, f"observations[{index}]", now, with_last_seen=False
            )
            if parsed["id"] in observation_ids:
                raise PerceptionError(f"duplicate observation id {parsed['id']!r}")
            observation_ids.add(parsed["id"])
            parsed_observations.append(parsed)

        by_id = {record["id"]: record for record in records}
        seen: list[str] = []
        for observation in parsed_observations:
            entity_id = observation["id"]
            seen.append(entity_id)
            if entity_id in by_id:
                record = by_id[entity_id]
                record["kind"] = observation["kind"]
                record["confidence"] = observation["confidence"]
                record["position"] = observation["position"]
                record["data"] = observation["data"]
                record["last_seen"] = now
            else:
                record = {
                    "id": entity_id,
                    "kind": observation["kind"],
                    "last_seen": now,
                    "confidence": observation["confidence"],
                    "position": observation["position"],
                    "data": observation["data"],
                }
                by_id[entity_id] = record
                records.append(record)

        updated: list[dict] = []
        forgotten: list[str] = []
        for record in records:
            entity_id = record["id"]
            if entity_id not in observation_ids and now - record["last_seen"] >= retention:
                forgotten.append(entity_id)
                continue
            updated.append(
                {
                    "id": record["id"],
                    "kind": record["kind"],
                    "last_seen": record["last_seen"],
                    "confidence": record["confidence"],
                    "position": dict(record["position"]),
                    "data": deepcopy(record["data"]),
                }
            )

        return {"status": "UPDATED", "memory": updated, "seen": seen, "forgotten": forgotten}
