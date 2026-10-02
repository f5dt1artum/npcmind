"""Core service surface for NpcMind.

Provides process health reporting, a stateless behavior-tree evaluator, a
single-step finite-state-machine driver, one-shot goal-oriented action
planning, and a stateless two-dimensional grid path search. Each
``evaluate_behavior`` call executes exactly one tick of the supplied tree
against the supplied blackboard; each ``step_state_machine`` call advances a
machine by exactly one event; each ``plan_goap`` call searches for a
minimum-cost action sequence from the supplied world state to the supplied
goal; each ``find_path`` call searches a supplied grid for a minimum-cost
four-neighbour route. No state is kept between calls.
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
    """Raised when a navigation request fails structural validation."""


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


def _validate_nav_coordinate(coord: Any, where: str) -> tuple[int, int]:
    """Validate a grid coordinate object; return its (x, y) tuple."""
    if not isinstance(coord, dict):
        raise NavigationError(f"{where} must be an object")
    if "x" not in coord or "y" not in coord:
        raise NavigationError(f"{where} requires integer x and y")
    x = coord["x"]
    y = coord["y"]
    if isinstance(x, bool) or not isinstance(x, int) or isinstance(y, bool) or not isinstance(y, int):
        raise NavigationError(f"{where} requires integer x and y")
    return x, y


class Service:
    """Health, behavior-tree evaluation, FSM stepping, GOAP, navigation."""
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
        """Find a minimum-cost four-neighbour route through ``grid``.

        The request is validated in full before any search happens. The grid
        must be a non-empty rectangular list whose cells are either ``None``
        (impassable) or a positive integer giving the cost of entering that
        cell; ``start`` and ``goal`` must be objects with integer ``x`` and
        ``y`` fields inside the grid on passable cells. The returned route
        minimises total cost (the start cell costs nothing); ties are broken
        by comparing the full (y, x) coordinate sequences lexicographically
        and choosing the smallest. A start equal to its goal yields the
        single-point path at zero cost, and an unreachable goal is not an
        error. Raises ValueError (NavigationError) on any invalid input; the
        request is never mutated and no map or search state is retained.
        """
        if not isinstance(request, dict):
            raise NavigationError("request must be a JSON object")
        if "grid" not in request or "start" not in request or "goal" not in request:
            raise NavigationError("request requires grid, start and goal")
        grid = request["grid"]
        if not isinstance(grid, list) or not grid:
            raise NavigationError("grid must be a non-empty two-dimensional array")
        width = len(grid[0]) if isinstance(grid[0], list) else -1
        if width <= 0:
            raise NavigationError("grid must be a non-empty two-dimensional array")
        for row in grid:
            if not isinstance(row, list) or len(row) != width:
                raise NavigationError("grid must be a non-empty rectangular array")
            for cell in row:
                if cell is not None and (isinstance(cell, bool) or not isinstance(cell, int) or cell <= 0):
                    raise NavigationError("grid cells must be null or positive integers")

        start_x, start_y = _validate_nav_coordinate(request["start"], "start")
        goal_x, goal_y = _validate_nav_coordinate(request["goal"], "goal")
        height = len(grid)
        for name, x, y in (("start", start_x, start_y), ("goal", goal_x, goal_y)):
            if not (0 <= x < width and 0 <= y < height):
                raise NavigationError(f"{name} is outside the grid")
            if grid[y][x] is None:
                raise NavigationError(f"{name} must be on a passable cell")

        if start_x == goal_x and start_y == goal_y:
            return {"status": "SUCCESS", "path": [{"x": start_x, "y": start_y}], "cost": 0}

        # Dijkstra with the full (y, x) cell sequence carried in the heap
        # priority: equal-cost routes are resolved lexicographically by
        # their coordinate sequences, so the first time the goal is popped it
        # carries the uniquely-determined minimum route.
        start_seq = ((start_y, start_x),)
        distances: dict[tuple[int, int], int] = {(start_y, start_x): 0}
        heap: list[tuple[int, tuple[tuple[int, int], ...], int]] = []
        counter = 0
        heapq.heappush(heap, (0, start_seq, counter))
        while heap:
            cost, seq, _ = heapq.heappop(heap)
            y, x = seq[-1]
            if distances.get((y, x)) != cost:
                continue  # stale entry superseded by a better path
            if (y, x) == (goal_y, goal_x):
                return {
                    "status": "SUCCESS",
                    "path": [{"x": cell_x, "y": cell_y} for cell_y, cell_x in seq],
                    "cost": cost,
                }
            for next_y, next_x in ((y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)):
                if not (0 <= next_y < height and 0 <= next_x < width):
                    continue
                cell_cost = grid[next_y][next_x]
                if cell_cost is None:
                    continue
                next_cost = cost + cell_cost
                known = distances.get((next_y, next_x))
                next_seq = seq + ((next_y, next_x),)
                if known is not None and known <= next_cost:
                    continue
                distances[(next_y, next_x)] = next_cost
                counter += 1
                heapq.heappush(heap, (next_cost, next_seq, counter))
        return {"status": "UNREACHABLE", "path": [], "cost": None}
