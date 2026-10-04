"""Core service surface for NpcMind.

Provides process health reporting, a stateless behavior-tree evaluator, a
stateless behavior-tree graph projection, a single-step finite-state-machine
driver, one-shot goal-oriented action
planning, stateless 2D grid navigation, stateless utility scoring, and a
one-shot perception-memory merge. Each ``evaluate_behavior`` call executes
exactly one tick of the supplied tree against the supplied blackboard; each
``step_state_machine`` call advances a machine by exactly one event; each
``plan_goap`` call searches for a minimum-cost action sequence from the
supplied world state to the supplied goal; each ``find_path`` call searches
a one-shot grid request for a minimum-cost orthogonal route; each
``select_utility`` call scores the supplied options against the supplied
context once and picks the best; each ``update_perception`` call validates
the supplied memory and observations once and returns the merged memory;
each ``select_avoidance`` call predicts disc collisions for the supplied
candidate velocities over one time horizon and picks one admissible
velocity; each ``select_attention`` call scores the supplied memory
entities against the supplied observer once and picks the most salient;
each ``match_dialogue_intent`` call matches the supplied utterance
against the supplied rule-based intent patterns once; each
``select_schedule_activity`` call evaluates the supplied activities
against the supplied minute and need levels once and picks at most one;
each ``assign_team_roles`` call validates the supplied roles and agents
once and computes one total-score-maximising role assignment; each
``adjust_difficulty`` call scores the supplied performance signals once
and returns one difficulty recommendation.
No state is kept between calls.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import re
from copy import deepcopy
from fractions import Fraction
from typing import Any

from . import __version__

STATUSES = ("SUCCESS", "FAILURE", "RUNNING")
_COMPOSITE_TYPES = ("sequence", "selector")
# Node types that carry a ``children`` array (used by the graph projection).
_BRANCH_TYPES = ("sequence", "selector", "random_selector")
_CONDITION_OPS = ("exists", "equals", "not_equals")
_ACTION_OPS = ("set", "delete", "status")
_MAX_SEED = 18446744073709551615  # 2**64 - 1


class TreeError(ValueError):
    """Raised when a behavior-tree request fails structural validation."""


class BehaviorVisualizationError(ValueError):
    """Raised when a behavior-tree visualization request fails validation."""


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


class SteeringError(ValueError):
    """Raised when a local-avoidance steering request fails validation."""


class AttentionError(ValueError):
    """Raised when an attention-selection request fails validation."""


class DialogueError(ValueError):
    """Raised when a dialogue-intent request fails validation."""


class ScheduleError(ValueError):
    """Raised when a schedule-decision request fails validation."""


class TeamAssignmentError(ValueError):
    """Raised when a team-role-assignment request fails validation."""


class DifficultyError(ValueError):
    """Raised when a difficulty-adjustment request fails validation."""


_SLOT_TOKEN_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _minute_in_window(now: int, start: int, end: int) -> bool:
    """Whether ``now`` lies in the start-inclusive, end-exclusive window.

    A start before the end is a same-day interval, a start after the end
    wraps past midnight, and equal endpoints cover the whole day.
    """
    if start == end:
        return True
    if start < end:
        return start <= now < end
    return now >= start or now < end


def _parse_dialogue_pattern(pattern: Any, where: str) -> list[tuple[str, str]]:
    """Validate one pattern string; return its ``(kind, value)`` tokens.

    Literal tokens are stored casefolded; slot tokens store their name.
    """
    if not isinstance(pattern, str):
        raise DialogueError(f"{where} must be a string")
    tokens = pattern.split()
    if not tokens:
        raise DialogueError(f"{where} must not be empty")
    parsed: list[tuple[str, str]] = []
    for token in tokens:
        if "{" in token or "}" in token:
            match = _SLOT_TOKEN_RE.fullmatch(token)
            if match is None:
                raise DialogueError(f"{where}: invalid slot token {token!r}")
            parsed.append(("slot", match.group(1)))
        else:
            parsed.append(("literal", token.casefold()))
    return parsed


def _match_dialogue_pattern(parsed: list[tuple[str, str]], tokens: list[str]) -> dict | None:
    """Match a parsed pattern against the utterance tokens.

    The pattern must cover the whole utterance. Returns the slot mapping
    (original utterance text) on a match, ``None`` otherwise.
    """
    if len(parsed) != len(tokens):
        return None
    slots: dict[str, str] = {}
    normalised: dict[str, str] = {}
    for (kind, value), token in zip(parsed, tokens):
        folded = token.casefold()
        if kind == "literal":
            if value != folded:
                return None
        elif value in normalised:
            if normalised[value] != folded:
                return None
        else:
            normalised[value] = folded
            slots[value] = token
    return slots


_UTILITY_CURVES = ("linear", "inverse")


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False  # integer too large to represent as a finite double


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


def _validate_node(node: Any, path: str, seen_ids: set[str]) -> bool:
    """Validate one node subtree; return whether it contains a random_selector."""
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
    has_random_selector = node_type == "random_selector"
    if node_type in _COMPOSITE_TYPES:
        children = node.get("children", [])
        if not isinstance(children, list):
            raise TreeError(f"{where}: children must be a list")
        for index, child in enumerate(children):
            if _validate_node(child, f"{path}.children[{index}]", seen_ids):
                has_random_selector = True
    elif node_type == "random_selector":
        children = node.get("children")
        if not isinstance(children, list) or not children:
            raise TreeError(f"{where}: children must be a non-empty list")
        if "weights" in node:
            weights = node["weights"]
            if not isinstance(weights, list) or len(weights) != len(children):
                raise TreeError(
                    f"{where}: weights must be a list of positive integers "
                    "matching the children count"
                )
            for weight in weights:
                if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
                    raise TreeError(f"{where}: weights must be positive integers")
        for index, child in enumerate(children):
            if _validate_node(child, f"{path}.children[{index}]", seen_ids):
                has_random_selector = True
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
    return has_random_selector


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


def _select_weighted_child(node: dict, seed: int) -> dict:
    """Pick one direct child of a random_selector, deterministically.

    The digest input is the decimal text of ``seed``, a half-width colon and
    the node id, hashed as UTF-8 with SHA-256; the first eight digest bytes
    read as an unsigned big-endian integer are reduced modulo the total
    weight and the remainder selects the first cumulative weight interval
    it falls into, in children order.
    """
    children = node["children"]
    weights = node.get("weights")
    if weights is None:
        weights = [1] * len(children)
    total = sum(weights)
    digest = hashlib.sha256(f"{seed}:{node['id']}".encode("utf-8")).digest()
    remainder = int.from_bytes(digest[:8], "big") % total
    cumulative = 0
    for child, weight in zip(children, weights):
        cumulative += weight
        if remainder < cumulative:
            return child
    return children[-1]  # unreachable: remainder < total


def _tick(node: dict, blackboard: dict, trace: list, seed: int | None = None) -> str:
    node_type = node["type"]
    if node_type == "sequence":
        status = "SUCCESS"
        for child in node.get("children", []):
            status = _tick(child, blackboard, trace, seed)
            if status != "SUCCESS":
                break
    elif node_type == "selector":
        status = "FAILURE"
        for child in node.get("children", []):
            status = _tick(child, blackboard, trace, seed)
            if status != "FAILURE":
                break
    elif node_type == "random_selector":
        # Exactly one direct child runs; the other branches are never
        # executed and produce no trace entries.
        status = _tick(_select_weighted_child(node, seed), blackboard, trace, seed)
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


def _expand_to_leaf(state_id: str, initials: dict[str, str]) -> str:
    """Follow ``initial`` links from ``state_id`` down to a leaf state."""
    seen = {state_id}
    current = state_id
    while current in initials:
        current = initials[current]
        if current in seen:
            raise MachineError(
                f"state {state_id!r}: initial expansion does not reach a leaf state"
            )
        seen.add(current)
    return current


def _validate_machine(machine: Any) -> tuple[set[str], dict[str, str], dict[str, str]]:
    """Validate a machine definition fully.

    Returns ``(state_ids, parents, initials)``: the declared state ids, the
    child-to-parent map, and the composite-state-to-initial-child map. A
    state given as a plain string or an object with only an ``id`` is a
    leaf; an object may also name a ``parent`` (its direct parent state)
    and a composite state must name an ``initial`` direct child.
    """
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

    parents: dict[str, str] = {}
    declared_initials: dict[str, str] = {}
    for state in states:
        if not isinstance(state, dict):
            continue
        state_id = state["id"]
        if "parent" in state:
            parent = state["parent"]
            if not isinstance(parent, str) or parent not in state_ids:
                raise MachineError(
                    f"state {state_id!r}: parent references unknown state {parent!r}"
                )
            if parent == state_id:
                raise MachineError(f"state {state_id!r} must not be its own parent")
            parents[state_id] = parent
        if "initial" in state:
            child = state["initial"]
            if not isinstance(child, str) or child not in state_ids:
                raise MachineError(
                    f"state {state_id!r}: initial references unknown state {child!r}"
                )
            declared_initials[state_id] = child

    # Each state has at most one parent (a single ``parent`` field) and the
    # parent relation must be acyclic: walking upward from any state must
    # terminate at a root.
    for state_id in state_ids:
        seen = {state_id}
        node = state_id
        while node in parents:
            node = parents[node]
            if node in seen:
                raise MachineError(f"state {state_id!r}: parent relation forms a cycle")
            seen.add(node)

    children: dict[str, list[str]] = {state_id: [] for state_id in state_ids}
    for child_id, parent_id in parents.items():
        children[parent_id].append(child_id)

    initials: dict[str, str] = {}
    for state_id in state_ids:
        if children[state_id]:
            child = declared_initials.get(state_id)
            if child is None:
                raise MachineError(
                    f"composite state {state_id!r} requires an initial child state"
                )
            if parents.get(child) != state_id:
                raise MachineError(
                    f"state {state_id!r}: initial {child!r} is not a direct child"
                )
            initials[state_id] = child
        elif state_id in declared_initials:
            raise MachineError(f"leaf state {state_id!r} must not declare an initial state")

    initial = machine.get("initial")
    if not isinstance(initial, str) or initial not in state_ids:
        raise MachineError(f"initial references unknown state {initial!r}")
    if initial in parents:
        raise MachineError(f"initial must reference a root state, but {initial!r} has a parent")

    # Following initial links from any composite state must reach a leaf.
    for state_id in state_ids:
        _expand_to_leaf(state_id, initials)

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
    return state_ids, parents, initials


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


def _validate_perception_entity(entity: Any, where: str, now: float, *, last_seen_required: bool) -> dict:
    """Validate one memory entity or observation; return a normalised copy."""
    if not isinstance(entity, dict):
        raise PerceptionError(f"{where} must be an object")
    entity_id = entity.get("id")
    if not _is_valid_key(entity_id):
        raise PerceptionError(f"{where} requires a non-empty string id")
    kind = entity.get("kind")
    if not _is_valid_key(kind):
        raise PerceptionError(f"{where} requires a non-empty string kind")
    confidence = entity.get("confidence")
    if not _is_finite_number(confidence) or not 0 <= confidence <= 1:
        raise PerceptionError(f"{where}: confidence must be a finite number between 0 and 1")
    position = entity.get("position")
    if not isinstance(position, dict):
        raise PerceptionError(f"{where}: position must be an object")
    if "x" not in position or not _is_finite_number(position["x"]):
        raise PerceptionError(f"{where}: position.x must be a finite number")
    if "y" not in position or not _is_finite_number(position["y"]):
        raise PerceptionError(f"{where}: position.y must be a finite number")
    if "data" in entity:
        data = entity["data"]
        if not isinstance(data, dict):
            raise PerceptionError(f"{where}: data must be an object")
        _validate_perception_json(data, f"{where}.data")
    else:
        data = {}
    parsed = {
        "id": entity_id,
        "kind": kind,
        "confidence": confidence,
        "position": {"x": position["x"], "y": position["y"]},
        "data": deepcopy(data),
    }
    if last_seen_required:
        last_seen = entity.get("last_seen")
        if not _is_finite_number(last_seen) or last_seen < 0:
            raise PerceptionError(f"{where}: last_seen must be a non-negative finite number")
        if last_seen > now:
            raise PerceptionError(f"{where}: last_seen must not be later than now")
        parsed["last_seen"] = last_seen
    return parsed


def _validate_vector(value: Any, where: str) -> tuple[float, float]:
    """Validate ``{"x": finite, "y": finite}``; booleans are not numbers."""
    if not isinstance(value, dict):
        raise SteeringError(f"{where} must be an object")
    if "x" not in value or "y" not in value:
        raise SteeringError(f"{where} requires finite-number fields x and y")
    x = value["x"]
    y = value["y"]
    if not _is_finite_number(x):
        raise SteeringError(f"{where}.x must be a finite number")
    if not _is_finite_number(y):
        raise SteeringError(f"{where}.y must be a finite number")
    return float(x), float(y)


def _validate_attention_vector(value: Any, where: str) -> tuple[float, float]:
    """Validate ``{"x": finite, "y": finite}`` for an attention request."""
    if not isinstance(value, dict):
        raise AttentionError(f"{where} must be an object")
    if "x" not in value or "y" not in value:
        raise AttentionError(f"{where} requires finite-number fields x and y")
    x = value["x"]
    y = value["y"]
    if not _is_finite_number(x):
        raise AttentionError(f"{where}.x must be a finite number")
    if not _is_finite_number(y):
        raise AttentionError(f"{where}.y must be a finite number")
    return float(x), float(y)


def _disc_collides(
    rel_pos: tuple[float, float],
    rel_vel: tuple[float, float],
    radius_sum: float,
    horizon: float,
) -> bool:
    """Whether two discs touch within ``[0, horizon]`` under linear motion.

    The other disc moves at ``rel_vel`` relative to the agent; ``rel_pos`` is
    the other centre relative to the agent at ``t = 0``. Squared centre
    distance along the trajectory is a convex quadratic in ``t``, so over the
    closed interval its minimum is at the closest-approach time clamped to
    ``[0, horizon]``; touching (distance equal to the radius sum), including at
    either endpoint, counts.
    """
    px, py = rel_pos
    vx, vy = rel_vel
    speed_sq = vx * vx + vy * vy
    if speed_sq == 0.0:
        return px * px + py * py <= radius_sum * radius_sum
    closest_time = -(px * vx + py * vy) / speed_sq
    if closest_time > horizon:
        closest_time = horizon
    if closest_time > 0.0:
        px += vx * closest_time
        py += vy * closest_time
    return px * px + py * py <= radius_sum * radius_sum


class Service:
    """Health, behavior trees (evaluate and export), FSM, GOAP, navigation, utility, perception, steering, attention, dialogue, schedules, teams, difficulty."""
    name = "npcmind"
    version = __version__

    def health(self) -> dict[str, str]:
        return {"status": "ok", "service": self.name, "version": self.version}

    def evaluate_behavior(self, request: dict) -> dict:
        """Execute one tick of ``request['tree']`` against the blackboard.

        When the tree contains a ``random_selector`` node the request must
        also carry a ``seed``: an integer (booleans are not integers) between
        0 and 18446744073709551615 inclusive. The same tree, blackboard and
        seed always produce the same status, blackboard and trace; a ``seed``
        supplied with a tree without random selectors is ignored. Raises
        ValueError (TreeError) when the request structure, the tree, the
        blackboard or the seed is invalid; no partial result is produced.
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

        has_random_selector = _validate_node(tree, "root", set())

        seed = None
        if has_random_selector:
            if "seed" not in request:
                raise TreeError("request is missing 'seed' required by random_selector")
            seed = request["seed"]
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TreeError("seed must be an integer")
            if not 0 <= seed <= _MAX_SEED:
                raise TreeError(f"seed must be between 0 and {_MAX_SEED}")

        board = dict(blackboard)
        trace: list = []
        status = _tick(tree, board, trace, seed)
        return {"status": status, "blackboard": board, "trace": trace}

    def export_behavior_tree(self, request: Any) -> dict:
        """Project ``request['tree']`` and an optional trace to a flat graph.

        Only structure and already-computed trace entries are reorganised;
        no node is executed and no blackboard is touched. ``trace`` defaults
        to an empty array when omitted and must be the trace array returned
        by :meth:`evaluate_behavior`: each entry is an object with a
        non-empty string ``id`` that exists in the tree, a ``type`` matching
        that node, a ``status`` of ``SUCCESS``, ``FAILURE`` or ``RUNNING``,
        and no id may occur twice.

        On success returns ``status`` ``EXPORTED``, ``nodes`` in
        depth-first pre-order (root first, children in declaration order)
        each with ``id``, ``type``, ``depth``, ``visited`` and ``status``
        (the trace value for visited nodes, ``null`` for the rest),
        ``edges`` ordered by the parent's pre-order position and within a
        parent by child order, each with ``from``, ``to`` and a zero-based
        ``index``, and ``trace_order`` listing the trace ids in input order.
        Raises ValueError (BehaviorVisualizationError) when the request,
        tree or trace is invalid; no partial graph is produced. The request
        and its nested values are never mutated and no state is kept.
        """
        if not isinstance(request, dict):
            raise BehaviorVisualizationError("request must be a JSON object")
        if "tree" not in request:
            raise BehaviorVisualizationError("request is missing 'tree'")
        tree = request["tree"]
        trace = request.get("trace", [])
        if not isinstance(trace, list):
            raise BehaviorVisualizationError("trace must be an array")

        seen_ids: set[str] = set()
        _validate_node(tree, "root", seen_ids)

        # Depth-first pre-order walk; nodes carry their depth and keep a
        # reference to the raw node so edges can be emitted grouped by the
        # parent's pre-order position.
        nodes: list[dict] = []
        node_types: dict[str, str] = {}
        node_objects: dict[str, dict] = {}

        def walk(node: dict, depth: int) -> None:
            node_id = node["id"]
            node_type = node["type"]
            node_types[node_id] = node_type
            node_objects[node_id] = node
            nodes.append(
                {"id": node_id, "type": node_type, "depth": depth, "visited": False, "status": None}
            )
            if node_type in _BRANCH_TYPES:
                for child in node.get("children", []):
                    walk(child, depth + 1)

        walk(tree, 0)

        # Edges follow the parent's pre-order position; within one parent
        # the declared children order is kept, and the global index runs
        # continuously from zero. Leaves contribute nothing.
        edges: list[dict] = []
        edge_index = 0
        for entry in nodes:
            node = node_objects[entry["id"]]
            if node["type"] not in _BRANCH_TYPES:
                continue
            for child in node.get("children", []):
                edges.append({"from": node["id"], "to": child["id"], "index": edge_index})
                edge_index += 1

        nodes_by_id = {entry["id"]: entry for entry in nodes}
        trace_order: list[str] = []
        trace_ids: set[str] = set()
        for position, entry in enumerate(trace):
            where = f"trace[{position}]"
            if not isinstance(entry, dict):
                raise BehaviorVisualizationError(f"{where} must be an object")
            entry_id = entry.get("id")
            if not _is_valid_key(entry_id):
                raise BehaviorVisualizationError(f"{where} requires a non-empty string id")
            if entry_id in trace_ids:
                raise BehaviorVisualizationError(f"{where}: duplicate trace id {entry_id!r}")
            if entry_id not in node_types:
                raise BehaviorVisualizationError(
                    f"{where}: id {entry_id!r} does not exist in the tree"
                )
            entry_type = entry.get("type")
            if entry_type != node_types[entry_id]:
                raise BehaviorVisualizationError(
                    f"{where}: type {entry_type!r} does not match tree node {entry_id!r}"
                )
            entry_status = entry.get("status")
            if entry_status not in STATUSES:
                raise BehaviorVisualizationError(f"{where}: illegal status {entry_status!r}")
            trace_ids.add(entry_id)
            trace_order.append(entry_id)
            nodes_by_id[entry_id]["visited"] = True
            nodes_by_id[entry_id]["status"] = entry_status

        return {"status": "EXPORTED", "nodes": nodes, "edges": edges, "trace_order": trace_order}

    def step_state_machine(self, request: dict) -> dict:
        """Advance ``request['machine']`` by exactly one ``request['event']``.

        The machine and the request are validated in full before any action
        runs. States may form a hierarchy: a state object may name a
        ``parent`` state, and a composite state (one with children) must
        name an ``initial`` direct child. ``machine.initial`` must be a root
        state; an explicit ``current_state`` may name any state. A composite
        start or target state is expanded along ``initial`` links to its
        unique leaf. Candidate transitions are sought from the current leaf
        upward towards the root: a deeper source level is considered as a
        whole before any parent level, and within one level transitions are
        checked in declaration order, the first whose ``event`` matches
        exactly and whose condition holds being taken; its actions then run
        in order, exactly once. The expansion itself produces no actions and
        no trace entries. Raises ValueError (MachineError) on any structural
        problem; a step that matches no transition is not an error.
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

        state_ids, parents, initials = _validate_machine(machine)

        current = request.get("current_state", machine["initial"])
        if not isinstance(current, str) or current not in state_ids:
            raise MachineError(f"current_state references unknown state {current!r}")

        # A composite start state expands along initial links to its leaf.
        start_leaf = _expand_to_leaf(current, initials)

        # Source states are tried from the current leaf up to the root; a
        # deeper level is exhausted before any ancestor level is considered.
        chain = [start_leaf]
        node = start_leaf
        while node in parents:
            node = parents[node]
            chain.append(node)

        board = dict(blackboard)
        trace: list = []
        matched = None
        for source in chain:
            for transition in machine["transitions"]:
                if transition["from"] != source or transition["event"] != event:
                    continue
                holds = _sm_condition_holds(transition.get("condition"), board)
                trace.append({"id": transition["id"], "condition": holds})
                if holds:
                    matched = transition
                    break
            if matched is not None:
                break

        if matched is None:
            return {
                "previous_state": start_leaf,
                "state": start_leaf,
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
            "previous_state": start_leaf,
            "state": _expand_to_leaf(matched["to"], initials),
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
        """Merge one batch of observations into a one-shot perception memory.

        The request is validated in full before any merge happens. ``now`` is
        a non-negative finite number and ``retention`` a positive finite
        number; ``memory`` and ``observations`` are lists of entities. Each
        memory entity needs a unique non-empty string ``id``, a non-empty
        string ``kind``, a ``last_seen`` non-negative finite number no later
        than ``now``, a ``confidence`` in ``[0, 1]``, a ``position`` of
        finite ``x``/``y`` numbers, and an optional JSON object ``data``
        (default ``{}``); observations are identical apart from lacking
        ``last_seen``. Ids must be unique within each list; an observation
        sharing an id with a memory entity re-observes it.

        A re-observed entity keeps its slot in the memory order but has its
        ``kind``, ``confidence``, ``position`` and ``data`` replaced and
        ``last_seen`` set to ``now``. Entities seen for the first time are
        appended in observation order. Unobserved old entities whose age
        ``now - last_seen`` is at least ``retention`` are forgotten; younger
        ones pass through untouched. Returns ``status`` ``UPDATED``, the new
        ``memory`` (each record carrying an explicit ``data``), the observed
        ids in observation order as ``seen``, and the forgotten ids in old
        memory order as ``forgotten``. Raises ValueError (PerceptionError) on
        any invalid input; the request is never mutated and no state is kept
        between calls.
        """
        if not isinstance(request, dict):
            raise PerceptionError("request must be a JSON object")
        now = request.get("now")
        if not _is_finite_number(now) or now < 0:
            raise PerceptionError("now must be a non-negative finite number")
        retention = request.get("retention")
        if not _is_finite_number(retention) or retention <= 0:
            raise PerceptionError("retention must be a positive finite number")
        memory = request.get("memory")
        if not isinstance(memory, list):
            raise PerceptionError("memory must be a list")
        observations = request.get("observations")
        if not isinstance(observations, list):
            raise PerceptionError("observations must be a list")

        parsed_memory: list[dict] = []
        memory_ids: set[str] = set()
        for index, entity in enumerate(memory):
            parsed = _validate_perception_entity(entity, f"memory[{index}]", now, last_seen_required=True)
            if parsed["id"] in memory_ids:
                raise PerceptionError(f"duplicate memory id {parsed['id']!r}")
            memory_ids.add(parsed["id"])
            parsed_memory.append(parsed)

        parsed_observations: list[dict] = []
        observation_ids: set[str] = set()
        for index, entity in enumerate(observations):
            parsed = _validate_perception_entity(
                entity, f"observations[{index}]", now, last_seen_required=False
            )
            if parsed["id"] in observation_ids:
                raise PerceptionError(f"duplicate observation id {parsed['id']!r}")
            observation_ids.add(parsed["id"])
            parsed_observations.append(parsed)

        observations_by_id = {entity["id"]: entity for entity in parsed_observations}

        merged: list[dict] = []
        forgotten: list[str] = []
        for entity in parsed_memory:
            entity_id = entity["id"]
            observed = observations_by_id.get(entity_id)
            if observed is not None:
                merged.append(
                    {
                        "id": entity_id,
                        "kind": deepcopy(observed["kind"]),
                        "last_seen": now,
                        "confidence": deepcopy(observed["confidence"]),
                        "position": deepcopy(observed["position"]),
                        "data": deepcopy(observed["data"]),
                    }
                )
            elif now - entity["last_seen"] >= retention:
                forgotten.append(entity_id)
            else:
                merged.append(deepcopy(entity))

        for entity in parsed_observations:
            if entity["id"] not in memory_ids:
                merged.append(
                    {
                        "id": entity["id"],
                        "kind": deepcopy(entity["kind"]),
                        "last_seen": now,
                        "confidence": deepcopy(entity["confidence"]),
                        "position": deepcopy(entity["position"]),
                        "data": deepcopy(entity["data"]),
                    }
                )

        return {
            "status": "UPDATED",
            "memory": merged,
            "seen": [entity["id"] for entity in parsed_observations],
            "forgotten": forgotten,
        }

    def select_avoidance(self, request: Any) -> dict:
        """Pick one admissible candidate velocity for local obstacle avoidance.

        The request is validated in full before any candidate is evaluated.
        The agent has a finite ``position`` vector, a positive finite
        ``radius``, a non-negative finite ``max_speed``, a finite
        ``desired_velocity`` vector, a positive finite ``time_horizon``, and a
        non-empty ``candidates`` list; each candidate has a unique non-empty
        string ``id`` and a finite ``velocity`` vector. ``neighbors`` are
        moving discs (``position``, positive finite ``radius``, finite
        ``velocity``) and ``obstacles`` static discs (``position``, positive
        finite ``radius``); their ids are unique across the two lists.

        Every candidate is advanced as a constant straight-line velocity; a
        collision is predicted when the agent centre and an object centre come
        within the radius sum at any time in the closed interval
        ``[0, time_horizon]``. A candidate is admissible only when its speed
        does not exceed ``max_speed`` and it predicts no collision. Among the
        admissible candidates the one closest to ``desired_velocity`` wins,
        ties going to the candidate listed first; with none admissible the
        result is ``BLOCKED`` with null ``selected`` and a zero ``velocity``.
        ``evaluations`` always lists every candidate in input order, each with
        ``speed_ok``, ``collision_ids`` (neighbours first, then obstacles) and
        ``admissible``. Raises ValueError (SteeringError) on any invalid
        input; the request is never mutated and no state is kept between calls.
        """
        if not isinstance(request, dict):
            raise SteeringError("request must be a JSON object")

        for field in (
            "position",
            "radius",
            "max_speed",
            "desired_velocity",
            "time_horizon",
            "candidates",
        ):
            if field not in request:
                raise SteeringError(f"request is missing {field!r}")

        agent_pos = _validate_vector(request["position"], "position")
        radius = request["radius"]
        if not _is_finite_number(radius) or radius <= 0:
            raise SteeringError("radius must be a positive finite number")
        max_speed = request["max_speed"]
        if not _is_finite_number(max_speed) or max_speed < 0:
            raise SteeringError("max_speed must be a non-negative finite number")
        desired_velocity = _validate_vector(request["desired_velocity"], "desired_velocity")
        horizon = request["time_horizon"]
        if not _is_finite_number(horizon) or horizon <= 0:
            raise SteeringError("time_horizon must be a positive finite number")
        candidates = request["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise SteeringError("candidates must be a non-empty list")

        neighbors_in = request.get("neighbors", [])
        if not isinstance(neighbors_in, list):
            raise SteeringError("neighbors must be a list")
        obstacles_in = request.get("obstacles", [])
        if not isinstance(obstacles_in, list):
            raise SteeringError("obstacles must be a list")

        parsed_candidates: list[tuple[str, tuple[float, float], Any, Any]] = []
        candidate_ids: set[str] = set()
        for index, candidate in enumerate(candidates):
            where = f"candidate at candidates[{index}]"
            if not isinstance(candidate, dict):
                raise SteeringError(f"{where} must be an object")
            candidate_id = candidate.get("id")
            if not _is_valid_key(candidate_id):
                raise SteeringError(f"{where} requires a non-empty string id")
            if candidate_id in candidate_ids:
                raise SteeringError(f"duplicate candidate id {candidate_id!r}")
            candidate_ids.add(candidate_id)
            if "velocity" not in candidate:
                raise SteeringError(f"{where} is missing 'velocity'")
            raw_velocity = candidate["velocity"]
            velocity = _validate_vector(raw_velocity, f"{where} velocity")
            # Keep the original numeric values so the selected velocity is
            # echoed verbatim (integers stay integers).
            parsed_candidates.append(
                (candidate_id, velocity, raw_velocity["x"], raw_velocity["y"])
            )

        # (id, relative position, relative velocity, radius sum); objects keep
        # their list order, neighbours ahead of obstacles, so collision ids
        # naturally emit in the required ordering.
        objects: list[tuple[str, tuple[float, float], tuple[float, float], float]] = []
        object_ids: set[str] = set()

        def parse_object(item: Any, index: int, kind: str, moving: bool) -> None:
            where = f"{kind[:-1]} at {kind}[{index}]"
            if not isinstance(item, dict):
                raise SteeringError(f"{where} must be an object")
            object_id = item.get("id")
            if not _is_valid_key(object_id):
                raise SteeringError(f"{where} requires a non-empty string id")
            if object_id in object_ids:
                raise SteeringError(f"duplicate {kind[:-1]} id {object_id!r}")
            object_ids.add(object_id)
            pos = _validate_vector(item.get("position"), f"{where} position")
            obj_radius = item.get("radius")
            if not _is_finite_number(obj_radius) or obj_radius <= 0:
                raise SteeringError(f"{where} radius must be a positive finite number")
            if moving:
                if "velocity" not in item:
                    raise SteeringError(f"{where} is missing 'velocity'")
                vel = _validate_vector(item["velocity"], f"{where} velocity")
            else:
                # Obstacles are static; any extra fields (including a stray
                # velocity) are ignored rather than treated as motion.
                vel = (0.0, 0.0)
            rel_pos = (pos[0] - agent_pos[0], pos[1] - agent_pos[1])
            objects.append((object_id, rel_pos, vel, float(radius) + float(obj_radius)))

        for index, item in enumerate(neighbors_in):
            parse_object(item, index, "neighbors", True)
        for index, item in enumerate(obstacles_in):
            parse_object(item, index, "obstacles", False)

        evaluations: list[dict] = []
        selected_id: str | None = None
        selected_velocity: dict | None = None
        best_distance: float | None = None
        for candidate_id, velocity, raw_x, raw_y in parsed_candidates:
            speed_ok = math.hypot(velocity[0], velocity[1]) <= float(max_speed)
            collision_ids: list[str] = []
            for object_id, rel_pos, obj_vel, radius_sum in objects:
                relative_velocity = (obj_vel[0] - velocity[0], obj_vel[1] - velocity[1])
                if _disc_collides(rel_pos, relative_velocity, radius_sum, float(horizon)):
                    collision_ids.append(object_id)
            admissible = speed_ok and not collision_ids
            evaluations.append(
                {
                    "id": candidate_id,
                    "speed_ok": speed_ok,
                    "collision_ids": collision_ids,
                    "admissible": admissible,
                }
            )
            if admissible:
                dx = velocity[0] - desired_velocity[0]
                dy = velocity[1] - desired_velocity[1]
                distance = math.hypot(dx, dy)
                # Candidates are visited in input order and the strict
                # comparison keeps the first one on an equal distance.
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    selected_id = candidate_id
                    # New dict holding the original numeric values so the
                    # velocity is echoed verbatim without aliasing the request.
                    selected_velocity = {"x": raw_x, "y": raw_y}

        if selected_id is None:
            return {
                "status": "BLOCKED",
                "selected": None,
                "velocity": {"x": 0, "y": 0},
                "evaluations": evaluations,
            }
        return {
            "status": "SELECTED",
            "selected": selected_id,
            "velocity": selected_velocity,
            "evaluations": evaluations,
        }

    def select_attention(self, request: Any) -> dict:
        """Score remembered entities for attention and pick the most salient.

        The request is validated in full before any entity is evaluated.
        ``observer`` has a finite ``position`` vector, a non-zero finite
        ``forward`` vector, a positive finite ``max_distance`` and a
        ``field_of_view_degrees`` in ``(0, 360]``; ``now`` is a non-negative
        finite number and ``memory_horizon`` a positive finite number.
        ``memory`` is a list (possibly empty) of perception-memory records,
        each with a unique non-empty string ``id``, ``kind``, ``last_seen``
        (non-negative finite, no later than ``now``), ``confidence`` in
        ``[0, 1]``, a finite ``position`` and an optional JSON ``data``;
        ``data.threat`` defaults to ``0`` and, when present, must be a finite
        number in ``[0, 1]``.

        An entity coincident with the observer is visible with proximity
        ``1``. Otherwise it is visible only when its distance is at most
        ``max_distance`` and the angle between the (un-normalised) entity
        direction and ``forward`` is at most half the field of view (a
        360-degree view ignores facing); either boundary counts as visible.
        Every record gets ``proximity = max(0, 1 - distance / max_distance)``,
        ``freshness = max(0, 1 - (now - last_seen) / memory_horizon)`` and a
        ``score`` of ``confidence * threat * proximity * freshness`` when
        visible (``0`` when not). The highest positive score wins, ties going
        to the record listed first; with no positive score the result is
        ``NO_TARGET`` with null ``selected`` and ``score``. ``evaluations``
        always lists every record in input order with ``id``, ``visible``,
        ``distance``, ``threat``, ``proximity``, ``freshness`` and ``score``.
        Raises ValueError (AttentionError) on any invalid input; the request
        is never mutated and no state is kept between calls.
        """
        if not isinstance(request, dict):
            raise AttentionError("request must be a JSON object")

        for field in ("observer", "now", "memory_horizon", "memory"):
            if field not in request:
                raise AttentionError(f"request is missing {field!r}")

        observer = request["observer"]
        if not isinstance(observer, dict):
            raise AttentionError("observer must be an object")
        observer_pos = _validate_attention_vector(observer.get("position"), "observer.position")
        forward = _validate_attention_vector(observer.get("forward"), "observer.forward")
        if forward == (0.0, 0.0):
            raise AttentionError("observer.forward must be a non-zero vector")
        max_distance = observer.get("max_distance")
        if not _is_finite_number(max_distance) or max_distance <= 0:
            raise AttentionError("observer.max_distance must be a positive finite number")
        fov = observer.get("field_of_view_degrees")
        if not _is_finite_number(fov) or not 0 < fov <= 360:
            raise AttentionError(
                "observer.field_of_view_degrees must be a finite number in (0, 360]"
            )

        now = request["now"]
        if not _is_finite_number(now) or now < 0:
            raise AttentionError("now must be a non-negative finite number")
        horizon = request["memory_horizon"]
        if not _is_finite_number(horizon) or horizon <= 0:
            raise AttentionError("memory_horizon must be a positive finite number")
        memory = request["memory"]
        if not isinstance(memory, list):
            raise AttentionError("memory must be a list")

        parsed_memory: list[dict] = []
        memory_ids: set[str] = set()
        for index, entity in enumerate(memory):
            where = f"memory[{index}]"
            try:
                parsed = _validate_perception_entity(entity, where, now, last_seen_required=True)
            except PerceptionError as exc:
                raise AttentionError(str(exc)) from exc
            if parsed["id"] in memory_ids:
                raise AttentionError(f"duplicate memory id {parsed['id']!r}")
            memory_ids.add(parsed["id"])
            if "threat" in parsed["data"]:
                threat = parsed["data"]["threat"]
                if not _is_finite_number(threat) or not 0 <= threat <= 1:
                    raise AttentionError(
                        f"{where}.data.threat must be a finite number between 0 and 1"
                    )
            else:
                parsed["data"]["threat"] = 0
            parsed_memory.append(parsed)

        max_distance_f = float(max_distance)
        half_view = math.radians(fov) / 2.0
        forward_len = math.hypot(forward[0], forward[1])
        fx, fy = forward[0] / forward_len, forward[1] / forward_len

        evaluations: list[dict] = []
        selected: str | None = None
        selected_score: float | None = None
        for entity in parsed_memory:
            dx = float(entity["position"]["x"]) - observer_pos[0]
            dy = float(entity["position"]["y"]) - observer_pos[1]
            distance = math.hypot(dx, dy)
            threat = entity["data"]["threat"]
            proximity = max(0.0, 1.0 - distance / max_distance_f)
            age = float(now) - float(entity["last_seen"])
            freshness = max(0.0, 1.0 - age / float(horizon))
            if distance == 0.0:
                visible = True
                proximity = 1.0
            elif distance > max_distance_f:
                visible = False
            elif fov == 360:
                visible = True
            else:
                # atan2(|cross|, dot) gives the unsigned angle in [0, pi];
                # a tiny tolerance keeps an exact cone boundary visible.
                angle = math.atan2(abs(fx * dy - fy * dx), fx * dx + fy * dy)
                visible = angle <= half_view + 1e-9
            score = (
                float(entity["confidence"]) * float(threat) * proximity * freshness
                if visible
                else 0.0
            )
            evaluations.append(
                {
                    "id": entity["id"],
                    "visible": visible,
                    "distance": distance,
                    "threat": threat,
                    "proximity": proximity,
                    "freshness": freshness,
                    "score": score,
                }
            )
            # Records are visited in input order and the strict comparison
            # keeps the first one on an equal score; zero never selects.
            if score > 0.0 and (selected_score is None or score > selected_score):
                selected = entity["id"]
                selected_score = score

        if selected is None:
            return {
                "status": "NO_TARGET",
                "selected": None,
                "score": None,
                "evaluations": evaluations,
            }
        return {
            "status": "SELECTED",
            "selected": selected,
            "score": selected_score,
            "evaluations": evaluations,
        }

    def match_dialogue_intent(self, request: Any) -> dict:
        """Match an utterance against rule-based intent patterns once.

        The request is validated in full before any matching happens.
        ``utterance`` is a string whose whitespace-normalised form is
        non-empty; ``intents`` is a non-empty list; ``context`` is an
        optional object (default ``{}``). Each intent requires a unique
        non-empty string ``id``, a non-empty ``patterns`` list of
        non-empty strings, an optional integer ``priority`` (default
        ``0``; booleans are not integers) and an optional ``requires``
        object constraining the context.

        Utterance and patterns are tokenised on runs of Unicode
        whitespace after stripping; comparison uses Unicode casefold
        while reported slot values keep the original utterance text. A
        pattern token of the exact form ``{name}`` (an ASCII letter or
        underscore followed by ASCII letters, digits or underscores) is
        a slot matching exactly one token; a repeated slot name must
        capture casefold-equal values. Every other token is a literal
        that must equal the utterance token. A pattern matches only when
        it covers the whole utterance, and an intent matches only when
        every ``requires`` key exists in the context with an equal value
        under JSON type-sensitive equality.

        When several patterns of one intent match, the one with the most
        literal tokens wins, ties going to the earliest pattern. Matched
        intents are ordered by descending ``priority``, then descending
        literal count, then declaration order. The result is ``MATCHED``
        with the first intent's ``id`` and ``slots`` plus a
        ``candidates`` list (``id``, ``priority``, ``literal_count``,
        ``pattern_index``, ``slots``) in that order, or ``NO_MATCH``
        with null ``intent`` and empty ``slots``/``candidates``. Raises
        ValueError (DialogueError) on any invalid input; the request is
        never mutated and no state is kept between calls.
        """
        if not isinstance(request, dict):
            raise DialogueError("request must be a JSON object")
        utterance = request.get("utterance")
        if not isinstance(utterance, str):
            raise DialogueError("utterance must be a string")
        utterance_tokens = utterance.split()
        if not utterance_tokens:
            raise DialogueError("utterance must not be empty")
        intents = request.get("intents")
        if not isinstance(intents, list) or not intents:
            raise DialogueError("intents must be a non-empty list")
        context = request.get("context", {})
        if not isinstance(context, dict):
            raise DialogueError("context must be an object")

        parsed_intents: list[tuple[str, int, dict, list]] = []
        seen_ids: set[str] = set()
        for index, intent in enumerate(intents):
            where = f"intent at intents[{index}]"
            if not isinstance(intent, dict):
                raise DialogueError(f"{where} must be an object")
            intent_id = intent.get("id")
            if not _is_valid_key(intent_id):
                raise DialogueError(f"{where} requires a non-empty string id")
            if intent_id in seen_ids:
                raise DialogueError(f"duplicate intent id {intent_id!r}")
            seen_ids.add(intent_id)
            where = f"intent {intent_id!r}"
            priority = intent.get("priority", 0)
            if isinstance(priority, bool) or not isinstance(priority, int):
                raise DialogueError(f"{where}: priority must be an integer")
            requires = intent.get("requires", {})
            if not isinstance(requires, dict):
                raise DialogueError(f"{where}: requires must be an object")
            patterns = intent.get("patterns")
            if not isinstance(patterns, list) or not patterns:
                raise DialogueError(f"{where}: patterns must be a non-empty list")
            parsed_patterns = [
                _parse_dialogue_pattern(pattern, f"{where} pattern at patterns[{p_index}]")
                for p_index, pattern in enumerate(patterns)
            ]
            parsed_intents.append((intent_id, priority, requires, parsed_patterns))

        matched: list[tuple[str, int, int, int, dict, int]] = []
        for order, (intent_id, priority, requires, parsed_patterns) in enumerate(parsed_intents):
            if not all(
                key in context and _json_equal(context[key], value)
                for key, value in requires.items()
            ):
                continue
            best: tuple[int, int, dict] | None = None
            for pattern_index, parsed in enumerate(parsed_patterns):
                slots = _match_dialogue_pattern(parsed, utterance_tokens)
                if slots is None:
                    continue
                literal_count = sum(1 for kind, _ in parsed if kind == "literal")
                # Patterns are visited in declaration order and the strict
                # comparison keeps the earliest one on an equal count.
                if best is None or literal_count > best[0]:
                    best = (literal_count, pattern_index, slots)
            if best is not None:
                matched.append((intent_id, priority, best[0], best[1], best[2], order))

        matched.sort(key=lambda item: (-item[1], -item[2], item[5]))
        if not matched:
            return {"status": "NO_MATCH", "intent": None, "slots": {}, "candidates": []}
        candidates = [
            {
                "id": intent_id,
                "priority": priority,
                "literal_count": literal_count,
                "pattern_index": pattern_index,
                "slots": slots,
            }
            for intent_id, priority, literal_count, pattern_index, slots, _ in matched
        ]
        first = matched[0]
        return {
            "status": "MATCHED",
            "intent": first[0],
            "slots": first[4],
            "candidates": candidates,
        }

    def select_schedule_activity(self, request: Any) -> dict:
        """Pick at most one activity for the current minute and need levels.

        The request is validated in full before any activity is evaluated.
        ``now`` is an integer minute in ``[0, 1439]`` (booleans are not
        integers); ``needs`` is an object mapping non-empty string keys to
        finite numbers in ``[0, 1]``; ``activities`` is a non-empty list.
        Each activity requires a unique non-empty string ``id``, integer
        ``start_minute``/``end_minute`` in ``[0, 1439]``, an optional
        integer ``priority`` (default ``0``) and an optional ``need``
        naming a declared needs key; an activity carrying ``need`` must
        also provide ``trigger`` and ``relief``, both finite numbers in
        ``[0, 1]``.

        The schedule window is start-inclusive and end-exclusive: a start
        before the end is a same-day interval, a start after the end wraps
        past midnight, and equal endpoints mean the whole day. An activity
        is ``scheduled`` when ``now`` falls in its window and ``urgent``
        when it carries a need whose current level is at least ``trigger``;
        either makes it ``eligible``. Urgent activities always outrank
        merely scheduled ones: among the urgent, the highest need level
        wins, then the highest priority, then the earliest input position;
        among the merely scheduled, the highest priority wins, ties going
        to the earliest input position. Selecting an activity with a need
        updates that need to ``max(0, level - relief)``; activities without
        a need leave the needs untouched.

        The result is ``SELECTED`` with the winning ``selected`` id and the
        updated ``needs``, or ``IDLE`` with null ``selected`` and the
        original ``needs`` when nothing is eligible; both carry
        ``evaluations`` listing every activity in input order with ``id``,
        ``scheduled``, ``urgent``, ``eligible`` and ``need_level`` (null
        for activities without a need). Raises ValueError (ScheduleError)
        on any invalid input; the request is never mutated and no state is
        kept between calls.
        """
        if not isinstance(request, dict):
            raise ScheduleError("request must be a JSON object")
        now = request.get("now")
        if isinstance(now, bool) or not isinstance(now, int) or not 0 <= now <= 1439:
            raise ScheduleError("now must be an integer between 0 and 1439")
        needs = request.get("needs")
        if not isinstance(needs, dict):
            raise ScheduleError("needs must be an object")
        for key, value in needs.items():
            if not _is_valid_key(key):
                raise ScheduleError(f"needs key {key!r} must be a non-empty string")
            if not _is_finite_number(value) or not 0 <= value <= 1:
                raise ScheduleError(f"needs[{key!r}] must be a finite number between 0 and 1")
        activities = request.get("activities")
        if not isinstance(activities, list) or not activities:
            raise ScheduleError("activities must be a non-empty list")

        parsed: list[tuple[str, int, int, int, str | None, float, float]] = []
        seen_ids: set[str] = set()
        for index, activity in enumerate(activities):
            where = f"activity at activities[{index}]"
            if not isinstance(activity, dict):
                raise ScheduleError(f"{where} must be an object")
            activity_id = activity.get("id")
            if not _is_valid_key(activity_id):
                raise ScheduleError(f"{where} requires a non-empty string id")
            if activity_id in seen_ids:
                raise ScheduleError(f"duplicate activity id {activity_id!r}")
            seen_ids.add(activity_id)
            where = f"activity {activity_id!r}"
            start = activity.get("start_minute")
            if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start <= 1439:
                raise ScheduleError(
                    f"{where}: start_minute must be an integer between 0 and 1439"
                )
            end = activity.get("end_minute")
            if isinstance(end, bool) or not isinstance(end, int) or not 0 <= end <= 1439:
                raise ScheduleError(
                    f"{where}: end_minute must be an integer between 0 and 1439"
                )
            priority = activity.get("priority", 0)
            if isinstance(priority, bool) or not isinstance(priority, int):
                raise ScheduleError(f"{where}: priority must be an integer")
            need: str | None = None
            trigger = 0.0
            relief = 0.0
            if "need" in activity:
                need = activity["need"]
                if not _is_valid_key(need) or need not in needs:
                    raise ScheduleError(f"{where}: need references unknown need {need!r}")
                if "trigger" not in activity:
                    raise ScheduleError(f"{where}: need requires a trigger")
                trigger = activity["trigger"]
                if not _is_finite_number(trigger) or not 0 <= trigger <= 1:
                    raise ScheduleError(
                        f"{where}: trigger must be a finite number between 0 and 1"
                    )
                if "relief" not in activity:
                    raise ScheduleError(f"{where}: need requires a relief")
                relief = activity["relief"]
                if not _is_finite_number(relief) or not 0 <= relief <= 1:
                    raise ScheduleError(
                        f"{where}: relief must be a finite number between 0 and 1"
                    )
            parsed.append((activity_id, start, end, priority, need, trigger, relief))

        evaluations: list[dict] = []
        urgent_pool: list[tuple[float, int, int, str, str, float]] = []
        scheduled_pool: list[tuple[int, int, str, str | None, float]] = []
        for index, (activity_id, start, end, priority, need, trigger, relief) in enumerate(parsed):
            scheduled = _minute_in_window(now, start, end)
            need_level = needs[need] if need is not None else None
            urgent = need is not None and need_level >= trigger
            eligible = scheduled or urgent
            evaluations.append(
                {
                    "id": activity_id,
                    "scheduled": scheduled,
                    "urgent": urgent,
                    "eligible": eligible,
                    "need_level": need_level,
                }
            )
            if urgent:
                urgent_pool.append((need_level, priority, index, activity_id, need, relief))
            elif scheduled:
                scheduled_pool.append((priority, index, activity_id, need, relief))

        selected_id: str | None = None
        selected_need: str | None = None
        selected_relief = 0.0
        if urgent_pool:
            # Highest need level, then highest priority, then earliest input.
            _, _, _, selected_id, selected_need, selected_relief = max(
                urgent_pool, key=lambda entry: (entry[0], entry[1], -entry[2])
            )
        elif scheduled_pool:
            # Highest priority, then earliest input.
            _, _, selected_id, selected_need, selected_relief = max(
                scheduled_pool, key=lambda entry: (entry[0], -entry[1])
            )

        updated_needs = dict(needs)
        if selected_id is None:
            return {
                "status": "IDLE",
                "selected": None,
                "needs": updated_needs,
                "evaluations": evaluations,
            }
        if selected_need is not None:
            updated_needs[selected_need] = max(0, needs[selected_need] - selected_relief)
        return {
            "status": "SELECTED",
            "selected": selected_id,
            "needs": updated_needs,
            "evaluations": evaluations,
        }

    def assign_team_roles(self, request: Any) -> dict:
        """Assign agents to at most one role each, maximising total score.

        The request is validated in full before any assignment is computed.
        ``roles`` and ``agents`` are non-empty lists. Each role has a unique
        non-empty string ``id`` and a positive integer ``capacity`` (booleans
        are not integers); each agent has a unique non-empty string ``id`` and
        a ``scores`` object keyed by role id. A score must be a finite number
        in ``[0, 1]`` (booleans are not numbers); roles not listed are not
        available to that agent, and a listed zero score never participates.

        Among all feasible assignments the one with the highest total score
        wins; ties are broken by encoding each agent's choice as the role's
        position in ``roles`` (unassigned last) and taking the
        lexicographically smallest sequence in agent order. All scores are
        finite floats and therefore exact binary fractions, so optimisation
        scales them to integers and compares equal decimal totals exactly.
        Returns ``status`` ``ASSIGNED``, ``assignments`` in agent input order
        (each ``{"agent", "role", "score"}``, assigned agents only),
        ``unassigned`` ids in input order, ``total_score`` and ``roles`` in
        role input order with each role's ``id``, ``capacity``, assigned
        ``agents`` and score subtotal. With no positive-score candidate the
        assignments are empty and the total score is zero; that is still
        ``ASSIGNED``. Raises ValueError (TeamAssignmentError) on any invalid
        input; the request is never mutated and no state is kept between calls.
        """
        if not isinstance(request, dict):
            raise TeamAssignmentError("request must be a JSON object")
        roles_in = request.get("roles")
        if not isinstance(roles_in, list) or not roles_in:
            raise TeamAssignmentError("roles must be a non-empty list")
        agents_in = request.get("agents")
        if not isinstance(agents_in, list) or not agents_in:
            raise TeamAssignmentError("agents must be a non-empty list")

        role_ids: list[str] = []
        capacities: list[int] = []
        role_index_by_id: dict[str, int] = {}
        for index, role in enumerate(roles_in):
            where = f"role at roles[{index}]"
            if not isinstance(role, dict):
                raise TeamAssignmentError(f"{where} must be an object")
            role_id = role.get("id")
            if not _is_valid_key(role_id):
                raise TeamAssignmentError(f"{where} requires a non-empty string id")
            if role_id in role_index_by_id:
                raise TeamAssignmentError(f"duplicate role id {role_id!r}")
            capacity = role.get("capacity")
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
                raise TeamAssignmentError(f"{where}: capacity must be a positive integer")
            role_index_by_id[role_id] = len(role_ids)
            role_ids.append(role_id)
            capacities.append(capacity)

        agent_ids: list[str] = []
        # Per agent: role index -> (original score value, exact Fraction).
        options: list[dict[int, tuple[Any, Fraction]]] = []
        seen_agent_ids: set[str] = set()
        for index, agent in enumerate(agents_in):
            where = f"agent at agents[{index}]"
            if not isinstance(agent, dict):
                raise TeamAssignmentError(f"{where} must be an object")
            agent_id = agent.get("id")
            if not _is_valid_key(agent_id):
                raise TeamAssignmentError(f"{where} requires a non-empty string id")
            if agent_id in seen_agent_ids:
                raise TeamAssignmentError(f"duplicate agent id {agent_id!r}")
            seen_agent_ids.add(agent_id)
            scores = agent.get("scores")
            if not isinstance(scores, dict):
                raise TeamAssignmentError(f"{where}: scores must be an object")
            agent_options: dict[int, tuple[Any, Fraction]] = {}
            for key, value in scores.items():
                if not _is_valid_key(key):
                    raise TeamAssignmentError(
                        f"{where}.scores key {key!r} must be a non-empty string"
                    )
                if key not in role_index_by_id:
                    raise TeamAssignmentError(
                        f"{where}.scores references unknown role {key!r}"
                    )
                if not _is_finite_number(value) or not 0 <= value <= 1:
                    raise TeamAssignmentError(
                        f"{where}.scores[{key!r}] must be a finite number between 0 and 1"
                    )
                if value == 0:
                    continue  # zero scores never participate in assignment
                agent_options[role_index_by_id[key]] = (value, Fraction(value))
            agent_ids.append(agent_id)
            options.append(agent_options)

        # Every finite float is an exact binary fraction, so every score's
        # denominator is a power of two; scale all scores by the LCM of those
        # denominators to get plain integer weights. The flow then compares
        # totals exactly (0.1 + 0.2 and a genuinely equal total tie) with gcd-
        # free integer arithmetic instead of Fraction operations.
        scale = 1
        for agent_options in options:
            for _, fraction in agent_options.values():
                scale = math.lcm(scale, fraction.denominator)

        def weight_of(fraction: Fraction) -> int:
            return fraction.numerator * (scale // fraction.denominator)

        role_count = len(role_ids)
        agent_count = len(agent_ids)
        unassigned_token = role_count

        # One min-cost max-flow over source -> agents -> roles -> sink, with an
        # extra agent -> sink edge for leaving an agent unassigned, so pushing
        # one unit per agent is always feasible. Edge costs are ordered pairs
        # ``(negated scaled score, lexicographic weight)`` compared
        # lexicographically and added componentwise: the first component
        # maximises the total score, and among equal totals the second
        # minimises the base-(role_count + 1) integer with digit choice_i
        # (role position, unassigned last), which is exactly the
        # lexicographically smallest choice sequence in agent order.
        base = role_count + 1
        source = 0
        agent_base = 1
        role_base = agent_base + agent_count
        sink = role_base + role_count
        graph: list[list[int]] = [[] for _ in range(sink + 1)]
        edges: list[list] = []  # [u, v, residual capacity, score cost, tie-break cost]

        def add_edge(u: int, v: int, capacity: int, score_cost: int, tie_cost: int) -> int:
            index = len(edges)
            graph[u].append(index)
            edges.append([u, v, capacity, score_cost, tie_cost])
            graph[v].append(index + 1)
            edges.append([v, u, 0, -score_cost, -tie_cost])
            return index

        used_edge: dict[tuple[int, int], int] = {}
        for k, agent_options in enumerate(options):
            agent_node = agent_base + k
            add_edge(source, agent_node, 1, 0, 0)
            add_edge(agent_node, sink, 1, 0, base ** (agent_count - 1 - k) * role_count)
            for role_index, (_, score) in agent_options.items():
                edge_index = add_edge(
                    agent_node,
                    role_base + role_index,
                    1,
                    -weight_of(score),
                    base ** (agent_count - 1 - k) * role_index,
                )
                used_edge[(k, role_index)] = edge_index
        for role_index in range(role_count):
            add_edge(role_base + role_index, sink, capacities[role_index], 0, 0)

        # Successive shortest paths with Johnson potentials. The initial
        # network is a DAG in node-index order, so the first shortest-path
        # labelling is a single forward relaxation pass (negative edges
        # included); afterwards reduced costs on the residual network are
        # non-negative and each augmentation uses a heap Dijkstra. Labels are
        # the ordered ``(int score cost, int tie-break cost)`` pairs.
        Label = tuple[int, int]
        potentials: list[Label] = [(0, 0)] * (sink + 1)
        first_dist: list[Label | None] = [None] * (sink + 1)
        first_dist[source] = (0, 0)
        for u in range(sink + 1):
            here = first_dist[u]
            if here is None:
                continue
            for edge_index in graph[u]:
                edge = edges[edge_index]
                _, v, residual, score_cost, tie_cost = edge
                if residual <= 0:
                    continue
                tentative = (here[0] + score_cost, here[1] + tie_cost)
                current = first_dist[v]
                if current is None or tentative < current:
                    first_dist[v] = tentative
        for v, distance in enumerate(first_dist):
            if distance is not None:
                potentials[v] = distance

        for _ in range(agent_count):
            distances: list[Label | None] = [None] * (sink + 1)
            previous: list[int | None] = [None] * (sink + 1)
            distances[source] = (0, 0)
            heap: list[tuple[int, int, int]] = [(0, 0, source)]
            while heap:
                score_dist, tie_dist, u = heapq.heappop(heap)
                current = distances[u]
                if current is None or (score_dist, tie_dist) != current:
                    continue
                pot_u_score, pot_u_tie = potentials[u]
                for edge_index in graph[u]:
                    edge = edges[edge_index]
                    _, v, residual, score_cost, tie_cost = edge
                    if residual <= 0:
                        continue
                    pot_v_score, pot_v_tie = potentials[v]
                    reduced = (
                        score_cost + pot_u_score - pot_v_score,
                        tie_cost + pot_u_tie - pot_v_tie,
                    )
                    tentative = (score_dist + reduced[0], tie_dist + reduced[1])
                    known = distances[v]
                    if known is None or tentative < known:
                        distances[v] = tentative
                        previous[v] = edge_index
                        heapq.heappush(heap, (tentative[0], tentative[1], v))
            if distances[sink] is None:
                break  # unreachable: cannot happen thanks to unassigned edges
            for v, distance in enumerate(distances):
                if distance is not None:
                    pot = potentials[v]
                    potentials[v] = (pot[0] + distance[0], pot[1] + distance[1])
            v = sink
            while v != source:
                edge_index = previous[v]
                assert edge_index is not None
                edge = edges[edge_index]
                edge[2] -= 1
                edges[edge_index ^ 1][2] += 1
                v = edge[0]

        chosen: list[int] = []
        for k in range(agent_count):
            picked = unassigned_token
            for role_index in options[k]:
                if edges[used_edge[(k, role_index)]][2] == 0:
                    picked = role_index
                    break
            chosen.append(picked)

        assignments: list[dict] = []
        unassigned: list[str] = []
        role_agents: list[list[str]] = [[] for _ in range(role_count)]
        role_subtotals = [Fraction(0) for _ in range(role_count)]
        total = Fraction(0)
        for i, choice in enumerate(chosen):
            if choice == unassigned_token:
                unassigned.append(agent_ids[i])
                continue
            value = options[i][choice][0]
            assignments.append({"agent": agent_ids[i], "role": role_ids[choice], "score": value})
            role_agents[choice].append(agent_ids[i])
            role_subtotals[choice] += options[i][choice][1]
            total += options[i][choice][1]

        def fraction_to_number(value: Fraction) -> Any:
            if value.denominator == 1:
                return value.numerator
            return float(value)

        return {
            "status": "ASSIGNED",
            "assignments": assignments,
            "unassigned": unassigned,
            "total_score": fraction_to_number(total),
            "roles": [
                {
                    "id": role_ids[j],
                    "capacity": capacities[j],
                    "agents": role_agents[j],
                    "score": fraction_to_number(role_subtotals[j]),
                }
                for j in range(role_count)
            ],
        }

    def adjust_difficulty(self, request: Any) -> dict:
        """Recommend one difficulty adjustment from recent performance signals.

        The request is validated in full before any score is computed.
        ``current_difficulty`` is a finite number in ``[0, 1]``; ``target``
        is an object whose ``min`` and ``max`` are finite numbers in
        ``[0, 1]`` with ``min <= max``; ``max_step`` is a finite number in
        ``(0, 1]``; ``now`` and ``cooldown`` are non-negative finite
        numbers; ``signals`` is a non-empty list whose items each carry a
        unique non-empty string ``id``, a ``value`` finite number in
        ``[0, 1]`` and a positive finite ``weight``. ``last_adjusted_at``
        may be omitted or null; when present it must be a non-negative
        finite number no later than ``now``.

        The weighted average of the signal values is the ``score``. When
        ``last_adjusted_at`` is not null and ``now - last_adjusted_at`` is
        less than ``cooldown`` the difficulty is left untouched and the
        status is ``COOLDOWN`` (a difference equal to ``cooldown`` allows
        adjustment). Otherwise a ``score`` above ``target.max`` raises the
        difficulty by the smallest of the overshoot, ``max_step`` and the
        remaining headroom to 1; a ``score`` below ``target.min`` lowers it
        by the smallest of the undershoot, ``max_step`` and the current
        difficulty; a score inside the closed interval leaves it unchanged.
        The result carries ``status`` (``INCREASED``, ``DECREASED``,
        ``UNCHANGED`` — also used when an adjustment is requested but the
        difficulty already sits at the boundary — or ``COOLDOWN``),
        ``previous_difficulty``, ``difficulty``, ``score`` and
        ``adjustment`` (new minus old difficulty). Raises ValueError
        (DifficultyError) on any invalid input; the request is never
        mutated and no state is kept between calls.
        """
        if not isinstance(request, dict):
            raise DifficultyError("request must be a JSON object")
        current = request.get("current_difficulty")
        if not _is_finite_number(current) or not 0 <= current <= 1:
            raise DifficultyError(
                "current_difficulty must be a finite number between 0 and 1"
            )
        target = request.get("target")
        if not isinstance(target, dict):
            raise DifficultyError("target must be an object")
        low = target.get("min")
        if not _is_finite_number(low) or not 0 <= low <= 1:
            raise DifficultyError("target.min must be a finite number between 0 and 1")
        high = target.get("max")
        if not _is_finite_number(high) or not 0 <= high <= 1:
            raise DifficultyError("target.max must be a finite number between 0 and 1")
        if low > high:
            raise DifficultyError("target.min must not exceed target.max")
        max_step = request.get("max_step")
        if not _is_finite_number(max_step) or not 0 < max_step <= 1:
            raise DifficultyError("max_step must be a finite number in (0, 1]")
        now = request.get("now")
        if not _is_finite_number(now) or now < 0:
            raise DifficultyError("now must be a non-negative finite number")
        cooldown = request.get("cooldown")
        if not _is_finite_number(cooldown) or cooldown < 0:
            raise DifficultyError("cooldown must be a non-negative finite number")
        signals = request.get("signals")
        if not isinstance(signals, list) or not signals:
            raise DifficultyError("signals must be a non-empty list")
        last_adjusted_at = request.get("last_adjusted_at")
        if last_adjusted_at is not None:
            if not _is_finite_number(last_adjusted_at) or last_adjusted_at < 0:
                raise DifficultyError(
                    "last_adjusted_at must be null or a non-negative finite number"
                )
            if last_adjusted_at > now:
                raise DifficultyError("last_adjusted_at must not be later than now")

        parsed_signals: list[tuple[float, float]] = []
        seen_ids: set[str] = set()
        for index, signal in enumerate(signals):
            where = f"signal at signals[{index}]"
            if not isinstance(signal, dict):
                raise DifficultyError(f"{where} must be an object")
            signal_id = signal.get("id")
            if not _is_valid_key(signal_id):
                raise DifficultyError(f"{where} requires a non-empty string id")
            if signal_id in seen_ids:
                raise DifficultyError(f"duplicate signal id {signal_id!r}")
            seen_ids.add(signal_id)
            where = f"signal {signal_id!r}"
            value = signal.get("value")
            if not _is_finite_number(value) or not 0 <= value <= 1:
                raise DifficultyError(
                    f"{where}: value must be a finite number between 0 and 1"
                )
            weight = signal.get("weight")
            if not _is_finite_number(weight) or weight <= 0:
                raise DifficultyError(f"{where}: weight must be a positive finite number")
            parsed_signals.append((float(value), float(weight)))

        weighted_sum = sum(value * weight for value, weight in parsed_signals)
        weight_total = sum(weight for _, weight in parsed_signals)
        score = weighted_sum / weight_total

        previous = current
        if last_adjusted_at is not None and now - last_adjusted_at < cooldown:
            return {
                "status": "COOLDOWN",
                "previous_difficulty": previous,
                "difficulty": previous,
                "score": score,
                "adjustment": 0,
            }

        if score > high:
            step = min(score - high, float(max_step), 1.0 - float(current))
            if step > 0:
                status = "INCREASED"
                new_difficulty: Any = float(current) + step
            else:  # already at the upper boundary
                status = "UNCHANGED"
                new_difficulty = current
        elif score < low:
            step = min(low - score, float(max_step), float(current))
            if step > 0:
                status = "DECREASED"
                new_difficulty = float(current) - step
            else:  # already at the lower boundary
                status = "UNCHANGED"
                new_difficulty = current
        else:
            status = "UNCHANGED"
            new_difficulty = current

        return {
            "status": status,
            "previous_difficulty": previous,
            "difficulty": new_difficulty,
            "score": score,
            "adjustment": new_difficulty - previous,
        }
