import copy
import json
import unittest

from npcmind.service import GoapError, Service


def action(action_id, cost=None, preconditions=None, effects=None):
    act = {"id": action_id}
    if cost is not None:
        act["cost"] = cost
    if preconditions is not None:
        act["preconditions"] = preconditions
    if effects is not None:
        act["effects"] = effects
    return act


class PlanGoapTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def plan(self, world, goal, actions):
        return self.service.plan_goap({"world": world, "goal": goal, "actions": actions})

    def test_goal_already_satisfied(self):
        world = {"has_key": True, "extra": [1, 2]}
        result = self.plan(world, {"has_key": True}, [])
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["plan"], [])
        self.assertEqual(result["cost"], 0)
        self.assertEqual(result["final_world"], world)

    def test_simple_plan(self):
        result = self.plan(
            {"has_key": False},
            {"door_open": True},
            [
                action("open", cost=5, preconditions={"has_key": True}, effects={"door_open": True}),
                action("take_key", cost=2, effects={"has_key": True}),
            ],
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["plan"], ["take_key", "open"])
        self.assertEqual(result["cost"], 7)
        self.assertEqual(
            result["final_world"],
            {"has_key": True, "door_open": True},
        )

    def test_lowest_cost_wins(self):
        result = self.plan(
            {},
            {"fed": True},
            [
                action("cook", cost=6, effects={"fed": True}),
                action("order", cost=3, effects={"fed": True}),
            ],
        )
        self.assertEqual(result["plan"], ["order"])
        self.assertEqual(result["cost"], 3)

    def test_cost_tie_broken_by_input_position_sequence(self):
        # Plans [0, 1], [1, 0] and [2] all cost 4; [0, 1] is lexicographically
        # smallest by input position.
        result = self.plan(
            {},
            {"a": True, "b": True},
            [
                action("set_b", cost=2, effects={"b": True}),
                action("set_a", cost=2, effects={"a": True}),
                action("set_both", cost=4, effects={"a": True, "b": True}),
            ],
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["plan"], ["set_b", "set_a"])
        self.assertEqual(result["cost"], 4)

    def test_default_cost_is_one(self):
        result = self.plan(
            {},
            {"x": 1},
            [action("bump", effects={"x": 1})],
        )
        self.assertEqual(result["cost"], 1)

    def test_unreachable_goal(self):
        world = {"gold": 0}
        result = self.plan(world, {"gold": 100}, [action("rest", effects={"rested": True})])
        self.assertEqual(result["status"], "UNREACHABLE")
        self.assertEqual(result["plan"], [])
        self.assertIsNone(result["cost"])
        self.assertEqual(result["final_world"], world)

    def test_numeric_equality_and_bool_distinction(self):
        # int 1 equals float 1.0; True does not equal 1.
        result = self.plan(
            {"level": 1.0},
            {"level": 1},
            [action("noop")],
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["plan"], [])
        result = self.plan(
            {"flag": True},
            {"flag": 1},
            [],
        )
        self.assertEqual(result["status"], "UNREACHABLE")

    def test_nested_values_match_recursively(self):
        result = self.plan(
            {"inv": {"items": ["sword", 3]}},
            {"inv": {"items": ["sword", 3.0]}},
            [],
        )
        self.assertEqual(result["status"], "SUCCESS")

    def test_effects_preserve_other_keys(self):
        result = self.plan(
            {"a": 1, "b": 2},
            {"a": 9},
            [action("bump", effects={"a": 9})],
        )
        self.assertEqual(result["final_world"], {"a": 9, "b": 2})

    def test_request_is_not_mutated(self):
        request = {
            "world": {"inv": ["coin"]},
            "goal": {"rich": True},
            "actions": [
                action("work", preconditions={"inv": ["coin"]}, effects={"rich": True, "inv": []})
            ],
        }
        snapshot = copy.deepcopy(request)
        result = self.service.plan_goap(request)
        self.assertEqual(request, snapshot)
        self.assertEqual(result["plan"], ["work"])
        # Mutating the result must not leak into the request either.
        result["final_world"]["inv"].append("hacked")
        self.assertEqual(request, snapshot)

    def test_result_is_json_serializable(self):
        result = self.plan({}, {"x": None}, [action("set_x", effects={"x": None})])
        json.dumps(result, sort_keys=True)

    def test_invalid_requests_raise_value_error(self):
        bad_requests = [
            None,
            [],
            {"goal": {}, "actions": []},  # missing world
            {"world": [], "goal": {}, "actions": []},
            {"world": {}, "goal": "x", "actions": []},
            {"world": {}, "goal": {}, "actions": {}},
            {"world": {"": 1}, "goal": {}, "actions": []},
            {"world": {}, "goal": {"": 1}, "actions": []},
            {"world": {}, "goal": {}, "actions": [{"id": ""}]},
            {"world": {}, "goal": {}, "actions": [{"id": 3}]},
            {"world": {}, "goal": {}, "actions": [{"name": "a"}]},
            {"world": {}, "goal": {}, "actions": [{"id": "a"}, {"id": "a"}]},
            {"world": {}, "goal": {}, "actions": [{"id": "a", "cost": True}]},
            {"world": {}, "goal": {}, "actions": [{"id": "a", "cost": 1.5}]},
            {"world": {}, "goal": {}, "actions": [{"id": "a", "cost": 0}]},
            {"world": {}, "goal": {}, "actions": [{"id": "a", "cost": -2}]},
            {"world": {}, "goal": {}, "actions": [{"id": "a", "preconditions": []}]},
            {"world": {}, "goal": {}, "actions": [{"id": "a", "effects": "x"}]},
            {"world": {}, "goal": {}, "actions": [{"id": "a", "effects": {"": 1}}]},
            {"world": {"x": float("nan")}, "goal": {}, "actions": []},
            {"world": {"x": [float("inf")]}, "goal": {}, "actions": []},
            {"world": {}, "goal": {"x": float("-inf")}, "actions": []},
            {"world": {}, "goal": {}, "actions": [{"id": "a", "effects": {"x": float("nan")}}]},
            {"world": {"x": object()}, "goal": {}, "actions": []},
            {"world": {"x": {1: "bad"}}, "goal": {}, "actions": []},
        ]
        for bad in bad_requests:
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    self.service.plan_goap(bad)

    def test_goap_error_is_value_error(self):
        with self.assertRaises(GoapError):
            self.service.plan_goap({"world": {}, "goal": {}, "actions": [{"id": "a", "cost": 0}]})


if __name__ == "__main__":
    unittest.main()
