import copy
import unittest

from npcmind.service import Service


def request(**overrides):
    base = {
        "world": {"has_wood": False, "has_planks": False},
        "goal": {"has_planks": True},
        "actions": [
            {
                "id": "chop",
                "cost": 2,
                "preconditions": {},
                "effects": {"has_wood": True},
            },
            {
                "id": "saw",
                "cost": 3,
                "preconditions": {"has_wood": True},
                "effects": {"has_planks": True},
            },
        ],
    }
    base.update(overrides)
    return base


class PlanGoapTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_goal_already_satisfied_returns_empty_plan(self):
        result = self.service.plan_goap(
            request(world={"has_wood": True, "has_planks": True, "extra": 1})
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["plan"], [])
        self.assertEqual(result["cost"], 0)
        self.assertEqual(
            result["final_world"],
            {"has_wood": True, "has_planks": True, "extra": 1},
        )

    def test_finds_minimum_cost_plan(self):
        result = self.service.plan_goap(request())
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["plan"], ["chop", "saw"])
        self.assertEqual(result["cost"], 5)
        self.assertEqual(
            result["final_world"], {"has_wood": True, "has_planks": True}
        )

    def test_cheaper_plan_wins_over_shorter_one(self):
        result = self.service.plan_goap(
            request(
                world={"has_wood": True},
                actions=[
                    {"id": "buy", "cost": 10, "effects": {"has_planks": True}},
                    {"id": "saw", "cost": 3, "effects": {"has_planks": True}},
                ],
            )
        )
        self.assertEqual(result["plan"], ["saw"])
        self.assertEqual(result["cost"], 3)

    def test_cost_tie_breaks_on_input_position_sequence(self):
        result = self.service.plan_goap(
            request(
                world={},
                goal={"x": 1},
                actions=[
                    {"id": "direct", "cost": 2, "effects": {"x": 1}},
                    {"id": "step1", "cost": 1, "effects": {"y": 1}},
                    {
                        "id": "step2",
                        "cost": 1,
                        "preconditions": {"y": 1},
                        "effects": {"x": 1},
                    },
                ],
            )
        )
        self.assertEqual(result["cost"], 2)
        self.assertEqual(result["plan"], ["direct"])  # (0,) < (1, 2)

    def test_cost_tie_prefers_earlier_positions_over_shorter_plan(self):
        result = self.service.plan_goap(
            request(
                world={},
                goal={"x": 1},
                actions=[
                    {"id": "prep", "cost": 1, "effects": {"z": 1}},
                    {
                        "id": "use",
                        "cost": 1,
                        "preconditions": {"z": 1},
                        "effects": {"x": 1},
                    },
                    {"id": "direct", "cost": 2, "effects": {"x": 1}},
                ],
            )
        )
        self.assertEqual(result["cost"], 2)
        self.assertEqual(result["plan"], ["prep", "use"])  # (0, 1) < (2,)

    def test_default_cost_and_optional_fields(self):
        result = self.service.plan_goap(
            request(
                world={},
                goal={"done": True},
                actions=[{"id": "only", "effects": {"done": True}}],
            )
        )
        self.assertEqual(result["plan"], ["only"])
        self.assertEqual(result["cost"], 1)

    def test_effects_overwrite_and_preserve_other_keys(self):
        result = self.service.plan_goap(
            request(
                world={"a": 1, "keep": [1, 2]},
                goal={"a": 2},
                actions=[{"id": "bump", "effects": {"a": 2}}],
            )
        )
        self.assertEqual(result["final_world"], {"a": 2, "keep": [1, 2]})

    def test_unreachable_goal_returns_original_world(self):
        world = {"has_wood": False, "has_planks": False}
        result = self.service.plan_goap(request(world=world, actions=[]))
        self.assertEqual(result["status"], "UNREACHABLE")
        self.assertEqual(result["plan"], [])
        self.assertIsNone(result["cost"])
        self.assertEqual(result["final_world"], world)

    def test_int_and_float_compare_by_value(self):
        result = self.service.plan_goap(
            request(world={"n": 1}, goal={"n": 1.0}, actions=[])
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["cost"], 0)

    def test_bool_never_equals_number(self):
        result = self.service.plan_goap(
            request(world={"flag": True}, goal={"flag": 1}, actions=[])
        )
        self.assertEqual(result["status"], "UNREACHABLE")

    def test_nested_values_compare_recursively(self):
        result = self.service.plan_goap(
            request(
                world={"inv": {"items": [1, 2.0]}},
                goal={"inv": {"items": [1.0, 2]}},
                actions=[],
            )
        )
        self.assertEqual(result["status"], "SUCCESS")

    def test_request_is_not_mutated(self):
        req = request()
        snapshot = copy.deepcopy(req)
        self.service.plan_goap(req)
        self.assertEqual(req, snapshot)

    def test_repeated_calls_are_independent(self):
        first = self.service.plan_goap(request())
        second = self.service.plan_goap(request())
        self.assertEqual(first, second)

    def assert_invalid(self, req):
        with self.assertRaises(ValueError):
            self.service.plan_goap(req)

    def test_request_must_be_object(self):
        self.assert_invalid([1, 2])

    def test_world_goal_actions_types_checked(self):
        self.assert_invalid(request(world=[]))
        self.assert_invalid(request(goal="x"))
        self.assert_invalid(request(actions={}))
        self.assert_invalid({"world": {}, "goal": {}})

    def test_state_keys_must_be_non_empty_strings(self):
        self.assert_invalid(request(world={"": 1}))
        self.assert_invalid(request(goal={"": 1}))
        self.assert_invalid(request(actions=[{"id": "a", "effects": {"": 1}}]))
        self.assert_invalid(request(actions=[{"id": "a", "preconditions": {"": 1}}]))

    def test_action_id_validation(self):
        self.assert_invalid(request(actions=[{"cost": 1}]))
        self.assert_invalid(request(actions=[{"id": ""}]))
        self.assert_invalid(request(actions=[{"id": 3}]))
        self.assert_invalid(request(actions=["not-an-object"]))
        self.assert_invalid(request(actions=[{"id": "a"}, {"id": "a"}]))

    def test_cost_must_be_positive_integer(self):
        for bad in (True, 0, -1, 1.5, 2.0, "1"):
            self.assert_invalid(request(actions=[{"id": "a", "cost": bad}]))

    def test_preconditions_and_effects_must_be_objects(self):
        self.assert_invalid(request(actions=[{"id": "a", "preconditions": []}]))
        self.assert_invalid(request(actions=[{"id": "a", "effects": "x"}]))

    def test_values_must_be_legal_json(self):
        self.assert_invalid(request(world={"x": float("nan")}))
        self.assert_invalid(request(goal={"x": float("inf")}))
        self.assert_invalid(request(world={"x": (1, 2)}))
        self.assert_invalid(request(world={"x": {1: "bad-key"}}))
        self.assert_invalid(
            request(actions=[{"id": "a", "effects": {"x": float("-inf")}}])
        )


if __name__ == "__main__":
    unittest.main()
