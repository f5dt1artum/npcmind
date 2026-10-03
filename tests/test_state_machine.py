import unittest

from npcmind.service import Service


def machine(**overrides):
    base = {
        "states": [{"id": "idle"}, {"id": "walk"}, {"id": "run"}],
        "initial": "idle",
        "transitions": [
            {"id": "t1", "from": "idle", "to": "walk", "event": "go"},
            {
                "id": "t2",
                "from": "walk",
                "to": "run",
                "event": "go",
                "condition": {"op": "equals", "key": "speed", "value": 2},
                "actions": [
                    {"op": "set", "key": "moving", "value": True},
                    {"op": "delete", "key": "speed"},
                ],
            },
            {"id": "t3", "from": "walk", "to": "run", "event": "go"},
        ],
    }
    base.update(overrides)
    return base


class StepStateMachineTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def step(self, request=None, **kwargs):
        if request is None:
            request = {"machine": machine(), "event": "go"}
            request.update(kwargs)
        return self.service.step_state_machine(request)

    def test_default_current_state_is_initial(self):
        result = self.step()
        self.assertEqual(result["previous_state"], "idle")
        self.assertEqual(result["state"], "walk")
        self.assertEqual(result["transition"], "t1")
        self.assertEqual(result["blackboard"], {})
        self.assertEqual(result["trace"], [{"id": "t1", "condition": True}])

    def test_condition_selects_first_matching_transition(self):
        result = self.step(current_state="walk", blackboard={"speed": 2})
        self.assertEqual(result["state"], "run")
        self.assertEqual(result["transition"], "t2")
        self.assertEqual(result["blackboard"], {"moving": True})
        self.assertEqual(result["trace"], [{"id": "t2", "condition": True}])

    def test_condition_failure_falls_through_to_next_candidate(self):
        result = self.step(current_state="walk", blackboard={"speed": 1})
        self.assertEqual(result["state"], "run")
        self.assertEqual(result["transition"], "t3")
        self.assertEqual(result["blackboard"], {"speed": 1})
        self.assertEqual(
            result["trace"],
            [{"id": "t2", "condition": False}, {"id": "t3", "condition": True}],
        )

    def test_no_match_keeps_state_and_blackboard(self):
        result = self.step(current_state="run", blackboard={"k": 1})
        self.assertEqual(result["previous_state"], "run")
        self.assertEqual(result["state"], "run")
        self.assertIsNone(result["transition"])
        self.assertEqual(result["blackboard"], {"k": 1})
        self.assertEqual(result["trace"], [])

    def test_no_match_keeps_examined_candidates_in_trace(self):
        request = {
            "machine": machine(
                transitions=[
                    {
                        "id": "t1",
                        "from": "idle",
                        "to": "walk",
                        "event": "go",
                        "condition": {"op": "exists", "key": "missing"},
                    }
                ]
            ),
            "event": "go",
        }
        result = self.step(request)
        self.assertIsNone(result["transition"])
        self.assertEqual(result["state"], "idle")
        self.assertEqual(result["trace"], [{"id": "t1", "condition": False}])

    def test_condition_type_sensitive_equality(self):
        request = {
            "machine": machine(
                transitions=[
                    {
                        "id": "t1",
                        "from": "idle",
                        "to": "walk",
                        "event": "go",
                        "condition": {"op": "equals", "key": "flag", "value": 1},
                    }
                ]
            ),
            "event": "go",
            "blackboard": {"flag": True},
        }
        result = self.step(request)
        self.assertIsNone(result["transition"])  # true != 1

    def test_not_equals_and_missing_key(self):
        transitions = [
            {
                "id": "t1",
                "from": "idle",
                "to": "walk",
                "event": "go",
                "condition": {"op": "not_equals", "key": "k", "value": 1},
            }
        ]
        result = self.step({"machine": machine(transitions=transitions), "event": "go", "blackboard": {"k": 2}})
        self.assertEqual(result["transition"], "t1")
        result = self.step({"machine": machine(transitions=transitions), "event": "go"})
        self.assertIsNone(result["transition"])  # missing key fails

    def test_input_objects_are_not_mutated(self):
        board = {"speed": 2}
        request = {"machine": machine(), "event": "go", "current_state": "walk", "blackboard": board}
        self.step(request)
        self.assertEqual(board, {"speed": 2})
        self.assertEqual(request["machine"]["transitions"][1]["from"], "walk")

    def test_blackboard_defaults_to_empty_object(self):
        result = self.step()
        self.assertEqual(result["blackboard"], {})

    def test_states_may_be_plain_strings(self):
        request = {
            "machine": {
                "states": ["a", "b"],
                "initial": "a",
                "transitions": [{"id": "t", "from": "a", "to": "b", "event": "x"}],
            },
            "event": "x",
        }
        result = self.step(request)
        self.assertEqual(result["state"], "b")
        self.assertEqual(result["transition"], "t")


def hierarchical_machine(**overrides):
    base = {
        "states": [
            {"id": "active", "initial": "idle"},
            {"id": "idle", "parent": "active"},
            {"id": "walk", "parent": "active"},
            {"id": "combat", "initial": "melee"},
            {"id": "melee", "parent": "combat"},
            {"id": "ranged", "parent": "combat"},
            "dead",
        ],
        "initial": "active",
        "transitions": [
            {"id": "t_idle_walk", "from": "idle", "to": "walk", "event": "go"},
            {
                "id": "t_active_combat",
                "from": "active",
                "to": "combat",
                "event": "fight",
                "actions": [{"op": "set", "key": "fighting", "value": True}],
            },
            {"id": "t_any_dead", "from": "active", "to": "dead", "event": "die"},
            {
                "id": "t_walk_guarded",
                "from": "walk",
                "to": "idle",
                "event": "stop",
                "condition": {"op": "exists", "key": "tired"},
            },
            {"id": "t_active_stop", "from": "active", "to": "idle", "event": "stop"},
        ],
    }
    base.update(overrides)
    return base


class HierarchicalStepTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def step(self, request=None, **kwargs):
        if request is None:
            request = {"machine": hierarchical_machine(), "event": "go"}
            request.update(kwargs)
        return self.service.step_state_machine(request)

    def test_initial_expands_to_leaf(self):
        result = self.step()
        self.assertEqual(result["previous_state"], "idle")
        self.assertEqual(result["state"], "walk")
        self.assertEqual(result["transition"], "t_idle_walk")

    def test_current_state_may_be_composite(self):
        result = self.step(current_state="active")
        self.assertEqual(result["previous_state"], "idle")
        self.assertEqual(result["state"], "walk")

    def test_composite_target_expands_to_leaf(self):
        result = self.step(event="fight")
        self.assertEqual(result["previous_state"], "idle")
        self.assertEqual(result["state"], "melee")
        self.assertEqual(result["transition"], "t_active_combat")
        self.assertEqual(result["blackboard"], {"fighting": True})
        self.assertEqual(result["trace"], [{"id": "t_active_combat", "condition": True}])

    def test_deeper_level_wins_over_parent_level(self):
        # "stop" matches both walk (deeper, guarded) and active (ancestor).
        result = self.step(current_state="walk", event="stop", blackboard={"tired": 1})
        self.assertEqual(result["transition"], "t_walk_guarded")
        self.assertEqual(
            result["trace"],
            [{"id": "t_walk_guarded", "condition": True}],
        )

    def test_parent_level_checked_after_deeper_level_fails(self):
        result = self.step(current_state="walk", event="stop")
        self.assertEqual(result["transition"], "t_active_stop")
        self.assertEqual(result["state"], "idle")
        self.assertEqual(
            result["trace"],
            [
                {"id": "t_walk_guarded", "condition": False},
                {"id": "t_active_stop", "condition": True},
            ],
        )

    def test_no_match_anywhere_keeps_leaf_state_and_blackboard(self):
        result = self.step(current_state="ranged", event="go", blackboard={"k": 1})
        self.assertEqual(result["previous_state"], "ranged")
        self.assertEqual(result["state"], "ranged")
        self.assertIsNone(result["transition"])
        self.assertEqual(result["blackboard"], {"k": 1})
        self.assertEqual(result["trace"], [])

    def test_deep_nesting_expands_through_levels(self):
        request = {
            "machine": {
                "states": [
                    {"id": "root", "initial": "mid"},
                    {"id": "mid", "parent": "root", "initial": "leaf"},
                    {"id": "leaf", "parent": "mid"},
                    "other",
                ],
                "initial": "root",
                "transitions": [
                    {"id": "t", "from": "root", "to": "other", "event": "x"},
                ],
            },
            "event": "x",
        }
        result = self.step(request)
        self.assertEqual(result["previous_state"], "leaf")
        self.assertEqual(result["state"], "other")
        self.assertEqual(result["transition"], "t")

    def test_inputs_are_not_mutated(self):
        board = {"tired": 1}
        request = {
            "machine": hierarchical_machine(),
            "event": "stop",
            "current_state": "walk",
            "blackboard": board,
        }
        self.step(request)
        self.assertEqual(board, {"tired": 1})
        self.assertEqual(request["machine"]["states"][0], {"id": "active", "initial": "idle"})


class HierarchicalMachineErrorsTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def assert_invalid(self, machine, *fragments):
        with self.assertRaises(ValueError) as ctx:
            self.service.step_state_machine({"machine": machine, "event": "go"})
        for fragment in fragments:
            self.assertIn(fragment, str(ctx.exception))

    def test_parent_unknown(self):
        self.assert_invalid(
            hierarchical_machine(states=[{"id": "a", "parent": "ghost"}, "b"], initial="b"),
            "parent",
        )

    def test_parent_self(self):
        self.assert_invalid(
            hierarchical_machine(states=[{"id": "a", "parent": "a"}], initial="a"),
            "parent",
        )

    def test_parent_cycle(self):
        self.assert_invalid(
            hierarchical_machine(
                states=[{"id": "a", "parent": "b"}, {"id": "b", "parent": "a"}],
                initial="a",
            ),
            "cycle",
        )

    def test_composite_missing_initial(self):
        self.assert_invalid(
            hierarchical_machine(states=[{"id": "a"}, {"id": "b", "parent": "a"}], initial="a"),
            "initial",
        )

    def test_composite_initial_not_direct_child(self):
        self.assert_invalid(
            hierarchical_machine(
                states=[
                    {"id": "a", "initial": "c"},
                    {"id": "b", "parent": "a"},
                    {"id": "c"},
                ],
                initial="a",
            ),
            "initial",
        )

    def test_composite_initial_unknown(self):
        self.assert_invalid(
            hierarchical_machine(
                states=[{"id": "a", "initial": "ghost"}, {"id": "b", "parent": "a"}],
                initial="a",
            ),
            "initial",
        )

    def test_leaf_declares_initial(self):
        self.assert_invalid(
            hierarchical_machine(states=[{"id": "a", "initial": "b"}, "b"], initial="a"),
            "leaf",
        )

    def test_machine_initial_not_root(self):
        self.assert_invalid(
            hierarchical_machine(
                states=[{"id": "a", "initial": "b"}, {"id": "b", "parent": "a"}],
                initial="b",
            ),
            "root",
        )


class StepStateMachineErrorsTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def assert_invalid(self, request, *fragments):
        with self.assertRaises(ValueError) as ctx:
            self.service.step_state_machine(request)
        for fragment in fragments:
            self.assertIn(fragment, str(ctx.exception))

    def test_request_not_object(self):
        self.assert_invalid([1], "object")

    def test_missing_machine(self):
        self.assert_invalid({"event": "go"}, "machine")

    def test_machine_not_object(self):
        self.assert_invalid({"machine": 7, "event": "go"}, "machine")

    def test_event_not_string(self):
        self.assert_invalid({"machine": machine(), "event": 3}, "event")
        self.assert_invalid({"machine": machine()}, "event")

    def test_blackboard_not_object(self):
        self.assert_invalid({"machine": machine(), "event": "go", "blackboard": []}, "blackboard")

    def test_empty_states(self):
        self.assert_invalid({"machine": machine(states=[]), "event": "go"}, "states")

    def test_states_not_list(self):
        self.assert_invalid({"machine": machine(states="idle"), "event": "go"}, "states")

    def test_state_empty_id(self):
        self.assert_invalid({"machine": machine(states=[{"id": ""}]), "event": "go"}, "id")

    def test_duplicate_state_id(self):
        self.assert_invalid({"machine": machine(states=["a", "a"], initial="a"), "event": "go"}, "duplicate")

    def test_initial_unknown(self):
        self.assert_invalid({"machine": machine(initial="ghost"), "event": "go"}, "initial")

    def test_current_state_unknown(self):
        self.assert_invalid({"machine": machine(), "event": "go", "current_state": "ghost"}, "current_state")

    def test_current_state_not_string(self):
        self.assert_invalid({"machine": machine(), "event": "go", "current_state": 1}, "current_state")

    def test_transitions_not_list(self):
        self.assert_invalid({"machine": machine(transitions={}), "event": "go"}, "transitions")

    def test_duplicate_transition_id(self):
        transitions = [
            {"id": "t", "from": "idle", "to": "walk", "event": "go"},
            {"id": "t", "from": "idle", "to": "run", "event": "go"},
        ]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "go"}, "duplicate")

    def test_transition_from_unknown(self):
        transitions = [{"id": "t", "from": "ghost", "to": "walk", "event": "go"}]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "go"}, "from")

    def test_transition_to_unknown(self):
        transitions = [{"id": "t", "from": "idle", "to": "ghost", "event": "go"}]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "go"}, "to")

    def test_transition_event_not_string(self):
        transitions = [{"id": "t", "from": "idle", "to": "walk", "event": 5}]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "go"}, "event")

    def test_transition_empty_id(self):
        transitions = [{"id": "", "from": "idle", "to": "walk", "event": "go"}]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "go"}, "id")

    def test_condition_illegal(self):
        transitions = [
            {"id": "t", "from": "idle", "to": "walk", "event": "go", "condition": {"op": "gt", "key": "k"}}
        ]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "go"}, "condition")

    def test_condition_missing_value(self):
        transitions = [
            {"id": "t", "from": "idle", "to": "walk", "event": "go", "condition": {"op": "equals", "key": "k"}}
        ]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "go"}, "value")

    def test_action_illegal_op(self):
        transitions = [
            {"id": "t", "from": "idle", "to": "walk", "event": "go", "actions": [{"op": "status", "status": "SUCCESS"}]}
        ]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "go"}, "action")

    def test_action_set_missing_value(self):
        transitions = [
            {"id": "t", "from": "idle", "to": "walk", "event": "go", "actions": [{"op": "set", "key": "k"}]}
        ]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "go"}, "value")

    def test_actions_not_list(self):
        transitions = [{"id": "t", "from": "idle", "to": "walk", "event": "go", "actions": {}}]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "go"}, "actions")

    def test_validation_happens_before_actions(self):
        # An invalid later transition must abort the call even though an
        # earlier transition would have matched and mutated the blackboard.
        transitions = [
            {"id": "t1", "from": "idle", "to": "walk", "event": "go", "actions": [{"op": "set", "key": "x", "value": 1}]},
            {"id": "t2", "from": "ghost", "to": "walk", "event": "go"},
        ]
        with self.assertRaises(ValueError):
            self.service.step_state_machine({"machine": machine(transitions=transitions), "event": "go"})


if __name__ == "__main__":
    unittest.main()
