import unittest

from npcmind.service import Service


def machine(**overrides):
    definition = {
        "states": [{"id": "idle"}, {"id": "alert"}, {"id": "gone"}],
        "initial": "idle",
        "transitions": [
            {
                "id": "t-see",
                "from": "idle",
                "to": "alert",
                "event": "see",
                "actions": [{"op": "set", "key": "seen", "value": True}],
            },
            {
                "id": "t-hear",
                "from": "idle",
                "to": "alert",
                "event": "hear",
                "condition": {"op": "equals", "key": "loud", "value": True},
            },
            {
                "id": "t-calm",
                "from": "alert",
                "to": "idle",
                "event": "calm",
                "actions": [{"op": "delete", "key": "seen"}],
            },
            {
                "id": "t-leave",
                "from": "alert",
                "to": "gone",
                "event": "leave",
            },
        ],
    }
    definition.update(overrides)
    return definition


class StepStateMachineTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def step(self, request):
        return self.service.step_state_machine(request)

    def test_defaults_to_initial_state(self):
        result = self.step({"machine": machine(), "event": "see"})
        self.assertEqual(result["previous_state"], "idle")
        self.assertEqual(result["state"], "alert")
        self.assertEqual(result["transition"], "t-see")
        self.assertEqual(result["blackboard"], {"seen": True})
        self.assertEqual(result["trace"], [{"id": "t-see", "condition": True}])

    def test_explicit_current_state(self):
        result = self.step({"machine": machine(), "event": "leave", "current_state": "alert"})
        self.assertEqual(result["previous_state"], "alert")
        self.assertEqual(result["state"], "gone")
        self.assertEqual(result["transition"], "t-leave")

    def test_no_match_keeps_state_and_blackboard(self):
        board = {"k": 1}
        result = self.step({"machine": machine(), "event": "see", "current_state": "gone", "blackboard": board})
        self.assertEqual(result["state"], "gone")
        self.assertEqual(result["transition"], None)
        self.assertEqual(result["blackboard"], {"k": 1})
        self.assertEqual(result["trace"], [])

    def test_no_match_keeps_examined_candidates_in_trace(self):
        result = self.step({"machine": machine(), "event": "hear"})
        self.assertEqual(result["state"], "idle")
        self.assertEqual(result["transition"], None)
        self.assertEqual(result["trace"], [{"id": "t-hear", "condition": False}])

    def test_condition_gates_transition(self):
        request = {"machine": machine(), "event": "hear", "blackboard": {"loud": True}}
        result = self.step(request)
        self.assertEqual(result["state"], "alert")
        self.assertEqual(result["transition"], "t-hear")

    def test_condition_type_sensitive(self):
        # JSON type-sensitive: 1 does not equal true
        request = {"machine": machine(), "event": "hear", "blackboard": {"loud": 1}}
        result = self.step(request)
        self.assertEqual(result["state"], "idle")
        self.assertEqual(result["transition"], None)

    def test_first_matching_transition_wins(self):
        definition = machine()
        definition["transitions"] = [
            {"id": "t-first", "from": "idle", "to": "alert", "event": "go",
             "condition": {"op": "exists", "key": "missing"}},
            {"id": "t-second", "from": "idle", "to": "gone", "event": "go"},
            {"id": "t-third", "from": "idle", "to": "alert", "event": "go"},
        ]
        result = self.step({"machine": definition, "event": "go"})
        self.assertEqual(result["state"], "gone")
        self.assertEqual(result["transition"], "t-second")
        self.assertEqual(
            result["trace"],
            [{"id": "t-first", "condition": False}, {"id": "t-second", "condition": True}],
        )

    def test_actions_run_in_order(self):
        definition = machine()
        definition["transitions"] = [
            {"id": "t", "from": "idle", "to": "alert", "event": "go", "actions": [
                {"op": "set", "key": "a", "value": 1},
                {"op": "set", "key": "b", "value": 2},
                {"op": "delete", "key": "a"},
            ]},
        ]
        result = self.step({"machine": definition, "event": "go"})
        self.assertEqual(result["blackboard"], {"b": 2})

    def test_inputs_not_mutated(self):
        definition = machine()
        board = {"loud": True}
        request = {"machine": definition, "event": "see", "blackboard": board}
        self.step(request)
        self.assertEqual(board, {"loud": True})
        self.assertEqual(definition["transitions"][0]["actions"], [{"op": "set", "key": "seen", "value": True}])

    def test_default_blackboard_is_empty_object(self):
        result = self.step({"machine": machine(), "event": "see"})
        self.assertEqual(result["blackboard"], {"seen": True})


class StepStateMachineErrorsTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def assert_invalid(self, request, *fragments):
        with self.assertRaises(ValueError) as ctx:
            self.service.step_state_machine(request)
        for fragment in fragments:
            self.assertIn(fragment, str(ctx.exception))

    def test_request_not_object(self):
        self.assert_invalid([1, 2], "object")

    def test_missing_machine(self):
        self.assert_invalid({"event": "go"}, "machine")

    def test_machine_not_object(self):
        self.assert_invalid({"machine": [], "event": "go"}, "machine")

    def test_event_not_string(self):
        self.assert_invalid({"machine": machine(), "event": 3}, "event")

    def test_event_missing(self):
        self.assert_invalid({"machine": machine()}, "event")

    def test_states_empty(self):
        self.assert_invalid({"machine": machine(states=[]), "event": "go"}, "states")

    def test_states_not_list(self):
        self.assert_invalid({"machine": machine(states={}), "event": "go"}, "states")

    def test_state_not_object(self):
        self.assert_invalid({"machine": machine(states=["idle"]), "event": "go"}, "states[0]")

    def test_state_empty_id(self):
        self.assert_invalid({"machine": machine(states=[{"id": ""}]), "event": "go"}, "id")

    def test_duplicate_state_id(self):
        self.assert_invalid(
            {"machine": machine(states=[{"id": "a"}, {"id": "a"}], initial="a"), "event": "go"},
            "duplicate",
        )

    def test_initial_unknown_state(self):
        self.assert_invalid({"machine": machine(initial="nowhere"), "event": "go"}, "initial")

    def test_current_state_unknown(self):
        self.assert_invalid(
            {"machine": machine(), "event": "go", "current_state": "nowhere"}, "current_state"
        )

    def test_transitions_not_list(self):
        self.assert_invalid({"machine": machine(transitions={}), "event": "go"}, "transitions")

    def test_duplicate_transition_id(self):
        transitions = [
            {"id": "dup", "from": "idle", "to": "alert", "event": "a"},
            {"id": "dup", "from": "idle", "to": "alert", "event": "b"},
        ]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "a"}, "dup")

    def test_from_unknown_state(self):
        transitions = [{"id": "t", "from": "ghost", "to": "alert", "event": "a"}]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "a"}, "from")

    def test_to_unknown_state(self):
        transitions = [{"id": "t", "from": "idle", "to": "ghost", "event": "a"}]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "a"}, "to")

    def test_transition_event_not_string(self):
        transitions = [{"id": "t", "from": "idle", "to": "alert", "event": 1}]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "a"}, "event")

    def test_condition_unknown_op(self):
        transitions = [{"id": "t", "from": "idle", "to": "alert", "event": "a",
                        "condition": {"op": "gt", "key": "k", "value": 1}}]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "a"}, "gt")

    def test_condition_missing_value(self):
        transitions = [{"id": "t", "from": "idle", "to": "alert", "event": "a",
                        "condition": {"op": "equals", "key": "k"}}]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "a"}, "value")

    def test_action_unknown_op(self):
        transitions = [{"id": "t", "from": "idle", "to": "alert", "event": "a",
                        "actions": [{"op": "status", "status": "SUCCESS"}]}]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "a"}, "status")

    def test_action_set_missing_value(self):
        transitions = [{"id": "t", "from": "idle", "to": "alert", "event": "a",
                        "actions": [{"op": "set", "key": "k"}]}]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "a"}, "value")

    def test_blackboard_not_object(self):
        self.assert_invalid({"machine": machine(), "event": "go", "blackboard": []}, "blackboard")

    def test_validation_happens_before_actions(self):
        # An invalid later transition must prevent earlier actions from running;
        # the call raises and produces no result at all.
        transitions = [
            {"id": "t-ok", "from": "idle", "to": "alert", "event": "go",
             "actions": [{"op": "set", "key": "x", "value": 1}]},
            {"id": "t-bad", "from": "idle", "to": "ghost", "event": "go"},
        ]
        self.assert_invalid({"machine": machine(transitions=transitions), "event": "go"}, "ghost")


if __name__ == "__main__":
    unittest.main()
