import copy
import math
import unittest

from npcmind.service import PerceptionError, Service


def entity(entity_id, kind="npc", confidence=0.5, position=None, data=..., last_seen=0):
    record = {"id": entity_id, "kind": kind, "confidence": confidence,
              "position": position if position is not None else {"x": 1, "y": 2}}
    if last_seen is not ...:
        record["last_seen"] = last_seen
    if data is not ...:
        record["data"] = data
    return record


def observation(entity_id, **overrides):
    record = entity(entity_id, last_seen=...)
    record.update(overrides)
    return record


def request(**overrides):
    base = {"now": 10.0, "retention": 5.0, "memory": [], "observations": []}
    base.update(overrides)
    return base


class UpdatePerceptionTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_new_observations_append_in_order(self):
        result = self.service.update_perception(
            request(observations=[observation("a"), observation("b", kind="item")])
        )
        self.assertEqual(result["status"], "UPDATED")
        self.assertEqual([r["id"] for r in result["memory"]], ["a", "b"])
        self.assertEqual(result["seen"], ["a", "b"])
        self.assertEqual(result["forgotten"], [])
        first = result["memory"][0]
        self.assertEqual(first["last_seen"], 10.0)
        self.assertEqual(first["data"], {})
        self.assertEqual(first["position"], {"x": 1, "y": 2})
        self.assertEqual(result["memory"][1]["kind"], "item")

    def test_output_records_have_memory_structure_with_data(self):
        result = self.service.update_perception(
            request(observations=[{"id": "a", "kind": "npc", "confidence": 1,
                                   "position": {"x": 0, "y": 0}, "data": {"k": [1, True, None]}}])
        )
        record = result["memory"][0]
        self.assertEqual(
            set(record), {"id", "kind", "last_seen", "confidence", "position", "data"}
        )
        self.assertEqual(record["data"], {"k": [1, True, None]})

    def test_reobserved_keeps_position_and_replaces_fields(self):
        result = self.service.update_perception(
            request(
                memory=[
                    entity("old", kind="npc", confidence=0.1, position={"x": 9, "y": 9},
                           data={"v": 1}, last_seen=1.0),
                    entity("mid", last_seen=8.0),
                ],
                observations=[
                    observation("old", kind="enemy", confidence=0.9,
                                position={"x": 3, "y": 4}, data={"v": 2})
                ],
            )
        )
        self.assertEqual([r["id"] for r in result["memory"]], ["old", "mid"])
        old = result["memory"][0]
        self.assertEqual(old["kind"], "enemy")
        self.assertEqual(old["confidence"], 0.9)
        self.assertEqual(old["position"], {"x": 3, "y": 4})
        self.assertEqual(old["data"], {"v": 2})
        self.assertEqual(old["last_seen"], 10.0)
        self.assertEqual(result["seen"], ["old"])
        self.assertEqual(result["forgotten"], [])

    def test_forgotten_when_age_meets_retention(self):
        result = self.service.update_perception(
            request(
                now=10,
                retention=5,
                memory=[
                    entity("expired", last_seen=5.0),   # now - last_seen == retention
                    entity("stale", last_seen=1.0),
                    entity("fresh", last_seen=5.5),
                    entity("seen_again", last_seen=0.0),
                ],
                observations=[observation("seen_again")],
            )
        )
        self.assertEqual([r["id"] for r in result["memory"]], ["fresh", "seen_again"])
        self.assertEqual(result["forgotten"], ["expired", "stale"])
        self.assertEqual(result["seen"], ["seen_again"])
        fresh = result["memory"][0]
        self.assertEqual(fresh["last_seen"], 5.5)
        self.assertEqual(fresh["kind"], "npc")
        self.assertEqual(fresh["confidence"], 0.5)
        self.assertEqual(fresh["position"], {"x": 1, "y": 2})
        self.assertEqual(fresh["data"], {})

    def test_reobserved_never_forgotten_even_when_stale(self):
        result = self.service.update_perception(
            request(
                memory=[entity("a", last_seen=0.0), entity("b", last_seen=0.0)],
                observations=[observation("a")],
            )
        )
        # b ages out (10 - 0 >= 5); a is re-observed and survives.
        self.assertEqual([r["id"] for r in result["memory"]], ["a"])
        self.assertEqual(result["forgotten"], ["b"])

    def test_new_entities_interleave_after_old(self):
        result = self.service.update_perception(
            request(
                memory=[entity("a", last_seen=8.0), entity("c", last_seen=8.0)],
                observations=[observation("c"), observation("b"), observation("a"),
                              observation("d")],
            )
        )
        self.assertEqual([r["id"] for r in result["memory"]], ["a", "c", "b", "d"])
        self.assertEqual(result["seen"], ["c", "b", "a", "d"])

    def test_request_and_nested_objects_are_not_mutated(self):
        payload = request(
            memory=[entity("a", last_seen=6.0)],
            observations=[observation("a", data={"nested": {"x": 1}})],
        )
        snapshot = copy.deepcopy(payload)
        self.service.update_perception(payload)
        self.assertEqual(payload, snapshot)

    def test_no_state_retained_between_calls(self):
        first = self.service.update_perception(request(observations=[observation("a")]))
        self.assertEqual([r["id"] for r in first["memory"]], ["a"])
        second = self.service.update_perception(request(memory=[], observations=[]))
        self.assertEqual(second["memory"], [])

    def test_returned_memory_mutation_does_not_affect_input(self):
        obs = observation("a", data={"v": 1})
        result = self.service.update_perception(request(observations=[obs]))
        result["memory"][0]["data"]["v"] = 99
        result["memory"][0]["position"]["x"] = 99
        self.assertEqual(obs["data"], {"v": 1})
        self.assertEqual(obs["position"], {"x": 1, "y": 2})

    def test_missing_memory_and_observations_default_to_empty(self):
        result = self.service.update_perception({"now": 0, "retention": 1})
        self.assertEqual(result, {"status": "UPDATED", "memory": [], "seen": [], "forgotten": []})

    # --- validation -------------------------------------------------------

    def assert_invalid(self, payload):
        with self.assertRaises(PerceptionError):
            self.service.update_perception(payload)

    def test_invalid_inputs(self):
        valid_entity = entity("a", last_seen=1.0)
        valid_obs = observation("b")
        cases = [
            {},
            {"retention": 1},
            {"now": -1, "retention": 1},
            {"now": True, "retention": 1},
            {"now": 1, "retention": 0},
            {"now": 1, "retention": -2.0},
            {"now": 1, "retention": float("inf")},
            {"now": float("nan"), "retention": 1},
            {"now": "1", "retention": 1},
            {"now": 1, "retention": 1, "memory": {}},
            {"now": 1, "retention": 1, "observations": {}},
            {"now": 1, "retention": 1, "memory": [{"id": "a"}]},
            {"now": 1, "retention": 1, "memory": ["nope"]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": ""}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "kind": ""}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "kind": 5}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "last_seen": 2}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "last_seen": -0.1}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "last_seen": True}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "last_seen": float("inf")}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "confidence": 1.1}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "confidence": -0.01}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "confidence": True}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "position": {"x": 1}}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "position": {"x": "1", "y": 0}}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "position": {"x": 1, "y": float("nan")}}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "position": [0, 0]}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "data": []}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "data": {"bad": object()}}]},
            {"now": 1, "retention": 1,
             "memory": [{**valid_entity, "id": "x", "data": {"bad": float("inf")}}]},
            {"now": 1, "retention": 1,
             "memory": [valid_entity, entity("a", last_seen=1.0)]},
            {"now": 1, "retention": 1, "observations": ["nope"]},
            {"now": 1, "retention": 1,
             "observations": [{**valid_obs, "id": ""}]},
            {"now": 1, "retention": 1,
             "observations": [valid_obs, observation("b")]},
            "not an object",
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assert_invalid(case)

    def test_validation_error_is_value_error(self):
        with self.assertRaises(ValueError):
            self.service.update_perception(request(memory=[{"id": "bad"}]))

    def test_invalid_request_leaves_no_partial_result(self):
        # An observation batch with a duplicate id must not merge anything.
        with self.assertRaises(PerceptionError):
            self.service.update_perception(
                request(memory=[entity("a", last_seen=9.0)],
                        observations=[observation("b"), observation("b")])
            )


if __name__ == "__main__":
    unittest.main()
