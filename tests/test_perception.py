import copy
import math
import unittest

from npcmind.service import PerceptionError, Service


def entity(id, kind="item", last_seen=0.0, confidence=1.0, x=0.0, y=0.0, data=None, **extra):
    record = {
        "id": id,
        "kind": kind,
        "last_seen": last_seen,
        "confidence": confidence,
        "position": {"x": x, "y": y},
    }
    if data is not None:
        record["data"] = data
    record.update(extra)
    return record


def observation(id, kind="item", confidence=1.0, x=0.0, y=0.0, data=None, **extra):
    record = {"id": id, "kind": kind, "confidence": confidence, "position": {"x": x, "y": y}}
    if data is not None:
        record["data"] = data
    record.update(extra)
    return record


def request(**overrides):
    base = {"now": 10.0, "retention": 5.0, "memory": [], "observations": []}
    base.update(overrides)
    return base


class UpdatePerceptionTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_empty_memory_and_observations(self):
        result = self.service.update_perception(request())
        self.assertEqual(result["status"], "UPDATED")
        self.assertEqual(result["memory"], [])
        self.assertEqual(result["seen"], [])
        self.assertEqual(result["forgotten"], [])

    def test_new_observations_append_in_order_with_last_seen_now(self):
        result = self.service.update_perception(
            request(
                observations=[
                    observation("a", x=1.0, y=2.0),
                    observation("b", kind="npc", confidence=0.5, data={"v": 1}),
                ]
            )
        )
        self.assertEqual([r["id"] for r in result["memory"]], ["a", "b"])
        first = result["memory"][0]
        self.assertEqual(first["last_seen"], 10.0)
        self.assertEqual(first["position"], {"x": 1.0, "y": 2.0})
        self.assertEqual(first["data"], {})
        self.assertEqual(set(first.keys()), {"id", "kind", "last_seen", "confidence", "position", "data"})
        second = result["memory"][1]
        self.assertEqual(second["kind"], "npc")
        self.assertEqual(second["confidence"], 0.5)
        self.assertEqual(second["data"], {"v": 1})
        self.assertEqual(result["seen"], ["a", "b"])
        self.assertEqual(result["forgotten"], [])

    def test_reobserved_entity_keeps_slot_and_replaces_fields(self):
        result = self.service.update_perception(
            request(
                now=10.0,
                memory=[
                    entity("old", kind="a", last_seen=1.0, confidence=0.1, x=0.0, y=0.0),
                    entity("keep", kind="b", last_seen=9.0, confidence=0.2, x=5.0, y=5.0),
                    entity("gone", kind="c", last_seen=2.0, confidence=0.3, x=9.0, y=9.0),
                ],
                observations=[
                    observation("new", kind="d"),
                    observation("old", kind="a2", confidence=0.9, x=3.0, y=4.0, data={"z": True}),
                ],
            )
        )
        records = result["memory"]
        self.assertEqual([r["id"] for r in records], ["old", "keep", "new"])
        old = records[0]
        self.assertEqual(old["kind"], "a2")
        self.assertEqual(old["confidence"], 0.9)
        self.assertEqual(old["position"], {"x": 3.0, "y": 4.0})
        self.assertEqual(old["data"], {"z": True})
        self.assertEqual(old["last_seen"], 10.0)
        keep = records[1]
        self.assertEqual(keep["kind"], "b")
        self.assertEqual(keep["last_seen"], 9.0)
        self.assertEqual(keep["position"], {"x": 5.0, "y": 5.0})
        self.assertEqual(result["seen"], ["new", "old"])
        self.assertEqual(result["forgotten"], ["gone"])

    def test_retention_boundary_is_inclusive(self):
        result = self.service.update_perception(
            request(
                now=10.0,
                retention=5.0,
                memory=[
                    entity("at_boundary", last_seen=5.0),
                    entity("just_young", last_seen=5.5),
                ],
            )
        )
        self.assertEqual([r["id"] for r in result["memory"]], ["just_young"])
        self.assertEqual(result["forgotten"], ["at_boundary"])

    def test_reobserved_older_than_retention_is_not_forgotten(self):
        result = self.service.update_perception(
            request(
                now=100.0,
                retention=1.0,
                memory=[entity("a", last_seen=0.0)],
                observations=[observation("a", kind="fresh")],
            )
        )
        self.assertEqual([r["id"] for r in result["memory"]], ["a"])
        self.assertEqual(result["seen"], ["a"])
        self.assertEqual(result["forgotten"], [])
        self.assertEqual(result["memory"][0]["kind"], "fresh")
        self.assertEqual(result["memory"][0]["last_seen"], 100.0)

    def test_retained_records_get_explicit_data(self):
        result = self.service.update_perception(
            request(memory=[{"id": "a", "kind": "k", "last_seen": 9.0,
                             "confidence": 1, "position": {"x": 0, "y": 0}}])
        )
        self.assertEqual(result["memory"][0]["data"], {})

    def test_integer_numbers_accepted(self):
        result = self.service.update_perception(
            request(now=10, retention=5, memory=[entity("a", last_seen=9, x=1, y=2)])
        )
        self.assertEqual(result["memory"][0]["position"], {"x": 1, "y": 2})

    def test_zero_now_allowed(self):
        result = self.service.update_perception(
            request(now=0, retention=1, memory=[entity("a", last_seen=0)])
        )
        self.assertEqual(result["memory"][0]["last_seen"], 0)

    def test_confidence_endpoints_allowed(self):
        result = self.service.update_perception(
            request(observations=[observation("a", confidence=0), observation("b", confidence=1)])
        )
        self.assertEqual([r["confidence"] for r in result["memory"]], [0, 1])

    def test_request_is_not_mutated(self):
        payload = request(
            memory=[entity("a", last_seen=1.0, data={"n": 1})],
            observations=[observation("a", data={"n": 2})],
        )
        snapshot = copy.deepcopy(payload)
        self.service.update_perception(payload)
        self.assertEqual(payload, snapshot)

    def test_output_does_not_alias_input_data(self):
        obs = observation("a", data={"nested": {"v": [1, 2]}})
        result = self.service.update_perception(request(observations=[obs]))
        result["memory"][0]["data"]["nested"]["v"].append(3)
        self.assertEqual(obs["data"]["nested"]["v"], [1, 2])

    def test_no_state_between_calls(self):
        first = self.service.update_perception(request(observations=[observation("a")]))
        second = self.service.update_perception(request())
        self.assertEqual([r["id"] for r in first["memory"]], ["a"])
        self.assertEqual(second["memory"], [])

    def test_duplicate_memory_id_rejected(self):
        with self.assertRaises(PerceptionError):
            self.service.update_perception(
                request(memory=[entity("a", last_seen=1.0), entity("a", last_seen=2.0)])
            )

    def test_duplicate_observation_id_rejected(self):
        with self.assertRaises(PerceptionError):
            self.service.update_perception(
                request(observations=[observation("a"), observation("a")])
            )

    def test_observation_id_matching_memory_is_not_duplicate_error(self):
        result = self.service.update_perception(
            request(memory=[entity("a", last_seen=9.0)], observations=[observation("a")])
        )
        self.assertEqual([r["id"] for r in result["memory"]], ["a"])

    def test_invalid_inputs(self):
        valid_entity = entity("a", last_seen=1.0)
        valid_obs = observation("b")

        cases = [
            request(now=-1.0),
            request(now=math.inf),
            request(now=math.nan),
            request(now=True),
            request(now="10"),
            request(now=None),
            request(retention=0),
            request(retention=-1.0),
            request(retention=math.inf),
            request(retention=True),
            request(memory=None),
            request(observations=None),
            request(memory=[{"id": "", "kind": "k", "last_seen": 1.0,
                             "confidence": 1, "position": {"x": 0, "y": 0}}]),
            request(memory=[{"id": 1, "kind": "k", "last_seen": 1.0,
                             "confidence": 1, "position": {"x": 0, "y": 0}}]),
            request(memory=[entity("a", last_seen=1.0, kind="")]),
            request(memory=[entity("a", last_seen=1.0, kind=None)]),
            request(memory=[{"id": "a", "kind": "k", "confidence": 1, "position": {"x": 0, "y": 0}}]),
            request(memory=[{"id": "a", "kind": "k", "last_seen": 1.0, "position": {"x": 0, "y": 0}}]),
            request(memory=[entity("a", last_seen=1.0, confidence=1.5)]),
            request(memory=[entity("a", last_seen=1.0, confidence=-0.1)]),
            request(memory=[entity("a", last_seen=1.0, confidence=True)]),
            request(memory=[entity("a", last_seen=1.0, confidence=math.nan)]),
            request(memory=[entity("a", last_seen=11.0)]),
            request(memory=[entity("a", last_seen=math.inf)]),
            request(memory=[entity("a", last_seen=-0.1)]),
            request(memory=[entity("a", last_seen=True)]),
            request(memory=[entity("a", last_seen=1.0, x=math.inf)]),
            request(memory=[entity("a", last_seen=1.0, x=True)]),
            request(memory=[{"id": "a", "kind": "k", "last_seen": 1.0,
                            "confidence": 1, "position": {"x": 0}}]),
            request(memory=[{"id": "a", "kind": "k", "last_seen": 1.0,
                            "confidence": 1, "position": "bad"}]),
            request(memory=[entity("a", last_seen=1.0, data=object())]),
            request(memory=[entity("a", last_seen=1.0, data=[])]),
            request(memory=[entity("a", last_seen=1.0, data={"v": math.inf})]),
            request(memory=[entity("a", last_seen=1.0, data={"v": object()})]),
            request(observations=[{"id": "b", "confidence": 1, "position": {"x": 0, "y": 0}}]),
            request(observations=[{"id": "b", "kind": "k", "confidence": 1}]),
            request(observations=["not-an-object"]),
            request(memory=["not-an-object"]),
            "not-an-object",
            request(memory=[valid_entity, {"id": "", "kind": "k", "last_seen": 1.0,
                                           "confidence": 1, "position": {"x": 0, "y": 0}}]),
            request(memory=[valid_entity], observations=[valid_obs, observation("b")]),
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    self.service.update_perception(case)

    def test_failed_validation_leaves_no_partial_result_and_no_mutation(self):
        payload = request(
            memory=[entity("good", last_seen=9.0), entity("bad", last_seen=1.0, confidence=2.0)],
            observations=[observation("good")],
        )
        snapshot = copy.deepcopy(payload)
        with self.assertRaises(PerceptionError):
            self.service.update_perception(payload)
        self.assertEqual(payload, snapshot)
        # service still stateless and usable after a rejected call
        follow = self.service.update_perception(request(observations=[observation("z")]))
        self.assertEqual([r["id"] for r in follow["memory"]], ["z"])


if __name__ == "__main__":
    unittest.main()
