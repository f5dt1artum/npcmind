import copy
import math
import unittest

from npcmind.service import AttentionError, Service


def entity(id, kind="npc", last_seen=0.0, confidence=1.0, x=0.0, y=0.0, data=None, **extra):
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


def request(**overrides):
    base = {
        "observer": {
            "position": {"x": 0.0, "y": 0.0},
            "forward": {"x": 1.0, "y": 0.0},
            "max_distance": 10.0,
            "field_of_view_degrees": 180.0,
        },
        "now": 10.0,
        "memory_horizon": 5.0,
        "memory": [],
    }
    base.update(overrides)
    return base


def observer(**overrides):
    base = {
        "position": {"x": 0.0, "y": 0.0},
        "forward": {"x": 1.0, "y": 0.0},
        "max_distance": 10.0,
        "field_of_view_degrees": 180.0,
    }
    base.update(overrides)
    return base


class SelectAttentionTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_empty_memory_yields_no_target(self):
        result = self.service.select_attention(request())
        self.assertEqual(result["status"], "NO_TARGET")
        self.assertIsNone(result["selected"])
        self.assertIsNone(result["score"])
        self.assertEqual(result["evaluations"], [])

    def test_visible_entity_is_scored_and_selected(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, x=5.0, y=0.0, data={"threat": 0.5})])
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "a")
        self.assertEqual(result["score"], 0.25)
        (evaluation,) = result["evaluations"]
        self.assertEqual(evaluation["id"], "a")
        self.assertTrue(evaluation["visible"])
        self.assertEqual(evaluation["distance"], 5.0)
        self.assertEqual(evaluation["threat"], 0.5)
        self.assertEqual(evaluation["proximity"], 0.5)
        self.assertEqual(evaluation["freshness"], 1.0)
        self.assertEqual(evaluation["score"], 0.25)

    def test_threat_defaults_to_zero(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, x=5.0, y=0.0)])
        )
        self.assertEqual(result["status"], "NO_TARGET")
        (evaluation,) = result["evaluations"]
        self.assertTrue(evaluation["visible"])
        self.assertEqual(evaluation["threat"], 0)
        self.assertEqual(evaluation["score"], 0)

    def test_coincident_entity_is_visible_with_proximity_one(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, x=0.0, y=0.0, data={"threat": 1})])
        )
        (evaluation,) = result["evaluations"]
        self.assertTrue(evaluation["visible"])
        self.assertEqual(evaluation["distance"], 0.0)
        self.assertEqual(evaluation["proximity"], 1.0)
        self.assertEqual(result["status"], "SELECTED")

    def test_beyond_max_distance_is_invisible(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, x=10.5, y=0.0, data={"threat": 1})])
        )
        (evaluation,) = result["evaluations"]
        self.assertFalse(evaluation["visible"])
        self.assertEqual(evaluation["proximity"], 0.0)
        self.assertEqual(evaluation["score"], 0.0)
        self.assertEqual(result["status"], "NO_TARGET")

    def test_distance_boundary_is_visible_with_zero_proximity(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, x=10.0, y=0.0, data={"threat": 1})])
        )
        (evaluation,) = result["evaluations"]
        self.assertTrue(evaluation["visible"])
        self.assertEqual(evaluation["proximity"], 0.0)
        self.assertEqual(evaluation["score"], 0.0)

    def test_outside_field_of_view_is_invisible(self):
        result = self.service.select_attention(
            request(
                observer=observer(field_of_view_degrees=90.0),
                memory=[entity("a", last_seen=10.0, x=0.0, y=5.0, data={"threat": 1})],
            )
        )
        (evaluation,) = result["evaluations"]
        self.assertFalse(evaluation["visible"])
        self.assertEqual(evaluation["score"], 0.0)

    def test_field_of_view_boundary_is_visible(self):
        result = self.service.select_attention(
            request(
                observer=observer(field_of_view_degrees=90.0),
                memory=[entity("a", last_seen=10.0, x=5.0, y=5.0, data={"threat": 1})],
            )
        )
        (evaluation,) = result["evaluations"]
        self.assertTrue(evaluation["visible"])
        self.assertAlmostEqual(evaluation["distance"], math.hypot(5.0, 5.0))

    def test_full_circle_field_of_view_ignores_orientation(self):
        result = self.service.select_attention(
            request(
                observer=observer(field_of_view_degrees=360.0),
                memory=[entity("a", last_seen=10.0, x=-5.0, y=0.0, data={"threat": 1})],
            )
        )
        (evaluation,) = result["evaluations"]
        self.assertTrue(evaluation["visible"])
        self.assertEqual(result["status"], "SELECTED")

    def test_narrow_field_of_view_excludes_behind(self):
        result = self.service.select_attention(
            request(
                observer=observer(field_of_view_degrees=60.0),
                memory=[entity("a", last_seen=10.0, x=-5.0, y=0.0, data={"threat": 1})],
            )
        )
        self.assertFalse(result["evaluations"][0]["visible"])
        self.assertEqual(result["status"], "NO_TARGET")

    def test_freshness_decays_with_age(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=7.5, x=0.0, y=0.0, data={"threat": 1})])
        )
        (evaluation,) = result["evaluations"]
        self.assertEqual(evaluation["freshness"], 0.5)
        self.assertEqual(evaluation["score"], 0.5)

    def test_freshness_floors_at_zero(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=1.0, x=0.0, y=0.0, data={"threat": 1})])
        )
        (evaluation,) = result["evaluations"]
        self.assertEqual(evaluation["freshness"], 0.0)
        self.assertEqual(evaluation["score"], 0.0)
        self.assertEqual(result["status"], "NO_TARGET")

    def test_highest_positive_score_wins(self):
        result = self.service.select_attention(
            request(
                memory=[
                    entity("near_weak", last_seen=10.0, x=2.0, y=0.0, data={"threat": 0.5}),
                    entity("far_strong", last_seen=10.0, x=8.0, y=0.0, data={"threat": 1.0}),
                ]
            )
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "near_weak")
        self.assertEqual(result["score"], 0.4)
        self.assertEqual([e["id"] for e in result["evaluations"]], ["near_weak", "far_strong"])

    def test_tie_goes_to_earlier_record(self):
        result = self.service.select_attention(
            request(
                memory=[
                    entity("first", last_seen=10.0, x=5.0, y=0.0, data={"threat": 1}),
                    entity("second", last_seen=10.0, x=0.0, y=5.0, data={"threat": 1}),
                ]
            )
        )
        self.assertEqual(result["selected"], "first")
        self.assertEqual(result["score"], 0.5)

    def test_zero_confidence_scores_zero(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, confidence=0.0, data={"threat": 1})])
        )
        self.assertEqual(result["status"], "NO_TARGET")
        self.assertTrue(result["evaluations"][0]["visible"])

    def test_integer_numbers_accepted(self):
        result = self.service.select_attention(
            request(
                observer=observer(
                    position={"x": 0, "y": 0},
                    forward={"x": 1, "y": 0},
                    max_distance=10,
                    field_of_view_degrees=180,
                ),
                now=10,
                memory_horizon=5,
                memory=[entity("a", last_seen=10, x=5, y=0, data={"threat": 1})],
            )
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "a")
        self.assertEqual(result["score"], 0.5)

    def test_request_is_not_mutated(self):
        payload = request(memory=[entity("a", last_seen=9.0, x=1.0, data={"threat": 0.5})])
        snapshot = copy.deepcopy(payload)
        self.service.select_attention(payload)
        self.assertEqual(payload, snapshot)

    def test_no_state_between_calls(self):
        first = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, data={"threat": 1})])
        )
        second = self.service.select_attention(request())
        self.assertEqual(first["status"], "SELECTED")
        self.assertEqual(second["status"], "NO_TARGET")
        self.assertEqual(second["evaluations"], [])

    def test_duplicate_memory_id_rejected(self):
        with self.assertRaises(AttentionError):
            self.service.select_attention(
                request(memory=[entity("a", last_seen=1.0), entity("a", last_seen=2.0)])
            )

    def test_invalid_inputs(self):
        valid = entity("a", last_seen=1.0)
        cases = [
            "not-an-object",
            request(observer=None),
            request(observer="bad"),
            request(observer=observer(position=None)),
            request(observer=observer(position={"x": 0})),
            request(observer=observer(position={"x": 0, "y": True})),
            request(observer=observer(position={"x": math.inf, "y": 0})),
            request(observer=observer(forward={"x": 0.0, "y": 0.0})),
            request(observer=observer(forward={"x": math.nan, "y": 0.0})),
            request(observer=observer(forward="north")),
            request(observer=observer(max_distance=0)),
            request(observer=observer(max_distance=-1.0)),
            request(observer=observer(max_distance=math.inf)),
            request(observer=observer(max_distance=True)),
            request(observer=observer(field_of_view_degrees=0)),
            request(observer=observer(field_of_view_degrees=-90)),
            request(observer=observer(field_of_view_degrees=361)),
            request(observer=observer(field_of_view_degrees=math.inf)),
            request(observer=observer(field_of_view_degrees=True)),
            request(now=-1.0),
            request(now=math.inf),
            request(now=math.nan),
            request(now=True),
            request(memory_horizon=0),
            request(memory_horizon=-1.0),
            request(memory_horizon=math.inf),
            request(memory_horizon=True),
            request(memory=None),
            request(memory="bad"),
            request(memory=["not-an-object"]),
            request(memory=[{"id": "", "kind": "k", "last_seen": 1.0,
                             "confidence": 1, "position": {"x": 0, "y": 0}}]),
            request(memory=[entity("a", last_seen=1.0, kind="")]),
            request(memory=[entity("a", last_seen=1.0, confidence=1.5)]),
            request(memory=[entity("a", last_seen=1.0, confidence=True)]),
            request(memory=[entity("a", last_seen=11.0)]),
            request(memory=[entity("a", last_seen=-0.1)]),
            request(memory=[entity("a", last_seen=1.0, x=math.inf)]),
            request(memory=[entity("a", last_seen=1.0, data=[])]),
            request(memory=[entity("a", last_seen=1.0, data={"v": math.inf})]),
            request(memory=[entity("a", last_seen=1.0, data={"threat": 1.5})]),
            request(memory=[entity("a", last_seen=1.0, data={"threat": -0.1})]),
            request(memory=[entity("a", last_seen=1.0, data={"threat": True})]),
            request(memory=[entity("a", last_seen=1.0, data={"threat": math.nan})]),
            request(memory=[entity("a", last_seen=1.0, data={"threat": "high"})]),
            request(memory=[valid, valid]),
            request(memory=[valid, {"id": "", "kind": "k", "last_seen": 1.0,
                                    "confidence": 1, "position": {"x": 0, "y": 0}}]),
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    self.service.select_attention(case)

    def test_failed_validation_leaves_no_partial_result_and_no_mutation(self):
        payload = request(
            memory=[entity("good", last_seen=9.0), entity("bad", last_seen=1.0, confidence=2.0)]
        )
        snapshot = copy.deepcopy(payload)
        with self.assertRaises(AttentionError):
            self.service.select_attention(payload)
        self.assertEqual(payload, snapshot)
        # service still stateless and usable after a rejected call
        follow = self.service.select_attention(
            request(memory=[entity("z", last_seen=10.0, data={"threat": 1})])
        )
        self.assertEqual(follow["selected"], "z")


if __name__ == "__main__":
    unittest.main()
