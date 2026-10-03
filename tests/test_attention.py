import copy
import math
import unittest

from npcmind.service import AttentionError, Service


def entity(id, last_seen=0.0, confidence=1.0, x=0.0, y=0.0, kind="npc", data=None, **extra):
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
            "field_of_view_degrees": 90.0,
        },
        "now": 10.0,
        "memory_horizon": 20.0,
        "memory": [],
    }
    base.update(overrides)
    return base


class SelectAttentionTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_empty_memory_is_no_target(self):
        result = self.service.select_attention(request())
        self.assertEqual(result["status"], "NO_TARGET")
        self.assertIsNone(result["selected"])
        self.assertIsNone(result["score"])
        self.assertEqual(result["evaluations"], [])

    def test_selects_entity_straight_ahead(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, x=5.0, y=0.0, data={"threat": 1})])
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "a")
        evaluation = result["evaluations"][0]
        self.assertTrue(evaluation["visible"])
        self.assertEqual(evaluation["distance"], 5.0)
        self.assertEqual(evaluation["threat"], 1)
        self.assertEqual(evaluation["proximity"], 0.5)
        self.assertEqual(evaluation["freshness"], 1.0)
        self.assertAlmostEqual(evaluation["score"], 0.5)
        self.assertAlmostEqual(result["score"], 0.5)

    def test_entity_behind_is_invisible_and_scores_zero(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, x=-1.0, y=0.0, data={"threat": 1})])
        )
        self.assertEqual(result["status"], "NO_TARGET")
        self.assertIsNone(result["selected"])
        self.assertIsNone(result["score"])
        evaluation = result["evaluations"][0]
        self.assertFalse(evaluation["visible"])
        self.assertEqual(evaluation["distance"], 1.0)
        self.assertEqual(evaluation["proximity"], 0.9)
        self.assertEqual(evaluation["score"], 0.0)

    def test_entity_outside_max_distance_is_invisible(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, x=10.5, y=0.0, data={"threat": 1})])
        )
        evaluation = result["evaluations"][0]
        self.assertFalse(evaluation["visible"])
        self.assertEqual(evaluation["proximity"], 0.0)
        self.assertEqual(evaluation["score"], 0.0)
        self.assertEqual(result["status"], "NO_TARGET")

    def test_coincident_entity_visible_with_proximity_one(self):
        result = self.service.select_attention(
            request(
                observer={
                    "position": {"x": 3.0, "y": 4.0},
                    "forward": {"x": 0.0, "y": 1.0},
                    "max_distance": 2.0,
                    "field_of_view_degrees": 10.0,
                },
                memory=[entity("a", last_seen=10.0, x=3.0, y=4.0, data={"threat": 0.5})],
            )
        )
        evaluation = result["evaluations"][0]
        self.assertTrue(evaluation["visible"])
        self.assertEqual(evaluation["distance"], 0.0)
        self.assertEqual(evaluation["proximity"], 1.0)
        self.assertAlmostEqual(evaluation["score"], 0.5)
        self.assertEqual(result["selected"], "a")

    def test_fov_360_ignores_facing(self):
        result = self.service.select_attention(
            request(
                observer={
                    "position": {"x": 0.0, "y": 0.0},
                    "forward": {"x": 1.0, "y": 0.0},
                    "max_distance": 10.0,
                    "field_of_view_degrees": 360.0,
                },
                memory=[entity("behind", last_seen=10.0, x=-5.0, y=0.0, data={"threat": 1})],
            )
        )
        self.assertTrue(result["evaluations"][0]["visible"])
        self.assertEqual(result["selected"], "behind")

    def test_cone_boundary_is_visible(self):
        result = self.service.select_attention(
            request(
                observer={
                    "position": {"x": 0.0, "y": 0.0},
                    "forward": {"x": 1.0, "y": 0.0},
                    "max_distance": 10.0,
                    "field_of_view_degrees": 90.0,
                },
                memory=[
                    entity("edge", last_seen=10.0, x=1.0, y=1.0, data={"threat": 1}),
                    entity("past", last_seen=10.0, x=1.0, y=1.0001, data={"threat": 1}),
                ],
            )
        )
        edge, past = result["evaluations"]
        self.assertTrue(edge["visible"])
        self.assertFalse(past["visible"])

    def test_distance_boundary_is_visible_but_proximity_zero(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, x=10.0, y=0.0, data={"threat": 1})])
        )
        evaluation = result["evaluations"][0]
        self.assertTrue(evaluation["visible"])
        self.assertEqual(evaluation["proximity"], 0.0)
        self.assertEqual(evaluation["score"], 0.0)
        self.assertEqual(result["status"], "NO_TARGET")

    def test_freshness_scales_with_age(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=0.0, x=0.0, y=0.0, data={"threat": 1})])
        )
        evaluation = result["evaluations"][0]
        self.assertEqual(evaluation["freshness"], 0.5)
        self.assertAlmostEqual(evaluation["score"], 0.5)

    def test_stale_record_scores_zero(self):
        result = self.service.select_attention(
            request(
                now=20.0,
                memory=[entity("a", last_seen=0.0, x=0.0, y=0.0, data={"threat": 1})],
            )
        )
        evaluation = result["evaluations"][0]
        self.assertEqual(evaluation["freshness"], 0.0)
        self.assertEqual(evaluation["score"], 0.0)
        self.assertEqual(result["status"], "NO_TARGET")

    def test_missing_threat_defaults_to_zero(self):
        result = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, x=0.0, y=0.0)])
        )
        evaluation = result["evaluations"][0]
        self.assertEqual(evaluation["threat"], 0)
        self.assertEqual(evaluation["score"], 0.0)
        self.assertEqual(result["status"], "NO_TARGET")

    def test_zero_confidence_scores_zero(self):
        result = self.service.select_attention(
            request(
                memory=[
                    entity("a", last_seen=10.0, confidence=0.0, x=0.0, y=0.0,
                           data={"threat": 1})
                ]
            )
        )
        self.assertEqual(result["evaluations"][0]["score"], 0.0)
        self.assertEqual(result["status"], "NO_TARGET")

    def test_highest_positive_score_wins(self):
        result = self.service.select_attention(
            request(
                memory=[
                    entity("far", last_seen=10.0, x=8.0, y=0.0, data={"threat": 1}),
                    entity("near", last_seen=10.0, x=1.0, y=0.0, data={"threat": 1}),
                ]
            )
        )
        self.assertEqual(result["selected"], "near")
        self.assertEqual([e["id"] for e in result["evaluations"]], ["far", "near"])

    def test_equal_scores_keep_earlier_record(self):
        result = self.service.select_attention(
            request(
                memory=[
                    entity("first", last_seen=10.0, x=2.0, y=0.0, data={"threat": 1}),
                    entity("second", last_seen=10.0, x=2.0, y=0.0, data={"threat": 1}),
                ]
            )
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "first")

    def test_invisible_records_still_appear_in_order(self):
        result = self.service.select_attention(
            request(
                memory=[
                    entity("behind", last_seen=10.0, x=-1.0, y=0.0, data={"threat": 1}),
                    entity("ahead", last_seen=10.0, x=1.0, y=0.0, data={"threat": 1}),
                ]
            )
        )
        evaluations = result["evaluations"]
        self.assertEqual([e["id"] for e in evaluations], ["behind", "ahead"])
        self.assertFalse(evaluations[0]["visible"])
        self.assertTrue(evaluations[1]["visible"])
        self.assertEqual(result["selected"], "ahead")
        self.assertEqual(set(evaluations[0].keys()),
                         {"id", "visible", "distance", "threat", "proximity",
                          "freshness", "score"})

    def test_integer_numbers_accepted(self):
        result = self.service.select_attention(
            request(
                now=10,
                memory_horizon=20,
                observer={
                    "position": {"x": 0, "y": 0},
                    "forward": {"x": 1, "y": 0},
                    "max_distance": 10,
                    "field_of_view_degrees": 360,
                },
                memory=[entity("a", last_seen=10, x=3, y=4, data={"threat": 1})],
            )
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["evaluations"][0]["distance"], 5.0)

    def test_request_is_not_mutated(self):
        payload = request(
            memory=[
                entity("a", last_seen=10.0, x=1.0, y=0.0, data={"keep": True})
            ]
        )
        snapshot = copy.deepcopy(payload)
        self.service.select_attention(payload)
        self.assertEqual(payload, snapshot)
        # the injected default threat is never written back into the request
        self.assertEqual(set(payload["memory"][0]["data"].keys()), {"keep"})

    def test_output_does_not_alias_input(self):
        payload = request(memory=[entity("a", last_seen=10.0, x=0.0, y=0.0)])
        result = self.service.select_attention(payload)
        result["evaluations"][0]["id"] = "changed"
        self.assertEqual(payload["memory"][0]["id"], "a")

    def test_no_state_between_calls(self):
        first = self.service.select_attention(
            request(memory=[entity("a", last_seen=10.0, x=0.0, y=0.0, data={"threat": 1})])
        )
        second = self.service.select_attention(request())
        self.assertEqual(first["selected"], "a")
        self.assertEqual(second["status"], "NO_TARGET")
        self.assertEqual(second["evaluations"], [])

    def test_invalid_inputs(self):
        valid_observer = {
            "position": {"x": 0.0, "y": 0.0},
            "forward": {"x": 1.0, "y": 0.0},
            "max_distance": 10.0,
            "field_of_view_degrees": 90.0,
        }
        valid_entity = entity("a", last_seen=1.0, x=1.0, y=0.0, data={"threat": 1})

        cases = [
            "not-an-object",
            None,
            [],
            request(observer=None),
            request(observer=[]),
            request(observer={**valid_observer, "position": {"x": 0, "y": "north"}}),
            request(observer={**valid_observer, "position": {"x": 0}}),
            request(observer={**valid_observer, "position": "bad"}),
            request(observer={**valid_observer, "position": {"x": math.inf, "y": 0}}),
            request(observer={**valid_observer, "position": {"x": math.nan, "y": 0}}),
            request(observer={**valid_observer, "position": {"x": True, "y": 0}}),
            request(observer={**valid_observer, "forward": {"x": 0.0, "y": 0.0}}),
            request(observer={**valid_observer, "forward": {"x": 0, "y": 0}}),
            request(observer={**valid_observer, "forward": {"x": 1.0, "y": math.inf}}),
            request(observer={**valid_observer, "forward": {"x": 1.0}}),
            request(observer={**valid_observer, "max_distance": 0}),
            request(observer={**valid_observer, "max_distance": -1.0}),
            request(observer={**valid_observer, "max_distance": math.inf}),
            request(observer={**valid_observer, "max_distance": True}),
            request(observer={**valid_observer, "max_distance": "10"}),
            request(observer={**valid_observer, "field_of_view_degrees": 0}),
            request(observer={**valid_observer, "field_of_view_degrees": -90.0}),
            request(observer={**valid_observer, "field_of_view_degrees": 360.0001}),
            request(observer={**valid_observer, "field_of_view_degrees": math.nan}),
            request(observer={**valid_observer, "field_of_view_degrees": True}),
            request(now=-0.1),
            request(now=math.inf),
            request(now=math.nan),
            request(now=True),
            request(now="10"),
            request(now=None),
            request(memory_horizon=0),
            request(memory_horizon=-1.0),
            request(memory_horizon=math.inf),
            request(memory_horizon=True),
            request(memory=None),
            request(memory="nope"),
            request(memory=[entity("a", last_seen=11.0, data={"threat": 1})]),
            request(memory=[entity("a", last_seen=math.inf, data={"threat": 1})]),
            request(memory=[entity("a", last_seen=-1.0, data={"threat": 1})]),
            request(memory=[entity("a", last_seen=True, data={"threat": 1})]),
            request(memory=[entity("a", last_seen=1.0, confidence=1.5, data={"threat": 1})]),
            request(memory=[entity("a", last_seen=1.0, confidence=True, data={"threat": 1})]),
            request(memory=[entity("a", last_seen=1.0, x=math.inf, data={"threat": 1})]),
            request(memory=[{"id": "", "kind": "k", "last_seen": 1.0, "confidence": 1,
                            "position": {"x": 0, "y": 0}, "data": {"threat": 0}}]),
            request(memory=[{"id": 1, "kind": "k", "last_seen": 1.0, "confidence": 1,
                            "position": {"x": 0, "y": 0}, "data": {"threat": 0}}]),
            request(memory=[{"id": "a", "last_seen": 1.0, "confidence": 1,
                            "position": {"x": 0, "y": 0}, "data": {"threat": 0}}]),
            request(memory=[{"id": "a", "kind": "k", "confidence": 1,
                            "position": {"x": 0, "y": 0}, "data": {"threat": 0}}]),
            request(memory=[{"id": "a", "kind": "k", "last_seen": 1.0, "confidence": 1,
                            "position": {"x": 0}, "data": {"threat": 0}}]),
            request(memory=["not-an-object"]),
            request(memory=[valid_entity, entity("a", last_seen=2.0, data={"threat": 1})]),
            request(memory=[entity("a", last_seen=1.0, data={"threat": 1.1})]),
            request(memory=[entity("a", last_seen=1.0, data={"threat": -0.01})]),
            request(memory=[entity("a", last_seen=1.0, data={"threat": math.nan})]),
            request(memory=[entity("a", last_seen=1.0, data={"threat": math.inf})]),
            request(memory=[entity("a", last_seen=1.0, data={"threat": True})]),
            request(memory=[entity("a", last_seen=1.0, data={"threat": "1"})]),
            request(memory=[entity("a", last_seen=1.0, data="no")]),
            request(memory=[entity("a", last_seen=1.0, data={"v": math.inf})]),
            {k: v for k, v in request().items() if k != "observer"},
            {k: v for k, v in request().items() if k != "now"},
            {k: v for k, v in request().items() if k != "memory_horizon"},
            {k: v for k, v in request().items() if k != "memory"},
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    self.service.select_attention(case)

    def test_fov_endpoints(self):
        base = {
            "position": {"x": 0.0, "y": 0.0},
            "forward": {"x": 1.0, "y": 0.0},
            "max_distance": 10.0,
        }
        with self.assertRaises(AttentionError):
            self.service.select_attention(
                request(observer={**base, "field_of_view_degrees": 0.0})
            )
        result = self.service.select_attention(
            request(
                observer={**base, "field_of_view_degrees": 360.0},
                memory=[entity("a", last_seen=10.0, x=-1.0, y=0.0, data={"threat": 1})],
            )
        )
        self.assertEqual(result["selected"], "a")

    def test_failed_validation_leaves_no_partial_result_or_mutation(self):
        payload = request(
            memory=[
                entity("good", last_seen=9.0, x=1.0, y=0.0, data={"threat": 1}),
                entity("bad", last_seen=1.0, confidence=2.0, data={"threat": 1}),
            ]
        )
        snapshot = copy.deepcopy(payload)
        with self.assertRaises(AttentionError):
            self.service.select_attention(payload)
        self.assertEqual(payload, snapshot)
        follow = self.service.select_attention(
            request(memory=[entity("z", last_seen=10.0, x=0.0, y=0.0, data={"threat": 1})])
        )
        self.assertEqual(follow["selected"], "z")


if __name__ == "__main__":
    unittest.main()
