import copy
import math
import unittest

from npcmind.service import Service, SteeringError


def request(**overrides):
    base = {
        "agent": {"position": {"x": 0.0, "y": 0.0}, "radius": 1.0},
        "max_speed": 10.0,
        "desired_velocity": {"x": 1.0, "y": 0.0},
        "time_horizon": 1.0,
        "candidates": [
            {"id": "halt", "velocity": {"x": 0.0, "y": 0.0}},
            {"id": "ahead", "velocity": {"x": 5.0, "y": 0.0}},
        ],
        "neighbors": [],
        "obstacles": [],
    }
    base.update(overrides)
    return base


class SelectAvoidanceTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_picks_candidate_closest_to_desired_velocity(self):
        result = self.service.select_avoidance(
            request(
                desired_velocity={"x": 4.0, "y": 0.0},
                candidates=[
                    {"id": "slow", "velocity": {"x": 1.0, "y": 0.0}},
                    {"id": "fast", "velocity": {"x": 5.0, "y": 0.0}},
                    {"id": "back", "velocity": {"x": -1.0, "y": 0.0}},
                ],
            )
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "fast")
        self.assertEqual(result["velocity"], {"x": 5.0, "y": 0.0})

    def test_equal_distance_prefers_earlier_candidate(self):
        result = self.service.select_avoidance(
            request(
                desired_velocity={"x": 0.0, "y": 0.0},
                candidates=[
                    {"id": "up", "velocity": {"x": 0.0, "y": 1.0}},
                    {"id": "right", "velocity": {"x": 1.0, "y": 0.0}},
                    {"id": "down", "velocity": {"x": 0.0, "y": -1.0}},
                ],
            )
        )
        self.assertEqual(result["selected"], "up")

    def test_too_fast_candidate_is_inadmissible_even_without_collision(self):
        result = self.service.select_avoidance(
            request(
                max_speed=2.0,
                desired_velocity={"x": 9.0, "y": 0.0},
                candidates=[
                    {"id": "fast", "velocity": {"x": 5.0, "y": 0.0}},
                    {"id": "slow", "velocity": {"x": 2.0, "y": 0.0}},
                ],
            )
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "slow")
        self.assertEqual(result["velocity"], {"x": 2.0, "y": 0.0})
        evaluations = {e["id"]: e for e in result["evaluations"]}
        self.assertFalse(evaluations["fast"]["speed_ok"])
        self.assertEqual(evaluations["fast"]["collision_ids"], [])
        self.assertFalse(evaluations["fast"]["admissible"])
        self.assertTrue(evaluations["slow"]["speed_ok"])
        self.assertTrue(evaluations["slow"]["admissible"])

    def test_over_speed_candidate_still_reports_collisions(self):
        # Collisions are evaluated for every candidate, even over-speed ones;
        # both flags are simply false/inadmissible together.
        result = self.service.select_avoidance(
            request(
                max_speed=1.0,
                candidates=[
                    {"id": "fast-into-wall", "velocity": {"x": 5.0, "y": 0.0}},
                    {"id": "hold", "velocity": {"x": 0.0, "y": 0.0}},
                ],
                obstacles=[{"id": "wall", "position": {"x": 3.0, "y": 0.0}, "radius": 1.0}],
            )
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "hold")
        fast = result["evaluations"][0]
        self.assertFalse(fast["speed_ok"])
        self.assertEqual(fast["collision_ids"], ["wall"])
        self.assertFalse(fast["admissible"])

    def test_speed_limit_boundary_is_inclusive(self):
        result = self.service.select_avoidance(
            request(
                max_speed=3.0,
                desired_velocity={"x": 3.0, "y": 0.0},
                candidates=[{"id": "exact", "velocity": {"x": 3.0, "y": 0.0}}],
            )
        )
        self.assertEqual(result["selected"], "exact")
        self.assertTrue(result["evaluations"][0]["speed_ok"])

    def test_zero_max_speed_allows_only_zero_velocity(self):
        result = self.service.select_avoidance(
            request(
                max_speed=0.0,
                desired_velocity={"x": 0.0, "y": 0.0},
                candidates=[
                    {"id": "creep", "velocity": {"x": 0.001, "y": 0.0}},
                    {"id": "stop", "velocity": {"x": 0.0, "y": 0.0}},
                ],
            )
        )
        self.assertEqual(result["selected"], "stop")
        self.assertEqual(result["velocity"], {"x": 0.0, "y": 0.0})

    def test_stationary_obstacle_on_path_blocks_candidate(self):
        # At t = 1 the agent centre reaches the obstacle centre.
        result = self.service.select_avoidance(
            request(
                desired_velocity={"x": 5.0, "y": 0.0},
                candidates=[
                    {"id": "straight", "velocity": {"x": 5.0, "y": 0.0}},
                    {"id": "sideways", "velocity": {"x": 0.0, "y": 5.0}},
                ],
                obstacles=[{"id": "wall", "position": {"x": 5.0, "y": 0.0}, "radius": 1.0}],
            )
        )
        self.assertEqual(result["selected"], "sideways")
        evaluations = {e["id"]: e for e in result["evaluations"]}
        self.assertEqual(evaluations["straight"]["collision_ids"], ["wall"])
        self.assertFalse(evaluations["straight"]["admissible"])
        self.assertEqual(evaluations["sideways"]["collision_ids"], [])

    def test_touching_at_radius_sum_counts_as_collision(self):
        # At t = 1 centre distance is exactly agent + obstacle radius.
        result = self.service.select_avoidance(
            request(
                desired_velocity={"x": 2.0, "y": 0.0},
                candidates=[
                    {"id": "creep", "velocity": {"x": 2.0, "y": 0.0}},
                    {"id": "wait", "velocity": {"x": 0.0, "y": 0.0}},
                ],
                obstacles=[{"id": "wall", "position": {"x": 4.0, "y": 0.0}, "radius": 1.0}],
            )
        )
        self.assertEqual(result["selected"], "wait")
        self.assertEqual(result["evaluations"][0]["collision_ids"], ["wall"])

    def test_contact_exactly_at_horizon_is_a_collision(self):
        result = self.service.select_avoidance(
            request(
                candidates=[{"id": "go", "velocity": {"x": 5.0, "y": 0.0}}],
                obstacles=[{"id": "wall", "position": {"x": 7.0, "y": 0.0}, "radius": 1.0}],
            )
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["evaluations"][0]["collision_ids"], ["wall"])

    def test_contact_after_horizon_is_safe(self):
        result = self.service.select_avoidance(
            request(
                candidates=[{"id": "go", "velocity": {"x": 5.0, "y": 0.0}}],
                obstacles=[{"id": "wall", "position": {"x": 7.5, "y": 0.0}, "radius": 1.0}],
            )
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "go")
        self.assertEqual(result["evaluations"][0]["collision_ids"], [])

    def test_overlap_at_time_zero_is_a_collision(self):
        result = self.service.select_avoidance(
            request(
                candidates=[
                    {"id": "flee", "velocity": {"x": 0.0, "y": 5.0}},
                    {"id": "wait", "velocity": {"x": 0.0, "y": 0.0}},
                ],
                obstacles=[{"id": "wall", "position": {"x": 1.0, "y": 0.0}, "radius": 1.0}],
            )
        )
        # Both candidates already overlap at t = 0.
        self.assertEqual(result["status"], "BLOCKED")
        for evaluation in result["evaluations"]:
            self.assertEqual(evaluation["collision_ids"], ["wall"])

    def test_moving_neighbor_head_on_blocks_straight_candidates(self):
        # Neighbour reaches x = 0 at t = 1; perpendicular motion stays clear.
        result = self.service.select_avoidance(
            request(
                desired_velocity={"x": 8.0, "y": 0.0},
                candidates=[
                    {"id": "charge", "velocity": {"x": 4.0, "y": 0.0}},
                    {"id": "hold", "velocity": {"x": 0.0, "y": 0.0}},
                    {"id": "dodge", "velocity": {"x": 0.0, "y": 4.0}},
                ],
                neighbors=[
                    {"id": "bike", "position": {"x": 8.0, "y": 0.0}, "radius": 1.0,
                     "velocity": {"x": -8.0, "y": 0.0}}
                ],
            )
        )
        self.assertEqual(result["selected"], "dodge")
        evaluations = {e["id"]: e for e in result["evaluations"]}
        self.assertEqual(evaluations["charge"]["collision_ids"], ["bike"])
        self.assertEqual(evaluations["hold"]["collision_ids"], ["bike"])
        self.assertEqual(evaluations["dodge"]["collision_ids"], [])

    def test_collision_ids_list_neighbors_then_obstacles_in_order(self):
        result = self.service.select_avoidance(
            request(
                candidates=[{"id": "go", "velocity": {"x": 0.0, "y": 0.0}}],
                neighbors=[
                    {"id": "n1", "position": {"x": 0.5, "y": 0.0}, "radius": 1.0,
                     "velocity": {"x": 0.0, "y": 0.0}},
                    {"id": "n2", "position": {"x": -0.5, "y": 0.0}, "radius": 1.0,
                     "velocity": {"x": 0.0, "y": 0.0}},
                ],
                obstacles=[
                    {"id": "o1", "position": {"x": 0.0, "y": 0.5}, "radius": 1.0},
                    {"id": "o2", "position": {"x": 0.0, "y": -0.5}, "radius": 1.0},
                ],
            )
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["evaluations"][0]["collision_ids"], ["n1", "n2", "o1", "o2"])

    def test_blocked_returns_null_selection_zero_velocity_and_all_evaluations(self):
        result = self.service.select_avoidance(
            request(
                max_speed=1.0,
                desired_velocity={"x": 1.0, "y": 0.0},
                candidates=[
                    {"id": "fast", "velocity": {"x": 5.0, "y": 0.0}},
                    {"id": "into-wall", "velocity": {"x": 1.0, "y": 0.0}},
                ],
                obstacles=[{"id": "wall", "position": {"x": 1.0, "y": 0.0}, "radius": 1.0}],
            )
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIsNone(result["selected"])
        self.assertEqual(result["velocity"], {"x": 0, "y": 0})
        self.assertEqual([e["id"] for e in result["evaluations"]], ["fast", "into-wall"])

    def test_evaluations_follow_candidate_order(self):
        result = self.service.select_avoidance(
            request(
                candidates=[
                    {"id": cid, "velocity": {"x": float(i), "y": 0.0}}
                    for i, cid in enumerate(("a", "b", "c"))
                ]
            )
        )
        self.assertEqual([e["id"] for e in result["evaluations"]], ["a", "b", "c"])
        for evaluation in result["evaluations"]:
            self.assertIn("speed_ok", evaluation)
            self.assertIn("collision_ids", evaluation)
            self.assertIn("admissible", evaluation)

    def test_returned_velocity_is_the_original_vector_unchanged(self):
        result = self.service.select_avoidance(
            request(
                desired_velocity={"x": 1, "y": 2},
                candidates=[{"id": "v", "velocity": {"x": 1, "y": 2}}],
            )
        )
        self.assertEqual(result["velocity"], {"x": 1, "y": 2})

    def test_request_is_not_mutated(self):
        req = request(
            neighbors=[
                {"id": "bike", "position": {"x": 3.0, "y": 0.0}, "radius": 1.0,
                 "velocity": {"x": -1.0, "y": 0.0}}
            ],
            obstacles=[{"id": "wall", "position": {"x": 4.0, "y": 0.0}, "radius": 1.0}],
        )
        snapshot = copy.deepcopy(req)
        self.service.select_avoidance(req)
        self.assertEqual(req, snapshot)

    def test_returned_velocity_is_not_the_callers_object(self):
        req = request(candidates=[{"id": "v", "velocity": {"x": 0.0, "y": 0.0}}])
        result = self.service.select_avoidance(req)
        self.assertIsNot(result["velocity"], req["candidates"][0]["velocity"])

    def test_repeated_calls_are_independent(self):
        first = self.service.select_avoidance(request())
        second = self.service.select_avoidance(request())
        self.assertEqual(first, second)

    def assert_invalid(self, req):
        with self.assertRaises(ValueError):
            self.service.select_avoidance(req)

    def test_request_must_be_object(self):
        self.assert_invalid(None)
        self.assert_invalid([1, 2])
        self.assert_invalid("nope")

    def test_required_fields_must_be_present(self):
        req = request()
        for key in ("agent", "max_speed", "desired_velocity", "time_horizon", "candidates"):
            partial = {k: v for k, v in req.items() if k != key}
            self.assert_invalid(partial)

    def test_collections_must_have_correct_type(self):
        self.assert_invalid(request(neighbors={"id": "x"}))
        self.assert_invalid(request(obstacles="wall"))
        self.assert_invalid(request(candidates=[]))
        self.assert_invalid(request(candidates={}))
        self.assert_invalid(request(candidates=None))

    def test_agent_must_be_valid_disc(self):
        self.assert_invalid(request(agent=[]))
        self.assert_invalid(request(agent={"position": {"x": 0.0, "y": 0.0}}))
        self.assert_invalid(request(agent={"position": {"x": 0.0, "y": 0.0}, "radius": 0.0}))
        self.assert_invalid(request(agent={"position": {"x": 0.0, "y": 0.0}, "radius": -1.0}))
        self.assert_invalid(request(agent={"position": {"x": 0.0, "y": 0.0}, "radius": True}))
        self.assert_invalid(request(agent={"position": {"x": 0.0}, "radius": 1.0}))
        self.assert_invalid(request(agent={"position": {"x": "0", "y": 0}, "radius": 1.0}))
        self.assert_invalid(request(agent={"position": [0.0, 0.0], "radius": 1.0}))

    def test_scalar_fields_reject_bad_numbers(self):
        self.assert_invalid(request(max_speed=-0.1))
        self.assert_invalid(request(max_speed=True))
        self.assert_invalid(request(max_speed="10"))
        self.assert_invalid(request(time_horizon=0.0))
        self.assert_invalid(request(time_horizon=-2.0))
        self.assert_invalid(request(time_horizon=math.inf))
        self.assert_invalid(request(time_horizon=math.nan))
        self.assert_invalid(request(desired_velocity={"x": math.inf, "y": 0.0}))

    def test_vectors_reject_malformed_structures(self):
        self.assert_invalid(request(desired_velocity={"x": 1.0}))
        self.assert_invalid(request(desired_velocity={"x": 1.0, "y": False}))
        self.assert_invalid(request(desired_velocity=(1.0, 0.0)))

    def test_candidate_ids_must_be_unique_non_empty_strings(self):
        good = {"x": 0.0, "y": 0.0}
        self.assert_invalid(request(candidates=[{"id": "", "velocity": good}]))
        self.assert_invalid(request(candidates=[{"id": 4, "velocity": good}]))
        self.assert_invalid(
            request(
                candidates=[
                    {"id": "same", "velocity": good},
                    {"id": "same", "velocity": good},
                ]
            )
        )
        self.assert_invalid(request(candidates=[{"id": "no-velocity"}]))
        self.assert_invalid(request(candidates=[{"id": "bad-vel", "velocity": {"x": 1}}]))
        self.assert_invalid(request(candidates=["just-a-string"]))

    def test_neighbors_and_obstacles_must_be_valid_discs(self):
        self.assert_invalid(request(neighbors=[{"id": "n", "position": {"x": 1.0, "y": 0.0}}]))
        self.assert_invalid(
            request(
                neighbors=[
                    {"id": "n", "position": {"x": 1.0, "y": 0.0}, "radius": 1.0,
                     "velocity": {"x": 0.0}}
                ]
            )
        )
        self.assert_invalid(
            request(
                neighbors=[
                    {"id": "n", "position": {"x": 1.0, "y": 0.0}, "radius": 0.0,
                     "velocity": {"x": 0.0, "y": 0.0}}
                ]
            )
        )
        self.assert_invalid(
            request(obstacles=[{"position": {"x": 1.0, "y": 0.0}, "radius": 1.0}])
        )
        self.assert_invalid(
            request(obstacles=[{"id": "o", "position": {"x": 1.0, "y": 0.0}}])
        )

    def test_object_ids_must_be_unique_across_neighbors_and_obstacles(self):
        disk = {"position": {"x": 9.0, "y": 9.0}, "radius": 1.0}
        self.assert_invalid(
            request(
                neighbors=[{"id": "dup", **disk, "velocity": {"x": 0.0, "y": 0.0}}],
                obstacles=[{"id": "dup", **disk}],
            )
        )
        self.assert_invalid(
            request(
                neighbors=[
                    {"id": "dup", **disk, "velocity": {"x": 0.0, "y": 0.0}},
                    {"id": "dup", **disk, "velocity": {"x": 0.0, "y": 0.0}},
                ]
            )
        )
        self.assert_invalid(request(obstacles=[{"id": "dup", **disk}, {"id": "dup", **disk}]))

    def test_steering_error_is_value_error(self):
        with self.assertRaises(SteeringError):
            self.service.select_avoidance(request(candidates=[]))


if __name__ == "__main__":
    unittest.main()
