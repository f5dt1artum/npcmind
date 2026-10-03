import copy
import math
import unittest

from npcmind.service import Service, SteeringError


def request(**overrides):
    base = {
        "position": {"x": 0.0, "y": 0.0},
        "radius": 1.0,
        "max_speed": 10.0,
        "desired_velocity": {"x": 1.0, "y": 0.0},
        "time_horizon": 1.0,
        "candidates": [
            {"id": "stop", "velocity": {"x": 0.0, "y": 0.0}},
            {"id": "go", "velocity": {"x": 1.0, "y": 0.0}},
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
        result = self.service.select_avoidance(request())
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "go")
        self.assertEqual(result["velocity"], {"x": 1.0, "y": 0.0})

    def test_evaluations_follow_candidate_order_with_all_fields(self):
        result = self.service.select_avoidance(request())
        self.assertEqual(
            result["evaluations"],
            [
                {"id": "stop", "speed_ok": True, "collision_ids": [], "admissible": True},
                {"id": "go", "speed_ok": True, "collision_ids": [], "admissible": True},
            ],
        )

    def test_equidistant_candidates_keep_input_order(self):
        result = self.service.select_avoidance(
            request(
                desired_velocity={"x": 0.0, "y": 0.0},
                candidates=[
                    {"id": "up", "velocity": {"x": 0.0, "y": 1.0}},
                    {"id": "down", "velocity": {"x": 0.0, "y": -1.0}},
                ],
            )
        )
        self.assertEqual(result["selected"], "up")
        self.assertEqual(result["velocity"], {"x": 0.0, "y": 1.0})

    def test_speed_at_the_limit_is_allowed(self):
        result = self.service.select_avoidance(
            request(
                max_speed=1.0,
                candidates=[{"id": "go", "velocity": {"x": 1.0, "y": 0.0}}],
            )
        )
        self.assertTrue(result["evaluations"][0]["speed_ok"])
        self.assertEqual(result["selected"], "go")

    def test_too_fast_candidate_is_inadmissible_but_still_collision_checked(self):
        result = self.service.select_avoidance(
            request(
                position={"x": 0.0, "y": 0.0},
                radius=0.5,
                max_speed=5.0,
                desired_velocity={"x": 1.0, "y": 0.0},
                candidates=[
                    {"id": "fast", "velocity": {"x": 20.0, "y": 0.0}},
                    {"id": "hit", "velocity": {"x": 1.0, "y": 0.0}},
                    {"id": "safe", "velocity": {"x": 0.0, "y": 1.0}},
                ],
                neighbors=[
                    {"id": "n1", "position": {"x": 2.0, "y": 0.0},
                     "radius": 0.5, "velocity": {"x": 0.0, "y": 0.0}}
                ],
                obstacles=[
                    {"id": "o1", "position": {"x": 2.0, "y": 0.0}, "radius": 0.5}
                ],
            )
        )
        fast, hit, safe = result["evaluations"]
        self.assertFalse(fast["speed_ok"])
        self.assertEqual(fast["collision_ids"], ["n1", "o1"])
        self.assertFalse(fast["admissible"])
        self.assertTrue(hit["speed_ok"])
        self.assertEqual(hit["collision_ids"], ["n1", "o1"])
        self.assertFalse(hit["admissible"])
        self.assertTrue(safe["speed_ok"])
        self.assertEqual(safe["collision_ids"], [])
        self.assertTrue(safe["admissible"])
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "safe")

    def test_blocked_when_every_candidate_is_inadmissible(self):
        result = self.service.select_avoidance(
            request(
                radius=0.5,
                desired_velocity={"x": 1.0, "y": 0.0},
                candidates=[
                    {"id": "fast", "velocity": {"x": 99.0, "y": 0.0}},
                    {"id": "hit", "velocity": {"x": 1.0, "y": 0.0}},
                ],
                obstacles=[{"id": "wall", "position": {"x": 2.0, "y": 0.0}, "radius": 0.5}],
            )
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIsNone(result["selected"])
        self.assertEqual(result["velocity"], {"x": 0, "y": 0})
        self.assertEqual([e["id"] for e in result["evaluations"]], ["fast", "hit"])
        self.assertFalse(any(e["admissible"] for e in result["evaluations"]))

    def test_touching_at_time_zero_is_a_collision_even_when_moving_apart(self):
        # Centres start exactly one radius-sum apart (2); the overlap already
        # holds in the closed interval no matter how the discs then separate.
        result = self.service.select_avoidance(
            request(
                candidates=[{"id": "flee", "velocity": {"x": -100.0, "y": 0.0}}],
                max_speed=200.0,
                neighbors=[
                    {"id": "n", "position": {"x": 2.0, "y": 0.0}, "radius": 1.0,
                     "velocity": {"x": 100.0, "y": 0.0}}
                ],
            )
        )
        self.assertEqual(result["evaluations"][0]["collision_ids"], ["n"])
        self.assertEqual(result["status"], "BLOCKED")

    def test_touching_at_the_horizon_endpoint_is_a_collision(self):
        # v = 1 reaches centre distance 2 (the radius sum) exactly at t = 1.
        touching = self.service.select_avoidance(
            request(
                candidates=[{"id": "v1", "velocity": {"x": 1.0, "y": 0.0}}],
                obstacles=[{"id": "wall", "position": {"x": 3.0, "y": 0.0}, "radius": 1.0}],
            )
        )
        self.assertEqual(touching["evaluations"][0]["collision_ids"], ["wall"])
        # Slightly slower never closes to the radius sum within the horizon.
        short = self.service.select_avoidance(
            request(
                candidates=[{"id": "v09", "velocity": {"x": 0.9, "y": 0.0}}],
                obstacles=[{"id": "wall", "position": {"x": 3.0, "y": 0.0}, "radius": 1.0}],
            )
        )
        self.assertEqual(short["evaluations"][0]["collision_ids"], [])

    def test_static_obstacle_grazing_tangent_counts_but_clear_swervve_does_not(self):
        # Obstacle at (5, 0): a straight run crosses it, a steep swerve clears
        # it, and a shallow swerve remains on a tangential course.
        straight = self.service.select_avoidance(
            request(
                candidates=[{"id": "straight", "velocity": {"x": 5.0, "y": 0.0}}],
                max_speed=10.0,
                obstacles=[{"id": "wall", "position": {"x": 5.0, "y": 0.0}, "radius": 1.0}],
            )
        )
        self.assertEqual(straight["evaluations"][0]["collision_ids"], ["wall"])
        shallow = self.service.select_avoidance(
            request(
                candidates=[{"id": "shallow", "velocity": {"x": 5.0, "y": 2.0}}],
                max_speed=10.0,
                obstacles=[{"id": "wall", "position": {"x": 5.0, "y": 0.0}, "radius": 1.0}],
            )
        )
        # Closest centre distance is 10/sqrt(29) ~ 1.86 < 2.
        self.assertEqual(shallow["evaluations"][0]["collision_ids"], ["wall"])
        steep = self.service.select_avoidance(
            request(
                candidates=[{"id": "steep", "velocity": {"x": 5.0, "y": 3.0}}],
                max_speed=10.0,
                obstacles=[{"id": "wall", "position": {"x": 5.0, "y": 0.0}, "radius": 1.0}],
            )
        )
        # Closest centre distance is 15/sqrt(34) ~ 2.57 > 2.
        self.assertEqual(steep["evaluations"][0]["collision_ids"], [])

    def test_neighbor_velocity_moves_the_threat(self):
        # A neighbor heading across the agent's path: standing still gets hit
        # but steering upward stays clear.
        req = request(
            desired_velocity={"x": 0.0, "y": 0.0},
            candidates=[
                {"id": "stop", "velocity": {"x": 0.0, "y": 0.0}},
                {"id": "up", "velocity": {"x": 0.0, "y": 3.0}},
            ],
            neighbors=[
                {"id": "cross", "position": {"x": 5.0, "y": 0.0}, "radius": 1.0,
                 "velocity": {"x": -4.0, "y": 0.0}}
            ],
        )
        stop, up = self.service.select_avoidance(req)["evaluations"]
        self.assertEqual(stop["collision_ids"], ["cross"])
        self.assertEqual(up["collision_ids"], [])

    def test_neighbor_moving_away_never_collides(self):
        result = self.service.select_avoidance(
            request(
                candidates=[
                    {"id": "stop", "velocity": {"x": 0.0, "y": 0.0}},
                    {"id": "chase", "velocity": {"x": 0.0, "y": 3.0}},
                ],
                desired_velocity={"x": 0.0, "y": 3.0},
                neighbors=[
                    {"id": "away", "position": {"x": 0.0, "y": 3.0}, "radius": 1.0,
                     "velocity": {"x": 0.0, "y": 3.0}}
                ],
            )
        )
        self.assertEqual(result["selected"], "chase")
        self.assertEqual(
            [e["collision_ids"] for e in result["evaluations"]],
            [[], []],
        )

    def test_collision_ids_order_neighbors_then_obstacles(self):
        hit_pos = {"x": 2.0, "y": 0.0}
        result = self.service.select_avoidance(
            request(
                radius=0.5,
                candidates=[{"id": "hit", "velocity": {"x": 1.0, "y": 0.0}}],
                neighbors=[
                    {"id": "n1", "position": hit_pos, "radius": 0.5,
                     "velocity": {"x": 0.0, "y": 0.0}},
                    {"id": "n2", "position": hit_pos, "radius": 0.5,
                     "velocity": {"x": 0.0, "y": 0.0}},
                ],
                obstacles=[
                    {"id": "o1", "position": hit_pos, "radius": 0.5},
                    {"id": "o2", "position": hit_pos, "radius": 0.5},
                ],
            )
        )
        self.assertEqual(
            result["evaluations"][0]["collision_ids"], ["n1", "n2", "o1", "o2"]
        )

    def test_selected_velocity_is_echoed_verbatim(self):
        result = self.service.select_avoidance(
            request(
                desired_velocity={"x": 1, "y": 0},
                candidates=[{"id": "go", "velocity": {"x": 1, "y": 0}}],
            )
        )
        self.assertEqual(result["velocity"], {"x": 1, "y": 0})
        self.assertIsInstance(result["velocity"]["x"], int)
        self.assertIsInstance(result["velocity"]["y"], int)

    def test_neighbors_and_obstacles_default_to_empty(self):
        minimal = {
            "position": {"x": 0.0, "y": 0.0},
            "radius": 1.0,
            "max_speed": 1.0,
            "desired_velocity": {"x": 1.0, "y": 0.0},
            "time_horizon": 1.0,
            "candidates": [{"id": "go", "velocity": {"x": 1.0, "y": 0.0}}],
        }
        result = self.service.select_avoidance(minimal)
        self.assertEqual(result["selected"], "go")
        self.assertEqual(result["evaluations"][0]["collision_ids"], [])

    def test_obstacle_extra_fields_including_velocity_are_ignored(self):
        result = self.service.select_avoidance(
            request(
                candidates=[{"id": "go", "velocity": {"x": 1.0, "y": 0.0}}],
                obstacles=[
                    {"id": "wall", "position": {"x": 9.0, "y": 0.0}, "radius": 1.0,
                     "velocity": {"x": -100.0, "y": 0.0}}
                ],
            )
        )
        # Treated as static at x = 9: never reached within the horizon.
        self.assertEqual(result["evaluations"][0]["collision_ids"], [])

    def test_request_is_not_mutated(self):
        req = request(
            neighbors=[
                {"id": "n", "position": {"x": 2.0, "y": 0.0}, "radius": 1.0,
                 "velocity": {"x": 0.0, "y": 0.0}}
            ],
            obstacles=[{"id": "o", "position": {"x": -2.0, "y": 0.0}, "radius": 1.0}],
        )
        snapshot = copy.deepcopy(req)
        self.service.select_avoidance(req)
        self.assertEqual(req, snapshot)
        result = self.service.select_avoidance(req)
        self.assertIsNot(result["velocity"], req["candidates"][1]["velocity"])

    def test_repeated_calls_are_independent(self):
        first = self.service.select_avoidance(request())
        second = self.service.select_avoidance(request())
        self.assertEqual(first, second)

    def assert_invalid(self, req):
        with self.assertRaises(ValueError):
            self.service.select_avoidance(req)

    def test_request_must_be_an_object(self):
        for bad in (None, [], "x", 42, True):
            self.assert_invalid(bad)

    def test_required_fields_must_be_present(self):
        req = request()
        for key in (
            "position",
            "radius",
            "max_speed",
            "desired_velocity",
            "time_horizon",
            "candidates",
        ):
            partial = {k: v for k, v in req.items() if k != key}
            self.assert_invalid(partial)

    def test_collections_must_have_the_right_type(self):
        self.assert_invalid(request(neighbors={}))
        self.assert_invalid(request(neighbors="n"))
        self.assert_invalid(request(obstacles={}))
        self.assert_invalid(request(obstacles=1))
        self.assert_invalid(request(candidates=[]))
        self.assert_invalid(request(candidates={}))
        self.assert_invalid(request(candidates=None))
        self.assert_invalid(request(candidates=[{"id": "x", "velocity": {"x": 0, "y": 0}}] * 0))

    def test_candidate_fields_are_validated(self):
        self.assert_invalid(request(candidates=[{"id": "", "velocity": {"x": 0, "y": 0}}]))
        self.assert_invalid(request(candidates=[{"id": 1, "velocity": {"x": 0, "y": 0}}]))
        self.assert_invalid(
            request(
                candidates=[
                    {"id": "a", "velocity": {"x": 0, "y": 0}},
                    {"id": "a", "velocity": {"x": 1, "y": 0}},
                ]
            )
        )
        self.assert_invalid(request(candidates=[{"id": "a"}]))
        self.assert_invalid(request(candidates=[{"id": "a", "velocity": [0, 0]}]))
        self.assert_invalid(request(candidates=[{"id": "a", "velocity": {"x": 0}}]))
        self.assert_invalid(request(candidates=[{"id": "a", "velocity": {"y": 0}}]))
        self.assert_invalid(request(candidates=[{"id": "a", "velocity": {"x": True, "y": 0}}]))
        self.assert_invalid(request(candidates=[{"id": "a", "velocity": {"x": 0, "y": float("nan")}}]))
        self.assert_invalid(request(candidates=[{"id": "a", "velocity": {"x": float("inf"), "y": 0}}]))
        self.assert_invalid(request(candidates=["not-an-object"]))

    def test_scalar_fields_reject_bools_non_finite_and_bad_ranges(self):
        self.assert_invalid(request(radius=True))
        self.assert_invalid(request(radius=0))
        self.assert_invalid(request(radius=-1.0))
        self.assert_invalid(request(radius=float("inf")))
        self.assert_invalid(request(radius=float("nan")))
        self.assert_invalid(request(max_speed=True))
        self.assert_invalid(request(max_speed=-0.1))
        self.assert_invalid(request(max_speed=float("inf")))
        self.assert_invalid(request(time_horizon=True))
        self.assert_invalid(request(time_horizon=0))
        self.assert_invalid(request(time_horizon=-2.0))
        self.assert_invalid(request(time_horizon=float("nan")))

    def test_integers_too_large_for_a_double_are_rejected_cleanly(self):
        big = 10**400
        self.assert_invalid(request(position={"x": big, "y": 0}))
        self.assert_invalid(request(radius=big))
        self.assert_invalid(request(max_speed=big))
        self.assert_invalid(request(time_horizon=big))
        self.assert_invalid(
            request(candidates=[{"id": "go", "velocity": {"x": big, "y": 0}}])
        )

    def test_vectors_must_be_finite_x_y_objects(self):
        for bad in (None, [], "0,0", {"x": 1}, {"y": 1}, {"x": True, "y": 0},
                    {"x": 1, "y": float("inf")}):
            self.assert_invalid(request(position=bad))
            self.assert_invalid(request(desired_velocity=bad))

    def test_neighbor_fields_are_validated(self):
        good_pos = {"x": 5.0, "y": 0.0}
        good_vel = {"x": 0.0, "y": 0.0}
        self.assert_invalid(request(neighbors=["no"]))
        self.assert_invalid(request(neighbors=[{"id": "", "position": good_pos,
                                               "radius": 1.0, "velocity": good_vel}]))
        self.assert_invalid(request(neighbors=[{"id": "n", "position": good_pos,
                                               "radius": 0.0, "velocity": good_vel}]))
        self.assert_invalid(request(neighbors=[{"id": "n", "position": good_pos,
                                               "radius": -1.0, "velocity": good_vel}]))
        self.assert_invalid(request(neighbors=[{"id": "n", "position": good_pos,
                                               "radius": True, "velocity": good_vel}]))
        self.assert_invalid(request(neighbors=[{"id": "n", "position": [0, 0],
                                               "radius": 1.0, "velocity": good_vel}]))
        self.assert_invalid(request(neighbors=[{"id": "n", "position": good_pos,
                                               "radius": 1.0}]))
        self.assert_invalid(request(neighbors=[{"id": "n", "position": good_pos,
                                               "radius": 1.0, "velocity": [0, 0]}]))

    def test_obstacle_fields_are_validated(self):
        good_pos = {"x": 5.0, "y": 0.0}
        self.assert_invalid(request(obstacles=[{"id": "", "position": good_pos, "radius": 1.0}]))
        self.assert_invalid(request(obstacles=[{"id": "o", "position": good_pos, "radius": 0}]))
        self.assert_invalid(request(obstacles=[{"id": "o", "position": good_pos, "radius": False}]))
        self.assert_invalid(request(obstacles=[{"id": "o", "position": None, "radius": 1.0}]))
        self.assert_invalid(request(obstacles=[{"id": "o", "position": good_pos,
                                               "radius": float("nan")}]))

    def test_object_ids_must_be_unique_across_neighbors_and_obstacles(self):
        good_pos = {"x": 5.0, "y": 0.0}
        good_vel = {"x": 0.0, "y": 0.0}
        self.assert_invalid(
            request(
                neighbors=[{"id": "dup", "position": good_pos, "radius": 1.0, "velocity": good_vel}],
                obstacles=[{"id": "dup", "position": good_pos, "radius": 1.0}],
            )
        )
        self.assert_invalid(
            request(
                neighbors=[
                    {"id": "dup", "position": good_pos, "radius": 1.0, "velocity": good_vel},
                    {"id": "dup", "position": good_pos, "radius": 1.0, "velocity": good_vel},
                ],
            )
        )

    def test_validation_happens_before_any_evaluation(self):
        # A perfectly good candidate list never saves a structurally bad
        # request: the whole call fails rather than returning partial output.
        with self.assertRaises(SteeringError):
            self.service.select_avoidance(request(radius=0))

    def test_steering_error_is_value_error(self):
        with self.assertRaises(SteeringError):
            self.service.select_avoidance(request(candidates=[]))


if __name__ == "__main__":
    unittest.main()
