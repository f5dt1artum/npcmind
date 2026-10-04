import copy
import math
import unittest

from npcmind.service import Service, FlockingError


def request(**overrides):
    base = {
        "id": "self",
        "position": {"x": 0.0, "y": 0.0},
        "velocity": {"x": 1.0, "y": 0.0},
        "neighbors": [
            {"id": "n1", "position": {"x": 1.0, "y": 0.0}, "velocity": {"x": 0.0, "y": 1.0}},
            {"id": "n2", "position": {"x": -1.0, "y": 0.0}, "velocity": {"x": 0.0, "y": -1.0}},
        ],
        "perception_radius": 5.0,
        "separation_radius": 2.0,
        "max_speed": 10.0,
        "max_acceleration": 10.0,
        "delta_time": 0.5,
        "separation": 2.0,
        "alignment": 1.0,
        "cohesion": 1.0,
    }
    base.update(overrides)
    return base


class SteerFlockTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_basic_steered_result(self):
        # separation cancels ((-1,0) + (1,0)), alignment is (0,0)-(1,0),
        # cohesion is (0,0)-(0,0); weighted sum is (-1,0).
        result = self.service.steer_flock(request())
        self.assertEqual(result["status"], "STEERED")
        self.assertEqual(result["neighbor_ids"], ["n1", "n2"])
        self.assertEqual(result["separation"], {"x": 0.0, "y": 0.0})
        self.assertEqual(result["alignment"], {"x": -1.0, "y": 0.0})
        self.assertEqual(result["cohesion"], {"x": 0.0, "y": 0.0})
        self.assertEqual(result["acceleration"], {"x": -1.0, "y": 0.0})
        self.assertEqual(result["velocity"], {"x": 0.5, "y": 0.0})

    def test_neighbors_beyond_perception_radius_are_ignored(self):
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "far", "position": {"x": 10.0, "y": 0.0},
                     "velocity": {"x": 9.0, "y": 9.0}},
                    {"id": "near", "position": {"x": 1.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 1.0}},
                ]
            )
        )
        self.assertEqual(result["neighbor_ids"], ["near"])
        self.assertEqual(result["alignment"], {"x": -1.0, "y": 1.0})
        self.assertEqual(result["cohesion"], {"x": 1.0, "y": 0.0})

    def test_perception_boundary_is_inclusive(self):
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "edge", "position": {"x": 5.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}}
                ]
            )
        )
        self.assertEqual(result["status"], "STEERED")
        self.assertEqual(result["neighbor_ids"], ["edge"])

    def test_neighbor_ids_follow_input_order_not_distance(self):
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "far", "position": {"x": 4.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                    {"id": "near", "position": {"x": 1.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                ]
            )
        )
        self.assertEqual(result["neighbor_ids"], ["far", "near"])

    def test_separation_uses_inverse_square_and_boundary_is_inclusive(self):
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "a", "position": {"x": 2.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                    {"id": "b", "position": {"x": 0.0, "y": 1.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                ],
                separation=1.0,
                alignment=0.0,
                cohesion=0.0,
            )
        )
        # a sits exactly on the separation radius: (-2, 0) / 4 = (-0.5, 0);
        # b at distance 1: (0, -1) / 1 = (0, -1).
        self.assertEqual(result["separation"], {"x": -0.5, "y": -1.0})
        self.assertEqual(result["acceleration"], {"x": -0.5, "y": -1.0})

    def test_separation_ignores_neighbors_outside_separation_radius(self):
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "a", "position": {"x": 3.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}}
                ],
                separation=1.0,
                alignment=0.0,
                cohesion=0.0,
            )
        )
        self.assertEqual(result["separation"], {"x": 0.0, "y": 0.0})
        self.assertEqual(result["acceleration"], {"x": 0.0, "y": 0.0})

    def test_coincident_neighbor_contributes_zero_separation(self):
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "here", "position": {"x": 0.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}}
                ],
                separation=1.0,
                alignment=0.0,
                cohesion=0.0,
            )
        )
        self.assertEqual(result["status"], "STEERED")
        self.assertEqual(result["separation"], {"x": 0.0, "y": 0.0})
        self.assertEqual(result["acceleration"], {"x": 0.0, "y": 0.0})

    def test_components_are_reported_unweighted(self):
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "a", "position": {"x": 2.0, "y": 0.0},
                     "velocity": {"x": 3.0, "y": 0.0}}
                ],
                separation=10.0,
                alignment=10.0,
                cohesion=10.0,
            )
        )
        self.assertEqual(result["separation"], {"x": -0.5, "y": 0.0})
        self.assertEqual(result["alignment"], {"x": 2.0, "y": 0.0})
        self.assertEqual(result["cohesion"], {"x": 2.0, "y": 0.0})
        self.assertEqual(result["acceleration"], {"x": 10.0, "y": 0.0})

    def test_zero_weights_leave_velocity_unchanged_but_still_steered(self):
        result = self.service.steer_flock(
            request(separation=0.0, alignment=0.0, cohesion=0.0)
        )
        self.assertEqual(result["status"], "STEERED")
        self.assertEqual(result["acceleration"], {"x": 0.0, "y": 0.0})
        self.assertEqual(result["velocity"], {"x": 1.0, "y": 0.0})

    def test_acceleration_is_truncated_to_max_acceleration(self):
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "close", "position": {"x": 0.5, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}}
                ],
                max_acceleration=5.0,
                separation=10.0,
                alignment=0.0,
                cohesion=0.0,
            )
        )
        # Raw separation is (-2, 0); weighted to (-20, 0), truncated to 5.
        self.assertEqual(result["separation"], {"x": -2.0, "y": 0.0})
        self.assertEqual(result["acceleration"], {"x": -5.0, "y": 0.0})
        self.assertEqual(result["velocity"], {"x": -1.5, "y": 0.0})

    def test_velocity_is_truncated_to_max_speed(self):
        result = self.service.steer_flock(
            request(
                velocity={"x": 3.0, "y": 0.0},
                neighbors=[
                    {"id": "ahead", "position": {"x": 10.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}}
                ],
                perception_radius=20.0,
                separation_radius=1.0,
                max_speed=4.0,
                delta_time=1.0,
                separation=0.0,
                alignment=0.0,
                cohesion=1.0,
            )
        )
        # cohesion (10, 0) fits max_acceleration; 3 + 10 = 13 truncated to 4.
        self.assertEqual(result["acceleration"], {"x": 10.0, "y": 0.0})
        self.assertEqual(result["velocity"], {"x": 4.0, "y": 0.0})

    def test_diagonal_truncation_preserves_direction(self):
        result = self.service.steer_flock(
            request(
                velocity={"x": 0.0, "y": 0.0},
                neighbors=[
                    {"id": "a", "position": {"x": 3.0, "y": 4.0},
                     "velocity": {"x": 0.0, "y": 0.0}}
                ],
                perception_radius=20.0,
                separation_radius=1.0,
                max_speed=10.0,
                max_acceleration=2.0,
                delta_time=1.0,
                separation=0.0,
                alignment=0.0,
                cohesion=1.0,
            )
        )
        # cohesion (3, 4) has length 5; truncated to length 2 -> (1.2, 1.6).
        self.assertEqual(result["cohesion"], {"x": 3.0, "y": 4.0})
        self.assertAlmostEqual(result["acceleration"]["x"], 1.2)
        self.assertAlmostEqual(result["acceleration"]["y"], 1.6)
        self.assertAlmostEqual(
            math.hypot(result["acceleration"]["x"], result["acceleration"]["y"]), 2.0
        )
        self.assertAlmostEqual(result["velocity"]["x"], result["acceleration"]["x"])
        self.assertAlmostEqual(result["velocity"]["y"], result["acceleration"]["y"])

    def test_no_neighbors_returns_original_velocity_and_zero_acceleration(self):
        result = self.service.steer_flock(request(neighbors=[]))
        self.assertEqual(result["status"], "NO_NEIGHBORS")
        self.assertEqual(result["neighbor_ids"], [])
        self.assertEqual(result["separation"], {"x": 0, "y": 0})
        self.assertEqual(result["alignment"], {"x": 0, "y": 0})
        self.assertEqual(result["cohesion"], {"x": 0, "y": 0})
        self.assertEqual(result["acceleration"], {"x": 0, "y": 0})
        self.assertEqual(result["velocity"], {"x": 1.0, "y": 0.0})

    def test_no_perceived_neighbors_is_not_an_error(self):
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "far", "position": {"x": 100.0, "y": 0.0},
                     "velocity": {"x": 5.0, "y": 5.0}}
                ]
            )
        )
        self.assertEqual(result["status"], "NO_NEIGHBORS")
        self.assertEqual(result["neighbor_ids"], [])
        self.assertEqual(result["velocity"], {"x": 1.0, "y": 0.0})

    def test_original_velocity_is_echoed_verbatim_without_neighbors(self):
        result = self.service.steer_flock(
            request(velocity={"x": 3, "y": -2}, neighbors=[])
        )
        self.assertEqual(result["velocity"], {"x": 3, "y": -2})
        self.assertIsInstance(result["velocity"]["x"], int)
        self.assertIsInstance(result["velocity"]["y"], int)

    def test_integer_inputs_are_accepted(self):
        result = self.service.steer_flock(
            request(
                position={"x": 0, "y": 0},
                velocity={"x": 1, "y": 0},
                perception_radius=5,
                separation_radius=2,
                max_speed=10,
                max_acceleration=10,
                delta_time=1,
                separation=2,
                alignment=1,
                cohesion=1,
            )
        )
        self.assertEqual(result["status"], "STEERED")
        self.assertEqual(result["velocity"], {"x": 0.0, "y": 0.0})

    def test_request_is_not_mutated(self):
        req = request()
        snapshot = copy.deepcopy(req)
        self.service.steer_flock(req)
        self.assertEqual(req, snapshot)
        result = self.service.steer_flock(request(neighbors=[]))
        self.assertIsNot(result["velocity"], req["velocity"])

    def test_repeated_calls_are_independent(self):
        first = self.service.steer_flock(request())
        second = self.service.steer_flock(request())
        self.assertEqual(first, second)

    def assert_invalid(self, req):
        with self.assertRaises(ValueError):
            self.service.steer_flock(req)

    def test_request_must_be_an_object(self):
        for bad in (None, [], "x", 42, True):
            self.assert_invalid(bad)

    def test_required_fields_must_be_present(self):
        req = request()
        for key in (
            "id",
            "position",
            "velocity",
            "neighbors",
            "perception_radius",
            "separation_radius",
            "max_speed",
            "max_acceleration",
            "delta_time",
            "separation",
            "alignment",
            "cohesion",
        ):
            partial = {k: v for k, v in req.items() if k != key}
            self.assert_invalid(partial)

    def test_id_must_be_a_non_empty_string(self):
        self.assert_invalid(request(id=""))
        self.assert_invalid(request(id=1))
        self.assert_invalid(request(id=None))
        self.assert_invalid(request(id=["self"]))

    def test_vectors_must_be_finite_x_y_objects(self):
        for bad in (None, [], "0,0", {"x": 1}, {"y": 1}, {"x": True, "y": 0},
                    {"x": 1, "y": float("inf")}, {"x": float("nan"), "y": 0}):
            self.assert_invalid(request(position=bad))
            self.assert_invalid(request(velocity=bad))

    def test_neighbors_must_be_a_list_of_valid_members(self):
        self.assert_invalid(request(neighbors={}))
        self.assert_invalid(request(neighbors="n"))
        self.assert_invalid(request(neighbors=["no"]))
        self.assert_invalid(request(neighbors=[{"id": "n"}]))
        self.assert_invalid(
            request(neighbors=[{"id": "n", "position": {"x": 1, "y": 0}}])
        )
        self.assert_invalid(
            request(neighbors=[{"id": "", "position": {"x": 1, "y": 0},
                                "velocity": {"x": 0, "y": 0}}])
        )
        self.assert_invalid(
            request(neighbors=[{"id": 1, "position": {"x": 1, "y": 0},
                                "velocity": {"x": 0, "y": 0}}])
        )
        self.assert_invalid(
            request(neighbors=[{"id": "n", "position": [1, 0],
                                "velocity": {"x": 0, "y": 0}}])
        )
        self.assert_invalid(
            request(neighbors=[{"id": "n", "position": {"x": 1, "y": 0},
                                "velocity": {"x": float("inf"), "y": 0}}])
        )

    def test_ids_must_be_unique_across_self_and_neighbors(self):
        good = {"position": {"x": 1.0, "y": 0.0}, "velocity": {"x": 0.0, "y": 0.0}}
        self.assert_invalid(request(neighbors=[{"id": "self", **good}]))
        self.assert_invalid(
            request(neighbors=[{"id": "n", **good}, {"id": "n", **good}])
        )

    def test_scalar_fields_reject_bools_non_finite_and_bad_ranges(self):
        for key in ("perception_radius", "separation_radius", "max_speed",
                    "max_acceleration", "delta_time"):
            self.assert_invalid(request(**{key: True}))
            self.assert_invalid(request(**{key: 0}))
            self.assert_invalid(request(**{key: -1.0}))
            self.assert_invalid(request(**{key: float("inf")}))
            self.assert_invalid(request(**{key: float("nan")}))
            self.assert_invalid(request(**{key: "1"}))

    def test_weights_must_be_non_negative_finite_numbers(self):
        for key in ("separation", "alignment", "cohesion"):
            self.assert_invalid(request(**{key: -0.1}))
            self.assert_invalid(request(**{key: True}))
            self.assert_invalid(request(**{key: float("inf")}))
            self.assert_invalid(request(**{key: float("nan")}))
            self.assert_invalid(request(**{key: None}))

    def test_separation_radius_must_not_exceed_perception_radius(self):
        self.assert_invalid(request(perception_radius=2.0, separation_radius=2.5))
        ok = self.service.steer_flock(
            request(perception_radius=2.0, separation_radius=2.0)
        )
        self.assertEqual(ok["status"], "STEERED")

    def test_integers_too_large_for_a_double_are_rejected_cleanly(self):
        big = 10**400
        self.assert_invalid(request(position={"x": big, "y": 0}))
        self.assert_invalid(request(perception_radius=big))
        self.assert_invalid(request(separation=big))
        self.assert_invalid(
            request(neighbors=[{"id": "n", "position": {"x": big, "y": 0},
                                "velocity": {"x": 0, "y": 0}}])
        )

    def test_validation_happens_before_any_steering(self):
        with self.assertRaises(FlockingError):
            self.service.steer_flock(request(max_speed=0))

    def test_flocking_error_is_value_error(self):
        with self.assertRaises(FlockingError):
            self.service.steer_flock(request(neighbors={}))


if __name__ == "__main__":
    unittest.main()
