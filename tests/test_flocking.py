import copy
import math
import unittest

from npcmind.service import FlockingError, Service


def request(**overrides):
    base = {
        "id": "self",
        "position": {"x": 0.0, "y": 0.0},
        "velocity": {"x": 0.0, "y": 0.0},
        "neighbors": [
            {"id": "n1", "position": {"x": 1.0, "y": 0.0},
             "velocity": {"x": 0.0, "y": 0.0}},
        ],
        "perception_radius": 5.0,
        "separation_radius": 1.0,
        "max_speed": 10.0,
        "max_acceleration": 2.0,
        "delta_time": 1.0,
        "separation": 1.0,
        "alignment": 1.0,
        "cohesion": 1.0,
    }
    base.update(overrides)
    return base


class SteerFlockTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def assert_vector_close(self, vector, x, y):
        self.assertTrue(math.isclose(vector["x"], x, abs_tol=1e-12), vector)
        self.assertTrue(math.isclose(vector["y"], y, abs_tol=1e-12), vector)

    def test_no_neighbors_returns_original_velocity_and_zero_components(self):
        result = self.service.steer_flock(request(neighbors=[]))
        self.assertEqual(result["status"], "NO_NEIGHBORS")
        self.assertEqual(result["neighbor_ids"], [])
        self.assertEqual(result["separation"], {"x": 0.0, "y": 0.0})
        self.assertEqual(result["alignment"], {"x": 0.0, "y": 0.0})
        self.assertEqual(result["cohesion"], {"x": 0.0, "y": 0.0})
        self.assertEqual(result["acceleration"], {"x": 0.0, "y": 0.0})
        self.assertEqual(result["velocity"], {"x": 0.0, "y": 0.0})

    def test_no_neighbors_never_clamps_an_over_speed_velocity(self):
        result = self.service.steer_flock(
            request(neighbors=[], velocity={"x": 10.0, "y": 0.0}, max_speed=1.0)
        )
        self.assertEqual(result["status"], "NO_NEIGHBORS")
        self.assertEqual(result["velocity"], {"x": 10.0, "y": 0.0})

    def test_no_neighbors_echoes_velocity_values_verbatim(self):
        result = self.service.steer_flock(request(neighbors=[], velocity={"x": 3, "y": -2}))
        self.assertEqual(result["velocity"], {"x": 3, "y": -2})
        self.assertIsInstance(result["velocity"]["x"], int)
        self.assertIsInstance(result["velocity"]["y"], int)

    def test_neighbors_outside_perception_are_ignored(self):
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "far", "position": {"x": 6.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                ]
            )
        )
        self.assertEqual(result["status"], "NO_NEIGHBORS")
        self.assertEqual(result["neighbor_ids"], [])

    def test_perception_radius_boundary_is_inclusive(self):
        on = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "edge", "position": {"x": 5.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                ],
                separation=0.0,
                alignment=0.0,
            )
        )
        self.assertEqual(on["status"], "STEERED")
        self.assertEqual(on["neighbor_ids"], ["edge"])
        self.assert_vector_close(on["cohesion"], 5.0, 0.0)
        beyond = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "edge", "position": {"x": 5.0 + 1e-9, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                ],
            )
        )
        self.assertEqual(beyond["status"], "NO_NEIGHBORS")

    def test_neighbor_ids_keep_input_order_and_drop_far_ones(self):
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "a", "position": {"x": 1.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                    {"id": "b", "position": {"x": 0.0, "y": -4.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                    {"id": "far", "position": {"x": 99.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                    {"id": "c", "position": {"x": 0.0, "y": 2.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                ]
            )
        )
        self.assertEqual(result["status"], "STEERED")
        self.assertEqual(result["neighbor_ids"], ["a", "b", "c"])

    def test_separation_sums_difference_over_squared_distance(self):
        result = self.service.steer_flock(
            request(
                position={"x": 0.0, "y": 0.0},
                neighbors=[
                    # On the separation-radius boundary: counts.
                    {"id": "axis-x", "position": {"x": 1.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                    # Coincident: contributes nothing.
                    {"id": "same", "position": {"x": 0.0, "y": 0.0},
                     "velocity": {"x": 2.0, "y": 0.0}},
                    # Outside separation radius but perceived: no separation.
                    {"id": "axis-y", "position": {"x": 0.0, "y": 2.0},
                     "velocity": {"x": 4.0, "y": 0.0}},
                ],
                perception_radius=5.0,
                separation_radius=1.0,
                separation=1.0,
                alignment=0.0,
                cohesion=0.0,
            )
        )
        self.assertEqual(result["neighbor_ids"], ["axis-x", "same", "axis-y"])
        self.assert_vector_close(result["separation"], -1.0, 0.0)
        # Alignment/cohesion are still returned unweighted over all three.
        self.assert_vector_close(result["alignment"], 2.0, 0.0)
        self.assert_vector_close(result["cohesion"], 1.0 / 3.0, 2.0 / 3.0)
        self.assert_vector_close(result["acceleration"], -1.0, 0.0)
        self.assert_vector_close(result["velocity"], -1.0, 0.0)

    def test_separation_weighted_neighbor(self):
        # Neighbour at distance 2: difference (-2, 0) / 4 = (-0.5, 0).
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "n", "position": {"x": 2.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                ],
                perception_radius=5.0,
                separation_radius=3.0,
                max_acceleration=10.0,
                separation=2.0,
                alignment=0.0,
                cohesion=0.0,
            )
        )
        self.assert_vector_close(result["separation"], -0.5, 0.0)
        self.assert_vector_close(result["acceleration"], -1.0, 0.0)

    def test_separation_boundary_is_inclusive_but_just_outside_is_excluded(self):
        neighbors = [
            {"id": "edge", "position": {"x": 2.0, "y": 0.0},
             "velocity": {"x": 0.0, "y": 0.0}},
        ]
        on = self.service.steer_flock(
            request(
                neighbors=neighbors,
                perception_radius=5.0,
                separation_radius=2.0,
                alignment=0.0,
                cohesion=0.0,
            )
        )
        self.assert_vector_close(on["separation"], -0.5, 0.0)
        off = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "edge", "position": {"x": 2.0 + 1e-9, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                ],
                perception_radius=5.0,
                separation_radius=2.0,
                alignment=0.0,
                cohesion=0.0,
            )
        )
        self.assert_vector_close(off["separation"], 0.0, 0.0)

    def test_alignment_and_cohesion_use_means_minus_self(self):
        result = self.service.steer_flock(
            request(
                position={"x": 1.0, "y": 1.0},
                velocity={"x": 1.0, "y": 0.0},
                neighbors=[
                    {"id": "a", "position": {"x": 3.0, "y": 1.0},
                     "velocity": {"x": 1.0, "y": 2.0}},
                    {"id": "b", "position": {"x": 1.0, "y": 3.0},
                     "velocity": {"x": 3.0, "y": 0.0}},
                ],
                separation=0.0,
                alignment=0.0,
                cohesion=0.0,
            )
        )
        self.assert_vector_close(result["alignment"], 1.0, 1.0)
        self.assert_vector_close(result["cohesion"], 1.0, 1.0)
        self.assert_vector_close(result["acceleration"], 0.0, 0.0)

    def test_acceleration_is_truncated_to_max_acceleration(self):
        # Alignment alone: (6, 8) * 0.5 = (3, 4), length 5, clamped to 2.5.
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "n", "position": {"x": 1.0, "y": 0.0},
                     "velocity": {"x": 6.0, "y": 8.0}},
                ],
                perception_radius=5.0,
                separation_radius=0.5,
                max_acceleration=2.5,
                max_speed=10.0,
                delta_time=1.0,
                separation=0.0,
                alignment=0.5,
                cohesion=0.0,
            )
        )
        self.assert_vector_close(result["acceleration"], 1.5, 2.0)
        self.assertAlmostEqual(math.hypot(*result["acceleration"].values()), 2.5)
        self.assert_vector_close(result["velocity"], 1.5, 2.0)

    def test_velocity_is_truncated_to_max_speed(self):
        # Same setup but the speed cap bites: length 2.5 -> 1.25.
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "n", "position": {"x": 1.0, "y": 0.0},
                     "velocity": {"x": 6.0, "y": 8.0}},
                ],
                perception_radius=5.0,
                separation_radius=0.5,
                max_acceleration=2.5,
                max_speed=1.25,
                delta_time=1.0,
                separation=0.0,
                alignment=0.5,
                cohesion=0.0,
            )
        )
        self.assert_vector_close(result["velocity"], 0.75, 1.0)
        self.assertAlmostEqual(math.hypot(*result["velocity"].values()), 1.25)

    def test_delta_time_scales_the_velocity_update(self):
        # Acceleration (1, 0); dt = 0.5 moves velocity to (0.5, 0).
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "n", "position": {"x": -1.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                ],
                perception_radius=5.0,
                separation_radius=1.0,
                max_acceleration=10.0,
                max_speed=10.0,
                delta_time=0.5,
                separation=1.0,
                alignment=0.0,
                cohesion=0.0,
            )
        )
        self.assert_vector_close(result["acceleration"], 1.0, 0.0)
        self.assert_vector_close(result["velocity"], 0.5, 0.0)

    def test_zero_weighted_acceleration_leaves_velocity_but_still_clamps_speed(self):
        steered = self.service.steer_flock(
            request(
                velocity={"x": 3.0, "y": 4.0},
                max_speed=10.0,
                separation=0.0,
                alignment=0.0,
                cohesion=0.0,
            )
        )
        self.assertEqual(steered["status"], "STEERED")
        self.assert_vector_close(steered["acceleration"], 0.0, 0.0)
        self.assert_vector_close(steered["velocity"], 3.0, 4.0)
        clamped = self.service.steer_flock(
            request(
                velocity={"x": 10.0, "y": 0.0},
                max_speed=5.0,
                separation=0.0,
                alignment=0.0,
                cohesion=0.0,
            )
        )
        self.assert_vector_close(clamped["velocity"], 5.0, 0.0)

    def test_tiny_max_speed_still_clamps_the_velocity(self):
        # max_speed must be positive; a very small positive cap still clamps.
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "n", "position": {"x": 1.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                ],
                separation_radius=0.1,
                max_speed=0.1,
                max_acceleration=2.0,
                separation=0.0,
                alignment=0.0,
                cohesion=1.0,
            )
        )
        self.assertEqual(result["status"], "STEERED")
        self.assert_vector_close(result["velocity"], 0.1, 0.0)

    def test_cohesion_only_steers_toward_the_centroid(self):
        result = self.service.steer_flock(
            request(
                neighbors=[
                    {"id": "a", "position": {"x": 2.0, "y": 0.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                    {"id": "b", "position": {"x": 0.0, "y": 2.0},
                     "velocity": {"x": 0.0, "y": 0.0}},
                ],
                separation_radius=0.5,
                max_acceleration=10.0,
                separation=0.0,
                alignment=0.0,
                cohesion=1.0,
            )
        )
        self.assert_vector_close(result["cohesion"], 1.0, 1.0)
        self.assert_vector_close(result["acceleration"], 1.0, 1.0)
        self.assert_vector_close(result["velocity"], 1.0, 1.0)

    def test_request_is_not_mutated(self):
        req = request(
            neighbors=[
                {"id": "n", "position": {"x": 1.0, "y": 0.0},
                 "velocity": {"x": 0.0, "y": 0.0}},
            ]
        )
        snapshot = copy.deepcopy(req)
        self.service.steer_flock(req)
        self.assertEqual(req, snapshot)

    def test_repeated_calls_are_independent_and_identical(self):
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
        good_neighbor = req["neighbors"][0]
        for key in ("id", "position", "velocity"):
            bad = copy.deepcopy(req)
            del bad["neighbors"][0][key]
            self.assert_invalid(bad)
            self.assertEqual(req["neighbors"][0], good_neighbor)

    def test_neighbors_must_be_a_list(self):
        self.assert_invalid(request(neighbors={}))
        self.assert_invalid(request(neighbors="n"))
        self.assert_invalid(request(neighbors=None))
        self.assert_invalid(request(neighbors=[{"id": "n"}]))
        self.assert_invalid(request(neighbors=["no"]))

    def test_ids_must_be_non_empty_and_unique(self):
        good_vel = {"x": 0.0, "y": 0.0}
        good_pos = {"x": 1.0, "y": 0.0}
        self.assert_invalid(request(id=""))
        self.assert_invalid(request(id=7))
        self.assert_invalid(request(neighbors=[{"id": "", "position": good_pos, "velocity": good_vel}]))
        self.assert_invalid(request(neighbors=[{"id": 3, "position": good_pos, "velocity": good_vel}]))
        self.assert_invalid(
            request(
                id="dup",
                neighbors=[{"id": "dup", "position": good_pos, "velocity": good_vel}],
            )
        )
        self.assert_invalid(
            request(
                neighbors=[
                    {"id": "dup", "position": good_pos, "velocity": good_vel},
                    {"id": "dup", "position": {"x": 0.0, "y": 1.0}, "velocity": good_vel},
                ]
            )
        )

    def test_vectors_must_be_finite_x_y_objects(self):
        for bad in (None, [], "0,0", {"x": 1}, {"y": 1}, {"x": True, "y": 0},
                    {"x": 1, "y": float("inf")}, {"x": float("nan"), "y": 0}):
            self.assert_invalid(request(position=bad))
            self.assert_invalid(request(velocity=bad))
            self.assert_invalid(
                request(neighbors=[{"id": "n", "position": {"x": 1.0, "y": 0.0},
                                   "velocity": bad}])
            )
            self.assert_invalid(
                request(neighbors=[{"id": "n", "position": bad,
                                   "velocity": {"x": 0.0, "y": 0.0}}])
            )

    def test_scalar_fields_reject_bools_non_finite_and_bad_ranges(self):
        self.assert_invalid(request(perception_radius=True))
        self.assert_invalid(request(perception_radius=0))
        self.assert_invalid(request(perception_radius=-1.0))
        self.assert_invalid(request(perception_radius=float("inf")))
        self.assert_invalid(request(perception_radius=float("nan")))
        self.assert_invalid(request(separation_radius=0))
        self.assert_invalid(request(separation_radius=True))
        self.assert_invalid(request(separation_radius=float("nan")))
        self.assert_invalid(request(separation_radius=6.0))
        self.assert_invalid(request(max_speed=True))
        self.assert_invalid(request(max_speed=0))
        self.assert_invalid(request(max_speed=-0.01))
        self.assert_invalid(request(max_speed=float("inf")))
        self.assert_invalid(request(max_acceleration=0))
        self.assert_invalid(request(max_acceleration=-3.0))
        self.assert_invalid(request(max_acceleration=float("nan")))
        self.assert_invalid(request(delta_time=0))
        self.assert_invalid(request(delta_time=-1.0))
        self.assert_invalid(request(delta_time=float("inf")))
        for name in ("separation", "alignment", "cohesion"):
            self.assert_invalid(request(**{name: -0.1}))
            self.assert_invalid(request(**{name: True}))
            self.assert_invalid(request(**{name: float("inf")}))
            self.assert_invalid(request(**{name: float("nan")}))

    def test_equal_radii_are_allowed(self):
        result = self.service.steer_flock(
            request(perception_radius=2.0, separation_radius=2.0)
        )
        self.assertEqual(result["status"], "STEERED")

    def test_zero_weights_are_allowed(self):
        result = self.service.steer_flock(
            request(separation=0.0, alignment=0.0, cohesion=0.0)
        )
        self.assertEqual(result["status"], "STEERED")

    def test_integers_too_large_for_a_double_are_rejected_cleanly(self):
        big = 10**400
        self.assert_invalid(request(position={"x": big, "y": 0}))
        self.assert_invalid(request(perception_radius=big))
        self.assert_invalid(request(separation_radius=big))
        self.assert_invalid(request(max_speed=big))
        self.assert_invalid(request(max_acceleration=big))
        self.assert_invalid(request(delta_time=big))
        self.assert_invalid(request(separation=big))

    def test_validation_happens_before_any_computation(self):
        with self.assertRaises(FlockingError):
            self.service.steer_flock(request(perception_radius=0))

    def test_flocking_error_is_value_error(self):
        with self.assertRaises(FlockingError):
            self.service.steer_flock(request(neighbors=None))


if __name__ == "__main__":
    unittest.main()
