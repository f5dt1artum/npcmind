import copy
import unittest

from npcmind.service import Service, NavigationError


def request(**overrides):
    base = {
        "grid": [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        "start": {"x": 0, "y": 0},
        "goal": {"x": 2, "y": 2},
    }
    base.update(overrides)
    return base


class FindPathTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_finds_lexicographically_smallest_minimum_cost_route(self):
        result = self.service.find_path(request())
        self.assertEqual(result["status"], "SUCCESS")
        # Uniform grid: several cost-4 routes; the (y, x)-smallest keeps to
        # the top row before descending.
        self.assertEqual(
            result["path"],
            [
                {"x": 0, "y": 0},
                {"x": 1, "y": 0},
                {"x": 2, "y": 0},
                {"x": 2, "y": 1},
                {"x": 2, "y": 2},
            ],
        )
        self.assertEqual(result["cost"], 4)

    def test_two_by_two_tie_prefers_x_before_y_at_same_y(self):
        result = self.service.find_path(
            request(
                grid=[[1, 1], [1, 1]],
                start={"x": 0, "y": 0},
                goal={"x": 1, "y": 1},
            )
        )
        self.assertEqual(
            result["path"],
            [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
        )
        self.assertEqual(result["cost"], 2)

    def test_cheaper_detour_beats_expensive_direct_route(self):
        result = self.service.find_path(
            request(
                grid=[
                    [1, 100, 1],
                    [1, 1, 1],
                ],
                start={"x": 0, "y": 0},
                goal={"x": 2, "y": 0},
            )
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(
            result["path"],
            [
                {"x": 0, "y": 0},
                {"x": 0, "y": 1},
                {"x": 1, "y": 1},
                {"x": 2, "y": 1},
                {"x": 2, "y": 0},
            ],
        )
        # Entering the start cell is free; four cost-1 cells are entered.
        self.assertEqual(result["cost"], 4)

    def test_start_equals_goal_returns_single_point_zero_cost(self):
        result = self.service.find_path(
            request(start={"x": 1, "y": 1}, goal={"x": 1, "y": 1})
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["path"], [{"x": 1, "y": 1}])
        self.assertEqual(result["cost"], 0)

    def test_walls_block_movement_to_unreachable_goal(self):
        result = self.service.find_path(
            request(
                grid=[
                    [1, None, 1],
                    [1, None, 1],
                    [1, None, 1],
                ],
                start={"x": 0, "y": 0},
                goal={"x": 2, "y": 0},
            )
        )
        self.assertEqual(result["status"], "UNREACHABLE")
        self.assertEqual(result["path"], [])
        self.assertIsNone(result["cost"])

    def test_single_cell_grid_path_to_self(self):
        result = self.service.find_path(
            request(grid=[[42]], start={"x": 0, "y": 0}, goal={"x": 0, "y": 0})
        )
        self.assertEqual(result["path"], [{"x": 0, "y": 0}])
        self.assertEqual(result["cost"], 0)

    def test_request_is_not_mutated(self):
        req = request(
            grid=[[1, 100, 1], [1, 1, 1]],
            start={"x": 0, "y": 0},
            goal={"x": 2, "y": 0},
        )
        snapshot = copy.deepcopy(req)
        self.service.find_path(req)
        self.assertEqual(req, snapshot)

    def test_repeated_calls_are_independent(self):
        first = self.service.find_path(request())
        second = self.service.find_path(request())
        self.assertEqual(first, second)

    def assert_invalid(self, req):
        with self.assertRaises(ValueError):
            self.service.find_path(req)

    def test_request_and_coordinates_must_be_objects(self):
        self.assert_invalid([1, 2])
        self.assert_invalid(None)
        self.assert_invalid(request(start=[0, 0]))
        self.assert_invalid(request(goal=(1, 2)))

    def test_required_fields_must_be_present(self):
        req = request()
        for key in ("grid", "start", "goal"):
            partial = {k: v for k, v in req.items() if k != key}
            self.assert_invalid(partial)

    def test_grid_must_be_non_empty_rectangle(self):
        self.assert_invalid(request(grid=[]))
        self.assert_invalid(request(grid=[[]]))
        self.assert_invalid(request(grid={}))
        self.assert_invalid(request(grid=None))
        self.assert_invalid(request(grid=[[1, 2], [3]]))
        self.assert_invalid(request(grid=[[1], [2, 3]]))
        self.assert_invalid(request(grid="not-a-grid"))

    def test_grid_cells_must_be_null_or_positive_integers(self):
        for bad in (True, False, 0, -1, 1.5, 2.0, "1", [], {}):
            self.assert_invalid(request(grid=[[1, bad], [1, 1]]))

    def test_coordinates_require_integer_x_and_y(self):
        self.assert_invalid(request(start={}))
        self.assert_invalid(request(start={"x": 0}))
        self.assert_invalid(request(start={"y": 0}))
        self.assert_invalid(request(start={"x": True, "y": 0}))
        self.assert_invalid(request(goal={"x": 2, "y": False}))
        self.assert_invalid(request(start={"x": 0.5, "y": 0}))
        self.assert_invalid(request(goal={"x": "2", "y": 2}))

    def test_coordinates_must_be_within_grid(self):
        self.assert_invalid(request(start={"x": -1, "y": 0}))
        self.assert_invalid(request(start={"x": 0, "y": -1}))
        self.assert_invalid(request(goal={"x": 3, "y": 2}))
        self.assert_invalid(request(goal={"x": 2, "y": 3}))

    def test_endpoints_must_be_passable(self):
        walled = [
            [None, 1, 1],
            [1, 1, 1],
            [1, 1, None],
        ]
        self.assert_invalid(request(grid=walled, start={"x": 0, "y": 0}, goal={"x": 2, "y": 2}))
        self.assert_invalid(request(grid=walled, start={"x": 2, "y": 2}, goal={"x": 1, "y": 1}))

    def test_navigation_error_is_value_error(self):
        with self.assertRaises(NavigationError):
            self.service.find_path(request(grid=[]))


if __name__ == "__main__":
    unittest.main()
