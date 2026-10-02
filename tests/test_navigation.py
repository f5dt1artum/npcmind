import copy
import json
import unittest

from npcmind.service import Service


def grid_request(**overrides):
    base = {
        "grid": [
            [1, 1, 1],
            [1, None, 1],
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

    def test_finds_route_around_obstacle(self):
        result = self.service.find_path(grid_request())
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["cost"], 4)
        # The two routes around the obstacle tie on cost; comparing their
        # full (y, x) sequences lexicographically favours the top route.
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

    def test_path_includes_both_endpoints_and_adjacent_steps(self):
        req = grid_request(grid=[[1, 1, 1]], start={"x": 0, "y": 0}, goal={"x": 2, "y": 0})
        result = self.service.find_path(req)
        path = result["path"]
        self.assertEqual(path[0], {"x": 0, "y": 0})
        self.assertEqual(path[-1], {"x": 2, "y": 0})
        for a, b in zip(path, path[1:]):
            self.assertEqual(abs(a["x"] - b["x"]) + abs(a["y"] - b["y"]), 1)

    def test_cost_excludes_start_and_sums_entered_cells(self):
        req = grid_request(
            grid=[[10, 20, 30]],
            start={"x": 0, "y": 0},
            goal={"x": 2, "y": 0},
        )
        result = self.service.find_path(req)
        self.assertEqual(result["cost"], 20 + 30)

    def test_cheapest_route_beats_shortest(self):
        # Direct route (up the middle column) costs 9+9; the detour along the
        # top row costs 1+1+1+1 even though it takes more steps.
        req = grid_request(
            grid=[
                [1, 1, 1, 1, 1],
                [1, None, 9, None, 1],
                [1, None, 9, None, 1],
                [1, None, 9, None, 1],
                [1, 1, 1, 1, 1],
            ],
            start={"x": 2, "y": 4},
            goal={"x": 2, "y": 0},
        )
        result = self.service.find_path(req)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["cost"], 8)
        self.assertEqual(result["path"][1], {"x": 1, "y": 4})

    def test_equal_cost_tie_breaks_on_y_then_x_sequences(self):
        # From (1,1) to (1,0) on a fully open uniform grid the routes via
        # (0,1) and (2,1) tie; the one reaching the smaller y first wins.
        req = grid_request(
            grid=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            start={"x": 1, "y": 1},
            goal={"x": 1, "y": 0},
        )
        result = self.service.find_path(req)
        self.assertEqual(
            result["path"],
            [
                {"x": 1, "y": 1},
                {"x": 1, "y": 0},
            ],
        )

    def test_tie_among_equal_length_detours_prefers_smaller_x(self):
        # Goal is above an obstacle; going via x=0 vs x=2 costs the same.
        req = grid_request(
            grid=[
                [1, 1, 1],
                [1, None, 1],
                [1, 1, 1],
            ],
            start={"x": 1, "y": 2},
            goal={"x": 1, "y": 0},
        )
        result = self.service.find_path(req)
        self.assertEqual(result["cost"], 4)
        self.assertEqual(
            result["path"],
            [
                {"x": 1, "y": 2},
                {"x": 0, "y": 2},
                {"x": 0, "y": 1},
                {"x": 0, "y": 0},
                {"x": 1, "y": 0},
            ],
        )

    def test_start_equals_goal(self):
        req = grid_request(start={"x": 1, "y": 1}, goal={"x": 1, "y": 1})
        req["grid"][1][1] = 5
        result = self.service.find_path(req)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["path"], [{"x": 1, "y": 1}])
        self.assertEqual(result["cost"], 0)

    def test_unreachable_goal_is_not_an_error(self):
        req = grid_request(
            grid=[[1, 1, None], [None, None, 1], [1, 1, None]],
            start={"x": 0, "y": 0},
            goal={"x": 1, "y": 2},
        )
        result = self.service.find_path(req)
        self.assertEqual(result["status"], "UNREACHABLE")
        self.assertEqual(result["path"], [])
        self.assertIsNone(result["cost"])

    def test_single_cell_grid(self):
        result = self.service.find_path(
            grid_request(grid=[[7]], start={"x": 0, "y": 0}, goal={"x": 0, "y": 0})
        )
        self.assertEqual(result["path"], [{"x": 0, "y": 0}])
        self.assertEqual(result["cost"], 0)

    def test_request_is_not_mutated(self):
        req = grid_request()
        snapshot = copy.deepcopy(req)
        self.service.find_path(req)
        self.assertEqual(req, snapshot)
        # Deep equality also catches added/removed keys on nested objects.
        self.assertEqual(json.dumps(req, sort_keys=True), json.dumps(snapshot, sort_keys=True))

    def test_repeated_calls_are_independent(self):
        req = grid_request()
        first = self.service.find_path(req)
        second = self.service.find_path(req)
        self.assertEqual(first, second)

    def test_returned_path_is_fresh_data(self):
        req = grid_request()
        result = self.service.find_path(req)
        result["path"].append({"x": 9, "y": 9})
        again = self.service.find_path(req)
        self.assertNotIn({"x": 9, "y": 9}, again["path"])

    def assert_invalid(self, req):
        with self.assertRaises(ValueError):
            self.service.find_path(req)

    def test_request_must_be_object(self):
        self.assert_invalid([])
        self.assert_invalid("nope")
        self.assert_invalid(None)

    def test_required_fields(self):
        self.assert_invalid({"grid": [[1]], "start": {"x": 0, "y": 0}})
        self.assert_invalid({"start": {"x": 0, "y": 0}, "goal": {"x": 0, "y": 0}})
        self.assert_invalid({"grid": [[1]], "goal": {"x": 0, "y": 0}})

    def test_coordinates_must_be_objects(self):
        self.assert_invalid(grid_request(start=[0, 0]))
        self.assert_invalid(grid_request(goal="0,0"))
        self.assert_invalid(grid_request(start=None))

    def test_coordinate_fields_required(self):
        self.assert_invalid(grid_request(start={"y": 0}))
        self.assert_invalid(grid_request(goal={"x": 2}))
        self.assert_invalid(grid_request(start={}))

    def test_coordinate_fields_must_be_integers(self):
        for bad in (True, False, 1.0, "1", 0.5, None):
            self.assert_invalid(grid_request(start={"x": bad, "y": 0}))
            self.assert_invalid(grid_request(start={"x": 0, "y": bad}))
            self.assert_invalid(grid_request(goal={"x": bad, "y": 2}))
            self.assert_invalid(grid_request(goal={"x": 2, "y": bad}))

    def test_grid_must_be_non_empty_rectangle(self):
        self.assert_invalid(grid_request(grid=[]))
        self.assert_invalid(grid_request(grid=[[]]))
        self.assert_invalid(grid_request(grid="x"))
        self.assert_invalid(grid_request(grid={}))
        self.assert_invalid(grid_request(grid=[[1, 1], [1]]))
        self.assert_invalid(grid_request(grid=[[1], [1, 1]]))
        self.assert_invalid(grid_request(grid=[1, 1]))
        self.assert_invalid(grid_request(grid=[[1], "x"]))

    def test_cell_values_must_be_null_or_positive_integer(self):
        for bad in (0, -1, True, False, 1.5, 2.0, "1", [], {}):
            self.assert_invalid(grid_request(grid=[[1, 1], [1, bad]]))

    def test_null_cells_are_passable_only_when_entering_is_not_required(self):
        # null cells are legal grid values; merely surrounding them works.
        result = self.service.find_path(
            grid_request(
                grid=[[1, None, 1], [1, 1, 1]],
                start={"x": 0, "y": 0},
                goal={"x": 0, "y": 1},
            )
        )
        self.assertEqual(result["status"], "SUCCESS")

    def test_coordinates_out_of_bounds(self):
        self.assert_invalid(grid_request(start={"x": -1, "y": 0}))
        self.assert_invalid(grid_request(start={"x": 3, "y": 0}))
        self.assert_invalid(grid_request(start={"x": 0, "y": 3}))
        self.assert_invalid(grid_request(goal={"x": 0, "y": -1}))
        self.assert_invalid(grid_request(goal={"x": 99, "y": 0}))

    def test_endpoints_must_be_passable(self):
        self.assert_invalid(grid_request(start={"x": 1, "y": 1}))
        self.assert_invalid(grid_request(goal={"x": 1, "y": 1}))

    def test_validation_runs_before_any_partial_route(self):
        # Valid-looking start but an illegal cell deep in the grid still
        # raises rather than returning a path.
        self.assert_invalid(
            grid_request(
                grid=[[1, 1, 1], [1, None, 1], [1, 1, -2]],
            )
        )


if __name__ == "__main__":
    unittest.main()
