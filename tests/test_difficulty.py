import copy
import unittest

from npcmind.service import DifficultyError, Service


def request(**overrides):
    base = {
        "current_difficulty": 0.5,
        "target": {"min": 0.4, "max": 0.6},
        "max_step": 0.1,
        "now": 100.0,
        "cooldown": 10.0,
        "signals": [
            {"id": "win_rate", "value": 0.9, "weight": 2.0},
            {"id": "deaths", "value": 0.3, "weight": 1.0},
        ],
    }
    base.update(overrides)
    return base


class AdjustDifficultyTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def adjust(self, payload):
        return self.service.adjust_difficulty(copy.deepcopy(payload))

    def test_score_above_target_increases_difficulty(self):
        result = self.adjust(request())
        # score = (0.9 * 2 + 0.3 * 1) / 3 = 0.7; overshoot 0.1, step 0.1.
        self.assertEqual(result["status"], "INCREASED")
        self.assertEqual(result["previous_difficulty"], 0.5)
        self.assertAlmostEqual(result["score"], 0.7)
        self.assertAlmostEqual(result["difficulty"], 0.6)
        self.assertAlmostEqual(result["adjustment"], 0.1)

    def test_increase_is_capped_by_max_step(self):
        result = self.adjust(request(max_step=0.02))
        self.assertEqual(result["status"], "INCREASED")
        self.assertAlmostEqual(result["difficulty"], 0.52)
        self.assertAlmostEqual(result["adjustment"], 0.02)

    def test_increase_is_capped_by_headroom(self):
        result = self.adjust(request(current_difficulty=0.95))
        self.assertEqual(result["status"], "INCREASED")
        self.assertAlmostEqual(result["difficulty"], 1.0)
        self.assertAlmostEqual(result["adjustment"], 0.05)

    def test_increase_at_upper_boundary_is_unchanged(self):
        result = self.adjust(request(current_difficulty=1.0))
        self.assertEqual(result["status"], "UNCHANGED")
        self.assertEqual(result["difficulty"], 1.0)
        self.assertEqual(result["adjustment"], 0)

    def test_score_below_target_decreases_difficulty(self):
        result = self.adjust(
            request(
                signals=[
                    {"id": "win_rate", "value": 0.1, "weight": 1.0},
                    {"id": "deaths", "value": 0.4, "weight": 1.0},
                ]
            )
        )
        # score = 0.25; undershoot 0.15, step 0.1.
        self.assertEqual(result["status"], "DECREASED")
        self.assertAlmostEqual(result["score"], 0.25)
        self.assertAlmostEqual(result["difficulty"], 0.4)
        self.assertAlmostEqual(result["adjustment"], -0.1)

    def test_decrease_is_capped_by_current_difficulty(self):
        result = self.adjust(
            request(
                current_difficulty=0.03,
                signals=[{"id": "win_rate", "value": 0.0, "weight": 1.0}],
            )
        )
        self.assertEqual(result["status"], "DECREASED")
        self.assertAlmostEqual(result["difficulty"], 0.0)
        self.assertAlmostEqual(result["adjustment"], -0.03)

    def test_decrease_at_lower_boundary_is_unchanged(self):
        result = self.adjust(
            request(
                current_difficulty=0.0,
                signals=[{"id": "win_rate", "value": 0.0, "weight": 1.0}],
            )
        )
        self.assertEqual(result["status"], "UNCHANGED")
        self.assertEqual(result["difficulty"], 0.0)
        self.assertEqual(result["adjustment"], 0)

    def test_score_inside_target_interval_is_unchanged(self):
        for value in (0.4, 0.5, 0.6):  # endpoints included
            result = self.adjust(
                request(signals=[{"id": "s", "value": value, "weight": 1.0}])
            )
            self.assertEqual(result["status"], "UNCHANGED", value)
            self.assertEqual(result["difficulty"], 0.5)
            self.assertEqual(result["adjustment"], 0)
            self.assertAlmostEqual(result["score"], value)

    def test_score_is_the_weighted_average(self):
        result = self.adjust(
            request(
                signals=[
                    {"id": "a", "value": 1.0, "weight": 3.0},
                    {"id": "b", "value": 0.0, "weight": 1.0},
                ]
            )
        )
        self.assertAlmostEqual(result["score"], 0.75)

    def test_cooldown_blocks_adjustment(self):
        result = self.adjust(request(last_adjusted_at=95.0))
        self.assertEqual(result["status"], "COOLDOWN")
        self.assertEqual(result["previous_difficulty"], 0.5)
        self.assertEqual(result["difficulty"], 0.5)
        self.assertEqual(result["adjustment"], 0)
        self.assertAlmostEqual(result["score"], 0.7)

    def test_cooldown_boundary_allows_adjustment(self):
        result = self.adjust(request(last_adjusted_at=90.0))
        self.assertEqual(result["status"], "INCREASED")
        self.assertAlmostEqual(result["difficulty"], 0.6)

    def test_null_or_missing_last_adjusted_at_allows_adjustment(self):
        for payload in (request(last_adjusted_at=None), request()):
            result = self.adjust(payload)
            self.assertEqual(result["status"], "INCREASED")

    def test_integer_fields_are_accepted(self):
        result = self.adjust(
            request(
                current_difficulty=1,
                target={"min": 0, "max": 1},
                max_step=1,
                now=100,
                cooldown=0,
                signals=[{"id": "s", "value": 1, "weight": 1}],
                last_adjusted_at=100,
            )
        )
        self.assertEqual(result["status"], "UNCHANGED")
        self.assertEqual(result["difficulty"], 1)

    def test_request_is_not_mutated(self):
        payload = request(last_adjusted_at=80.0)
        snapshot = copy.deepcopy(payload)
        self.service.adjust_difficulty(payload)
        self.assertEqual(payload, snapshot)

    def test_invalid_requests_raise_value_error(self):
        bad_requests = [
            "not a dict",
            {},
            request(current_difficulty=True),
            request(current_difficulty=-0.1),
            request(current_difficulty=1.1),
            request(current_difficulty=float("nan")),
            request(current_difficulty=float("inf")),
            request(target=None),
            request(target={"max": 0.6}),
            request(target={"min": 0.4}),
            request(target={"min": True, "max": 0.6}),
            request(target={"min": 0.4, "max": False}),
            request(target={"min": -0.1, "max": 0.6}),
            request(target={"min": 0.4, "max": 1.1}),
            request(target={"min": 0.7, "max": 0.6}),
            request(target={"min": float("nan"), "max": 0.6}),
            request(max_step=0),
            request(max_step=-0.5),
            request(max_step=1.5),
            request(max_step=True),
            request(max_step=float("inf")),
            request(now=-1.0),
            request(now=True),
            request(now=float("nan")),
            request(cooldown=-1.0),
            request(cooldown=False),
            request(cooldown=float("inf")),
            request(signals=[]),
            request(signals="nope"),
            request(signals=["nope"]),
            request(signals=[{"value": 0.5, "weight": 1.0}]),
            request(signals=[{"id": "", "value": 0.5, "weight": 1.0}]),
            request(signals=[{"id": 1, "value": 0.5, "weight": 1.0}]),
            request(
                signals=[
                    {"id": "s", "value": 0.5, "weight": 1.0},
                    {"id": "s", "value": 0.6, "weight": 1.0},
                ]
            ),
            request(signals=[{"id": "s", "weight": 1.0}]),
            request(signals=[{"id": "s", "value": True, "weight": 1.0}]),
            request(signals=[{"id": "s", "value": -0.1, "weight": 1.0}]),
            request(signals=[{"id": "s", "value": 1.1, "weight": 1.0}]),
            request(signals=[{"id": "s", "value": float("nan"), "weight": 1.0}]),
            request(signals=[{"id": "s", "value": 0.5}]),
            request(signals=[{"id": "s", "value": 0.5, "weight": 0}]),
            request(signals=[{"id": "s", "value": 0.5, "weight": -1.0}]),
            request(signals=[{"id": "s", "value": 0.5, "weight": True}]),
            request(signals=[{"id": "s", "value": 0.5, "weight": float("inf")}]),
            request(last_adjusted_at=-1.0),
            request(last_adjusted_at=True),
            request(last_adjusted_at=float("nan")),
            request(last_adjusted_at=101.0),
        ]
        for bad in bad_requests:
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.service.adjust_difficulty(copy.deepcopy(bad))

    def test_error_type_is_difficulty_error(self):
        with self.assertRaises(DifficultyError):
            self.service.adjust_difficulty({})

    def test_no_partial_result_on_invalid_input(self):
        payload = request(signals=[{"id": "s", "value": 0.5, "weight": 1.0}], now=-1.0)
        with self.assertRaises(ValueError):
            self.service.adjust_difficulty(payload)


if __name__ == "__main__":
    unittest.main()
