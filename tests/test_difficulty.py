import copy
import unittest

from npcmind.service import Service


def request(**overrides):
    base = {
        "current_difficulty": 0.5,
        "target": {"min": 0.3, "max": 0.6},
        "max_step": 0.2,
        "now": 100.0,
        "cooldown": 10.0,
        "last_adjusted_at": None,
        "signals": [
            {"id": "win_rate", "value": 0.9, "weight": 1.0},
            {"id": "clear_time", "value": 0.3, "weight": 1.0},
        ],
    }
    base.update(overrides)
    return base


class AdjustDifficultyTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_score_is_weighted_average(self):
        # (0.9 * 3 + 0.6 * 1) / 4 == 0.825
        result = self.service.adjust_difficulty(
            request(
                signals=[
                    {"id": "a", "value": 0.9, "weight": 3},
                    {"id": "b", "value": 0.6, "weight": 1},
                ]
            )
        )
        self.assertAlmostEqual(result["score"], 0.825)

    def test_increase_takes_minimum_of_gap_step_and_headroom(self):
        # score 0.6 -> above max; step = min(0.6 - 0.5, 0.2, 1 - 0.5)
        result = self.service.adjust_difficulty(
            request(
                target={"min": 0.2, "max": 0.5},
                signals=[{"id": "a", "value": 0.6, "weight": 1}],
            )
        )
        self.assertEqual(result["status"], "INCREASED")
        self.assertEqual(result["previous_difficulty"], 0.5)
        self.assertAlmostEqual(result["difficulty"], 0.6)
        self.assertAlmostEqual(result["adjustment"], 0.1)

    def test_increase_is_capped_by_max_step(self):
        # gap 0.4, max_step 0.2, headroom 0.5 -> 0.2
        result = self.service.adjust_difficulty(
            request(
                target={"min": 0.2, "max": 0.5},
                signals=[{"id": "a", "value": 0.9, "weight": 1}],
            )
        )
        self.assertEqual(result["status"], "INCREASED")
        self.assertAlmostEqual(result["difficulty"], 0.7)
        self.assertAlmostEqual(result["adjustment"], 0.2)

    def test_increase_is_capped_by_headroom(self):
        # gap 0.4, max_step 0.2, headroom 0.05 -> 0.05; still INCREASED
        result = self.service.adjust_difficulty(
            request(
                current_difficulty=0.95,
                target={"min": 0.2, "max": 0.5},
                signals=[{"id": "a", "value": 0.9, "weight": 1}],
            )
        )
        self.assertEqual(result["status"], "INCREASED")
        self.assertAlmostEqual(result["difficulty"], 1.0)
        self.assertAlmostEqual(result["adjustment"], 0.05)

    def test_increase_at_the_top_boundary_is_unchanged(self):
        result = self.service.adjust_difficulty(
            request(
                current_difficulty=1.0,
                signals=[{"id": "a", "value": 0.9, "weight": 1}],
            )
        )
        self.assertEqual(result["status"], "UNCHANGED")
        self.assertEqual(result["difficulty"], 1.0)
        self.assertEqual(result["adjustment"], 0)

    def test_decrease_takes_minimum_of_gap_step_and_current(self):
        # score 0.2 -> below min; step = min(0.3 - 0.2, 0.2, 0.5) = 0.1
        result = self.service.adjust_difficulty(
            request(
                target={"min": 0.3, "max": 0.6},
                signals=[{"id": "a", "value": 0.2, "weight": 1}],
            )
        )
        self.assertEqual(result["status"], "DECREASED")
        self.assertAlmostEqual(result["difficulty"], 0.4)
        self.assertAlmostEqual(result["adjustment"], -0.1)

    def test_decrease_is_capped_by_max_step(self):
        # gap 0.25, max_step 0.2, current 0.5 -> 0.2
        result = self.service.adjust_difficulty(
            request(signals=[{"id": "a", "value": 0.05, "weight": 1}])
        )
        self.assertEqual(result["status"], "DECREASED")
        self.assertAlmostEqual(result["difficulty"], 0.3)
        self.assertAlmostEqual(result["adjustment"], -0.2)

    def test_decrease_is_capped_at_zero(self):
        # gap 0.25, max_step 0.2, current 0.05 -> 0.05; still DECREASED
        result = self.service.adjust_difficulty(
            request(
                current_difficulty=0.05,
                signals=[{"id": "a", "value": 0.05, "weight": 1}],
            )
        )
        self.assertEqual(result["status"], "DECREASED")
        self.assertAlmostEqual(result["difficulty"], 0.0)
        self.assertAlmostEqual(result["adjustment"], -0.05)

    def test_decrease_at_zero_is_unchanged(self):
        result = self.service.adjust_difficulty(
            request(
                current_difficulty=0,
                signals=[{"id": "a", "value": 0.05, "weight": 1}],
            )
        )
        self.assertEqual(result["status"], "UNCHANGED")
        self.assertEqual(result["difficulty"], 0)
        self.assertEqual(result["adjustment"], 0)

    def test_closed_interval_including_endpoints_is_unchanged(self):
        for value in (0.3, 0.45, 0.6):
            result = self.service.adjust_difficulty(
                request(signals=[{"id": "a", "value": value, "weight": 1}])
            )
            self.assertEqual(result["status"], "UNCHANGED", value)
            self.assertEqual(result["difficulty"], 0.5)
            self.assertEqual(result["adjustment"], 0)
            self.assertAlmostEqual(result["score"], value)

    def test_cooldown_holds_difficulty_but_still_reports_score(self):
        result = self.service.adjust_difficulty(
            request(
                now=105.0,
                cooldown=10.0,
                last_adjusted_at=100.0,
                signals=[{"id": "a", "value": 0.9, "weight": 1}],
            )
        )
        self.assertEqual(result["status"], "COOLDOWN")
        self.assertEqual(result["previous_difficulty"], 0.5)
        self.assertEqual(result["difficulty"], 0.5)
        self.assertEqual(result["adjustment"], 0)
        self.assertAlmostEqual(result["score"], 0.9)

    def test_gap_equal_to_cooldown_allows_adjustment(self):
        result = self.service.adjust_difficulty(
            request(
                now=110.0,
                cooldown=10.0,
                last_adjusted_at=100.0,
                signals=[{"id": "a", "value": 0.9, "weight": 1}],
            )
        )
        self.assertEqual(result["status"], "INCREASED")
        self.assertAlmostEqual(result["difficulty"], 0.7)

    def test_zero_cooldown_with_past_timestamp_allows_adjustment(self):
        result = self.service.adjust_difficulty(
            request(
                now=100.0,
                cooldown=0.0,
                last_adjusted_at=99.0,
                signals=[{"id": "a", "value": 0.9, "weight": 1}],
            )
        )
        self.assertEqual(result["status"], "INCREASED")

    def test_null_or_omitted_timestamp_adjusts(self):
        payload = request(signals=[{"id": "a", "value": 0.9, "weight": 1}])
        self.assertEqual(self.service.adjust_difficulty(payload)["status"], "INCREASED")
        payload.pop("last_adjusted_at")
        self.assertEqual(self.service.adjust_difficulty(payload)["status"], "INCREASED")

    def test_request_is_not_mutated(self):
        payload = request()
        snapshot = copy.deepcopy(payload)
        self.service.adjust_difficulty(payload)
        self.assertEqual(payload, snapshot)

    def test_invalid_requests_raise_value_error(self):
        bad_requests = [
            "not a dict",
            None,
            request(current_difficulty=None),
            request(current_difficulty=-0.1),
            request(current_difficulty=1.1),
            request(current_difficulty=True),
            request(current_difficulty="0.5"),
            request(current_difficulty=float("nan")),
            request(current_difficulty=float("inf")),
            request(target=None),
            request(target={"min": 0.3}),
            request(target={"max": 0.6}),
            request(target={"min": -0.1, "max": 0.6}),
            request(target={"min": 0.3, "max": 1.1}),
            request(target={"min": True, "max": 0.6}),
            request(target={"min": 0.3, "max": float("nan")}),
            request(target={"min": 0.7, "max": 0.6}),
            request(max_step=0),
            request(max_step=-0.1),
            request(max_step=1.01),
            request(max_step=True),
            request(max_step=float("nan")),
            request(now=-0.01),
            request(now=True),
            request(now=float("inf")),
            request(cooldown=-1),
            request(cooldown=False),
            request(cooldown=float("nan")),
            request(last_adjusted_at=-1),
            request(last_adjusted_at=101.0),
            request(last_adjusted_at=True),
            request(last_adjusted_at="100"),
            request(last_adjusted_at=float("inf")),
            request(signals=[]),
            request(signals="not a list"),
            request(signals=["not a dict"]),
            request(signals=[{"value": 0.5, "weight": 1}]),
            request(signals=[{"id": "", "value": 0.5, "weight": 1}]),
            request(
                signals=[
                    {"id": "a", "value": 0.5, "weight": 1},
                    {"id": "a", "value": 0.5, "weight": 1},
                ]
            ),
            request(signals=[{"id": "a", "weight": 1}]),
            request(signals=[{"id": "a", "value": -0.1, "weight": 1}]),
            request(signals=[{"id": "a", "value": 1.1, "weight": 1}]),
            request(signals=[{"id": "a", "value": True, "weight": 1}]),
            request(signals=[{"id": "a", "value": float("nan"), "weight": 1}]),
            request(signals=[{"id": "a", "value": 0.5}]),
            request(signals=[{"id": "a", "value": 0.5, "weight": 0}]),
            request(signals=[{"id": "a", "value": 0.5, "weight": -1}]),
            request(signals=[{"id": "a", "value": 0.5, "weight": True}]),
            request(signals=[{"id": "a", "value": 0.5, "weight": float("inf")}]),
        ]
        for bad in bad_requests:
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.service.adjust_difficulty(bad)

    def test_no_partial_result_on_invalid_signal_after_valid_ones(self):
        with self.assertRaises(ValueError):
            self.service.adjust_difficulty(
                request(
                    signals=[
                        {"id": "ok", "value": 0.9, "weight": 1},
                        {"id": "bad", "value": 0.9, "weight": 0},
                    ]
                )
            )

    def test_health_unchanged(self):
        self.assertEqual(
            self.service.health(),
            {"status": "ok", "service": "npcmind", "version": Service.version},
        )


if __name__ == "__main__":
    unittest.main()
