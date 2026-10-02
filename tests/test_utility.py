import copy
import math
import unittest

from npcmind.service import Service, UtilityError


def request(**overrides):
    base = {
        "context": {"hunger": 0.8, "danger": 0.2},
        "options": [
            {
                "id": "eat",
                "base": 2,
                "considerations": [
                    {"key": "hunger", "min": 0, "max": 1, "curve": "linear"},
                    {"key": "danger", "min": 0, "max": 1, "curve": "inverse", "weight": 3},
                ],
            },
            {"id": "idle"},
        ],
    }
    base.update(overrides)
    return base


class SelectUtilityTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_weighted_average_and_defaults(self):
        result = self.service.select_utility(request())
        self.assertEqual(result["status"], "SELECTED")
        # eat: responses 0.8 (linear) and 0.8 (inverse of 0.2), weights 1 and 3
        # score = 2 * (0.8*1 + 0.8*3) / 4 = 1.6; idle: no considerations -> base 1
        self.assertEqual(result["selected"], "eat")
        self.assertAlmostEqual(result["score"], 1.6)
        candidates = result["candidates"]
        self.assertEqual([c["id"] for c in candidates], ["eat", "idle"])
        eat, idle = candidates
        self.assertTrue(eat["enabled"])
        self.assertAlmostEqual(eat["score"], 1.6)
        self.assertEqual(
            [c["key"] for c in eat["considerations"]], ["hunger", "danger"]
        )
        hunger, danger = eat["considerations"]
        self.assertEqual(hunger["value"], 0.8)
        self.assertAlmostEqual(hunger["response"], 0.8)
        self.assertEqual(hunger["weight"], 1)
        self.assertEqual(danger["value"], 0.2)
        self.assertAlmostEqual(danger["response"], 0.8)
        self.assertEqual(danger["weight"], 3)
        self.assertTrue(idle["enabled"])
        self.assertEqual(idle["score"], 1)
        self.assertEqual(idle["considerations"], [])

    def test_normalization_is_clamped_to_unit_interval(self):
        result = self.service.select_utility(
            request(
                context={"hunger": 5},
                options=[
                    {
                        "id": "a",
                        "considerations": [
                            {"key": "hunger", "min": 0, "max": 1, "curve": "linear"}
                        ],
                    },
                    {
                        "id": "b",
                        "considerations": [
                            {"key": "hunger", "min": 0, "max": 1, "curve": "inverse"}
                        ],
                    },
                ],
            )
        )
        self.assertEqual(result["selected"], "a")
        a, b = result["candidates"]
        self.assertEqual(a["score"], 1)
        self.assertEqual(b["score"], 0)

    def test_tie_prefers_earlier_option(self):
        result = self.service.select_utility(
            request(options=[{"id": "first", "base": 1}, {"id": "second", "base": 1}])
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "first")
        self.assertEqual(result["score"], 1)

    def test_disabled_options_are_skipped_and_report_null_score(self):
        result = self.service.select_utility(
            request(
                options=[
                    {"id": "off", "enabled": False, "base": 100},
                    {"id": "on", "base": 1},
                ]
            )
        )
        self.assertEqual(result["selected"], "on")
        off, on = result["candidates"]
        self.assertFalse(off["enabled"])
        self.assertIsNone(off["score"])
        self.assertEqual(off["considerations"], [])
        self.assertEqual(on["score"], 1)

    def test_all_disabled_is_no_selection_not_an_error(self):
        result = self.service.select_utility(
            request(options=[{"id": "a", "enabled": False}, {"id": "b", "enabled": False}])
        )
        self.assertEqual(result["status"], "NO_SELECTION")
        self.assertIsNone(result["selected"])
        self.assertIsNone(result["score"])
        self.assertEqual(len(result["candidates"]), 2)

    def test_disabled_option_may_reference_missing_context_key(self):
        result = self.service.select_utility(
            request(
                options=[
                    {
                        "id": "off",
                        "enabled": False,
                        "considerations": [
                            {"key": "absent", "min": 0, "max": 1, "curve": "linear"}
                        ],
                    },
                    {"id": "on"},
                ]
            )
        )
        self.assertEqual(result["selected"], "on")

    def test_request_is_not_mutated(self):
        original = request()
        snapshot = copy.deepcopy(original)
        self.service.select_utility(original)
        self.assertEqual(original, snapshot)

    def test_enabled_option_with_missing_context_key_raises(self):
        with self.assertRaises(UtilityError):
            self.service.select_utility(
                request(
                    options=[
                        {
                            "id": "a",
                            "considerations": [
                                {"key": "absent", "min": 0, "max": 1, "curve": "linear"}
                            ],
                        }
                    ]
                )
            )

    def test_duplicate_option_id_raises(self):
        with self.assertRaises(UtilityError):
            self.service.select_utility(request(options=[{"id": "a"}, {"id": "a"}]))

    def test_empty_options_raises(self):
        with self.assertRaises(UtilityError):
            self.service.select_utility(request(options=[]))

    def test_non_object_request_raises(self):
        with self.assertRaises(UtilityError):
            self.service.select_utility([1, 2, 3])

    def test_context_must_be_object_with_finite_number_values(self):
        with self.assertRaises(UtilityError):
            self.service.select_utility(request(context=None))
        with self.assertRaises(UtilityError):
            self.service.select_utility(request(context={"hunger": True}))
        with self.assertRaises(UtilityError):
            self.service.select_utility(request(context={"hunger": float("nan")}))
        with self.assertRaises(UtilityError):
            self.service.select_utility(request(context={"hunger": "high"}))

    def test_base_must_be_non_negative_finite_number(self):
        with self.assertRaises(UtilityError):
            self.service.select_utility(request(options=[{"id": "a", "base": -1}]))
        with self.assertRaises(UtilityError):
            self.service.select_utility(request(options=[{"id": "a", "base": True}]))
        with self.assertRaises(UtilityError):
            self.service.select_utility(
                request(options=[{"id": "a", "base": float("inf")}])
            )

    def test_enabled_must_be_boolean(self):
        with self.assertRaises(UtilityError):
            self.service.select_utility(request(options=[{"id": "a", "enabled": 1}]))

    def test_invalid_consideration_range_raises(self):
        for bounds in ({"min": 1, "max": 1}, {"min": 2, "max": 1}):
            with self.assertRaises(UtilityError):
                self.service.select_utility(
                    request(
                        options=[
                            {
                                "id": "a",
                                "considerations": [
                                    {"key": "hunger", "curve": "linear", **bounds}
                                ],
                            }
                        ]
                    )
                )

    def test_unknown_curve_raises(self):
        with self.assertRaises(UtilityError):
            self.service.select_utility(
                request(
                    options=[
                        {
                            "id": "a",
                            "considerations": [
                                {"key": "hunger", "min": 0, "max": 1, "curve": "sigmoid"}
                            ],
                        }
                    ]
                )
            )

    def test_weight_must_be_positive_finite_number(self):
        for bad_weight in (0, -1, True, float("nan")):
            with self.assertRaises(UtilityError):
                self.service.select_utility(
                    request(
                        options=[
                            {
                                "id": "a",
                                "considerations": [
                                    {
                                        "key": "hunger",
                                        "min": 0,
                                        "max": 1,
                                        "curve": "linear",
                                        "weight": bad_weight,
                                    }
                                ],
                            }
                        ]
                    )
                )

    def test_non_finite_consideration_bound_raises(self):
        with self.assertRaises(UtilityError):
            self.service.select_utility(
                request(
                    options=[
                        {
                            "id": "a",
                            "considerations": [
                                {
                                    "key": "hunger",
                                    "min": 0,
                                    "max": math.inf,
                                    "curve": "linear",
                                }
                            ],
                        }
                    ]
                )
            )

    def test_disabled_option_is_still_validated_structurally(self):
        with self.assertRaises(UtilityError):
            self.service.select_utility(
                request(
                    options=[
                        {
                            "id": "off",
                            "enabled": False,
                            "considerations": [
                                {"key": "hunger", "min": 0, "max": 1, "curve": "bogus"}
                            ],
                        }
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
