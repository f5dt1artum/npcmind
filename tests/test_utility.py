import copy
import unittest

from npcmind.service import Service


def request(**overrides):
    base = {
        "context": {"hunger": 0.8, "danger": 0.2},
        "options": [
            {
                "id": "eat",
                "base": 2,
                "considerations": [
                    {"key": "hunger", "min": 0, "max": 1, "curve": "linear"},
                    {"key": "danger", "min": 0, "max": 1, "curve": "inverse", "weight": 2},
                ],
            },
            {"id": "idle"},
            {"id": "sleep", "enabled": False},
        ],
    }
    base.update(overrides)
    return base


class SelectUtilityTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_scores_and_selects_highest(self):
        result = self.service.select_utility(request())
        self.assertEqual(result["status"], "SELECTED")
        # eat: 2 * (0.8*1 + 0.8*2) / 3 = 1.6; idle: 1
        self.assertEqual(result["selected"], "eat")
        self.assertAlmostEqual(result["score"], 1.6)
        self.assertEqual([d["id"] for d in result["options"]], ["eat", "idle", "sleep"])

    def test_enabled_option_details(self):
        result = self.service.select_utility(request())
        eat = result["options"][0]
        self.assertTrue(eat["enabled"])
        self.assertAlmostEqual(eat["score"], 1.6)
        self.assertEqual(
            [(c["key"], c["weight"]) for c in eat["considerations"]],
            [("hunger", 1), ("danger", 2)],
        )
        hunger, danger = eat["considerations"]
        self.assertEqual(hunger["value"], 0.8)
        self.assertAlmostEqual(hunger["response"], 0.8)
        self.assertEqual(danger["value"], 0.2)
        self.assertAlmostEqual(danger["response"], 0.8)

    def test_option_without_considerations_scores_base(self):
        result = self.service.select_utility(request())
        idle = result["options"][1]
        self.assertEqual(idle["score"], 1)
        self.assertEqual(idle["considerations"], [])

    def test_disabled_option_has_null_score_and_no_considerations(self):
        result = self.service.select_utility(request())
        sleep = result["options"][2]
        self.assertFalse(sleep["enabled"])
        self.assertIsNone(sleep["score"])
        self.assertEqual(sleep["considerations"], [])

    def test_disabled_option_may_reference_missing_context_keys(self):
        result = self.service.select_utility(
            request(
                options=[
                    {
                        "id": "flee",
                        "enabled": False,
                        "considerations": [
                            {"key": "threat", "min": 0, "max": 10, "curve": "linear"}
                        ],
                    },
                    {"id": "idle"},
                ]
            )
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "idle")
        self.assertIsNone(result["options"][0]["score"])

    def test_all_disabled_returns_no_selection(self):
        result = self.service.select_utility(
            request(options=[{"id": "a", "enabled": False}, {"id": "b", "enabled": False}])
        )
        self.assertEqual(result["status"], "NO_SELECTION")
        self.assertIsNone(result["selected"])
        self.assertIsNone(result["score"])
        self.assertEqual(len(result["options"]), 2)

    def test_tie_goes_to_earlier_option(self):
        result = self.service.select_utility(
            request(options=[{"id": "first", "base": 3}, {"id": "second", "base": 3}])
        )
        self.assertEqual(result["selected"], "first")
        self.assertEqual(result["score"], 3)

    def test_normalisation_is_clamped(self):
        result = self.service.select_utility(
            request(
                context={"level": 99},
                options=[
                    {
                        "id": "fight",
                        "considerations": [
                            {"key": "level", "min": 0, "max": 10, "curve": "linear"}
                        ],
                    }
                ],
            )
        )
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["options"][0]["considerations"][0]["response"], 1.0)

    def test_inverse_curve_below_range_clamps_to_zero_response(self):
        result = self.service.select_utility(
            request(
                context={"level": -5},
                options=[
                    {
                        "id": "fight",
                        "considerations": [
                            {"key": "level", "min": 0, "max": 10, "curve": "inverse"}
                        ],
                    }
                ],
            )
        )
        self.assertEqual(result["options"][0]["considerations"][0]["response"], 1.0)
        self.assertEqual(result["score"], 1)

    def test_weighted_average_uses_weights(self):
        result = self.service.select_utility(
            request(
                context={"a": 1.0, "b": 0.0},
                options=[
                    {
                        "id": "x",
                        "base": 4,
                        "considerations": [
                            {"key": "a", "min": 0, "max": 1, "curve": "linear", "weight": 3},
                            {"key": "b", "min": 0, "max": 1, "curve": "linear", "weight": 1},
                        ],
                    }
                ],
            )
        )
        # 4 * (1*3 + 0*1) / 4 = 3
        self.assertEqual(result["score"], 3)

    def test_request_is_not_mutated(self):
        payload = request()
        snapshot = copy.deepcopy(payload)
        self.service.select_utility(payload)
        self.assertEqual(payload, snapshot)


class SelectUtilityValidationTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def assert_invalid(self, payload):
        with self.assertRaises(ValueError):
            self.service.select_utility(payload)

    def test_request_must_be_object(self):
        self.assert_invalid([])

    def test_context_must_be_object(self):
        self.assert_invalid(request(context=None))

    def test_context_keys_must_be_non_empty_strings(self):
        self.assert_invalid(request(context={"": 1}))

    def test_context_values_must_be_finite_numbers(self):
        self.assert_invalid(request(context={"hunger": "high"}))
        self.assert_invalid(request(context={"hunger": True}))
        self.assert_invalid(request(context={"hunger": float("nan")}))
        self.assert_invalid(request(context={"hunger": float("inf")}))

    def test_options_must_be_non_empty_list(self):
        self.assert_invalid(request(options=[]))
        self.assert_invalid(request(options=None))

    def test_option_requires_unique_non_empty_id(self):
        self.assert_invalid(request(options=[{"id": ""}]))
        self.assert_invalid(request(options=[{"id": "a"}, {"id": "a"}]))

    def test_enabled_must_be_boolean(self):
        self.assert_invalid(request(options=[{"id": "a", "enabled": 1}]))

    def test_base_must_be_non_negative_finite_number(self):
        self.assert_invalid(request(options=[{"id": "a", "base": -1}]))
        self.assert_invalid(request(options=[{"id": "a", "base": True}]))
        self.assert_invalid(request(options=[{"id": "a", "base": float("nan")}]))

    def test_consideration_validation(self):
        def option(consideration):
            return request(options=[{"id": "a", "considerations": [consideration]}])

        self.assert_invalid(option({"min": 0, "max": 1, "curve": "linear"}))  # no key
        self.assert_invalid(option({"key": "hunger", "max": 1, "curve": "linear"}))
        self.assert_invalid(option({"key": "hunger", "min": 0, "curve": "linear"}))
        self.assert_invalid(option({"key": "hunger", "min": 1, "max": 1, "curve": "linear"}))
        self.assert_invalid(option({"key": "hunger", "min": 2, "max": 1, "curve": "linear"}))
        self.assert_invalid(option({"key": "hunger", "min": 0, "max": 1, "curve": "ease"}))
        self.assert_invalid(option({"key": "hunger", "min": 0, "max": 1, "curve": "linear", "weight": 0}))
        self.assert_invalid(option({"key": "hunger", "min": 0, "max": 1, "curve": "linear", "weight": -2}))
        self.assert_invalid(option({"key": "hunger", "min": 0, "max": 1, "curve": "linear", "weight": True}))

    def test_enabled_option_requires_context_key(self):
        self.assert_invalid(
            request(
                options=[
                    {
                        "id": "eat",
                        "considerations": [
                            {"key": "missing", "min": 0, "max": 1, "curve": "linear"}
                        ],
                    }
                ]
            )
        )

    def test_invalid_request_produces_no_partial_result(self):
        payload = request(
            options=[
                {"id": "ok"},
                {"id": "bad", "considerations": [{"key": "hunger", "min": 1, "max": 0, "curve": "linear"}]},
            ]
        )
        with self.assertRaises(ValueError):
            self.service.select_utility(payload)


if __name__ == "__main__":
    unittest.main()
