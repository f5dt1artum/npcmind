import copy
import math
import unittest

from npcmind.service import ScheduleError, Service


def activity(id, start_minute, end_minute, **extra):
    record = {
        "id": id,
        "start_minute": start_minute,
        "end_minute": end_minute,
    }
    record.update(extra)
    return record


def request(**overrides):
    base = {
        "now": 600,
        "needs": {"hunger": 0.2, "fun": 0.1},
        "activities": [activity("patrol", 540, 660)],
    }
    base.update(overrides)
    return base


class SelectScheduleActivityTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_scheduled_activity_is_selected(self):
        result = self.service.select_schedule_activity(request())
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "patrol")
        self.assertEqual(result["needs"], {"hunger": 0.2, "fun": 0.1})
        self.assertEqual(
            result["evaluations"],
            [
                {
                    "id": "patrol",
                    "scheduled": True,
                    "urgent": False,
                    "eligible": True,
                    "need_level": None,
                }
            ],
        )

    def test_window_is_start_inclusive_end_exclusive(self):
        for now, in_window in ((540, True), (599, True), (660, False), (539, False)):
            with self.subTest(now=now):
                result = self.service.select_schedule_activity(request(now=now))
                self.assertEqual(result["status"], "SELECTED" if in_window else "IDLE")
                self.assertEqual(result["evaluations"][0]["scheduled"], in_window)

    def test_wraparound_window(self):
        payload = request(activities=[activity("night", 1320, 240)])
        for now, in_window in (
            (1319, False),
            (1320, True),
            (1439, True),
            (0, True),
            (239, True),
            (240, False),
            (800, False),
        ):
            with self.subTest(now=now):
                result = self.service.select_schedule_activity(request(now=now, activities=[activity("night", 1320, 240)]))
                self.assertEqual(result["evaluations"][0]["scheduled"], in_window)

    def test_equal_bounds_cover_the_whole_day(self):
        for now in (0, 600, 1439):
            with self.subTest(now=now):
                result = self.service.select_schedule_activity(
                    request(now=now, activities=[activity("allday", 600, 600)])
                )
                self.assertEqual(result["status"], "SELECTED")
                self.assertTrue(result["evaluations"][0]["scheduled"])

    def test_scheduled_uses_priority_then_input_order(self):
        activities = [
            activity("first_low", 540, 660, priority=0),
            activity("high", 540, 660, priority=5),
            activity("second_low", 540, 660, priority=0),
        ]
        result = self.service.select_schedule_activity(request(activities=activities))
        self.assertEqual(result["selected"], "high")

        tied = [
            activity("first", 540, 660, priority=2),
            activity("second", 540, 660, priority=2),
        ]
        result = self.service.select_schedule_activity(request(activities=tied))
        self.assertEqual(result["selected"], "first")

    def test_priority_defaults_to_zero(self):
        result = self.service.select_schedule_activity(
            request(
                activities=[
                    activity("plain", 540, 660),
                    activity("negative", 540, 660, priority=-1),
                ]
            )
        )
        self.assertEqual(result["selected"], "plain")

    def test_urgent_activity_beats_scheduled_even_with_lower_priority(self):
        payload = request(
            needs={"hunger": 0.9},
            activities=[
                activity("patrol", 540, 660, priority=10),
                activity(
                    "eat", 0, 300, priority=0, need="hunger", trigger=0.5, relief=0.8
                ),
            ],
        )
        result = self.service.select_schedule_activity(payload)
        self.assertEqual(result["selected"], "eat")
        eat_eval = result["evaluations"][1]
        self.assertFalse(eat_eval["scheduled"])
        self.assertTrue(eat_eval["urgent"])
        self.assertTrue(eat_eval["eligible"])

    def test_urgent_orders_by_need_level_then_priority_then_input_order(self):
        def urgent(ids_with_needs):
            return [
                activity(
                    item[0],
                    0,
                    300,
                    priority=item[2],
                    need=item[1],
                    trigger=0.5,
                    relief=0.1,
                )
                for item in ids_with_needs
            ]

        # Higher need value wins regardless of priority.
        result = self.service.select_schedule_activity(
            request(
                needs={"hunger": 0.9, "fun": 0.8},
                activities=urgent([("low_need_high_prio", "fun", 10), ("hungry", "hunger", 0)]),
            )
        )
        self.assertEqual(result["selected"], "hungry")

        # Equal need value: higher priority wins.
        result = self.service.select_schedule_activity(
            request(
                needs={"hunger": 0.8, "fun": 0.8},
                activities=urgent([("prio_one", "fun", 1), ("prio_five", "hunger", 5)]),
            )
        )
        self.assertEqual(result["selected"], "prio_five")

        # Same activity need, equal priority: input order wins.
        result = self.service.select_schedule_activity(
            request(
                needs={"hunger": 0.8},
                activities=urgent([("first", "hunger", 0), ("second", "hunger", 0)]),
            )
        )
        self.assertEqual(result["selected"], "first")

    def test_trigger_boundary(self):
        activities = [
            activity("eat", 0, 300, need="hunger", trigger=0.5, relief=0.1)
        ]
        self.assertEqual(
            self.service.select_schedule_activity(
                request(now=600, needs={"hunger": 0.5}, activities=activities)
            )["status"],
            "SELECTED",
        )
        idle = self.service.select_schedule_activity(
            request(now=600, needs={"hunger": 0.499999}, activities=activities)
        )
        self.assertEqual(idle["status"], "IDLE")
        self.assertFalse(idle["evaluations"][0]["urgent"])

    def test_activity_in_window_and_urgent_reports_both_flags(self):
        payload = request(
            needs={"hunger": 0.9},
            activities=[
                activity(
                    "eat", 540, 660, need="hunger", trigger=0.5, relief=0.1, priority=0
                ),
                activity("patrol", 540, 660, priority=10),
            ],
        )
        result = self.service.select_schedule_activity(payload)
        self.assertEqual(result["selected"], "eat")
        evaluation = result["evaluations"][0]
        self.assertTrue(evaluation["scheduled"])
        self.assertTrue(evaluation["urgent"])
        self.assertTrue(evaluation["eligible"])
        self.assertEqual(evaluation["need_level"], 0.9)

    def test_relief_updates_only_the_associated_need(self):
        payload = request(
            now=600,
            needs={"hunger": 1.0, "fun": 0.4},
            activities=[
                activity(
                    "eat", 0, 300, need="hunger", trigger=0.5, relief=0.25
                )
            ],
        )
        result = self.service.select_schedule_activity(payload)
        self.assertEqual(result["needs"], {"hunger": 0.75, "fun": 0.4})

    def test_relief_clamps_at_zero(self):
        result = self.service.select_schedule_activity(
            request(
                now=600,
                needs={"hunger": 0.2},
                activities=[
                    activity("eat", 0, 300, need="hunger", trigger=0.1, relief=0.5)
                ],
            )
        )
        self.assertEqual(result["needs"], {"hunger": 0})

    def test_needless_activity_leaves_needs_untouched(self):
        payload = request(needs={"hunger": 0.2}, activities=[activity("patrol", 540, 660)])
        result = self.service.select_schedule_activity(payload)
        self.assertEqual(result["needs"], {"hunger": 0.2})

    def test_idle_returns_original_needs_and_full_evaluations(self):
        activities = [
            activity("patrol", 100, 200),
            activity("eat", 100, 200, need="hunger", trigger=0.9, relief=0.5),
        ]
        payload = request(now=0, needs={"hunger": 0.1}, activities=activities)
        result = self.service.select_schedule_activity(payload)
        self.assertEqual(result["status"], "IDLE")
        self.assertIsNone(result["selected"])
        self.assertEqual(result["needs"], {"hunger": 0.1})
        self.assertEqual(
            result["evaluations"],
            [
                {
                    "id": "patrol",
                    "scheduled": False,
                    "urgent": False,
                    "eligible": False,
                    "need_level": None,
                },
                {
                    "id": "eat",
                    "scheduled": False,
                    "urgent": False,
                    "eligible": False,
                    "need_level": 0.1,
                },
            ],
        )

    def test_evaluations_follow_input_order(self):
        activities = [
            activity(
                "eat", 0, 300, priority=2, need="hunger", trigger=0.5, relief=0.1
            ),
            activity("patrol", 540, 660, priority=9),
            activity(
                "play", 0, 300, priority=1, need="fun", trigger=0.9, relief=0.1
            ),
        ]
        result = self.service.select_schedule_activity(
            request(needs={"hunger": 0.9, "fun": 0.2}, activities=activities)
        )
        self.assertEqual([e["id"] for e in result["evaluations"]], ["eat", "patrol", "play"])
        self.assertEqual(result["selected"], "eat")
        self.assertEqual(
            [(e["scheduled"], e["urgent"], e["eligible"], e["need_level"]) for e in result["evaluations"]],
            [(False, True, True, 0.9), (True, False, True, None), (False, False, False, 0.2)],
        )

    def test_request_is_not_mutated(self):
        payload = request(
            needs={"hunger": 0.9, "fun": 0.4},
            activities=[
                activity("patrol", 540, 660, priority=3),
                activity(
                    "eat", 0, 300, need="hunger", trigger=0.5, relief=0.25, priority=1
                ),
            ],
        )
        snapshot = copy.deepcopy(payload)
        self.service.select_schedule_activity(payload)
        self.assertEqual(payload, snapshot)

    def test_no_state_between_calls(self):
        first = self.service.select_schedule_activity(
            request(
                now=600,
                needs={"hunger": 0.9},
                activities=[
                    activity("eat", 0, 300, need="hunger", trigger=0.5, relief=0.9)
                ],
            )
        )
        self.assertEqual(first["needs"], {"hunger": 0.0})
        second = self.service.select_schedule_activity(
            request(
                now=600,
                needs={"hunger": 0.9},
                activities=[
                    activity("eat", 0, 300, need="hunger", trigger=0.5, relief=0.9)
                ],
            )
        )
        self.assertEqual(second, first)

    def test_invalid_requests_raise_value_error(self):
        bad_requests = [
            "not a dict",
            {},
            request(now=600.0),
            request(now=True),
            request(now="600"),
            request(now=-1),
            request(now=1440),
            request(needs=[]),
            request(needs={}),
            request(needs={"": 0.5}),
            request(needs={"hunger": "x"}),
            request(needs={"hunger": True}),
            request(needs={"hunger": -0.1}),
            request(needs={"hunger": 1.1}),
            request(needs={"hunger": float("nan")}),
            request(needs={"hunger": float("inf")}),
            request(activities="nope"),
            request(activities=[]),
            request(activities=["nope"]),
            request(activities=[{}]),
            request(activities=[{"start_minute": 0, "end_minute": 1}]),
            request(activities=[activity("", 0, 1)]),
            request(activities=[activity("a", 0, 1), activity("a", 2, 3)]),
            request(activities=[activity("a", 0.0, 1)]),
            request(activities=[activity("a", True, 1)]),
            request(activities=[activity("a", -1, 1)]),
            request(activities=[activity("a", 1440, 1)]),
            request(activities=[activity("a", 0, 1.0)]),
            request(activities=[activity("a", 0, 1440)]),
            request(activities=[activity("a", 0, 1, priority=1.5)]),
            request(activities=[activity("a", 0, 1, priority=True)]),
            request(activities=[activity("a", 0, 1, priority="1")]),
            request(activities=[activity("a", 0, 1, need="")]),
            request(activities=[activity("a", 0, 1, need=None)]),
            request(activities=[activity("a", 0, 1, need="missing")]),
            request(activities=[activity("a", 0, 1, need=42)]),
            request(
                activities=[
                    activity("a", 0, 1, need="hunger", relief=0.5)
                ]
            ),
            request(
                activities=[
                    activity("a", 0, 1, need="hunger", trigger=0.5)
                ]
            ),
            request(
                activities=[
                    activity("a", 0, 1, need="hunger", trigger="x", relief=0.5)
                ]
            ),
            request(
                activities=[
                    activity("a", 0, 1, need="hunger", trigger=True, relief=0.5)
                ]
            ),
            request(
                activities=[
                    activity("a", 0, 1, need="hunger", trigger=-0.01, relief=0.5)
                ]
            ),
            request(
                activities=[
                    activity("a", 0, 1, need="hunger", trigger=1.01, relief=0.5)
                ]
            ),
            request(
                activities=[
                    activity("a", 0, 1, need="hunger", trigger=0.5, relief=float("nan"))
                ]
            ),
        ]
        for bad in bad_requests:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.service.select_schedule_activity(bad)

    def test_errors_are_schedule_errors(self):
        with self.assertRaises(ScheduleError):
            self.service.select_schedule_activity(request(activities=[]))
        self.assertTrue(issubclass(ScheduleError, ValueError))

    def test_validation_completes_before_any_decision(self):
        # A valid urgent activity followed by an invalid one must not select.
        payload = request(
            needs={"hunger": 0.9},
            activities=[
                activity("eat", 0, 300, need="hunger", trigger=0.5, relief=0.5),
                activity("bad", 0, 1, need="ghost"),
            ],
        )
        with self.assertRaises(ValueError):
            self.service.select_schedule_activity(payload)


if __name__ == "__main__":
    unittest.main()
