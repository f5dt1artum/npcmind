import copy
import unittest

from npcmind.service import Service


def request(**overrides):
    base = {
        "now": 600,
        "needs": {"hunger": 0.8, "fatigue": 0.3},
        "activities": [
            {"id": "work", "start_minute": 540, "end_minute": 1020, "priority": 1},
            {
                "id": "eat",
                "start_minute": 1200,
                "end_minute": 1260,
                "priority": 3,
                "need": "hunger",
                "trigger": 0.7,
                "relief": 0.5,
            },
            {"id": "idle", "start_minute": 0, "end_minute": 0},
        ],
    }
    base.update(overrides)
    return base


class SelectScheduleActivityTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_urgent_beats_scheduled_and_applies_relief(self):
        result = self.service.select_schedule_activity(request())
        self.assertEqual(result["status"], "SELECTED")
        # eat is urgent (hunger 0.8 >= 0.7) though outside its window.
        self.assertEqual(result["selected"], "eat")
        self.assertEqual(result["needs"], {"hunger": 0.30000000000000004, "fatigue": 0.3})
        self.assertAlmostEqual(result["needs"]["hunger"], 0.3)

    def test_evaluations_in_input_order(self):
        result = self.service.select_schedule_activity(request())
        evaluations = result["evaluations"]
        self.assertEqual([e["id"] for e in evaluations], ["work", "eat", "idle"])
        work, eat, idle = evaluations
        self.assertEqual(
            (work["scheduled"], work["urgent"], work["eligible"], work["need_level"]),
            (True, False, True, None),
        )
        self.assertEqual(
            (eat["scheduled"], eat["urgent"], eat["eligible"]),
            (False, True, True),
        )
        self.assertEqual(eat["need_level"], 0.8)
        # Equal start/end means the whole day.
        self.assertEqual(
            (idle["scheduled"], idle["urgent"], idle["eligible"], idle["need_level"]),
            (True, False, True, None),
        )

    def test_scheduled_only_picks_highest_priority_then_input_order(self):
        result = self.service.select_schedule_activity(
            request(
                needs={"hunger": 0.2},
                activities=[
                    {"id": "a", "start_minute": 540, "end_minute": 1020, "priority": 2},
                    {"id": "b", "start_minute": 540, "end_minute": 1020, "priority": 5},
                    {"id": "c", "start_minute": 540, "end_minute": 1020, "priority": 5},
                ],
            )
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected"], "b")
        # Needs are untouched when the selected activity has no need.
        self.assertEqual(result["needs"], {"hunger": 0.2})

    def test_urgent_ties_break_on_need_level_then_priority_then_order(self):
        activities = [
            {
                "id": "low",
                "start_minute": 0,
                "end_minute": 0,
                "priority": 9,
                "need": "hunger",
                "trigger": 0.5,
                "relief": 0.1,
            },
            {
                "id": "high",
                "start_minute": 0,
                "end_minute": 0,
                "priority": 1,
                "need": "fatigue",
                "trigger": 0.5,
                "relief": 0.1,
            },
        ]
        result = self.service.select_schedule_activity(
            request(needs={"hunger": 0.6, "fatigue": 0.9}, activities=activities)
        )
        # Higher need level wins even with lower priority.
        self.assertEqual(result["selected"], "high")

        same_level = [
            dict(activity, need="hunger") for activity in activities
        ]
        result = self.service.select_schedule_activity(
            request(needs={"hunger": 0.6, "fatigue": 0.9}, activities=same_level)
        )
        # Equal need level: higher priority wins.
        self.assertEqual(result["selected"], "low")

        same_priority = [dict(activity, priority=4, need="hunger") for activity in activities]
        result = self.service.select_schedule_activity(
            request(needs={"hunger": 0.6, "fatigue": 0.9}, activities=same_priority)
        )
        # Equal level and priority: earliest input wins.
        self.assertEqual(result["selected"], "low")

    def test_overnight_window(self):
        payload = request(
            needs={},
            activities=[
                {"id": "night", "start_minute": 1320, "end_minute": 360},
                {"id": "day", "start_minute": 360, "end_minute": 1320},
            ],
        )
        for now, expected in ((0, "night"), (359, "night"), (360, "day"), (1320, "night")):
            result = self.service.select_schedule_activity({**payload, "now": now})
            self.assertEqual(result["selected"], expected, now)

    def test_window_end_is_exclusive(self):
        result = self.service.select_schedule_activity(
            request(
                needs={},
                now=1020,
                activities=[{"id": "work", "start_minute": 540, "end_minute": 1020}],
            )
        )
        self.assertEqual(result["status"], "IDLE")
        self.assertIsNone(result["selected"])

    def test_idle_returns_original_needs_and_full_evaluations(self):
        result = self.service.select_schedule_activity(
            request(
                now=300,
                needs={"hunger": 0.4},
                activities=[
                    {
                        "id": "eat",
                        "start_minute": 1200,
                        "end_minute": 1260,
                        "need": "hunger",
                        "trigger": 0.7,
                        "relief": 0.5,
                    }
                ],
            )
        )
        self.assertEqual(result["status"], "IDLE")
        self.assertIsNone(result["selected"])
        self.assertEqual(result["needs"], {"hunger": 0.4})
        self.assertEqual(len(result["evaluations"]), 1)
        evaluation = result["evaluations"][0]
        self.assertFalse(evaluation["scheduled"])
        self.assertFalse(evaluation["urgent"])
        self.assertFalse(evaluation["eligible"])
        self.assertEqual(evaluation["need_level"], 0.4)

    def test_relief_clamps_at_zero(self):
        result = self.service.select_schedule_activity(
            request(
                needs={"hunger": 0.2},
                activities=[
                    {
                        "id": "eat",
                        "start_minute": 0,
                        "end_minute": 0,
                        "need": "hunger",
                        "trigger": 0.1,
                        "relief": 0.9,
                    }
                ],
            )
        )
        self.assertEqual(result["selected"], "eat")
        self.assertEqual(result["needs"], {"hunger": 0})

    def test_trigger_boundary_is_urgent(self):
        result = self.service.select_schedule_activity(
            request(
                needs={"hunger": 0.7},
                activities=[
                    {
                        "id": "eat",
                        "start_minute": 1200,
                        "end_minute": 1260,
                        "need": "hunger",
                        "trigger": 0.7,
                        "relief": 0.5,
                    }
                ],
            )
        )
        self.assertEqual(result["status"], "SELECTED")
        self.assertTrue(result["evaluations"][0]["urgent"])

    def test_request_is_not_mutated(self):
        payload = request()
        snapshot = copy.deepcopy(payload)
        self.service.select_schedule_activity(payload)
        self.assertEqual(payload, snapshot)

    def test_invalid_requests_raise_value_error(self):
        bad_requests = [
            "not a dict",
            request(now=-1),
            request(now=1440),
            request(now=1.5),
            request(now=True),
            request(now="600"),
            request(needs=None),
            request(needs={"": 0.5}),
            request(needs={"hunger": -0.1}),
            request(needs={"hunger": 1.1}),
            request(needs={"hunger": True}),
            request(needs={"hunger": float("nan")}),
            request(needs={"hunger": float("inf")}),
            request(activities=[]),
            request(activities="not a list"),
            request(activities=["not a dict"]),
            request(activities=[{"id": "", "start_minute": 0, "end_minute": 1}]),
            request(activities=[{"id": "a", "start_minute": 0, "end_minute": 1}] * 2),
            request(activities=[{"id": "a", "start_minute": -1, "end_minute": 1}]),
            request(activities=[{"id": "a", "start_minute": 0, "end_minute": 1440}]),
            request(activities=[{"id": "a", "start_minute": 0.5, "end_minute": 1}]),
            request(activities=[{"id": "a", "start_minute": 0, "end_minute": True}]),
            request(
                activities=[{"id": "a", "start_minute": 0, "end_minute": 1, "priority": 1.5}]
            ),
            request(
                activities=[{"id": "a", "start_minute": 0, "end_minute": 1, "priority": True}]
            ),
            request(
                activities=[
                    {"id": "a", "start_minute": 0, "end_minute": 1, "need": "unknown",
                     "trigger": 0.5, "relief": 0.5}
                ]
            ),
            request(
                activities=[
                    {"id": "a", "start_minute": 0, "end_minute": 1, "need": "hunger",
                     "relief": 0.5}
                ]
            ),
            request(
                activities=[
                    {"id": "a", "start_minute": 0, "end_minute": 1, "need": "hunger",
                     "trigger": 0.5}
                ]
            ),
            request(
                activities=[
                    {"id": "a", "start_minute": 0, "end_minute": 1, "need": "hunger",
                     "trigger": 1.5, "relief": 0.5}
                ]
            ),
            request(
                activities=[
                    {"id": "a", "start_minute": 0, "end_minute": 1, "need": "hunger",
                     "trigger": 0.5, "relief": -0.1}
                ]
            ),
            request(
                activities=[
                    {"id": "a", "start_minute": 0, "end_minute": 1, "need": "hunger",
                     "trigger": 0.5, "relief": float("nan")}
                ]
            ),
        ]
        for bad in bad_requests:
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.service.select_schedule_activity(bad)

    def test_health_unchanged(self):
        self.assertEqual(
            self.service.health(),
            {"status": "ok", "service": "npcmind", "version": Service.version},
        )


if __name__ == "__main__":
    unittest.main()
