import copy
import unittest

from npcmind.service import DialogueError, Service


def intent(id, patterns, **extra):
    record = {"id": id, "patterns": patterns}
    record.update(extra)
    return record


def request(**overrides):
    base = {
        "utterance": "attack the goblin",
        "intents": [intent("attack", ["attack the {target}"])],
    }
    base.update(overrides)
    return base


class MatchDialogueIntentTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_literal_match_is_case_insensitive(self):
        result = self.service.match_dialogue_intent(
            request(utterance="  ATTACK   THE\tGOBLIN  ")
        )
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["intent"], "attack")
        self.assertEqual(result["slots"], {"target": "GOBLIN"})

    def test_slot_values_keep_original_text(self):
        result = self.service.match_dialogue_intent(
            request(utterance="attack the GoBlIn")
        )
        self.assertEqual(result["slots"], {"target": "GoBlIn"})

    def test_pattern_must_cover_whole_utterance(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="attack the goblin now",
                intents=[intent("attack", ["attack the {target}"])],
            )
        )
        self.assertEqual(result["status"], "NO_MATCH")
        self.assertIsNone(result["intent"])
        self.assertEqual(result["slots"], {})
        self.assertEqual(result["candidates"], [])

    def test_repeated_slot_must_capture_equal_values(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="go Home home",
                intents=[intent("echo", ["go {place} {place}"])],
            )
        )
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["slots"], {"place": "Home"})

    def test_repeated_slot_rejects_different_values(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="go home there",
                intents=[intent("echo", ["go {place} {place}"])],
            )
        )
        self.assertEqual(result["status"], "NO_MATCH")

    def test_pattern_with_most_literals_wins_within_intent(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="cast fire ball",
                intents=[
                    intent(
                        "cast",
                        ["cast {spell} ball", "cast {a} {b}"],
                    )
                ],
            )
        )
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["slots"], {"spell": "fire"})
        candidate = result["candidates"][0]
        self.assertEqual(candidate["pattern_index"], 0)
        self.assertEqual(candidate["literal_count"], 2)

    def test_pattern_tie_goes_to_earliest(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="hello world",
                intents=[
                    intent(
                        "greet",
                        ["hello {a}", "{b} world"],
                    )
                ],
            )
        )
        self.assertEqual(result["slots"], {"a": "world"})
        self.assertEqual(result["candidates"][0]["pattern_index"], 0)

    def test_intents_ordered_by_priority_then_literals_then_declaration(self):
        intents = [
            intent("mid_one", ["{a} {b} world"], priority=1),
            intent("low", ["hello {a} {b}"], priority=0),
            intent("high", ["hello brave {a}"], priority=5),
            intent("mid_two", ["hello {a} world"], priority=1),
        ]
        result = self.service.match_dialogue_intent(
            request(utterance="hello brave world", intents=intents)
        )
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["intent"], "high")
        self.assertEqual(
            [c["id"] for c in result["candidates"]],
            ["high", "mid_two", "mid_one", "low"],
        )
        self.assertEqual(
            [(c["priority"], c["literal_count"]) for c in result["candidates"]],
            [(5, 2), (1, 2), (1, 1), (0, 1)],
        )

    def test_candidates_carry_slots_and_pattern_index(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="open the gate",
                intents=[
                    intent("open", ["open the {thing}"]),
                    intent("act", ["{verb} the gate"]),
                ],
            )
        )
        self.assertEqual(
            result["candidates"],
            [
                {
                    "id": "open",
                    "priority": 0,
                    "literal_count": 2,
                    "pattern_index": 0,
                    "slots": {"thing": "gate"},
                },
                {
                    "id": "act",
                    "priority": 0,
                    "literal_count": 2,
                    "pattern_index": 0,
                    "slots": {"verb": "open"},
                },
            ],
        )

    def test_requires_gates_intent(self):
        intents = [
            intent("trade", ["trade with {who}"], requires={"mode": "shop"}),
            intent("talk", ["trade with {who}"]),
        ]
        result = self.service.match_dialogue_intent(
            request(
                utterance="trade with anna",
                intents=intents,
                context={"mode": "field"},
            )
        )
        self.assertEqual(result["intent"], "talk")
        self.assertEqual([c["id"] for c in result["candidates"]], ["talk"])

    def test_requires_uses_json_type_sensitive_equality(self):
        intents = [intent("go", ["go now"], requires={"flag": 1})]
        result = self.service.match_dialogue_intent(
            request(utterance="go now", intents=intents, context={"flag": True})
        )
        self.assertEqual(result["status"], "NO_MATCH")
        result = self.service.match_dialogue_intent(
            request(utterance="go now", intents=intents, context={"flag": 1})
        )
        self.assertEqual(result["status"], "MATCHED")

    def test_unicode_casefold_and_whitespace(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="straße weg",
                intents=[intent("de", ["STRASSE {slot}"])],
            )
        )
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["slots"], {"slot": "weg"})

    def test_request_is_not_mutated(self):
        payload = request(
            utterance="attack the goblin",
            intents=[
                intent("attack", ["attack the {target}"], priority=2, requires={"a": 1})
            ],
            context={"a": 1},
        )
        snapshot = copy.deepcopy(payload)
        self.service.match_dialogue_intent(payload)
        self.assertEqual(payload, snapshot)

    def test_invalid_requests_raise_value_error(self):
        bad_requests = [
            "not a dict",
            request(utterance=42),
            request(utterance="   "),
            request(intents="nope"),
            request(intents=[]),
            request(intents=["nope"]),
            request(intents=[intent("", ["hi"])]),
            request(intents=[intent("a", ["hi"]), intent("a", ["yo"])]),
            request(intents=[intent("a", ["hi"], priority=True)]),
            request(intents=[intent("a", ["hi"], priority=1.5)]),
            request(intents=[intent("a", ["hi"], requires=["mode"])]),
            request(intents=[intent("a", "hi")]),
            request(intents=[intent("a", [])]),
            request(intents=[intent("a", ["hi", 3])]),
            request(intents=[intent("a", ["hi", "  "])]),
            request(intents=[intent("a", ["{1bad}"])]),
            request(intents=[intent("a", ["{bad-name}"])]),
            request(intents=[intent("a", ["{}"])]),
            request(intents=[intent("a", ["{unclosed"])]),
            request(intents=[intent("a", ["x{slot}"])]),
            request(context="nope"),
        ]
        for bad in bad_requests:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.service.match_dialogue_intent(bad)

    def test_errors_are_dialogue_errors(self):
        with self.assertRaises(DialogueError):
            self.service.match_dialogue_intent(request(intents=[]))

    def test_valid_slot_name_characters(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="go north",
                intents=[intent("move", ["go {_dir_2}"])],
            )
        )
        self.assertEqual(result["slots"], {"_dir_2": "north"})

    def test_no_state_between_calls(self):
        first = self.service.match_dialogue_intent(request())
        second = self.service.match_dialogue_intent(request(utterance="nothing here"))
        self.assertEqual(first["status"], "MATCHED")
        self.assertEqual(second["status"], "NO_MATCH")
        third = self.service.match_dialogue_intent(request())
        self.assertEqual(third, first)


if __name__ == "__main__":
    unittest.main()
