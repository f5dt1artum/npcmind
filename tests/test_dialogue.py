import copy
import unittest

from npcmind.service import DialogueError, Service


def request(**overrides):
    base = {
        "utterance": "attack the goblin",
        "intents": [
            {"id": "attack", "patterns": ["attack the {target}"], "priority": 1},
            {"id": "greet", "patterns": ["hello", "hi there"]},
        ],
    }
    base.update(overrides)
    return base


class MatchDialogueIntentTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_basic_match_binds_slot_with_original_text(self):
        result = self.service.match_dialogue_intent(request())
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["intent"], "attack")
        self.assertEqual(result["slots"], {"target": "goblin"})
        self.assertEqual(len(result["candidates"]), 1)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["id"], "attack")
        self.assertEqual(candidate["priority"], 1)
        self.assertEqual(candidate["words"], 2)
        self.assertEqual(candidate["pattern"], 0)
        self.assertEqual(candidate["slots"], {"target": "goblin"})

    def test_plain_words_compare_by_casefold(self):
        result = self.service.match_dialogue_intent(
            request(utterance="ATTACK THE Goblin")
        )
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["slots"], {"target": "Goblin"})

    def test_casefold_handles_unicode_forms(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="gehe nach STRASSE",
                intents=[{"id": "go", "patterns": ["gehe nach straße"]}],
            )
        )
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["intent"], "go")

    def test_unicode_whitespace_tokenization(self):
        result = self.service.match_dialogue_intent(
            request(utterance="  attack  the goblin  ")
        )
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["slots"], {"target": "goblin"})

    def test_pattern_must_cover_the_whole_utterance(self):
        result = self.service.match_dialogue_intent(
            request(utterance="attack the goblin now")
        )
        self.assertEqual(result["status"], "NO_MATCH")
        self.assertIsNone(result["intent"])
        self.assertEqual(result["slots"], {})
        self.assertEqual(result["candidates"], [])

    def test_no_match_result_shape(self):
        result = self.service.match_dialogue_intent(request(utterance="run away quickly"))
        self.assertEqual(
            result,
            {"status": "NO_MATCH", "intent": None, "slots": {}, "candidates": []},
        )

    def test_repeated_slot_requires_equal_normalized_values(self):
        intent = {"id": "echo", "patterns": ["to {who} and {who}"]}
        matched = self.service.match_dialogue_intent(
            request(utterance="to Alice and ALICE", intents=[intent])
        )
        self.assertEqual(matched["status"], "MATCHED")
        self.assertEqual(matched["slots"], {"who": "Alice"})
        missed = self.service.match_dialogue_intent(
            request(utterance="to Alice and Bob", intents=[intent])
        )
        self.assertEqual(missed["status"], "NO_MATCH")

    def test_intent_keeps_pattern_with_most_plain_words(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="attack the goblin",
                intents=[
                    {
                        "id": "attack",
                        "patterns": ["attack the {target}", "attack {a} {b}"],
                    }
                ],
            )
        )
        candidate = result["candidates"][0]
        self.assertEqual(candidate["pattern"], 0)
        self.assertEqual(candidate["words"], 2)
        self.assertEqual(result["slots"], {"target": "goblin"})

    def test_pattern_word_tie_goes_to_earlier_pattern(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="hi there",
                intents=[{"id": "greet", "patterns": ["hi there", "hi {who}"]}],
            )
        )
        candidate = result["candidates"][0]
        self.assertEqual(candidate["pattern"], 0)
        self.assertEqual(candidate["slots"], {})

    def test_candidates_order_by_priority_then_words_then_declaration(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="attack the goblin",
                intents=[
                    {"id": "low", "patterns": ["attack the goblin"], "priority": 0},
                    {"id": "generic", "patterns": ["attack the {target}"], "priority": 5},
                    {"id": "specific", "patterns": ["attack the goblin"], "priority": 5},
                    {"id": "also_specific", "patterns": ["attack the goblin"], "priority": 5},
                ],
            )
        )
        self.assertEqual(
            [c["id"] for c in result["candidates"]],
            ["specific", "also_specific", "generic", "low"],
        )
        self.assertEqual(result["intent"], "specific")

    def test_requires_gates_intent(self):
        intents = [
            {
                "id": "attack",
                "patterns": ["attack the {target}"],
                "requires": {"armed": True},
            },
            {"id": "talk", "patterns": ["attack the {target}"]},
        ]
        unarmed = self.service.match_dialogue_intent(
            request(context={"armed": False}, intents=intents)
        )
        self.assertEqual([c["id"] for c in unarmed["candidates"]], ["talk"])
        armed = self.service.match_dialogue_intent(
            request(context={"armed": True}, intents=intents)
        )
        self.assertEqual(armed["intent"], "attack")

    def test_requires_uses_type_sensitive_json_equality(self):
        intents = [
            {
                "id": "attack",
                "patterns": ["attack"],
                "requires": {"armed": 1},
            }
        ]
        # True does not equal the number 1 under JSON type-sensitive rules.
        result = self.service.match_dialogue_intent(
            request(utterance="attack", context={"armed": True}, intents=intents)
        )
        self.assertEqual(result["status"], "NO_MATCH")
        result = self.service.match_dialogue_intent(
            request(utterance="attack", context={"armed": 1.0}, intents=intents)
        )
        self.assertEqual(result["status"], "MATCHED")

    def test_requires_key_must_exist_in_context(self):
        intents = [
            {"id": "attack", "patterns": ["attack"], "requires": {"armed": True}}
        ]
        result = self.service.match_dialogue_intent(
            request(utterance="attack", intents=intents)
        )
        self.assertEqual(result["status"], "NO_MATCH")

    def test_request_is_not_mutated(self):
        payload = request(context={"armed": True})
        snapshot = copy.deepcopy(payload)
        self.service.match_dialogue_intent(payload)
        self.assertEqual(payload, snapshot)

    def test_calls_keep_no_state(self):
        first = self.service.match_dialogue_intent(request())
        second = self.service.match_dialogue_intent(request(utterance="hello"))
        self.assertEqual(first["intent"], "attack")
        self.assertEqual(second["intent"], "greet")
        self.assertEqual(second["candidates"][0]["pattern"], 0)

    def test_invalid_requests_raise_value_error(self):
        bad_requests = [
            None,
            [],
            {},
            request(utterance=None),
            request(utterance=42),
            request(utterance=""),
            request(utterance="   \n\t "),
            request(intents=None),
            request(intents=[]),
            request(intents="attack"),
            request(intents=[None]),
            request(intents=[{"patterns": ["hi"]}]),
            request(intents=[{"id": "", "patterns": ["hi"]}]),
            request(intents=[{"id": 7, "patterns": ["hi"]}]),
            request(
                intents=[
                    {"id": "a", "patterns": ["hi"]},
                    {"id": "a", "patterns": ["yo"]},
                ]
            ),
            request(intents=[{"id": "a", "patterns": ["hi"], "priority": True}]),
            request(intents=[{"id": "a", "patterns": ["hi"], "priority": 1.5}]),
            request(intents=[{"id": "a", "patterns": ["hi"], "priority": "1"}]),
            request(intents=[{"id": "a", "patterns": ["hi"], "requires": []}]),
            request(intents=[{"id": "a", "patterns": ["hi"], "requires": "armed"}]),
            request(intents=[{"id": "a"}]),
            request(intents=[{"id": "a", "patterns": []}]),
            request(intents=[{"id": "a", "patterns": "hi"}]),
            request(intents=[{"id": "a", "patterns": [""]}]),
            request(intents=[{"id": "a", "patterns": ["   "]}]),
            request(intents=[{"id": "a", "patterns": [None]}]),
            request(intents=[{"id": "a", "patterns": [7]}]),
            request(intents=[{"id": "a", "patterns": ["{1target}"]}]),
            request(intents=[{"id": "a", "patterns": ["{-target}"]}]),
            request(intents=[{"id": "a", "patterns": ["{tar-get}"]}]),
            request(intents=[{"id": "a", "patterns": ["{}"]}]),
            request(intents=[{"id": "a", "patterns": ["{target"]}]),
            request(intents=[{"id": "a", "patterns": ["target}"]}]),
            request(intents=[{"id": "a", "patterns": ["{target}ish"]}]),
            request(context=[]),
            request(context="ctx"),
        ]
        for bad in bad_requests:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.service.match_dialogue_intent(bad)

    def test_error_is_dialogue_error(self):
        with self.assertRaises(DialogueError):
            self.service.match_dialogue_intent(request(utterance=""))

    def test_valid_slot_names(self):
        result = self.service.match_dialogue_intent(
            request(
                utterance="go north quickly",
                intents=[{"id": "go", "patterns": ["go {direction_2} {_speed}"]}],
            )
        )
        self.assertEqual(
            result["slots"], {"direction_2": "north", "_speed": "quickly"}
        )


if __name__ == "__main__":
    unittest.main()
