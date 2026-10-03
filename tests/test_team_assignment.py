import copy
import unittest

from npcmind.service import Service


def request(**overrides):
    base = {
        "roles": [
            {"id": "tank", "capacity": 1},
            {"id": "healer", "capacity": 2},
            {"id": "dps", "capacity": 3},
        ],
        "agents": [
            {"id": "a", "scores": {"tank": 0.9, "healer": 0.5, "dps": 0.2}},
            {"id": "b", "scores": {"tank": 0.8, "dps": 0.7}},
            {"id": "c", "scores": {"tank": 0.3, "healer": 0.6, "dps": 0.4}},
            {"id": "d", "scores": {"healer": 0.1}},
        ],
    }
    base.update(overrides)
    return base


class AssignTeamRolesTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def assign(self, payload):
        return self.service.assign_team_roles(copy.deepcopy(payload))

    def test_maximises_total_score_and_respects_capacity(self):
        result = self.assign(request())
        self.assertEqual(result["status"], "ASSIGNED")
        # a->tank 0.9, b->dps 0.7, c->healer 0.6, d->healer 0.1 = 2.3
        self.assertEqual(
            result["assignments"],
            [
                {"agent": "a", "role": "tank", "score": 0.9},
                {"agent": "b", "role": "dps", "score": 0.7},
                {"agent": "c", "role": "healer", "score": 0.6},
                {"agent": "d", "role": "healer", "score": 0.1},
            ],
        )
        self.assertEqual(result["unassigned"], [])
        self.assertAlmostEqual(result["total_score"], 2.3)
        self.assertEqual(
            result["roles"],
            [
                {"id": "tank", "capacity": 1, "agents": ["a"], "score": 0.9},
                {"id": "healer", "capacity": 2, "agents": ["c", "d"], "score": 0.7},
                {"id": "dps", "capacity": 3, "agents": ["b"], "score": 0.7},
            ],
        )

    def test_role_capacity_is_enforced(self):
        payload = request(
            roles=[{"id": "r", "capacity": 1}],
            agents=[
                {"id": "a", "scores": {"r": 0.9}},
                {"id": "b", "scores": {"r": 0.8}},
                {"id": "c", "scores": {"r": 0.7}},
            ],
        )
        result = self.assign(payload)
        self.assertEqual(result["status"], "ASSIGNED")
        self.assertEqual([item["agent"] for item in result["assignments"]], ["a"])
        self.assertEqual(result["unassigned"], ["b", "c"])
        self.assertEqual(result["roles"][0]["agents"], ["a"])
        self.assertEqual(result["total_score"], 0.9)

    def test_each_agent_gets_at_most_one_role(self):
        payload = request(
            roles=[{"id": "r1", "capacity": 5}, {"id": "r2", "capacity": 5}],
            agents=[{"id": "a", "scores": {"r1": 0.9, "r2": 0.8}}],
        )
        result = self.assign(payload)
        self.assertEqual(len(result["assignments"]), 1)
        self.assertEqual(result["assignments"][0]["role"], "r1")
        self.assertEqual(result["unassigned"], [])

    def test_unlisted_role_is_unavailable(self):
        payload = request(
            roles=[{"id": "r", "capacity": 2}],
            agents=[{"id": "a", "scores": {}}],
        )
        result = self.assign(payload)
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["unassigned"], ["a"])
        self.assertEqual(result["total_score"], 0)
        self.assertEqual(result["roles"], [{"id": "r", "capacity": 2, "agents": [], "score": 0}])

    def test_zero_score_never_participates(self):
        # c scores 0 for the sole role; even with spare capacity it must not
        # be assigned, and assigning it would not be distinguishable by score
        # but unassigned must win.
        payload = request(
            roles=[{"id": "r", "capacity": 3}],
            agents=[
                {"id": "a", "scores": {"r": 0.5}},
                {"id": "b", "scores": {"r": 0}},
                {"id": "c", "scores": {"r": 0.0}},
            ],
        )
        result = self.assign(payload)
        self.assertEqual([item["agent"] for item in result["assignments"]], ["a"])
        self.assertEqual(result["unassigned"], ["b", "c"])
        self.assertEqual(result["roles"][0]["agents"], ["a"])

    def test_no_positive_candidates_still_assigned_empty(self):
        payload = request(
            roles=[{"id": "r", "capacity": 2}, {"id": "s", "capacity": 2}],
            agents=[
                {"id": "a", "scores": {"r": 0}},
                {"id": "b", "scores": {}},
            ],
        )
        result = self.assign(payload)
        self.assertEqual(result["status"], "ASSIGNED")
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["unassigned"], ["a", "b"])
        self.assertEqual(result["total_score"], 0)
        self.assertEqual(
            result["roles"],
            [
                {"id": "r", "capacity": 2, "agents": [], "score": 0},
                {"id": "s", "capacity": 2, "agents": [], "score": 0},
            ],
        )

    def test_tie_break_is_lexicographic_choice_sequence(self):
        # One role: both agents score 0.5, capacity 1. Assigning either ties
        # on total; sequences (role=0, unassigned=1) vs (1, 0): the earlier
        # agent gets the role.
        payload = request(
            roles=[{"id": "r", "capacity": 1}],
            agents=[
                {"id": "a", "scores": {"r": 0.5}},
                {"id": "b", "scores": {"r": 0.5}},
            ],
        )
        result = self.assign(payload)
        self.assertEqual([item["agent"] for item in result["assignments"]], ["a"])
        self.assertEqual(result["unassigned"], ["b"])

    def test_tie_break_prefers_earlier_role_index_for_first_agent(self):
        # a fits both roles equally; earlier role index must win even though
        # b then gets displaced.
        payload = request(
            roles=[{"id": "r1", "capacity": 1}, {"id": "r2", "capacity": 1}],
            agents=[
                {"id": "a", "scores": {"r1": 0.5, "r2": 0.5}},
                {"id": "b", "scores": {"r1": 0.5, "r2": 0.5}},
            ],
        )
        result = self.assign(payload)
        self.assertEqual(
            [(item["agent"], item["role"]) for item in result["assignments"]],
            [("a", "r1"), ("b", "r2")],
        )

    def test_tie_break_unassigned_is_after_all_roles(self):
        # Agent a ties on total whether it takes r1 (0.4) and frees nothing
        # or stays unassigned while b takes r1: totals 0.4 either way.
        # sequences: a->r1 = (0,1); a unassigned, b->r1 = (1,0); (0,1) wins.
        payload = request(
            roles=[{"id": "r1", "capacity": 1}, {"id": "r2", "capacity": 1}],
            agents=[
                {"id": "a", "scores": {"r1": 0.4, "r2": 0.0}},
                {"id": "b", "scores": {"r1": 0.4}},
            ],
        )
        result = self.assign(payload)
        self.assertEqual(result["total_score"], 0.4)
        self.assertEqual(
            [(item["agent"], item["role"]) for item in result["assignments"]],
            [("a", "r1")],
        )
        self.assertEqual(result["unassigned"], ["b"])

    def test_score_dominates_lexicographic_tie_break(self):
        # Lexicographically smaller sequence (a->r1) scores less; score wins.
        payload = request(
            roles=[{"id": "r1", "capacity": 1}, {"id": "r2", "capacity": 1}],
            agents=[
                {"id": "a", "scores": {"r1": 0.1, "r2": 0.9}},
                {"id": "b", "scores": {"r1": 0.9}},
            ],
        )
        result = self.assign(payload)
        self.assertAlmostEqual(result["total_score"], 1.8)
        self.assertEqual(
            [(item["agent"], item["role"]) for item in result["assignments"]],
            [("a", "r2"), ("b", "r1")],
        )

    def test_assignments_and_unassigned_follow_agent_order(self):
        payload = request(
            roles=[{"id": "r", "capacity": 2}],
            agents=[
                {"id": "a", "scores": {}},
                {"id": "b", "scores": {"r": 0.5}},
                {"id": "c", "scores": {}},
                {"id": "d", "scores": {"r": 0.6}},
            ],
        )
        result = self.assign(payload)
        self.assertEqual([item["agent"] for item in result["assignments"]], ["b", "d"])
        self.assertEqual(result["unassigned"], ["a", "c"])
        self.assertEqual(result["roles"][0]["agents"], ["b", "d"])

    def test_integer_scores_keep_integer_types(self):
        payload = request(
            roles=[{"id": "r", "capacity": 2}],
            agents=[
                {"id": "a", "scores": {"r": 1}},
                {"id": "b", "scores": {"r": 0}},
            ],
        )
        result = self.assign(payload)
        self.assertEqual(result["assignments"][0]["score"], 1)
        self.assertIsInstance(result["assignments"][0]["score"], int)
        self.assertEqual(result["total_score"], 1)
        self.assertIsInstance(result["total_score"], int)
        self.assertEqual(result["roles"][0]["score"], 1)
        self.assertIsInstance(result["roles"][0]["score"], int)

    def test_decimal_ties_compare_equal(self):
        # Both permutations fill both roles and total 0.75 exactly
        # (0.5 + 0.25 versus 0.25 + 0.5); summation noise must not invent a
        # winner, so the lexicographic rule decides: a takes the earlier role.
        payload = request(
            roles=[{"id": "r1", "capacity": 1}, {"id": "r2", "capacity": 1}],
            agents=[
                {"id": "a", "scores": {"r1": 0.5, "r2": 0.25}},
                {"id": "b", "scores": {"r1": 0.5, "r2": 0.25}},
            ],
        )
        result = self.assign(payload)
        self.assertEqual(result["total_score"], 0.75)
        self.assertEqual(
            [(item["agent"], item["role"]) for item in result["assignments"]],
            [("a", "r1"), ("b", "r2")],
        )

    def test_request_is_not_mutated(self):
        payload = request()
        snapshot = copy.deepcopy(payload)
        self.service.assign_team_roles(payload)
        self.assertEqual(payload, snapshot)

    def test_no_state_kept_between_calls(self):
        first = self.assign(request())
        second = self.service.assign_team_roles(copy.deepcopy(request()))
        self.assertEqual(first, second)

    def test_invalid_requests_raise_value_error(self):
        bad_requests = [
            "not a dict",
            None,
            [],
            {},
            request(roles="not a list"),
            request(roles=[]),
            request(roles=None),
            request(agents="not a list"),
            request(agents=[]),
            request(agents=None),
            request(roles=["not a dict"]),
            request(agents=["not a dict"]),
            request(roles=[{"id": "", "capacity": 1}]),
            request(roles=[{"id": 42, "capacity": 1}]),
            request(roles=[{"id": None, "capacity": 1}]),
            request(roles=[{"capacity": 1}]),
            request(
                roles=[
                    {"id": "r", "capacity": 1},
                    {"id": "r", "capacity": 1},
                ]
            ),
            request(roles=[{"id": "r"}]),
            request(roles=[{"id": "r", "capacity": 0}]),
            request(roles=[{"id": "r", "capacity": -1}]),
            request(roles=[{"id": "r", "capacity": 1.5}]),
            request(roles=[{"id": "r", "capacity": True}]),
            request(roles=[{"id": "r", "capacity": "1"}]),
            request(roles=[{"id": "r", "capacity": None}]),
            request(agents=[{"id": "", "scores": {}}]),
            request(agents=[{"id": 1, "scores": {}}]),
            request(agents=[{"scores": {}}]),
            request(
                agents=[
                    {"id": "a", "scores": {}},
                    {"id": "a", "scores": {}},
                ]
            ),
            request(agents=[{"id": "a"}]),
            request(agents=[{"id": "a", "scores": []}]),
            request(agents=[{"id": "a", "scores": None}]),
            request(agents=[{"id": "a", "scores": {"unknown": 0.5}}]),
            request(agents=[{"id": "a", "scores": {42: 0.5}}]),
            request(agents=[{"id": "a", "scores": {"": 0.5}}]),
            request(agents=[{"id": "a", "scores": {"tank": True}}]),
            request(agents=[{"id": "a", "scores": {"tank": False}}]),
            request(agents=[{"id": "a", "scores": {"tank": "0.9"}}]),
            request(agents=[{"id": "a", "scores": {"tank": None}}]),
            request(agents=[{"id": "a", "scores": {"tank": -0.1}}]),
            request(agents=[{"id": "a", "scores": {"tank": 1.1}}]),
            request(agents=[{"id": "a", "scores": {"tank": float("nan")}}]),
            request(agents=[{"id": "a", "scores": {"tank": float("inf")}}]),
            request(agents=[{"id": "a", "scores": {"tank": float("-inf")}}]),
        ]
        for bad in bad_requests:
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.service.assign_team_roles(bad)

    def test_boundary_scores_are_accepted(self):
        payload = request(
            roles=[{"id": "r", "capacity": 2}],
            agents=[
                {"id": "a", "scores": {"r": 0}},
                {"id": "b", "scores": {"r": 1}},
                {"id": "c", "scores": {"r": 0.0}},
                {"id": "d", "scores": {"r": 1.0}},
            ],
        )
        result = self.assign(payload)
        self.assertEqual([item["agent"] for item in result["assignments"]], ["b", "d"])
        self.assertEqual(result["unassigned"], ["a", "c"])

    def test_health_unchanged(self):
        self.assertEqual(
            self.service.health(),
            {"status": "ok", "service": "npcmind", "version": Service.version},
        )


if __name__ == "__main__":
    unittest.main()
