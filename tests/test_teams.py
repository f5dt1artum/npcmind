import copy
import unittest

from npcmind.service import Service


def request(**overrides):
    base = {
        "agents": [
            {"id": "alice", "scores": {"tank": 0.9, "healer": 0.4}},
            {"id": "bob", "scores": {"tank": 0.7, "dps": 0.8}},
            {"id": "carol", "scores": {"healer": 0.6}},
        ],
        "roles": [
            {"id": "tank", "capacity": 1},
            {"id": "healer", "capacity": 1},
            {"id": "dps", "capacity": 2},
        ],
    }
    base.update(overrides)
    return base


class AssignTeamRolesTest(unittest.TestCase):
    def setUp(self):
        self.service = Service()

    def test_optimal_assignment_and_role_subtotals(self):
        result = self.service.assign_team_roles(request())
        self.assertEqual(result["status"], "ASSIGNED")
        self.assertEqual(
            result["assignments"],
            [
                {"agent": "alice", "role": "tank", "score": 0.9},
                {"agent": "bob", "role": "dps", "score": 0.8},
                {"agent": "carol", "role": "healer", "score": 0.6},
            ],
        )
        self.assertEqual(result["unassigned"], [])
        self.assertAlmostEqual(result["total_score"], 2.3)
        self.assertEqual(
            result["roles"],
            [
                {"id": "tank", "capacity": 1, "agents": ["alice"], "score": 0.9},
                {"id": "healer", "capacity": 1, "agents": ["carol"], "score": 0.6},
                {"id": "dps", "capacity": 2, "agents": ["bob"], "score": 0.8},
            ],
        )

    def test_capacity_forces_second_best(self):
        result = self.service.assign_team_roles(
            request(
                agents=[
                    {"id": "a", "scores": {"tank": 0.9, "dps": 0.1}},
                    {"id": "b", "scores": {"tank": 0.8, "dps": 0.7}},
                ],
                roles=[{"id": "tank", "capacity": 1}, {"id": "dps", "capacity": 1}],
            )
        )
        # Both prefer tank, but the single slot goes to a and b takes dps.
        self.assertEqual(
            result["assignments"],
            [
                {"agent": "a", "role": "tank", "score": 0.9},
                {"agent": "b", "role": "dps", "score": 0.7},
            ],
        )
        self.assertAlmostEqual(result["total_score"], 1.6)

    def test_capacity_allows_multiple_agents_per_role(self):
        result = self.service.assign_team_roles(
            request(
                agents=[
                    {"id": "a", "scores": {"dps": 0.5}},
                    {"id": "b", "scores": {"dps": 0.6}},
                    {"id": "c", "scores": {"dps": 0.7}},
                ],
                roles=[{"id": "dps", "capacity": 2}],
            )
        )
        self.assertEqual([a["agent"] for a in result["assignments"]], ["b", "c"])
        self.assertEqual(result["unassigned"], ["a"])
        self.assertAlmostEqual(result["total_score"], 1.3)
        self.assertEqual(result["roles"][0]["agents"], ["b", "c"])

    def test_tie_breaks_to_lexicographically_smallest_role_sequence(self):
        result = self.service.assign_team_roles(
            request(
                agents=[
                    {"id": "a", "scores": {"r0": 0.5, "r1": 0.5}},
                    {"id": "b", "scores": {"r0": 0.5, "r1": 0.5}},
                ],
                roles=[{"id": "r0", "capacity": 1}, {"id": "r1", "capacity": 1}],
            )
        )
        # (r0, r1) and (r1, r0) both total 1.0; the earlier role wins.
        self.assertEqual(
            result["assignments"],
            [
                {"agent": "a", "role": "r0", "score": 0.5},
                {"agent": "b", "role": "r1", "score": 0.5},
            ],
        )

    def test_tie_break_prefers_assigning_the_earlier_agent(self):
        result = self.service.assign_team_roles(
            request(
                agents=[
                    {"id": "a", "scores": {"r0": 0.5}},
                    {"id": "b", "scores": {"r0": 0.5}},
                ],
                roles=[{"id": "r0", "capacity": 1}],
            )
        )
        # Assigning a (sequence (0, unassigned)) beats assigning b.
        self.assertEqual(
            result["assignments"], [{"agent": "a", "role": "r0", "score": 0.5}]
        )
        self.assertEqual(result["unassigned"], ["b"])

    def test_zero_scores_never_assigned(self):
        result = self.service.assign_team_roles(
            request(
                agents=[
                    {"id": "a", "scores": {"r0": 0}},
                    {"id": "b", "scores": {}},
                ],
                roles=[{"id": "r0", "capacity": 2}],
            )
        )
        self.assertEqual(result["status"], "ASSIGNED")
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["unassigned"], ["a", "b"])
        self.assertEqual(result["total_score"], 0)
        self.assertEqual(
            result["roles"], [{"id": "r0", "capacity": 2, "agents": [], "score": 0}]
        )

    def test_integer_scores_stay_integers(self):
        result = self.service.assign_team_roles(
            request(
                agents=[{"id": "a", "scores": {"r0": 1}}],
                roles=[{"id": "r0", "capacity": 1}],
            )
        )
        self.assertEqual(result["assignments"], [{"agent": "a", "role": "r0", "score": 1}])
        self.assertEqual(result["total_score"], 1)
        self.assertEqual(result["roles"][0]["score"], 1)

    def test_unlisted_roles_are_not_assignable(self):
        result = self.service.assign_team_roles(
            request(
                agents=[{"id": "a", "scores": {"r1": 0.5}}],
                roles=[{"id": "r0", "capacity": 1}, {"id": "r1", "capacity": 1}],
            )
        )
        self.assertEqual(
            result["assignments"], [{"agent": "a", "role": "r1", "score": 0.5}]
        )
        self.assertEqual(result["roles"][0]["agents"], [])

    def test_request_is_not_mutated(self):
        payload = request()
        snapshot = copy.deepcopy(payload)
        self.service.assign_team_roles(payload)
        self.assertEqual(payload, snapshot)

    def test_invalid_requests_raise_value_error(self):
        bad_requests = [
            "not a dict",
            request(agents=[]),
            request(agents="not a list"),
            request(agents=None),
            request(roles=[]),
            request(roles="not a list"),
            request(roles=None),
            request(agents=["not a dict"]),
            request(roles=["not a dict"]),
            request(agents=[{"scores": {"tank": 0.5}}]),
            request(agents=[{"id": "", "scores": {}}]),
            request(agents=[{"id": 1, "scores": {}}]),
            request(agents=[{"id": "a", "scores": {}}, {"id": "a", "scores": {}}]),
            request(roles=[{"capacity": 1}]),
            request(roles=[{"id": "", "capacity": 1}]),
            request(roles=[{"id": "r", "capacity": 1}, {"id": "r", "capacity": 2}]),
            request(roles=[{"id": "r", "capacity": 0}]),
            request(roles=[{"id": "r", "capacity": -1}]),
            request(roles=[{"id": "r", "capacity": 1.5}]),
            request(roles=[{"id": "r", "capacity": True}]),
            request(roles=[{"id": "r", "capacity": "1"}]),
            request(roles=[{"id": "r"}]),
            request(agents=[{"id": "a"}]),
            request(agents=[{"id": "a", "scores": None}]),
            request(agents=[{"id": "a", "scores": [("tank", 0.5)]}]),
            request(agents=[{"id": "a", "scores": {"unknown": 0.5}}]),
            request(agents=[{"id": "a", "scores": {"tank": True}}]),
            request(agents=[{"id": "a", "scores": {"tank": float("nan")}}]),
            request(agents=[{"id": "a", "scores": {"tank": float("inf")}}]),
            request(agents=[{"id": "a", "scores": {"tank": -0.1}}]),
            request(agents=[{"id": "a", "scores": {"tank": 1.1}}]),
            request(agents=[{"id": "a", "scores": {"tank": "0.5"}}]),
        ]
        for bad in bad_requests:
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.service.assign_team_roles(bad)

    def test_health_unchanged(self):
        self.assertEqual(
            self.service.health(),
            {"status": "ok", "service": "npcmind", "version": Service.version},
        )


if __name__ == "__main__":
    unittest.main()
