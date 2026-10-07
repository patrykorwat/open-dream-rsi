"""Sentinel features in the policy observation (commit 3 of the
Sentinel-powered Dream-RSI integration).

TreeNode.sentinel (commit 2) becomes a FIRST-CLASS policy feature:
frontier_entry() exposes sentinel_signature / sentinel_repeat /
sentinel_blocked, the same dict feeds online expansion (loop._frontier)
and the counterfactual replay world (rollout_world_payload), and both
harness mirrors (sandboxed + in-process) pass it through to the policy.

Pinned properties:
* Absent metadata (cold arm, legacy trees) degrades to clean-and-unblocked
  defaults — an ON-arm policy cannot crash on an OFF-tree, so an
  ablation (sentinel_engine=None) changes VALUES, never the interface.
* frontier_entry only ever reads the replay-safe keys; a malformed node
  carrying a cross-session counter still cannot leak it into the world
  view (defense-in-depth behind to_world_dict's construction rule).
* The features are ACTIONABLE: on a tree where the trap leaf is a gate
  refusal, a policy written against sentinel_blocked beats the greedy
  baseline in the real rollout harness — and the same policy on the same
  tree WITHOUT metadata ties greedy, attributing the win to the feature,
  not to the code (ablation pattern required for any new arm).
"""
import unittest

from open_dream_rsi.core.policygen import (
    POLICY_CONTRACT,
    PolicySandbox,
    _simulate_step_rewards,
    frontier_entry,
    greedy_rollout_score,
    outcome_map,
    rollout_score,
    rollout_world_payload,
)
from open_dream_rsi.core.tree import DiscoveryTree

SANDBOX = PolicySandbox()


def trap_tree(blocked: bool) -> DiscoveryTree:
    """root -> A (high score, sentinel-blocked when blocked=True, never
    expanded) and root -> B (low score, recorded child worth 1.0).

    Greedy camps A (0.9 > 0.2) and is paid A's own score forever; a policy
    that reads sentinel_blocked picks B and gets the 1.0 ladder."""
    t = DiscoveryTree()
    t.add_node("root", "seed", {}, 0.0)
    t.add_node("A", "try_a", {"errors": ["gate refused"]}, 0.9, "root",
               sentinel=({"signature": "deadbeefcafe1234", "repeat_world": 4,
                          "recurring": True, "blocked": blocked}
                         if blocked else None))
    t.add_node("B", "try_b", {}, 0.2, "root")
    t.add_node("B/win", "try_b2", {}, 1.0, "B")
    return t


class TestFrontierEntryFields(unittest.TestCase):
    def test_metadata_becomes_primitive_features(self):
        t = DiscoveryTree()
        t.add_node("r", "seed", {}, 0.0)
        t.add_node("n", "try", {"errors": ["boom"]}, 0.5, "r",
                   sentinel={"signature": "sig-abc", "repeat_world": 3,
                             "recurring": True, "blocked": True,
                             "budget_fraction": 0.82,
                             "cross_session_count": 17})   # malformed: must not leak
        e = frontier_entry(t.nodes["n"], outcome_map(t))
        self.assertEqual(e["sentinel_signature"], "sig-abc")
        self.assertEqual(e["sentinel_repeat"], 3)
        self.assertIs(e["sentinel_blocked"], True)
        # defense-in-depth: frontier_entry reads ONLY the three safe keys
        self.assertNotIn("cross_session_count", e)
        self.assertNotIn("sessions", e)
        self.assertNotIn("budget_fraction", e)
        self.assertNotIn("repeat_world", e)

    def test_absent_metadata_degrades_to_clean_defaults(self):
        t = DiscoveryTree()
        t.add_node("r", "seed", {}, 0.0)
        e = frontier_entry(t.nodes["r"], {})
        self.assertEqual(e["sentinel_signature"], "")
        self.assertEqual(e["sentinel_repeat"], 0)
        self.assertIs(e["sentinel_blocked"], False)
        # every frontier dict has the SAME keys whether or not the node was
        # observed — the OFF-arm/ON-arm ablation changes values, not shape
        t.add_node("s", "try", {}, 0.1, "r",
                   sentinel={"signature": "x", "repeat_world": 1,
                             "recurring": False, "blocked": False})
        self.assertEqual(set(frontier_entry(t.nodes["r"], {})),
                         set(frontier_entry(t.nodes["s"], {})))

    def test_contract_text_names_the_fields(self):
        for name in ("sentinel_signature", "sentinel_repeat",
                     "sentinel_blocked"):
            self.assertIn(name, POLICY_CONTRACT)


class TestRolloutCarriesFeatures(unittest.TestCase):
    def test_world_payload_and_harness_pass_fields_through(self):
        t = trap_tree(blocked=True)
        world = rollout_world_payload(t, horizon=3)
        a = next(n for n in world["nodes"] if n["node_id"] == "A")
        self.assertEqual(a["sentinel_signature"], "deadbeefcafe1234")
        self.assertEqual(a["sentinel_repeat"], 4)
        self.assertTrue(a["sentinel_blocked"])
        seen = []

        def record_and_root(visible, step):
            seen.append({n["node_id"]: (n["sentinel_blocked"],
                                        n["sentinel_repeat"])
                         for n in visible})
            return "root"

        _simulate_step_rewards(world, record_and_root)
        # the in-process mirror hands the policy the recorded facts once the
        # node is revealed (step 0 shows only the root — the prefix rule;
        # the values were fixed at node creation, nothing future rides along)
        self.assertNotIn("A", seen[0])
        self.assertEqual(seen[1]["A"], (True, 4))

    def test_feature_driven_policy_beats_greedy_only_with_features(self):
        """The ablation assertion for commit 4: the SAME policy code,
        the SAME tree; metadata on -> beats greedy, metadata off -> ties
        greedy. The delta is attributable to the feature alone. Uses the
        real promotion gate (rollout_score), not a hand-rolled objective."""
        policy = (
            "def choose_action(frontier, step):\n"
            "    if not frontier:\n"
            "        return None\n"
            "    safe = [n for n in frontier if not n.get('sentinel_blocked')]\n"
            "    pool = safe or frontier\n"
            "    best = pool[0]\n"
            "    for n in pool:\n"
            "        if n['score'] > best['score']:\n"
            "            best = n\n"
            "    return best['node_id']\n")

        on = trap_tree(blocked=True)
        off = trap_tree(blocked=False)

        obj_on, err_on = rollout_score(SANDBOX, policy, on, horizon=8)
        self.assertEqual(err_on, "")
        self.assertGreater(obj_on, greedy_rollout_score(on),
                           "sentinel_blocked feature must let the policy "
                           "escape the refused high-score trap")

        obj_off, err_off = rollout_score(SANDBOX, policy, off, horizon=8)
        self.assertEqual(err_off, "")
        self.assertAlmostEqual(obj_off, greedy_rollout_score(off),
                               msg="without metadata the policy is plain "
                                   "greedy — any delta on the ON tree is "
                                   "the feature, not the code")

    def test_online_frontier_field_parity(self):
        """loop._frontier delegates to frontier_entry (same code path), so
        online and replay show identical field sets — pinned via the
        runtime helper the loop uses."""
        from open_dream_rsi.loop import AutoRSIRuntime, DreamMemory, Task
        import tempfile
        mem = DreamMemory(tempfile.mkdtemp())
        rt = AutoRSIRuntime(client=None, memory=mem, tasks=[])
        t = trap_tree(blocked=True)
        online = rt._frontier(t)
        replay = rollout_world_payload(t)["nodes"]
        self.assertEqual(set(online[1]), set(replay[1]))
        a_online = next(e for e in online if e["node_id"] == "A")
        self.assertTrue(a_online["sentinel_blocked"])


if __name__ == "__main__":
    unittest.main()
