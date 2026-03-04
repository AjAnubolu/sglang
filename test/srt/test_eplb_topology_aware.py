"""
CPU-only unit tests for the topology-aware EPLB algorithm.
No GPU or server required.
"""

import unittest

import torch

from sglang.srt.eplb.eplb_algorithms.deepseek import (
    rebalance_experts_hierarchical,
    rebalance_experts_topology_aware,
    rebalance_experts_topology_aware_entry,
)
from sglang.srt.eplb.utils import inter_node_traffic_ratio

# Common test configuration: 2 nodes, 4 GPUs, 8 logical experts, 16 physical
NUM_LAYERS = 2
NUM_LOGICAL = 8
NUM_PHYSICAL = 16
NUM_GROUPS = 4
NUM_NODES = 2
NUM_GPUS = 4


def _make_skewed_weight():
    """Create weights where expert 0 is much hotter than others."""
    w = torch.ones(NUM_LAYERS, NUM_LOGICAL)
    w[:, 0] = 100.0  # expert 0 is very hot
    return w


class TestTopologyAwareOutputShapes(unittest.TestCase):
    def test_output_shapes(self):
        weight = _make_skewed_weight()
        phy2log, phyrank, logcnt = rebalance_experts_topology_aware(
            weight, NUM_PHYSICAL, NUM_GROUPS, NUM_NODES, NUM_GPUS
        )
        self.assertEqual(phy2log.shape, (NUM_LAYERS, NUM_PHYSICAL))
        self.assertEqual(phyrank.shape, (NUM_LAYERS, NUM_PHYSICAL))
        self.assertEqual(logcnt.shape, (NUM_LAYERS, NUM_LOGICAL))

    def test_entry_point_shapes(self):
        weight = _make_skewed_weight()
        phy2log, log2phy, logcnt = rebalance_experts_topology_aware_entry(
            weight, NUM_PHYSICAL, NUM_GROUPS, NUM_NODES, NUM_GPUS
        )
        self.assertEqual(phy2log.shape, (NUM_LAYERS, NUM_PHYSICAL))
        self.assertEqual(log2phy.shape[0], NUM_LAYERS)
        self.assertEqual(log2phy.shape[1], NUM_LOGICAL)
        self.assertEqual(logcnt.shape, (NUM_LAYERS, NUM_LOGICAL))


class TestTopologyAwareValidity(unittest.TestCase):
    def test_valid_expert_indices(self):
        weight = _make_skewed_weight()
        phy2log, _, _ = rebalance_experts_topology_aware(
            weight, NUM_PHYSICAL, NUM_GROUPS, NUM_NODES, NUM_GPUS
        )
        self.assertTrue((phy2log >= 0).all())
        self.assertTrue((phy2log < NUM_LOGICAL).all())

    def test_logcnt_sum(self):
        weight = _make_skewed_weight()
        _, _, logcnt = rebalance_experts_topology_aware(
            weight, NUM_PHYSICAL, NUM_GROUPS, NUM_NODES, NUM_GPUS
        )
        for layer in range(NUM_LAYERS):
            self.assertEqual(logcnt[layer].sum().item(), NUM_PHYSICAL)

    def test_every_expert_covered(self):
        weight = _make_skewed_weight()
        _, _, logcnt = rebalance_experts_topology_aware(
            weight, NUM_PHYSICAL, NUM_GROUPS, NUM_NODES, NUM_GPUS
        )
        self.assertTrue((logcnt >= 1).all())


class TestTopologyAwarePlacement(unittest.TestCase):
    def test_hot_expert_cross_node(self):
        """Hot expert should be replicated onto both nodes."""
        weight = _make_skewed_weight()
        phy2log, _, _ = rebalance_experts_topology_aware(
            weight, NUM_PHYSICAL, NUM_GROUPS, NUM_NODES, NUM_GPUS
        )
        num_phy_per_node = NUM_PHYSICAL // NUM_NODES
        for layer in range(NUM_LAYERS):
            hot_expert = 0
            # Find which nodes have the hot expert
            nodes_with_hot = set()
            for node in range(NUM_NODES):
                start = node * num_phy_per_node
                end = start + num_phy_per_node
                if hot_expert in phy2log[layer, start:end].tolist():
                    nodes_with_hot.add(node)
            self.assertGreater(
                len(nodes_with_hot),
                1,
                f"Layer {layer}: hot expert 0 should be on both nodes, "
                f"but only on nodes {nodes_with_hot}",
            )

    def test_lower_inter_node_traffic(self):
        """Topology-aware should have <= inter-node traffic than hierarchical."""
        weight = _make_skewed_weight()
        topo_phy2log, _, topo_logcnt = rebalance_experts_topology_aware(
            weight, NUM_PHYSICAL, NUM_GROUPS, NUM_NODES, NUM_GPUS
        )
        hier_phy2log, _, hier_logcnt = rebalance_experts_hierarchical(
            weight, NUM_PHYSICAL, NUM_GROUPS, NUM_NODES, NUM_GPUS
        )
        topo_ratio = inter_node_traffic_ratio(topo_phy2log, weight, NUM_NODES)
        hier_ratio = inter_node_traffic_ratio(hier_phy2log, weight, NUM_NODES)
        for layer in range(NUM_LAYERS):
            self.assertLessEqual(
                topo_ratio[layer].item(),
                hier_ratio[layer].item() + 1e-6,
                f"Layer {layer}: topology-aware inter-node traffic "
                f"({topo_ratio[layer]:.4f}) should be <= hierarchical "
                f"({hier_ratio[layer]:.4f})",
            )


class TestTopologyAwareEdgeCases(unittest.TestCase):
    def test_single_node(self):
        """With 1 node, should degenerate correctly."""
        weight = _make_skewed_weight()
        phy2log, phyrank, logcnt = rebalance_experts_topology_aware(
            weight,
            num_physical_experts=16,
            num_groups=4,
            num_nodes=1,
            num_gpus=4,
        )
        self.assertEqual(phy2log.shape, (NUM_LAYERS, 16))
        self.assertEqual(logcnt.sum(dim=-1).tolist(), [16, 16])
        self.assertTrue((logcnt >= 1).all())

    def test_uniform_weights(self):
        """Uniform weights should give balanced replication (logcnt close to equal)."""
        weight = torch.ones(NUM_LAYERS, NUM_LOGICAL)
        _, _, logcnt = rebalance_experts_topology_aware(
            weight, NUM_PHYSICAL, NUM_GROUPS, NUM_NODES, NUM_GPUS
        )
        # With uniform weights, each expert should get exactly 2 replicas
        expected = NUM_PHYSICAL // NUM_LOGICAL
        self.assertTrue(
            (logcnt == expected).all(),
            f"Expected all logcnt={expected}, got {logcnt}",
        )

    def test_no_redundancy(self):
        """When num_phy == num_log, logcnt should be all 1s."""
        weight = _make_skewed_weight()
        _, _, logcnt = rebalance_experts_topology_aware(
            weight,
            num_physical_experts=8,
            num_groups=4,
            num_nodes=2,
            num_gpus=4,
        )
        self.assertTrue(
            (logcnt == 1).all(),
            f"Expected all logcnt=1 with no redundancy, got {logcnt}",
        )


class TestLog2PhyConsistency(unittest.TestCase):
    def test_log2phy_consistency(self):
        """Round-trip: every entry in log2phy should map back correctly via phy2log."""
        weight = _make_skewed_weight()
        phy2log, log2phy, logcnt = rebalance_experts_topology_aware_entry(
            weight, NUM_PHYSICAL, NUM_GROUPS, NUM_NODES, NUM_GPUS
        )
        for layer in range(NUM_LAYERS):
            for expert in range(NUM_LOGICAL):
                cnt = logcnt[layer, expert].item()
                for r in range(cnt):
                    phy_idx = log2phy[layer, expert, r].item()
                    self.assertNotEqual(phy_idx, -1)
                    self.assertEqual(
                        phy2log[layer, phy_idx].item(),
                        expert,
                        f"Layer {layer}, expert {expert}, rank {r}: "
                        f"phy2log[{phy_idx}]={phy2log[layer, phy_idx].item()} != {expert}",
                    )


class TestMetrics(unittest.TestCase):
    def test_metrics_perfect_coverage(self):
        """If every expert is on every node, inter-node ratio should be 0."""
        num_nodes = 2
        num_log = 4
        # Each node has all 4 experts
        phy2log = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        weight = torch.ones(1, num_log)
        ratio = inter_node_traffic_ratio(phy2log, weight, num_nodes)
        self.assertAlmostEqual(ratio[0].item(), 0.0, places=5)

    def test_metrics_no_coverage(self):
        """If experts are disjoint across nodes, inter-node ratio should be 0.5."""
        num_nodes = 2
        # Node 0 has experts 0,1; Node 1 has experts 2,3
        phy2log = torch.tensor([[0, 1, 2, 3]])
        weight = torch.ones(1, 4)
        ratio = inter_node_traffic_ratio(phy2log, weight, num_nodes)
        # Each expert is on 1/2 nodes, so inter-node fraction = 1 - 1/2 = 0.5
        self.assertAlmostEqual(ratio[0].item(), 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
