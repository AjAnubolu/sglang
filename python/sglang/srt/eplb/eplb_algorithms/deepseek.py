# This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package
from typing import Tuple

import torch


def balanced_packing(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (i for i in range(num_packs) if pack_items[i] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
                perm.shape
            ),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy
    )  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """

    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if enable_hierarchical:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
            num_layers, -1
        ),
    )
    return phy2log, log2phy, logcnt


def rebalance_experts_topology_aware(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Topology-aware expert rebalancing that distributes replicas across nodes
    to reduce inter-node traffic. Unlike hierarchical, which replicates only
    within each node, this algorithm places replicas on different nodes so
    that the runtime router can prefer NVLink (same-node) over RDMA.

    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes
        num_gpus: number of GPUs, must be a multiple of num_nodes

    Returns:
        phy2log: [num_moe_layers, num_physical_experts]
        phyrank: [num_moe_layers, num_physical_experts]
        logcnt: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    gpus_per_node = num_gpus // num_nodes
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus
    num_log_per_node = num_logical_experts // num_nodes
    num_phy_per_node = num_physical_experts // num_nodes

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
                perm.shape
            ),
        )
        return inv

    # Step 1: pack groups to nodes (identical to hierarchical)
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    tokens_per_mlog_full = weight.gather(-1, mlog2log)

    # Step 2: topology-aware replication across nodes + Step 3: pack to GPUs
    num_redundant = num_physical_experts - num_logical_experts
    final_phy2mlog = torch.zeros(num_layers, num_physical_experts, dtype=torch.int64)
    final_phyrank = torch.zeros(num_layers, num_physical_experts, dtype=torch.int64)
    final_logcnt = torch.zeros(num_layers, num_logical_experts, dtype=torch.int64)

    for layer in range(num_layers):
        w = tokens_per_mlog_full[layer]
        cnt = torch.ones(num_logical_experts, dtype=torch.int64)
        remaining = [phy_experts_per_gpu * gpus_per_node - num_log_per_node] * num_nodes
        node_has = [
            set(range(n * num_log_per_node, (n + 1) * num_log_per_node))
            for n in range(num_nodes)
        ]

        # Each node starts with its initial experts
        # node_experts[n] = list of (mlog_idx, rank)
        node_experts = [
            [(m, 0) for m in range(n * num_log_per_node, (n + 1) * num_log_per_node)]
            for n in range(num_nodes)
        ]

        for _ in range(num_redundant):
            scores = w.float() / cnt.float()
            best_expert = scores.argmax().item()

            # Priority 1: node without this expert, most remaining capacity
            best_node = -1
            best_remaining = -1
            for n in range(num_nodes):
                if remaining[n] > 0 and best_expert not in node_has[n]:
                    if remaining[n] > best_remaining:
                        best_remaining = remaining[n]
                        best_node = n

            # Priority 2: any node with remaining capacity
            if best_node < 0:
                best_remaining = -1
                for n in range(num_nodes):
                    if remaining[n] > 0 and remaining[n] > best_remaining:
                        best_remaining = remaining[n]
                        best_node = n

            if best_node >= 0:
                node_has[best_node].add(best_expert)
                remaining[best_node] -= 1
                node_experts[best_node].append((best_expert, cnt[best_expert].item()))
                cnt[best_expert] += 1

        final_logcnt[layer] = cnt

        # Step 3: pack within each node to GPUs using balanced_packing
        offset = 0
        for n in range(num_nodes):
            experts_on_node = node_experts[n]
            num_on_node = len(experts_on_node)
            assert num_on_node == num_phy_per_node

            # Build weight for packing: weight of each phy expert = w[mlog] / cnt[mlog]
            phy_weights = torch.zeros(1, num_on_node)
            phy_mlogs = []
            phy_ranks = []
            for j, (mlog, rank) in enumerate(experts_on_node):
                phy_weights[0, j] = w[mlog].float() / cnt[mlog].float()
                phy_mlogs.append(mlog)
                phy_ranks.append(rank)

            pack_index, rank_in_pack = balanced_packing(phy_weights, gpus_per_node)
            phy2pphy = pack_index[0] * phy_experts_per_gpu + rank_in_pack[0]

            for j in range(num_on_node):
                pphy_idx = offset + phy2pphy[j].item()
                final_phy2mlog[layer, pphy_idx] = phy_mlogs[j]
                final_phyrank[layer, pphy_idx] = phy_ranks[j]

            offset += num_phy_per_node

    # Convert mlog back to log
    final_phy2log = mlog2log.gather(-1, final_phy2mlog)

    final_logcnt_log = final_logcnt.gather(-1, log2mlog)
    return final_phy2log, final_phyrank, final_logcnt_log


def rebalance_experts_topology_aware_entry(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for topology-aware expert rebalancing.

    Parameters:
        weight: [layers, num_logical_experts]
        num_replicas: number of physical experts
        num_groups: number of expert groups
        num_nodes: number of server nodes
        num_gpus: number of GPUs

    Returns:
        phy2log: [layers, num_replicas]
        log2phy: [layers, num_logical_experts, X]
        logcnt: [layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    phy2log, phyrank, logcnt = rebalance_experts_topology_aware(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
            num_layers, -1
        ),
    )
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts", "rebalance_experts_topology_aware_entry"]
