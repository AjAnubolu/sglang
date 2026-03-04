import torch


def inter_node_traffic_ratio(
    phy2log: torch.Tensor,
    weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Estimate the fraction of token traffic that must go inter-node.

    For each expert, count how many distinct nodes host it (coverage K out of N).
    Traffic that must go inter-node = weight[expert] * (1 - K/N).
    Returns the weighted fraction per layer.

    Parameters:
        phy2log: [num_layers, num_physical_experts]
        weight: [num_layers, num_logical_experts]
        num_nodes: number of server nodes

    Returns:
        ratio: [num_layers], fraction of traffic that is inter-node
    """
    num_layers, num_physical_experts = phy2log.shape
    num_logical_experts = weight.shape[1]
    num_phy_per_node = num_physical_experts // num_nodes

    ratios = torch.zeros(num_layers)
    for layer in range(num_layers):
        total_weight = weight[layer].sum().item()
        if total_weight == 0:
            continue
        inter_node_weight = 0.0
        for expert in range(num_logical_experts):
            # Count how many distinct nodes host this expert
            nodes_hosting = set()
            for node in range(num_nodes):
                start = node * num_phy_per_node
                end = start + num_phy_per_node
                if expert in phy2log[layer, start:end].tolist():
                    nodes_hosting.add(node)
            coverage = len(nodes_hosting) / num_nodes
            inter_node_weight += weight[layer, expert].item() * (1.0 - coverage)
        ratios[layer] = inter_node_weight / total_weight

    return ratios


def inter_rank_imbalance_ratio(
    phy2log: torch.Tensor,
    weight: torch.Tensor,
    logcnt: torch.Tensor,
    num_gpus: int,
) -> torch.Tensor:
    """
    Compute the load imbalance ratio across GPUs.

    Per-GPU load = sum of weight[expert] / count[expert] for each physical expert
    assigned to that GPU. Returns max(gpu_load) / mean(gpu_load) per layer.
    A value of 1.0 means perfect balance.

    Parameters:
        phy2log: [num_layers, num_physical_experts]
        weight: [num_layers, num_logical_experts]
        logcnt: [num_layers, num_logical_experts]
        num_gpus: total number of GPUs

    Returns:
        ratio: [num_layers], imbalance ratio (>= 1.0)
    """
    num_layers, num_physical_experts = phy2log.shape
    phy_per_gpu = num_physical_experts // num_gpus

    ratios = torch.zeros(num_layers)
    for layer in range(num_layers):
        gpu_loads = torch.zeros(num_gpus)
        for gpu in range(num_gpus):
            start = gpu * phy_per_gpu
            end = start + phy_per_gpu
            for phy_idx in range(start, end):
                expert = phy2log[layer, phy_idx].item()
                cnt = logcnt[layer, expert].item()
                gpu_loads[gpu] += weight[layer, expert].item() / cnt
        mean_load = gpu_loads.mean().item()
        if mean_load > 0:
            ratios[layer] = gpu_loads.max().item() / mean_load
        else:
            ratios[layer] = 1.0

    return ratios
