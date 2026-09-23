"""The empty walking tour used by the paper's start-immediately policy."""

import networkx as nx

from ware_ops_algos.algorithms.algorithm_interfaces import BatchObject, NodeType, Route, RouteNode
from ware_ops_algos.domain_models import LayoutData


def all_aisles_patrol_route(layout: LayoutData) -> Route:
    """Walk every aisle end-to-end in a one-block layout, without fictitious picks."""
    params = layout.graph_data
    network = layout.layout_network
    if params.n_blocks != 1:
        raise ValueError("Empty patrol currently supports one-block layouts only")
    graph = network.graph
    front, back = network.min_aisle_position, network.max_aisle_position
    waypoints = [network.start_node]
    for aisle in range(1, params.n_aisles + 1):
        waypoints.extend(((aisle, front), (aisle, back)) if aisle % 2 else
                         ((aisle, back), (aisle, front)))
    waypoints.append(network.end_node)
    if any(node not in graph for node in waypoints):
        raise ValueError("Empty patrol needs both cross-aisle endpoints for every aisle")

    nodes = [waypoints[0]]
    for target in waypoints[1:]:
        nodes.extend(nx.shortest_path(graph, nodes[-1], target, weight="weight")[1:])
    distance = sum(graph[a][b]["weight"] for a, b in zip(nodes, nodes[1:]))
    return Route(
        distance=float(distance),
        route=nodes,
        item_sequence=[],
        annotated_route=[RouteNode(node, NodeType.ROUTE) for node in nodes],
        batch=BatchObject(batch_id=0, orders=[]),
    )
