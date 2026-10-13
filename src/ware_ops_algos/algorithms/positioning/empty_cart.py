"""Empty-cart movement chosen after ordinary jobs have been scheduled."""

from __future__ import annotations

import networkx as nx

from ware_ops_algos.domain_models import LayoutData

from ..algorithm_interfaces import BatchObject, NodeType, Route, RouteNode
from ..routing.patrol import all_aisles_patrol_route
from .positioning import Positioning


def center_relocation_route(layout: LayoutData) -> Route:
    """Move to an accessible central graph node and hold there."""
    network = layout.layout_network
    graph = network.graph
    pick_nodes = [node for node in graph if 1 <= node[1] <= layout.graph_data.n_pick_locations]
    if not pick_nodes:
        raise ValueError("Center relocation needs accessible pick-location nodes")
    midpoint = (sum(node[0] for node in pick_nodes) / len(pick_nodes),
                sum(node[1] for node in pick_nodes) / len(pick_nodes))
    center = min(pick_nodes, key=lambda node:
                 ((node[0] - midpoint[0]) ** 2 + (node[1] - midpoint[1]) ** 2,
                  node))
    nodes = nx.shortest_path(graph, network.start_node, center, weight="weight")
    distance = nx.path_weight(graph, nodes, weight="weight")
    return Route(distance=float(distance), route=nodes, item_sequence=[],
                 annotated_route=[RouteNode(node, NodeType.ROUTE) for node in nodes],
                 batch=BatchObject(batch_id=0, orders=[]))


class AislePatrolPositioning(Positioning):
    algo_name = "AislePatrolPositioning"
    mode = "aisle_patrol"

    def _route(self, layout: LayoutData) -> Route:
        return all_aisles_patrol_route(layout)


class CenterRelocationPositioning(Positioning):
    algo_name = "CenterRelocationPositioning"
    mode = "center_relocation"

    def _route(self, layout: LayoutData) -> Route:
        return center_relocation_route(layout)
