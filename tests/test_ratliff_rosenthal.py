import pytest
import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import shortest_path

from ware_ops_algos.algorithms import RatliffRosenthalRouting, GreedyItemAssignment
from ware_ops_algos.algorithms.algorithm_interfaces import NodeType
from ware_ops_algos.domain_models import (
    Resource, Location, StorageLocations, StorageType,
    OrderPosition, Order, OrdersDomain, OrderType,
)
from ware_ops_algos.domain_models.layout.graph_generators import ShelfStorageGraphGenerator

# Synthetic instance from the original paper
LOCATIONS = [
    Location(x=1, y=3,  article_id=1,  amount=1),
    Location(x=1, y=8,  article_id=2,  amount=1),
    Location(x=1, y=11, article_id=3,  amount=1),
    Location(x=2, y=5,  article_id=4,  amount=1),
    Location(x=2, y=11, article_id=5,  amount=1),
    Location(x=3, y=3,  article_id=6,  amount=1),
    Location(x=3, y=9,  article_id=7,  amount=1),
    Location(x=3, y=12, article_id=8,  amount=1),
    Location(x=5, y=7,  article_id=9,  amount=1),
    Location(x=6, y=2,  article_id=10, amount=1),
    Location(x=6, y=9,  article_id=11, amount=1),
    Location(x=6, y=10, article_id=12, amount=1),
]

ORDER_POSITIONS = [
    OrderPosition(order_number=1, article_id=i, amount=1)
    for i in range(1, 13)
]

RR_KWARGS = dict(
    start_node=(0, 0),
    end_node=(4, 0),
    closest_node_to_start=(0, 0),
    min_aisle_position=1,
    max_aisle_position=6,
    picker=[Resource(id=1)],
    n_aisles=6,
    n_pick_locations=15,
    dist_aisle=2,
    dist_pick_locations=1,
    dist_aisle_location=1,
    dist_start=1,
    dist_end=1,
)


@pytest.fixture
def pick_list():
    storage = StorageLocations(StorageType.DEDICATED, locations=LOCATIONS)
    storage.build_article_location_mapping()
    order = Order(order_id=1, order_positions=ORDER_POSITIONS)
    orders = OrdersDomain(OrderType.STANDARD, orders=[order])
    ia_sol = GreedyItemAssignment(storage).solve(orders.orders)
    pl = [pos for order in ia_sol.resolved_orders for pos in order.pick_positions]
    return pl


def test_rr_returns_solution(pick_list):
    router = RatliffRosenthalRouting(**RR_KWARGS)
    sol = router.solve(pick_list)
    assert sol is not None
    assert sol.route.distance > 0


def test_rr_reconstructed_tour_has_the_same_distance_and_all_picks(pick_list):
    """The DP objective must be executable as a route through every picked node."""
    graph = ShelfStorageGraphGenerator(
        n_aisles=6, n_pick_locations=15, dist_aisle=2, dist_pick_locations=1,
        dist_aisle_location=1, dist_start=1, dist_end=1,
        start_location=(0, -1), end_location=(0, -2),
        start_connection_point=(1, 0), end_connection_point=(1, 0),
    )
    graph.populate_graph()
    nodes = list(graph.G.nodes)
    distances, predecessors = shortest_path(
        nx.to_scipy_sparse_array(graph.G, nodelist=nodes, weight="weight"),
        directed=False, return_predecessors=True,
    )
    matrix = pd.DataFrame(distances, index=nodes, columns=nodes)
    router = RatliffRosenthalRouting(
        **{**RR_KWARGS, "start_node": (0, -1), "end_node": (0, -2),
           "closest_node_to_start": (1, 0), "distance_matrix": matrix,
           "predecessor_matrix": predecessors,
           "node_to_idx": {node: index for index, node in enumerate(nodes)},
           "idx_to_node": dict(enumerate(nodes)),
           "gen_tour": True, "gen_item_sequence": True},
    )
    route = router.solve(pick_list).route
    positions = [node.position for node in route.annotated_route]
    picks = [node.position for node in route.annotated_route
             if node.node_type == NodeType.PICK]

    assert positions[0] == (0, -1)
    assert positions[-1] == (0, -2)
    assert sorted(picks) == sorted(position.pick_node for position in pick_list)
    assert len(route.item_sequence) == len(pick_list)
    assert all(a == b or graph.G.has_edge(a, b) for a, b in zip(positions, positions[1:]))
    assert sum(matrix.at[a, b] for a, b in zip(positions, positions[1:])) == pytest.approx(
        route.distance)
