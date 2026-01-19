from pathlib import Path
import random

from ware_ops_algos.algorithms import (GreedyItemAssignment,
                                       OrderNrFifoBatching,
                                       ExactTSPRoutingDistance,
                                       NearestNeighbourhoodRouting,
                                       SShapeRouting)
from ware_ops_algos.data_loaders import FoodmartLoader
from ware_ops_algos.utils.visualization import plot_route

instance_set = "FoodmartData"

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src" / "ware_ops_algos"

instances_dir = DATA_DIR / "instances" / instance_set
results_dir = DATA_DIR / "results"

il = FoodmartLoader(instances_dir=instances_dir)
domain = il.load("instances_d5_ord5_MAL.txt")

orders = domain.orders
layout = domain.layout
resources = domain.resources
articles = domain.articles
storage_locations = domain.storage

layout_network = layout.layout_network
graph_data = layout.graph_data
graph = layout_network.graph
graph_params = layout.graph_data
dima = layout_network.distance_matrix


selector = GreedyItemAssignment(storage_locations)
ia_sol = selector.solve(orders.orders)
orders.orders = ia_sol.resolved_orders

batcher = OrderNrFifoBatching(
            pick_cart=resources.resources[0].pick_cart,
            articles=articles
        )

batching_sol = batcher.solve(orders.orders)

print(batching_sol.execution_time)

# Build pick list from batches
batches = batching_sol.batches
pick_lists = []
for batch in batches:
    pick_list = []
    for order in batch.orders:
        for pos in order.pick_positions:
            pick_list.append(pos)
    pick_lists.append(pick_list)

tsp_routing = ExactTSPRoutingDistance(
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            big_m=1000,
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            set_time_limit=120,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
        )

nn_router = NearestNeighbourhoodRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))}
        )

ss_router = SShapeRouting(
            start_node=layout_network.start_node,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            distance_matrix=layout_network.distance_matrix,
            predecessor_matrix=layout_network.predecessor_matrix,
            picker=resources.resources,
            gen_tour=True,
            gen_item_sequence=True,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))}
        )

total_dist = 0
for pl in pick_lists:
    sol = ss_router.solve(pl)
    total_dist += sol.route.distance
    plot_route(graph, sol.route.route)
    ss_router.reset_parameters()
print(total_dist)
