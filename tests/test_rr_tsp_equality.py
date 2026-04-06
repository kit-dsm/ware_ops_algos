import unittest

import networkx as nx
import pandas as pd

from ware_ops_algos.algorithms import RatliffRosenthalRouting, GreedyItemAssignment, ExactTSPRoutingDistance
from ware_ops_algos.data_loaders import HesslerIrnichLoader
from ware_ops_algos.domain_models import Resource, Location, StorageLocations, StorageType, OrderPosition, Order, \
    OrdersDomain, OrderType
from ware_ops_algos.utils.io_helpers import find_project_root


class RRTSPEquality(unittest.TestCase):
    instance_set = "SPRP"
    instance_name = "unit_F1_m5_C30_a7_12.txt"

    def test_rr_tsp_equality(self):
        PROJECT_ROOT = find_project_root()

        DATA_DIR = PROJECT_ROOT / "data"

        instances_base = DATA_DIR / "instances"
        cache_base = DATA_DIR / "instances" / "caches"
        loader_hi = HesslerIrnichLoader(
            instances_base / self.instance_set,
            cache_base / self.instance_set
        )

        results_sprp = pd.read_csv(DATA_DIR / f"results/results_{self.instance_set}.csv",
                                  sep=";",
                                  decimal=",",
                                  thousands=".")
        results_sprp["filename"] = results_sprp.apply(
            lambda
                row: f"unit_F1_m{row['num aisles']}_C{row['num cells']}_a{row['num articles']}_{row['random seed']}.txt",
            axis=1
        )

        instance_result = results_sprp[results_sprp["filename"] == self.instance_name]

        lit_result = instance_result["DP cost"].item()

        domain = loader_hi.load(filepath=self.instance_name)
        layout = domain.layout
        resources = domain.resources
        layout_network = layout.layout_network
        graph = layout_network.graph
        graph_params = layout.graph_data
        storage_locations = domain.storage
        orders = domain.orders

        tsp_router = ExactTSPRoutingDistance(
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
            set_time_limit=3600,
            node_list=layout_network.node_list,
            node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
            idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
        )

        rr_router = RatliffRosenthalRouting(
            start_node=layout.graph_data.start_connection_point,
            end_node=layout_network.end_node,
            closest_node_to_start=layout_network.closest_node_to_start,
            min_aisle_position=layout_network.min_aisle_position,
            max_aisle_position=layout_network.max_aisle_position,
            picker=resources.resources,
            n_aisles=graph_params.n_aisles,
            n_pick_locations=graph_params.n_pick_locations,
            dist_aisle=graph_params.dist_aisle,
            dist_pick_locations=graph_params.dist_pick_locations,
            dist_aisle_location=graph_params.dist_bottom_to_pick_location,
            dist_start=graph_params.dist_start,
            dist_end=graph_params.dist_end,
        )

        selector = GreedyItemAssignment(storage_locations)
        # for order in orders:
        ia_sol = selector.solve(orders.orders)

        pick_lists = []
        for order in ia_sol.resolved_orders:
            pick_list = []
            for pos in order.pick_positions:
                pick_list.append(pos)
            pick_lists.append(pick_list)

        assert len(pick_lists) == 1

        dist_rr = 0
        dist_tsp = 0
        for pl in pick_lists:
            sol_rr = rr_router.solve(pl)
            sol_tsp = tsp_router.solve(pl)
            rr_router.reset_parameters()
            tsp_router.reset_parameters()

            dist_rr += sol_rr.route.distance
            dist_tsp += sol_tsp.route.distance
        self.assertEqual(dist_tsp, dist_rr)
        self.assertEqual(dist_rr, lit_result)


if __name__ == '__main__':
    unittest.main()
