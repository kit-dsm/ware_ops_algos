import pandas as pd
import pytest

from ware_ops_algos.algorithms import ExactCombinedBatchingRouting, GreedyItemAssignment, CombinedRoutingSolution
from ware_ops_algos.data_loaders import HesslerIrnichResultsLoader
from ware_ops_algos.utils.io_helpers import find_project_root

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
INSTANCE_NAME = "Pr_20_1_20_Store1_01.txt"


@pytest.fixture(scope="module")
def literature_lb():
    results = pd.read_csv(
        DATA_DIR / "results/results_BahceciOencan.csv",
        sep=";", decimal=",", thousands=".",
    )
    return results[results["filename"] == INSTANCE_NAME]["LB"].item()


@pytest.fixture(scope="module")
def cbr_solution(bahceci_domain):
    domain = bahceci_domain
    layout_network = domain.layout.layout_network
    graph = layout_network.graph

    router = ExactCombinedBatchingRouting(
        start_node=layout_network.start_node,
        end_node=layout_network.end_node,
        distance_matrix=layout_network.distance_matrix,
        predecessor_matrix=layout_network.predecessor_matrix,
        picker=domain.resources.resources,
        gen_tour=True,
        gen_item_sequence=True,
        time_limit=3600,
        node_list=layout_network.node_list,
        node_to_idx={node: idx for idx, node in enumerate(graph.nodes)},
        idx_to_node={idx: node for idx, node in enumerate(graph.nodes)},
    )

    ia_sol = GreedyItemAssignment(domain.storage).solve(domain.orders.orders)
    return router.solve(ia_sol.resolved_orders)


def test_cbr_matches_literature_lb(cbr_solution, literature_lb):
    total_distance = sum(r.distance for r in cbr_solution.routes)
    assert total_distance == literature_lb