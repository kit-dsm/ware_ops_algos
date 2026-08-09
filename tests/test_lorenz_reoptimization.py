from __future__ import annotations

import networkx as nx

from ware_ops_algos.algorithms import (
    PickPosition,
    WarehouseOrder,
    exact_lorenz_makespan,
)
from ware_ops_algos.data_loaders.generators.layout.graph_generator import (
    distance_matrix_generator,
)
from ware_ops_algos.domain_models import (
    LayoutData,
    LayoutNetwork,
    LayoutParameters,
    LayoutType,
    Resource,
)


def _layout() -> LayoutData:
    start = (0, 0)
    pick = (0, 3)
    end = (-1, 0)
    graph = nx.Graph()
    graph.add_edge(start, pick, weight=3.0)
    graph.add_edge(pick, end, weight=3.0)
    graph.add_edge(start, end, weight=0.0)
    nodes = [start, pick, end]
    predecessors = {
        start: {pick: start, end: start},
        pick: {start: pick, end: pick},
        end: {start: end, pick: end},
    }
    return LayoutData(
        tpe=LayoutType.CONVENTIONAL,
        graph_data=LayoutParameters(
            n_aisles=1,
            n_pick_locations=1,
            n_blocks=1,
            dist_top_to_pick_location=1.0,
            dist_bottom_to_pick_location=1.0,
            dist_pick_locations=1.0,
            dist_aisle=1.0,
            dist_start=0.0,
            dist_end=0.0,
            start_location=start,
            end_location=end,
        ),
        layout_network=LayoutNetwork(
            graph=graph,
            distance_matrix=distance_matrix_generator(graph),
            predecessor_matrix=predecessors,
            closest_node_to_start=start,
            start_node=start,
            end_node=end,
            node_list=nodes,
            min_aisle_position=0,
            max_aisle_position=4,
        ),
    )


def test_exact_lorenz_makespan_respects_release_and_reconstructs_route():
    order = WarehouseOrder(
        order_id=1,
        order_date=10.0,
        pick_positions=(
            PickPosition(
                order_number=1,
                article_id=1,
                amount=1,
                pick_node=(0, 3),
                in_store=1,
            ),
        ),
    )
    picker = Resource(
        id=0,
        speed=1.0,
        time_per_pick=5.0,
        tour_setup_time=0.0,
    )
    solution = exact_lorenz_makespan(
        [order],
        _layout(),
        picker,
        batch_capacity_orders=1,
    )
    assert solution.is_optimal is True
    assert solution.solver_status == "optimal"
    assert solution.objective_value == 18.0
    assert solution.jobs[0].start_time == 0.0
    assert solution.jobs[0].end_time == 18.0
    assert solution.jobs[0].job.route.item_sequence == [(0, 3)]
    assert solution.jobs[0].job.route.picking_times == [15.0]
