import pytest

from ware_ops_algos.algorithms import (
    OrderNrFifoBatching,
    LocalSearchBatching,
    ClarkAndWrightBatching,
    NearestNeighbourhoodRouting,
    RatliffRosenthalRouting, Batching, BatchingSolution,
)
from ware_ops_algos.domain_models import Order


def assert_valid_batching_result(batching_sol: BatchingSolution, resolved_orders: list[Order], algo: Batching):
    assert batching_sol is not None
    assert len(batching_sol.batches) > 0
    assert all(algo.capacity_checker.orders_fit(b.orders) for b in batching_sol.batches)
    input_ids = {o.order_id for o in resolved_orders}
    output_ids = {o.order_id for b in batching_sol.batches for o in b.orders}
    assert input_ids == output_ids


def _routing_kwargs(domain):
    ln = domain.layout.layout_network
    return {
        "start_node": ln.start_node,
        "end_node": ln.end_node,
        "closest_node_to_start": ln.closest_node_to_start,
        "min_aisle_position": ln.min_aisle_position,
        "max_aisle_position": ln.max_aisle_position,
        "distance_matrix": ln.distance_matrix,
        "predecessor_matrix": ln.predecessor_matrix,
        "picker": domain.resources.resources,
        "gen_tour": False,
        "gen_item_sequence": False,
        "node_list": ln.node_list,
        "node_to_idx": {node: idx for idx, node in enumerate(ln.graph.nodes)},
        "idx_to_node": {idx: node for idx, node in enumerate(ln.graph.nodes)},
    }


def _rr_routing_kwargs(domain):
    ln = domain.layout.layout_network
    gd = domain.layout.graph_data
    return {
        "start_node": ln.start_node,
        "end_node": ln.end_node,
        "closest_node_to_start": ln.closest_node_to_start,
        "min_aisle_position": ln.min_aisle_position,
        "max_aisle_position": ln.max_aisle_position,
        "distance_matrix": ln.distance_matrix,
        "predecessor_matrix": ln.predecessor_matrix,
        "picker": domain.resources.resources,
        "n_aisles": gd.n_aisles,
        "n_pick_locations": gd.n_pick_locations,
        "dist_aisle": gd.dist_aisle,
        "dist_pick_locations": gd.dist_pick_locations,
        "dist_aisle_location": gd.dist_bottom_to_pick_location,
        "dist_start": gd.dist_start,
        "dist_end": gd.dist_end,
        "gen_tour": False,
        "gen_item_sequence": False,
    }

@pytest.mark.benchmark(group="constructive_batching")
def test_fifo_batching_henn(benchmark, henn_domain, henn_resolved_orders):
    pick_cart = henn_domain.resources.resources[0].pick_cart
    articles = henn_domain.articles
    algo = OrderNrFifoBatching(pick_cart=pick_cart, articles=articles)
    result = benchmark(algo.solve, henn_resolved_orders)
    assert_valid_batching_result(result, henn_resolved_orders, algo)

@pytest.mark.benchmark(group="local_search_batching")
def test_local_search_batching_henn(benchmark, henn_domain, henn_resolved_orders):
    pick_cart = henn_domain.resources.resources[0].pick_cart
    articles = henn_domain.articles
    algo = LocalSearchBatching(
        pick_cart=pick_cart,
        articles=articles,
        routing_class=RatliffRosenthalRouting,
        routing_class_kwargs=_rr_routing_kwargs(henn_domain),
        start_batching_class=OrderNrFifoBatching,
    )
    result = benchmark(algo.solve, henn_resolved_orders)
    assert_valid_batching_result(result, henn_resolved_orders, algo)


@pytest.mark.benchmark(group="savings_batching")
def test_clark_wright_batching_henn(benchmark, henn_domain, henn_resolved_orders):
    pick_cart = henn_domain.resources.resources[0].pick_cart
    articles = henn_domain.articles
    algo = ClarkAndWrightBatching(
        pick_cart=pick_cart,
        articles=articles,
        routing_class=NearestNeighbourhoodRouting,
        routing_class_kwargs=_routing_kwargs(henn_domain),
    )
    result = benchmark(algo.solve, henn_resolved_orders)
    assert_valid_batching_result(result, henn_resolved_orders, algo)


# ---------------------------------------------------------------------------
# MuterOencan
# ---------------------------------------------------------------------------

@pytest.mark.benchmark(group="constructive_batching")
def test_fifo_batching_muter(benchmark, muter_domain, muter_resolved_orders):
    pick_cart = muter_domain.resources.resources[0].pick_cart
    articles = muter_domain.articles
    algo = OrderNrFifoBatching(pick_cart=pick_cart, articles=articles)
    result = benchmark(algo.solve, muter_resolved_orders)
    assert_valid_batching_result(result, muter_resolved_orders, algo)


@pytest.mark.benchmark(group="local_search_batching")
def test_local_search_batching_muter(benchmark, muter_domain, muter_resolved_orders):
    pick_cart = muter_domain.resources.resources[0].pick_cart
    articles = muter_domain.articles
    algo = LocalSearchBatching(
        pick_cart=pick_cart,
        articles=articles,
        routing_class=RatliffRosenthalRouting,
        routing_class_kwargs=_rr_routing_kwargs(muter_domain),
        start_batching_class=OrderNrFifoBatching,
    )
    result = benchmark(algo.solve, muter_resolved_orders)
    assert_valid_batching_result(result, muter_resolved_orders, algo)


@pytest.mark.benchmark(group="savings_batching")
def test_clark_wright_batching_muter(benchmark, muter_domain, muter_resolved_orders):
    pick_cart = muter_domain.resources.resources[0].pick_cart
    articles = muter_domain.articles
    algo = ClarkAndWrightBatching(
        pick_cart=pick_cart,
        articles=articles,
        routing_class=NearestNeighbourhoodRouting,
        routing_class_kwargs=_routing_kwargs(muter_domain),
    )
    result = benchmark(algo.solve, muter_resolved_orders)
    assert_valid_batching_result(result, muter_resolved_orders, algo)