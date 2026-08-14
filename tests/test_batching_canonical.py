from collections import Counter

import pytest

from ware_ops_algos.algorithms.algorithm_interfaces import BatchingSolution
from ware_ops_algos.algorithms.batching.batching import Batching, OrderNrFifoBatching
from ware_ops_algos.algorithms.batching.local_search_batching import LocalSearchBatching
from ware_ops_algos.algorithms.batching.moves import ShiftNeighborhood, SwapNeighborhood
from ware_ops_algos.algorithms.batching.savings_batching import ClarkAndWrightBatching
from ware_ops_algos.algorithms.batching.seed_batching import (
    ClosestToDepotSeed,
    FewestPositionsSeed,
    MinDistanceSimilarity,
    MostPositionsSeed,
    RandomSeed,
    SeedBatching,
    SharedArticlesSimilarity,
)
from ware_ops_algos.algorithms.routing.routing import (
    NearestNeighbourhoodRouting,
    RatliffRosenthalRouting,
)


CASES = [
    pytest.param(
        "henn_domain",
        "henn_resolved_orders",
        ((0, 1), (4, 7, 11)),
        903.0,
        ((0, 1), (7, 4, 11)),
        id="henn",
    ),
    pytest.param(
        "muter_domain",
        "muter_resolved_orders",
        ((91, 0, 1, 3, 5, 10, 13, 17, 18, 28, 36, 48, 55), (4,)),
        178.4,
        ((4,), (55, 17, 18, 3, 5, 48, 1, 0, 36, 91, 28, 10, 13)),
        id="muter",
    ),
]


def _routing_kwargs(domain):
    network = domain.layout.layout_network
    return {
        "start_node": network.start_node,
        "end_node": network.end_node,
        "closest_node_to_start": network.closest_node_to_start,
        "min_aisle_position": network.min_aisle_position,
        "max_aisle_position": network.max_aisle_position,
        "distance_matrix": network.distance_matrix,
        "predecessor_matrix": network.predecessor_matrix,
        "picker": domain.resources.resources,
        "gen_tour": False,
        "gen_item_sequence": False,
        "node_list": network.node_list,
        "node_to_idx": {node: i for i, node in enumerate(network.graph.nodes)},
        "idx_to_node": {i: node for i, node in enumerate(network.graph.nodes)},
    }


def _rr_kwargs(domain):
    result = _routing_kwargs(domain)
    data = domain.layout.graph_data
    result.update({
        "n_aisles": data.n_aisles,
        "n_pick_locations": data.n_pick_locations,
        "dist_aisle": data.dist_aisle,
        "dist_pick_locations": data.dist_pick_locations,
        "dist_aisle_location": data.dist_bottom_to_pick_location,
        "dist_start": data.dist_start,
        "dist_end": data.dist_end,
    })
    return result


def _memberships(solution: BatchingSolution):
    return tuple(
        tuple(order.order_id for order in batch.orders)
        for batch in solution.batches
    )


def _assert_valid(solution, orders, algorithm: Batching):
    assert all(batch.orders for batch in solution.batches)
    assert all(
        algorithm.capacity_checker.orders_fit(batch.orders)
        for batch in solution.batches
    )
    assert Counter(
        order.order_id for batch in solution.batches for order in batch.orders
    ) == Counter(order.order_id for order in orders)


@pytest.mark.parametrize(
    "domain_fixture,orders_fixture,expected,objective,_savings",
    CASES,
)
def test_canonical_local_search_regression(
    request,
    domain_fixture,
    orders_fixture,
    expected,
    objective,
    _savings,
):
    domain = request.getfixturevalue(domain_fixture)
    orders = request.getfixturevalue(orders_fixture)
    algorithm = LocalSearchBatching(
        pick_cart=domain.resources.resources[0].pick_cart,
        articles=domain.articles,
        routing_class=RatliffRosenthalRouting,
        routing_class_kwargs=_rr_kwargs(domain),
        start_batching_class=OrderNrFifoBatching,
        neighborhood_classes=[SwapNeighborhood, ShiftNeighborhood],
        time_limit=30.0,
    )
    solution = algorithm.solve(orders)

    _assert_valid(solution, orders, algorithm)
    assert _memberships(solution) == expected
    assert solution.objective_trajectory[-1][1] == pytest.approx(objective)
    assert algorithm.search_statistics["final_objective"] == pytest.approx(objective)
    assert algorithm.search_statistics["full_capacity_rescans"] == 0
    assert algorithm.search_statistics["full_objective_resums"] == 0
    assert algorithm.search_statistics["candidate_batch_objects"] == 0


@pytest.mark.parametrize(
    "domain_fixture,orders_fixture,_local,_objective,expected",
    CASES,
)
def test_canonical_savings_regression(
    request,
    domain_fixture,
    orders_fixture,
    _local,
    _objective,
    expected,
):
    domain = request.getfixturevalue(domain_fixture)
    orders = request.getfixturevalue(orders_fixture)
    algorithm = ClarkAndWrightBatching(
        pick_cart=domain.resources.resources[0].pick_cart,
        articles=domain.articles,
        routing_class=NearestNeighbourhoodRouting,
        routing_class_kwargs=_routing_kwargs(domain),
    )
    solution = algorithm.solve(orders)

    _assert_valid(solution, orders, algorithm)
    assert _memberships(solution) == expected


@pytest.mark.parametrize(
    "seed_name,similarity_name,expected",
    [
        ("fewest", "shared", ((1, 0), (4, 7, 11))),
        ("most", "shared", ((0, 1), (11, 4, 7))),
        ("closest", "shared", ((0, 1), (4, 7, 11))),
        ("fewest", "distance", ((1, 4, 7), (11,), (0,))),
        ("most", "distance", ((0, 1), (11, 4, 7))),
        ("closest", "distance", ((0, 1), (4, 7, 11))),
    ],
)
def test_seed_strategy_combinations_match_baseline(
    henn_domain,
    henn_resolved_orders,
    seed_name,
    similarity_name,
    expected,
):
    network = henn_domain.layout.layout_network
    seeds = {
        "fewest": FewestPositionsSeed,
        "most": MostPositionsSeed,
        "closest": lambda: ClosestToDepotSeed(
            network.distance_matrix,
            network.start_node,
        ),
    }
    similarities = {
        "shared": SharedArticlesSimilarity,
        "distance": lambda: MinDistanceSimilarity(network.distance_matrix),
    }
    algorithm = SeedBatching(
        pick_cart=henn_domain.resources.resources[0].pick_cart,
        articles=henn_domain.articles,
        seed_criterion=seeds[seed_name](),
        similarity_measure=similarities[similarity_name](),
    )

    solution = algorithm.solve(henn_resolved_orders)
    _assert_valid(solution, henn_resolved_orders, algorithm)
    assert _memberships(solution) == expected


def test_random_seed_contract_is_repeatable(henn_domain, henn_resolved_orders):
    algorithm = SeedBatching(
        pick_cart=henn_domain.resources.resources[0].pick_cart,
        articles=henn_domain.articles,
        seed_criterion=RandomSeed(seed=17),
        similarity_measure=SharedArticlesSimilarity(),
    )
    first = algorithm.solve(henn_resolved_orders)
    second = algorithm.solve(henn_resolved_orders)
    assert _memberships(second) == _memberships(first)
