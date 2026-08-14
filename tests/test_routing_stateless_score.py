import importlib

import pytest

routing_module = importlib.import_module(
    "ware_ops_algos.algorithms.routing.routing"
)
from ware_ops_algos.algorithms.algorithm_interfaces import NodeType, PickPosition
from ware_ops_algos.algorithms.routing.routing import (
    LargestGapRouting,
    MidpointRouting,
    NearestNeighbourhoodRouting,
    PickListRouting,
    RatliffRosenthalRouting,
    ReturnRouting,
    SShapeRouting,
    UShapeRouting,
)
from ware_ops_algos.algorithms import (
    ClarkAndWrightBatching,
    LocalSearchBatching,
    OrderNrFifoBatching,
)


HEURISTICS = (
    SShapeRouting,
    ReturnRouting,
    MidpointRouting,
    LargestGapRouting,
    NearestNeighbourhoodRouting,
    UShapeRouting,
    PickListRouting,
)

CAPTURED_MUTER_RESULTS = {
    SShapeRouting: (131.2, [(2, 5), (3, 3), (5, 2), (6, 9), (6, 6), (6, 1), (7, 4), (8, 2), (9, 4), (10, 1)]),
    ReturnRouting: (165.2, [(2, 5), (3, 3), (5, 2), (6, 1), (6, 6), (6, 9), (7, 4), (8, 2), (9, 4), (10, 1)]),
    MidpointRouting: (107.2, [(3, 3), (5, 2), (6, 1), (7, 4), (8, 2), (9, 4), (10, 1), (6, 6), (6, 9), (2, 5)]),
    LargestGapRouting: (107.2, [(3, 3), (5, 2), (6, 1), (7, 4), (8, 2), (9, 4), (10, 1), (6, 6), (6, 9), (2, 5)]),
    NearestNeighbourhoodRouting: (99.2, [(2, 5), (3, 3), (5, 2), (6, 1), (6, 6), (6, 9), (7, 4), (8, 2), (10, 1), (9, 4)]),
    UShapeRouting: (172.4, [(2, 5), (3, 3), (6, 1), (6, 6), (6, 9), (5, 2), (8, 2), (7, 4), (10, 1), (9, 4)]),
    PickListRouting: (141.2, [(6, 9), (8, 2), (6, 6), (3, 3), (10, 1), (9, 4), (7, 4), (6, 1), (5, 2), (2, 5)]),
}

TRANSIENT_NAMES = {
    "pick_list",
    "current_order",
    "distance",
    "route",
    "annotated_route",
    "item_sequence",
    "decisions",
    "path",
    "T",
    "execution_time",
}


def _routing_kwargs(domain, **flags):
    network = domain.layout.layout_network
    graph = network.graph
    kwargs = {
        "start_node": network.start_node,
        "end_node": network.end_node,
        "closest_node_to_start": network.closest_node_to_start,
        "min_aisle_position": network.min_aisle_position,
        "max_aisle_position": network.max_aisle_position,
        "distance_matrix": network.distance_matrix,
        "predecessor_matrix": network.predecessor_matrix,
        "picker": domain.resources.resources,
        "node_list": network.node_list,
        "node_to_idx": {node: index for index, node in enumerate(graph.nodes)},
        "idx_to_node": {index: node for index, node in enumerate(graph.nodes)},
        "gen_tour": False,
        "gen_item_sequence": False,
    }
    kwargs.update(flags)
    return kwargs


def _rr_kwargs(domain, **flags):
    network = domain.layout.layout_network
    data = domain.layout.graph_data
    kwargs = {
        "start_node": network.start_node,
        "end_node": network.end_node,
        "closest_node_to_start": network.closest_node_to_start,
        "min_aisle_position": network.min_aisle_position,
        "max_aisle_position": network.max_aisle_position,
        "picker": domain.resources.resources,
        "n_aisles": data.n_aisles,
        "n_pick_locations": data.n_pick_locations,
        "dist_aisle": data.dist_aisle,
        "dist_pick_locations": data.dist_pick_locations,
        "dist_aisle_location": data.dist_bottom_to_pick_location,
        "dist_start": data.dist_start,
        "dist_end": data.dist_end,
        "gen_tour": False,
        "gen_item_sequence": False,
    }
    kwargs.update(flags)
    return kwargs


def _pick_sets(orders):
    first = [position for order in orders[:4] for position in order.pick_positions]
    second = [position for order in orders[4:7] for position in order.pick_positions]
    return first, second


@pytest.mark.parametrize("routing_class", HEURISTICS)
def test_constructive_score_matches_solve_and_calls_are_independent(
    routing_class,
    muter_domain,
    muter_resolved_orders,
):
    first, second = _pick_sets(muter_resolved_orders)
    router = routing_class(**_routing_kwargs(muter_domain))

    first_score = router.score(first)
    second_score = router.score(second)

    assert router.solve(first).route.distance == pytest.approx(first_score)
    assert first_score == pytest.approx(CAPTURED_MUTER_RESULTS[routing_class][0])
    assert router.solve(second).route.distance == pytest.approx(second_score)
    assert router.score(first) == pytest.approx(first_score)
    assert TRANSIENT_NAMES.isdisjoint(router.__dict__)
    for pick_count in (0, 1):
        edge_picks = first[:pick_count]
        assert router.score(edge_picks) == pytest.approx(
            router.solve(edge_picks).route.distance
        )


@pytest.mark.parametrize("routing_class", HEURISTICS)
def test_distance_only_calls_allocate_no_output_work(
    routing_class,
    muter_domain,
    muter_resolved_orders,
    monkeypatch,
):
    picks, _ = _pick_sets(muter_resolved_orders)
    router = routing_class(**_routing_kwargs(muter_domain))

    def fail(*args, **kwargs):
        raise AssertionError("path expansion reached the distance-only path")

    monkeypatch.setattr(router, "_get_route_segment", fail)
    assert router.score(picks) == pytest.approx(router.solve(picks).route.distance)

    def fail_route(*args, **kwargs):
        raise AssertionError("score allocated a Route")

    monkeypatch.setattr(routing_module, "Route", fail_route)
    assert router.score(picks) > 0


@pytest.mark.parametrize("routing_class", HEURISTICS)
def test_output_flags_and_duplicate_pick_positions(
    routing_class,
    muter_domain,
    muter_resolved_orders,
):
    picks, _ = _pick_sets(muter_resolved_orders)
    original = picks[0]
    duplicate = PickPosition(
        order_number=999,
        article_id=999,
        amount=1,
        pick_node=original.pick_node,
        in_store=1,
    )
    score_router = routing_class(**_routing_kwargs(muter_domain))
    output_router = routing_class(**_routing_kwargs(
        muter_domain, gen_tour=True, gen_item_sequence=True,
    ))
    assert output_router.solve(picks).route.item_sequence == (
        CAPTURED_MUTER_RESULTS[routing_class][1]
    )
    picks = [*picks, duplicate]
    route = output_router.solve(picks).route

    assert route.distance == pytest.approx(score_router.score(picks))
    assert route.route
    assert route.item_sequence.count(original.pick_node) == 2
    assert sum(
        node.node_type == NodeType.PICK and node.position == original.pick_node
        for node in route.annotated_route
    ) == 2
    assert TRANSIENT_NAMES.isdisjoint(output_router.__dict__)


def test_rr_score_is_stateless_and_skips_backtracking(
    muter_domain,
    muter_resolved_orders,
    monkeypatch,
):
    first, second = _pick_sets(muter_resolved_orders)
    router = RatliffRosenthalRouting(**_rr_kwargs(muter_domain))
    first_score = router.score(first)
    second_score = router.score(second)
    assert router.solve(second).route.distance == pytest.approx(second_score)

    def fail(*args, **kwargs):
        raise AssertionError("score allocated reconstruction state")

    monkeypatch.setattr(router, "_backtrack_decisions", fail)
    monkeypatch.setattr(routing_module, "Route", fail)
    assert router.score(first) == pytest.approx(first_score)
    assert TRANSIENT_NAMES.isdisjoint(router.__dict__)


@pytest.mark.parametrize("batching_kind", ("local_search", "clark_wright"))
def test_candidate_batch_scoring_allocates_no_routes(
    batching_kind,
    muter_domain,
    muter_resolved_orders,
    monkeypatch,
):
    cart = muter_domain.resources.resources[0].pick_cart
    articles = muter_domain.articles
    if batching_kind == "local_search":
        algorithm = LocalSearchBatching(
            cart,
            articles,
            RatliffRosenthalRouting,
            _rr_kwargs(muter_domain),
            OrderNrFifoBatching,
            time_limit=1.0,
        )
    else:
        algorithm = ClarkAndWrightBatching(
            cart,
            articles,
            NearestNeighbourhoodRouting,
            _routing_kwargs(muter_domain),
        )

    def fail(*args, **kwargs):
        raise AssertionError("candidate scoring allocated a Route")

    monkeypatch.setattr(routing_module, "Route", fail)
    solution = algorithm.solve(muter_resolved_orders)
    assert solution.batches
    assert algorithm._route_cache
    assert all(isinstance(value, (int, float)) for value in algorithm._route_cache.values())
