import pandas as pd
import pytest

from ware_ops_algos.algorithms import (
    BatchObject,
    Batching,
    BatchingSolution,
    ClarkAndWrightBatching,
    LargestGapRouting,
    LocalSearchBatching,
    MidpointRouting,
    NearestNeighbourhoodRouting,
    OrderNrFifoBatching,
    PickListRouting,
    PickPosition,
    RatliffRosenthalRouting,
    ReturnRouting,
    SShapeRouting,
    UShapeRouting,
    WarehouseOrder,
)
from ware_ops_algos.algorithms.batching.moves import ShiftNeighborhood, SwapNeighborhood
from ware_ops_algos.domain_models import (
    Article,
    Articles,
    ArticleType,
    DimensionType,
    PickCart,
    Resource,
)


def _rr_kwargs(**flags):
    kwargs = dict(
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
    kwargs.update(flags)
    return kwargs


def _rr_picks():
    return [
        PickPosition(index, index, 1, node, 1)
        for index, node in enumerate(
            [(1, 3), (1, 8), (2, 5), (3, 3), (5, 7), (6, 2), (6, 9)],
            1,
        )
    ]


def test_rr_score_solve_stateless_and_route_reconstruction():
    picks = _rr_picks()
    router = RatliffRosenthalRouting(**_rr_kwargs())

    score = router.score(picks)
    first = router.solve(picks)
    second = router.solve(picks)

    assert score == pytest.approx(82.0)
    assert first.route.distance == pytest.approx(score)
    assert second.route.distance == pytest.approx(score)

    output = RatliffRosenthalRouting(
        **_rr_kwargs(gen_tour=True, gen_item_sequence=True)
    ).solve(picks).route
    assert output.route
    assert output.annotated_route
    assert len(output.item_sequence) == len(picks)


def _heuristic_kwargs():
    nodes = [
        (0, 0), (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2), (3, 0),
    ]
    distance = pd.DataFrame(
        [
            [abs(x - u) + abs(y - v) for u, v in nodes]
            for x, y in nodes
        ],
        index=nodes,
        columns=nodes,
    )
    predecessor = [[index for _ in nodes] for index in range(len(nodes))]
    return dict(
        start_node=(0, 0),
        end_node=(3, 0),
        closest_node_to_start=(0, 0),
        min_aisle_position=0,
        max_aisle_position=2,
        picker=[Resource(id=1)],
        distance_matrix=distance,
        predecessor_matrix=predecessor,
        node_list=nodes,
        node_to_idx={node: index for index, node in enumerate(nodes)},
        idx_to_node={index: node for index, node in enumerate(nodes)},
    )


@pytest.mark.parametrize(
    "routing_class",
    [
        SShapeRouting,
        ReturnRouting,
        MidpointRouting,
        LargestGapRouting,
        NearestNeighbourhoodRouting,
        UShapeRouting,
        PickListRouting,
    ],
)
def test_heuristic_score_matches_solve_on_repeated_calls(routing_class):
    picks = [
        PickPosition(1, 1, 1, (1, 1), 1),
        PickPosition(2, 2, 1, (2, 2), 1),
    ]
    router = routing_class(**_heuristic_kwargs())

    score = router.score(picks)
    assert router.solve(picks).route.distance == pytest.approx(score)
    assert router.solve([]).route.distance >= 0
    assert router.score(picks) == pytest.approx(score)


class MembershipScoreRouting:
    algo_name = "MembershipScore"

    def __init__(self, scores, default_score=100.0, **_kwargs):
        self.scores = scores
        self.default_score = default_score
        self.calls = 0

    def score(self, picks):
        self.calls += 1
        membership = frozenset(pick.order_number for pick in picks)
        return self.scores.get(membership, self.default_score)


class FixedStartBatching(Batching):
    algo_name = "FixedStart"

    def __init__(self, pick_cart, articles, groups):
        super().__init__(pick_cart, articles)
        self.groups = groups

    def _run(self, input_data):
        orders = {order.order_id: order for order in input_data}
        return BatchingSolution(
            batches=[
                BatchObject(
                    batch_id=index,
                    orders=[orders[order_id] for order_id in group],
                )
                for index, group in enumerate(self.groups)
            ]
        )


def _order(order_id, quantity=1):
    return WarehouseOrder(
        order_id=order_id,
        pick_positions=(
            PickPosition(order_id, 1, quantity, (order_id, 0), quantity),
        ),
    )


def _cart(capacity=2):
    return PickCart(
        n_dimension=1,
        capacities=[capacity],
        dimensions=[DimensionType.ITEMS],
        n_boxes=1,
        box_can_mix_orders=True,
    )


def _articles():
    return Articles(ArticleType.STANDARD, [Article(article_id=1, weight=1.0)])


def test_local_search_incremental_state_preserves_objective_and_cache():
    orders = [_order(index) for index in range(1, 5)]
    scores = {
        frozenset((1, 2)): 10.0,
        frozenset((3, 4)): 10.0,
        frozenset((2, 3)): 4.0,
        frozenset((1, 4)): 4.0,
    }
    algorithm = LocalSearchBatching(
        _cart(),
        _articles(),
        MembershipScoreRouting,
        {"scores": scores},
        FixedStartBatching,
        start_batching_kwargs={"groups": [(1, 2), (3, 4)]},
        neighborhood_classes=[SwapNeighborhood],
        time_limit=5.0,
    )

    solution = algorithm.solve(orders)
    recomputed = sum(
        algorithm._router.score(batch.pick_positions)
        for batch in solution.batches
    )
    assert recomputed == pytest.approx(algorithm.search_statistics["final_objective"])
    assert algorithm.search_statistics["accepted_swaps"] == 1
    assert algorithm.search_statistics["full_objective_resums"] == 0
    assert algorithm.search_statistics["candidate_batch_objects"] == 0


def test_clark_wright_uses_cached_scalar_candidate_scores():
    orders = [_order(index) for index in range(1, 4)]
    scores = {frozenset((index,)): 10.0 for index in range(1, 4)}
    scores.update({frozenset(pair): 10.0 for pair in [(1, 2), (1, 3), (2, 3)]})
    algorithm = ClarkAndWrightBatching(
        _cart(),
        _articles(),
        MembershipScoreRouting,
        {"scores": scores},
    )

    solution = algorithm.solve(orders)
    assert solution.batches
    assert algorithm._route_cache
    assert all(isinstance(value, (int, float)) for value in algorithm._route_cache.values())
    assert algorithm._batch_score(solution.batches[0].orders) == pytest.approx(
        scores[frozenset(solution.batches[0].order_numbers)]
    )
