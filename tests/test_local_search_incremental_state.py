import pytest

from ware_ops_algos.algorithms.algorithm_interfaces import (
    BatchObject,
    BatchingSolution,
    PickPosition,
    WarehouseOrder,
)
from ware_ops_algos.algorithms.batching.batching import Batching
from ware_ops_algos.algorithms.batching.local_search_batching import LocalSearchBatching
from ware_ops_algos.algorithms.batching.moves import ShiftNeighborhood, SwapNeighborhood
from ware_ops_algos.algorithms.batching.savings_batching import ClarkAndWrightBatching
from ware_ops_algos.domain_models import (
    Article,
    Articles,
    ArticleType,
    DimensionType,
    PickCart,
)


class FixedStartBatching(Batching):
    algo_name = "FixedStart"

    def __init__(self, pick_cart, articles, groups):
        super().__init__(pick_cart, articles)
        self.groups = groups

    def _run(self, input_data):
        orders = {order.order_id: order for order in input_data}
        return BatchingSolution(batches=[
            BatchObject(
                batch_id=index,
                orders=[orders[order_id] for order_id in group],
            )
            for index, group in enumerate(self.groups)
        ])


class MembershipScoreRouting:
    algo_name = "MembershipScore"

    def __init__(self, scores, default_score=100.0, **_kwargs):
        self.scores = scores
        self.default_score = default_score

    def score(self, picks):
        membership = frozenset(pick.order_number for pick in picks)
        return self.scores.get(membership, self.default_score)


def _order(order_id, article_id=1, quantity=1):
    return WarehouseOrder(
        order_id=order_id,
        pick_positions=(PickPosition(
            order_number=order_id,
            article_id=article_id,
            amount=quantity,
            pick_node=(order_id, 0),
            in_store=quantity,
        ),),
    )


def _algorithm(
    orders,
    groups,
    scores,
    neighborhood,
    *,
    capacities=(2,),
    dimensions=(DimensionType.ITEMS,),
    articles=None,
    n_boxes=1,
    box_can_mix_orders=True,
):
    if articles is None:
        articles = Articles(
            ArticleType.STANDARD,
            [Article(article_id=1, weight=1.0)],
        )
    cart = PickCart(
        n_dimension=len(dimensions),
        capacities=list(capacities),
        dimensions=list(dimensions),
        n_boxes=n_boxes,
        box_can_mix_orders=box_can_mix_orders,
    )
    return LocalSearchBatching(
        pick_cart=cart,
        articles=articles,
        routing_class=MembershipScoreRouting,
        routing_class_kwargs={"scores": scores},
        start_batching_class=FixedStartBatching,
        start_batching_kwargs={"groups": groups},
        neighborhood_classes=[neighborhood],
        time_limit=30.0,
    )


def _memberships(solution):
    return tuple(
        tuple(order.order_id for order in batch.orders)
        for batch in solution.batches
    )


def _assert_incremental_invariants(algorithm, solution):
    recomputed = sum(
        algorithm._router.score([
            position
            for order in batch.orders
            for position in order.pick_positions
        ])
        for batch in solution.batches
    )
    assert algorithm.search_statistics["final_objective"] == pytest.approx(
        recomputed
    )
    assert all(
        algorithm.capacity_checker.orders_fit(batch.orders)
        for batch in solution.batches
    )
    assert algorithm.search_statistics["full_capacity_rescans"] == 0
    assert algorithm.search_statistics["full_objective_resums"] == 0
    assert algorithm.search_statistics["candidate_batch_objects"] == 0


def test_feasible_swap_updates_scores_and_total():
    orders = [_order(i) for i in range(1, 5)]
    scores = {
        frozenset((1, 2)): 10.0,
        frozenset((3, 4)): 10.0,
        frozenset((2, 3)): 4.0,
        frozenset((1, 4)): 4.0,
    }
    algorithm = _algorithm(
        orders,
        [(1, 2), (3, 4)],
        scores,
        SwapNeighborhood,
    )
    solution = algorithm.solve(orders)

    assert _memberships(solution) == ((3, 2), (1, 4))
    assert algorithm.search_statistics["accepted_swaps"] == 1
    assert [value for _, value in solution.objective_trajectory] == [20.0, 8.0]
    _assert_incremental_invariants(algorithm, solution)


def test_feasible_shift_removes_empty_source_batch():
    orders = [_order(1), _order(2)]
    algorithm = _algorithm(
        orders,
        [(1,), (2,)],
        {
            frozenset((1,)): 10.0,
            frozenset((2,)): 10.0,
            frozenset((1, 2)): 5.0,
        },
        ShiftNeighborhood,
    )
    solution = algorithm.solve(orders)

    assert _memberships(solution) == ((2, 1),)
    assert algorithm.search_statistics["accepted_shifts"] == 1
    assert [value for _, value in solution.objective_trajectory] == [20.0, 5.0]
    _assert_incremental_invariants(algorithm, solution)


def test_infeasible_destination_is_rejected_before_routing():
    orders = [_order(1, quantity=2), _order(2, quantity=2)]
    algorithm = _algorithm(
        orders,
        [(1,), (2,)],
        {frozenset((1,)): 5.0, frozenset((2,)): 5.0},
        ShiftNeighborhood,
        capacities=(2,),
    )
    solution = algorithm.solve(orders)

    assert _memberships(solution) == ((1,), (2,))
    assert algorithm.search_statistics["accepted_shifts"] == 0
    assert algorithm.search_statistics["routing_score_requests"] == 2
    assert algorithm.search_statistics["capacity_checks"] == 4


def test_no_improving_swap_reuses_candidate_membership_scores():
    orders = [_order(i) for i in range(1, 5)]
    scores = {
        frozenset((1, 2)): 10.0,
        frozenset((3, 4)): 10.0,
        frozenset((2, 3)): 10.0,
        frozenset((1, 4)): 10.0,
        frozenset((2, 4)): 10.0,
        frozenset((1, 3)): 10.0,
    }
    algorithm = _algorithm(
        orders,
        [(1, 2), (3, 4)],
        scores,
        SwapNeighborhood,
    )
    solution = algorithm.solve(orders)

    assert _memberships(solution) == ((1, 2), (3, 4))
    assert algorithm.search_statistics["accepted_swaps"] == 0
    assert algorithm.search_statistics["routing_cache_hits"] >= 4
    assert algorithm.search_statistics["routing_cache_misses"] == 6


def test_multidimensional_capacity_rejects_weight_infeasible_shift():
    orders = [_order(1, article_id=1), _order(2, article_id=2)]
    articles = Articles(
        ArticleType.STANDARD,
        [
            Article(article_id=1, weight=8.0),
            Article(article_id=2, weight=4.0),
        ],
    )
    algorithm = _algorithm(
        orders,
        [(1,), (2,)],
        {frozenset((1,)): 10.0, frozenset((2,)): 10.0},
        ShiftNeighborhood,
        capacities=(2, 10.0),
        dimensions=(DimensionType.ITEMS, DimensionType.WEIGHT),
        articles=articles,
    )
    solution = algorithm.solve(orders)

    assert _memberships(solution) == ((1,), (2,))
    assert algorithm.search_statistics["accepted_shifts"] == 0
    assert algorithm.search_statistics["routing_score_requests"] == 2


def test_savings_ties_use_batch_ids_deterministically():
    orders = [_order(i) for i in range(1, 4)]
    articles = Articles(
        ArticleType.STANDARD,
        [Article(article_id=1, weight=1.0)],
    )
    algorithm = ClarkAndWrightBatching(
        pick_cart=PickCart(
            n_dimension=1,
            capacities=[2],
            dimensions=[DimensionType.ITEMS],
            n_boxes=1,
            box_can_mix_orders=True,
        ),
        articles=articles,
        routing_class=MembershipScoreRouting,
        routing_class_kwargs={
            "scores": {
                frozenset((1,)): 10.0,
                frozenset((2,)): 10.0,
                frozenset((3,)): 10.0,
                frozenset((1, 2)): 10.0,
                frozenset((1, 3)): 10.0,
                frozenset((2, 3)): 10.0,
            },
        },
    )

    solution = algorithm.solve(orders)
    assert _memberships(solution) == ((3,), (1, 2))


def test_non_mixing_box_count_is_updated_incrementally():
    orders = [_order(1, quantity=4), _order(2, quantity=4)]
    algorithm = _algorithm(
        orders,
        [(1,), (2,)],
        {frozenset((1,)): 10.0, frozenset((2,)): 10.0},
        ShiftNeighborhood,
        capacities=(3,),
        n_boxes=2,
        box_can_mix_orders=False,
    )

    solution = algorithm.solve(orders)
    assert _memberships(solution) == ((1,), (2,))
    assert algorithm.search_statistics["accepted_shifts"] == 0
