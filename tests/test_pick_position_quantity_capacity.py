import pytest

from ware_ops_algos.algorithms import GreedyItemAssignment, PickPosition
from ware_ops_algos.algorithms.batching.batching_utils import CapacityChecker
from ware_ops_algos.domain_models import (
    Article,
    Articles,
    ArticleType,
    DimensionType,
    Location,
    Order,
    OrderPosition,
    PickCart,
    StorageLocations,
    StorageType,
)


@pytest.fixture
def articles():
    return Articles(
        ArticleType.STANDARD,
        [
            Article(article_id=101, weight=2.5, volume=1.2),
            Article(article_id=202, weight=7.0, volume=0.5),
        ],
    )


@pytest.fixture
def resolved_orders():
    storage = StorageLocations(
        StorageType.DEDICATED,
        [
            Location(x=1, y=1, article_id=101, amount=6),
            Location(x=2, y=1, article_id=101, amount=4),
            Location(x=3, y=1, article_id=202, amount=4),
        ],
    )
    storage.build_article_location_mapping()
    orders = [
        Order(
            order_id=1,
            order_positions=[
                OrderPosition(1, 101, 10),
                OrderPosition(1, 202, 3),
            ],
        ),
        Order(
            order_id=2,
            order_positions=[OrderPosition(2, 202, 1)],
        ),
    ]
    return GreedyItemAssignment(storage).solve(orders).resolved_orders


def _checker(articles, dimensions, capacities, n_boxes=1):
    return CapacityChecker(
        PickCart(
            n_dimension=len(dimensions),
            capacities=capacities,
            dimensions=dimensions,
            n_boxes=n_boxes,
            box_can_mix_orders=True,
        ),
        articles,
    )


def test_pick_position_default_and_quantity_invariant():
    pick = PickPosition(1, 101, 4, (1, 1), 4)
    assert pick.article_name is None
    assert pick.picked_quantity == 4
    with pytest.raises(ValueError, match="quantity picked"):
        PickPosition(1, 101, 10, (1, 1), 4)


def test_item_assignment_uses_resolved_quantity_at_each_location(resolved_orders):
    split_picks = [
        pick for pick in resolved_orders[0].pick_positions
        if pick.article_id == 101
    ]
    single_location_pick = next(
        pick for pick in resolved_orders[0].pick_positions
        if pick.article_id == 202
    )

    assert [pick.picked_quantity for pick in split_picks] == [6, 4]
    assert [pick.amount for pick in split_picks] == [6, 4]
    assert sum(pick.picked_quantity for pick in split_picks) == 10
    assert single_location_pick.picked_quantity == 3
    assert single_location_pick.amount == 3


@pytest.mark.parametrize(
    ("dimension", "expected"),
    [
        (DimensionType.ITEMS, 13),
        (DimensionType.WEIGHT, 46.0),
        (DimensionType.VOLUME, 13.5),
        (DimensionType.ORDERLINES, 2),
        (DimensionType.ORDERS, 1),
    ],
)
def test_capacity_dimensions_use_resolved_pick_quantity(
    articles, resolved_orders, dimension, expected,
):
    order = resolved_orders[0]
    checker = _checker(articles, [dimension], [expected])

    assert checker._compute_order_consumption(order) == pytest.approx([expected])
    assert checker.orders_fit([order])
    checker.pick_cart.capacities[0] = expected - 0.1
    assert not checker.orders_fit([order])


def test_split_line_weight_and_orderline_are_not_double_counted(
    articles, resolved_orders,
):
    order = resolved_orders[0]
    split_only = type(order)(
        order_id=order.order_id,
        pick_positions=tuple(
            pick for pick in order.pick_positions if pick.article_id == 101
        ),
    )
    checker = _checker(
        articles,
        [DimensionType.ITEMS, DimensionType.WEIGHT, DimensionType.ORDERLINES],
        [10, 25.0, 1],
    )

    assert checker._compute_order_consumption(split_only) == pytest.approx(
        [10, 25.0, 1]
    )
    assert checker.orders_fit([split_only])


def test_mixed_capacity_dimensions_and_order_count(articles, resolved_orders):
    dimensions = [
        DimensionType.ITEMS,
        DimensionType.WEIGHT,
        DimensionType.VOLUME,
        DimensionType.ORDERLINES,
        DimensionType.ORDERS,
    ]
    checker = _checker(articles, dimensions, [14, 53, 14, 3, 2])

    assert checker._compute_consumption(resolved_orders) == pytest.approx(
        [14, 53.0, 14.0, 3, 2]
    )
    assert checker.orders_fit(resolved_orders)
    checker.pick_cart.capacities[-1] = 1
    assert not checker.orders_fit(resolved_orders)
