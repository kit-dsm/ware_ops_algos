from copy import deepcopy

import pytest

from ware_ops_algos.algorithms import (
    BatchObject,
    GreedyItemAssignment,
    PickPosition,
    ResidualBatchingInput,
    ResidualFifoBatching,
    WarehouseOrder,
)
from ware_ops_algos.domain_models import (
    DimensionType,
    Location,
    Order,
    OrderPosition,
    PickCart,
    StorageLocations,
    StorageType,
)


def _cart(n_boxes=3, *, dimension=DimensionType.ORDERS, mixing=False):
    return PickCart(
        n_dimension=1,
        capacities=[1],
        dimensions=[dimension],
        n_boxes=n_boxes,
        box_can_mix_orders=mixing,
    )


def _resolved(order_id, release=0):
    return WarehouseOrder(
        order_id=order_id,
        order_date=release,
        pick_positions=(
            PickPosition(order_id, order_id, 1, (1, order_id), 1),
        ),
    )


@pytest.mark.parametrize(
    ("n_boxes", "owners", "expected"),
    [
        (2, ((1,), (2,)), ()),
        (3, ((1,), (2,), ()), (3,)),
        (4, ((1,), (2,), (), ()), (3, 4)),
    ],
)
def test_residual_fifo_fills_only_genuinely_empty_bins(
    n_boxes,
    owners,
    expected,
):
    active = BatchObject(1, [_resolved(1), _resolved(2)])
    result = ResidualFifoBatching().solve(
        ResidualBatchingInput(
            active_batch=active,
            candidate_orders=(_resolved(4, 2), _resolved(3, 1)),
            bin_order_ids=owners,
            locked_bin_ids=frozenset(),
            pick_cart=_cart(n_boxes),
        )
    )
    inserted = tuple(
        sorted(
            result.batches[0].order_numbers
            - active.order_numbers
        )
    )
    assert inserted == expected
    assert tuple(
        order.order_id for order in result.batches[0].orders
    ) == (1, 2, *expected)


def test_residual_fifo_preserves_owned_and_locked_bins():
    result = ResidualFifoBatching().solve(
        ResidualBatchingInput(
            active_batch=BatchObject(1, [_resolved(1), _resolved(2)]),
            candidate_orders=(_resolved(3), _resolved(4)),
            bin_order_ids=((1,), (2,), (), ()),
            locked_bin_ids=frozenset({2}),
            pick_cart=_cart(4),
        )
    )
    assert result.batches[0].bin_assignments == {
        0: (1,),
        1: (2,),
        2: (),
        3: (3,),
    }


@pytest.mark.parametrize(
    "cart",
    [
        _cart(mixing=True),
        _cart(dimension=DimensionType.ITEMS),
    ],
)
def test_residual_fifo_rejects_unsupported_cart_semantics(cart):
    with pytest.raises(RuntimeError, match="Residual FIFO insertion"):
        ResidualFifoBatching().solve(
            ResidualBatchingInput(
                active_batch=BatchObject(1, [_resolved(1)]),
                candidate_orders=(_resolved(2),),
                bin_order_ids=((1,), (), ()),
                locked_bin_ids=frozenset(),
                pick_cart=cart,
            )
        )


def test_residual_fifo_is_repeat_call_safe_and_does_not_mutate_input():
    input_data = ResidualBatchingInput(
        active_batch=BatchObject(1, [_resolved(1), _resolved(2)]),
        candidate_orders=(_resolved(3),),
        bin_order_ids=((1,), (2,), ()),
        locked_bin_ids=frozenset(),
        pick_cart=_cart(),
    )
    before = deepcopy(input_data)
    algorithm = ResidualFifoBatching()
    first = algorithm.solve(input_data)
    second = algorithm.solve(input_data)
    assert first.batches == second.batches
    assert input_data == before


def test_greedy_assignment_splits_scattered_stock_and_rolls_back_shortage():
    storage = StorageLocations(
        StorageType.DEDICATED,
        [
            Location(x=1, y=2, article_id=101, amount=3),
            Location(x=3, y=10, article_id=101, amount=2),
        ],
    )
    storage.build_article_location_mapping()
    orders = [
        Order(
            order_id=1,
            order_date=0,
            order_positions=[OrderPosition(1, 101, 4)],
        ),
        Order(
            order_id=2,
            order_date=1,
            order_positions=[OrderPosition(2, 101, 2)],
        ),
    ]
    storage_before = deepcopy(storage)
    orders_before = deepcopy(orders)
    algorithm = GreedyItemAssignment(storage)
    first = algorithm.solve(orders)
    second = algorithm.solve(orders)

    assert [pick.in_store for pick in first.resolved_orders[0].pick_positions] == [
        3,
        1,
    ]
    assert [order.order_id for order in first.unassigned_orders] == [2]
    assert first.shortages == [
        {
            "order_id": 2,
            "article_id": 101,
            "missing": 1,
        }
    ]
    assert first.resolved_orders == second.resolved_orders
    assert first.unassigned_orders == second.unassigned_orders
    assert storage == storage_before
    assert orders == orders_before
