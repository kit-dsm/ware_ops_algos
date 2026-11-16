from __future__ import annotations

import pytest

from ware_ops_algos.algorithms import RecordedItemAssignment
from ware_ops_algos.domain_models import (
    Order, OrderPosition, RecordedLine, RecordedPickingPlan, RecordedTour,
    StorageLocations, StorageType,
)


def test_repeated_article_keeps_each_recorded_pick_node():
    order = Order(order_id=7, order_positions=[
        OrderPosition(order_number=7, article_id=42, amount=2),
        OrderPosition(order_number=7, article_id=42, amount=3),
    ])
    plan = RecordedPickingPlan(tours=(RecordedTour(
        tour_id=0, picker_id=0, lines=(
            RecordedLine(7, 42, (0.0, 0.0)),
            RecordedLine(7, 42, (8.0, 0.0)),
        )),))
    storage = StorageLocations(tpe=StorageType.DEDICATED, locations=[])
    solution = RecordedItemAssignment(storage, plan).solve([order])
    positions = solution.resolved_orders[0].pick_positions
    assert [position.pick_node for position in positions] == [(0.0, 0.0), (8.0, 0.0)]
    assert [position.amount for position in positions] == [2, 3]


def test_missing_recorded_line_is_rejected():
    order = Order(order_id=7, order_positions=[
        OrderPosition(order_number=7, article_id=42, amount=1),
        OrderPosition(order_number=7, article_id=42, amount=1),
    ])
    plan = RecordedPickingPlan(tours=(RecordedTour(
        tour_id=0, picker_id=0, lines=(RecordedLine(7, 42, (0.0, 0.0)),)),))
    storage = StorageLocations(tpe=StorageType.DEDICATED, locations=[])
    with pytest.raises(RuntimeError, match="no pick node"):
        RecordedItemAssignment(storage, plan).solve([order])
