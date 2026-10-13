"""The analytic model uses the picker's actual cart capacity as q."""

from types import SimpleNamespace

import pytest

from ware_ops_algos.algorithms import (
    AnalyticStochasticWaiting, BatchObject, Job, PickPosition, Route,
    ScheduledJob, WaitingInput, WarehouseOrder,
)
from ware_ops_algos.algorithms.waiting import policies
from ware_ops_algos.domain_models import DimensionType, PickCart, Resource


def _input(n_boxes: int, n_orders: int) -> WaitingInput:
    orders = [WarehouseOrder(
        order_id=index, order_date=0,
        pick_positions=(PickPosition(index, index, 1, (index, 1), 1),),
    ) for index in range(n_orders)]
    batch = BatchObject(1, orders)
    route = Route(distance=4, batch=batch)
    job = Job(1, 4, 0, 100, n_orders, route=route, batch=batch)
    cart = PickCart(n_boxes=n_boxes, capacities=[1],
                    dimensions=[DimensionType.ORDERS], box_can_mix_orders=False)
    return WaitingInput(
        candidates=(ScheduledJob(job, picker_id=0, start_time=0, end_time=4),),
        current_time=0, input_closed=False,
        picker=Resource(id=0, speed=1, time_per_pick=1, pick_cart=cart),
        layout=None,
        information=SimpleNamespace(require=lambda *_: SimpleNamespace(
            mean_interarrival_time_s=10)),
    )


def test_q_comes_from_picker_cart(monkeypatch):
    seen = []

    def optimum(data, mean_interarrival_time, remaining_arrivals_after_miss):
        seen.append(remaining_arrivals_after_miss)
        return SimpleNamespace(wait_duration=0.0)

    monkeypatch.setattr(policies, "solve_optimal_wait", optimum)
    result = AnalyticStochasticWaiting().solve(_input(3, 2))
    assert result.action == "release"
    assert seen == [2]


@pytest.mark.parametrize("n_orders", [1, 2])
def test_incomplete_q_minus_one_batch_raises(n_orders):
    with pytest.raises(ValueError, match="exactly q-1 known orders"):
        AnalyticStochasticWaiting()._run(_input(4, n_orders))
