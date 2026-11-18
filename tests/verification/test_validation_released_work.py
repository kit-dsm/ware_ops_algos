"""F62: orders of a tour cancelled by the same proposal are released work for every command.

The 61s-80-30-0 case at t = 2255.438 reduced to its essentials. Two planned
drafts that the decision may rebuild: tour 1 holds order 10, tour 2 holds order
11; the buffer holds order 12. The candidate merges the drafts: it cancels tour 2
and replaces the route of tour 1 by one over orders 10 and 11, naming 11 as the
addition. Order 11 is not in the buffer, but the same proposal releases it by
cancelling tour 2, exactly as a CreateTour over orders 10 and 11 would be
accepted. Adding an order of a tour that the proposal does not cancel stays a
state-information violation, as does adding an order nobody exposed.
"""

from ware_ops_algos.algorithms import BatchObject, Route, WarehouseOrder
from ware_ops_algos.planning.commands import CancelTour, ReplaceSuffix, ViolationCode
from ware_ops_algos.planning.lifecycle import Accepts, TourStates
from ware_ops_algos.planning.models import PlanView, TourView
from ware_ops_algos.planning.validation import check


def _order(order_id: int) -> WarehouseOrder:
    return WarehouseOrder(order_id=order_id, order_date=0)


def _plan() -> PlanView:
    first = TourView(1, 0, TourStates.PLANNED, Accepts.REBUILD, batch=BatchObject(1, [_order(10)]))
    second = TourView(2, 0, TourStates.PLANNED, Accepts.REBUILD, batch=BatchObject(2, [_order(11)]))
    return PlanView(2255.438, order_buffer=(_order(12),), tours=(first, second),
                    available_from={0: 2255.438})


def _merge(*order_ids: int) -> ReplaceSuffix:
    route = Route(0, batch=BatchObject(1, [_order(order_id) for order_id in (10, *order_ids)]))
    return ReplaceSuffix(1, route, tuple(order_ids))


def test_merging_two_drafts_adds_the_released_orders():
    assert check((CancelTour(2), _merge(11)), _plan()) == ()


def test_buffered_orders_remain_admissible_additions():
    assert check((_merge(12),), _plan()) == ()


def test_orders_of_a_tour_that_stays_are_not_released():
    violations = check((_merge(11),), _plan())
    assert [violation.code for violation in violations] == [ViolationCode.STATE_INFORMATION]


def test_orders_nobody_exposed_are_not_released():
    violations = check((CancelTour(2), _merge(13)), _plan())
    assert [violation.code for violation in violations] == [ViolationCode.STATE_INFORMATION]
