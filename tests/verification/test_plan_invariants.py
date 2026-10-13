from dataclasses import replace

from ware_ops_algos.algorithms import (
    BatchObject, PickPosition, Route, RouteNode, NodeType, WarehouseOrder,
)
from ware_ops_algos.domain_models import PickCart
from ware_ops_algos.planning.invariants import check_plan, check_progress
from ware_ops_algos.planning.lifecycle import Accepts, TourStates
from ware_ops_algos.planning.models import PlanView, TourView


def _order(order_id=1, date=0):
    return WarehouseOrder(
        order_id=order_id, order_date=date,
        pick_positions=(PickPosition(order_id, 1, 1, (1, 1), 1),),
    )


def _tour(status=TourStates.STARTED):
    batch = BatchObject(1, [_order()])
    route = Route(2, batch=batch, annotated_route=[
        RouteNode((0, 0), NodeType.ROUTE),
        RouteNode((1, 1), NodeType.PICK),
        RouteNode((0, 0), NodeType.ROUTE),
    ])
    return TourView(
        tour_id=1, picker_id=0, status=status, accepts=Accepts.INSERT,
        batch=batch, route=route, start_time=0, end_time=10,
        pick_cart=PickCart(n_boxes=1), occupied_bins=1,
        pinned_prefix=(route.annotated_route[0],),
    )


def test_plan_detects_duplicate_order_and_lost_pick():
    tour = _tour()
    assert check_plan(PlanView(0, tours=(tour,))) == ()

    duplicate = PlanView(0, order_buffer=(_order(),), tours=(tour,))
    assert any(v.code == "membership" for v in check_plan(duplicate))

    short_route = replace(tour, route=replace(tour.route,
        annotated_route=[RouteNode((0, 0), NodeType.ROUTE)]))
    assert any(v.code == "route" for v in check_plan(PlanView(0, tours=(short_route,))))


def test_progress_detects_lost_order_and_backward_time():
    before = PlanView(10, order_buffer=(_order(),))
    after = PlanView(9)
    codes = {violation.code for violation in check_progress(before, after)}
    assert codes == {"membership", "time"}


def test_progress_keeps_order_when_tour_completes():
    before = PlanView(0, tours=(_tour(TourStates.STARTED),))
    after = PlanView(10, tours=(_tour(TourStates.DONE),))
    assert check_progress(before, after) == ()
