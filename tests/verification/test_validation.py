from ware_ops_algos.algorithms import (
    BatchObject, PickCart, Route, RoutingOrigin, WarehouseOrder,
)
from ware_ops_algos.planning.commands import (
    CancelTour, CommitTour, CreatePickList, ReplaceSuffix, ViolationCode,
)
from ware_ops_algos.planning.lifecycle import Accepts, TourStates
from ware_ops_algos.planning.models import PlanView, TourView
from ware_ops_algos.planning.validation import check


def _order(order_id):
    return WarehouseOrder(order_id=order_id, order_date=0)


def test_validation_uses_visible_tours_and_revision_level():
    tour = TourView(1, 0, TourStates.SCHEDULED, Accepts.NONE,
                    batch=BatchObject(1, [_order(1)]))
    plan = PlanView(0, tours=(tour,))
    assert check((CancelTour(99),), plan)[0].code is ViolationCode.STATE_INFORMATION
    assert check((CancelTour(1),), plan)[0].code is ViolationCode.ACCEPTS
    assert check((CommitTour(tour_id=1),), plan)[0].code is ViolationCode.PHYSICAL


def test_started_replacement_checks_membership_origin_and_boxes():
    old = BatchObject(1, [_order(1)])
    origin = RoutingOrigin((0, 0))
    tour = TourView(
        1, 0, TourStates.STARTED, Accepts.INSERT, batch=old,
        routing_origin=origin, occupied_bins=1,
        pick_cart=PickCart(n_boxes=1, box_can_mix_orders=False),
    )
    plan = PlanView(0, order_buffer=(_order(2),), tours=(tour,),
                    available_from={0: 0})
    route = Route(0, batch=BatchObject(2, [_order(1), _order(2)]),
                  routing_origin=RoutingOrigin((1, 1)))
    codes = {v.code for v in check((ReplaceSuffix(1, route, (2,)),), plan)}
    assert codes == {ViolationCode.EXECUTED_PREFIX, ViolationCode.PHYSICAL}
    route.routing_origin = origin
    assert check((ReplaceSuffix(1, route, (3,)),), plan)[0].code is ViolationCode.MEMBERSHIP


def test_pick_list_uses_exposed_orders():
    plan = PlanView(0, order_buffer=(_order(1),))
    assert check((CreatePickList(BatchObject(1, [_order(1)])),), plan) == ()
    assert check((CreatePickList(BatchObject(2, [_order(2)])),), plan)[0].code is (
        ViolationCode.STATE_INFORMATION)
