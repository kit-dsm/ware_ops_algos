import pytest

from ware_ops_algos.algorithms import RatliffRosenthalRouting, GreedyItemAssignment
from ware_ops_algos.domain_models import (
    Resource, Location, StorageLocations, StorageType,
    OrderPosition, Order, OrdersDomain, OrderType,
)

# Synthetic instance from the original paper
LOCATIONS = [
    Location(x=1, y=3,  article_id=1,  amount=1),
    Location(x=1, y=8,  article_id=2,  amount=1),
    Location(x=1, y=11, article_id=3,  amount=1),
    Location(x=2, y=5,  article_id=4,  amount=1),
    Location(x=2, y=11, article_id=5,  amount=1),
    Location(x=3, y=3,  article_id=6,  amount=1),
    Location(x=3, y=9,  article_id=7,  amount=1),
    Location(x=3, y=12, article_id=8,  amount=1),
    Location(x=5, y=7,  article_id=9,  amount=1),
    Location(x=6, y=2,  article_id=10, amount=1),
    Location(x=6, y=9,  article_id=11, amount=1),
    Location(x=6, y=10, article_id=12, amount=1),
]

ORDER_POSITIONS = [
    OrderPosition(order_number=1, article_id=i, amount=1)
    for i in range(1, 13)
]

RR_KWARGS = dict(
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


@pytest.fixture
def pick_list():
    storage = StorageLocations(StorageType.DEDICATED, locations=LOCATIONS)
    storage.build_article_location_mapping()
    order = Order(order_id=1, order_positions=ORDER_POSITIONS)
    orders = OrdersDomain(OrderType.STANDARD, orders=[order])
    ia_sol = GreedyItemAssignment(storage).solve(orders.orders)
    pl = [pos for order in ia_sol.resolved_orders for pos in order.pick_positions]
    return pl


def test_rr_returns_solution(pick_list):
    router = RatliffRosenthalRouting(**RR_KWARGS)
    sol = router.solve(pick_list)
    assert sol is not None
    assert sol.route.distance > 0


def test_rr_visits_all_picks(pick_list):
    router = RatliffRosenthalRouting(**RR_KWARGS)
    sol = router.solve(pick_list)
    assert len(sol.route.route) >= len(pick_list)