from ware_ops_algos.algorithms import GreedyItemAssignment
from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards
from ware_ops_algos.domain_models import (
    Location,
    Order,
    OrderPosition,
    OrdersDomain,
    OrderType,
    StorageLocations,
    StorageType,
)


def test_packaged_algorithm_cards_are_loadable_and_unique():
    cards = load_packaged_algo_cards()

    assert cards
    assert len({card.algo_name for card in cards}) == len(cards)
    assert {card.problem_type for card in cards} >= {
        "batching",
        "item_assignment",
        "routing",
        "scheduling",
    }
    assert all(card.implementation.get("class_name") for card in cards)


def test_greedy_assignment_resolves_a_physical_pick_node():
    storage = StorageLocations(
        StorageType.DEDICATED,
        locations=[Location(x=2, y=4, article_id=7, amount=2)],
    )
    storage.build_article_location_mapping()
    orders = OrdersDomain(
        OrderType.STANDARD,
        orders=[
            Order(
                order_id=1,
                order_positions=[OrderPosition(order_number=1, article_id=7, amount=1)],
            )
        ],
    )

    solution = GreedyItemAssignment(storage).solve(orders.orders)

    assert solution.resolved_orders[0].pick_positions[0].pick_node == (2, 4)
