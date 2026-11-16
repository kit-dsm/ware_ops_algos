"""First-fit and next-fit FCFS batching on a hand-made order sequence.

One box of 10 items that holds several orders; orders arrive in the order 0, 1, 2, 3 with 6, 5, 4 and 5 items.
Next-fit (FifoBatching) fills only the last batch: {0} (6; 5 more do not fit),
{1, 2} (9; 5 more do not fit), {3}. First-fit (FirstFitFifoBatching) puts each
order into the first batch with room: 0 opens batch A (6), 1 opens B (5),
2 joins A (10), 3 joins B (10): {0, 2}, {1, 3}, two batches instead of three.
"""

from ware_ops_algos.algorithms import (
    FifoBatching, FirstFitFifoBatching, PickPosition, WarehouseOrder,
)
from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards
from ware_ops_algos.domain_models import Articles, ArticleType, Article, PickCart
from ware_ops_algos.domain_models.resources.models import DimensionType


def _order(order_id: int, items: int) -> WarehouseOrder:
    return WarehouseOrder(order_id=order_id, order_date=float(order_id), pick_positions=(
        PickPosition(order_number=order_id, article_id=order_id, amount=items,
                     pick_node=(1, order_id + 1), in_store=items),))


def _batches(algorithm) -> list[list[int]]:
    orders = [_order(3, 5), _order(1, 5), _order(0, 6), _order(2, 4)]
    return [[order.order_id for order in batch.orders]
            for batch in algorithm.solve(orders).batches]


CART = PickCart(n_dimension=1, capacities=[10], dimensions=[DimensionType.ITEMS], n_boxes=1,
                box_can_mix_orders=True)
ARTICLES = Articles(ArticleType.STANDARD, [Article(article_id=index) for index in range(4)])


def test_next_fit_fills_only_the_last_batch():
    assert _batches(FifoBatching(CART, ARTICLES)) == [[0], [1, 2], [3]]


def test_first_fit_fills_the_first_batch_with_room():
    assert _batches(FirstFitFifoBatching(CART, ARTICLES)) == [[0, 2], [1, 3]]


def test_first_fit_card_names_its_class_and_component():
    card = next(card for card in load_packaged_algo_cards() if card.algo_name == "FirstFitFiFo")
    assert card.problem_type == "batching"
    assert card.implementation["class_name"] == "FirstFitFifoBatching"
    assert card.implementation["component_name"] == "FirstFitFiFo"
