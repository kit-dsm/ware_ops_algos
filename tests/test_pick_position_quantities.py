import pandas as pd
import pytest

from ware_ops_algos.algorithms import (
    ExactCombinedBatchingRouting,
    ExactTSPBatchingAndRoutingDistance,
    ExactTSPBatchingAndRoutingMaxCompletionTime,
    GreedyItemAssignment,
    MinMaxItemAssignment,
    MinMinItemAssignment,
    NearestNeighborItemAssignment,
    PickPosition,
    SinglePositionItemAssignment,
    WarehouseOrder,
)
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
    Resource,
    StorageLocations,
    StorageType,
)


class _ManhattanScore:
    def __init__(self, start_node=(0, 0), **_kwargs):
        self.start_node = start_node

    def score(self, positions):
        current = self.start_node
        distance = 0
        for position in positions:
            distance += abs(current[0] - position.pick_node[0])
            distance += abs(current[1] - position.pick_node[1])
            current = position.pick_node
        return float(distance)


def _split_fixture():
    nodes = [(0, 0), (1, 1), (2, 1)]
    distances = pd.DataFrame(
        [
            [abs(x - u) + abs(y - v) for u, v in nodes]
            for x, y in nodes
        ],
        index=nodes,
        columns=nodes,
    )
    storage = StorageLocations(
        tpe=StorageType.SCATTERED,
        locations=[
            Location(x=1, y=1, article_id=7, amount=2),
            Location(x=2, y=1, article_id=7, amount=3),
        ],
    )
    storage.build_article_location_mapping()
    order = Order(
        order_id=11,
        parent_order_id="parent-11",
        due_date=42.0,
        order_date=3.0,
        order_positions=[
            OrderPosition(
                order_number=11,
                article_id=7,
                amount=4,
                article_name="split sku",
            )
        ],
    )
    return storage, distances, order


@pytest.mark.parametrize(
    "factory",
    [
        lambda storage, distances: GreedyItemAssignment(storage),
        lambda storage, distances: NearestNeighborItemAssignment(
            storage, distances, start_node=(0, 0)
        ),
        lambda storage, distances: SinglePositionItemAssignment(
            storage,
            distances,
            _ManhattanScore,
            {"start_node": (0, 0)},
        ),
        lambda storage, distances: MinMaxItemAssignment(
            storage, distances, start_node=(0, 0)
        ),
        lambda storage, distances: MinMinItemAssignment(
            storage, distances, start_node=(0, 0)
        ),
    ],
    ids=["greedy", "nearest-neighbor", "single-position", "min-max", "min-min"],
)
def test_all_item_assignment_paths_preserve_split_pick_quantities(factory):
    storage, distances, order = _split_fixture()

    resolved_order = factory(storage, distances).solve([order]).resolved_orders[0]
    picks = resolved_order.pick_positions

    assert resolved_order.order_id == order.order_id
    assert resolved_order.parent_order_id == order.parent_order_id
    assert resolved_order.due_date == order.due_date
    assert resolved_order.order_date == order.order_date
    assert len(picks) == 2
    assert sum(position.picked_quantity for position in picks) == 4
    assert all(position.amount == 4 for position in picks)
    assert all(position.article_name == "split sku" for position in picks)

    stock_by_node = {
        (location.x, location.y): location.amount
        for location in storage.locations
    }
    assert all(
        0 < position.picked_quantity <= stock_by_node[position.pick_node]
        for position in picks
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda storage, distances: GreedyItemAssignment(storage),
        lambda storage, distances: NearestNeighborItemAssignment(
            storage, distances, start_node=(0, 0)
        ),
        lambda storage, distances: SinglePositionItemAssignment(
            storage,
            distances,
            _ManhattanScore,
            {"start_node": (0, 0)},
        ),
        lambda storage, distances: MinMaxItemAssignment(
            storage, distances, start_node=(0, 0)
        ),
        lambda storage, distances: MinMinItemAssignment(
            storage, distances, start_node=(0, 0)
        ),
    ],
    ids=["greedy", "nearest-neighbor", "single-position", "min-max", "min-min"],
)
def test_dedicated_location_with_sufficient_stock_fulfils_order_demand(factory):
    nodes = [(0, 0), (1, 1)]
    distances = pd.DataFrame(
        [[0, 2], [2, 0]],
        index=nodes,
        columns=nodes,
    )
    storage = StorageLocations(
        tpe=StorageType.DEDICATED,
        locations=[Location(x=1, y=1, article_id=7, amount=4)],
    )
    storage.build_article_location_mapping()
    order = Order(
        order_id=11,
        order_positions=[OrderPosition(order_number=11, article_id=7, amount=4)],
    )

    resolved = factory(storage, distances).solve([order]).resolved_orders[0]

    assert len(resolved.pick_positions) == 1
    assert resolved.pick_positions[0].amount == 4
    assert resolved.pick_positions[0].picked_quantity == 4


@pytest.mark.parametrize(
    "factory",
    [
        lambda storage, distances: GreedyItemAssignment(storage),
        lambda storage, distances: NearestNeighborItemAssignment(
            storage, distances, start_node=(0, 0)
        ),
        lambda storage, distances: SinglePositionItemAssignment(
            storage,
            distances,
            _ManhattanScore,
            {"start_node": (0, 0)},
        ),
        lambda storage, distances: MinMaxItemAssignment(
            storage, distances, start_node=(0, 0)
        ),
        lambda storage, distances: MinMinItemAssignment(
            storage, distances, start_node=(0, 0)
        ),
    ],
    ids=["greedy", "nearest-neighbor", "single-position", "min-max", "min-min"],
)
def test_all_item_assignment_paths_reject_insufficient_stock(factory):
    nodes = [(0, 0), (1, 1)]
    distances = pd.DataFrame([[0, 2], [2, 0]], index=nodes, columns=nodes)
    storage = StorageLocations(
        tpe=StorageType.DEDICATED,
        locations=[Location(x=1, y=1, article_id=7, amount=1)],
    )
    storage.build_article_location_mapping()
    order = Order(
        order_id=11,
        order_positions=[OrderPosition(order_number=11, article_id=7, amount=4)],
    )

    with pytest.raises(RuntimeError, match="Insufficient stock for article 7"):
        factory(storage, distances).solve([order])


def _split_order(*, second_article=False):
    picks = [
        PickPosition(11, 7, 4, (1, 1), 2),
        PickPosition(11, 7, 4, (2, 1), 2),
    ]
    if second_article:
        picks.extend(
            [
                PickPosition(11, 8, 3, (3, 1), 1),
                PickPosition(11, 8, 3, (4, 1), 2),
            ]
        )
    return WarehouseOrder(order_id=11, pick_positions=tuple(picks))


@pytest.mark.parametrize(
    ("dimension", "expected"),
    [
        (DimensionType.ITEMS, 4.0),
        (DimensionType.WEIGHT, 10.0),
        (DimensionType.VOLUME, 12.0),
        (DimensionType.ORDERLINES, 1.0),
        (DimensionType.ORDERS, 1.0),
    ],
)
def test_capacity_dimensions_use_physical_quantity_and_semantic_lines(
    dimension, expected
):
    cart = PickCart(
        n_dimension=1,
        capacities=[100.0],
        dimensions=[dimension],
        n_boxes=1,
        box_can_mix_orders=True,
    )
    articles = Articles(
        ArticleType.STANDARD,
        [Article(article_id=7, weight=2.5, volume=3.0)],
    )

    assert CapacityChecker(cart, articles).order_consumption(
        _split_order()
    ) == pytest.approx((expected,))


def test_foodmart_box_rounding_and_kris_orderline_semantics_survive_splits():
    articles = Articles(
        ArticleType.STANDARD,
        [Article(article_id=7), Article(article_id=8)],
    )
    foodmart_cart = PickCart(
        n_dimension=1,
        capacities=[3],
        dimensions=[DimensionType.ITEMS],
        n_boxes=2,
        box_can_mix_orders=False,
    )
    foodmart_checker = CapacityChecker(foodmart_cart, articles)
    assert foodmart_checker.order_box_count(_split_order()) == 2
    assert foodmart_checker.orders_fit([_split_order()])
    assert not foodmart_checker.orders_fit([_split_order(), _split_order()])

    kris_cart = PickCart(
        n_dimension=1,
        capacities=[2],
        dimensions=[DimensionType.ORDERLINES],
        n_boxes=1,
        box_can_mix_orders=True,
    )
    assert CapacityChecker(kris_cart, articles).order_consumption(
        _split_order(second_article=True)
    ) == (2,)


@pytest.mark.parametrize(
    "model_class",
    [
        ExactTSPBatchingAndRoutingDistance,
        ExactTSPBatchingAndRoutingMaxCompletionTime,
        ExactCombinedBatchingRouting,
    ],
)
def test_integrated_exact_models_do_not_duplicate_split_line_quantity(model_class):
    picks = list(_split_order().pick_positions)
    model = model_class.__new__(model_class)
    model.pick_list = picks
    model.picker = [Resource(id=0, capacity=10, speed=1.0)]

    if model_class is not ExactCombinedBatchingRouting:
        nodes = [(0, 0), (1, 1), (2, 1), (3, 0)]
        model.start_node = (0, 0)
        model.end_node = (3, 0)
        model.distance_matrix = pd.DataFrame(
            [
                [abs(x - u) + abs(y - v) for u, v in nodes]
                for x, y in nodes
            ],
            index=nodes,
            columns=nodes,
        )

    model._set_routing_parameters()

    assert model.list_item_amounts == [2, 2]
    order_sizes = model.s_o if model_class is ExactCombinedBatchingRouting else model.s_i
    assert order_sizes == {11: 4}
