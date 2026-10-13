"""Incumbent batching is applicable only when every picker has the same cart."""

from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards
from ware_ops_algos.domain_algo_mapper.domain_algo_mapper import DomainAlgorithmMapper
from ware_ops_algos.domain_models import DataCard, PickCart, Resource, Resources
from ware_ops_algos.domain_models.resources.models import DimensionType, ResourceType


def _data_card(resources: Resources) -> DataCard:
    empty = {"type": "any", "features": {}}
    return DataCard(
        name="incumbent", problem_class="OBRP", objective="Distance",
        layout=empty, articles=empty,
        orders={"type": "any", "features": {"order_date": True}},
        resources={"type": "human", "features": resources.get_features()},
        storage=empty, warehouse_info=empty,
        decision_features={"type": "decision", "features": {"incumbent": True}},
    )


def test_incumbent_card_rejects_heterogeneous_carts():
    card = next(card for card in load_packaged_algo_cards()
                if card.algo_name == "IncumbentStartBatching")
    mapper = DomainAlgorithmMapper({"OBRP": {"variables": ["batching"]}})
    cart = PickCart(capacities=[1], dimensions=[DimensionType.ORDERS], n_boxes=1)
    same = Resources(ResourceType.HUMAN, [
        Resource(id=0, pick_cart=cart), Resource(id=1, pick_cart=cart),
    ])
    different = Resources(ResourceType.HUMAN, [
        Resource(id=0, pick_cart=cart),
        Resource(id=1, pick_cart=PickCart(
            capacities=[1], dimensions=[DimensionType.ORDERS], n_boxes=2)),
    ])
    assert mapper.filter([card], _data_card(same)) == [card]
    assert mapper.filter([card], _data_card(different)) == []
