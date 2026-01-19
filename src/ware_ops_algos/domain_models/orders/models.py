from dataclasses import dataclass, field
import datetime
from enum import Enum
from typing import Optional

from ware_ops_algos.domain_models import BaseDomainObject


class OrderType(str, Enum):
    STANDARD = "standard"
    UNIT_DEMAND = "unit_demand"
    GENERAL_DEMAND = "general_demand"


@dataclass
class OrderPosition:
    order_number: int
    article_id: int
    amount: int
    article_name: Optional[str] = None

    @staticmethod
    def from_dict(order_number: int, data: dict) -> "OrderPosition":
        return OrderPosition(
            order_number=order_number,
            article_id=data["article_id"],
            article_name=data.get("article_name"),
            amount=data["amount"],
        )


@dataclass
class ResolvedOrderPosition:
    position: OrderPosition
    pick_node: tuple[int, int]
    fulfilled: Optional[int] = None  # TODO: ambiguous, rename
    # dynamic info
    picked: Optional[bool] = None


@dataclass
class Order:
    order_id: int
    due_date: Optional[datetime.datetime | float] = None
    order_date: Optional[datetime.datetime | float] = None
    order_positions: list[OrderPosition] = field(default_factory=list)

    @staticmethod
    def from_dict(order_number: int, data: dict) -> "Order":
        return Order(
            order_id=order_number,
            due_date=data.get("due_date") or None,
            order_date=data.get("order_date"),
            order_positions=[
                OrderPosition.from_dict(order_number, pos) for pos in data.get("order_positions", [])
            ]
        )


@dataclass
class OrdersDomain(BaseDomainObject):
    tpe: OrderType
    orders: Optional[list[Order]] = None

    def get_features(self) -> dict[str, any]:
        features = {}
        features["n_orders"] = len(self.orders)

        if self.orders:
            features["due_date"] = any(o.due_date is not None for o in self.orders)
            features["order_date"] = any(o.order_date is not None for o in self.orders)
            features["article_id"] = any(o.order_positions[0].article_id is not None for o in self.orders)
            features["amount"] = any(o.order_positions[0].amount is not None for o in self.orders)
        return features





