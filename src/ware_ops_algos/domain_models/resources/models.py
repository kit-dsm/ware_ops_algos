from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ware_ops_algos.domain_models import BaseDomainObject


class ResourceType(str, Enum):
    HUMAN = "human"
    ROBOT = "robot"
    COBOT = "cobot"
    MIXED = "mixed"


class DimensionType(str, Enum):
    WEIGHT = "weight"
    ITEMS = "items"
    VOLUME = "volume"
    ORDERS = "orders"
    ORDERLINES = "orderline"


@dataclass
class Box:
    box_id: int
    order_ids: set[int]
    items: list[tuple[int, int]] = field(default_factory=list)  # [(article_id, quantity), ...]
    consumption: list[float] = field(default_factory=list)  # [weight, volume, ...]


@dataclass
class PickCart:
    n_dimension: Optional[int] = None
    capacities: Optional[list[float]] = None  # capa per box
    dimensions: Optional[list[DimensionType]] = None  # dimension in which we meassure box cap
    n_boxes: Optional[int] = None  # n boxes that fit on the cart
    box_can_mix_orders: Optional[bool] = None


@dataclass
class Resource:
    id: int
    capacity: Optional[int] = None
    speed: Optional[float] = None
    time_per_pick: Optional[float] = None
    pick_cart: Optional[PickCart] = None
    tpe: ResourceType = ResourceType.HUMAN
    # Dynamic Information
    occupied: Optional[bool] = None
    current_location: Optional[tuple[float, float]] = None


@dataclass
class CobotPicker(Resource):
    mode: str = "driving"
    speed_follow_mode: float = None
    aisle_congestion_rate: float = None
    tour_setup_time: float = None


@dataclass
class ManualPicker(Resource):
    aisle_congestion_rate: float = None
    tour_setup_time: float = None


@dataclass
class Resources(BaseDomainObject):
    tpe: ResourceType
    resources: Optional[list[Resource]] = None

    def get_features(self) -> dict[str, any]:
        features = {}

        # Aggregate features
        features["n_resources"] = len(self.resources)

        features["capacity"] = any(r.capacity is not None for r in self.resources)
        features["speed"] = any(r.speed is not None for r in self.resources)
        features["time_per_pick"] = any(r.time_per_pick is not None for r in self.resources)
        features["current_location"] = any(r.current_location is not None for r in self.resources)
        return features