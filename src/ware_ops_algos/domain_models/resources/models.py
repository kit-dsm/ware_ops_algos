from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ware_ops_algos.domain_models import BaseDomainObject, OrderPosition, Article


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
class PalletSpec:
    """Physical pallet limits."""
    max_weight: float = 1500.0    # kg
    max_volume: float = 1500.0    # L
    max_height: float = 1800.0    # mm
    length: float = 1200.0        # mm
    width: float = 800.0          # mm

    @property
    def floor_area(self) -> float:
        return self.length * self.width


@dataclass
class Pallet:
    """A pallet under construction.

    Tracks total weight, volume, and a simple layer-based geometric state.
    """
    spec: PalletSpec
    positions: list[OrderPosition] = field(default_factory=list)
    weight: float = 0.0
    volume: float = 0.0
    height_below: float = 0.0
    layer_height: float = 0.0
    layer_area_used: float = 0.0

    @property
    def total_height(self) -> float:
        return self.height_below + self.layer_height

    def can_fit_kolli(self, art: Article) -> bool:
        ks = art.kolli_size or 1
        if self.weight + art.weight * ks > self.spec.max_weight:
            return False
        if self.volume + art.volume * ks > self.spec.max_volume:
            return False
        if art.length is None or art.width is None or art.height is None:
            return True
        footprint = art.length * art.width
        if (art.height <= self.layer_height
                and self.layer_area_used + footprint <= self.spec.floor_area):
            return True
        return self.total_height + art.height <= self.spec.max_height

    def add_kolli(self, art: Article) -> None:
        ks = art.kolli_size or 1
        self.weight += art.weight * ks
        self.volume += art.volume * ks
        if art.length is None or art.width is None or art.height is None:
            return
        footprint = art.length * art.width
        if (art.height <= self.layer_height
                and self.layer_area_used + footprint <= self.spec.floor_area):
            self.layer_area_used += footprint
            return
        self.height_below += self.layer_height
        self.layer_height = art.height
        self.layer_area_used = footprint


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
    tour_setup_time: float = None
    # Dynamic Information
    available: Optional[bool] = None # "stronger" compared to occupied -> E.g a picker is not available for a shift
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


@dataclass
class Resources(BaseDomainObject):
    tpe: ResourceType
    resources: Optional[list[Resource]] = None

    def get_features(self) -> dict[str, any]:
        features = {}

        # Aggregate features
        features["n_resources"] = len(self.resources)

        features["capacity"] = any(r.capacity is not None for r in self.resources)
        features["capacities"] = True
        features["dimensions_type"] = True
        features["speed"] = any(r.speed is not None for r in self.resources)
        features["time_per_pick"] = any(r.time_per_pick is not None for r in self.resources)
        features["current_location"] = any(r.current_location is not None for r in self.resources)
        return features