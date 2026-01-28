from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ware_ops_algos.domain_models import BaseDomainObject


class StorageType(str, Enum):
    SCATTERED = "scattered"
    DEDICATED = "dedicated"


@dataclass
class Position:
    x: float
    y: float


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class StorageSlot:
    id: str
    # bbox: BoundingBox
    level: int
    # aisle_id: Optional[int] = None
    # pick_node: Optional[tuple[float, float]] = None


@dataclass
class StorageLocation:
    id: str
    bbox: BoundingBox
    aisle_id: Optional[int] = None
    slots: list[StorageSlot] = field(default_factory=list)
    pick_node: Optional[tuple[float, float]] = None


@dataclass
class Location:
    x: int | float
    y: int | float
    article_id: int
    amount: int | float


@dataclass
class StorageLocations(BaseDomainObject):
    tpe: StorageType
    locations: Optional[list[Location]] = None
    storage_slots: Optional[list[StorageLocation]] = None
    article_location_mapping: Optional[dict[int, list[Location]]] = field(init=False, repr=False)
    location_article_mapping: Optional[dict[tuple[int | float, int | float], int]] = field(init=False, repr=False)

    def build_article_location_mapping(self):
        self.article_location_mapping = {}
        self.location_article_mapping = {}
        for loc in self.locations:
            if loc.article_id not in self.article_location_mapping.keys():
                self.article_location_mapping[loc.article_id] = []

            self.article_location_mapping[loc.article_id].append(loc)

        for loc in self.locations:
            if (loc.x, loc.y) not in self.location_article_mapping.keys():
                self.location_article_mapping[(loc.x, loc.y)] = loc.article_id

    def get_locations_by_article_id(self, article_id: int) -> list[Location]:
        return self.article_location_mapping[article_id]

    def get_features(self) -> dict[str, Any]:
        features = {}

        # Aggregate features
        features["n_locations"] = len(self.locations)
        features["n_unique_articles"] = len(self.article_location_mapping)

        features["amount"] = any(loc.amount is not None for loc in self.locations)
        x_coords = [loc.x for loc in self.locations]
        y_coords = [loc.y for loc in self.locations]
        features["x"] = x_coords  # Keep for compatibility
        features["y"] = y_coords
        features["article_id"] = [loc.article_id for loc in self.locations]
        return features
