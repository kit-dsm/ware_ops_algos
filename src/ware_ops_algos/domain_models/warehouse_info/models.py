from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ware_ops_algos.domain_models import BaseDomainObject


class WarehouseInfoType(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass
class WarehouseInfo(BaseDomainObject):
    tpe: WarehouseInfoType
    # Forecast available to a planner. Realised arrivals belong to the event stream.
    arrival_process: str | None = None
    mean_interarrival_time_s: float | None = None
    order_lines_per_order: int | None = None
    pick_location_distribution: str | None = None


