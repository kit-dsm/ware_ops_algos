from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ware_ops_algos.domain_models.base_domain_object import BaseDomainObject

if TYPE_CHECKING:
    from ware_ops_algos.algorithms.algorithm_interfaces import AdmissionTour, BatchObject
    from ware_ops_algos.algorithms.algorithm_interfaces import WarehouseOrder
    from ware_ops_algos.domain_models.resources.models import Resource


class WarehouseInfoType(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass
class WarehouseInfo(BaseDomainObject):
    tpe: WarehouseInfoType


@dataclass(kw_only=True)
class DynamicInfo(WarehouseInfo):
    time: float | None = None
    congestion_rate: dict[str, float] = field(default_factory=dict)
    active_tours: list = field(default_factory=list)
    admission_tours: tuple[AdmissionTour, ...] = ()
    considered_active_orders: frozenset[tuple[int, int]] = frozenset()
    current_picker: Resource | None = None
    buffered_batches: list[BatchObject] = field(default_factory=list)
    done: bool = False
    wait_expired: bool = False
    is_break: bool = False
    n_staged_pallets: int = 0
    active_candidate_ids: frozenset[int] = frozenset()
    fixed_resolved_orders: tuple[WarehouseOrder, ...] = ()
