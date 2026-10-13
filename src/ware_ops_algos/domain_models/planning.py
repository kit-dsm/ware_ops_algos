"""Read-only decision data shared with warehouse algorithms.

CASIM owns the live state and constructs these views for each decision. The
algorithm interfaces remain independent of CASIM's configuration and commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .lifecycle import Accepts, TourStates

if TYPE_CHECKING:
    from ware_ops_algos.algorithms.algorithm_interfaces import (
        AdmissionTour, BatchObject, Route, RouteNode, RoutingOrigin, WarehouseOrder,
    )
    from ware_ops_algos.domain_models.resources.models import PickCart, Resource


@dataclass(frozen=True)
class DecisionContext:
    time: float
    cause: tuple[str, ...]
    decision: str
    version: int
    input_closed: bool
    solve: bool = True


@dataclass(frozen=True)
class TourView:
    tour_id: int
    picker_id: int | None
    status: TourStates
    route_version: int
    cursor: int
    pinned_prefix: tuple[RouteNode, ...]
    accepts: Accepts
    pick_cart: PickCart | None
    active_order_ids: frozenset[int]
    remaining_route: tuple[tuple[float, float], ...]
    occupied_bins: int
    routing_origin: RoutingOrigin
    picking_until: float | None = None
    route: Route | None = None
    batch: BatchObject | None = None
    start_time: float | None = None
    end_time: float | None = None
    processing_time: float | None = None
    positioning_mode: str | None = None
    positioning_hold: bool = False

@dataclass(frozen=True)
class PlanView:
    orders: tuple[WarehouseOrder, ...] = ()
    pick_lists: tuple[BatchObject, ...] = ()
    tours: tuple[TourView, ...] = ()
    resources: tuple[Resource, ...] = ()
    available_from: dict[int, float] = field(default_factory=dict)
    active_candidate_ids: frozenset[int] = frozenset()
