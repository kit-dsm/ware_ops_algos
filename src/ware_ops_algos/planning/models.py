"""Decision-visible facts about a warehouse plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from .lifecycle import Accepts, TourStates

if TYPE_CHECKING:
    from ware_ops_algos.algorithms.algorithm_interfaces import (
        BatchObject, Route, RouteNode, RoutingOrigin, WarehouseOrder,
    )
    from ware_ops_algos.domain_models.resources.models import PickCart


@dataclass(frozen=True)
class TourView:
    tour_id: int
    picker_id: int | None
    status: TourStates
    accepts: Accepts
    batch: BatchObject | None = None
    route: Route | None = None
    start_time: float | None = None
    end_time: float | None = None
    processing_time: float | None = None
    pick_cart: PickCart | None = None
    active_order_ids: frozenset[int] = frozenset()
    cursor: int | None = None
    pinned_prefix: tuple[RouteNode, ...] = ()
    remaining_route: tuple[tuple[float, float], ...] = ()
    routing_origin: RoutingOrigin | None = None
    occupied_bins: int | None = None
    picking_until: float | None = None
    holding: bool = False
    positioning_mode: str | None = None
    positioning_hold: bool = False


@dataclass(frozen=True)
class PlanView:
    time: float
    order_buffer: tuple[WarehouseOrder, ...] = ()
    pick_list_buffer: tuple[BatchObject, ...] = ()
    tours: tuple[TourView, ...] = ()
    available_from: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.order_buffer, key=lambda order:
                               (float("-inf") if order.order_date is None
                                else order.order_date, order.order_id)))
        object.__setattr__(self, "order_buffer", ordered)
