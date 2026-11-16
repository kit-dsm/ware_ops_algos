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
class TourProgress:
    """Execution facts of a tour that has started (status started or done).

    ``active_order_ids`` are the orders with an unpicked position; ``cursor``
    indexes the route node last reached; ``pinned_prefix`` is the executed part
    of the route, including an edge in progress; ``remaining_route`` the
    positions still to visit; ``routing_origin`` where a new route must start;
    ``occupied_bins`` the cart boxes in use; ``picking_until`` the end of a pick
    in progress; ``holding`` that the tour has reached the end of its route
    away from the depot and waits there.
    """

    active_order_ids: frozenset[int]
    cursor: int
    pinned_prefix: tuple[RouteNode, ...]
    remaining_route: tuple[tuple[float, float], ...]
    routing_origin: RoutingOrigin
    occupied_bins: int
    picking_until: float | None = None
    holding: bool = False


@dataclass(frozen=True)
class TourView:
    """A tour as a decision sees it; ``progress`` is set once the tour has started."""

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
    progress: TourProgress | None = None


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
