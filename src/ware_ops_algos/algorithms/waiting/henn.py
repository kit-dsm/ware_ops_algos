"""Stateless deterministic online waiting algorithms."""

from __future__ import annotations

from dataclasses import dataclass

from ware_ops_algos.algorithms.algorithm_interfaces import (
    Algorithm,
    CombinedRoutingSolution,
    Route,
    WaitingSolution,
    WarehouseOrder,
)
from ware_ops_algos.domain_models import Resource


@dataclass(frozen=True)
class DeterministicWaitingInput:
    """Observable state shared by deterministic waiting algorithms."""

    candidate: CombinedRoutingSolution
    picker: Resource
    orders: tuple[WarehouseOrder, ...]
    current_time: float
    input_closed: bool
    single_services: dict[int, float]


def route_service_time(route: Route, picker: Resource) -> float:
    """Return setup, travel, and item-picking time in seconds."""
    item_count = sum(
        int(position.in_store)
        for position in route.batch.pick_positions
    )
    if route.item_sequence is not None and len(route.item_sequence) != item_count:
        raise ValueError(
            "The routed item sequence does not match the batch item count: "
            f"{len(route.item_sequence)} != {item_count}"
        )
    return (
        float(picker.tour_setup_time)
        + float(route.distance) / float(picker.speed)
        + item_count * float(picker.time_per_pick)
    )


def _candidate_routes(
    input_data: DeterministicWaitingInput,
) -> tuple[Route, ...]:
    if not isinstance(input_data.candidate, CombinedRoutingSolution):
        raise TypeError("Waiting candidates must be CombinedRoutingSolution")
    routes = tuple(input_data.candidate.routes)
    if not routes:
        raise ValueError("Open orders produced no candidate route")
    return routes


def _details(
    input_data: DeterministicWaitingInput,
    routes: tuple[Route, ...],
) -> dict[str, object]:
    return {
        "candidate_batches": [
            {
                "batch_id": route.batch.batch_id,
                "order_ids": sorted(route.batch.order_numbers),
                "distance": float(route.distance),
            }
            for route in routes
        ],
        "input_closed": input_data.input_closed,
    }


def _dispatch_all(
    routes: tuple[Route, ...],
    details: dict[str, object],
) -> WaitingSolution:
    details["selected_order_ids"] = [
        sorted(route.batch.order_numbers) for route in routes
    ]
    return WaitingSolution(
        action="dispatch",
        routes=routes,
        reason="input_closed",
        details=details,
    )


def _dispatch_first(
    routes: tuple[Route, ...],
    reason: str,
    details: dict[str, object],
) -> WaitingSolution:
    route = routes[0]
    details["selected_order_ids"] = sorted(route.batch.order_numbers)
    return WaitingSolution(
        action="dispatch",
        routes=(route,),
        reason=reason,
        details=details,
    )


class HennWaiting(Algorithm[DeterministicWaitingInput, WaitingSolution]):
    """Original Henn Algorithm 4.1 threshold with alpha equal to one."""

    algo_name = "HennWaiting"

    def _run(self, input_data: DeterministicWaitingInput) -> WaitingSolution:
        routes = _candidate_routes(input_data)
        details = _details(input_data, routes)
        details["candidate_batches"] = [
            {
                **batch,
                "service_time_s": route_service_time(route, input_data.picker),
            }
            for batch, route in zip(details["candidate_batches"], routes)
        ]
        if input_data.input_closed:
            return _dispatch_all(routes, details)
        if len(routes) > 1:
            return _dispatch_first(
                routes,
                "multiple_batches_release_selected",
                details,
            )

        route = routes[0]
        critical_order = min(
            route.batch.orders,
            key=lambda order: (
                -input_data.single_services[order.order_id],
                int(order.order_id),
            ),
        )
        critical_service = input_data.single_services[
            critical_order.order_id
        ]
        batch_service = route_service_time(route, input_data.picker)
        threshold = (
            2 * float(critical_order.order_date)
            + critical_service
            - batch_service
        )
        details.update(
            {
                "critical_order_id": int(critical_order.order_id),
                "critical_order_service_time_s": critical_service,
                "batch_service_time_s": batch_service,
                "threshold_s": threshold,
            }
        )
        if input_data.current_time < threshold:
            return WaitingSolution(
                action="wait",
                routes=(route,),
                dispatch_at=threshold,
                reason="single_batch_threshold",
                details=details,
            )
        return _dispatch_first(routes, "single_batch_release", details)


class NoWaiting(Algorithm[DeterministicWaitingInput, WaitingSolution]):
    """Release the upstream-selected candidate immediately."""

    algo_name = "NoWaiting"

    def _run(self, input_data: DeterministicWaitingInput) -> WaitingSolution:
        routes = _candidate_routes(input_data)
        details = _details(input_data, routes)
        if input_data.input_closed:
            return _dispatch_all(routes, details)
        return _dispatch_first(routes, "no_wait_release", details)


class FillOrAgeWaiting(
    Algorithm[DeterministicWaitingInput, WaitingSolution]
):
    """In-house benchmark using visible fill and oldest-order age."""

    algo_name = "FillOrAgeWaiting"

    def __init__(
        self,
        fill_threshold: float = 0.75,
        max_age_s: float = 300.0,
    ):
        super().__init__()
        if not 0.0 <= fill_threshold <= 1.0:
            raise ValueError("fill_threshold must be between zero and one")
        if max_age_s < 0.0:
            raise ValueError("max_age_s must be non-negative")
        self.fill_threshold = fill_threshold
        self.max_age_s = max_age_s

    def _run(self, input_data: DeterministicWaitingInput) -> WaitingSolution:
        routes = _candidate_routes(input_data)
        details = _details(input_data, routes)
        if input_data.input_closed:
            return _dispatch_all(routes, details)

        visible_items = sum(
            int(position.in_store)
            for order in input_data.orders
            for position in order.pick_positions
        )
        fill = min(
            1.0,
            visible_items / max(1.0, float(input_data.picker.capacity)),
        )
        oldest_age = max(
            (
                input_data.current_time - float(order.order_date)
                for order in input_data.orders
            ),
            default=0.0,
        )
        details.update(
            {
                "fill": fill,
                "fill_threshold": self.fill_threshold,
                "oldest_age_s": oldest_age,
                "max_age_s": self.max_age_s,
            }
        )
        if fill < self.fill_threshold and oldest_age < self.max_age_s:
            return WaitingSolution(
                action="wait",
                routes=routes,
                dispatch_at=(
                    input_data.current_time
                    + self.max_age_s
                    - oldest_age
                ),
                reason="fill_or_age_below_threshold",
                details=details,
            )
        return _dispatch_first(routes, "fill_or_age_release", details)


class QueueThresholdWaiting(
    Algorithm[DeterministicWaitingInput, WaitingSolution]
):
    """Published VTW-QO rule based on the visible queued-order count."""

    algo_name = "QueueThresholdWaiting"

    def __init__(self, minimum_orders: int):
        super().__init__()
        if minimum_orders < 1:
            raise ValueError("minimum_orders must be positive")
        self.minimum_orders = int(minimum_orders)

    def _run(self, input_data: DeterministicWaitingInput) -> WaitingSolution:
        routes = _candidate_routes(input_data)
        details = _details(input_data, routes)
        details.update(
            {
                "visible_orders": len(input_data.orders),
                "minimum_orders": self.minimum_orders,
            }
        )
        if input_data.input_closed:
            return _dispatch_all(routes, details)
        if len(input_data.orders) < self.minimum_orders:
            return WaitingSolution(
                action="wait",
                routes=routes,
                reason="queue_below_threshold",
                details=details,
            )
        return _dispatch_first(routes, "queue_threshold_release", details)
