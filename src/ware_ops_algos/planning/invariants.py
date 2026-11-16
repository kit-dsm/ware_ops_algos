"""Simulator-independent checks of plan state and progress."""

from __future__ import annotations

from collections import Counter, defaultdict

from ware_ops_algos.algorithms import NodeType

from .commands import Violation, ViolationCode
from .lifecycle import TOUR_TRANSITIONS, TourStates
from .models import PlanView, TourView


def _violation(code: ViolationCode, message: str, tour_id: int | None = None) -> Violation:
    return Violation(code=code, tour_id=tour_id, message=message)


def _orders_of(tour: TourView) -> tuple[int, ...]:
    return tuple(tour.batch.order_numbers) if tour.batch is not None else ()


def _visible_order_ids(plan: PlanView) -> set[int]:
    ids = {order.order_id for order in plan.order_buffer}
    ids.update(order_id for batch in plan.pick_list_buffer
               for order_id in batch.order_numbers)
    ids.update(order_id for tour in plan.tours
               if tour.status is not TourStates.CANCELLED
               for order_id in _orders_of(tour))
    return ids


def check_plan(plan: PlanView) -> tuple[Violation, ...]:
    violations: list[Violation] = []
    locations: dict[int, list[str]] = defaultdict(list)
    for order in plan.order_buffer:
        locations[order.order_id].append("buffer")
    for batch in plan.pick_list_buffer:
        for order_id in batch.order_numbers:
            locations[order_id].append(f"pick_list:{batch.batch_id}")
    for tour in plan.tours:
        if tour.status is TourStates.CANCELLED:
            continue
        for order_id in _orders_of(tour):
            place = "delivered" if tour.status is TourStates.DONE else f"tour:{tour.tour_id}"
            locations[order_id].append(place)
    for order_id, places in locations.items():
        if len(places) != 1:
            violations.append(_violation(
                ViolationCode.MEMBERSHIP, f"Order {order_id} occurs in {places}"))

    by_picker: dict[int, list[TourView]] = defaultdict(list)
    for tour in plan.tours:
        if tour.status in (TourStates.SCHEDULED, TourStates.PENDING, TourStates.STARTED):
            if tour.picker_id is None or tour.start_time is None or tour.end_time is None:
                violations.append(_violation(
                    ViolationCode.SCHEDULE, "Committed tour needs picker and times", tour.tour_id))
        if (tour.status is not TourStates.CANCELLED and tour.picker_id is not None
                and tour.start_time is not None and tour.end_time is not None):
            by_picker[tour.picker_id].append(tour)
        if tour.status is TourStates.STARTED:
            if tour.route is None or tour.route.annotated_route is None:
                violations.append(_violation(
                    ViolationCode.ROUTE, "Started tour needs an annotated route", tour.tour_id))
            elif tour.progress is None:
                violations.append(_violation(
                    ViolationCode.ROUTE, "Started tour needs its progress", tour.tour_id))
            elif tour.batch is not None:
                picks = Counter(position.pick_node for position in tour.batch.pick_positions)
                completed = Counter(node.position for node in tour.progress.pinned_prefix
                                    if node.node_type is NodeType.PICK)
                remaining = picks - completed
                route_picks = Counter(tour.progress.remaining_route)
                if not all(route_picks[node] >= count for node, count in remaining.items()):
                    violations.append(_violation(
                        ViolationCode.ROUTE, "Route omits an unpicked position", tour.tour_id))
            if (tour.progress is not None and tour.pick_cart is not None
                    and tour.pick_cart.n_boxes is not None
                    and tour.progress.occupied_bins > tour.pick_cart.n_boxes):
                violations.append(_violation(
                    ViolationCode.CAPACITY, "Occupied boxes exceed cart capacity", tour.tour_id))

    for picker_id, tours in by_picker.items():
        ordered = sorted(tours, key=lambda tour: (tour.start_time, tour.tour_id))
        for previous, current in zip(ordered, ordered[1:]):
            if previous.end_time > current.start_time:
                violations.append(_violation(
                    ViolationCode.OVERLAP, f"Picker {picker_id} has overlapping tours",
                    current.tour_id))
    return tuple(violations)


def check_progress(before: PlanView, after: PlanView) -> tuple[Violation, ...]:
    violations: list[Violation] = []
    if after.time < before.time:
        violations.append(_violation(
            ViolationCode.TIME, f"Plan time moved backward from {before.time} to {after.time}"))
    for order_id in sorted(_visible_order_ids(before) - _visible_order_ids(after)):
        violations.append(_violation(
            ViolationCode.MEMBERSHIP, f"Order {order_id} disappeared from the plan"))
    previous = {tour.tour_id: tour for tour in before.tours}
    current = {tour.tour_id: tour for tour in after.tours}
    for tour_id, old in previous.items():
        new = current.get(tour_id)
        if new is None:
            violations.append(_violation(
                ViolationCode.MEMBERSHIP, "Tour disappeared from the full plan", tour_id))
            continue
        if new.status != old.status and new.status not in TOUR_TRANSITIONS[old.status]:
            violations.append(_violation(
                ViolationCode.STATUS, f"Illegal transition {old.status.value} to {new.status.value}",
                tour_id))
        if old.status in (TourStates.DONE, TourStates.CANCELLED) and old != new:
            violations.append(_violation(
                ViolationCode.TERMINAL, "Terminal tour changed", tour_id))
        if (old.status is TourStates.STARTED
                and new.status in (TourStates.STARTED, TourStates.DONE)
                and old.progress is not None and new.progress is not None):
            old_prefix, new_prefix = old.progress.pinned_prefix, new.progress.pinned_prefix
            if new_prefix[:len(old_prefix)] != old_prefix:
                violations.append(_violation(
                    ViolationCode.EXECUTED_PREFIX, "Executed route prefix changed", tour_id))
            old_picks = Counter(node.position for node in old_prefix
                                if node.node_type is NodeType.PICK)
            new_picks = Counter(node.position for node in new_prefix
                                if node.node_type is NodeType.PICK)
            if not all(new_picks[node] >= count for node, count in old_picks.items()):
                violations.append(_violation(
                    ViolationCode.EXECUTED_PREFIX, "Completed pick was undone", tour_id))
    return tuple(violations)
