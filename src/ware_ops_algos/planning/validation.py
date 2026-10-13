"""Pure checks of revision scope and proposed plan changes."""

from __future__ import annotations

from .commands import (
    CancelTour, Command, CommitTour, CreatePickList, CreateTour,
    ReplaceSuffix, Reschedule, Violation, ViolationCode,
)
from .lifecycle import Accepts, PHYSICAL, TourStates, accepts_at_least
from .models import PlanView, TourView
from .scope import RevisionScope, StateInformation


def check_scope(state_information: StateInformation,
                scope: RevisionScope) -> tuple[TourStates, ...]:
    """Statuses whose configured revision level exceeds their physical bound."""
    return tuple(status for status in TourStates
                 if status in state_information.tours and not accepts_at_least(
                     PHYSICAL[status], scope.accepts(status)))


def _violation(code: ViolationCode, message: str,
               tour_id: int | None = None) -> Violation:
    return Violation(code=code, tour_id=tour_id, message=message)


def _check_tour(view: TourView, required: Accepts) -> tuple[Violation, ...]:
    violations = []
    bound = PHYSICAL[view.status]
    if not accepts_at_least(bound, view.accepts):
        violations.append(_violation(
            ViolationCode.PHYSICAL,
            f"{view.status.value} physically permits {bound.value}, not {view.accepts.value}",
            view.tour_id))
    if not accepts_at_least(view.accepts, required):
        violations.append(_violation(
            ViolationCode.ACCEPTS,
            f"Tour accepts {view.accepts.value}, {required.value} required",
            view.tour_id))
    return tuple(violations)


def check(commands: tuple[Command, ...], plan: PlanView) -> tuple[Violation, ...]:
    """Return every decision-visible violation without mutating the plan."""
    views = {tour.tour_id: tour for tour in plan.tours}
    violations: list[Violation] = []
    created_count = 0
    free_ids = {order.order_id for order in plan.order_buffer}
    free_ids.update(order_id for batch in plan.pick_list_buffer
                    for order_id in batch.order_numbers)
    free_ids.update(order_id for command in commands
                    if isinstance(command, CancelTour)
                    for order_id in (views[command.tour_id].batch.order_numbers
                                     if command.tour_id in views and
                                     views[command.tour_id].batch is not None
                                     else ()))
    for command in commands:
        if isinstance(command, CreateTour):
            if not set(command.job.order_numbers) <= free_ids:
                violations.append(_violation(
                    ViolationCode.STATE_INFORMATION,
                    "New tour contains orders outside exposed or released work"))
            created_count += 1
            continue
        if isinstance(command, CreatePickList):
            exposed_ids = {order.order_id for order in plan.order_buffer}
            if not command.batch.order_numbers <= exposed_ids:
                violations.append(_violation(
                    ViolationCode.STATE_INFORMATION,
                    "Pick list includes an order outside the exposed buffer"))
            continue
        if isinstance(command, CommitTour) and command.creation_index is not None:
            if not 0 <= command.creation_index < created_count:
                violations.append(_violation(
                    ViolationCode.MEMBERSHIP,
                    "CommitTour references a missing new tour"))
            continue
        tour_id = command.tour_id
        view = views.get(tour_id)
        if view is None:
            violations.append(_violation(
                ViolationCode.STATE_INFORMATION,
                f"Tour {tour_id} is not exposed", tour_id))
            continue
        if isinstance(command, CommitTour):
            if view.status is not TourStates.PLANNED:
                violations.append(_violation(
                    ViolationCode.PHYSICAL,
                    "Only a planned tour can be committed", tour_id))
            continue
        required = Accepts.INSERT if isinstance(command, ReplaceSuffix) else Accepts.REBUILD
        violations.extend(_check_tour(view, required))
        if isinstance(command, Reschedule):
            if command.picker_id not in plan.available_from:
                violations.append(_violation(
                    ViolationCode.STATE_INFORMATION,
                    "Reschedule picker is not exposed", tour_id))
        if not isinstance(command, ReplaceSuffix):
            continue
        if command.route.batch is None:
            violations.append(_violation(
                ViolationCode.MEMBERSHIP, "Replacement route has no batch", tour_id))
            continue
        old_ids = set(view.batch.order_numbers) if view.batch else set()
        new_ids = set(command.route.batch.order_numbers)
        added = set(command.added_order_ids)
        if not old_ids <= new_ids or added != new_ids - old_ids:
            violations.append(_violation(
                ViolationCode.MEMBERSHIP,
                "Replacement must retain orders and name exactly the additions",
                tour_id))
        exposed_ids = {order.order_id for order in plan.order_buffer}
        if not added <= exposed_ids:
            violations.append(_violation(
                ViolationCode.STATE_INFORMATION,
                "Added orders are not in the exposed order buffer", tour_id))
        if view.status is TourStates.STARTED:
            if (command.route.routing_origin is None or
                    command.route.routing_origin != view.routing_origin):
                violations.append(_violation(
                    ViolationCode.EXECUTED_PREFIX,
                    "Replacement changes the running tour's routing origin",
                    tour_id))
            if (view.pick_cart is not None and view.pick_cart.n_boxes is not None
                    and not view.pick_cart.box_can_mix_orders and
                    view.occupied_bins is not None and
                    view.occupied_bins + len(added) > view.pick_cart.n_boxes):
                violations.append(_violation(
                    ViolationCode.PHYSICAL,
                    "Added orders exceed free cart boxes", tour_id))
    return tuple(violations)
