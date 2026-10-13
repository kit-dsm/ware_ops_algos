"""Convert algorithm solutions into simulator-independent plan edits."""

from __future__ import annotations

from functools import singledispatch

from ware_ops_algos.algorithms.algorithm_interfaces import (
    ActiveTourRoutingSolution, BatchingSolution, CombinedRoutingSolution,
    SchedulingSolution, WaitingSolution,
)

from .commands import (
    CancelTour, Command, CreatePickList, CreateTour, ReplaceSuffix, Reschedule,
)
from .lifecycle import Accepts, TourStates
from .models import PlanView, TourView


def _same_route(view: TourView, route) -> bool:
    return (view.route is not None
            and tuple(view.route.annotated_route) == tuple(route.annotated_route)
            and view.route.distance == route.distance
            and view.route.batch.order_numbers == route.batch.order_numbers)


def _matching_candidates(jobs, drafts: tuple[TourView, ...]):
    """Pair by largest overlap, then lowest incumbent ID and candidate index."""
    pairs = []
    for index, job in enumerate(jobs):
        for view in drafts:
            overlap = len(view.batch.order_numbers & job.order_numbers)
            if overlap and view.batch.order_numbers <= job.order_numbers:
                pairs.append((-overlap, view.tour_id, index, view))
    pairs.sort(key=lambda pair: pair[:3])
    matched_indices = {}
    matched_ids = set()
    for _, tour_id, index, view in pairs:
        if index not in matched_indices and tour_id not in matched_ids:
            matched_indices[index] = view
            matched_ids.add(tour_id)
    return matched_indices, matched_ids


def _schedule_commands(jobs, plan: PlanView) -> tuple[Command, ...]:
    jobs = tuple(jobs)
    drafts = tuple(tour for tour in plan.tours
                   if tour.status is TourStates.PLANNED
                   and tour.accepts is Accepts.REBUILD
                   and tour.batch is not None)
    matches, matched_ids = _matching_candidates(jobs, drafts)
    commands: list[Command] = [CancelTour(tour.tour_id) for tour in drafts
                               if tour.tour_id not in matched_ids]
    for index, job in enumerate(jobs):
        view = matches.get(index)
        if view is None:
            commands.append(CreateTour(job))
            continue
        if not _same_route(view, job.job.route):
            added = tuple(sorted(job.order_numbers - view.batch.order_numbers))
            commands.append(ReplaceSuffix(view.tour_id, job.job.route, added))
        if (view.picker_id != job.picker_id or view.start_time != job.start_time
                or view.end_time != job.end_time):
            commands.append(Reschedule(view.tour_id, job.picker_id,
                                       job.start_time, job.end_time))
    return tuple(commands)


@singledispatch
def to_commands(solution, plan: PlanView) -> tuple[Command, ...]:
    """Return the delta a candidate solution proposes for the visible plan."""
    raise TypeError(f"No command conversion for {type(solution).__name__}")


@to_commands.register
def _(solution: BatchingSolution, plan: PlanView) -> tuple[Command, ...]:
    return tuple(CreatePickList(batch) for batch in solution.batches)


@to_commands.register
def _(solution: SchedulingSolution, plan: PlanView) -> tuple[Command, ...]:
    return _schedule_commands(solution.jobs, plan)


@to_commands.register
def _(solution: WaitingSolution, plan: PlanView) -> tuple[Command, ...]:
    return _schedule_commands(solution.jobs, plan)


@to_commands.register
def _(solution: CombinedRoutingSolution, plan: PlanView) -> tuple[Command, ...]:
    return ()


@to_commands.register
def _(solution: ActiveTourRoutingSolution, plan: PlanView) -> tuple[Command, ...]:
    views = {tour.tour_id: tour for tour in plan.tours}
    admitted: dict[int, set[int]] = {}
    for tour_id, order_id in solution.assignments:
        admitted.setdefault(tour_id, set()).add(order_id)
    return tuple(ReplaceSuffix(
        route.batch.tour_id, route,
        tuple(sorted(admitted.get(route.batch.tour_id, set())
                     | (route.batch.order_numbers
                        - views[route.batch.tour_id].batch.order_numbers))),
    ) for route in solution.routes)


@singledispatch
def decision_effects(solution) -> tuple[bool, float | None, frozenset[tuple[int, int]]]:
    """Return waiting and considered-admission bookkeeping for a solution."""
    return False, None, frozenset()


@decision_effects.register
def _(solution: WaitingSolution) -> tuple[bool, float | None, frozenset[tuple[int, int]]]:
    return solution.action == "wait", solution.reconsider_at, frozenset()


@decision_effects.register
def _(solution: ActiveTourRoutingSolution) -> tuple[bool, float | None, frozenset[tuple[int, int]]]:
    return False, None, frozenset(solution.considered_pairs)
