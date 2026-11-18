from types import SimpleNamespace

from ware_ops_algos.algorithms import (
    ActiveTourRoutingSolution, BatchingSolution, SchedulingSolution,
)
from ware_ops_algos.planning import (
    Accepts, PlanView, TourStates, TourView, to_commands,
)
from ware_ops_algos.planning.commands import (
    CancelTour, CreatePickList, CreateTour, ReplaceSuffix, Reschedule,
)


def _job(order_ids, picker=1, start=10, end=20):
    batch = SimpleNamespace(order_numbers=frozenset(order_ids))
    route = SimpleNamespace(annotated_route=(), distance=0, batch=batch)
    return SimpleNamespace(
        job=SimpleNamespace(route=route),
        order_numbers=frozenset(order_ids), picker_id=picker,
        start_time=start, end_time=end,
    )


def _tour(tour_id, order_ids):
    job = _job(order_ids)
    return TourView(tour_id=tour_id, picker_id=job.picker_id,
                    status=TourStates.PLANNED, accepts=Accepts.REBUILD,
                    batch=job.job.route.batch, route=job.job.route,
                    start_time=job.start_time, end_time=job.end_time)


def test_batching_creates_pick_lists():
    batch = SimpleNamespace(order_numbers=frozenset((1,)))
    assert to_commands(BatchingSolution(batches=[batch]), PlanView(time=0)) == (
        CreatePickList(batch),)


def test_schedule_matches_largest_overlap_and_rebuilds_unmatched():
    plan = PlanView(time=0, tours=(_tour(2, (1, 2)), _tour(1, (3,))))
    jobs = [_job((1, 2, 4)), _job((5,))]
    commands = to_commands(SchedulingSolution(jobs=jobs), plan)
    assert commands[0] == CancelTour(1)
    assert commands[1] == ReplaceSuffix(2, jobs[0].job.route, (4,))
    assert commands[2] == CreateTour(jobs[1])


def test_schedule_reschedules_matched_tour():
    plan = PlanView(time=0, tours=(_tour(7, (1,)),))
    job = _job((1,), picker=2, start=30, end=40)
    assert to_commands(SchedulingSolution(jobs=[job]), plan) == (
        Reschedule(7, 2, 30, 40),)


def test_admission_without_assignments_has_no_commands():
    solution = ActiveTourRoutingSolution(
        routes=[], assignments=(), considered_pairs=((1, 3),))
    assert to_commands(solution, PlanView(time=12, tours=(_tour(1, (1,)),))) == ()


def _scheduled(tour_id, order_ids, accepts=Accepts.REBUILD):
    view = _tour(tour_id, order_ids)
    return TourView(tour_id=view.tour_id, picker_id=view.picker_id,
                    status=TourStates.SCHEDULED, accepts=accepts, batch=view.batch,
                    route=view.route, start_time=view.start_time, end_time=view.end_time)


def test_a_rebuildable_scheduled_tour_is_revised_not_duplicated():
    # The pipelines receive every tour whose scope gives REBUILD, scheduled ones
    # included; the comparison must use the same set.
    plan = PlanView(time=0, tours=(_scheduled(8, (2,)),))
    job = _job((2,), picker=3, start=30, end=40)
    assert to_commands(SchedulingSolution(jobs=[job]), plan) == (Reschedule(8, 3, 30, 40),)


def test_a_rebuildable_scheduled_tour_missing_from_the_solution_is_cancelled():
    plan = PlanView(time=0, tours=(_scheduled(8, (2,)), _tour(9, (3,))))
    assert to_commands(SchedulingSolution(jobs=[_job((3,))]), plan) == (CancelTour(8),)


def test_a_frozen_scheduled_tour_is_not_compared():
    plan = PlanView(time=0, tours=(_scheduled(8, (2,), accepts=Accepts.NONE),))
    job = _job((5,))
    assert to_commands(SchedulingSolution(jobs=[job]), plan) == (CreateTour(job),)


def test_a_candidate_with_exactly_a_tours_orders_revises_that_tour():
    # Tour 1 holds order 1. Candidate 0 holds orders 1 and 2, candidate 1 only
    # order 1: both overlap tour 1 by one order, and the exact candidate wins.
    plan = PlanView(time=0, tours=(_tour(1, (1,)),))
    jobs = [_job((1, 2)), _job((1,))]
    assert to_commands(SchedulingSolution(jobs=jobs), plan) == (CreateTour(jobs[0]),)
