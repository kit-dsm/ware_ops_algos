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
