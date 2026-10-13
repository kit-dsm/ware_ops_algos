"""Henn waiting receives a whole schedule and uses its configured router."""

from types import SimpleNamespace

from ware_ops_algos.algorithms import (
    BatchObject, HennWaiting, Job, PickPosition, Route, ScheduledJob,
    WaitingInput, WarehouseOrder,
)
from ware_ops_algos.domain_models import Resource


class Router:
    def __init__(self):
        self.calls = 0

    def reset_parameters(self):
        pass

    def solve(self, picks):
        self.calls += 1
        return SimpleNamespace(route=SimpleNamespace(distance=4))


def _job(order_id, start):
    order = WarehouseOrder(
        order_id=order_id, order_date=10,
        pick_positions=(PickPosition(order_id, 1, 1, (1, 1), 1),),
    )
    batch = BatchObject(order_id, [order])
    route = Route(distance=4, batch=batch)
    job = Job(order_id, 5, 10, 100, 1, route=route, batch=batch)
    return ScheduledJob(job, picker_id=0, start_time=start, end_time=start + 7)


def _input(jobs, now=15):
    return WaitingInput(
        candidates=tuple(jobs), current_time=now, input_closed=False,
        picker=Resource(id=0, speed=1, time_per_pick=1, tour_setup_time=2),
        layout=None, information=None,
    )


def test_henn_threshold_and_schedule_delay_use_injected_router():
    router = Router()
    waiter = HennWaiting(router)
    result = waiter.solve(_input([_job(1, 15)]))
    assert router.calls == 1
    assert result.action == "wait"
    assert result.reconsider_at == 20
    assert [(job.start_time, job.end_time) for job in result.jobs] == [(20, 27)]


def test_multiple_candidates_keep_the_whole_schedule():
    router = Router()
    jobs = (_job(1, 15), _job(2, 22))
    result = HennWaiting(router).solve(_input(jobs))
    assert result.action == "release"
    assert result.jobs == jobs
    assert router.calls == 0
