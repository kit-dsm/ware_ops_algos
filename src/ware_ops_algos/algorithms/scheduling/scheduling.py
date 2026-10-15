import heapq
from abc import abstractmethod
from dataclasses import replace

from ware_ops_algos.algorithms import Algorithm
from ware_ops_algos.algorithms.algorithm_interfaces import (
    Route,
    SchedulingSolution,
    Job,
    ScheduledJob,
)
from ware_ops_algos.domain_models import Resource, Resources


def _processing_time(route: Route, picker: Resource) -> float:
    if picker.speed is None or picker.speed <= 0:
        raise ValueError(f"Resource {picker.id} needs a positive 'speed'.")
    if picker.time_per_pick is None:
        raise ValueError(f"Resource {picker.id} needs 'time_per_pick'.")
    # setup = picker.tour_setup_time
    return (route.distance / picker.speed) + picker.time_per_pick * len(route.item_sequence) #+ setup

def build_jobs(
    routes: list[Route],
    resources: Resources,
    release_time: float | None = None,
) -> list[Job]:
    ref_picker = resources.resources[0]
    return [
        Job(
            job_id=i,
            route=route,
            processing_time=_processing_time(route, ref_picker),
            release_time=route.arrival_time if release_time is None else release_time,
            due_date=route.earliest_due_date,
            n_picks=len(route.item_sequence)
        )
        for i, route in enumerate(routes)
    ]


def earliest_free_assignment(
    ordered_jobs: list[Job],
    resources: Resources,
    available_from: dict[int, float] | None = None,
) -> list[ScheduledJob]:
    heap: list[tuple[float, int, Resource]] = [
        ((available_from or {}).get(r.id, 0.0), i, r)
        for i, r in enumerate(resources.resources)
    ]
    heapq.heapify(heap)

    scheduled: list[ScheduledJob] = []
    for job in ordered_jobs:
        avail, ridx, picker = heapq.heappop(heap)
        start = max(job.release_time, avail)
        actual_job = replace(job, processing_time=_processing_time(job.route, picker))
        end = start + picker.tour_setup_time + actual_job.processing_time
        scheduled.append(ScheduledJob(
            job=actual_job,
            picker_id=picker.id,
            start_time=start,
            end_time=end,
        ))
        heapq.heappush(heap, (end, ridx, picker))
    return scheduled


class Scheduler(Algorithm[list[Job], SchedulingSolution]):
    def __init__(self, resources: Resources,
                 available_from: dict[int, float] | None = None, **kwargs):
        super().__init__(**kwargs)
        if not resources.resources:
            raise ValueError("Scheduler requires at least one resource.")
        self.resources = resources
        self.available_from = available_from or {}

    def _run(self, input_data: list[Job]) -> SchedulingSolution:
        scheduled = self._schedule(input_data)
        return SchedulingSolution(jobs=scheduled, algo_name=self.algo_name)

    @abstractmethod
    def _schedule(self, jobs: list[Job]) -> list[ScheduledJob]:
        ...


class PriorityScheduler(Scheduler):
    def _schedule(self, jobs: list[Job]) -> list[ScheduledJob]:
        ordered = sorted(jobs, key=self._priority)
        return earliest_free_assignment(ordered, self.resources,
                                        self.available_from)

    @abstractmethod
    def _priority(self, job: Job) -> float:
        ...


class SPTScheduling(PriorityScheduler):
    algo_name = "SPTScheduler"
    def _priority(self, job: Job) -> float:
        return job.processing_time


class LPTScheduling(PriorityScheduler):
    algo_name = "LPTScheduler"
    def _priority(self, job: Job) -> float:
        return -job.processing_time


class EDDScheduling(PriorityScheduler):
    algo_name = "EDDScheduler"
    def _priority(self, job: Job) -> float:
        return job.due_date


class ERDScheduling(PriorityScheduler):
    algo_name = "ERDScheduler"
    def _priority(self, job: Job) -> float:
        return job.release_time


class FIFOScheduling(PriorityScheduler):
    algo_name = "FIFOScheduler"
    def _priority(self, job: Job) -> float:
        return job.job_id


class SAVScheduling(Scheduler):
    """Henn's batch selection by sum of single-order services saved."""

    algo_name = "SAVScheduler"

    def __init__(self, resources: Resources, router,
                 available_from: dict[int, float] | None = None):
        super().__init__(resources, available_from=available_from)
        if len(resources.resources) != 1:
            raise ValueError("SAV selection is defined here for one picker")
        self.router = router

    def _schedule(self, jobs: list[Job]) -> list[ScheduledJob]:
        picker = self.resources.resources[0]
        services = {}
        for job in jobs:
            for order in job.route.batch.orders:
                if order.order_id in services:
                    continue
                self.router.reset_parameters()
                route = self.router.solve(order.pick_positions).route
                services[order.order_id] = (
                    picker.tour_setup_time + route.distance / picker.speed
                    + sum(pick.in_store for pick in order.pick_positions)
                    * picker.time_per_pick)

        def priority(job: Job):
            saved = (sum(services[order.order_id]
                         for order in job.route.batch.orders)
                     - job.processing_time - picker.tour_setup_time)
            return (-saved, job.job_id)

        return earliest_free_assignment(sorted(jobs, key=priority),
                                        self.resources, self.available_from)


class MSScheduling(PriorityScheduler):
    """Minimum Slack: due_date - processing_time. Urgency weighted by workload."""
    algo_name = "MSScheduler"
    def _priority(self, job: Job) -> float:
        return job.due_date - job.processing_time


class MDDScheduling(PriorityScheduler):
    """Modified Due Date: behaves like EDD when slack exists, SPT-ish when late."""
    algo_name = "MDDScheduler"
    def _priority(self, job: Job) -> float:
        return max(job.due_date, job.release_time + job.processing_time)


class EDDThenSPTScheduling(PriorityScheduler):
    """EDD with SPT tie-break."""
    algo_name = "EDDThenSPTScheduler"
    def _priority(self, job: Job) -> tuple[float, float]:
        return (job.due_date, job.processing_time)
