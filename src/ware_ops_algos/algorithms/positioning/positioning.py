"""Shared empty-cart positioning mechanics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ware_ops_algos.domain_models import LayoutData

from ..algorithm_interfaces import (
    Algorithm, Job, PositioningInput, PositioningSolution, Route, ScheduledJob,
)


class Positioning(Algorithm[PositioningInput, PositioningSolution], ABC):
    """Add ordinary empty-cart jobs to the proposed schedule."""

    mode: str

    @abstractmethod
    def _route(self, layout: LayoutData) -> Route:
        """Choose the route of an empty-cart tour."""

    def _run(self, input_data: PositioningInput) -> PositioningSolution:
        jobs = list(input_data.schedule.jobs)
        modes: dict[int, str] = {}
        if input_data.input_closed:
            return PositioningSolution(jobs=jobs, modes=modes)
        assigned = {job.picker_id for job in jobs} | set(input_data.occupied_picker_ids)
        for picker in input_data.resources:
            if picker.id in assigned or picker.speed <= 0:
                continue
            route = self._route(input_data.layout)
            duration = route.distance / picker.speed
            job = Job(job_id=-picker.id - 1, processing_time=duration,
                      release_time=input_data.current_time,
                      due_date=float("inf"), n_picks=0,
                      route=route, batch=route.batch)
            jobs.append(ScheduledJob(job=job, picker_id=picker.id,
                                     start_time=input_data.current_time,
                                     end_time=input_data.current_time + duration))
            modes[job.job_id] = self.mode
        return PositioningSolution(jobs=jobs, modes=modes)
