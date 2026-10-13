"""CASOP's six-job scheduling case keeps its end-time objectives after setup moves."""

import pytest

from ware_ops_algos.algorithms.algorithm_interfaces import BatchObject, Job, Route
from ware_ops_algos.algorithms.scheduling.scheduling import (
    EDDScheduling, EDDThenSPTScheduling, ERDScheduling, FIFOScheduling,
    LPTScheduling, MDDScheduling, MSScheduling, SPTScheduling,
)
from ware_ops_algos.domain_models import Resource, Resources
from ware_ops_algos.domain_models.resources.models import ResourceType


# (job id, picker id, end, lateness, tardiness), recorded before Task 4.8.
EDD = ((2, 1, 10, 0, 0), (6, 2, 15, 1, 1), (1, 1, 21, 6, 6),
       (5, 2, 24, 6, 6), (4, 1, 31, 6, 6), (3, 2, 35, 5, 5))
ERD = ((1, 1, 13, -2, 0), (2, 2, 14, 4, 4), (3, 1, 22, -8, 0),
       (4, 2, 26, 1, 1), (6, 1, 29, 15, 15), (5, 2, 35, 17, 17))
FIFO = ((1, 1, 13, -2, 0), (2, 2, 14, 4, 4), (3, 1, 22, -8, 0),
        (4, 2, 26, 1, 1), (5, 1, 29, 11, 11), (6, 2, 35, 21, 21))
LPT = ((1, 1, 13, -2, 0), (4, 2, 16, -9, 0), (3, 1, 22, -8, 0),
       (2, 2, 26, 16, 16), (5, 1, 29, 11, 11), (6, 2, 35, 21, 21))
MSS = ((2, 1, 10, 0, 0), (1, 2, 17, 2, 2), (6, 1, 17, 3, 3),
       (5, 1, 24, 6, 6), (4, 2, 29, 4, 4), (3, 1, 33, 3, 3))
SPT = ((5, 1, 17, -1, 0), (6, 2, 15, 1, 1), (2, 2, 25, 15, 15),
       (3, 1, 26, -4, 0), (4, 2, 37, 12, 12), (1, 1, 37, 22, 22))


@pytest.mark.parametrize("scheduler_class,expected", [
    (EDDScheduling, EDD), (EDDThenSPTScheduling, EDD),
    (ERDScheduling, ERD), (FIFOScheduling, FIFO),
    (LPTScheduling, LPT), (MDDScheduling, EDD),
    (MSScheduling, MSS), (SPTScheduling, SPT),
])
def test_casop_end_times_and_objectives_unchanged(scheduler_class, expected):
    specs = ((1, 8, 0, 15), (2, 5, 0, 10), (3, 6, 0, 30),
             (4, 7, 0, 25), (5, 4, 10, 18), (6, 4, 6, 14))
    jobs = [Job(job_id, duration, release, due, 0,
                route=Route(duration, item_sequence=[],
                            batch=BatchObject(job_id, [])))
            for job_id, duration, release, due in specs]
    resources = Resources(ResourceType.HUMAN, [
        Resource(id=1, speed=1, time_per_pick=0, tour_setup_time=3),
        Resource(id=2, speed=1, time_per_pick=0, tour_setup_time=5),
    ])
    scheduled = scheduler_class(resources, available_from={1: 2, 2: 4}).solve(jobs).jobs
    assert tuple((entry.job.job_id, entry.picker_id, entry.end_time,
                  entry.lateness, entry.tardiness) for entry in scheduled) == expected

    available = {1: 2, 2: 4}
    setup = {1: 3, 2: 5}
    for entry in scheduled:
        # start_time marks the beginning of the setup, including available_from.
        assert entry.start_time == max(entry.job.release_time,
                                       available[entry.picker_id])
        assert entry.end_time == (entry.start_time + setup[entry.picker_id]
                                  + entry.job.processing_time)
        available[entry.picker_id] = entry.end_time
