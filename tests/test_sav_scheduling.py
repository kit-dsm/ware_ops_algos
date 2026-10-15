"""Henn's savings selection prefers a batch that avoids single-order work."""

from types import SimpleNamespace

from ware_ops_algos.algorithms.algorithm_interfaces import (
    BatchObject, Job, PickPosition, Route, WarehouseOrder,
)
from ware_ops_algos.algorithms.scheduling.scheduling import SAVScheduling
from ware_ops_algos.domain_models import Resource, Resources, ResourceType


class SingleOrderRouter:
    def reset_parameters(self):
        pass

    def solve(self, _positions):
        return SimpleNamespace(route=SimpleNamespace(distance=10.0))


def test_sav_selects_greater_single_order_saving_first():
    picker = Resource(id=0, speed=1.0, time_per_pick=0.0,
                      tour_setup_time=0.0)
    orders = [WarehouseOrder(order_id=i, order_date=0.0,
                             pick_positions=(PickPosition(
                                 order_number=i, article_id=i, amount=1,
                                 pick_node=(i, 0), in_store=1),))
              for i in range(3)]
    routes = [Route(distance=10.0, batch=BatchObject(0, orders[:1]),
                    item_sequence=[0]),
              Route(distance=12.0, batch=BatchObject(1, orders[1:]),
                    item_sequence=[1, 2])]
    jobs = [Job(job_id=i, processing_time=route.distance, release_time=0.0,
                due_date=float("inf"), n_picks=len(route.item_sequence), route=route)
            for i, route in enumerate(routes)]
    scheduler = SAVScheduling(Resources(ResourceType.HUMAN, [picker]),
                              router=SingleOrderRouter())
    scheduled = scheduler._schedule(jobs)
    assert [item.job.job_id for item in scheduled] == [1, 0]
