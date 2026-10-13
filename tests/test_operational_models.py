from ware_ops_algos.algorithms import BatchObject, Route
from ware_ops_algos.domain_models import PlanEntry, TourPlanningState, TourStates


def test_tour_state_and_plan_entry_work_after_package_import():
    batch = BatchObject(batch_id=1, orders=[])
    route = Route(distance=0.0, batch=batch, annotated_route=[])
    tour = TourPlanningState(
        tour_id=1,
        order_numbers=[],
        original_route=route,
        batch=batch,
        annotated_route=[],
    )

    assert tour.status == TourStates.PLANNED
    assert PlanEntry(id=1, decision=batch).batch is batch
    assert PlanEntry(id=2, decision=route).batch is batch
