from ware_ops_algos.algorithms import BatchObject, Route
from ware_ops_algos.domain_models import TourPlanningState, TourStates


def test_tour_state_works_after_package_import():
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
