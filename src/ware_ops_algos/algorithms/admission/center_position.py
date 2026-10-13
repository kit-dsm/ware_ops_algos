"""Admit work to an empty cart waiting at a center node."""

from ..algorithm_interfaces import AdmissionInput, AdmissionSolution, Algorithm
from ..batching.batching_utils import CapacityChecker


class CenterPositionAdmission(Algorithm[AdmissionInput, AdmissionSolution]):
    algo_name = "CenterPositionAdmission"

    def __init__(self, articles):
        super().__init__()
        self.articles = articles

    def _run(self, input_data: AdmissionInput) -> AdmissionSolution:
        orders = {order.order_id: order for order in input_data.orders}
        if len(orders) != len(input_data.orders):
            raise ValueError("Admission input contains duplicate order IDs")
        assignments = []
        considered = []
        reserved: set[int] = set()
        for tour in sorted(input_data.tours, key=lambda item: item.tour_id):
            if tour.picking_until is not None or tour.active_order_ids:
                continue
            checker = CapacityChecker(tour.pick_cart, self.articles)
            accepted = []
            for order_id in sorted(input_data.candidate_order_ids):
                if order_id in reserved or (tour.tour_id, order_id) in input_data.considered_pairs:
                    continue
                order = orders[order_id]
                pair = (tour.tour_id, order_id)
                considered.append(pair)
                if order.pick_positions and checker.can_add_order(accepted, order):
                    assignments.append(pair)
                    accepted.append(order)
                    reserved.add(order_id)
        return AdmissionSolution(assignments=tuple(assignments),
                                 considered_pairs=tuple(considered))
