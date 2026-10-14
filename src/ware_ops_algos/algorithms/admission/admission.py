"""Admission base and both active-tour admission algorithms."""

from abc import ABC

from ..algorithm_interfaces import AdmissionInput, AdmissionSolution, Algorithm
from ..batching.batching_utils import CapacityChecker


class Admission(Algorithm[AdmissionInput, AdmissionSolution], ABC):
    """Assign newly visible orders to running tours."""


class CenterPositionAdmission(Admission):
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


class RemainingRouteAdmission(Admission):
    """Greedily assign visible orders to tours that can still pick them."""

    algo_name = "RemainingRouteAdmission"

    def _run(self, input_data: AdmissionInput) -> AdmissionSolution:
        by_id = {order.order_id: order for order in input_data.orders}
        if len(by_id) != len(input_data.orders):
            raise ValueError("Admission input contains duplicate order IDs")
        if not input_data.candidate_order_ids <= by_id.keys():
            raise ValueError("A candidate order was not resolved")

        tours = sorted(input_data.tours, key=lambda tour: tour.tour_id)
        free_bins = {}
        remaining = {}
        for tour in tours:
            cart = tour.pick_cart
            free_bins[tour.tour_id] = cart.n_boxes - tour.occupied_bins
            remaining[tour.tour_id] = frozenset(tour.remaining_route)

        candidates = sorted(
            (by_id[order_id] for order_id in input_data.candidate_order_ids),
            key=lambda order: (order.order_date or 0.0, order.order_id),
        )
        assignments = []
        considered = []
        for order in candidates:
            if not order.pick_positions:
                raise ValueError(f"Candidate order {order.order_id} has no resolved picks")
            for tour in tours:
                pair = (tour.tour_id, order.order_id)
                if (tour.picking_until is not None or
                        pair in input_data.considered_pairs or
                        order.order_id in tour.active_order_ids):
                    continue
                considered.append(pair)
                if (free_bins[tour.tour_id] > 0 and
                        all(pick.pick_node in remaining[tour.tour_id]
                            for pick in order.pick_positions)):
                    assignments.append(pair)
                    free_bins[tour.tour_id] -= 1
                    break
        return AdmissionSolution(
            assignments=tuple(assignments),
            considered_pairs=tuple(considered),
        )
