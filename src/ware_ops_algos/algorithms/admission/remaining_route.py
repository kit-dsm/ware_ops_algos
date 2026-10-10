"""Original simulator's active-tour admission rule, independent of CASIM."""

from ware_ops_algos.algorithms.algorithm_interfaces import (
    AdmissionInput, AdmissionSolution, Algorithm,
)


class RemainingRouteAdmission(Algorithm[AdmissionInput, AdmissionSolution]):
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
