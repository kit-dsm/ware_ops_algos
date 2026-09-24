"""Original simulator's active-tour admission rule, independent of CASIM."""

from ware_ops_algos.algorithms.algorithm_interfaces import (
    AdmissionInput, AdmissionSolution, Algorithm,
)


class RemainingRouteAdmission(Algorithm[AdmissionInput, AdmissionSolution]):
    """Accept visible arrivals with a free bin and every pick still ahead."""

    algo_name = "RemainingRouteAdmission"

    def _run(self, input_data: AdmissionInput) -> AdmissionSolution:
        if input_data.pick_cart.box_can_mix_orders:
            raise ValueError("Remaining-route admission requires one cart bin per order")
        capacity = input_data.pick_cart.n_boxes or len(input_data.pick_cart.capacities or [])
        if not 0 <= input_data.occupied_bins <= capacity:
            raise ValueError("Invalid active cart occupancy")
        by_id = {order.order_id: order for order in input_data.orders}
        if len(by_id) != len(input_data.orders):
            raise ValueError("Admission input contains duplicate order IDs")
        if input_data.active_order_ids & input_data.candidate_order_ids:
            raise ValueError("Active and candidate order IDs overlap")
        if not (input_data.active_order_ids | input_data.candidate_order_ids) <= by_id.keys():
            raise ValueError("An active or candidate order was not resolved")

        remaining = frozenset(input_data.remaining_route)
        candidates = sorted(
            (by_id[order_id] for order_id in input_data.candidate_order_ids),
            key=lambda order: (order.order_date, order.order_id),
        )
        accepted = []
        for order in candidates:
            if not order.pick_positions:
                raise ValueError(f"Candidate order {order.order_id} has no resolved picks")
            if input_data.occupied_bins + len(accepted) >= capacity:
                break
            if all(pick.pick_node in remaining for pick in order.pick_positions):
                accepted.append(order.order_id)
        return AdmissionSolution(accepted_order_ids=tuple(accepted))
