"""Active batch membership under the original simulator's remaining-route rule."""

from ware_ops_algos.algorithms.algorithm_interfaces import (
    BatchObject, BatchingSolution, WarehouseOrder,
)
from ware_ops_algos.domain_models import Articles, PickCart

from .batching import Batching


class RemainingRouteAdmission(Batching):
    """Admit an arrived order only if a free bin and its unserved pick nodes remain."""

    algo_name = "RemainingRouteAdmission"

    def __init__(self, pick_cart: PickCart, articles: Articles, *,
                 active_order_ids: frozenset[int], candidate_order_ids: frozenset[int],
                 remaining_route: tuple[tuple[float, float], ...], occupied_bins: int):
        super().__init__(pick_cart, articles)
        self.active_order_ids = active_order_ids
        self.candidate_order_ids = candidate_order_ids
        self.remaining_route = frozenset(remaining_route)
        self.occupied_bins = occupied_bins

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        if self.pick_cart.box_can_mix_orders:
            raise ValueError("Remaining-route admission requires one cart bin per order")
        capacity = self.pick_cart.n_boxes or len(self.pick_cart.capacities or [])
        if not 0 <= self.occupied_bins <= capacity:
            raise ValueError("Invalid active cart occupancy")
        by_id = {order.order_id: order for order in input_data}
        if not self.candidate_order_ids <= by_id.keys():
            raise ValueError("A newly arrived order was not resolved")
        active = [order for order in input_data if order.order_id in self.active_order_ids]
        candidates = sorted(
            (by_id[order_id] for order_id in self.candidate_order_ids),
            key=lambda order: (order.order_date, order.order_id),
        )
        accepted = []
        for order in candidates:
            if not order.pick_positions:
                raise ValueError(f"Candidate order {order.order_id} has no resolved picks")
            if self.occupied_bins + len(accepted) >= capacity:
                break
            if all(pick.pick_node in self.remaining_route for pick in order.pick_positions):
                accepted.append(order)
        if not accepted:
            return BatchingSolution(batches=[])
        return BatchingSolution(batches=[BatchObject(batch_id=0, orders=active + accepted)])
