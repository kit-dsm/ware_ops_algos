"""Release policies using only information available at decision time."""

from __future__ import annotations

from ware_ops_algos.algorithms.algorithm_interfaces import WaitingInput, WaitingSolution
from ware_ops_algos.domain_models import ExponentialSingleLineUniformLocationOrderStream

from .analytic_progress.optimal_waiting import solve_optimal_wait
from .waiting import Waiting


class NoWaiting(Waiting):
    algo_name = "NoWaiting"

    def _run(self, data: WaitingInput) -> WaitingSolution:
        return WaitingSolution(jobs=data.candidates)


class OrderCountWaiting(Waiting):
    """Release the first FCFS batch once ``min_orders`` are present (wait-k)."""

    algo_name = "OrderCountWaiting"

    def __init__(self, min_orders: int):
        super().__init__()
        if min_orders < 1:
            raise ValueError("wait-0 requires an initial empty tour, not OrderCountWaiting")
        self.min_orders = min_orders

    def _run(self, data: WaitingInput) -> WaitingSolution:
        if not data.candidates:
            return WaitingSolution(action="wait")
        first = data.candidates[0]
        count = len(first.job.route.batch.orders)
        if data.input_closed or count >= self.min_orders:
            return WaitingSolution(jobs=(first,))
        return WaitingSolution(action="wait")


class HennWaiting(Waiting):
    """Henn's release threshold, applied to the first scheduled batch."""

    algo_name = "HennWaiting"

    def _run(self, data: WaitingInput) -> WaitingSolution:
        if not data.candidates:
            return WaitingSolution(jobs=())
        job = data.candidates[0]
        if data.input_closed or len(data.candidates) > 1:
            return WaitingSolution(jobs=(job,))
        services = data.single_order_service_times
        if not services:
            raise ValueError("Henn waiting needs single-order service times")
        orders = job.job.route.batch.orders
        critical = max(orders, key=lambda o: services[o.order_id])
        batch_service = job.job.processing_time + data.picker.tour_setup_time
        threshold = 2 * critical.order_date + services[critical.order_id] - batch_service
        if data.current_time < threshold:
            return WaitingSolution(action="wait", reconsider_at=threshold)
        return WaitingSolution(jobs=(job,))


class AnalyticStochasticWaiting(Waiting):
    """Causal single-line exponential-arrival policy; no realised future order."""

    algo_name = "AnalyticStochasticWaiting"

    def __init__(self, target_batch_size_orders: int = 4):
        super().__init__()
        self.target_batch_size_orders = target_batch_size_orders

    def _run(self, data: WaitingInput) -> WaitingSolution:
        if len(data.candidates) != 1:
            raise ValueError("Analytical waiting requires one scheduled candidate")
        job = data.candidates[0]
        if data.information is None:
            raise ValueError("Analytical waiting requires planner information")
        orders = job.job.route.batch.orders
        if (data.input_closed or data.deadline_reached
                or len(orders) >= self.target_batch_size_orders):
            return WaitingSolution(jobs=(job,))
        if len(orders) < self.target_batch_size_orders - 1:
            return WaitingSolution(action="wait")
        if len(orders) != self.target_batch_size_orders - 1:
            raise ValueError(
                "Analytical waiting received more than q-1 known orders"
            )
        if any(len(order.pick_positions) != 1 for order in orders):
            raise ValueError("Analytical waiting supports one line per order")
        order_stream = data.information.require(
            "incoming_orders", ExponentialSingleLineUniformLocationOrderStream
        )
        result = solve_optimal_wait(
            data,
            mean_interarrival_time=order_stream.mean_interarrival_time_s,
            remaining_arrivals_after_miss=self.target_batch_size_orders - 1,
        )
        if result.wait_duration <= 1e-9:
            return WaitingSolution(jobs=(job,))
        return WaitingSolution(action="wait", reconsider_at=data.current_time + result.wait_duration)
