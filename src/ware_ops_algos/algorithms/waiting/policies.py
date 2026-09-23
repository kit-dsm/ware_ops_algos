"""Release policies using only information available at decision time."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ware_ops_algos.algorithms.algorithm_interfaces import (
    Algorithm, Job, ScheduledJob, WaitingSolution,
)
from ware_ops_algos.algorithms.routing.patrol import all_aisles_patrol_route
from ware_ops_algos.domain_models import LayoutData, Resource, WarehouseInfo

from .analytic_progress.optimal_waiting import solve_optimal_wait
from .analytic_progress.models import WarehouseInstance


def _warehouse_instance(layout: LayoutData, picker: Resource, orders) -> WarehouseInstance:
    """Translate visible orders into the existing single-line analytical model."""
    params = layout.graph_data
    positions = [
        pick.pick_node
        for order in orders
        for pick in order.pick_positions
        for _ in range(int(pick.in_store or pick.amount))
    ]
    aisle_counts = Counter(int(aisle) for aisle, _ in positions)
    aisles = sorted(aisle_counts)
    return WarehouseInstance(
        M=int(params.n_aisles),
        N_L=int(params.n_pick_locations),
        w=float(params.dist_aisle),
        L=float(params.dist_pick_locations) * params.n_pick_locations,
        A=aisles,
        n_list=[aisle_counts[aisle] for aisle in aisles],
        v=float(picker.speed),
        t_p=float(picker.time_per_pick),
        P=[(int(aisle), int(y)) for aisle, y in positions],
        arrival_times=[float(order.order_date) for order in orders],
    )


@dataclass(frozen=True)
class WaitingInput:
    candidates: tuple[ScheduledJob, ...]
    current_time: float
    input_closed: bool
    picker: Resource
    layout: LayoutData
    warehouse_info: WarehouseInfo | None
    deadline_reached: bool = False
    single_order_service_times: dict[int, float] | None = None


class NoWaiting(Algorithm[WaitingInput, WaitingSolution]):
    algo_name = "NoWaiting"

    def _run(self, data: WaitingInput) -> WaitingSolution:
        return WaitingSolution(jobs=data.candidates)


class OrderCountWaiting(Algorithm[WaitingInput, WaitingSolution]):
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


class StartImmediatelyWaiting(Algorithm[WaitingInput, WaitingSolution]):
    """Release visible work, or start an empty aisle patrol while the stream is open."""

    algo_name = "StartImmediatelyWaiting"

    def _run(self, data: WaitingInput) -> WaitingSolution:
        if data.candidates:
            return WaitingSolution(jobs=(data.candidates[0],))
        if data.input_closed:
            return WaitingSolution(action="wait")
        route = all_aisles_patrol_route(data.layout)
        duration = route.distance / data.picker.speed
        job = Job(
            job_id=-1,
            processing_time=duration,
            release_time=data.current_time,
            due_date=float("inf"),
            n_picks=0,
            route=route,
            batch=route.batch,
        )
        return WaitingSolution(jobs=(ScheduledJob(
            job=job,
            picker_id=data.picker.id,
            start_time=data.current_time,
            end_time=data.current_time + duration,
        ),))


class HennWaiting(Algorithm[WaitingInput, WaitingSolution]):
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


class AnalyticStochasticWaiting(Algorithm[WaitingInput, WaitingSolution]):
    """Causal single-line exponential-arrival policy; no realised future order."""

    algo_name = "AnalyticStochasticWaiting"

    def __init__(self, target_batch_size_orders: int = 4):
        super().__init__()
        self.target_batch_size_orders = target_batch_size_orders

    def _run(self, data: WaitingInput) -> WaitingSolution:
        if len(data.candidates) != 1:
            raise ValueError("Analytical waiting requires one scheduled candidate")
        job = data.candidates[0]
        orders = job.job.route.batch.orders
        if data.input_closed or data.deadline_reached or len(orders) >= self.target_batch_size_orders:
            return WaitingSolution(jobs=(job,))
        if len(orders) != self.target_batch_size_orders - 1:
            raise ValueError(
                "Analytical waiting requires exactly q-1 known orders before the missing order"
            )
        info = data.warehouse_info
        if (info.arrival_process != "exponential" or
                info.order_lines_per_order != 1 or
                info.pick_location_distribution != "uniform_iid" or
                info.mean_interarrival_time_s is None):
            raise ValueError("Unsupported analytical waiting forecast")
        if any(len(order.pick_positions) != 1 for order in orders):
            raise ValueError("Analytical waiting supports one line per order")
        instance = _warehouse_instance(data.layout, data.picker, orders)
        result = solve_optimal_wait(
            instance,
            mean_interarrival_time=info.mean_interarrival_time_s,
            remaining_arrivals_after_miss=self.target_batch_size_orders - 1,
        )
        if result.wait_duration <= 1e-9:
            return WaitingSolution(jobs=(job,))
        return WaitingSolution(action="wait", reconsider_at=data.current_time + result.wait_duration)
