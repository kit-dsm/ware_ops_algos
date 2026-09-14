"""Stochastic waiting optimizer with MDP-based optimal stopping.

Implements the ``Algorithm`` interface.  Takes ``WaitingAnalysisInput``
(built from ware_ops_algos domain objects) and returns a
``WaitingSolution``.  Internally adapts to the analytic_progress
computation engine.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from ware_ops_algos.algorithms import (
    Algorithm,
    Route,
    WarehouseOrder,
)
from ware_ops_algos.domain_models import LayoutData, Resource

from .waiting_analysis import WaitingAnalysisInput, WaitingSolution
from .analytic_progress.models import (
    BatchComparisonResult,
    BatchWindow,
    OrderInput,
    WarehouseInstance,
)
from .analytic_progress.order_optimizer_discrete import (
    analyze_batch_window as _analyze_discrete,
)
from .analytic_progress.order_optimizer_continuous import (
    analyze_batch_window as _analyze_continuous,
)


def _build_warehouse_instance(
    layout: LayoutData,
    picker: Resource,
    all_orders: list[WarehouseOrder],
) -> WarehouseInstance:
    """Build the S-shape analysis context from domain objects.

    The ``WarehouseInstance`` is a batch-specific mathematical description:
    ``M``/``N_L``/``w``/``L`` are warehouse-wide (from layout), ``v``/``t_p``
    are picker-specific, and ``A``/``n_list``/``P``/``arrival_times`` are
    batch-specific (from the orders' pick positions).
    """
    params = layout.graph_data
    M = int(params.n_aisles)
    N_L = int(params.n_pick_locations)
    w = float(params.dist_aisle)
    L = float(params.dist_pick_locations) * N_L
    v = float(picker.speed)
    t_p = float(picker.time_per_pick)

    pick_positions: list[tuple[int, int]] = []
    arrival_times: list[float] = []
    for order in all_orders:
        for pp in order.pick_positions:
            qty = int(pp.in_store) if pp.in_store else int(pp.amount)
            aisle, y = pp.pick_node
            pick_positions.extend([(int(aisle), int(y))] * qty)
        arrival_times.append(float(order.order_date or 0.0))

    if not pick_positions:
        raise ValueError("Cannot build WarehouseInstance: no pick positions")

    aisle_counts: Counter[int] = Counter(a for a, _ in pick_positions)
    A = sorted(aisle_counts)
    n_list = [aisle_counts[a] for a in A]

    return WarehouseInstance(
        M=M,
        N_L=N_L,
        w=w,
        L=L,
        A=A,
        n_list=n_list,
        v=v,
        t_p=t_p,
        P=pick_positions,
        arrival_times=arrival_times,
    )


def _build_batch_window(
    base_orders: list[WarehouseOrder],
    insert_order: WarehouseOrder,
) -> BatchWindow:
    """Convert domain orders to the analytic_progress BatchWindow."""
    base_inputs = [
        OrderInput(
            order_id=int(o.order_id),
            items={
                str(pp.article_id): int(pp.in_store) if pp.in_store else int(pp.amount)
                for pp in o.pick_positions
            },
            arrival_time=float(o.order_date or 0.0),
            due_date=float(o.due_date) if o.due_date is not None else None,
        )
        for o in base_orders
    ]
    insert_input = OrderInput(
        order_id=int(insert_order.order_id),
        items={
            str(pp.article_id): int(pp.in_store) if pp.in_store else int(pp.amount)
            for pp in insert_order.pick_positions
        },
        arrival_time=float(insert_order.order_date or 0.0),
        due_date=float(insert_order.due_date)
        if insert_order.due_date is not None
        else None,
    )
    return BatchWindow(
        batch_index=1,
        base_orders=base_inputs,
        insert_order=insert_input,
    )


def _build_article_mapping(
    base_orders: list[WarehouseOrder],
    insert_order: WarehouseOrder,
) -> dict[str, tuple[int, int]]:
    """Build article_id -> (aisle, y) mapping from PickPosition data."""
    mapping: dict[str, tuple[int, int]] = {}
    for order in [*base_orders, insert_order]:
        for pp in order.pick_positions:
            key = str(pp.article_id)
            if key not in mapping:
                mapping[key] = (int(pp.pick_node[0]), int(pp.pick_node[1]))
    return mapping


def _to_waiting_solution(
    result: BatchComparisonResult,
    engine: str,
) -> WaitingSolution:
    """Convert the analytic_progress result to a WaitingSolution."""
    return WaitingSolution(
        should_wait=bool(result.phase1_should_integrate),
        best_integration_time=float(result.phase1_best_integration_t),
        predicted_completion_time=float(result.phase1_predicted_completion_time),
        mean_predicted_order_completion_time=float(
            result.samples[0].mean_predicted_order_completion_time
            if result.samples
            else 0.0
        ),
        initial_completion_time=float(result.samples[0].initial_completion_time
            if result.samples
            else 0.0),
        mean_initial_order_completion_time=float(
            result.samples[0].mean_initial_order_completion_time
            if result.samples
            else 0.0
        ),
        expected_detour=float(result.phase1_predicted_best_detour),
        actual_waiting_time=float(result.actual_waiting_time),
        engine=engine,
    )


class StochasticWaitingOptimizer(Algorithm[WaitingAnalysisInput, WaitingSolution]):
    """MDP-based optimal-stopping waiting policy for S-shape routing.

    Estimates the expected detour from a continuously arriving order and
    decides whether to wait for it or dispatch immediately.  Only applicable
    to conventional single-block layouts with S-shape routing and a single
    human picker.

    Parameters
    ----------
    engine
        Computation engine: "discrete" (exact S-shape positions) or
        "continuous" (continuous position approximation).
    expected_interarrival
        Expected time between order arrivals in seconds.
    """

    algo_name = "StochasticWaiting"

    def __init__(
        self,
        engine: Literal["discrete", "continuous"] = "discrete",
        expected_interarrival: float = 28.8,
    ):
        super().__init__()
        self.engine = engine
        self.expected_interarrival = float(expected_interarrival)

    def _run(self, input_data: WaitingAnalysisInput) -> WaitingSolution:
        all_orders = [*input_data.base_orders, input_data.insert_order]
        instance = _build_warehouse_instance(
            input_data.layout,
            input_data.picker,
            all_orders,
        )
        window = _build_batch_window(
            input_data.base_orders,
            input_data.insert_order,
        )
        article_mapping = _build_article_mapping(
            input_data.base_orders,
            input_data.insert_order,
        )

        from .analytic_progress.models import EXPECTED_INTERARRIVAL_TIME
        from .analytic_progress import optimizer_common
        optimizer_common.EXPECTED_INTERARRIVAL = self.expected_interarrival

        engine_fn = _analyze_discrete if self.engine == "discrete" else _analyze_continuous
        result = engine_fn(
            window,
            article_mapping,
            M=instance.M,
            N_L=instance.N_L,
            w=instance.w,
            L=instance.L,
            v=instance.v,
            t_p=instance.t_p,
        )
        return _to_waiting_solution(result, self.engine)
