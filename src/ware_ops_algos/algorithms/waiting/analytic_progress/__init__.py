"""Analytic-progress stochastic waiting optimizer.

Stateless S-shape route analysis with MDP-based optimal stopping
and continuous detour approximation.  Ported from project_4D4L.
"""

from __future__ import annotations

from .models import (
    EXPECTED_INTERARRIVAL_TIME,
    BatchComparisonResult,
    BatchWindow,
    OrderInput,
    TimeSampleResult,
    WarehouseInstance,
)
from .optimizer_common import (
    EngineName,
    OptimizerEngine,
    build_batch_windows,
    create_warehouse_instance_from_orders,
    expanded_pick_nodes_for_order,
    load_article_mapping,
    load_orders,
    order_pick_service_time,
)
from .mdp import phase_1_discrete
from .order_optimizer import analyze_batch_window, analyze_batches
from .interval_builder import IntervalData, build_interval_data
from .optimal_waiting import (
    AnalyticWaitingResult,
    critical_mean_interarrival_time,
    expected_waiting_cost,
    solve_optimal_wait,
)

__all__ = [
    "EXPECTED_INTERARRIVAL_TIME",
    "BatchComparisonResult",
    "BatchWindow",
    "OrderInput",
    "TimeSampleResult",
    "WarehouseInstance",
    "EngineName",
    "OptimizerEngine",
    "build_batch_windows",
    "create_warehouse_instance_from_orders",
    "expanded_pick_nodes_for_order",
    "load_article_mapping",
    "load_orders",
    "order_pick_service_time",
    "phase_1_discrete",
    "analyze_batch_window",
    "analyze_batches",
    "IntervalData",
    "build_interval_data",
    "AnalyticWaitingResult",
    "critical_mean_interarrival_time",
    "expected_waiting_cost",
    "solve_optimal_wait",
]
