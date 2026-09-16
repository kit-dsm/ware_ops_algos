"""Domain-facing types for stochastic waiting analysis.

These types bridge ware_ops_algos domain objects to the analytic_progress
computation engine.  ``WaitingAnalysisInput`` is built from domain objects
(``LayoutData``, ``Resource``, ``WarehouseOrder``, ``Route``); the optimizer
returns a ``WaitingAnalysisSolution``. This retrospective analysis is kept
separate from the causal ``WaitingSolution`` release contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from ware_ops_algos.algorithms import (
    AlgorithmSolution,
    WarehouseOrder,
)
from ware_ops_algos.domain_models import LayoutData, Resource


@dataclass
class WaitingAnalysisInput:
    """Input for stochastic waiting analysis, built from domain objects.

    The analyzer retrospectively compares the known ``insert_order`` with the
    ``base_orders``. It does not represent an online release decision.

    Parameters
    ----------
    layout
        Warehouse layout (must be conventional, single-block).
    picker
        The idle picker (must have speed and time_per_pick).
    base_orders
        Already-routed orders in the current batch.
    insert_order
        Newly arrived order to potentially integrate.
    The historical evaluator retains project_4D4L's fixed 28.8-second
    interarrival assumption.
    """

    layout: LayoutData
    picker: Resource
    base_orders: list[WarehouseOrder]
    insert_order: WarehouseOrder


@dataclass
class WaitingAnalysisSolution(AlgorithmSolution):
    """Result of retrospective insert-order analysis.

    Attributes
    ----------
    should_wait
        True if the picker should wait for the insert order; False to
        dispatch immediately.
    best_integration_time
        Route time at which the insert order should be integrated
        (seconds from route start), or None if dispatching.
    predicted_completion_time
        Predicted completion time if the insert order is integrated.
    mean_predicted_order_completion_time
        Mean predicted order completion time with integration.
    initial_completion_time
        Completion time without integration (dispatch now).
    mean_initial_order_completion_time
        Mean order completion time without integration.
    expected_detour
        Expected additional route time from integrating the insert order.
    actual_waiting_time
        Time the picker should wait before starting (seconds).
    engine
        Computation engine used: "discrete" or "continuous".
    """

    should_wait: bool = False
    best_integration_time: float | None = None
    predicted_completion_time: float = 0.0
    mean_predicted_order_completion_time: float = 0.0
    initial_completion_time: float = 0.0
    mean_initial_order_completion_time: float = 0.0
    expected_detour: float = 0.0
    actual_waiting_time: float = 0.0
    engine: str = "discrete"
