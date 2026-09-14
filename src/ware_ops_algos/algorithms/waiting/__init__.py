"""Waiting policies for online order batching.

Provides deterministic Henn threshold policies and stochastic
analytic-progress optimisation with MDP-based optimal stopping.
"""

from __future__ import annotations

from .henn import (
    HennDecision,
    decide_henn,
    route_service_time,
)
from .waiting_analysis import (
    WaitingAnalysisInput,
    WaitingSolution,
)
from .stochastic_waiting import StochasticWaitingOptimizer

__all__ = [
    "HennDecision",
    "decide_henn",
    "route_service_time",
    "WaitingAnalysisInput",
    "WaitingSolution",
    "StochasticWaitingOptimizer",
]
