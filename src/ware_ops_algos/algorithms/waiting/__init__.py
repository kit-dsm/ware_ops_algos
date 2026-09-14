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

__all__ = [
    "HennDecision",
    "decide_henn",
    "route_service_time",
]
