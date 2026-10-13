"""Waiting decisions share one input and output."""

# ruff: noqa: F401
from ..algorithm_interfaces import WaitingInput
from .waiting import Waiting
from .policies import NoWaiting, OrderCountWaiting, HennWaiting, AnalyticStochasticWaiting
