"""Warehouse waiting algorithms."""

from ..algorithm_interfaces import WaitingInput
from .waiting import Waiting, NoWaiting, OrderCountWaiting, HennWaiting, AnalyticStochasticWaiting
