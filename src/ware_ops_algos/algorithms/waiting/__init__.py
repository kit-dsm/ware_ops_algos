"""Waiting decisions share one input and output."""

from ..algorithm_interfaces import WaitingInput
from .policies import NoWaiting, OrderCountWaiting, StartImmediatelyWaiting, HennWaiting, AnalyticStochasticWaiting
