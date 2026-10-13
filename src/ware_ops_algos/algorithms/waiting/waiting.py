"""Shared contract for warehouse waiting algorithms."""

from abc import ABC

from ..algorithm_interfaces import Algorithm, WaitingInput, WaitingSolution


class Waiting(Algorithm[WaitingInput, WaitingSolution], ABC):
    """Decide when a scheduled candidate may start."""
