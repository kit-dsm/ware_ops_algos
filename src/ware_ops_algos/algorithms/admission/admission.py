"""Shared contract for admission to an active tour."""

from abc import ABC

from ..algorithm_interfaces import AdmissionInput, AdmissionSolution, Algorithm


class Admission(Algorithm[AdmissionInput, AdmissionSolution], ABC):
    """Assign newly visible orders to running tours."""
