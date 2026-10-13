"""Algorithms implementing a subproblem share its typed base."""

from ware_ops_algos.algorithms import (
    Admission, AnalyticStochasticWaiting, CenterPositionAdmission,
    HennWaiting, NoWaiting, OrderCountWaiting, RemainingRouteAdmission,
    Waiting,
)


def test_waiting_algorithms_share_the_waiting_contract():
    for implementation in (NoWaiting, OrderCountWaiting, HennWaiting,
                           AnalyticStochasticWaiting):
        assert issubclass(implementation, Waiting)


def test_admission_algorithms_share_the_admission_contract():
    for implementation in (RemainingRouteAdmission, CenterPositionAdmission):
        assert issubclass(implementation, Admission)
