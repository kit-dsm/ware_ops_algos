"""Exhaustive proofs over the finite lifecycle and scope domains."""

from itertools import combinations

import pytest

from ware_ops_algos.planning.lifecycle import (
    Accepts, PHYSICAL, TOUR_ORDER, TOUR_TRANSITIONS, TourStates,
    accepts_at_least,
)
from ware_ops_algos.planning.scope import RevisionScope, StateInformation
from ware_ops_algos.planning.validation import check_scope


def _subsets(values):
    for size in range(len(values) + 1):
        for chosen in combinations(values, size):
            yield frozenset(chosen)


def test_lifecycle_is_forward_reachable_and_terminal():
    assert set(TOUR_ORDER) == set(TourStates)
    assert set(TOUR_TRANSITIONS) == set(TourStates)
    for source, targets in TOUR_TRANSITIONS.items():
        for target in targets:
            assert TOUR_ORDER.index(target) > TOUR_ORDER.index(source)
    reached = {TourStates.PLANNED}
    frontier = [TourStates.PLANNED]
    while frontier:
        for target in TOUR_TRANSITIONS[frontier.pop()]:
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    assert reached == set(TourStates)
    assert {status for status, targets in TOUR_TRANSITIONS.items() if not targets} == {
        TourStates.DONE, TourStates.CANCELLED,
    }


def test_physical_rights_never_increase():
    assert set(PHYSICAL) == set(TourStates)
    assert tuple(Accepts) == (Accepts.NONE, Accepts.INSERT, Accepts.REBUILD)
    for earlier, later in zip(TOUR_ORDER, TOUR_ORDER[1:]):
        assert accepts_at_least(PHYSICAL[earlier], PHYSICAL[later])


def test_scope_check_is_exhaustive_over_exposed_status_sets():
    for freeze_from in TourStates:
        for frozen_accepts in (Accepts.NONE, Accepts.INSERT):
            scope = RevisionScope(freeze_from, frozen_accepts)
            levels = [scope.accepts(status)
                      for status in TOUR_ORDER]
            assert all(accepts_at_least(a, b)
                       for a, b in zip(levels, levels[1:]))
            for exposed in _subsets(tuple(TourStates)):
                information = StateInformation(tours=exposed)
                expected = tuple(status for status in TourStates
                                 if status in exposed and not accepts_at_least(
                                     PHYSICAL[status],
                                     scope.accepts(status)))
                assert check_scope(information, scope) == expected
    for freeze_from in TourStates:
        with pytest.raises(ValueError, match="frozen_accepts"):
            RevisionScope(freeze_from, Accepts.REBUILD)
