"""Pure checks of revision scope and proposed plan changes."""

from __future__ import annotations

from .lifecycle import PHYSICAL, TourStates, accepts_at_least, accepts_for
from .scope import RevisionScope, StateInformation


def check_scope(state_information: StateInformation,
                scope: RevisionScope) -> tuple[TourStates, ...]:
    """Statuses whose configured revision level exceeds their physical bound."""
    return tuple(status for status in TourStates
                 if status in state_information.tours and not accepts_at_least(
                     PHYSICAL[status],
                     accepts_for(status, scope.freeze_from, scope.frozen_accepts)))
