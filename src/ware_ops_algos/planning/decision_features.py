"""Algorithm-card features implied by a decision's visible plan and rights."""

from .lifecycle import Accepts, TourStates
from .scope import RevisionScope, StateInformation


def decision_features(state_information: StateInformation,
                      scope: RevisionScope) -> dict[str, bool | str]:
    def allows(status: TourStates, level: Accepts) -> bool:
        return (status in state_information.tours and
                scope.accepts(status) is level)

    incumbent = (allows(TourStates.PLANNED, Accepts.REBUILD) or
                 allows(TourStates.SCHEDULED, Accepts.REBUILD))
    live = (state_information.orders and
            allows(TourStates.STARTED, Accepts.INSERT))
    return {"incumbent": incumbent,
            "routing_origin": "live" if live else "depot"}
