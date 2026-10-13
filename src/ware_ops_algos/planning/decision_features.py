"""Algorithm-card features implied by a decision's visible plan and rights."""

from .lifecycle import Accepts, TourStates, accepts_for
from .scope import RevisionScope, StateInformation


def decision_features(state_information: StateInformation,
                      scope: RevisionScope) -> dict[str, bool | str]:
    def allows(status: TourStates, level: Accepts) -> bool:
        return (status in state_information.tours and
                accepts_for(status, scope.freeze_from,
                            scope.frozen_accepts) is level)

    incumbent = (allows(TourStates.PLANNED, Accepts.REBUILD) or
                 allows(TourStates.SCHEDULED, Accepts.REBUILD))
    live = (state_information.orders and
            allows(TourStates.STARTED, Accepts.INSERT))
    return {"incumbent": incumbent,
            "routing_origin": "live" if live else "depot"}
