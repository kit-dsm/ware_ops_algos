from itertools import combinations

from ware_ops_algos.planning.decision_features import decision_features
from ware_ops_algos.planning.lifecycle import Accepts, TourStates, accepts_for
from ware_ops_algos.planning.scope import RevisionScope, StateInformation
from ware_ops_algos.planning.validation import check_scope


def test_features_cover_all_admissible_scope_and_exposure_combinations():
    statuses = tuple(TourStates)
    for freeze_from in statuses:
        for frozen in (Accepts.NONE, Accepts.INSERT):
            scope = RevisionScope(freeze_from, frozen)
            for size in range(len(statuses) + 1):
                for exposed in combinations(statuses, size):
                    for orders in (False, True):
                        information = StateInformation(
                            orders=orders, tours=frozenset(exposed))
                        if check_scope(information, scope):
                            continue
                        features = decision_features(information, scope)
                        assert features["incumbent"] == any(
                            status in information.tours and
                            accepts_for(status, freeze_from, frozen) is Accepts.REBUILD
                            for status in (TourStates.PLANNED,
                                           TourStates.SCHEDULED))
                        assert features["routing_origin"] == (
                            "live" if orders and TourStates.STARTED in exposed and
                            accepts_for(TourStates.STARTED, freeze_from, frozen)
                            is Accepts.INSERT else "depot")
