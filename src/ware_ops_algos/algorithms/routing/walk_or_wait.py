"""Paper-style S-shape routing and its active-tour variant."""

from __future__ import annotations

from .routing import SShapeRouting, ReroutableRouting


class WalkOrWaitSShapeRouting(SShapeRouting):
    """S-shape traversal used by the 2_Stochastic_Waiting simulator."""

    algo_name = "WalkOrWaitSShapeRouting"

    def _walk_to_target_and_pick(self, source: tuple, target_y: list, walking_upwards: bool = None) -> tuple:
        for next_position in target_y:
            source = self._walk_to_target(source, (source[0], next_position), target_is_pick_node=True)

        if walking_upwards is not None:
            last_position = self.max_aisle_position if walking_upwards else 1
            source = self._walk_to_target(source, (source[0], last_position))
        return source

    def _process_aisle(self, current_source: tuple, aisle_to_visit: int, walking_up: bool = None) -> tuple:
        if current_source[0] == aisle_to_visit:
            return super()._process_aisle(current_source, aisle_to_visit, walking_up)
        return self._walk_to_target(current_source, (aisle_to_visit, current_source[1]))


class ReroutableWalkOrWaitSShapeRouting(ReroutableRouting):
    """Paper-style S-shape traversal with active rerouting."""

    algo_name = "ReroutableWalkOrWaitSShapeRouting"
    route_class = WalkOrWaitSShapeRouting
