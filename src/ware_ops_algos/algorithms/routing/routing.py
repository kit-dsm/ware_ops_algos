from typing import NamedTuple, Optional
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from gurobipy import GRB
from matplotlib import pyplot as plt
import gurobipy as gp

from ware_ops_algos.algorithms.algorithm_interfaces import Algorithm, RoutingSolution, Route, PickPosition, RouteNode, NodeType, \
    CombinedRoutingSolution
from ware_ops_algos.domain_models import Resource, OrderPosition, Article, StorageLocations
from ware_ops_algos.algorithms.routing.dynamic_programming_helpers import (
    equivalence_classes,
    cross_aisle_mapping,
    table_I,
    table_II,
    aisle_mapping,
    )




class Routing(Algorithm[list[PickPosition] | list[OrderPosition], RoutingSolution], ABC):
    def __init__(self,
                 start_node: tuple[int, int],
                 end_node: tuple[int, int],
                 closest_node_to_start: tuple[int, int],
                 min_aisle_position: int,
                 max_aisle_position: int,
                 picker: list[Resource],
                 gen_tour: bool = False,
                 gen_item_sequence: bool = False,
                 node_list: list[tuple[float, float]] = None,
                 node_to_idx: dict = None,
                 idx_to_node: dict = None,
                 distance_matrix: pd.DataFrame | None = None,
        predecessor_matrix: np.array = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.start_node = start_node
        self.end_node = end_node
        self.closest_node_to_start = closest_node_to_start
        self.min_aisle_position = min_aisle_position
        self.max_aisle_position = max_aisle_position

        self.distance_matrix = distance_matrix
        if self.distance_matrix is not None:
            self._dist_array = distance_matrix.values
            self._node_to_idx = {node: idx for idx, node in enumerate(distance_matrix.index)}
        self.predecessor_matrix = predecessor_matrix
        self.node_list: list[tuple[float, float]] = node_list
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.gen_item_sequence = gen_item_sequence
        self.gen_tour = gen_tour
        self.picker = picker

    @abstractmethod
    def _run(self, input_data: list[PickPosition]) -> RoutingSolution | CombinedRoutingSolution:
        """Concrete routing algorithms implement this and return a Route result."""
        ...

    def reset_parameters(self):
        """Compatibility hook for pipeline callers; state is solve-local."""
        return None

    def score(self, input_data: list[PickPosition] | list[OrderPosition]) -> float:
        """Return only the routing objective.

        Exact and specialised routers may use a full solve.  Constructive
        routers override this with their allocation-light computation path.
        """
        return self.solve(input_data).route.distance

    def _get_distance(self, source, target) -> float:
        """Fast distance lookup."""
        return self._dist_array[self._node_to_idx[source], self._node_to_idx[target]]

    def _get_aisle_entry_points(self) -> dict:
        """Find the entry point (min y) for each aisle."""
        aisles = defaultdict(list)
        for x, y in self.node_list:
            aisles[x].append(y)
        return {aisle: (aisle, min(ys)) for aisle, ys in aisles.items()}

    def _get_route_segment(self, source, target, with_last_element: bool = False):
        """Expand one shortest-path segment without mutating the router."""
        source_idx = self.node_to_idx[source]
        target_idx = self.node_to_idx[target]

        path_indices = []
        current_idx = target_idx

        while current_idx != source_idx:
            # path_indices.insert(0, current_idx)
            path_indices.append(current_idx)
            current_idx = self.predecessor_matrix[source_idx, current_idx]

            if current_idx == -9999:
                raise ValueError(f"No path from {source} to {target}")

        # path_indices.insert(0, source_idx)
        path_indices.append(source_idx)
        path_indices.reverse()

        # Convert back to node names
        path = [self.idx_to_node[idx] for idx in path_indices]
        path_nodes = [RouteNode(self.idx_to_node[idx], NodeType.ROUTE)
                      for idx in path_indices]
        if not with_last_element:
            path = path[:-1]
            path_nodes = path_nodes[:-1]
        return path, path_nodes


class HeuristicRouting(Routing, ABC):
    """Base class for heuristic routing algorithms."""

    def __init__(self,
                 start_node: tuple[int, int],
                 end_node: tuple[int, int],
                 closest_node_to_start: tuple[int, int],
                 min_aisle_position: int,
                 max_aisle_position: int,
                 distance_matrix,
                 predecessor_matrix,
                 picker,
                 fixed_depot=True,
                 **kwargs):

        super().__init__(start_node=start_node, end_node=end_node, closest_node_to_start=closest_node_to_start,
                         min_aisle_position=min_aisle_position, max_aisle_position=max_aisle_position,
                         picker=picker, distance_matrix=distance_matrix, predecessor_matrix=predecessor_matrix, **kwargs)
        self.fixed_depot = fixed_depot

    def score(self, input_data: list[PickPosition] | list[OrderPosition]) -> float:
        visit_nodes = self._compute_visit_order(
            [position.pick_node for position in input_data]
        )
        return self._distance_of_visit_order(visit_nodes)

    def _run(self, input_data: list[PickPosition]) -> RoutingSolution:
        pick_nodes = [position.pick_node for position in input_data]
        visit_nodes = self._compute_visit_order(pick_nodes)
        distance = self._distance_of_visit_order(visit_nodes)
        node_to_picks = self._group_semantic_picks(input_data) if self.gen_item_sequence else {}
        route, item_sequence, annotated = self._materialize_output(
            visit_nodes, node_to_picks
        )
        return RoutingSolution(
            algo_name=self.algo_name,
            route=Route(
                distance=distance,
                route=route,
                item_sequence=item_sequence,
                annotated_route=annotated,
            ),
        )

    @abstractmethod
    def _compute_visit_order(self, pick_nodes: list[tuple]) -> list[tuple]:
        """Return algorithm decision nodes, including start and final node."""
        ...

    @staticmethod
    def _group_aisle_nodes(pick_nodes: list[tuple]) -> dict:
        grouped = defaultdict(set)
        for x, y in pick_nodes:
            grouped[x].add(y)
        return {x: sorted(ys) for x, ys in grouped.items()}

    @staticmethod
    def _group_semantic_picks(picks) -> dict:
        grouped = defaultdict(list)
        for pick in picks:
            grouped[pick.pick_node].append(pick)
        return dict(grouped)

    def _distance_of_visit_order(self, visit_nodes: list[tuple]) -> float:
        return float(sum(
            self._get_distance(source, target)
            for source, target in zip(visit_nodes, visit_nodes[1:])
        ))

    def _materialize_output(self, visit_nodes, node_to_picks):
        item_sequence = []
        annotated = []
        route = []
        picked_nodes = set()

        if self.gen_tour:
            for index, (source, target) in enumerate(zip(visit_nodes, visit_nodes[1:])):
                path, path_nodes = self._get_route_segment(
                    source,
                    target,
                    with_last_element=index == len(visit_nodes) - 2,
                )
                route.extend(path)
                annotated.extend(path_nodes)
                if (
                    self.gen_item_sequence
                    and target in node_to_picks
                    and target not in picked_nodes
                ):
                    picked_nodes.add(target)
                    for _ in node_to_picks[target]:
                        item_sequence.append(target)
                        annotated.append(RouteNode(target, NodeType.PICK))
        elif self.gen_item_sequence:
            for node in visit_nodes[1:]:
                if node in node_to_picks and node not in picked_nodes:
                    picked_nodes.add(node)
                    for _ in node_to_picks[node]:
                        item_sequence.append(node)
                        annotated.append(RouteNode(node, NodeType.PICK))

        return route, item_sequence, annotated

    def _determine_walking_direction(self, current_source: tuple) -> bool:
        if current_source[1] == self.min_aisle_position:
            return True
        elif current_source[1] == self.max_aisle_position:
            return False
        else:
            raise ValueError(f"Start node is not connected to the beginning or end of the aisle. "
                             f"Min aisle: {self.min_aisle_position}, Max aisle: {self.max_aisle_position},"
                             f"Current source: {current_source}")

    def _end_for(self, current_source: tuple):
        if self.fixed_depot:
            return self.end_node
        return self._get_aisle_entry_points()[current_source[0]]

    def _visit_aisle(self, visit_nodes, remaining, aisle, walking_up):
        current = visit_nodes[-1]
        if current[0] != aisle:
            visit_nodes.append((aisle, current[1]))
            return
        y_values = remaining.pop(aisle)
        visit_nodes.extend(
            (aisle, y) for y in sorted(y_values, reverse=not walking_up)
        )
        if remaining:
            visit_nodes.append((
                aisle,
                self.max_aisle_position if walking_up else self.min_aisle_position,
            ))


class SShapeRouting(HeuristicRouting):
    """Implements S-shape routing."""
    algo_name = "SShapeRouting"

    def __init__(self,
                 start_node: tuple[int, int],
                 end_node: tuple[int, int],
                 closest_node_to_start: tuple[int, int],
                 min_aisle_position: int,
                 max_aisle_position: int,
                 distance_matrix,
                 predecessor_matrix,
                 picker,
                 fixed_depot=True,
                 **kwargs):
        super().__init__(start_node, end_node, closest_node_to_start, min_aisle_position, max_aisle_position,
                         distance_matrix, predecessor_matrix, picker, fixed_depot, **kwargs)

    def _compute_visit_order(self, pick_nodes: list[tuple]) -> list[tuple]:
        remaining = self._group_aisle_nodes(pick_nodes)
        visit_nodes = [self.start_node, self.closest_node_to_start]
        walking_up = not self._determine_walking_direction(visit_nodes[-1])

        while remaining:
            aisle = min(remaining)
            if visit_nodes[-1][0] == aisle:
                walking_up = not walking_up
            self._visit_aisle(visit_nodes, remaining, aisle, walking_up)

        visit_nodes.append(self._end_for(visit_nodes[-1]))
        return visit_nodes


class ReturnRouting(HeuristicRouting):
    """Implements Return routing."""
    algo_name = "ReturnRouting"

    def __init__(self,
                 start_node: tuple[int, int],
                 end_node: tuple[int, int],
                 closest_node_to_start: tuple[int, int],
                 min_aisle_position: int,
                 max_aisle_position: int,
                 distance_matrix,
                 predecessor_matrix,
                 picker,
                 fixed_depot=True,
                 **kwargs):
        super().__init__(start_node, end_node, closest_node_to_start, min_aisle_position, max_aisle_position,
                         distance_matrix, predecessor_matrix, picker, fixed_depot, **kwargs)

    def _compute_visit_order(self, pick_nodes: list[tuple]) -> list[tuple]:
        remaining = self._group_aisle_nodes(pick_nodes)
        visit_nodes = [self.start_node, self.closest_node_to_start]

        while remaining:
            self._visit_aisle(visit_nodes, remaining, min(remaining), True)

        visit_nodes.append(self._end_for(visit_nodes[-1]))
        return visit_nodes


class MidpointRouting(HeuristicRouting):
    """Implements Midpoint routing."""
    algo_name = "MidpointRouting"

    def __init__(self,
                 start_node: tuple[int, int],
                 end_node: tuple[int, int],
                 closest_node_to_start: tuple[int, int],
                 min_aisle_position: int,
                 max_aisle_position: int,
                 distance_matrix,
                 predecessor_matrix,
                 picker,
                 fixed_depot=True,
                 **kwargs):
        super().__init__(start_node, end_node, closest_node_to_start, min_aisle_position, max_aisle_position,
                         distance_matrix, predecessor_matrix, picker, fixed_depot, **kwargs)

    def _compute_visit_order(self, pick_nodes: list[tuple]) -> list[tuple]:
        visit_nodes = [self.start_node, self.closest_node_to_start]
        midpoint = round(self.max_aisle_position / 2)
        lower = [node for node in pick_nodes if node[1] < midpoint]
        upper = [node for node in pick_nodes if node[1] >= midpoint]
        min_upper = min((node[0] for node in upper), default=-99)

        if lower:
            max_lower = max(node[0] for node in lower)
            min_lower = min(node[0] for node in lower)
            if min_lower <= min_upper:
                moved = [node for node in lower if node[0] == min_lower]
                lower = [node for node in lower if node[0] != min_lower]
                upper.extend(moved)

            lower_by_aisle = self._group_aisle_nodes(lower)
            while lower_by_aisle:
                aisle = min(lower_by_aisle)
                walking_up = visit_nodes[-1][0] == max_lower
                self._visit_aisle(
                    visit_nodes, lower_by_aisle, aisle, walking_up,
                )

            upper_by_aisle = self._group_aisle_nodes(upper)
            self._transition_to_upper(visit_nodes, upper_by_aisle, max_lower)
        else:
            upper_by_aisle = self._group_aisle_nodes(upper)

        while upper_by_aisle:
            aisle = max(upper_by_aisle)
            walking_up = aisle != min(upper_by_aisle)
            self._visit_aisle(
                visit_nodes, upper_by_aisle, aisle, walking_up,
            )

        visit_nodes.append(self._end_for(visit_nodes[-1]))
        return visit_nodes

    @staticmethod
    def split_orders_by_pickzone(resolved_positions, mid_point: int) -> tuple[list[PickPosition], list[PickPosition]]:

        lower_half = []
        upper_half = []

        for pos in resolved_positions:
            x, y = pos.pick_node
            if y < mid_point:
                lower_half.append(pos)
            else:
                upper_half.append(pos)

        return lower_half, upper_half

    def _transition_to_upper(self, visit_nodes, upper_by_aisle, max_lower):
        if not upper_by_aisle:
            return
        max_upper = max(upper_by_aisle)
        current = visit_nodes[-1]
        if max_upper < max_lower:
            visit_nodes.append((current[0], self.max_aisle_position))
        elif max_upper == max_lower:
            self._visit_aisle(
                visit_nodes, upper_by_aisle, max_upper, True,
            )
        else:
            visit_nodes.append((current[0], self.min_aisle_position))
            visit_nodes.append((max_upper, self.min_aisle_position))
            visit_nodes.extend(
                (max_upper, y) for y in upper_by_aisle.pop(max_upper)
            )
            if upper_by_aisle:
                visit_nodes.append((max_upper, self.max_aisle_position))


class LargestGapRouting(HeuristicRouting):
    """
    Implements Largest Gap Routing for order picking in a warehouse.
    """
    algo_name = "LargestGapRouting"

    def __init__(self,
                 start_node: tuple[int, int],
                 end_node: tuple[int, int],
                 closest_node_to_start: tuple[int, int],
                 min_aisle_position: int,
                 max_aisle_position: int,
                 distance_matrix,
                 predecessor_matrix,
                 picker,
                 fixed_depot=True,
                 **kwargs):

        super().__init__(start_node, end_node, closest_node_to_start, min_aisle_position, max_aisle_position,
                         distance_matrix, predecessor_matrix, picker, fixed_depot, **kwargs)

    def _compute_visit_order(self, pick_nodes: list[tuple]) -> list[tuple]:
        remaining = self._group_aisle_nodes(pick_nodes)
        visit_nodes = [self.start_node, self.closest_node_to_start]
        if not remaining:
            visit_nodes.append(self._end_for(visit_nodes[-1]))
            return visit_nodes

        aisle_max = max(remaining)
        aisle_list = sorted(remaining)
        if max(node[1] for node in pick_nodes) > self.max_aisle_position / 2:
            aisle_list = aisle_list[1:] if len(aisle_list) >= 2 else aisle_list[:1]

        for aisle in aisle_list:
            current = visit_nodes[-1]
            if current[0] != aisle:
                visit_nodes.append((aisle, current[1]))
            y_values = remaining[aisle]
            if aisle == aisle_max:
                remaining.pop(aisle)
                visit_nodes.extend((aisle, y) for y in y_values)
                if remaining:
                    visit_nodes.append((aisle, self.max_aisle_position))
            else:
                split = self._get_largest_gap_pos_inside_aisle(y_values)
                selected = y_values[:split]
                if selected:
                    visit_nodes.extend((aisle, y) for y in selected)
                    unvisited = y_values[split:]
                    if unvisited:
                        remaining[aisle] = unvisited
                    else:
                        remaining.pop(aisle)
                if remaining:
                    visit_nodes.append((aisle, self.min_aisle_position))

        while remaining:
            aisle = max(remaining)
            if aisle == min(remaining):
                current = visit_nodes[-1]
                visit_nodes.append((aisle, current[1]))
                y_values = remaining.pop(aisle)
                visit_nodes.extend((aisle, y) for y in reversed(y_values))
                if remaining:
                    visit_nodes.append((aisle, self.min_aisle_position))
            if remaining:
                self._visit_aisle(visit_nodes, remaining, aisle, True)

        visit_nodes.append(self._end_for(visit_nodes[-1]))
        return visit_nodes

    def _get_largest_gap_pos_inside_aisle(self, y_values: list) -> int:
        """
        Determines the largest gap between pick locations in a given aisle.

        :param y_values: the Y-values of the pick locations in the aisle

        Returns the position of the largest gap and its size.
        """
        gaps = [min(y_values) - 1]
        gaps.extend([(y_values[i + 1] - y_values[i]) for i in range(0, len(y_values) - 1)])
        gaps.append(self.max_aisle_position - max(y_values))

        pos_largest_gap = gaps.index(max(gaps))

        return pos_largest_gap


class NearestNeighbourhoodRouting(HeuristicRouting):
    """
    A class to perform nearest neighbourhood routing for order picking in a warehouse using Dijkstra's algorithm.
    """
    algo_name = "NearestNeighbourhoodRouting"

    def __init__(self,
                 start_node: tuple[int, int],
                 end_node: tuple[int, int],
                 closest_node_to_start: tuple[int, int],
                 min_aisle_position: int,
                 max_aisle_position: int,
                 distance_matrix,
                 predecessor_matrix,
                 picker,
                 fixed_depot=True,
                 **kwargs):
        super().__init__(start_node, end_node, closest_node_to_start, min_aisle_position, max_aisle_position,
                         distance_matrix, predecessor_matrix, picker, fixed_depot, **kwargs)

    def score(self, input_data: list[PickPosition] | list[OrderPosition]) -> float:
        distance, _ = self._nearest_kernel(
            [position.pick_node for position in input_data],
            record_visits=False,
        )
        return distance

    def _compute_visit_order(self, pick_nodes: list[tuple]) -> list[tuple]:
        _, visit_nodes = self._nearest_kernel(pick_nodes, record_visits=True)
        return visit_nodes

    def _nearest_kernel(self, pick_nodes, record_visits):
        # First occurrence determines deterministic tie order. Node indices are
        # prepared once and kept aligned as candidates are deleted by position.
        candidates = list(dict.fromkeys(pick_nodes))
        candidate_indices = [self._node_to_idx[node] for node in candidates]
        current_node = self.closest_node_to_start
        current_index = self._node_to_idx[current_node]
        distance = float(self._get_distance(self.start_node, current_node))
        visit_nodes = [self.start_node, current_node] if record_visits else None

        while candidate_indices:
            candidate_distances = self._dist_array[
                current_index, candidate_indices
            ]
            best_position = int(candidate_distances.argmin())
            distance += float(candidate_distances[best_position])
            current_index = candidate_indices.pop(best_position)
            current_node = candidates.pop(best_position)
            if record_visits:
                visit_nodes.append(current_node)

        end_node = self._end_for(current_node)
        distance += float(self._get_distance(current_node, end_node))
        if record_visits:
            visit_nodes.append(end_node)
        return distance, visit_nodes


class UShapeRouting(HeuristicRouting):
    """U-shaped routing for ladder shaped layout.

    Rule:
    - enter a physical aisle on the right side at the bottom,
    - pick the right side bottom-up,
    - cross at the top rung,
    - pick the left side top-down,
    - leave the aisle at the bottom.
    """

    algo_name = "UShapeRouting"

    def __init__(
        self,
        start_node: tuple[int, int],
        end_node: tuple[int, int],
        closest_node_to_start: tuple[int, int],
        min_aisle_position: int,
        max_aisle_position: int,
        distance_matrix,
        predecessor_matrix,
        picker,
        fixed_depot=True,
        **kwargs,
    ):
        super().__init__(
            start_node=start_node,
            end_node=end_node,
            closest_node_to_start=closest_node_to_start,
            min_aisle_position=min_aisle_position,
            max_aisle_position=max_aisle_position,
            distance_matrix=distance_matrix,
            predecessor_matrix=predecessor_matrix,
            picker=picker,
            fixed_depot=fixed_depot,
            **kwargs,
        )
        self._aisle_pairs = self._derive_physical_aisle_pairs()

    def _rail_x_values(self) -> list[float]:
        nodes = list(self.distance_matrix.index)

        rail_x_values = []

        for node in nodes:
            if not isinstance(node, tuple) or len(node) != 2:
                continue

            x, y = node

            if y != self.min_aisle_position:
                continue

            if (x, self.max_aisle_position) in self._node_to_idx:
                rail_x_values.append(x)

        return sorted(set(rail_x_values))

    def _derive_physical_aisle_pairs(self) -> tuple[tuple[float, float], ...]:
        rails = self._rail_x_values()

        if len(rails) % 2 != 0:
            raise ValueError(f"Expected an even number of rails, got {len(rails)}: {rails}")

        return tuple(
            (rails[i], rails[i + 1])  # left_x, right_x
            for i in range(0, len(rails), 2)
        )

    def _compute_visit_order(self, pick_nodes: list[tuple]) -> list[tuple]:
        by_rail = self._group_aisle_nodes(pick_nodes)
        visit_nodes = [self.start_node, self.closest_node_to_start]

        if not by_rail:
            visit_nodes.append(self._end_for(visit_nodes[-1]))
            return visit_nodes

        rails_with_picks = set(by_rail)

        for left_x, right_x in self._aisle_pairs:
            if left_x not in rails_with_picks and right_x not in rails_with_picks:
                continue

            right_bottom = (right_x, self.min_aisle_position)
            right_top = (right_x, self.max_aisle_position)
            left_top = (left_x, self.max_aisle_position)
            left_bottom = (left_x, self.min_aisle_position)

            # Enter aisle on the right side.
            visit_nodes.append(right_bottom)

            # Right side: bottom -> top.
            visit_nodes.extend(
                (right_x, y) for y in by_rail.get(right_x, ())
            )

            # U-turn at the top.
            visit_nodes.extend((right_top, left_top))

            # Left side: top -> bottom.
            visit_nodes.extend(
                (left_x, y) for y in reversed(by_rail.get(left_x, ()))
            )

            # Exit at bottom left.
            visit_nodes.append(left_bottom)

        visit_nodes.append(self._end_for(visit_nodes[-1]))
        return visit_nodes


class PickListRouting(HeuristicRouting):
    algo_name = "PickListRouting"

    def __init__(self,
                 start_node: tuple[int, int],
                 end_node: tuple[int, int],
                 closest_node_to_start: tuple[int, int],
                 min_aisle_position: int,
                 max_aisle_position: int,
                 distance_matrix,
                 predecessor_matrix,
                 picker,
                 **kwargs):

        super().__init__(start_node, end_node, closest_node_to_start, min_aisle_position, max_aisle_position,
                         distance_matrix, predecessor_matrix, picker, **kwargs)

    def _compute_visit_order(self, pick_nodes: list[tuple]) -> list[tuple]:
        entry_points = self._get_aisle_entry_points()

        if self.fixed_depot:
            start_node = self.start_node
        else:
            picker_location = self.picker[0].current_location
            if isinstance(picker_location, RouteNode):
                start_node = picker_location.position
            else:
                start_node = picker_location
        if not pick_nodes:
            return [start_node, self._end_for(start_node)]

        first_aisle = pick_nodes[0][0]
        aisle_entry = entry_points[first_aisle]
        unique_picks = list(dict.fromkeys(pick_nodes))
        visit_nodes = [start_node, aisle_entry, *unique_picks]
        visit_nodes.append(self._end_for(visit_nodes[-1]))
        return visit_nodes


class ExactRouting(Routing, ABC):
    """
    Base class for exact routing algorithms.
    """

    def __init__(self,
                 start_node: tuple[int, int], end_node: tuple[int, int], distance_matrix: pd.DataFrame,
                 predecessor_matrix: dict, picker: list[Resource], big_m, set_time_limit, **kwargs):
        super().__init__(start_node, end_node, distance_matrix=distance_matrix, predecessor_matrix=predecessor_matrix, picker=picker,**kwargs)

        self.big_m = big_m
        self.time_limit = set_time_limit
@dataclass
class _ExactTSPWork:
    picks: tuple[PickPosition, ...]
    pick_nodes: list[tuple]
    model: object
    amounts: list[float] | None = None
    weights: list[float] | None = None
    x: object = None
    x_start: object = None
    x_end: object = None
    order: object = None
    completion: object = None


class ExactTSPRouting(ExactRouting):
    """
    Implements the exact routing algorithm for the Traveling Salesman Problem (TSP).
    """

    def __init__(self,
                 start_node: tuple[int, int], end_node: tuple[int, int], distance_matrix: pd.DataFrame,
                 predecessor_matrix: dict, picker: list[Resource], big_m, set_time_limit, **kwargs):
        super().__init__(start_node, end_node, distance_matrix=distance_matrix, predecessor_matrix=predecessor_matrix, picker=picker, big_m=big_m, set_time_limit=set_time_limit, **kwargs)

    def score(self, input_data: list[PickPosition]) -> float:
        distance, _, _ = self._optimize(input_data, materialize=False)
        return distance

    def _run(self, pick_list: list[PickPosition]):
        distance, output, model = self._optimize(
            pick_list,
            materialize=self.gen_tour or self.gen_item_sequence,
        )
        route, item_sequence, annotated = output
        return RoutingSolution(
            algo_name=self.algo_name,
            route=Route(
                route=route,
                annotated_route=annotated,
                item_sequence=item_sequence,
                distance=distance,
            ),
            solver_status=str(model.status),
            objective_value=distance,
            is_optimal=model.status == GRB.OPTIMAL,
        )

    def _optimize(self, pick_list, materialize):
        if not pick_list:
            distance = float(self._get_distance(self.start_node, self.end_node))
            output = self._materialize_exact_order([]) if materialize else ([], [], [])
            return distance, output, _EmptyExactModel()

        model = gp.Model(f"{self.algo_name}")
        model.setParam('OutputFlag', 1)
        if self.time_limit > 0:
            model.setParam('TimeLimit', self.time_limit)
        work = self._make_work(tuple(pick_list), model)
        self._set_decision_variables(work)
        self._set_objective(work)
        self._set_constraints(work)
        model.optimize()
        if not (
            model.status == GRB.OPTIMAL
            or (self.time_limit and model.SolCount > 0)
        ):
            raise RuntimeError(f"Exact routing model status {model.status}")

        distance = float(model.objVal)
        output = ([], [], [])
        if materialize:
            output = self._materialize_exact_order(
                self._extract_order(work)
            )
        return distance, output, model

    def _make_work(self, picks, model):
        return _ExactTSPWork(
            picks=picks,
            pick_nodes=[position.pick_node for position in picks],
            model=model,
        )

    def _set_decision_variables(self, work):
        length = len(work.pick_nodes)
        work.x = work.model.addVars(length, length, vtype=GRB.BINARY, name="x")
        work.x_start = work.model.addVars(length, vtype=GRB.BINARY, name="x0j")
        work.x_end = work.model.addVars(length, vtype=GRB.BINARY, name="xj0")
        work.order = work.model.addVars(length, vtype=GRB.CONTINUOUS, name="T")

    def _set_objective(self, work):
        raise NotImplementedError

    def _set_constraints(self, work):
        raise NotImplementedError

    def _add_constraint_each_node_is_visited_exactly_once(self, work):
        length = len(work.pick_nodes)
        for i in range(length):
            work.model.addConstr(gp.quicksum(work.x[i, j] for j in range(length) if i != j) + work.x_end[i] == 1,
                               name=f"constr1_{i}")
            work.model.addConstr(gp.quicksum(work.x[j, i] for j in range(length) if j != i) + work.x_start[i] == 1,
                               name=f"constr2_{i}")

    def _add_constraint_start_and_end_node_are_visited_once(self, work):
        length = len(work.pick_nodes)
        work.model.addConstr(gp.quicksum(work.x_start[j] for j in range(length)) == 1, name="constr3")
        work.model.addConstr(gp.quicksum(work.x_end[j] for j in range(length)) == 1, name="constr4")

    def _add_subtour_eliminiation_without_time(self, work):
        length = len(work.pick_nodes)
        for i in range(length):
            for j in range(length):
                if i != j:
                    work.model.addConstr(work.order[i] + 1 <= work.order[j] + self.big_m * (1 - work.x[i, j]), name=f"constr5_{i}_{j}")

    def _add_subtour_eliminiation_with_time(self, work, travel_time_matrix: pd.DataFrame):
        length = len(work.pick_nodes)
        for i in range(length):
            for j in range(length):
                if i != j:
                    work.model.addConstr(work.order[i] + travel_time_matrix[work.pick_nodes[i]][work.pick_nodes[j]]
                                       <= work.order[j] + self.big_m * (1 - work.x[i, j]), name=f"constr5_{i}_{j}")

    @staticmethod
    def _extract_order(work):
        length = len(work.pick_nodes)
        current = next(i for i in range(length) if work.x_start[i].X > 0.5)
        order = [work.pick_nodes[current]]
        visited = {current}
        while len(visited) < length:
            next_index = next(
                (
                    j for j in range(length)
                    if j not in visited and work.x[current, j].X > 0.5
                ),
                None,
            )
            if next_index is None:
                break
            current = next_index
            visited.add(current)
            order.append(work.pick_nodes[current])
        return order

    def _materialize_exact_order(self, pick_order):
        item_sequence = list(pick_order) if self.gen_item_sequence else []
        annotated = (
            [RouteNode(node, NodeType.PICK) for node in pick_order]
            if self.gen_item_sequence and not self.gen_tour
            else []
        )
        route = []
        if self.gen_tour:
            visit_nodes = [self.start_node, *pick_order, self.end_node]
            annotated = []
            for index, (source, target) in enumerate(zip(visit_nodes, visit_nodes[1:])):
                path, path_nodes = self._get_route_segment(
                    source, target, index == len(visit_nodes) - 2,
                )
                route.extend(path)
                annotated.extend(path_nodes)
                if self.gen_item_sequence and target in pick_order:
                    annotated.append(RouteNode(target, NodeType.PICK))
        return route, item_sequence, annotated


class _EmptyExactModel:
    status = GRB.OPTIMAL

class ExactTSPRoutingDistance(ExactTSPRouting):
    """
    Implements the exact routing algorithm for the Traveling Salesman Problem (TSP) using distance as the objective.
    """
    algo_name = "ExactTSPRoutingDistance"

    def __init__(self,
                 start_node: tuple[int, int], end_node: tuple[int, int], distance_matrix: pd.DataFrame,
                 predecessor_matrix: np.array, picker: list[Resource], gen_tour, gen_item_sequence, big_m=1000,
                 set_time_limit=300, **kwargs):
        super().__init__(start_node, end_node, distance_matrix=distance_matrix, predecessor_matrix=predecessor_matrix,
                         picker=picker, gen_tour=gen_tour, gen_item_sequence=gen_item_sequence, big_m=big_m, set_time_limit=set_time_limit, **kwargs)

    def _set_objective(self, work):
        """Set the objective function for the exact routing model."""
        length = len(work.pick_nodes)
        dist_x_i_x_j = gp.quicksum(self.distance_matrix.at[work.pick_nodes[i], work.pick_nodes[j]] * work.x[i, j]
                                   for i in range(length) for j in range(length) if i != j)
        dist_start_i = gp.quicksum(self.distance_matrix.at[self.start_node, work.pick_nodes[j]] * work.x_start[j]
                                  for j in range(length))
        dist_end_j = gp.quicksum(self.distance_matrix.at[work.pick_nodes[j], self.end_node] * work.x_end[j]
                                for j in range(length))
        work.model.setObjective(dist_x_i_x_j + dist_start_i + dist_end_j, GRB.MINIMIZE)

    def _set_constraints(self, work):
        """Set the constraints"""
        self._add_constraint_each_node_is_visited_exactly_once(work)
        self._add_constraint_start_and_end_node_are_visited_once(work)
        self._add_subtour_eliminiation_without_time(work)


class ExactTSPRoutingDistanceWithWeightPrecedence(ExactTSPRouting):
    """
    Implements the exact routing algorithm for TSP with weight-based precedence constraints.
    Heavy items must be picked before lighter items.
    """
    algo_name = "ExactTSPRoutingDistanceWithWeightPrecedence"

    def __init__(self,
                 start_node: tuple[int, int], end_node: tuple[int, int], distance_matrix: pd.DataFrame,
                 predecessor_matrix: np.array, picker: list[Resource], gen_tour, gen_item_sequence,
                 articles: list[Article],
                 big_m=1000,
                 set_time_limit=300, **kwargs):
        super().__init__(start_node, end_node, distance_matrix=distance_matrix, predecessor_matrix=predecessor_matrix,
                         picker=picker, gen_tour=gen_tour, gen_item_sequence=gen_item_sequence, big_m=big_m,
                         set_time_limit=set_time_limit, **kwargs)
        self.articles = articles

    def _make_work(self, picks, model):
        article_weight_map = {article.article_id: article.weight for article in self.articles}
        return _ExactTSPWork(
            picks=picks,
            pick_nodes=[position.pick_node for position in picks],
            model=model,
            weights=[article_weight_map[position.article_id] for position in picks],
        )

    def _set_decision_variables(self, work):
        length = len(work.pick_nodes)
        work.x = work.model.addVars(length, length, vtype=GRB.BINARY, name="x")
        work.x_start = work.model.addVars(length, vtype=GRB.BINARY, name="x0j")
        work.x_end = work.model.addVars(length, vtype=GRB.BINARY, name="xj0")
        work.order = work.model.addVars(length, vtype=GRB.CONTINUOUS, lb=0, ub=length - 1, name="T")

    def _set_objective(self, work):
        """Set the objective function for the exact routing model."""
        length = len(work.pick_nodes)
        dist_x_i_x_j = gp.quicksum(self.distance_matrix.at[work.pick_nodes[i], work.pick_nodes[j]] * work.x[i, j]
                                   for i in range(length) for j in range(length) if i != j)
        dist_start_i = gp.quicksum(self.distance_matrix.at[self.start_node, work.pick_nodes[j]] * work.x_start[j]
                                   for j in range(length))
        dist_end_j = gp.quicksum(self.distance_matrix.at[work.pick_nodes[j], self.end_node] * work.x_end[j]
                                 for j in range(length))
        work.model.setObjective(dist_x_i_x_j + dist_start_i + dist_end_j, GRB.MINIMIZE)

    def _set_constraints(self, work):
        self._add_constraint_each_node_is_visited_exactly_once(work)
        self._add_constraint_start_and_end_node_are_visited_once(work)
        self._add_subtour_eliminiation_without_time(work)
        self._add_weight_precedence_constraints(work)

    def _add_weight_precedence_constraints(self, work):
        """
        Add constraints ensuring that heavier items are picked before lighter items.
        For all pairs (i, j) where weight[i] > weight[j], ensure T[i] < T[j].
        """
        epsilon = 0.01  # Small value to ensure strict inequality

        length = len(work.pick_nodes)
        for i in range(length):
            for j in range(length):
                if i != j and work.weights[i] > work.weights[j]:
                    # If item i is heavier than item j, then i must be visited before j
                    # T[i] + epsilon <= T[j]
                    work.model.addConstr(
                        work.order[i] + epsilon <= work.order[j],
                        name=f"weight_precedence_{i}_{j}"
                    )


class ExactTSPRoutingTime(ExactTSPRouting):
    """
    Implements the exact routing algorithm for the Traveling Salesman Problem (TSP) using time as the objective.
    """
    algo_name = "ExactTSPRouting"

    def __init__(self,
                 start_node: tuple[int, int], end_node: tuple[int, int], distance_matrix: pd.DataFrame,
                 predecessor_matrix: dict, picker: list[Resource], gen_tour, gen_item_sequence, big_m, set_time_limit,
                 **kwargs):
        super().__init__(start_node, end_node, distance_matrix=distance_matrix, predecessor_matrix=predecessor_matrix,
                         picker=picker, gen_tour=gen_tour, gen_item_sequence=gen_item_sequence, big_m=big_m,
                         set_time_limit=set_time_limit, **kwargs)


        self.travel_time_matrix = self.distance_matrix / self.picker[0].speed

    def _make_work(self, picks, model):
        return _ExactTSPWork(
            picks=picks,
            pick_nodes=[position.pick_node for position in picks],
            model=model,
            amounts=[position.picked_quantity for position in picks],
        )

    def _set_objective(self, work):
        length = len(work.pick_nodes)
        time_x_i_x_j = gp.quicksum((self.travel_time_matrix[work.pick_nodes[i]][work.pick_nodes[j]] + work.amounts[j] * self.picker[0].time_per_pick) * work.x[i, j]
                                   for i in range(length) for j in range(length) if i != j)
        time_start_i = gp.quicksum((self.travel_time_matrix[self.start_node][work.pick_nodes[j]] + work.amounts[j] * self.picker[0].time_per_pick) * work.x_start[j]
                                   for j in range(length))
        time_end_j = gp.quicksum(self.travel_time_matrix[work.pick_nodes[j]][self.end_node] * work.x_end[j]
                                 for j in range(length))
        work.model.setObjective(time_x_i_x_j + time_start_i + time_end_j, GRB.MINIMIZE)

    def _set_constraints(self, work):
        self._add_constraint_each_node_is_visited_exactly_once(work)
        self._add_constraint_start_and_end_node_are_visited_once(work)
        self._add_subtour_eliminiation_with_time(work, self.travel_time_matrix)


class ExactTSPRoutingMaxCompletionTime(ExactTSPRouting):
    """
    Implements the exact routing algorithm for the Traveling Salesman Problem (TSP) using maximum completion time as the objective.
    """
    algo_name = 'ExactTSPRoutingMaxCompletionTime'

    def __init__(self, batched_list, distance_matrix, tour_matrix, picker, big_m, objective, **kwargs):
        super().__init__(batched_list, distance_matrix, tour_matrix, picker, big_m, objective, **kwargs)

    def _make_work(self, picks, model):
        return _ExactTSPWork(
            picks=picks,
            pick_nodes=[position.pick_node for position in picks],
            model=model,
            amounts=[position.picked_quantity for position in picks],
        )

    def _set_decision_variables(self, work):
        """Set the decision variables for the exact routing model."""
        length = len(work.pick_nodes)
        work.x = work.model.addVars(length, length, vtype=GRB.BINARY, name="x")
        work.x_start = work.model.addVars(length, vtype=GRB.BINARY, name="x0j")
        work.x_end = work.model.addVars(length, vtype=GRB.BINARY, name="xj0")
        work.order = work.model.addVars(length, vtype=GRB.CONTINUOUS, name="T")
        work.completion = work.model.addVar(vtype=GRB.CONTINUOUS, name="C_max")

    def _set_objective(self, work):
        """Set the objective function for the exact routing model."""
        work.model.setObjective(work.completion, GRB.MINIMIZE)

    def _set_constraints(self, work):
        self._add_constraint_each_node_is_visited_exactly_once(work)
        self._add_constraint_start_and_end_node_are_visited_once(work)
        self._add_subtour_eliminiation_with_time(work, self.distance_matrix)

        # Constraint for maximum completion time C_max >= T[i] + amount_at_pick_nodes[i] * time_to_pick for all i
        for i in range(len(work.pick_nodes)):
            work.model.addConstr(work.completion >= work.order[i] + work.amounts[i] * self.picker[0]['time_to_pick'],
                               name=f"constr_C_max_{i}")


class _RRAisleInfo(NamedTuple):
    picks: tuple[PickPosition, ...]
    min_y: int | None
    max_y: int | None
    largest_gap: float
    gap_nodes: tuple[int | None, int | None]


class _RRDecision(NamedTuple):
    aisle: int
    kind: str
    from_state: tuple[str, str, str]
    to_state: tuple[str, str, str]
    action: int | tuple[int, int]
    action_node: int | tuple[int | None, int | None] | None
    cost: float


class RatliffRosenthalRouting(Routing):
    """
    Dynamic Programming based approach to solve the picker routing problem in a single-block, parallel-aisle warehouse.

    Based on:
        Katrin Heßler, Stefan Irnich (2024) Exact Solution of the Single-Picker Routing Problem with Scattered Storage.
        INFORMS Journal on Computing 36(6):1417-1435.
        https://doi.org/10.1287/ijoc.2023.0075


    """
    algo_name = "RatliffRosenthalRouting"

    def __init__(self,
                 start_node: tuple[int, int],
                 end_node: tuple[int, int],
                 closest_node_to_start: tuple[int, int],
                 min_aisle_position: int,
                 max_aisle_position: int,
                 picker: list[Resource],
                 n_aisles: int,
                 n_pick_locations: int,
                 dist_aisle: float,
                 dist_pick_locations: float,
                 dist_aisle_location: float,
                 dist_start: float,
                 dist_end: float,
                 gen_tour: bool = False,
                 gen_item_sequence: bool = False,
                 **kwargs):
        super().__init__(start_node, end_node, closest_node_to_start, min_aisle_position, max_aisle_position,
                         picker, gen_tour, gen_item_sequence, **kwargs)

        self.n_aisles = n_aisles
        self.n_pick_locations = n_pick_locations
        self.dist_aisle = dist_aisle
        self.dist_pick_locations = dist_pick_locations
        self.dist_aisle_location = dist_aisle_location
        self.dist_start = dist_start
        self.dist_end = dist_end
        self.depot = closest_node_to_start

    def score(self, input_data: list[PickPosition]) -> float:
        picks_by_aisle, aisle_info = self._prepare_aisles(input_data)
        distance, _ = self._solve_direct_dp(aisle_info, backtrack=False)
        return distance

    def _run(self, input_data: list[PickPosition]):
        """Solve the seven-state Ratliff--Rosenthal layered dynamic program."""
        picks_by_aisle, aisle_info = self._prepare_aisles(input_data)
        need_decisions = self.gen_tour or self.gen_item_sequence
        distance, decisions = self._solve_direct_dp(
            aisle_info, backtrack=need_decisions,
        )
        ordered_picks = (
            self._item_sequence_from_decisions(decisions, picks_by_aisle)
            if self.gen_tour or self.gen_item_sequence
            else []
        )
        if self.gen_tour:
            route_nodes, tour_picks, annotated_route = self._materialize_rr_tour(
                decisions, picks_by_aisle,
            )
            if self.gen_item_sequence:
                ordered_picks = tour_picks
        else:
            route_nodes = []
            annotated_route = (
                [RouteNode(pos.pick_node, NodeType.PICK) for pos in ordered_picks]
                if self.gen_item_sequence
                else []
            )
        route = Route(
            route=route_nodes,
            item_sequence=[pos.pick_node for pos in ordered_picks] if self.gen_item_sequence else [],
            distance=distance,
            annotated_route=annotated_route,
        )
        return RoutingSolution(algo_name=self.algo_name, route=route)

    def _prepare_aisles(self, pick_list) -> tuple[dict, dict]:
        """Group semantic pick objects and derive each aisle's RR extrema once."""
        grouped: dict[int, list[PickPosition]] = defaultdict(list)
        for position in pick_list:
            grouped[position.pick_node[0]].append(position)

        picks_by_aisle = {
            aisle: tuple(positions)
            for aisle, positions in grouped.items()
        }
        aisle_info = {}
        for aisle in range(1, self.n_aisles + 1):
            aisle_picks = picks_by_aisle.get(aisle, ())
            y_coords = sorted(position.pick_node[1] for position in aisle_picks)
            if y_coords:
                min_y = y_coords[0]
                max_y = y_coords[-1]
            else:
                min_y = max_y = None

            if len(y_coords) < 2:
                largest_gap = 0
                gap_nodes = (None, None)
            else:
                gaps = (
                    (
                        y_coords[index + 1] - y_coords[index],
                        (y_coords[index], y_coords[index + 1]),
                    )
                    for index in range(len(y_coords) - 1)
                )
                coordinate_gap, gap_nodes = max(
                    gaps,
                    key=lambda gap: gap[0],
                )
                largest_gap = coordinate_gap * self.dist_pick_locations

            aisle_info[aisle] = _RRAisleInfo(
                picks=aisle_picks,
                min_y=min_y,
                max_y=max_y,
                largest_gap=largest_gap,
                gap_nodes=gap_nodes,
            )
        return picks_by_aisle, aisle_info

    def _aisle_action_data(
        self,
        aisle: int,
        action: int,
        aisle_info: dict[int, _RRAisleInfo],
    ) -> tuple[float, int | tuple[int | None, int | None] | None]:
        info = aisle_info[aisle]
        action_node = None
        if action == 1:
            cost = self.one_pass()
        elif action == 2:
            action_node = info.min_y
            cost = self.top(action_node)
        elif action == 3:
            action_node = info.max_y
            cost = self.bottom(action_node)
        elif action == 4:
            action_node = info.gap_nodes
            cost = self.gap(info.largest_gap)
        elif action == 5:
            cost = self.two_pass()
        elif action == 6:
            cost = self.void()
        else:
            raise ValueError(f"Unknown RR aisle action: {action}")

        if aisle == self.depot[0]:
            cost += 2 * self.dist_end
        return cost, action_node

    def _propagate_aisle_layer(
        self,
        aisle: int,
        costs: dict[tuple[str, str, str], float],
        aisle_info: dict[int, _RRAisleInfo],
        record_predecessors: bool = True,
    ) -> tuple[
        dict[tuple[str, str, str], float],
        dict[tuple[str, str, str], tuple[tuple[str, str, str], int, object, float]],
    ]:
        next_costs: dict[tuple[str, str, str], float] = {}
        predecessors = {}
        actions = (1, 2, 3, 4, 5) if aisle_info[aisle].picks else (6,)
        action_data = [
            (action, *self._aisle_action_data(aisle, action, aisle_info))
            for action in actions
        ]

        for previous_state, previous_cost in costs.items():
            for action, action_cost, action_node in action_data:
                next_state = table_I[previous_state].get(action)
                if next_state is None:
                    continue
                candidate = previous_cost + action_cost
                if candidate < next_costs.get(next_state, float("inf")):
                    next_costs[next_state] = candidate
                    if record_predecessors:
                        predecessors[next_state] = (
                            previous_state, action, action_node, action_cost,
                        )
        return next_costs, predecessors

    def _propagate_cross_aisle_layer(
        self,
        aisle: int,
        costs: dict[tuple[str, str, str], float],
        record_predecessors: bool = True,
    ) -> tuple[
        dict[tuple[str, str, str], float],
        dict[tuple[str, str, str], tuple[tuple[str, str, str], tuple[int, int], float]],
    ]:
        next_costs: dict[tuple[str, str, str], float] = {}
        predecessors = {}

        for previous_state, previous_cost in costs.items():
            for cross_id, next_state in table_II[previous_state].items():
                if next_state is None:
                    continue
                action = cross_aisle_mapping[cross_id]
                if not self._is_valid_cross_aisle_transition(
                    aisle, previous_state, action
                ):
                    continue
                action_cost = self.cross_aisle_cost(action)
                candidate = previous_cost + action_cost
                if candidate < next_costs.get(next_state, float("inf")):
                    next_costs[next_state] = candidate
                    if record_predecessors:
                        predecessors[next_state] = (
                            previous_state, action, action_cost,
                        )
        return next_costs, predecessors

    def _solve_direct_dp(self, aisle_info, backtrack: bool) -> tuple[float, tuple[_RRDecision, ...]]:
        initial_state = ("0", "0", "0C")
        final_state = ("0", "0", "1C")
        costs = {initial_state: 0.0}
        aisle_predecessors = {} if backtrack else None
        cross_predecessors = {} if backtrack else None

        for aisle in range(1, self.n_aisles + 1):
            costs, predecessors = self._propagate_aisle_layer(
                aisle, costs, aisle_info, backtrack
            )
            if backtrack:
                aisle_predecessors[aisle] = predecessors
            costs, predecessors = self._propagate_cross_aisle_layer(
                aisle, costs, backtrack
            )
            if backtrack:
                cross_predecessors[aisle] = predecessors

        if final_state not in costs:
            raise RuntimeError("No feasible Ratliff--Rosenthal state path")

        decisions = self._backtrack_decisions(
            aisle_predecessors, cross_predecessors, initial_state, final_state,
        ) if backtrack else ()
        return costs[final_state], decisions

    def _backtrack_decisions(
        self,
        aisle_predecessors,
        cross_predecessors,
        initial_state,
        final_state,
    ) -> tuple[_RRDecision, ...]:
        reversed_decisions: list[_RRDecision] = []
        state = final_state

        for aisle in range(self.n_aisles, 0, -1):
            previous_state, action, action_cost = cross_predecessors[aisle][state]
            reversed_decisions.append(_RRDecision(
                aisle=aisle,
                kind="cross",
                from_state=previous_state,
                to_state=state,
                action=action,
                action_node=None,
                cost=action_cost,
            ))
            state = previous_state

            previous_state, action, action_node, action_cost = aisle_predecessors[aisle][state]
            reversed_decisions.append(_RRDecision(
                aisle=aisle,
                kind="aisle",
                from_state=previous_state,
                to_state=state,
                action=action,
                action_node=action_node,
                cost=action_cost,
            ))
            state = previous_state

        if state != initial_state:
            raise RuntimeError("Broken Ratliff--Rosenthal predecessor chain")

        return tuple(reversed(reversed_decisions))

    def _materialize_rr_tour(self, decisions, picks_by_aisle):
        """Materialize the selected Eulerian RR subgraph, preserving multiplicity."""
        graph = self._construct_picker_tour(decisions)
        if graph.number_of_edges() == 0:
            nodes = [self.start_node, self.end_node]
            return nodes, [], [RouteNode(node, NodeType.ROUTE) for node in nodes]
        if not nx.is_eulerian(graph):
            raise RuntimeError("Selected RR subgraph is not Eulerian")

        source = (
            self.closest_node_to_start
            if self.closest_node_to_start in graph
            else min(graph.nodes)
        )
        euler_edges = list(nx.eulerian_circuit(graph, source=source, keys=True))
        route_nodes = [self.start_node]
        annotated = [RouteNode(self.start_node, NodeType.ROUTE)]
        if route_nodes[-1] != source:
            route_nodes.append(source)
            annotated.append(RouteNode(source, NodeType.ROUTE))

        picked_nodes = set()
        ordered_picks = []
        for edge_source, edge_target, _ in euler_edges:
            if route_nodes[-1] != edge_source:
                route_nodes.append(edge_source)
                annotated.append(RouteNode(edge_source, NodeType.ROUTE))

            traversed_picks = []
            if edge_source[0] == edge_target[0]:
                low, high = sorted((edge_source[1], edge_target[1]))
                traversed_picks = [
                    pick
                    for pick in picks_by_aisle.get(edge_source[0], ())
                    if low <= pick.pick_node[1] <= high
                ]
                traversed_picks.sort(
                    key=lambda pick: pick.pick_node[1],
                    reverse=edge_target[1] < edge_source[1],
                )

            for pick in traversed_picks:
                node = pick.pick_node
                if route_nodes[-1] != node:
                    route_nodes.append(node)
                    annotated.append(RouteNode(node, NodeType.ROUTE))
                if node not in picked_nodes:
                    picked_nodes.add(node)
                    same_node_picks = [
                        candidate
                        for candidate in picks_by_aisle.get(node[0], ())
                        if candidate.pick_node == node
                    ]
                    ordered_picks.extend(same_node_picks)
                    if self.gen_item_sequence:
                        annotated.extend(
                            RouteNode(node, NodeType.PICK)
                            for _ in same_node_picks
                        )

            if route_nodes[-1] != edge_target:
                route_nodes.append(edge_target)
                annotated.append(RouteNode(edge_target, NodeType.ROUTE))

        if route_nodes[-1] != self.end_node:
            route_nodes.append(self.end_node)
            annotated.append(RouteNode(self.end_node, NodeType.ROUTE))
        return route_nodes, ordered_picks, annotated

    def _rr_node_position(self, node):
        _, y = node
        if y == 0:
            return 0.0
        if y == self.n_pick_locations + 1:
            return self.one_pass()
        return self.dist_aisle_location + (y - 1) * self.dist_pick_locations

    def _rr_route_distance(self, route_nodes):
        """Evaluate a materialized RR route in the DP's physical units."""
        total = 0.0
        final_index = len(route_nodes) - 2
        depot_connection = (
            self.dist_end if 1 <= self.depot[0] <= self.n_aisles else 0.0
        )
        for index, (source, target) in enumerate(zip(route_nodes, route_nodes[1:])):
            if index == 0 and source == self.start_node and source != target:
                total += depot_connection
            elif index == final_index and target == self.end_node and source != target:
                total += depot_connection
            elif source[0] == target[0]:
                total += abs(
                    self._rr_node_position(source) - self._rr_node_position(target)
                )
            else:
                total += abs(source[0] - target[0]) * self.dist_aisle
        return total

    def one_pass(self):
        return self.dist_pick_locations * (self.n_pick_locations - 1) + 2 * self.dist_aisle_location

    def two_pass(self):
        return 2 * self.one_pass()

    def top(self, pick_node_y: int):
        distance = (self.n_pick_locations - pick_node_y) * self.dist_pick_locations
        return 2 * distance + 2 * self.dist_aisle_location

    def bottom(self, pick_node_y: int):
        distance = (pick_node_y - 1) * self.dist_pick_locations
        return 2 * distance + 2 * self.dist_aisle_location

    def gap(self, gap_size: int):
        return 2 * self.one_pass() - 2 * gap_size

    def void(self):
        return 0

    def cross_aisle_cost(self, cross_aisle_action: tuple[int, int]):
        return self.dist_aisle * sum(cross_aisle_action)

    def _is_valid_cross_aisle_transition(self, j, prev_eq_class, cross_aisle_action):
        is_depot_aisle = (j == self.depot[0])
        action = f"{cross_aisle_action[0]}{cross_aisle_action[1]}"

        if prev_eq_class == ("U", "U", "1C"):
            return action == "11"
        if prev_eq_class == ("E", "0", "1C"):
            return action in ["22", "20", "00"] if not is_depot_aisle else action == "22"
        if prev_eq_class == ("0", "E", "1C"):
            return action in ["02", "22", "00"]
        if prev_eq_class == ("E", "E", "1C"):
            return action in ["20", "02", "22", "00"]
        if prev_eq_class == ("E", "E", "2C"):
            return action == "22"
        if prev_eq_class == ("0", "0", "0C"):
            return action == "02" if is_depot_aisle else action == "00"
        if prev_eq_class == ("0", "0", "1C"):
            return action == "00" and not is_depot_aisle
        return False

    def _construct_picker_tour(self, decisions) -> nx.MultiGraph:
        T = nx.MultiGraph()

        for decision in decisions:
            current_aisle = decision.aisle
            action = decision.action
            node_info = decision.action_node

            # Handle cross-aisle transitions
            if isinstance(action, tuple):  # e.g., (1, 0) for front & back cross-aisle moves
                a_edge, b_edge = action
                for _ in range(a_edge):
                    T.add_edge((current_aisle, self.n_pick_locations + 1),
                               (current_aisle + 1, self.n_pick_locations + 1))
                for _ in range(b_edge):
                    T.add_edge((current_aisle, 0), (current_aisle + 1, 0))
                continue  # done with this transition

            # Aisle transition — determine pick structure
            transition_type = aisle_mapping.get(action)

            if transition_type == "one_pass":
                T.add_edge((current_aisle, 0), (current_aisle, self.n_pick_locations + 1))

            elif transition_type == "two_pass":
                T.add_edge((current_aisle, 0), (current_aisle, self.n_pick_locations + 1))
                T.add_edge((current_aisle, 0), (current_aisle, self.n_pick_locations + 1))

            elif transition_type == "top" and isinstance(node_info, int):
                T.add_edge((current_aisle, self.n_pick_locations + 1), (current_aisle, node_info))
                T.add_edge((current_aisle, node_info), (current_aisle, self.n_pick_locations + 1))

            elif transition_type == "bottom" and isinstance(node_info, int):
                T.add_edge((current_aisle, 0), (current_aisle, node_info))
                T.add_edge((current_aisle, node_info), (current_aisle, 0))

            elif transition_type == "gap" and isinstance(node_info, tuple):
                y_min, y_max = node_info
                if y_min is not None and y_max is not None:
                    T.add_edge((current_aisle, 0), (current_aisle, y_min))
                    T.add_edge((current_aisle, y_min), (current_aisle, 0))
                    T.add_edge((current_aisle, self.n_pick_locations + 1), (current_aisle, y_max))
                    T.add_edge((current_aisle, y_max), (current_aisle, self.n_pick_locations + 1))

            elif transition_type == "void":
                continue  # skip void

            else:
                print(f"Unhandled action: {action} ({transition_type}), node_info: {node_info}")

        return T

    def _item_sequence_from_decisions(self, decisions, picks_by_aisle) -> list[PickPosition]:
        """
        Extracts the ordered pick sequence from the dynamic programming path (self.path).
        Returns:
            list[PickPosition]: ordered item sequence along the optimal tour.
        """
        picked_items = []

        for decision in decisions:
            action = decision.action
            action_node = decision.action_node

            # Skip cross-aisle transitions
            if isinstance(action, tuple):
                continue

            transition_type = aisle_mapping.get(action)
            aisle = decision.aisle
            aisle_orders = picks_by_aisle.get(aisle, ())
            if not aisle_orders:
                continue

            if transition_type == "one_pass" or transition_type == "two_pass":
                # Sort by y-coordinate, front-to-back
                sorted_orders = sorted(aisle_orders, key=lambda o: o.pick_node[1])
                picked_items.extend(sorted_orders)

            elif transition_type == "top" and isinstance(action_node, int):
                # Top-down: high y → low y
                picked_items.extend(sorted(
                    [o for o in aisle_orders if o.pick_node[1] >= action_node],
                    key=lambda o: o.pick_node[1], reverse=True))

            elif transition_type == "bottom" and isinstance(action_node, int):
                # Bottom-up: low y → high y
                picked_items.extend(sorted(
                    [o for o in aisle_orders if o.pick_node[1] <= action_node],
                    key=lambda o: o.pick_node[1]))

            elif transition_type == "gap" and isinstance(action_node, tuple):
                y_min, y_max = action_node
                if y_min is not None and y_max is not None:
                    # Front-to-gap, then back-to-gap (typical assumption)
                    front_picks = [o for o in aisle_orders if o.pick_node[1] <= y_min]
                    back_picks = [o for o in aisle_orders if o.pick_node[1] >= y_max]
                    picked_items.extend(sorted(front_picks, key=lambda o: o.pick_node[1]))
                    picked_items.extend(sorted(back_picks, key=lambda o: o.pick_node[1], reverse=True))

            # Void means no picks
            elif transition_type == "void":
                continue

            else:
                print(f"Unhandled transition {transition_type} for aisle {aisle}, skipping.")

        return picked_items

    def plot_picker_tour(self, T: nx.MultiGraph):
        """
        Visualizes the picker tour graph T as a 2D warehouse layout.
        Nodes are (aisle, pick_y) positions.
        """
        pos = {}
        labels = {}
        # Place nodes in grid layout: aisle = x, pick_y = y
        for node in T.nodes:
            aisle, y = node
            x_pos = aisle
            y_pos = y
            pos[node] = (x_pos, y_pos)
            labels[node] = f"{aisle},{y}"
        plt.figure(figsize=(10, 6))
        nx.draw(T, pos, with_labels=True, labels=labels,
                node_size=500, node_color="skyblue", edge_color="gray", font_size=8)
        plt.title("Picker Tour")
        plt.xlabel("Aisle")
        plt.ylabel("Pick Position")
        plt.xlim(0, self.n_aisles + 2)
        plt.ylim(-1, self.n_pick_locations + 2)
        plt.grid(True)
        plt.show()
@dataclass
class _ScatteredWork:
    graph: nx.MultiDiGraph
    demand: dict[int, int]
    aisle_content: dict[int, list[dict]]
    total_warehouse_supply: dict[int, int]
    aisle_total_supply: dict[int, dict[int, int]]


class RatliffRosenthalScatteredRouting(RatliffRosenthalRouting):
    """Dynamic Programming based approach to solve the picker routing problem in a single-block,
    parallel-aisle warehouse with scattered storage.

    Based on:
        Katrin Heßler, Stefan Irnich (2024) Exact Solution of the Single-Picker Routing Problem with Scattered Storage.
        INFORMS Journal on Computing 36(6):1417-1435.
        https://doi.org/10.1287/ijoc.2023.0075
    """

    algo_name = "RatliffRosenthalScattered"

    def __init__(self, storage_locations: StorageLocations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_locations = storage_locations
        if not hasattr(storage_locations, "article_location_mapping") or \
                storage_locations.article_location_mapping is None:
            storage_locations.build_article_location_mapping()

    def score(self, input_data: list[OrderPosition]) -> float:
        """The scattered-storage IP has no cheaper objective-only kernel."""
        _, distance = self._optimize_scattered(input_data)
        return distance

    def _run(self, input_data: list[OrderPosition]):
        route_edges, distance = self._optimize_scattered(input_data)
        route_nodes = []
        if self.gen_tour:
            tour = self._construct_scattered_tour(route_edges)
            if tour.number_of_edges():
                if not nx.is_eulerian(tour):
                    raise RuntimeError("Selected scattered-storage tour is not Eulerian")
                source = (
                    self.closest_node_to_start
                    if self.closest_node_to_start in tour
                    else min(tour.nodes)
                )
                route_nodes = [self.start_node]
                if route_nodes[-1] != source:
                    route_nodes.append(source)
                route_nodes.extend(v for _, v, _ in nx.eulerian_circuit(
                    tour, source=source, keys=True,
                ))
                if route_nodes[-1] != self.end_node:
                    route_nodes.append(self.end_node)
        route = Route(route=route_nodes, item_sequence=[], distance=distance)
        return RoutingSolution(algo_name=self.algo_name, route=route)

    def _optimize_scattered(self, input_data: list[OrderPosition]):
        demand = defaultdict(int)
        for position in input_data:
            demand[position.article_id] += position.amount
        work = _ScatteredWork(
            graph=nx.MultiDiGraph(),
            demand=dict(demand),
            aisle_content=defaultdict(list),
            total_warehouse_supply=defaultdict(int),
            aisle_total_supply=defaultdict(lambda: defaultdict(int)),
        )
        self._build_aisle_content(work)
        self._build_scattered_state_space(work)
        route_edges = self._solve_ip(work)
        distance = float(sum(d["weight"] for _, _, _, d in route_edges))
        return route_edges, distance

    def _build_aisle_content(self, work: _ScatteredWork):
        for article_id in work.demand:
            for loc in self.storage_locations.get_locations_by_article_id(article_id):
                j, y = int(loc.x), int(loc.y)
                work.aisle_content[j].append({
                    "y": y,
                    "article_id": loc.article_id,
                    "amount": loc.amount,
                })
                work.total_warehouse_supply[article_id] += loc.amount
                work.aisle_total_supply[j][article_id] += loc.amount

    def _get_relevant_y(self, work: _ScatteredWork, aisle_index: int) -> list[int]:
        return sorted({item["y"] for item in work.aisle_content.get(aisle_index, [])})

    def _build_scattered_state_space(self, work: _ScatteredWork):
        graph = work.graph
        start_node = (1, ("0", "0", "0C"), "-")
        end_node = (self.n_aisles + 1, ("0", "0", "1C"), "-")
        graph.add_node(start_node, type="start_node", pos=(0, 6))
        graph.add_node(end_node, type="end_node", pos=(self.n_aisles + 1, 7))
        for j in range(1, self.n_aisles + 2):
            for i, eq_class in enumerate(equivalence_classes):
                for stage in ("-", "+"):
                    if (j == 1 and stage == "-") or (j == self.n_aisles + 1 and stage == "+"):
                        continue
                    graph.add_node((j, eq_class, stage), pos=(2 * (j - 1) + (1.5 if stage == "+" else 0.5), i))
        self._add_aisle_transitions_for(work, 1, [("0", "0", "0C")])
        self._add_cross_aisle_transitions(work, 1)
        for j in range(2, self.n_aisles + 1):
            self._add_aisle_transitions_for(work, j, equivalence_classes)
            self._add_cross_aisle_transitions(work, j)

    def _add_aisle_transitions_for(self, work: _ScatteredWork, j: int, prev_states: list):
        is_depot_aisle = (j == self.depot[0])
        depot_cost = 2 * self.dist_end if is_depot_aisle else 0
        relevant_y = self._get_relevant_y(work, j)

        for prev_eq in prev_states:
            candidates = []

            # --- Void ---
            next_eq = self._next_state(prev_eq, 6)
            if next_eq:
                candidates.append(self._make_candidate(work,
                    j, prev_eq, next_eq, "void", None,
                    self.void() + depot_cost, [],
                ))

            # --- One pass ---
            next_eq = self._next_state(prev_eq, 1)
            if next_eq:
                candidates.append(self._make_candidate(work,
                    j, prev_eq, next_eq, "one_pass", None,
                    self.one_pass() + depot_cost,
                    [(0, self.n_pick_locations)],
                ))

            # --- Two pass ---
            next_eq = self._next_state(prev_eq, 5)
            if next_eq:
                candidates.append(self._make_candidate(work,
                    j, prev_eq, next_eq, "two_pass", None,
                    self.two_pass() + depot_cost,
                    [(0, self.n_pick_locations)],
                ))

            # --- Top(p): increasing order, break on infeasibility (Eq 1) ---
            next_eq = self._next_state(prev_eq, 2)
            if next_eq and relevant_y:
                for p in relevant_y:
                    cand = self._make_candidate(work,
                        j, prev_eq, next_eq, "top", p,
                        self.top(p) + depot_cost,
                        [(p, self.n_pick_locations)],
                    )
                    if not self._is_edge_feasible(work, cand["supply"], j):
                        break
                    candidates.append(cand)

            # --- Bottom(p): decreasing order, break on infeasibility (Eq 1) ---
            next_eq = self._next_state(prev_eq, 3)
            if next_eq and relevant_y:
                for p in reversed(relevant_y):
                    cand = self._make_candidate(work,
                        j, prev_eq, next_eq, "bottom", p,
                        self.bottom(p) + depot_cost,
                        [(0, p)],
                    )
                    if not self._is_edge_feasible(work, cand["supply"], j):
                        break
                    candidates.append(cand)

            # --- Gap(h, i): structured enumeration with monotone pruning (Eq 1) ---
            next_eq = self._next_state(prev_eq, 4)
            if next_eq and len(relevant_y) >= 2:
                n = len(relevant_y)
                for h_idx in range(n - 1):
                    h = relevant_y[h_idx]
                    feasible_row = False
                    for i_idx in range(h_idx + 1, n):
                        i = relevant_y[i_idx]
                        gap_distance = (i - h) * self.dist_pick_locations
                        cand = self._make_candidate(work,
                            j, prev_eq, next_eq, "gap", (h, i),
                            self.gap(gap_distance) + depot_cost,
                            [(0, h), (i, self.n_pick_locations)],
                        )
                        if not self._is_edge_feasible(work, cand["supply"], j):
                            break
                        feasible_row = True
                        candidates.append(cand)
                    if not feasible_row:
                        break

            # Dominance pruning (Section 2.3): skip if too many candidates
            feasible = [c for c in candidates
                        if self._is_edge_feasible(work, c["supply"], j)]

            if len(feasible) > 5000:
                for cand in feasible:
                    self._emit_edge(work, cand)
            else:
                for cand in self._prune_dominated(work, feasible, j):
                    self._emit_edge(work, cand)

    @staticmethod
    def _next_state(prev_eq, action_id) -> Optional[tuple]:
        """Look up the next equivalence class from table_I, or None."""
        return table_I.get(prev_eq, {}).get(action_id)

    def _make_candidate(self, work, j, prev_eq, next_eq, action_type,
                        action_node, cost, coverage):
        """Build a candidate edge dict with supply calculation."""
        return {
            "from": (j, prev_eq, "-"),
            "to": (j, next_eq, "+"),
            "weight": cost,
            "action_type": action_type,
            "action_node": action_node,
            "aisle": j,
            "supply": self._calculate_supply(work, j, coverage),
            "coverage": coverage,
        }

    def _emit_edge(self, work: _ScatteredWork, cand):
        """Add a candidate edge to the MultiDiGraph."""
        f, t = cand["from"], cand["to"]
        if f in work.graph and t in work.graph:
            work.graph.add_edge(
                f, t,
                weight=cand["weight"],
                action=cand["action_type"],
                action_node=cand["action_node"],
                aisle=cand["aisle"],
                supply=cand["supply"],
                coverage=cand["coverage"],
            )

    def _add_cross_aisle_transitions(self, work: _ScatteredWork, j: int):
        """Cross-aisle transitions from stage j+ to (j+1)-.

        Uses the same depot connectivity constraints as the base class.
        These are structural requirements of the Ratliff-Rosenthal DP
        and apply identically to SPRP-SS (Section 2 of Lueke et al.).
        """
        for prev_eq in equivalence_classes:
            for cross_id, next_eq in table_II.get(prev_eq, {}).items():
                if not next_eq:
                    continue
                cross_action = cross_aisle_mapping[cross_id]
                if not self._is_valid_cross_aisle_transition(j, prev_eq, cross_action):
                    continue
                cost = self.cross_aisle_cost(cross_action)
                f = (j, prev_eq, "+")
                t = (j + 1, next_eq, "-")
                if f in work.graph and t in work.graph:
                    work.graph.add_edge(f, t, weight=cost, action=cross_action)

    def _calculate_supply(self, work: _ScatteredWork, aisle_index: int,
                          coverage: list[tuple[int, int]]) -> dict[int, int]:
        """Calculate b_se: units of each article collected by this action.

        Parameters
        ----------
        aisle_index : int
        coverage : list of (y_min, y_max) inclusive ranges visited
        """
        supply: dict[int, int] = defaultdict(int)
        for item in work.aisle_content.get(aisle_index, []):
            for y_min, y_max in coverage:
                if y_min <= item["y"] <= y_max:
                    supply[item["article_id"]] += item["amount"]
                    break  # don't double-count across overlapping ranges
        return dict(supply)

    # Feasibility pruning (Eq 1)
    def _is_edge_feasible(self, work: _ScatteredWork, edge_supply: dict, aisle_index: int) -> bool:
        """Check necessary feasibility condition (Eq 1 of Lueke et al.).

        If we take this edge in aisle j, can all article demands still be
        met using this edge's supply plus everything in other aisles?
        """
        for article_id, required in work.demand.items():
            supply_other = (
                    work.total_warehouse_supply[article_id]
                    - work.aisle_total_supply[aisle_index][article_id]
            )
            supply_this = edge_supply.get(article_id, 0)
            if supply_other + supply_this < required:
                return False
        return True

    # Dominance pruning (Section 3)
    def _prune_dominated(self, work: _ScatteredWork, candidates: list[dict], aisle_index: int) -> list[dict]:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for c in candidates:
            key = (c["from"], c["to"])
            groups[key].append(c)

        result = []
        for key, group in groups.items():
            group.sort(key=lambda c: c["weight"])
            survived: list[dict] = []
            for cand in group:
                dominated = False
                for surv in survived:
                    if surv["weight"] <= cand["weight"]:
                        if all(surv["supply"].get(a, 0) >= cand["supply"].get(a, 0)
                               for a in work.demand):
                            dominated = True
                            break
                if not dominated:
                    survived.append(cand)
            result.extend(survived)
        return result

    # IP formulation (Eq 1a-1d)
    def _solve_ip(self, work: _ScatteredWork) -> list[tuple]:
        """Solve the SPRP-SS integer programme with Gurobi.

        Returns ordered list of (u, v, key, data) edge tuples.
        """
        start_node = (1, ("0", "0", "0C"), "-")
        end_node = (self.n_aisles + 1, ("0", "0", "1C"), "-")

        mdl = gp.Model("SPRP_SS")
        mdl.setParam("OutputFlag", 0)
        mdl.setParam("Threads", 1)

        # Binary variable per edge
        edge_list = list(work.graph.edges(keys=True, data=True))
        x = {}
        for u, v, k, data in edge_list:
            x[(u, v, k)] = mdl.addVar(vtype=gp.GRB.BINARY, obj=data["weight"])
        mdl.update()

        # (1b) Flow conservation
        for node in work.graph.nodes():
            out_expr = gp.quicksum(
                x[(u, v, k)]
                for u, v, k in work.graph.out_edges(node, keys=True)
            )
            in_expr = gp.quicksum(
                x[(u, v, k)]
                for u, v, k in work.graph.in_edges(node, keys=True)
            )
            if node == start_node:
                rhs = 1
            elif node == end_node:
                rhs = -1
            else:
                rhs = 0
            mdl.addConstr(out_expr - in_expr == rhs)

        # (1c) Covering constraints
        for article_id, qty_needed in work.demand.items():
            terms = []
            for u, v, k, data in edge_list:
                b = data.get("supply", {}).get(article_id, 0)
                if b > 0:
                    terms.append(b * x[(u, v, k)])
            if not terms:
                raise ValueError(
                    f"Article {article_id} (demand={qty_needed}): "
                    f"no supply edge exists. Instance infeasible."
                )
            mdl.addConstr(gp.quicksum(terms) >= qty_needed)

        mdl.optimize()
        if mdl.status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"SPRP-SS IP status {mdl.status}")

        # Extract and order selected edges
        selected = [
            (u, v, k, data)
            for u, v, k, data in edge_list
            if x[(u, v, k)].X > 0.5
        ]
        return self._order_path(selected, start_node, end_node)

    @staticmethod
    def _order_path(edges, start_node, end_node):
        """Order selected edges into a sequential path from start to end."""
        succ = {u: (u, v, k, d) for u, v, k, d in edges}
        path, current = [], start_node
        while current != end_node:
            if current not in succ:
                raise RuntimeError(f"Path broken at {current}")
            edge = succ[current]
            path.append(edge)
            current = edge[1]
        return path

    def _construct_scattered_tour(self, route_edges) -> nx.MultiGraph:
        """Reconstruct the picker tour graph T from selected edges."""
        T = nx.MultiGraph()

        for u, v, k, data in route_edges:
            action = data.get("action")
            aisle = data.get("aisle", u[0])

            # Cross-aisle
            if isinstance(action, tuple):
                a_edge, b_edge = action
                for _ in range(a_edge):
                    T.add_edge((aisle, self.n_pick_locations + 1),
                               (aisle + 1, self.n_pick_locations + 1))
                for _ in range(b_edge):
                    T.add_edge((aisle, 0), (aisle + 1, 0))
                continue

            coverage = data.get("coverage", [])

            if action == "void":
                continue

            elif action in ("one_pass", "two_pass"):
                passes = 1 if action == "one_pass" else 2
                for _ in range(passes):
                    T.add_edge((aisle, 0), (aisle, self.n_pick_locations + 1))

            elif action in ("top", "bottom"):
                if coverage:
                    y_start, y_end = coverage[0]
                    anchor = self.n_pick_locations + 1 if action == "top" else 0
                    turn = y_start if action == "top" else y_end
                    T.add_edge((aisle, anchor), (aisle, turn))
                    T.add_edge((aisle, turn), (aisle, anchor))

            elif action == "gap":
                if len(coverage) == 2:
                    seg_bot, seg_top = coverage
                    T.add_edge((aisle, 0), (aisle, seg_bot[1]))
                    T.add_edge((aisle, seg_bot[1]), (aisle, 0))
                    T.add_edge((aisle, self.n_pick_locations + 1), (aisle, seg_top[0]))
                    T.add_edge((aisle, seg_top[0]), (aisle, self.n_pick_locations + 1))

        return T
