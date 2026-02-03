from typing import Optional
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from gurobipy import GRB
from matplotlib import pyplot as plt
import gurobipy as gp

from ware_ops_algos.algorithms.algorithm import Algorithm, RoutingSolution, Route, PickPosition, RouteNode, NodeType
from ware_ops_algos.domain_models import Resource, OrderPosition, Article
from ware_ops_algos.utils.dynamic_programming_helpers import (
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
                 distance_matrix: pd.DataFrame,
                 predecessor_matrix: np.array,
                 picker: list[Resource],
                 gen_tour: bool = False,
                 gen_item_sequence: bool = False,
                 node_list: list[tuple[float, float]] = None,
                 node_to_idx: dict = None,
                 idx_to_node: dict = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.start_node = start_node
        self.end_node = end_node
        self.closest_node_to_start = closest_node_to_start
        self.min_aisle_position = min_aisle_position
        self.max_aisle_position = max_aisle_position

        self.pick_list: Optional[list[PickPosition]] = None
        self.distance_matrix = distance_matrix
        self._dist_array = distance_matrix.values
        self._node_to_idx = {node: idx for idx, node in enumerate(distance_matrix.index)}
        self.predecessor_matrix = predecessor_matrix
        self.node_list: list[tuple[float, float]] = node_list
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.gen_item_sequence = gen_item_sequence
        self.gen_tour = gen_tour
        # self.algo_name = None
        self.item_sequence = []
        self.route = []
        self.annotated_route: list[RouteNode] = []
        self.distance = 0

        self.current_order: Optional[list[PickPosition]] = None
        self.picker = picker

        self.execution_time = None

    @abstractmethod
    def _run(self, input_data: list[PickPosition]) -> RoutingSolution:
        """Concrete routing algorithms implement this and return a Route result."""
        ...

    def _get_distance(self, source, target) -> float:
        """Fast distance lookup."""
        return self._dist_array[self._node_to_idx[source], self._node_to_idx[target]]

    def reset_parameters(self):
        self.distance = 0
        self.route = []
        self.item_sequence = []

    def _get_aisle_entry_points(self) -> dict:
        """Find the entry point (min y) for each aisle."""
        aisles = defaultdict(list)
        for x, y in self.node_list:
            aisles[x].append(y)
        return {aisle: (aisle, min(ys)) for aisle, ys in aisles.items()}

    def _get_route_for_tour(self, source, target, with_last_element: bool = False):
        source_idx = self.node_to_idx[source]
        target_idx = self.node_to_idx[target]

        path_indices = []
        current_idx = target_idx

        while current_idx != source_idx:
            path_indices.insert(0, current_idx)
            current_idx = self.predecessor_matrix[source_idx, current_idx]

            if current_idx == -9999:
                raise ValueError(f"No path from {source} to {target}")

        path_indices.insert(0, source_idx)

        # Convert back to node names
        path = [self.idx_to_node[idx] for idx in path_indices]
        self.route.extend(path if with_last_element else path[:-1])

        path_nodes = [RouteNode(self.idx_to_node[idx], NodeType.ROUTE)
                      for idx in path_indices]
        self.annotated_route.extend(path_nodes if with_last_element else path_nodes[:-1])


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

        super().__init__(start_node, end_node, closest_node_to_start, min_aisle_position, max_aisle_position,
                         distance_matrix, predecessor_matrix, picker, **kwargs)
        self.fixed_depot = fixed_depot

    def _determine_walking_direction(self, current_source: tuple) -> bool:
        if current_source[1] == self.min_aisle_position:
            return True
        elif current_source[1] == self.max_aisle_position:
            return False
        else:
            raise ValueError("Start node is not connected to the beginning or end of the aisle.")

    def _walk_to_target(self, source: tuple, target: tuple, target_is_pick_node: bool = False, target_is_end_node: bool = False) -> tuple:
        if self.gen_tour:
            self._get_route_for_tour(source, target, target_is_end_node)
            if target_is_pick_node:
                self.annotated_route.append(RouteNode(target, NodeType.PICK))
        # self.distance += self.distance_matrix.at[source, target]
        self.distance += self._get_distance(source, target)
        if target_is_pick_node:
            remaining = []
            for pos in self.current_order:
                if pos.pick_node == target:
                    if self.gen_item_sequence:
                        self.item_sequence.append(target)
                else:
                    remaining.append(pos)
            self.current_order = remaining

        return target

    def _walk_to_target_and_pick(self, source: tuple, target_y: list, walking_upwards: bool = None) -> tuple:
        for next_position in target_y:
            source = self._walk_to_target(source, (source[0], next_position), target_is_pick_node=True)

        if self.current_order:
            if walking_upwards is not None:
                last_position = self.max_aisle_position if walking_upwards else self.min_aisle_position
                source = self._walk_to_target(source, (source[0], last_position))
        return source

    # def _process_aisle(self, current_source: tuple, aisle_to_visit: int, walking_up: bool = None) -> tuple:
    #     if current_source[0] == aisle_to_visit:
    #         target_y_values = self._get_sorted_y_values_for_current_aisle(current_source, walking_up)
    #         current_source = self._walk_to_target_and_pick(current_source, target_y_values, walking_up)
    #     else:
    #         target = (aisle_to_visit, current_source[1])
    #         current_source = self._walk_to_target(current_source, target)
    #     return current_source

    def _process_aisle(self, current_source: tuple, aisle_to_visit: int, walking_up: bool = None) -> tuple:
        if current_source[0] == aisle_to_visit:
            target_y_values = self._get_sorted_y_values_for_current_aisle(current_source, walking_up)
            current_source = self._walk_to_target_and_pick(current_source, target_y_values, walking_up)
        else:
            # Stay on the same cross-aisle (same y), just change x
            target = (aisle_to_visit, current_source[1])

            # Bypass predecessor matrix - direct edge only
            if self.gen_tour:
                self.route.append(current_source)
                self.annotated_route.append(RouteNode(current_source, NodeType.ROUTE))
            self.distance += self.distance_matrix.at[current_source, target]
            current_source = target

        return current_source

    def _get_min_aisle(self) -> int:
        return min(pos.pick_node[0] for pos in self.current_order)

    def _get_max_aisle(self) -> int:
        return max(pos.pick_node[0] for pos in self.current_order)

    def _get_aisle_list(self, reverse: bool = False) -> list:
        aisle_list = list(set(pos.pick_node[0] for pos in self.current_order))
        return sorted(aisle_list, reverse=reverse)

    def _get_sorted_y_values_for_current_aisle(self, current_source: tuple, walking_up: bool) -> list:
        aisle_y_values = [pos.pick_node[1] for pos in self.current_order if pos.pick_node[0] == current_source[0]]
        return sorted(aisle_y_values) if walking_up else sorted(aisle_y_values, reverse=True)

    def _go_to_end_node(self, current_source: tuple):
        if self.fixed_depot:
            end_node = self.end_node
        else:
            entry_points = self._get_aisle_entry_points()
            current_aisle_x = current_source[0]
            current_aisle_y = entry_points[current_aisle_x][1]
            end_node = (current_aisle_x, current_aisle_y)
            node_idx = self.node_to_idx[end_node]
        self._walk_to_target(current_source, end_node, target_is_end_node=True)


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

    def _process_aisle(self, current_source: tuple, aisle_to_visit: int, walking_up: bool = None) -> tuple:
        if current_source[0] == aisle_to_visit:
            target_y_values = self._get_sorted_y_values_for_current_aisle(current_source, walking_up)
            current_source = self._walk_to_target_and_pick(current_source, target_y_values, walking_up)
        else:
            # Stay on the same cross-aisle (same y), just change x
            target = (aisle_to_visit, current_source[1])

            # Bypass predecessor matrix - direct edge only
            if self.gen_tour:
                self.route.append(current_source)
            self.distance += self.distance_matrix.at[current_source, target]
            current_source = target

        return current_source

    def _run(self, pick_list: list[PickPosition]) -> RoutingSolution:
        self.pick_list = list(pick_list)
        self.current_order = list(pick_list)
        current_source = self._walk_to_target(self.start_node, self.closest_node_to_start)
        walking_up = not self._determine_walking_direction(current_source)

        while self.current_order:
            aisle_min = self._get_min_aisle()

            if current_source[0] == aisle_min:
                walking_up = not walking_up

            current_source = self._process_aisle(current_source, aisle_min, walking_up)

        self._go_to_end_node(current_source)
        route = Route(route=self.route,
                      item_sequence=self.item_sequence,
                      distance=self.distance,
                      )
        return RoutingSolution(algo_name=self.algo_name, route=route)


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

    def _run(self, pick_list: list[PickPosition]) -> RoutingSolution:
        self.pick_list = list(pick_list)
        self.current_order = list(pick_list)

        current_source = self._walk_to_target(self.start_node, self.closest_node_to_start)
        walking_up = True

        while self.current_order:
            aisle_min = self._get_min_aisle()
            current_source = self._process_aisle(current_source, aisle_min, walking_up)

        self._go_to_end_node(current_source)

        route = Route(route=self.route,
                      item_sequence=self.item_sequence,
                      distance=self.distance,
                      # pick_positions=self.pick_list,
                      # order_numbers=list({pp.order_number for pp in self.pick_list})
                      )
        return RoutingSolution(algo_name=self.algo_name, route=route)


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

    def _run(self, pick_list: list[PickPosition]) -> RoutingSolution:
        """
        Implements Midpoint Routing: handles lower aisles first, navigates back to upper aisles
        while ensuring that orders in different halves of the warehouse are handled appropriately.
        """
        self.pick_list = list(pick_list)
        self.current_order = list(pick_list)

        current_source = self._walk_to_target(self.start_node, self.closest_node_to_start)
        mid_point = round(self.max_aisle_position/2)  # Define the midpoint of the warehouse

        orders_lower_half, orders_upper_half = self.split_orders_by_pickzone(self.current_order, mid_point)

        if orders_upper_half:
            min_aisle_upper_part = min(pos.pick_node[0] for pos in orders_upper_half)
        else:
            min_aisle_upper_part = -99

        if orders_lower_half:
            max_aisle_lower_part = max(pos.pick_node[0] for pos in orders_lower_half)
            min_aisle_lower_part = min(pos.pick_node[0] for pos in orders_lower_half)

            # Verschiebe nur, wenn min Aisles gleich sind

            if min_aisle_lower_part <= min_aisle_upper_part:
                to_move = [pos for pos in orders_lower_half if pos.pick_node[0] == min_aisle_lower_part]

                # Verschieben
                for order in to_move:
                    orders_lower_half.remove(order)
                    orders_upper_half.append(order)

            # Process orders in the lower half of the warehouse
            self.current_order = orders_lower_half
            current_source = self._process_lower_half(current_source, max_aisle_lower_part)


            # Process orders in the upper half of the warehouse
            self.current_order = orders_upper_half
            current_source = self._process_upper_half(current_source, max_aisle_lower_part)

        else:
            # If there are no orders in the lower half, process only the upper half
            current_source = self._process_upper_half(current_source, -99)

        # Walk to the end node after completing all orders
        self._go_to_end_node(current_source)
        route = Route(route=self.route,
                      item_sequence=self.item_sequence,
                      distance=self.distance,
                      )
        return RoutingSolution(algo_name=self.algo_name, route=route)

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

    def _process_lower_half(self, current_source, max_aisle_lower_part):
        """
        Process all orders in the lower half of the warehouse (y <= mid_point).

        :param current_source: the current source node

        Returns the current source node after completing all lower orders.
        """

        while self.current_order:
            # Find the minimum aisle in the current order
            aisle_to_visit = self._get_min_aisle()

            if current_source[0] == max_aisle_lower_part:
                current_source = self._process_aisle(current_source, aisle_to_visit, walking_up=True)
            else:
                # Process the aisle by walking and picking items and return back to the source
                current_source = self._process_aisle(current_source, aisle_to_visit, walking_up=False)

        return current_source

    def _process_upper_half(self, current_source, max_aisle_lower_part):
        """
        Process all orders in the upper half of the warehouse (y > mid_point).

        :param current_source: the current source node
        :param max_aisle_lower_part: the maximum aisle in the lower half of the warehouse

        Returns the current source node after completing all upper orders.
        """
        if max_aisle_lower_part != -99:
            current_source = self._process_transition_lower_to_upper_half(current_source, max_aisle_lower_part)

        while self.current_order:
            # Find the maximum aisle in the current order
            aisle_to_visit = self._get_max_aisle()
            min_aisle = self._get_min_aisle()

            if aisle_to_visit == min_aisle:
                # If the aisle to visit is the minimum aisle, process it directly
                current_source = self._process_aisle(current_source, aisle_to_visit, walking_up=False)
            else:
                # Process the current aisle by walking and picking items
                current_source = self._process_aisle(current_source, aisle_to_visit, walking_up=True)

        return current_source

    def _process_transition_lower_to_upper_half(self, current_source: tuple[int, int], max_aisle_lower_part: int) -> tuple[int, int]:
        """
        Process the transition from the lower to the upper half of the warehouse.

        :param current_source: the current source node
        :param max_aisle_lower_part: the maximum aisle in the lower half of the warehouse

        Returns the current source node after transitioning to the upper half.
        """
        if self.current_order:
            max_aisle_upper_part = self._get_max_aisle()

            if max_aisle_upper_part == -99:
                # If there are no orders in the upper half, walk to the end node
                current_source = self._walk_to_target(current_source, self.end_node, target_is_end_node=True)

            elif max_aisle_upper_part < max_aisle_lower_part:
                # Walk to the self.max_aisle_position of the current aisle
                current_source = self._walk_to_target(current_source, (current_source[0], self.max_aisle_position))
            elif max_aisle_upper_part == max_aisle_lower_part:
                # Process the current aisle by walking and picking items
                current_source = self._process_aisle(current_source, max_aisle_upper_part, walking_up=True)
            else:
                # Walk to the (max_aisle_upper_part, 1) position
                current_source= self._walk_to_target(current_source, (current_source[0], self.min_aisle_position))
                current_source = self._walk_to_target(current_source, (max_aisle_upper_part, self.min_aisle_position))
                positions_to_visit = self._get_sorted_y_values_for_current_aisle(current_source, walking_up=True)
                current_source = self._walk_to_target_and_pick(current_source, positions_to_visit, walking_upwards=None)
                if not self.current_order:
                    current_source = self._walk_to_target(current_source, self.end_node)
                else:
                    current_source = self._walk_to_target(current_source, (current_source[0],self.max_aisle_position))
        return current_source


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

    def _run(self, pick_list: list[PickPosition]) -> RoutingSolution:
        """
        Executes the Largest Gap Routing algorithm. This strategy identifies the largest gap between
        pick locations in an aisle and processes items accordingly to minimize walking distance.
        """
        self.pick_list = list(pick_list)
        self.current_order = list(pick_list)

        # Initialize routing by moving into the storage area
        # current_source = self._walk_from_start_to_graph()
        current_source = self._walk_to_target(self.start_node, self.closest_node_to_start)

        aisle_max = self._get_max_aisle()
        aisle_list = self._get_aisle_list()
        aisle_min = self._get_min_aisle()
        max_postion_of_orders_to_list = max(pos.pick_node[1] for pos in self.current_order)

        if max_postion_of_orders_to_list <= self.max_aisle_position/2:
            aisle_list = aisle_list
        elif len(aisle_list) >= 2:
            # if there are at least two aisles, we do not need to walk to the first aisle
            aisle_list = aisle_list[1:]
        else:
            # if there is only one aisle, we can start directly with the first aisle
            aisle_list = aisle_list[:1]

        for next_aisle in aisle_list:
            if current_source[0] is not next_aisle:
                current_source = self._walk_to_target(current_source, (next_aisle, current_source[1]))

            target_y_values = self._get_sorted_y_values_for_current_aisle(current_source, True)

            if aisle_max == next_aisle:
                current_source = self._walk_to_target_and_pick(current_source, target_y_values, True)
            else:
                pos_largest_gap = self._get_largest_gap_pos_inside_aisle(target_y_values)
                current_source = self._walk_to_target_and_pick(current_source, target_y_values[:pos_largest_gap],False)

        while self.current_order:
            next_aisle = self._get_max_aisle()
            if next_aisle == self._get_min_aisle():
                target = (next_aisle, current_source[1])
                current_source = self._walk_to_target(current_source, target)
                target_y_values = self._get_sorted_y_values_for_current_aisle(current_source, False)
                current_source = self._walk_to_target_and_pick(current_source, target_y_values, False)

            current_source = self._process_aisle(current_source, next_aisle, True)

        # walk to the end node
        self._go_to_end_node(current_source)
        route = Route(route=self.route,
                      item_sequence=self.item_sequence,
                      distance=self.distance,
                      )
        return RoutingSolution(algo_name=self.algo_name, route=route)

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

    def _run(self, pick_list: list[PickPosition]) -> RoutingSolution:
        self.pick_list = list(pick_list)
        self.current_order = list(pick_list)

        current_source = self._walk_to_target(self.start_node, self.closest_node_to_start)

        while self.current_order:
            nearest_pick_node = self._get_next_nearest_node_by_dijkstra(current_source)
            current_source = self._walk_to_target(current_source, nearest_pick_node, target_is_pick_node=True)

        # walk to the end node
        self._go_to_end_node(current_source)
        route = Route(route=self.route,
                      item_sequence=self.item_sequence,
                      distance=self.distance,
                      )
        return RoutingSolution(algo_name=self.algo_name, route=route)

    # def _get_next_nearest_node_by_dijkstra(self, current_source: tuple) -> tuple:
    #     """
    #     Determines the next nearest pick node using Dijkstra's algorithm.
    #
    #     :param current_source: the current source node
    #
    #     Returns the next nearest pick node.
    #     """
    #     return min(
    #         [pos.pick_node for pos in self.current_order],
    #         key=lambda item: self.distance_matrix.at[current_source, item]
    #     )

    def _get_next_nearest_node_by_dijkstra(self, current_source: tuple) -> tuple:
        """Vectorized nearest neighbor selection."""
        source_idx = self._node_to_idx[current_source]

        pick_nodes = [pos.pick_node for pos in self.current_order]
        pick_indices = [self._node_to_idx[node] for node in pick_nodes]

        # Single NumPy operation instead of repeated lookups
        distances = self._dist_array[source_idx, pick_indices]

        return pick_nodes[distances.argmin()]


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

    def _run(self, input_data: list[PickPosition]) -> RoutingSolution:
        self.reset_parameters()
        self.current_order = list(input_data)  # needed for _walk_to_target bookkeeping

        entry_points = self._get_aisle_entry_points()

        if self.fixed_depot == True:
            start_node = self.start_node
        else:
            picker_location = self.picker[0].current_location
            if isinstance(picker_location, RouteNode):
                start_node = picker_location.position
            else:
                start_node = picker_location
        # Start from start_node, go to first aisle entry
        first_aisle = input_data[0].pick_node[0]
        aisle_entry = entry_points[first_aisle]

        current_source = self._walk_to_target(start_node, aisle_entry)

        # Walk through picks in list order
        for pp in input_data:
            current_source = self._walk_to_target(current_source, pp.pick_node, target_is_pick_node=True)

        # Return to end node
        self._go_to_end_node(current_source)
        route = Route(route=self.route,
                      item_sequence=self.item_sequence,
                      distance=self.distance,
                      annotated_route=self.annotated_route
                      )
        return RoutingSolution(algo_name=self.algo_name, route=route)


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
        ##############################
        self.length = None
        self.pick_nodes = None
        self.mdl = None
        self.amount_at_pick_nodes = None


class ExactTSPRouting(ExactRouting):
    """
    Implements the exact routing algorithm for the Traveling Salesman Problem (TSP).
    """

    def __init__(self,
                 start_node: tuple[int, int], end_node: tuple[int, int], distance_matrix: pd.DataFrame,
                 predecessor_matrix: dict, picker: list[Resource], big_m, set_time_limit, **kwargs):
        super().__init__(start_node, end_node, distance_matrix=distance_matrix, predecessor_matrix=predecessor_matrix, picker=picker, big_m=big_m, set_time_limit=set_time_limit, **kwargs)

        self.T = None
        self.C_max = None

        self.x = None
        self.x_start = None
        self.x_end = None

    def _set_exact_routing_parameters(self):
        ...

    def _set_decision_variables(self):
        ...

    def _set_objective(self):
        ...

    def _set_constraints(self):
        ...

    def _run(self, pick_list: list[PickPosition]):
        """Solves the exact routing problem using the Gurobi optimization solver."""
        # Set the parameters for the exact routing algorithm
        self.pick_list = list(pick_list)
        self.current_order = list(pick_list)

        self._set_exact_routing_parameters()

        self.mdl = gp.Model(f"{self.algo_name}")
        self.mdl.setParam('OutputFlag', 1)
        if self.time_limit > 0:
            self.mdl.setParam('TimeLimit', self.time_limit)

        self._set_decision_variables()

        self._set_objective()

        self._set_constraints()
        self.mdl.optimize()
        if self.mdl.status == GRB.OPTIMAL or (self.time_limit and self.mdl.SolCount > 0):
            print(f"Model solved with status {self.mdl.status}")
            # self.mdl.write('model.lp')
            # for v in self.mdl.getVars():
                # if v.x > 0.5:
                #     print(f"{v.varName}: {v.x}")

            self.distance = self.mdl.objVal

            self.construct_route_and_item_sequence()
        elif self.mdl.status == GRB.INFEASIBLE:
            print(f"Model could not be solved. Status: {self.mdl.status}")
            self.mdl.computeIIS()
            # self.mdl.write("model.ilp")
            # self.mdl.write("model.lp")
            for c in self.mdl.getConstrs():
                if c.IISConstr:
                    print(f"Infeasible Constraint: {c.ConstrName}")
        else:
            print(f"Model could not be solved. Status: {self.mdl.status}")
        route = Route(route=self.route,
                      item_sequence=self.item_sequence,
                      distance=self.distance,
                      )
        return RoutingSolution(algo_name=self.algo_name, route=route)

    def _add_constraint_each_node_is_visited_exactly_once(self):
        for i in range(self.length):
            self.mdl.addConstr(gp.quicksum(self.x[i, j] for j in range(self.length) if i != j) + self.x_end[i] == 1,
                               name=f"constr1_{i}")
            self.mdl.addConstr(gp.quicksum(self.x[j, i] for j in range(self.length) if j != i) + self.x_start[i] == 1,
                               name=f"constr2_{i}")

    def _add_constraint_start_and_end_node_are_visited_once(self):
        self.mdl.addConstr(gp.quicksum(self.x_start[j] for j in range(self.length)) == 1, name="constr3")
        self.mdl.addConstr(gp.quicksum(self.x_end[j] for j in range(self.length)) == 1, name="constr4")

    def _add_subtour_eliminiation_without_time(self):
        for i in range(self.length):
            for j in range(self.length):
                if i != j:
                    self.mdl.addConstr(self.T[i] + 1 <= self.T[j] + self.big_m * (1 - self.x[i, j]), name=f"constr5_{i}_{j}")

    def _add_subtour_eliminiation_with_time(self, travel_time_matrix: pd.DataFrame):
        for i in range(self.length):
            for j in range(self.length):
                if i != j:
                    self.mdl.addConstr(self.T[i] + travel_time_matrix[self.pick_nodes[i]][self.pick_nodes[j]]
                                       <= self.T[j] + self.big_m * (1 - self.x[i, j]), name=f"constr5_{i}_{j}")

    def construct_route_and_item_sequence(self):
        """Generates the route for the exact routing algorithm from solution variables."""

        if self.gen_item_sequence or self.gen_tour:
            current_node = None

            for i in range(self.length):
                if self.x_start[i].x > 0.5:
                    if self.gen_tour:
                        self._get_route_for_tour(self.start_node, self.pick_nodes[i])
                    if self.gen_item_sequence:
                        self.item_sequence.append(self.pick_nodes[i])
                    current_node = i
                    break

            visited = set()
            while True:
                found = False
                for j in range(self.length):
                    if j in visited:
                        continue
                    if self.x[current_node, j].x > 0.5:
                        if self.gen_tour:
                            self._get_route_for_tour(self.pick_nodes[current_node], self.pick_nodes[j])
                        if self.gen_item_sequence:
                            self.item_sequence.append(self.pick_nodes[j])
                        visited.add(current_node)
                        current_node = j
                        found = True
                        break
                if not found:
                    break

            # Endknoten
            for i in range(self.length):
                if self.x_end[i].x > 0.5:
                    if self.gen_tour:
                        self._get_route_for_tour(self.pick_nodes[current_node], self.end_node, with_last_element=True)
                    break


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

    def _set_exact_routing_parameters(self):
        """Sets the parameters for the exact routing algorithm."""
        self.length = len(self.current_order)
        self.pick_nodes = [pos.pick_node for pos in self.pick_list]
        # self.pick_nodes = self.current_order['pick_node'].tolist()


    def _set_decision_variables(self):
        """Set the decision variables for the exact routing model."""
        self.x = self.mdl.addVars(self.length, self.length, vtype=GRB.BINARY, name="x")
        self.x_start = self.mdl.addVars(self.length, vtype=GRB.BINARY, name="x0j")
        self.x_end = self.mdl.addVars(self.length, vtype=GRB.BINARY, name="xj0")
        self.T = self.mdl.addVars(self.length, vtype=GRB.CONTINUOUS, name="T")

    def _set_objective(self):
        """Set the objective function for the exact routing model."""
        dist_x_i_x_j = gp.quicksum(self.distance_matrix.at[self.pick_nodes[i], self.pick_nodes[j]] * self.x[i, j]
                                   for i in range(self.length) for j in range(self.length) if i != j)
        dist_start_i = gp.quicksum(self.distance_matrix.at[self.start_node, self.pick_nodes[j]] * self.x_start[j]
                                  for j in range(self.length))
        dist_end_j = gp.quicksum(self.distance_matrix.at[self.pick_nodes[j], self.end_node] * self.x_end[j]
                                for j in range(self.length))
        self.mdl.setObjective(dist_x_i_x_j + dist_start_i + dist_end_j, GRB.MINIMIZE)

    def _set_constraints(self):
        """Set the constraints"""

        self._add_constraint_each_node_is_visited_exactly_once()
        self._add_constraint_start_and_end_node_are_visited_once()
        self._add_subtour_eliminiation_without_time()


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
        self.weights = []

    def _set_exact_routing_parameters(self):
        """Sets the parameters for the exact routing algorithm."""
        self.length = len(self.current_order)
        self.pick_nodes = [pos.pick_node for pos in self.pick_list]

        article_weight_map = {article.article_id: article.weight for article in self.articles}

        self.weights = [article_weight_map[pos.article_id] for pos in self.pick_list]

    def _set_decision_variables(self):
        """Set the decision variables for the exact routing model."""
        self.x = self.mdl.addVars(self.length, self.length, vtype=GRB.BINARY, name="x")
        self.x_start = self.mdl.addVars(self.length, vtype=GRB.BINARY, name="x0j")
        self.x_end = self.mdl.addVars(self.length, vtype=GRB.BINARY, name="xj0")
        self.T = self.mdl.addVars(self.length, vtype=GRB.CONTINUOUS, lb=0, ub=self.length - 1, name="T")

    def _set_objective(self):
        """Set the objective function for the exact routing model."""
        dist_x_i_x_j = gp.quicksum(self.distance_matrix.at[self.pick_nodes[i], self.pick_nodes[j]] * self.x[i, j]
                                   for i in range(self.length) for j in range(self.length) if i != j)
        dist_start_i = gp.quicksum(self.distance_matrix.at[self.start_node, self.pick_nodes[j]] * self.x_start[j]
                                   for j in range(self.length))
        dist_end_j = gp.quicksum(self.distance_matrix.at[self.pick_nodes[j], self.end_node] * self.x_end[j]
                                 for j in range(self.length))
        self.mdl.setObjective(dist_x_i_x_j + dist_start_i + dist_end_j, GRB.MINIMIZE)

    def _set_constraints(self):
        self._add_constraint_each_node_is_visited_exactly_once()
        self._add_constraint_start_and_end_node_are_visited_once()
        self._add_subtour_eliminiation_without_time()
        self._add_weight_precedence_constraints()

    def _add_weight_precedence_constraints(self):
        """
        Add constraints ensuring that heavier items are picked before lighter items.
        For all pairs (i, j) where weight[i] > weight[j], ensure T[i] < T[j].
        """
        epsilon = 0.01  # Small value to ensure strict inequality

        for i in range(self.length):
            for j in range(self.length):
                if i != j and self.weights[i] > self.weights[j]:
                    # If item i is heavier than item j, then i must be visited before j
                    # T[i] + epsilon <= T[j]
                    self.mdl.addConstr(
                        self.T[i] + epsilon <= self.T[j],
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

    def _set_exact_routing_parameters(self):
        """Sets the parameters for the exact routing algorithm."""
        self.length = len(self.current_order)
        self.pick_nodes = [pos.pick_node for pos in self.pick_list]
        self.amount_at_pick_nodes = [pos.amount for pos in self.pick_list]

    def _set_decision_variables(self):
        """Set the decision variables for the exact routing model."""
        self.x = self.mdl.addVars(self.length, self.length, vtype=GRB.BINARY, name="x")
        self.x_start = self.mdl.addVars(self.length, vtype=GRB.BINARY, name="x0j")
        self.x_end = self.mdl.addVars(self.length, vtype=GRB.BINARY, name="xj0")
        self.T = self.mdl.addVars(self.length, vtype=GRB.CONTINUOUS, name="T")

    def _set_objective(self):
        time_x_i_x_j = gp.quicksum((self.travel_time_matrix[self.pick_nodes[i]][self.pick_nodes[j]] + self.amount_at_pick_nodes[j] * self.picker[0].time_per_pick)  * self.x[i, j]
                                   for i in range(self.length) for j in range(self.length) if i != j)
        time_start_i = gp.quicksum((self.travel_time_matrix[self.start_node][self.pick_nodes[j]]  + self.amount_at_pick_nodes[j] * self.picker[0].time_per_pick) * self.x_start[j]
                                   for j in range(self.length))
        time_end_j = gp.quicksum(self.travel_time_matrix[self.pick_nodes[j]][self.end_node] * self.x_end[j]
                                 for j in range(self.length))
        self.mdl.setObjective(time_x_i_x_j + time_start_i + time_end_j, GRB.MINIMIZE)

    def _set_constraints(self):

        self._add_constraint_each_node_is_visited_exactly_once()
        self._add_constraint_start_and_end_node_are_visited_once()
        self._add_subtour_eliminiation_with_time(self.travel_time_matrix)


class ExactTSPRoutingMaxCompletionTime(ExactTSPRouting):
    """
    Implements the exact routing algorithm for the Traveling Salesman Problem (TSP) using maximum completion time as the objective.
    """
    algo_name = 'ExactTSPRoutingMaxCompletionTime'

    def __init__(self, batched_list, distance_matrix, tour_matrix, picker, big_m, objective, **kwargs):
        super().__init__(batched_list, distance_matrix, tour_matrix, picker, big_m, objective, **kwargs)

    def _set_exact_routing_parameters(self):
        """Sets the parameters for the exact routing algorithm."""
        self.length = len(self.current_order)
        self.pick_nodes = self.current_order['pick_node'].tolist()
        self.amount_at_pick_nodes = self.current_order['amount'].tolist()

    def _set_decision_variables(self):
        """Set the decision variables for the exact routing model."""
        self.x = self.mdl.addVars(self.length, self.length, vtype=GRB.BINARY, name="x")
        self.x_start = self.mdl.addVars(self.length, vtype=GRB.BINARY, name="x0j")
        self.x_end = self.mdl.addVars(self.length, vtype=GRB.BINARY, name="xj0")
        self.T = self.mdl.addVars(self.length, vtype=GRB.CONTINUOUS, name="T")
        self.C_max = self.mdl.addVar(vtype=GRB.CONTINUOUS, name="C_max")

    def _set_objective(self):
        """Set the objective function for the exact routing model."""
        self.mdl.setObjective(self.C_max, GRB.MINIMIZE)

    def _set_constraints(self):

        self._add_constraint_each_node_is_visited_exactly_once()
        self._add_constraint_start_and_end_node_are_visited_once()
        self._add_subtour_eliminiation_with_time()

        # Constraint for maximum completion time C_max >= T[i] + amount_at_pick_nodes[i] * time_to_pick for all i
        for i in range(self.length):
            self.mdl.addConstr(self.C_max >= self.T[i] + self.amount_at_pick_nodes[i] * self.picker[0]['time_to_pick'],
                               name=f"constr_C_max_{i}")


class RatliffRosenthalRouting(Routing):
    algo_name = "RatliffRosenthalRouting"

    def __init__(self,
                 start_node: tuple[int, int],
                 end_node: tuple[int, int],
                 closest_node_to_start: tuple[int, int],
                 min_aisle_position: int,
                 max_aisle_position: int,
                 distance_matrix: pd.DataFrame,
                 predecessor_matrix: np.array,
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
                         distance_matrix, predecessor_matrix, picker, gen_tour, gen_item_sequence, **kwargs)

        # self.algo_name = "RatliffRosenthal"
        self.state_graph = nx.DiGraph()
        self.n_aisles = n_aisles
        self.n_pick_locations = n_pick_locations
        self.dist_aisle = dist_aisle
        self.dist_pick_locations = dist_pick_locations
        self.dist_aisle_location = dist_aisle_location
        self.dist_start = dist_start
        self.dist_end = dist_end
        self.depot = closest_node_to_start

    def _run(self, input_data: list[PickPosition]):
        """ Entry point for solving the routing problem. """
        self.pick_list = input_data
        self.current_order = list(input_data)
        self.reset_parameters()
        self.state_graph = nx.DiGraph()
        self.build_state_space()
        start_node = (1, ("0", "0", "0C"), "-")
        end_node = (self.n_aisles + 1, ("0", "0", "1C"), "-")
        self.path = nx.shortest_path(self.state_graph, start_node, end_node, weight='weight', method="bellman-ford")
        self.distance = nx.path_weight(self.state_graph, self.path, weight='weight')
        self.T = self._construct_picker_tour()

        route = Route(route=self.route,
                      item_sequence=self.item_sequence,
                      distance=self.distance,
                      )
        return RoutingSolution(algo_name=self.algo_name, route=route)

    def build_state_space(self):
        start_node = (1, ("0", "0", "0C"), "-")
        end_node = (self.n_aisles + 1, ("0", "0", "1C"), "-")
        self.state_graph.add_node(start_node, type="start_node", pos=(0, 6))
        self.state_graph.add_node(end_node, type="end_node", pos=(self.n_aisles + 1, 7))

        for j in range(1, self.n_aisles + 2):
            for i, eq_class in enumerate(equivalence_classes):
                for stage in ["-", "+"]:
                    if (j == 1 and stage == "-") or (j == self.n_aisles + 1 and stage == "+"):
                        continue
                    x = 2 * (j - 1) + (1.5 if stage == "+" else 0.5)
                    self.state_graph.add_node((j, eq_class, stage), pos=(x, i))

        self._add_first_aisle_transitions()
        self._add_cross_aisle_transitions(1)
        for j in range(2, self.n_aisles + 1):
            self._add_aisle_transitions(j)
            self._add_cross_aisle_transitions(j)

    def _get_aisle_orders(self, j: int) -> list[PickPosition]:
        return [pos for pos in self.pick_list if pos.pick_node[0] == j]

    @staticmethod
    def largest_gap(order_list: list[PickPosition]):
        y_coords = sorted(pos.pick_node[1] for pos in order_list)
        if len(y_coords) < 2:
            return 0, (None, None)
        gaps = [(y_coords[i+1] - y_coords[i], (y_coords[i], y_coords[i+1])) for i in range(len(y_coords) - 1)]
        return max(gaps, key=lambda x: x[0], default=(0, (None, None)))

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

    def cost_fn_wrapper(self, order_subset: list[PickPosition], transition: str, node=None):
        if not order_subset:
            assert transition == "void", f"Invalid transition '{transition}' on empty order subset"
        cost = self.void()
        if transition == "one_pass":
            cost = self.one_pass()
        elif transition == "two_pass":
            cost = self.two_pass()
        elif transition == "top":
            pick_y = node if node else min(pos.pick_node[1] for pos in order_subset)
            cost = self.top(pick_y)
            node = pick_y
        elif transition == "bottom":
            pick_y = node if node else max(pos.pick_node[1] for pos in order_subset)
            cost = self.bottom(pick_y)
            node = pick_y
        elif transition == "gap":
            if node:
                lg = max(node) - min(node)
            else:
                lg, node = self.largest_gap(order_subset)
            cost = self.gap(lg)
        return cost, node

    def _add_first_aisle_transitions(self):
        aisle_orders = self._get_aisle_orders(1)
        initial_state = ("0", "0", "0C")
        if not aisle_orders:
            allowed_actions = [6]  # void only
        else:
            allowed_actions = [a for a in aisle_mapping.keys() if a != 6]
        for action in allowed_actions:
            if action == 6 and aisle_orders:
                continue
            cost, node = self.cost_fn_wrapper(aisle_orders, aisle_mapping[action])
            cost += 2 * self.dist_end if self.depot[0] == 1 else 0
            from_state = (1, initial_state, "-")
            to_state = (1, table_I[initial_state][action], "+")
            self.state_graph.add_edge(from_state, to_state, weight=cost, action=action, action_node=node)

    def _add_aisle_transitions(self, j: int):
        depot_aisle = self.depot[0]
        is_depot_aisle = (j == depot_aisle)
        aisle_orders = self._get_aisle_orders(j)
        for prev_eq_class in equivalence_classes:
            already_visited = {c: np.inf for c in equivalence_classes}
            if not aisle_orders:
                allowed_actions = [6]  # void only
            else:
                allowed_actions = [a for a in aisle_mapping.keys() if a != 6]

            for action in allowed_actions:
                if action in table_I[prev_eq_class]:
                    if action == 6:
                        if len(aisle_orders) > 0:
                            # skip void action if aisle has picks
                            continue
                    current_eq_class = table_I[prev_eq_class][action]
                    cost, node = self.cost_fn_wrapper(aisle_orders, aisle_mapping[action])
                    if is_depot_aisle:
                        cost += 2 * self.dist_end
                    if current_eq_class:
                        if cost < already_visited[current_eq_class]:
                            # Edges to the same state overwrite each other, only keep best
                            from_state = (j, prev_eq_class, "-")
                            to_state = (j, current_eq_class, "+")
                            self.state_graph.add_edge(from_state, to_state, weight=cost, action=action,
                                                      action_node=node)
                            already_visited[current_eq_class] = cost

    def _add_cross_aisle_transitions(self, j: int):
        for prev_eq_class in equivalence_classes:
            for cross_id, current_eq_class in table_II[prev_eq_class].items():
                if current_eq_class and self._is_valid_cross_aisle_transition(j, prev_eq_class, cross_aisle_mapping[cross_id]):
                    cost = self.cross_aisle_cost(cross_aisle_mapping[cross_id])
                    from_state = (j, prev_eq_class, "+")
                    to_state = (j+1, current_eq_class, "-")
                    self.state_graph.add_edge(from_state, to_state, weight=cost, action=cross_aisle_mapping[cross_id])

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

    def _construct_picker_tour(self) -> nx.MultiGraph:
        """
        Constructs a MultiGraph representing the picker tour based on the DP solution path.
        Uses self.path and self.state_graph edge attributes (action, action_node).
        Returns:
        nx.MultiGraph: constructed tour graph with nodes = (aisle, pick_y).
        """
        assert hasattr(self, "path") and self.path, "No DP path found."
        T = nx.MultiGraph()

        for i in range(len(self.path) - 1):
            from_state = self.path[i]
            to_state = self.path[i + 1]
            edge_data = self.state_graph.get_edge_data(from_state, to_state)

            current_aisle = from_state[0]
            action = edge_data.get("action")
            node_info = edge_data.get("action_node")

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
            aisle_orders = self._get_aisle_orders(current_aisle)
            pick_positions = sorted([pos.pick_node[1] for pos in aisle_orders])

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

    def get_item_sequence_from_path(self) -> list[PickPosition]:
        """
        Extracts the ordered pick sequence from the dynamic programming path (self.path).
        Returns:
            list[PickPosition]: ordered item sequence along the optimal tour.
        """
        assert hasattr(self, "path") and self.path, "No DP path found."
        picked_items = []

        for i in range(len(self.path) - 1):
            from_state = self.path[i]
            to_state = self.path[i + 1]
            edge_data = self.state_graph.get_edge_data(from_state, to_state)

            action = edge_data.get("action")
            action_node = edge_data.get("action_node")

            # Skip cross-aisle transitions
            if isinstance(action, tuple):
                continue

            transition_type = aisle_mapping.get(action)
            aisle = from_state[0]
            aisle_orders = self._get_aisle_orders(aisle)
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
                if aisle == 3:
                    print()
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
