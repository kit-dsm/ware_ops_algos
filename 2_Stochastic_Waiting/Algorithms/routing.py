import networkx as nx
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple

from Utils.utils import setup_matplotlib_backend
setup_matplotlib_backend()

from matplotlib import pyplot as plt
from collections import defaultdict



solution_total = []

class Routing(ABC):
    """Base class for routing algorithms."""
    def __init__(self,
                 network_graph: nx.Graph,
                 batched_list: pd.DataFrame,
                 distance_matrix: pd.DataFrame,
                 tour_matrix: pd.DataFrame,
                 picker,
                 time_to_pick: float = 1,
                 print_plot: bool = False,
                 batching_class: str = None,
                 batching_name: str = None,
                 **kwargs):

        self.batch_numbers = None
        self.graph = network_graph
        self.batched_list = batched_list
        self.distance_matrix = distance_matrix
        self.tour_matrix = tour_matrix
        self.picker = picker
        self.time_to_pick = time_to_pick
        self.print_plot = print_plot
        self.batching_class = batching_class
        self.batching_name = batching_name

        self.start_node = self._get_node_by_type('start_node')
        self.end_node = self._get_node_by_type('end_node')

        self.max_aisle_position = max(node[1] for node in self.graph.nodes)

        # Solution attributes
        self.tour = []
        self.item_sequence = []
        self.distance = 0
        self.total_distance = 0
        self.travel_time = 0
        self.path = []  # TODO fix this
        self.actions = []

        self.complete_solutions = []
        self.current_order = None
        self.batch_number = None
        self.routing_algo = None
        self._picker = None

    def _get_node_by_type(self, node_type: str) -> tuple:
        """Finds a node by type.
        :param node_type: the type of the node to find
        Returns the node if found, otherwise None."""
        return next((node for node, data in self.graph.nodes(data=True) if data.get('type') == node_type), None)

    def _reset_solution_parameters(self):
        """Resets output attributes for each batch."""
        self.tour = []
        self.item_sequence = []
        self.distance = 0
        self.travel_time = 0
        self.current_order = self.batched_list[self.batched_list['batch_number'] == self.batch_number]

    @abstractmethod
    def routing_algorithm(self):
        """Abstract method for routing algorithms."""
        pass

    def generate_routing(self):
        """Generates routing solutions for all batches.
        :param routing_algorithm: the routing algorithm to use"""
        self.batch_numbers = self.batched_list['batch_number'].unique()

        # if len(self.picker_list) == 1:
        #     self.picker = self.picker_list[0]
        #     self.generate_output_for_solution()
        #
        # else:
        #     for self.picker in self.picker_list:
        #         self.generate_output_for_solution()


        for self._picker in self.picker:
            self.generate_output_for_solution()

    def generate_output_for_solution(self):
        self.complete_solutions = []
        for self.batch_number in self.batch_numbers:
            batch_data = {'batch_number': self.batch_number, 'results': []}

            self._reset_solution_parameters()
            self.routing_algorithm()
            # print(self.distance)

            travel_time = self.distance / self._picker['speed'] + self.time_to_pick * self.batched_list['amount'].sum()
            order_in_batch = self.batched_list.loc[
                self.batched_list['batch_number'] == self.batch_number, 'order_number'].unique().tolist()

            batch_data['results'].append({
                'picker': self._picker['id'],
                'distance': self.distance,
                'batch_number': self.batch_number,
                'travel_time': travel_time,
                'routing_algorithm': self.routing_algo,
                'item_sequence': self.item_sequence,
                'number_of_items': len(self.item_sequence),
                'orders_in_batch': order_in_batch,
                'number_of_orders': len(order_in_batch),
                'tour': self.tour,
                'path': self.path,
                'actions': self.actions
            })

            self.add_output_to_solution(batch_data)

        # self.plot_solution()

    def add_output_to_solution(self, batch_data):
        """Generates the output for each batch.
        :param batch_data: the data for the current batch"""
        # Aktualisiere `solution_total`
        batch_found = False
        for solution in solution_total:
            if solution['batch_number'] == self.batch_number:
                solution['results'].extend(batch_data['results'])
                batch_found = True
                break

        if not batch_found:
            solution_total.append(batch_data)

        # Speichere die vollständigen Lösungen
        if self.batch_number not in [sol['batch_number'] for sol in self.complete_solutions]:
            self.complete_solutions.append(batch_data)

    # NOTE: Solution saving functionality has been removed.
    # If needed in the future, implement a method to save solutions to a file.

    def plot_solution(self):
        """Visualizes the routing output."""
        if self.print_plot:
            for j in range(len(self.complete_solutions)):
                pos = nx.get_node_attributes(self.graph, 'pos')
                plt.figure(figsize=(10, 8))
                nx.draw(self.graph, pos, with_labels=False, node_size=100)
                edges = [(self.complete_solutions[j]['results'][0]['tour'][i],
                          self.complete_solutions[j]['results'][0]['tour'][i + 1])
                         for i in range(len(self.complete_solutions[j]['results'][0]['tour']) - 1)]
                nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color='red', width=4, node_size=100)
                plt.title(f'Batch {self.batch_number} Routing Solution')
                plt.show()

class HeuristicRouting(Routing, ABC):
    """Base class for heuristic routing algorithms."""
    def __init__(self, network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs):
        super().__init__(network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs)

    def _initialize_routing(self) -> tuple:
        """Initializes routing by moving from start node to the first node in the aisle.
        Returns the current source node after initialization."""
        closest_node_to_start_point = list(self.graph.neighbors(self.start_node))[0]
        return self._walk_to_target(self.start_node, closest_node_to_start_point)

    def _determine_walking_direction(self, current_source: tuple) -> bool:
        """Determines if walking up or down.
        :param current_source: the current source node"""
        if current_source[1] == 0:
            return True  # Upward
        elif current_source[1] == self.max_aisle_position:
            return False  # Downward
        else:
            raise ValueError("Start node is not connected to the beginning or end of the aisle.")

    def _walk_to_target(self, source: tuple, target: tuple, target_is_pick_node: bool = False) -> tuple:
        """Walk to the target node using precomputed paths.
        :param source: the source node
        :param target: the target node
        :param target_is_pick_node: whether the target is a pick node
        Returns the target node after walking to it."""
        self.tour.extend(self.tour_matrix.at[source, target][:-1])
        self.distance += self.distance_matrix.at[source, target]

        if target_is_pick_node:
            for i in range(len( self.current_order[self.current_order['pick_node'] == target])):
                self.item_sequence.append(target)
            self.current_order = self.current_order[self.current_order['pick_node'] != target]

        return target

    def _walk_to_target_and_pick(self, source: tuple, target_y: list, walking_upwards: bool = None) -> \
    tuple:
        """Walk to each target position in the aisle and optionally to its end.
        :param source: the source node
        :param target_y: the target Y-values to visit as a list
        :param walking_upwards: whether to walk upwards or downwards, upwards if True, downwards if False. If walking_upwards is None there is no walking to the end of the aisle."""
        for next_position in target_y:
            source = self._walk_to_target(source, (source[0], next_position), target_is_pick_node=True)

        if walking_upwards is not None:
            last_position = self.max_aisle_position if walking_upwards else 1
            source = self._walk_to_target(source, (source[0], last_position))

        return source

    def _process_aisle(self, current_source: tuple, aisle_to_visit: int, walking_up: bool) -> tuple:
        """Handles the walking through an aisle.
        :param current_source: the current source node
        :param aisle_to_visit: the aisle to visit
        :param walking_up: whether to walk upwards or downwards, upwards if True, downwards if False
        Returns the current source node after completing the aisle
        """
        if current_source[0] == aisle_to_visit:
            target_y_values = self._get_sorted_y_values_for_current_aisle(current_source, walking_up)
            current_source = self._walk_to_target_and_pick(current_source, target_y_values, walking_up)
        else:
            target = (aisle_to_visit, current_source[1])
            current_source = self._walk_to_target(current_source, target)
        return current_source

    def _get_min_aisle(self) -> int:
        """Returns the next aisle containing items."""
        # self.current_order['pick_node'] = self.current_order['pick_node'].apply(ast.literal_eval)
        if min(self.current_order['pick_node'])[0] == "(":
            print()
        return min(self.current_order['pick_node'])[0]

    def _get_max_aisle(self) -> int:
        """Returns the next aisle containing items."""
        return max(self.current_order['pick_node'])[0]

    def _get_aisle_list(self, reverse = False) -> list:
        """
        Returns a list of unique aisles containing items to be picked.

        :param reverse: whether to return the aisles in reverse order. Default is False, i.e., aisles are returned in ascending order.
        """
        if reverse:
            return sorted(set(self.current_order['pick_node'].apply(lambda x: x[0])), reverse=True)

        return list(set(self.current_order['pick_node'].apply(lambda x: x[0])))

    def _get_sorted_y_values_for_current_aisle(self, current_source: tuple, walking_up: bool) -> list:
        """
        Returns the sorted Y-values for the current aisle based on walking direction.

        :param current_source: the current source node
        :param walking_up: whether to sort the Y-values in ascending order if True, descending order if False
        """
        aisle_y_values = [coord_x_y[1] for coord_x_y in self.current_order['pick_node'] if
                          coord_x_y[0] == current_source[0]]
        return sorted(aisle_y_values) if walking_up else sorted(aisle_y_values, reverse=True)

class SShapeRouting(HeuristicRouting):
    """Implements S-shape routing."""

    def __init__(self,
                 network_graph,
                 batched_list,
                 distance_matrix,
                 tour_matrix,
                 picker,
                 **kwargs):
        super().__init__(network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs)

        self.routing_algo = 'SShapeRouting'

    def routing_algorithm(self):
        current_source = self._initialize_routing()
        walking_up = not(self._determine_walking_direction(current_source))

        while not self.current_order.empty:
            aisle_min = self._get_min_aisle()

            if current_source[0] == aisle_min:
                walking_up = not walking_up  # Change direction at the end of the aisle

            current_source = self._process_aisle(current_source, aisle_min, walking_up)

        self._walk_to_target(current_source, self.end_node)

class ReturnRouting(HeuristicRouting):
    """Implements Return routing."""

    def __init__(self, network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs):
        super().__init__(network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs)

        self.routing_algo = 'ReturnRouting'

    def routing_algorithm(self):
        current_source = self._initialize_routing()
        walking_up = False

        while not self.current_order.empty:
            aisle_min = self._get_min_aisle()
            current_source = self._process_aisle(current_source, aisle_min, walking_up)

        current_source = self._walk_to_target(current_source, self.end_node)

class MidpointRouting(HeuristicRouting):
    """Implements Midpoint routing."""

    def __init__(self, network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs):
        super().__init__(network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs)

        self.routing_algo = 'MidpointRouting'

    def routing_algorithm(self):
        """
        Implements Midpoint Routing: handles lower aisles first, navigates back to upper aisles
        while ensuring that orders in different halves of the warehouse are handled appropriately.
        """
        current_source = self._initialize_routing()  # Start from the initial node
        mid_point = round(self.max_aisle_position / 2)  # Define the midpoint of the warehouse

        # Split orders based on their position relative to the midpoint
        orders_lower_half = self.current_order[
            self.current_order['pick_node'].apply(lambda x: x[1] <= mid_point)]
        orders_upper_half = self.current_order[
            self.current_order['pick_node'].apply(lambda x: x[1] > mid_point)]

        # Find the maximum aisle in the lower half of the warehouse
        max_aisle_lower_part = max(orders_lower_half['pick_node'].apply(lambda x: x[0]))

        # Process orders in the lower half of the warehouse
        self.current_order = orders_lower_half
        current_source = self._process_lower_half(current_source)

        # Process orders in the upper half of the warehouse
        self.current_order = orders_upper_half
        current_source = self._process_upper_half(current_source, max_aisle_lower_part)

        # Walk to the end node after completing all orders
        self._walk_to_target(current_source, self.end_node)

    def _process_lower_half(self, current_source):
        """
        Process all orders in the lower half of the warehouse (y <= mid_point).

        :param current_source: the current source node

        Returns the current source node after completing all lower orders.
        """
        while not self.current_order.empty:
            # Find the minimum aisle in the current order
            aisle_to_visit = self._get_min_aisle()

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
        current_source = self._process_transition_lower_to_upper_half(current_source, max_aisle_lower_part)

        while not self.current_order.empty:
            # Find the maximum aisle in the current order
            aisle_to_visit = self._get_max_aisle()

            # Process the current aisle by walking and picking items
            current_source = self._process_aisle(current_source, aisle_to_visit, walking_up=False)

        return current_source

    def _process_transition_lower_to_upper_half(self, current_source: Tuple[int, int], max_aisle_lower_part: int) -> Tuple[int, int]:
        """
        Process the transition from the lower to the upper half of the warehouse.

        :param current_source: the current source node
        :param max_aisle_lower_part: the maximum aisle in the lower half of the warehouse

        Returns the current source node after transitioning to the upper half.
        """
        if not self.current_order.empty:
            max_aisle_upper_part = self._get_max_aisle()

            if max_aisle_upper_part < max_aisle_lower_part:
                # Walk to the self.max_aisle_position of the current aisle
                current_source = self._walk_to_target(current_source, (current_source[0], self.max_aisle_position))
            elif max_aisle_upper_part == max_aisle_lower_part:
                # Process the current aisle by walking and picking items
                current_source = self._process_aisle(current_source, max_aisle_upper_part, walking_up=True)
            else:
                # Walk to the (max_aisle_upper_part, 1) position
                current_source = self._walk_to_target(current_source, (max_aisle_upper_part, 1))

        return current_source

class LargestGapRouting(HeuristicRouting):
    """
    Implements Largest Gap Routing for order picking in a warehouse.
    """

    def __init__(self, network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs):
        super().__init__(network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs)

        self.routing_algo = 'LargestGapRouting'

    def routing_algorithm(self):
        """
        Executes the Largest Gap Routing algorithm. This strategy identifies the largest gap between
        pick locations in an aisle and processes items accordingly to minimize walking distance.
        """
        # Initialize routing by moving into the storage area
        current_source = self._initialize_routing()

        aisle_max = self._get_max_aisle()
        aisle_list = self._get_aisle_list()

        for next_aisle in aisle_list:

            if current_source[0] is not next_aisle:
                current_source = self._walk_to_target(current_source, (next_aisle, current_source[1]))

            target_y_values = self._get_sorted_y_values_for_current_aisle(current_source, True)

            if aisle_max == next_aisle:
                current_source = self._walk_to_target_and_pick(current_source, target_y_values, True)
            else:
                pos_largest_gap = self._get_largest_gap_pos_inside_aisle(target_y_values)
                current_source = self._walk_to_target_and_pick(current_source, target_y_values[:pos_largest_gap],False)

        while not self.current_order.empty:
            next_aisle = self._get_max_aisle()
            current_source = self._process_aisle(current_source, next_aisle, True)

        # walk to the end node
        self._walk_to_target(current_source, self.end_node)

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

    def __init__(self, network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs):
        super().__init__(network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs)

        self.routing_algo = 'NearestNeighbourhoodRouting'

    def routing_algorithm(self):
        # walk from start node into the storage area
        current_source = self._initialize_routing()

        while not self.current_order.empty:
            nearest_pick_node = self._get_next_nearest_node_by_dijkstra(current_source)
            current_source = self._walk_to_target(current_source, nearest_pick_node, target_is_pick_node=True)

        # walk to the end node
        self._walk_to_target(current_source, self.end_node)

    def _get_next_nearest_node_by_dijkstra(self, current_source: tuple) -> tuple:
        """
        Determines the next nearest pick node using Dijkstra's algorithm.

        :param current_source: the current source node

        Returns the next nearest pick node.
        """
        return min(self.current_order['pick_node'], key=lambda item: nx.dijkstra_path_length(self.graph, current_source, item))

# class ExactRouting(Routing, ABC):
#     """
#     Base class for exact routing algorithms.
#     """
#     def __init__(self, network_graph, batched_list, distance_matrix, tour_matrix, picker,
#                  big_m: float = 1e3, objective: str = 'minimize_distance', **kwargs):
#         super().__init__(network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs)
#
#         self.objective = objective
#         self.big_m = big_m
#
#         ##############################
#         self.length = None
#         self.pick_nodes = None
#         self.travel_time_matrix_p = None
#         self.mdl = None
#         self.amount_at_pick_nodes = None
#
#     def _set_exact_routing_parameters(self):
#         """Sets the parameters for the exact routing algorithm."""
#         self.length = len(self.current_order)
#         self.pick_nodes = self.current_order['pick_node'].tolist()
#         self.amount_at_pick_nodes = self.current_order['amount'].tolist()
#         self.M = 1000
#         self.travel_time_matrix_p = self.distance_matrix / self._picker['speed']
#
# class ExactTSPRouting(ExactRouting):
#     """
#     Implements the exact routing algorithm for the Traveling Salesman Problem (TSP).
#     """
#
#     def __init__(self, network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs):
#         super().__init__(network_graph, batched_list, distance_matrix, tour_matrix, picker, **kwargs)
#
#         self.routing_algo = 'ExactTSPRouting'
#         self.x = None
#         self.x_start = None
#         self.x_end = None
#         self.T = None
#         self.C_max = None
#
#
#     def routing_algorithm(self):
#         """Solves the exact routing problem using the Gurobi optimization solver."""
#         # Set the parameters for the exact routing algorithm
#         self._set_exact_routing_parameters()
#
#         self.mdl = gp.Model(f"{self.routing_algo}")
#         self.mdl.setParam('OutputFlag', 0)
#
#         self.set_decision_variables()
#
#         self.set_objective()
#
#         self.set_constraint_each_node_visited_once()
#         self.set_constraint_only_one_start_and_end_node()
#         self.set_constraint_time()
#
#         self.mdl.optimize()
#
#         self.print_solution()
#         # print(self.batched_list)
#
#     def set_decision_variables(self):
#         """Set the decision variables for the exact routing model."""
#         self.x = self.mdl.addVars(self.length, self.length, vtype=GRB.BINARY, name="x")
#         self.x_start = self.mdl.addVars(self.length, vtype=GRB.BINARY, name="x0j")
#         self.x_end = self.mdl.addVars(self.length, vtype=GRB.BINARY, name="xj0")
#         self.T = self.mdl.addVars(self.length, vtype=GRB.CONTINUOUS, name="T")
#         self.C_max = self.mdl.addVars(self.length, lb=0, vtype=GRB.CONTINUOUS, name="C_max")
#
#     def set_constraint_each_node_visited_once(self):
#         """Set the constraint that each node is visited exactly once."""
#         for i in range(self.length):
#             self.mdl.addConstr(gp.quicksum(self.x[i, j] for j in range(self.length) if i != j) + self.x_end[i] == 1,
#                             name=f"constr1_{i}")
#             self.mdl.addConstr(gp.quicksum(self.x[j, i] for j in range(self.length) if j != i) + self.x_start[i] == 1,
#                             name=f"constr2_{i}")
#
#     def set_constraint_only_one_start_and_end_node(self):
#         """Set the constraint that only one start and end node is selected."""
#         self.mdl.addConstr(gp.quicksum(self.x_start[j] for j in range(self.length)) == 1, name="constr3")
#         self.mdl.addConstr(gp.quicksum(self.x_end[j] for j in range(self.length)) == 1, name="constr4")
#
#     def set_constraint_time(self):
#         """Set the constraint that the time is respected."""
#         for i in range(self.length):
#             for j in range(self.length):
#                 if i != j:
#                     self.mdl.addConstr(self.T[i] + self.travel_time_matrix_p[self.pick_nodes[i]][self.pick_nodes[j]]
#                                        <= self.T[j] + self.big_m * (1 - self.x[i, j]), name=f"constr5_{i}_{j}")
#
#             self.mdl.addConstr(self.C_max[i] >= self.T[i] + self.travel_time_matrix_p[self.pick_nodes[i]][self.end_node])
#
#     def set_objective(self):
#         """Set the objective function for the exact routing model."""
#         if self.objective == 'minimize_distance':
#             dist_x_i_x_j = gp.quicksum(self.distance_matrix.at[self.pick_nodes[i], self.pick_nodes[j]] * self.x[i, j]
#                                        for i in range(self.length) for j in range(self.length) if i != j)
#             dist_start_i = gp.quicksum(self.distance_matrix.at[self.start_node, self.pick_nodes[j]] * self.x_start[j]
#                                       for j in range(self.length))
#             dist_end_j = gp.quicksum(self.distance_matrix.at[self.pick_nodes[j], self.end_node] * self.x_end[j]
#                                     for j in range(self.length))
#             self.mdl.setObjective(dist_x_i_x_j + dist_start_i + dist_end_j, GRB.MINIMIZE)
#
#         elif self.objective == 'minimize_time':
#             time_x_i_x_j = gp.quicksum(self.travel_time_matrix_p.at[self.pick_nodes[i], self.pick_nodes[j]] * self.x[i, j] + self.amount_at_pick_nodes[j]
#                                        for i in range(self.length) for j in range(self.length) if i != j)
#             time_start_i = gp.quicksum(self.travel_time_matrix_p.at[self.start_node, self.pick_nodes[j]] * self.x_start[j] + self.amount_at_pick_nodes[j]
#                                        for j in range(self.length))
#             time_end_j = gp.quicksum(self.travel_time_matrix_p.at[self.pick_nodes[j], self.end_node] * self.x_end[j]
#                                      for j in range(self.length))
#             self.mdl.setObjective(time_x_i_x_j + time_start_i + time_end_j, GRB.MINIMIZE)
#
#         elif self.objective == 'minimize_maximum_completion_time':
#             self.mdl.setObjective(gp.max_(self.T), GRB.MINIMIZE)
#
#         else:
#             raise ValueError("The given objective function is not valid.")
#
#     def print_solution(self):
#         """Prints the solution of the exact routing model."""
#         # Ergebnisse ausgeben
#         if self.mdl.status == GRB.OPTIMAL:
#             print(f"Objective: {self.mdl.objVal}")
#             print("Routes:")
#             next_node = None
#
#             # get the start node
#             for j in range(self.length):
#                 if self.x_start[j].x > 0.5:
#                     next_node = j
#                     self.distance += self.distance_matrix.at[self.start_node, self.pick_nodes[j]]
#                     self.tour.extend(nx.dijkstra_path(self.graph, self.start_node, self.pick_nodes[j]))
#                     self.item_sequence.append(self.pick_nodes[j])
#                     break
#
#             # go to first item to pick from start node
#             for i in range(self.length):
#                 for j in range(self.length):
#                     if self.x[next_node, j].x > 0.5:
#                         self.item_sequence.append(self.pick_nodes[j])
#                         self.distance += self.distance_matrix.at[self.pick_nodes[next_node], self.pick_nodes[j]]
#                         self.tour.extend(nx.dijkstra_path(self.graph, self.pick_nodes[next_node], self.pick_nodes[j]))
#                         next_node = j
#
#             # go end node
#             for j in range(self.length):
#                 if self.x_end[j].x > 0.5:
#                     self.distance += self.distance_matrix.at[self.pick_nodes[j], self.end_node]
#                     self.tour.extend(nx.dijkstra_path(self.graph, self.pick_nodes[j], self.end_node))
#                     break
#
#         else:
#             print("No optimal Solution found")
#
# def do_routing(algo_class_and_name, *args, **kwargs):
#     """
#     Run the routing algorithm and print the results. Measure the execution time.
#
#     :param algo_class_and_name: the routing class and its name.
#                                 Tuple of (class, name), e.g.
#                                 (LargestGapRouting, "Largest Gap"),
#                                 (MidpointRouting, "Midpoint"),
#                                 (ReturnRouting, "Return"),
#                                 (SShapeRouting, "S-Shape"),
#                                 (ExactTSPRouting, "Exact TSP"),
#                                 (NearestNeighbourhood, "Nearest Neighbourhood")
#     :param args: the arguments for the routing class
#     :param kwargs: the keyword arguments for the routing class
#     """
#     routing_class, class_name = algo_class_and_name
#
#     print("************************************")
#     print(f"{class_name} Routing")
#     print("************************************")
#
#     # Instanziere die Routing-Strategie
#     routing_instance = routing_class(*args, **kwargs)
#
#     # Messe die Zeit und führe die Strategie aus
#     start_time = time.time()
#     routing_instance.generate_routing()
#     # print(routing_instance.distance)
#
#     end_time = time.time()
#
#     # Gib die Ergebnisse aus
#     solution = pd.DataFrame(routing_instance.complete_solutions)
#
#
#     # com_sol.append(routing_instance.complete_solutions)
#     # print(solution)
#
#
#     elapsed_time = end_time - start_time
#     print(f"Die Ausführungszeit '{class_name}' beträgt {elapsed_time:.2f} Sekunden.\n")

class AnalyticRouting(Routing, ABC):
    """
    Base class for analytical routing algorithms.
    Computes expected distances analytically instead of simulating a path on the graph.
    """

    def __init__(self,
                 network_graph: nx.Graph,
                 batched_list: pd.DataFrame,
                 distance_matrix: pd.DataFrame,
                 tour_matrix: pd.DataFrame,
                 picker,
                 time_to_pick: float = 1,
                 print_plot: bool = False,
                 batching_class: str = None,
                 batching_name: str = None,
                 **kwargs):
        super().__init__(network_graph, batched_list, distance_matrix, tour_matrix,
                         picker, time_to_pick, print_plot, batching_class, batching_name, **kwargs)

    @abstractmethod
    def routing_algorithm(self):
        """
        Must compute self.distance (expected distance),
        optionally travel time etc., but does NOT need to fill self.tour/path.
        """
        pass

class SShapeAnalyticRoutingAdvanced(AnalyticRouting):
    def __init__(self,
                 network_graph,
                 batched_list,
                 distance_matrix,
                 tour_matrix,
                 picker,
                 **kwargs):
        super().__init__(network_graph, batched_list, distance_matrix, tour_matrix,
                         picker, **kwargs)
        self.routing_algo = 'SShapeAnalyticRouting'
        # M = Anzahl Gassen (Aisles)
        self.M = self._get_number_of_aisles()
        # L = Länge einer Gasse und Anzahl der Knoten in einer Gasse aus dem Graph
        self.L, self.N_L = self._get_L_from_graph_weighted()
        # W = Abstand zwischen den Gassen aus dem Graph
        self.W = self._get_W_from_graph_weighted()

    def _get_number_of_aisles(self) -> int:
        # angenommen Nodes sind (aisle, position)
        aisles = {node[0] for node in self.graph.nodes
                  if isinstance(node, tuple) and len(node) == 2 and self.graph.nodes[node]['type']=='pick_node'}
        return len(aisles)

    def _get_L_from_graph_weighted(self) -> Tuple[float, int]:
        # Wähle irgendeine Gasse (z.B. aisle 0)
        aisle1_nodes = sorted(
            [n for n in self.graph.nodes if isinstance(n, tuple) and n[0] == 1
             and not (self.graph.nodes[n]['type'] == 'start_node' or self.graph.nodes[n]['type'] == 'end_node')],
            key=lambda x: x[1]
        )
        # Lauf die y-Koordinate hoch und summiere die Kantenlängen
        L = 0.0
        N_L = 0
        for u, v in zip(aisle1_nodes[:-1], aisle1_nodes[1:]):
            edge_data = self.graph.get_edge_data(u, v)
            w = edge_data.get('weight', 1.0)
            L += w
            N_L += 1
        return L, N_L

    def _get_W_from_graph_weighted(self) -> float:
        """
            Determines the horizontal distance W between adjacent aisles based on the graph structure and edge weights.

              - Find all aisle IDs and sort
              - Take two adjacent aisles (e.g., a1 and a2).
              - Find node with lowest position in these aisles
              - Search an edge between the found nodes
              - Return W
            """

        aisles = sorted({
            node[0] for node in self.graph.nodes
            if isinstance(node, tuple) and len(node) == 2
               and not (self.graph.nodes[node]['type'] == 'start_node' or self.graph.nodes[node]['type'] == 'end_node')
        })

        if len(aisles) < 2:
            W = 0.0
            print("Warning: Less than 2 aisles found, setting W=0.0")
            return W

        a1, a2 = aisles[0], aisles[1]

        def min_position_node(aisle_id):
            nodes_in_aisle = [
                n for n in self.graph.nodes
                if isinstance(n, tuple)
                   and len(n) == 2
                   and n[0] == aisle_id
                   and not (self.graph.nodes[n]["type"] == 'start_node' or self.graph.nodes[n]["type"] == 'end_node')
            ]
            if not nodes_in_aisle:
                raise ValueError(f"Keine pick_nodes in Aisle {aisle_id} gefunden.")
            return min(nodes_in_aisle, key=lambda x: x[1])

        n1 = min_position_node(a1)
        n2 = min_position_node(a2)

        edge_data = self.graph.get_edge_data(n1, n2)
        if edge_data is None:
            edge_data = self.graph.get_edge_data(n2, n1)
        if edge_data is None:
            raise ValueError(
                f"Found no edge between {n1} and {n2}.")

        W = edge_data.get('weight')
        return W


    def _expected_picks_aisles_opened_last_index(self, picks):
        """
        Markov chain with states (k,h,y):
          k = number of aisles used
          h = highest aisle index opened (0-based)
          y = picks in aisle h
        Returns:
          states: defaultdict((k,h,y) -> prob)
          distance_total: expected total distance (vertikal + horizontal)
          expected_highest_aisle: E[h+1]
        """
        M = self.M
        L = self.L
        W = self.W


        states = defaultdict(float)

        # --- first pick ---
        # in any of M aisles
        for j in range(M):
            states[(1, j, 1)] += 1.0 / M

        # --- remaining Picks ---
        for _ in range(picks - 1):

            new_states = defaultdict(float)
            for (k, h, y), prob in states.items():
                # 1) Pick in last open aisle h
                new_states[(k, h, y + 1)] += prob * (1.0 / M)

                # 2) Pick in already open aisle < h
                if k > 1:
                    new_states[(k, h, y)] += prob * ((k - 1.0) / M)

                # 3) Pick in new, not yet opened aisle < h
                unopened_below = h - (k - 1)
                if unopened_below > 0:
                    new_states[(k + 1, h, y)] += prob * (unopened_below / M)

                # 4) Pick in new, not yet opened aisle > h
                for j in range(h + 1, M):
                    new_states[(k + 1, j, 1)] += prob * (1.0 / M)

            states = new_states
            # print(f"{_}/{picks - 1} picks processed, number of states: {len(states)}")
            # print(states)

        def expected_max_aisle(L, y):

            return sum(
                k * ((k / L) ** y - ((k - 1) / L) ** y)
                for k in range(1, L + 1)
            )

        # Accumulators for statistics
        total_prob_odd = 0.0
        total_prob_even = 0.0
        sum_distance1_odd = 0.0
        sum_distance1_even = 0.0
        sum_distance2_total = 0.0
        sum_y_given_k_odd = 0.0
        sum_y_total = 0.0
        total_prob_total = 0.0
        total_distance_r = 0.0

        #print("Probability table with distances:")
        for (k, h, y), prob in sorted(states.items()):
            # --- Distance1: vertical distance ---
            if k % 2 == 0:
                distance1 = k * L
                total_prob_even += prob
                sum_distance1_even += prob * distance1
            else:

                #Advanced Variante mit präziserer Berechnung der letzten Gasse
                distance1 = (k - 1) * L + 2 * expected_max_aisle(self.N_L - 1, y)
                total_prob_odd += prob
                sum_distance1_odd += prob * distance1
                sum_y_given_k_odd += prob * y

            total_distance_r += prob * distance1

            # --- Distance2: horizontal distance ---
            distance2 = 2 * (h + 1) * W
            sum_distance2_total += prob * distance2

            # Statistik
            sum_y_total += prob * y
            total_prob_total += prob

            # print(
            #     f"  k={k:2d}, h={h + 1:2d}, y={y:2d} | "
            #     f"P={prob:.8f}, distance1={distance1:.6f}, distance2={distance2:.6f}"
            # )

        # expected values
        expected_distance1_odd = sum_distance1_odd / total_prob_odd if total_prob_odd > 0 else 0.0
        expected_distance1_even = sum_distance1_even / total_prob_even if total_prob_even > 0 else 0.0
        avg_y_given_k_odd = sum_y_given_k_odd / total_prob_odd if total_prob_odd > 0 else 0.0
        avg_y_total = sum_y_total / total_prob_total if total_prob_total > 0 else 0.0

        # expected highest aisle index opened
        expected_highest_aisle = sum((h + 1) * prob for (k, h, y), prob in states.items())

        # print(f"\nAverage highest index of aisles opened ≈ {expected_highest_aisle:.6f}")
        # print("\nExpected distance1 totals (conditional on k odd/even):")
        # print(f"  Expected distance1 | k odd  ≈ {expected_distance1_odd:.6f}")
        # print(f"  Expected distance1 | k even ≈ {expected_distance1_even:.6f}")
        # print(f"  Expected distance1 | total ≈ {total_distance_r:.6f}")
        # print(f"\nExpected distance2 (all states) ≈ {sum_distance2_total:.6f}")
        # print(f"Average picks in last | k odd ≈ {avg_y_given_k_odd:.6f}")
        # print(f"Average picks in last | all states ≈ {avg_y_total:.6f}")

        distance_total = total_distance_r + sum_distance2_total
        return states, distance_total, expected_highest_aisle

    def _expected_value_augmented_markov_full(self, picks: int) -> float:
        """
        Wrapper: returns expected total distance
        """
        _, distance_total, _ = self._expected_picks_aisles_opened_last_index(picks)
        return distance_total

    def routing_algorithm(self):
        # picks = Nr items in same batch
        picks_in_batch = len(self.current_order)
        if picks_in_batch == 0:
            self.distance = 0.0
            return

        self.distance = self._expected_value_augmented_markov_full(picks_in_batch)

        self.tour = []
        self.path = []
        self.actions = []
        self.item_sequence = list(self.current_order['pick_node'])

class SShapeAnalyticRouting(AnalyticRouting):
    def __init__(self,
                 network_graph,
                 batched_list,
                 distance_matrix,
                 tour_matrix,
                 picker,
                 **kwargs):
        super().__init__(network_graph, batched_list, distance_matrix, tour_matrix,
                         picker, **kwargs)
        self.routing_algo = 'SShapeAnalyticRouting'
        # M = Anzahl Gassen (Aisles)
        self.M = self._get_number_of_aisles()
        # L = Länge einer Gasse und Anzahl der Knoten in einer Gasse aus dem Graph
        self.L, _ = self._get_L_from_graph_weighted()
        # W = Abstand zwischen den Gassen aus dem Graph
        self.W = self._get_W_from_graph_weighted()

    def _get_number_of_aisles(self) -> int:
        # angenommen Nodes sind (aisle, position)
        aisles = {node[0] for node in self.graph.nodes
                  if isinstance(node, tuple) and len(node) == 2 and self.graph.nodes[node]['type'] == 'pick_node'}
        return len(aisles)

    def _get_L_from_graph_weighted(self) -> Tuple[float, int]:
        # Wähle irgendeine Gasse (z.B. aisle 0)
        aisle1_nodes = sorted(
            [n for n in self.graph.nodes if isinstance(n, tuple) and n[0] == 1
             and not (self.graph.nodes[n]['type'] == 'start_node' or self.graph.nodes[n]['type'] == 'end_node')],
            key=lambda x: x[1]
        )
        # Lauf die y-Koordinate hoch und summiere die Kantenlängen
        L = 0.0
        N_L = 0
        for u, v in zip(aisle1_nodes[:-1], aisle1_nodes[1:]):
            edge_data = self.graph.get_edge_data(u, v)
            w = edge_data.get('weight', 1.0)
            L += w
            N_L += 1
        return L, N_L

    def _get_W_from_graph_weighted(self) -> float:
        """
            Determines the horizontal distance W between adjacent aisles based on the graph structure and edge weights.

              - Find all aisle IDs and sort
              - Take two adjacent aisles (e.g., a1 and a2).
              - Find node with lowest position in these aisles
              - Search an edge between the found nodes
              - Return W
            """

        aisles = sorted({
            node[0] for node in self.graph.nodes
            if isinstance(node, tuple) and len(node) == 2
               and not (self.graph.nodes[node]['type'] == 'start_node' or self.graph.nodes[node][
                'type'] == 'end_node')
        })

        if len(aisles) < 2:
            W = 0.0
            print("Warning: Less than 2 aisles found, setting W=0.0")
            return W

        a1, a2 = aisles[0], aisles[1]

        def min_position_node(aisle_id):
            nodes_in_aisle = [
                n for n in self.graph.nodes
                if isinstance(n, tuple)
                   and len(n) == 2
                   and n[0] == aisle_id
                   and not (self.graph.nodes[n]["type"] == 'start_node' or self.graph.nodes[n][
                    "type"] == 'end_node')
            ]
            if not nodes_in_aisle:
                raise ValueError(f"Keine pick_nodes in Aisle {aisle_id} gefunden.")
            return min(nodes_in_aisle, key=lambda x: x[1])

        n1 = min_position_node(a1)
        n2 = min_position_node(a2)

        edge_data = self.graph.get_edge_data(n1, n2)
        if edge_data is None:
            edge_data = self.graph.get_edge_data(n2, n1)
        if edge_data is None:
            raise ValueError(
                f"Found no edge between {n1} and {n2}.")

        W = edge_data.get('weight')
        return W

    def _expected_picks_aisles_opened_last_index(self, picks):
        """
        Markov chain with states (k,h,y):
          k = number of aisles used
          h = highest aisle index opened (0-based)
          y = picks in aisle h
        Returns:
          states: defaultdict((k,h,y) -> prob)
          distance_total: expected total distance (vertikal + horizontal)
          expected_highest_aisle: E[h+1]
        """
        M = self.M
        L = self.L
        W = self.W

        states = defaultdict(float)

        # --- first pick ---
        # in any of M aisles
        for j in range(M):
            states[(1, j, 1)] += 1.0 / M

        # --- remaining Picks ---
        for _ in range(picks - 1):

            new_states = defaultdict(float)
            for (k, h, y), prob in states.items():
                # 1) Pick in last open aisle h
                new_states[(k, h, y + 1)] += prob * (1.0 / M)

                # 2) Pick in already open aisle < h
                if k > 1:
                    new_states[(k, h, y)] += prob * ((k - 1.0) / M)

                # 3) Pick in new, not yet opened aisle < h
                unopened_below = h - (k - 1)
                if unopened_below > 0:
                    new_states[(k + 1, h, y)] += prob * (unopened_below / M)

                # 4) Pick in new, not yet opened aisle > h
                for j in range(h + 1, M):
                    new_states[(k + 1, j, 1)] += prob * (1.0 / M)

            states = new_states
            # print(f"{_}/{picks - 1} picks processed, number of states: {len(states)}")
            # print(states)

        # Accumulators for statistics
        total_prob_odd = 0.0
        total_prob_even = 0.0
        sum_distance1_odd = 0.0
        sum_distance1_even = 0.0
        sum_distance2_total = 0.0
        sum_y_given_k_odd = 0.0
        sum_y_total = 0.0
        total_prob_total = 0.0
        total_distance_r = 0.0

        # print("Probability table with distances:")
        for (k, h, y), prob in sorted(states.items()):
            # --- Distance1: vertical distance ---
            if k % 2 == 0:
                distance1 = k * L
                total_prob_even += prob
                sum_distance1_even += prob * distance1
            else:

                # einfachere Variante ohne expected_max_aisle
                distance1 = k * L + L
                sum_distance1_odd += prob * distance1
                sum_y_given_k_odd += prob * y

            total_distance_r += prob * distance1

            # --- Distance2: horizontal distance ---
            distance2 = 2 * (h + 1) * W
            sum_distance2_total += prob * distance2

            # Statistik
            sum_y_total += prob * y
            total_prob_total += prob

            # print(
            #     f"  k={k:2d}, h={h + 1:2d}, y={y:2d} | "
            #     f"P={prob:.8f}, distance1={distance1:.6f}, distance2={distance2:.6f}"
            # )

        # expected values
        expected_distance1_odd = sum_distance1_odd / total_prob_odd if total_prob_odd > 0 else 0.0
        expected_distance1_even = sum_distance1_even / total_prob_even if total_prob_even > 0 else 0.0
        avg_y_given_k_odd = sum_y_given_k_odd / total_prob_odd if total_prob_odd > 0 else 0.0
        avg_y_total = sum_y_total / total_prob_total if total_prob_total > 0 else 0.0

        # expected highest aisle index opened
        expected_highest_aisle = sum((h + 1) * prob for (k, h, y), prob in states.items())

        # print(f"\nAverage highest index of aisles opened ≈ {expected_highest_aisle:.6f}")
        # print("\nExpected distance1 totals (conditional on k odd/even):")
        # print(f"  Expected distance1 | k odd  ≈ {expected_distance1_odd:.6f}")
        # print(f"  Expected distance1 | k even ≈ {expected_distance1_even:.6f}")
        # print(f"  Expected distance1 | total ≈ {total_distance_r:.6f}")
        # print(f"\nExpected distance2 (all states) ≈ {sum_distance2_total:.6f}")
        # print(f"Average picks in last | k odd ≈ {avg_y_given_k_odd:.6f}")
        # print(f"Average picks in last | all states ≈ {avg_y_total:.6f}")

        distance_total = total_distance_r + sum_distance2_total
        return states, distance_total, expected_highest_aisle

    def _expected_value_augmented_markov_full(self, picks: int) -> float:
        """
        Wrapper: returns expected total distance
        """
        _, distance_total, _ = self._expected_picks_aisles_opened_last_index(picks)
        return distance_total

    def routing_algorithm(self):
        # picks = Nr items in same batch
        picks_in_batch = len(self.current_order)
        if picks_in_batch == 0:
            self.distance = 0.0
            return

        self.distance = self._expected_value_augmented_markov_full(picks_in_batch)

        self.tour = []
        self.path = []
        self.actions = []
        self.item_sequence = list(self.current_order['pick_node'])


def run_routing_algo_for_batching(algo_class_and_name, *args, **kwargs):
    """
    Run the routing algorithm and print the results. Measure the execution time.

    :param algo_class_and_name: the routing class and its name.
                                Tuple of (class, name), e.g.
                                (LargestGapRouting, "Largest Gap"),
                                (MidpointRouting, "Midpoint"),
                                (ReturnRouting, "Return"),
                                (SShapeRouting, "S-Shape"),
                                (ExactTSPRouting, "Exact TSP"),
                                (NearestNeighbourhood, "Nearest Neighbourhood")
    :param args: the arguments for the routing class
    :param kwargs: the keyword arguments for the routing class

    Returns the distance of the order with the given routing strategy.
    """
    routing_class, class_name = algo_class_and_name

    # Instanziere die Routing-Strategie
    routing_instance = routing_class(*args, **kwargs)
    routing_instance.generate_routing()
    # print(routing_instance.distance)

    return routing_instance.distance
