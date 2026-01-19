import time

import pandas as pd
import numpy as np
from abc import ABC
from gurobipy import *
import gurobipy as gp

from ware_ops_algos.algorithms.algorithm import RoutingSolution, Route, CombinedRoutingSolution, PickPosition
from ware_ops_algos.domain_models import Resource

from ware_ops_algos.algorithms.routing import Routing


class RoutingBatchingAssigning(Routing, ABC):
    """Base class for routing algorithms.
    :param network_graph: the network graph
    :param order_list: list of orders or just one order, each order is a dictionary with at least keys 'batch_number' or 'order_number'
    :param distance_matrix: the distance matrix
    :param picker_list: list[{'id': int, 'speed': m/sec, 'time_to_pick': sec, 'capacity': int}], can be a list of pickers or a single picker
    :param kwargs: additional keyword arguments"""

    def __init__(self,
                 start_node: tuple[int, int],
                 end_node: tuple[int, int],
                 distance_matrix: pd.DataFrame,
                 predecessor_matrix: np.array,
                 picker: list[Resource],
                 gen_tour: bool = False,
                 gen_item_sequence: bool = False,
                 time_limit: int | None =None,
                 **kwargs):

        super().__init__(
            start_node=start_node,
            end_node=end_node,
            closest_node_to_start=(0,0),
            min_aisle_position=0,
            max_aisle_position=0,
            distance_matrix=distance_matrix,
            predecessor_matrix=predecessor_matrix,
            picker=picker,
            gen_tour=gen_tour,
            gen_item_sequence=gen_item_sequence,
            **kwargs
        )

        self.item_sequence = []
        self.route = []
        self.distance_per_batch = None # Placeholder for distances
        self.solution = None

        self.current_order = None
        self.pick_list = None
        self.routing_algo = None



        self.list_item_pick_locations = None
        self.list_item_numbers = None
        self.list_item_amounts = None
        self.list_order_numbers = None

        self.x_start = None  # Start decision variables
        self.x_end = None  # End decision variables
        self.x = None  # Routing decision variables
        self.w = None  # Assignment decision variables
        self.T = None  # Time decision variables
        self.y = None  # Order assignment decision variables
        self.z = None  # Item assignment decision variables
        self.C_bp = None  # Completion time per batch
        self.C_max = None  # Maximum completion time
        self.C_max_p = None  # Maximum completion time per picker

        self.range_item_number = None  # Range of item numbers
        self.set_order_numbers = None  # Set of order numbers
        self.set_batch_numbers = None  # Set of batch numbers



        self.gen_tour = gen_tour  # Generate tour
        self.gen_item_sequence = gen_item_sequence  # Generate item sequence
        self.time_limit = time_limit


    def construct_route_and_sequence(self, b):
        """Generates the route and/or item sequence from solution variables."""
        self.route = []
        self.item_sequence = []

        if not (self.gen_tour or self.gen_item_sequence):
            return

        current_index = None

        # Startknoten finden
        for i in self.range_item_number:
            if self.x_start[i, b].x > 0.5:
                if self.gen_tour:
                    self._get_route_for_tour(self.start_node, self.list_item_pick_locations[i])
                if self.gen_item_sequence:
                    self.item_sequence.append(self.list_item_pick_locations[i])
                current_index = i
                break  # Nur ein Startpunkt

        if current_index is None:
            print("⚠️ Kein Startpunkt gefunden – möglicherweise ungültige Lösung.")
            return

        # Iterativ durch die Pickknoten
        visited = {current_index}
        while True:
            found = False
            for j in self.range_item_number:
                if j in visited:
                    continue
                if self.x[current_index, j, b].x > 0.5:
                    if self.gen_tour:
                        self._get_route_for_tour(self.list_item_pick_locations[current_index],
                                                 self.list_item_pick_locations[j])
                    if self.gen_item_sequence:
                        self.item_sequence.append(self.list_item_pick_locations[j])
                    visited.add(j)
                    current_index = j
                    found = True
                    break
            if not found:
                break

        # Endknoten anhängen
        for i in self.range_item_number:
            if self.x_end[i, b].x > 0.5:
                if self.gen_tour:
                    self._get_route_for_tour(self.list_item_pick_locations[current_index], self.end_node,
                                             with_last_element=True)
                break


class ExactTSPBatchingAndRoutingDistance(RoutingBatchingAssigning, ABC):
    """
    Implements the exact routing algorithm for the Traveling Salesman Problem (TSP) with batching.
    This class extends ExactTSPRouting to handle batching of orders.
    """

    def __init__(self, start_node, end_node, distance_matrix, predecessor_matrix, picker, gen_tour, gen_item_sequence, big_m, **kwargs):
        super().__init__(start_node, end_node, distance_matrix, predecessor_matrix, picker,
                         gen_tour, gen_item_sequence, **kwargs)

        self.big_m = big_m
        self.routing_algo = 'ExactTSPBatchingAndRouting'
        self.mdl = gp.Model(self.routing_algo)

    def _run(self, pick_list: list[PickPosition] = None):

        """Generates the route for the exact routing algorithm from solution variables."""

        self.pick_list = pick_list

        start_time = time.time()
        self._set_routing_parameters()
        self._set_decision_variables()
        self._set_objective()
        self._set_constraints()

        if self.time_limit is not None:
            self.mdl.Params.TimeLimit = self.time_limit
        self.mdl.optimize()

        execution_time = time.time() - start_time
        routes = []
        if self.mdl.status == GRB.OPTIMAL or (self.time_limit and self.mdl.SolCount > 0):
            self.solution = {
                'picker': {}
            }
            for p in self.picker_id:
                self.solution['picker'][p] = {'batches': {}}
                for b in self.set_batch_numbers:
                    if self.w[b, p].x > 0.5:  # aktivierte Zuordnung

                        self.construct_route_and_sequence(b)
                        # self.distance_per_batch[b] = self.C_bp[b, p].x
                        # Speichern
                        self.solution['picker'][p]['batches'][b] = {
                            'order_in_batch': [o for o in self.set_order_numbers if self.y[o, b].x > 0.5],
                            'item_sequence': self.item_sequence,
                            'route': self.route,
                            'distance': self.C_bp[b, p].x,  # optional: du kannst hier auch die echte Distanz berechnen
                            'execution_time': execution_time
                        }
                        self.distance = self.C_bp[b, p].x

                        route = Route(route=self.route, item_sequence=self.item_sequence, distance=self.distance,
                                      order_numbers=[o for o in self.set_order_numbers if self.y[o, b].x > 0.5])
                        routes.append(route)

            return CombinedRoutingSolution(algo_name=self.algo_name, routes=routes)
        else:
            self.solution = {'status': 'infeasible', 'message': 'No optimal solution found.'}
            print("No optimal solution found. Status:", self.mdl.status)
        sol = CombinedRoutingSolution(routes=routes, algo_name=self.algo_name)
        return sol



    def _set_routing_parameters(self):
        """Sets the parameters for the exact routing algorithm."""

        self.list_item_pick_locations = [item.pick_node for item in self.pick_list]
        self.list_item_numbers = [item.position.article_id for item in self.pick_list]
        self.list_item_amounts = [item.position.amount for item in self.pick_list]
        self.list_order_numbers = [item.position.order_number for item in self.pick_list]

        self.set_order_numbers = set(self.list_order_numbers)
        self.set_batch_numbers = range(len(self.set_order_numbers))

        self.len_item_numbers = len(self.list_item_numbers)

        self.range_item_number = range(self.len_item_numbers)


        self.picker_id =  [p.id for p in self.picker]
        self.picker_capa = [p.capacity for p in self.picker]
        self.picker_speed = [p.speed for p in self.picker]

        # Amount items in each order
        df = pd.DataFrame([{
            'order_number': item.position.order_number,
            'amount': item.position.amount
        } for item in self.pick_list])

        self.s_i = df.groupby('order_number')['amount'].sum().to_dict()

        # Travel time cost
        self.c = {(i, j, p.id): self.distance_matrix[self.list_item_pick_locations[i]][self.list_item_pick_locations[j]]
                    for i in self.range_item_number for j in self.range_item_number for p in self.picker}
        self.c_start = { (j, p.id): self.distance_matrix[self.start_node][self.list_item_pick_locations[j]]
                        for j in self.range_item_number for p in self.picker}
        self.c_end = { (j, p.id): self.distance_matrix[self.list_item_pick_locations[j]][self.end_node]
                        for j in self.range_item_number for p in self.picker}

        self.u_i_j = {i: {j: self.list_item_amounts[j]
        if self.list_item_numbers[j] in self.list_item_numbers and self.list_order_numbers[j] == i
        else 0 for j in self.range_item_number} for i in self.set_order_numbers}

        print()


    def _set_decision_variables(self):
        """Set the decision variables for the exact routing model."""
        self.x = self.mdl.addVars(self.range_item_number, self.range_item_number, self.set_batch_numbers, vtype=GRB.BINARY, name="x")
        self.x_start = self.mdl.addVars(self.range_item_number, self.set_batch_numbers,vtype=GRB.BINARY, name="x0j")
        self.x_end = self.mdl.addVars(self.range_item_number, self.set_batch_numbers,vtype=GRB.BINARY, name="xj0")
        self.w = self.mdl.addVars(self.set_batch_numbers, self.picker_id, vtype=GRB.BINARY, name="w")
        self.T = self.mdl.addVars(self.range_item_number, self.set_batch_numbers, self.picker_id, vtype=GRB.CONTINUOUS, name="T")
        self.y = self.mdl.addVars(self.set_order_numbers, self.range_item_number, vtype=GRB.BINARY, name="y")
        self.z = self.mdl.addVars(self.range_item_number, self.set_batch_numbers,vtype=GRB.BINARY, name="z")
        self.C_bp = self.mdl.addVars(self.set_batch_numbers, self.picker_id, vtype=GRB.CONTINUOUS, lb=0, name="C_bp")
        self.C_max = self.mdl.addVar(lb=0, vtype=GRB.CONTINUOUS, name="C_max")
        self.C_max_p = self.mdl.addVars(self.picker_id, vtype=GRB.CONTINUOUS, lb=0, name="C_max_p")

        #self.help = self.mdl.addVar(vtype=GRB.BINARY, name="help")

    def _set_objective(self):
        self.mdl.setObjective(gp.quicksum(self.distance_matrix[self.start_node][self.list_item_pick_locations[j]] * self.x_start[j, b]
                                          for j in self.range_item_number for b in self.set_batch_numbers) +
                              gp.quicksum(self.distance_matrix[self.list_item_pick_locations[j]][self.end_node] * self.x_end[j, b]
                                          for j in self.range_item_number for b in self.set_batch_numbers) +
                                gp.quicksum(self.distance_matrix[self.list_item_pick_locations[i]][self.list_item_pick_locations[j]] * self.x[i, j, b]
                                            for i in self.range_item_number for j in self.range_item_number for b in self.set_batch_numbers if i != j)+
                              gp.quicksum(self.C_max_p[p] * 0.001 for p in self.picker_id), GRB.MINIMIZE)

    def _set_constraints(self):
        """Set the constraints for the exact routing model."""
        # Each item is picked exactly once
        for j in self.range_item_number:
            self.mdl.addConstr(gp.quicksum(gp.quicksum(
                self.x[i, j, b]
                for i in self.range_item_number if i != j) + self.x_start[j, b]  for b in self.set_batch_numbers)
                               == 1)
            self.mdl.addConstr(gp.quicksum(gp.quicksum(
                self.x[j, i, b]
                 for i in self.range_item_number if i != j) + self.x_end[j, b] for b in self.set_batch_numbers)
                               == 1)

        # # Each item is connected to the next item in the batch
        for j in self.range_item_number:
            for b in self.set_batch_numbers:
                self.mdl.addConstr(
                    gp.quicksum(self.x[j, i, b] for i in self.range_item_number if i != j) + self.x_end[j, b]
                    == gp.quicksum(self.x[i, j, b] for i in self.range_item_number if i != j) + self.x_start[j, b])

        for b in self.set_batch_numbers:
            self.mdl.addConstr(gp.quicksum(self.w[b, p] for p in self.picker_id) <= 1)

            for i in self.set_order_numbers:
                self.mdl.addConstr(gp.quicksum(self.y[i, b] for i in self.set_order_numbers) <= self.big_m * gp.quicksum(self.w[b, p] for p in self.picker_id))
        # Only one starting and one ending point per batch
        for b in self.set_batch_numbers:
            self.mdl.addConstr(gp.quicksum(self.x_start[j, b] for j in self.range_item_number) == gp.quicksum(self.w[b, p] for p in self.picker_id))
            self.mdl.addConstr(gp.quicksum(self.x_end[j, b] for j in self.range_item_number) == gp.quicksum(self.w[b, p] for p in self.picker_id))
            # self.mdl.addConstr(gp.quicksum(gp.quicksum(self.x_end[j, b, p] for j in self.range_item_number) for p in self.picker_id) == 1)

        # Each order is assigned to one batch
        for o in self.set_order_numbers:
            self.mdl.addConstr(gp.quicksum(self.y[o, b] for b in self.set_batch_numbers) == 1)

        # Start/Ende nur, wenn Auftrag im Batch ist
        for b in self.set_batch_numbers:
            for o in self.set_order_numbers:
                self.mdl.addConstr(gp.quicksum(self.x_start[j, b] for j in self.range_item_number if self.u_i_j[o][j] > 0) <= self.y[o, b])
                self.mdl.addConstr(gp.quicksum(self.x_end[j, b] for j in self.range_item_number if self.u_i_j[o][j] > 0) <= self.y[o, b])

        # Artikelzuordnung auf Batches konsistent mit Aufträgen
        for b in self.set_batch_numbers:
            for o in self.set_order_numbers:
                self.mdl.addConstr(gp.quicksum(self.u_i_j[o][j] * self.z[j, b] for j in self.range_item_number) == self.s_i[o] * self.y[o, b])

        # Routingkanten dürfen nur existieren, wenn Artikel im Batch
        for b in self.set_batch_numbers:
            for j1 in self.range_item_number:
                self.mdl.addConstr(gp.quicksum(self.x[j1, j2, b] for j2 in self.range_item_number if j1 != j2) <= self.z[j1, b])

            for j2 in self.range_item_number:
                self.mdl.addConstr(gp.quicksum(self.x[j1, j2, b] for j1 in self.range_item_number if j1 != j2) <= self.z[j2, b])

        # Kapazitätsbeschränkung pro Picker
        for index_p, p in enumerate(self.picker_id):
            for b in self.set_batch_numbers:
                self.mdl.addConstr(gp.quicksum(self.s_i[i] * self.y[i, b] * self.w[b, p] for i in self.set_order_numbers) <= self.picker_capa[index_p])

        # Zeitplanung (Reisedauer)
        for j1 in self.range_item_number:
            for j2 in self.range_item_number:
                for b in self.set_batch_numbers:
                    for p in self.picker_id:
                        self.mdl.addConstr(self.T[j1, b, p] + self.distance_matrix.at[self.list_item_pick_locations[j1], self.list_item_pick_locations[j2]] <= self.T[j2, b, p] + self.big_m * (1 - self.x[j1, j2, b] * self.w[b, p]))

        # Start- und Endzeit pro Picker und Batch
        for j in self.range_item_number:
            for b in self.set_batch_numbers:
                for p in self.picker_id:
                    # Start time
                    self.mdl.addConstr(self.distance_matrix[self.start_node][self.list_item_pick_locations[j]] * self.x_start[j, b] * self.w[b, p] <= self.T[j, b, p])
                    # End time
                    self.mdl.addConstr(self.T[j, b, p] + self.distance_matrix[self.list_item_pick_locations[j]][self.end_node] * self.x_end[j, b] * self.w[b, p] <= self.C_bp[b, p])


        # Abschlusszeiten aggregieren
        for p in self.picker_id:
            self.mdl.addConstr(self.C_max_p[p] >= gp.quicksum(self.C_bp[b, p] for b in self.set_batch_numbers))
            #self.mdl.addConstr(self.C_max >= self.C_max_p[p])

class ExactTSPBatchingAndRoutingMaxCompletionTime(RoutingBatchingAssigning):
    """
    Implements the exact routing algorithm for the Traveling Salesman Problem (TSP) with batching.
    This class extends ExactTSPRouting to handle batching of orders.
    """

    def __init__(self, start_node, end_node, distance_matrix, predecessor_matrix,
                 picker, gen_tour, gen_item_sequence, big_m: int, **kwargs):
        super().__init__(start_node, end_node, distance_matrix, predecessor_matrix,
                         picker, gen_tour, gen_item_sequence, **kwargs)

        self.routing_algo = 'ExactTSPBatchingAndRouting'
        self.mdl = gp.Model(self.routing_algo)

        self.big_m = big_m



    def _run(self, pick_list: list[PickPosition] = None):

        """Generates the route for the exact routing algorithm from solution variables."""

        self.pick_list = pick_list

        start_time = time.time()
        self._set_routing_parameters()
        self._set_decision_variables()
        self._set_objective()
        self._set_constraints()
        self.mdl.optimize()

        execution_time = time.time() - start_time

        if self.mdl.status == GRB.OPTIMAL:
            self.solution = {
                'picker': {}
            }
            for p in self.picker_id:
                self.solution['picker'][p] = {'batches': {}}
                for b in self.set_batch_numbers:
                    if self.w[b, p].x > 0.5:  # aktivierte Zuordnung

                        self.construct_route_and_sequence(b)
                        self.distance_per_batch[b] = self.C_bp[b, p].x
                        # Speichern
                        self.solution['picker'][p]['batches'][b] = {
                            'order_in_batch': [o for o in self.set_order_numbers if self.y[o, b].x > 0.5],
                            'item_sequence': self.item_sequence,
                            'route': self.route,
                            'distance': self.C_bp[b, p].x,  # optional: du kannst hier auch die echte Distanz berechnen
                            'execution_time': execution_time
                        }
            print(self.solution)
        else:
            self.solution = {'status': 'infeasible', 'message': 'No optimal solution found.'}
            print("No optimal solution found. Status:", self.mdl.status)


    def _set_routing_parameters(self):
        """Sets the parameters for the exact routing algorithm."""

        self.list_item_pick_locations = [item.pick_node for item in self.pick_list]
        self.list_item_numbers = [item.position.article_id for item in self.pick_list]
        self.list_item_amounts = [item.position.amount for item in self.pick_list]
        self.list_order_numbers = [item.position.order_number for item in self.pick_list]

        self.set_order_numbers = set(self.list_order_numbers)
        self.set_batch_numbers = range(len(self.set_order_numbers))

        self.len_item_numbers = len(self.list_item_numbers)

        self.range_item_number = range(self.len_item_numbers)


        self.picker_id =  [p.id for p in self.picker]
        self.picker_capa = [p.capacity for p in self.picker]
        self.picker_speed = [p.speed for p in self.picker]

        # Amount items in each order
        df = pd.DataFrame([{
            'order_number': item.position.order_number,
            'amount': item.position.amount
        } for item in self.pick_list])

        self.s_i = df.groupby('order_number')['amount'].sum().to_dict()

        # Travel time cost
        self.c = {(i, j, p.id): self.distance_matrix[self.list_item_pick_locations[i]][self.list_item_pick_locations[j]] / p.speed
                    for i in self.range_item_number for j in self.range_item_number for p in self.picker}
        self.c_start = { (j, p.id): self.distance_matrix[self.start_node][self.list_item_pick_locations[j]] / p.speed
                        for j in self.range_item_number for p in self.picker}
        self.c_end = { (j, p.id): self.distance_matrix[self.list_item_pick_locations[j]][self.end_node] / p.speed
                        for j in self.range_item_number for p in self.picker}

        self.u_i_j = {i: {j: self.list_item_amounts[j]
            if self.list_item_numbers[j] in self.list_item_numbers and self.list_order_numbers[j] == i
            else 0 for j in self.range_item_number} for i in self.set_order_numbers}

        print()



    def _set_decision_variables(self):
        """Set the decision variables for the exact routing model."""
        self.x = self.mdl.addVars(self.range_item_number, self.range_item_number, self.set_batch_numbers, vtype=GRB.BINARY, name="x")
        self.x_start = self.mdl.addVars(self.range_item_number, self.set_batch_numbers,vtype=GRB.BINARY, name="x0j")
        self.x_end = self.mdl.addVars(self.range_item_number, self.set_batch_numbers,vtype=GRB.BINARY, name="xj0")
        self.w = self.mdl.addVars(self.set_batch_numbers, self.picker_id, vtype=GRB.BINARY, name="w")
        self.T = self.mdl.addVars(self.range_item_number, self.set_batch_numbers, self.picker_id, vtype=GRB.CONTINUOUS, name="T")
        self.y = self.mdl.addVars(self.set_order_numbers, self.range_item_number, vtype=GRB.BINARY, name="y")
        self.z = self.mdl.addVars(self.range_item_number, self.set_batch_numbers,vtype=GRB.BINARY, name="z")
        self.C_bp = self.mdl.addVars(self.set_batch_numbers, self.picker_id, vtype=GRB.CONTINUOUS, lb=0, name="C_bp")
        self.C_max = self.mdl.addVar(lb=0, vtype=GRB.CONTINUOUS, name="C_max")
        self.C_max_p = self.mdl.addVars(self.picker_id, vtype=GRB.CONTINUOUS, lb=0, name="C_max_p")

        #self.help = self.mdl.addVar(vtype=GRB.BINARY, name="help")

    def _set_objective(self):
        self.mdl.setObjective(self.C_max + gp.quicksum(self.C_max_p[p] * 0.001 for p in self.picker_id), GRB.MINIMIZE)

    def _set_constraints(self):
        """Set the constraints for the exact routing model."""
        # Each item is picked exactly once
        for j in self.range_item_number:
            self.mdl.addConstr(gp.quicksum(gp.quicksum(
                self.x[i, j, b]
                for i in self.range_item_number if i != j) + self.x_start[j, b]  for b in self.set_batch_numbers)
                               == 1)
            self.mdl.addConstr(gp.quicksum(gp.quicksum(
                self.x[j, i, b]
                 for i in self.range_item_number if i != j) + self.x_end[j, b] for b in self.set_batch_numbers)
                               == 1)

        # # Each item is connected to the next item in the batch
        for j in self.range_item_number:
            for b in self.set_batch_numbers:
                self.mdl.addConstr(
                    gp.quicksum(self.x[j, i, b] for i in self.range_item_number if i != j) + self.x_end[j, b]
                    == gp.quicksum(self.x[i, j, b] for i in self.range_item_number if i != j) + self.x_start[j, b])

        for b in self.set_batch_numbers:
            self.mdl.addConstr(gp.quicksum(self.w[b, p] for p in self.picker_id) <= 1)

            for i in self.set_order_numbers:
                self.mdl.addConstr(gp.quicksum(self.y[i, b] for i in self.set_order_numbers) <= self.big_m * gp.quicksum(self.w[b, p] for p in self.picker_id))
        # Only one starting and one ending point per batch
        for b in self.set_batch_numbers:
            self.mdl.addConstr(gp.quicksum(self.x_start[j, b] for j in self.range_item_number) == gp.quicksum(self.w[b, p] for p in self.picker_id))
            self.mdl.addConstr(gp.quicksum(self.x_end[j, b] for j in self.range_item_number) == gp.quicksum(self.w[b, p] for p in self.picker_id))
            # self.mdl.addConstr(gp.quicksum(gp.quicksum(self.x_end[j, b, p] for j in self.range_item_number) for p in self.picker_id) == 1)

        # Each order is assigned to one batch
        for o in self.set_order_numbers:
            self.mdl.addConstr(gp.quicksum(self.y[o, b] for b in self.set_batch_numbers) == 1)

        # Start/Ende nur, wenn Auftrag im Batch ist
        for b in self.set_batch_numbers:
            for o in self.set_order_numbers:
                self.mdl.addConstr(gp.quicksum(self.x_start[j, b] for j in self.range_item_number if self.u_i_j[o][j] > 0) <= self.y[o, b])
                self.mdl.addConstr(gp.quicksum(self.x_end[j, b] for j in self.range_item_number if self.u_i_j[o][j] > 0) <= self.y[o, b])

        # Artikelzuordnung auf Batches konsistent mit Aufträgen
        for b in self.set_batch_numbers:
            for o in self.set_order_numbers:
                self.mdl.addConstr(gp.quicksum(self.u_i_j[o][j] * self.z[j, b] for j in self.range_item_number) == self.s_i[o] * self.y[o, b])

        # Routingkanten dürfen nur existieren, wenn Artikel im Batch
        for b in self.set_batch_numbers:
            for j1 in self.range_item_number:
                self.mdl.addConstr(gp.quicksum(self.x[j1, j2, b] for j2 in self.range_item_number if j1 != j2) <= self.z[j1, b])

            for j2 in self.range_item_number:
                self.mdl.addConstr(gp.quicksum(self.x[j1, j2, b] for j1 in self.range_item_number if j1 != j2) <= self.z[j2, b])

        # Kapazitätsbeschränkung pro Picker
        for index_p, p in enumerate(self.picker_id):
            for b in self.set_batch_numbers:
                self.mdl.addConstr(gp.quicksum(self.s_i[i] * self.y[i, b] * self.w[b, p] for i in self.set_order_numbers) <= self.picker_capa[index_p])

        # Zeitplanung (Reisedauer)
        for j1 in self.range_item_number:
            for j2 in self.range_item_number:
                for b in self.set_batch_numbers:
                    for p in self.picker_id:
                        self.mdl.addConstr(self.T[j1, b, p] + self.c[(j1,j2,p)] <= self.T[j2, b, p] + self.big_m * (1 - self.x[j1, j2, b] * self.w[b, p]))

        # Start- und Endzeit pro Picker und Batch
        for j in self.range_item_number:
            for b in self.set_batch_numbers:
                for p in self.picker_id:
                    # Start time
                    self.mdl.addConstr(self.c_start[(j,p)] * self.x_start[j, b] * self.w[b, p] <= self.T[j, b, p])
                    # End time
                    self.mdl.addConstr(self.T[j, b, p] + self.c_end[(j,p)] * self.x_end[j, b] * self.w[b, p] <= self.C_bp[b, p])


        # Abschlusszeiten aggregieren
        for p in self.picker_id:
            self.mdl.addConstr(self.C_max_p[p] >= gp.quicksum(self.C_bp[b, p] for b in self.set_batch_numbers))
            self.mdl.addConstr(self.C_max >= self.C_max_p[p])

