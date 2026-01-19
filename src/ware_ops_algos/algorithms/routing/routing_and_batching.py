import time

import gurobipy as gp
from gurobipy import GRB

import numpy as np


class ExactBatchingAndRoutingBase:
    pass

class ExactBatchingAndRouting(ExactBatchingAndRoutingBase):
    def __init__(self,
                 network_graph,
                 orders,
                 distance_matrix,
                 tour_matrix,
                 picker_list,
                 big_m,
                 objective: str,
                 # 'minimize_distance' or 'minimize_maximum_completion_time' or 'minimize_delay' or 'minimize_tardiness'
                 **kwargs):
        # super().__init__(network_graph,
        #                  orders,
        #                  distance_matrix,
        #                  tour_matrix,
        #                  **kwargs)
        self.network_graph = network_graph
        self.orders = orders
        self.distance_matrix = distance_matrix
        self.tour_matrix = tour_matrix

        self.picker_list = picker_list
        self.M = big_m
        self.objective = objective

        self.set_of_pickers = None  # P
        self.set_of_orders = None
        self.set_of_items_with_location = None

        self.number_of_orders = None  # i
        self.number_of_pickers = None  # p
        self.number_of_batches = None  # b
        self.number_of_pick_locations = None  # j

        self.number_of_items_in_order = {}  # s_i
        self.capa_of_picker_p = None  # capa_p
        self.speed_of_picker_p = None  # v_p
        # self.distance_matrix = None  # d_{j1}_{j2}
        self.travel_time_matrix_of_picker_p = {}  # c_{p}_{j1}_{j2}

        self.item_is_in_order_i = None  # u_{i}_{j}

    def _set_exact_parameters(self):
        self.set_of_pickers = self.picker_list['picker_id'].tolist()  # P
        self.set_of_orders = sorted(self.orders['order_number'].unique())  # O
        self.set_of_items = self.orders['article_id']  # V
        self.set_of_batches = list(range(len(self.set_of_orders)))  # B

        self.number_of_orders = len(self.set_of_orders)  # i
        self.number_of_pickers = len(self.set_of_pickers)  # p
        self.number_of_batches = len(self.set_of_orders)  # b
        self.number_of_pick_locations = len(self.set_of_items_with_location)  # j

        for order_k in self.set_of_orders:
            self.number_of_items_in_order[order_k] = (
                self.orders[self.orders['order_number'] == order_k]['amount'].sum())  # s_i

        self.capa_of_picker_p = self.picker_list['capacity'].tolist()  # capa_p

        self.speed_of_picker_p = self.picker_list['speed'].tolist()  # v_p

        for p in range(self.number_of_pickers):
            self.travel_time_matrix_of_picker_p[p] = np.array(self.distance_matrix) / self.speed_of_picker_p[
                p]  # c_{p}_{j1}_{j2}

        self.item_is_in_order_i = self.orders.pivot_table(index='order_number',
                                                                columns='article_id',
                                                                aggfunc='size',
                                                                fill_value=0)
        self.item_is_in_order_i_dict = self.item_is_in_order_i.to_dict()  # u_{i}_{j}


    def routing_algorithm(self):
        # --- Model ---
        model = gp.Model("Batching_and_Routing")

        # --- Variables ---
        xw = model.addVars(self.set_of_items, self.set_of_items, B, self.set_of_pickers, vtype=GRB.BINARY, name="xw")  # xw_j1j2bp
        xw_start = model.addVars(self.set_of_items, B, self.set_of_pickers, vtype=GRB.BINARY, name="xw_start")  # xw_0jbp
        xw_end = model.addVars(self.set_of_items, B, self.set_of_pickers, vtype=GRB.BINARY, name="xw_end")  # xw_j0bp
        y = model.addVars(self.set_of_orders, B, self.set_of_pickers, vtype=GRB.BINARY, name="y")  # y_ib
        z = model.addVars(self.set_of_items, B, self.set_of_pickers, vtype=GRB.BINARY, name="z")  # z_jb
        T_jbp = model.addVars(self.set_of_items, B, self.set_of_pickers, lb=0, vtype=GRB.CONTINUOUS, name="T_jbp")  # T_jbp
        C_bp = model.addVars(B, self.set_of_pickers, lb=0, vtype=GRB.CONTINUOUS, name="C_bp")  # C_bp

        # max time in general for all pickers
        C_max = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="C_max")

        # Cumulative time for each picker
        C_max_p = model.addVars(self.set_of_pickers, vtype=GRB.CONTINUOUS, lb=0, name="C_max_p")

        # --- Objective ---
        # Minimize the maximum completion time and the sum of the maximum completion times of all pickers
        model.setObjective(C_max + gp.quicksum(C_max_p[p] * 0.001 for p in self.set_of_pickers), GRB.MINIMIZE)

        # --- Constraints ---
        # Each item is picked exactly once
        for j in self.set_of_items:
            model.addConstr(1 == gp.quicksum(
                gp.quicksum(gp.quicksum(xw[j, j2, b, p] for j2 in self.set_of_items if j2 != j) + xw_end[j, b, p] for b in B) for p in self.set_of_pickers))
            model.addConstr(1 == gp.quicksum(
                gp.quicksum(gp.quicksum(xw[j1, j, b, p] for j1 in self.set_of_items if j1 != j) + xw_start[j, b, p] for b in B) for p in self.set_of_pickers))

        # Each item is connected to the next item in the batch
        for j in self.set_of_items:
            for b in self.set_of_batches:
                model.addConstr(
                    gp.quicksum(gp.quicksum(xw[j, j2, b, p] for j2 in self.set_of_items if j != j2) + xw_end[j, b, p] for p in self.set_of_pickers)
                    == gp.quicksum(gp.quicksum(xw[j1, j, b, p] for j1 in self.set_of_items if j != j1) + xw_start[j, b, p] for p in self.set_of_pickers))

        # Only one starting and one ending point per batch
        for b in self.set_of_batches:
            model.addConstr(gp.quicksum(gp.quicksum(xw_start[j, b, p] for j in self.set_of_items) for p in self.set_of_pickers) == 1)
            model.addConstr(gp.quicksum(gp.quicksum(xw_end[j, b, p] for j in self.set_of_items) for p in self.set_of_pickers) == 1)

        # Each order is assigned to one batch
        for i in self.set_of_orders:
            model.addConstr(gp.quicksum(gp.quicksum(y[i, b, p] for p in P) for b in B) == 1)

        # Ensure that the start and end points are only used if the order is assigned to a batch
        for b in self.set_of_batches:
            for i in self.set_of_orders:
                for p in self.set_of_pickers:
                    model.addConstr(gp.quicksum(xw_start[j, b, p] for j in self.set_of_items if u_ij[i, self.set_of_items[j]] > 0) <= y[i, b, p])
                    model.addConstr(gp.quicksum(xw_end[j, b, p] for j in self.set_of_items if u_ij[i, self.set_of_items[j]] > 0) <= y[i, b, p])

        # # Item assignments based on orders
        for b in self.set_of_batches:
            for i in self.set_of_orders:
                for p in self.set_of_pickers:
                    model.addConstr(gp.quicksum(u_ij[i, j] * z[j, b, p] for j in V) == s_i[i] * y[i, b, p])

        # Batching constraints
        for j1 in self.set_of_items:
            for b in self.set_of_batches:
                for p in self.set_of_pickers:
                    model.addConstr(gp.quicksum(xw[j1, j2, b, p] for j2 in self.set_of_items if j1 != j2) <= z[j1, b, p])

        for j2 in self.set_of_items:
            for b in self.set_of_batches:
                for p in self.set_of_pickers:
                    model.addConstr(gp.quicksum(xw[j1, j2, b, p] for j1 in self.set_of_items if j1 != j2) <= z[j2, b, p])

        # Capacity constraint has to be fulfilled only if the batch is assigned to a picker
        for b in self.set_of_batches:
            for p in self.set_of_pickers:
                model.addConstr(gp.quicksum(s_i[i] * y[i, b, p] for i in self.set_of_orders) <= capa_p[p])

        # Travel time and process time constraints
        for j1 in self.set_of_items:
            for j2 in self.set_of_items:
                for b in self.set_of_batches:
                    for p in self.set_of_pickers:
                        model.addConstr(T_jbp[j1, b, p] + c[j1, j2, p] <= T_jbp[j2, b, p] + M * (1 - xw[j1, j2, b, p]))

        # if self.objective == "minimize_maximum_completion_time":
        # Start and end time constraints
        for j in self.set_of_items:
            for b in self.set_of_batches:
                for p in self.set_of_pickers:
                    # Start time
                    model.addConstr(c_start[j, p] * xw_start[j, b, p] <= T_jbp[j, b, p])
                    # End time
                    model.addConstr(T_jbp[j, b, p] + c_end[j, p] * xw_end[j, b, p] <= C_bp[b, p])

        # # Ensure that C_max is at least as large as any picker's cumulative completion time
        for p in self.set_of_pickers:
            model.addConstr(C_max_p[p] >= gp.quicksum(C_bp[b, p] for b in self.set_of_batches))
            model.addConstr(C_max >= C_max_p[p])

        # --- Optimize ---
        model.optimize()

    @staticmethod
    def print_solution(model_):
        if model_.status != GRB.OPTIMAL:
            print("No optimal output found.")
        # Ergebnisse ausgeben
        else:
            print("Optimal output found:")
            print("C_max:", C_max.X)
            # print output
            for b in B:
                for p in P:
                    if w[b, p].X > 0.5:
                        print(f"Batch {b}, Picker {p}:")
                        for j in V:
                            if x_start[j, b].X > 0.5:
                                print(f"Start at {j}")
                        for j in V:
                            if x_end[j, b].X > 0.5:
                                print(f"End at {j}")
                        for j in V:
                            for j2 in V:
                                if x[j, j2, b].X > 0.5:
                                    print(f"Item {j} to {j2}")

        if model_.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model_.computeIIS()
            model_.write("infeasible_model.ilp")  # Speichere die IIS in einer Datei

    def generate_routing(self):
        self._routing_algorithm = 'ExactRouting'
        self._initialize_parameters()
        self._set_exact_parameters()
        self.routing_algorithm()
        self.complete_solutions[self.batch_number] = self._compile_solution()


if __name__ == "__main__":

    # batch_list = pd.read_csv("../../algorithms/batching/test.csv")
    # batch_list = batch_list.head(5)
    # # G = load_pickle("../../data/use_cases/shelf_storage/input/graph.pkl")
    # graph_generator = ShelfStorageGraphGenerator(
    #     10, 20,
    #     4, 1,
    #     1.5, 1.5, (0, 0),
    #     (3, 0))
    # print('....')
    # graph_generator.populate_graph()
    # print('.....')
    # # start_time = time.time()
    # # _distance_matrix = distance_matrix_generator(graph_generator.G)
    # # end_time = time.time()
    # # print(f"Die Ausführungszeit 'Distance Matrix' beträgt {end_time - start_time} Sekunden.")
    #
    # start_time = time.time()
    # _tour_matrix = tour_matrix_generator(graph_generator.G)
    # end_time = time.time()
    # print(f"Die Ausführungszeit 'Tour Matrix' beträgt {end_time - start_time} Sekunden.")
    # start_time = time.time()
    # _distance_matrix = distance_matrix_generator_old(graph_generator.G)
    # end_time = time.time()
    # print(f"Die Ausführungszeit 'Distance Matrix Old' beträgt {end_time - start_time} Sekunden.")
    #
    # # start_time = time.time()
    # # _tour_matrix_old = tour_matrix_generator_old(graph_generator.G)
    # # end_time = time.time()
    # # print(f"Die Ausführungszeit 'Tour Matrix Old' beträgt {end_time - start_time} Sekunden.")
    #
    # #
    # # print("************************************\n"
    # #       "S-Shape Routing\n"
    # #       "************************************")
    # #
    # # SShapeRouting = SShapeRouting(graph_generator.G, batch_list, _distance_matrix, _tour_matrix)
    # #
    # # start_time = time.time()
    # # SShapeRouting.generate_routing()
    # # end_time = time.time()
    # #
    # # # SShapeRouting.draw_solution()
    # # print(pd.DataFrame(SShapeRouting.complete_solutions))
    # #
    # # elapsed_time = end_time - start_time
    # # print(f"Die Ausführungszeit 'S-Shape' beträgt {elapsed_time} Sekunden.")  #
    # #
    # # print("\n************************************\n"
    # #       "Return Routing\n"
    #       "************************************")
    #
    # ReturnRouting = ReturnRouting(graph_generator.G, batch_list, _distance_matrix, _tour_matrix)
    #
    # start_time = time.time()
    # ReturnRouting.generate_routing()
    # end_time = time.time()
    #
    # # ReturnRouting.draw_solution()
    # print(pd.DataFrame(ReturnRouting.complete_solutions))
    #
    # elapsed_time = end_time - start_time
    #
    # print(f"Die Ausführungszeit 'Return' beträgt {elapsed_time} Sekunden.")
    #
    # print("\n************************************\n"
    #       "Nearest Neighbourhood Routing\n"
    #       "************************************")
    #
    # NearestNeighbourhood = NearestNeighbourhood(graph_generator.G, batch_list, _distance_matrix, _tour_matrix)
    #
    # start_time = time.time()
    # NearestNeighbourhood.generate_routing()
    # end_time = time.time()
    #
    # print(pd.DataFrame(NearestNeighbourhood.complete_solutions))
    # # NearestNeighbourhood.draw_solution()
    #
    # elapsed_time = end_time - start_time
    # print(f"Die Ausführungszeit 'Return' beträgt {elapsed_time} Sekunden.")
    #
    # print("\n************************************\n"
    #       "Midpoint Routing\n"
    #       "************************************")
    #
    # MidpointRouting = MidpointRouting(graph_generator.G, batch_list, _distance_matrix, _tour_matrix)
    #
    # start_time = time.time()
    # MidpointRouting.generate_routing()
    # end_time = time.time()
    #
    # print(pd.DataFrame(MidpointRouting.complete_solutions))
    # # MidpointRouting.draw_solution()
    #
    # elapsed_time = end_time - start_time
    # print(f"Die Ausführungszeit 'Midpoint' beträgt {elapsed_time} Sekunden.")
    #
    # print("\n************************************\n"
    #       "LargestGap Routing\n"
    #       "************************************")
    #
    # LargestGapRouting = LargestGapRouting(graph_generator.G, batch_list, _distance_matrix, _tour_matrix)
    #
    # start_time = time.time()
    # LargestGapRouting.generate_routing()
    # end_time = time.time()
    #
    # print(pd.DataFrame(LargestGapRouting.complete_solutions))
    # # LargestGapRouting.draw_solution()
    #
    # elapsed_time = end_time - start_time
    # print(f"Die Ausführungszeit 'LargestGap' beträgt {elapsed_time} Sekunden.")
    #
    # print("\n************************************\n"
    #         "Exact Routing\n"
    #         "************************************")
    #
    # ExactRouting = ExactRouting(graph_generator.G, batch_list, _distance_matrix, _tour_matrix)
    #
    # start_time = time.time()
    # ExactRouting.generate_routing()
    # end_time = time.time()
    #
    # print(pd.DataFrame(ExactRouting.complete_solutions))
    # # ExactRouting.draw_solution()
    #
    # elapsed_time = end_time - start_time
    # print(f"Die Ausführungszeit 'Exact' beträgt {elapsed_time} Sekunden.")

    # print("\n************************************\n"
    #       "Exact Batching And Routing\n"
    #       "************************************")
    # Dummy Picker List
    # _picker_list = pd.DataFrame({
    #     'picker_id': [1, 2],
    #     'capacity': [40, 40],
    #     'speed': [2, 6]
    # })
    #
    # ExactBatchingAndRouting = ExactBatchingAndRouting(graph_generator.G,
    #                                                   batch_list,
    #                                                   _distance_matrix,
    #                                                   _tour_matrix,
    #                                                   _picker_list,
    #                                                   1000)
    #
    # start_time = time.time()
    # ExactBatchingAndRouting.generate_routing()
    # end_time = time.time()
    #
    # print(pd.DataFrame(ExactBatchingAndRouting.complete_solutions))
    # # ExactRouting.draw_solution()
    #
    # elapsed_time = end_time - start_time
    # print(f"Die Ausführungszeit 'Exact' beträgt {elapsed_time} Sekunden.")

    ########## TEST ###########

    # --- Sets ---
    P = [1, 2, 3, 4]  # Set of pickers (4 pickers)
    V = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Set of items and pick locations (10 items)
    O = [1, 2, 3, 4, 5]  # Set of orders (5 orders)
    B = [1, 2, 3, 4, 5]  # Set of batches (3 batches)

    # --- Parameters ---
    K = len(P)  # Number of pickers
    N = len(V)  # Number of items and pick locations
    M = len(O)  # Number of orders
    D = len(B)  # Number of batches

    # Anzahl der Artikel in den Bestellungen (s_i)
    s_i = {
        1: 1,  # Bestellung 1 hat 6 Artikel
        2: 2,  # Bestellung 2 hat 3 Artikel
        3: 3,  # Bestellung 3 hat 5 Artikel
        4: 3,  # Bestellung 4 hat 2 Artikel
        5: 1  # Bestellung 5 hat 4 Artikel
    }

    # Kapazität der Picker (capa_p)
    capa_p = {
        1: 1,  # Picker 1 kann 5 Artikel tragen
        2: 6,  # Picker 2 kann 10 Artikel tragen
        3: 6,  # Picker 3 kann 7 Artikel tragen
        4: 1  # Picker 4 kann 5 Artikel tragen
    }

    # Geschwindigkeit der Picker (v_p)
    v_p = {
        1: 1.0,  # Geschwindigkeit von Picker 1
        2: 1.0,  # Geschwindigkeit von Picker 2
        3: 1.0,  # Geschwindigkeit von Picker 3
        4: 1.0  # Geschwindigkeit von Picker 4
    }

    # Distanzmatrix (distance_matrix)
    distance_matrix_ = {
        (1, 1): 0, (1, 2): 5, (1, 3): 8, (1, 4): 7, (1, 5): 10, (1, 6): 6, (1, 7): 4, (1, 8): 8, (1, 9): 9, (1, 10): 11,
        (2, 1): 5, (2, 2): 0, (2, 3): 3, (2, 4): 9, (2, 5): 6, (2, 6): 5, (2, 7): 8, (2, 8): 3, (2, 9): 7, (2, 10): 2,
        (3, 1): 8, (3, 2): 3, (3, 3): 0, (3, 4): 4, (3, 5): 6, (3, 6): 2, (3, 7): 5, (3, 8): 4, (3, 9): 8, (3, 10): 5,
        (4, 1): 7, (4, 2): 9, (4, 3): 4, (4, 4): 0, (4, 5): 3, (4, 6): 7, (4, 7): 6, (4, 8): 8, (4, 9): 5, (4, 10): 2,
        (5, 1): 10, (5, 2): 6, (5, 3): 6, (5, 4): 3, (5, 5): 0, (5, 6): 5, (5, 7): 4, (5, 8): 7, (5, 9): 8, (5, 10): 6,
        (6, 1): 6, (6, 2): 5, (6, 3): 2, (6, 4): 7, (6, 5): 5, (6, 6): 0, (6, 7): 3, (6, 8): 4, (6, 9): 5, (6, 10): 2,
        (7, 1): 4, (7, 2): 8, (7, 3): 5, (7, 4): 6, (7, 5): 4, (7, 6): 3, (7, 7): 0, (7, 8): 5, (7, 9): 4, (7, 10): 6,
        (8, 1): 8, (8, 2): 3, (8, 3): 4, (8, 4): 8, (8, 5): 7, (8, 6): 4, (8, 7): 5, (8, 8): 0, (8, 9): 4, (8, 10): 5,
        (9, 1): 9, (9, 2): 7, (9, 3): 8, (9, 4): 5, (9, 5): 8, (9, 6): 5, (9, 7): 4, (9, 8): 4, (9, 9): 0, (9, 10): 2,
        (10, 1): 11, (10, 2): 2, (10, 3): 5, (10, 4): 2, (10, 5): 6, (10, 6): 2, (10, 7): 6, (10, 8): 5, (10, 9): 2,
        (10, 10): 0,
        (0, 1): 1, (0, 2): 2, (0, 3): 4, (0, 4): 3, (0, 5): 5, (0, 6): 6, (0, 7): 7, (0, 8): 8, (0, 9): 9, (0, 10): 10,
        (1, 0): 1, (2, 0): 2, (3, 0): 4, (4, 0): 3, (5, 0): 5, (6, 0): 6, (7, 0): 7, (8, 0): 8, (9, 0): 9, (10, 0): 10
    }

    # Travel time cost c_j1j2p
    c = {(j1, j2, p): distance_matrix_.get((j1, j2), 0) / v_p[p] for j1 in V for j2 in V for p in P}
    c_start = {(j, p): distance_matrix_.get((0, j), 0) / v_p[p] for j in V for p in P}
    c_end = {(j, p): distance_matrix_.get((j, 0), 0) / v_p[p] for j in V for p in P}

    # Big M constant
    M = 1000

    # u_ij: Order-item assignment
    u_ij = {
        (1, 1): 1, (1, 2): 0, (1, 3): 0, (1, 4): 0, (1, 5): 0, (1, 6): 0, (1, 7): 0, (1, 8): 0, (1, 9): 0, (1, 10): 0,
        # Bestellung 1
        (2, 1): 0, (2, 2): 1, (2, 3): 1, (2, 4): 0, (2, 5): 0, (2, 6): 0, (2, 7): 0, (2, 8): 0, (2, 9): 0, (2, 10): 0,
        # Bestellung 2
        (3, 1): 0, (3, 2): 0, (3, 3): 0, (3, 4): 1, (3, 5): 1, (3, 6): 1, (3, 7): 0, (3, 8): 0, (3, 9): 0, (3, 10): 0,
        # Bestellung 3
        (4, 1): 0, (4, 2): 0, (4, 3): 0, (4, 4): 0, (4, 5): 0, (4, 6): 0, (4, 7): 1, (4, 8): 1, (4, 9): 1, (4, 10): 0,
        # Bestellung 4
        (5, 1): 0, (5, 2): 0, (5, 3): 0, (5, 4): 0, (5, 5): 0, (5, 6): 0, (5, 7): 0, (5, 8): 0, (5, 9): 0, (5, 10): 1
        # Bestellung 5
    }

    model_variante = True

    if model_variante:
        # --- Model ---
        model = gp.Model("Batching_and_Routing")

        # --- Variables ---
        # x = model.addVars(V, V, B, vtype=GRB.BINARY, name="x")  # x_j1j2b
        # x_start = model.addVars(V, B, vtype=GRB.BINARY, name="x_start")  # x_0jb
        # x_end = model.addVars(V, B, vtype=GRB.BINARY, name="x_end")  # x_j0b
        y = model.addVars(O, B, P, vtype=GRB.BINARY, name="y")  # y_ib
        z = model.addVars(V, B, P, vtype=GRB.BINARY, name="z")  # z_jb
        # w = model.addVars(B, P, vtype=GRB.BINARY, name="w")  # w_bp
        C_bp = model.addVars(B, P, lb=0, vtype=GRB.CONTINUOUS, name="C_bp")  # C_bp
        T_jbp = model.addVars(V, B, P, lb=0, vtype=GRB.CONTINUOUS, name="T_jbp")  # T_jbp
        xw = model.addVars(V, V, B, P, vtype=GRB.BINARY, name="xw")  # xw_j1j2bp
        xw_start = model.addVars(V, B, P, vtype=GRB.BINARY, name="xw_start")  # xw_0jbp
        xw_end = model.addVars(V, B, P, vtype=GRB.BINARY, name="xw_end")  # xw_j0bp

        # C_max
        C_max = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="C_max")

        # Cumulative time for each picker
        C_max_p = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="C_max_p")

        # --- Objective ---
        model.setObjective(C_max + gp.quicksum(C_max_p[p] * 0.001 for p in P), GRB.MINIMIZE)

        # --- Constraints ---
        # Each item is picked exactly once
        for j in V:
            model.addConstr(1 == gp.quicksum(
                gp.quicksum(gp.quicksum(xw[j, j2, b, p] for j2 in V if j2 != j) + xw_end[j, b, p] for b in B) for p in P))
            model.addConstr(1 == gp.quicksum(
                gp.quicksum(gp.quicksum(xw[j1, j, b, p] for j1 in V if j1 != j) + xw_start[j, b, p] for b in B) for p in P))

        # Each item is connected to the next item in the batch
        for j in V:
            for b in B:
                model.addConstr(
                    gp.quicksum(gp.quicksum(xw[j, j2, b, p] for j2 in V if j != j2) + xw_end[j, b, p] for p in P)
                    == gp.quicksum(gp.quicksum(xw[j1, j, b, p] for j1 in V if j != j1) + xw_start[j, b, p] for p in P))

        # Only one starting and one ending point per batch
        for b in B:
            model.addConstr(gp.quicksum(gp.quicksum(xw_start[j, b, p] for j in V) for p in P) == 1)
            model.addConstr(gp.quicksum(gp.quicksum(xw_end[j, b, p] for j in V) for p in P) == 1)

        # Each order is assigned to one batch
        for i in O:
            model.addConstr(gp.quicksum(gp.quicksum(y[i, b, p] for p in P) for b in B) == 1)

        # Ensure that the start and end points are only used if the order is assigned to a batch
        for b in B:
            for i in O:
                for p in P:
                    model.addConstr(gp.quicksum(xw_start[j, b, p] for j in V if u_ij[i, j] > 0) <= y[i, b, p])
                    model.addConstr(gp.quicksum(xw_end[j, b, p] for j in V if u_ij[i, j] > 0) <= y[i, b, p])

        # # Item assignments based on orders
        for b in B:
            for i in O:
                for p in P:
                    model.addConstr(gp.quicksum(u_ij[i, j] * z[j, b, p] for j in V) == s_i[i] * y[i, b, p])

        # Batching constraints
        for j1 in V:
            for b in B:
                for p in P:
                    model.addConstr(gp.quicksum(xw[j1, j2, b, p] for j2 in V if j1 != j2) <= z[j1, b, p])

        for j2 in V:
            for b in B:
                for p in P:
                    model.addConstr(gp.quicksum(xw[j1, j2, b, p] for j1 in V if j1 != j2) <= z[j2, b, p])

        # Capacity constraint has to be fulfilled only if the batch is assigned to a picker
        for b in B:
            for p in P:
                model.addConstr(gp.quicksum(s_i[i] * y[i, b, p] for i in O) <= capa_p[p])

        # Travel time and process time constraints
        for j1 in V:
            for j2 in V:
                for b in B:
                    for p in P:
                        model.addConstr(T_jbp[j1, b, p] + c[j1, j2, p] <= T_jbp[j2, b, p] + M * (1 - xw[j1, j2, b, p]))

        # if self.objective == "minimize_maximum_completion_time":
        # Start and end time constraints
        for j in V:
            for b in B:
                for p in P:
                    # Start time
                    model.addConstr(c_start[j, p] * xw_start[j, b, p] <= T_jbp[j, b, p])
                    # End time
                    model.addConstr(T_jbp[j, b, p] + c_end[j, p] * xw_end[j, b, p] <= C_bp[b, p])

        # # Ensure that C_max is at least as large as any picker's cumulative completion time
        for p in P:
            model.addConstr(C_max_p[p] >= gp.quicksum(C_bp[b, p] for b in B))
            model.addConstr(C_max >= C_max_p[p])


    else:
        # --- Model ---
        model = gp.Model("Batching_and_Routing")

        # --- Variables ---
        x = model.addVars(V, V, B, vtype=GRB.BINARY, name="x")  # x_j1j2b
        x_start = model.addVars(V, B, vtype=GRB.BINARY, name="x_start")  # x_0jb
        x_end = model.addVars(V, B, vtype=GRB.BINARY, name="x_end")  # x_j0b
        y = model.addVars(O, B, vtype=GRB.BINARY, name="y")  # y_ib
        z = model.addVars(V, B, vtype=GRB.BINARY, name="z")  # z_jb
        w = model.addVars(B, P, vtype=GRB.BINARY, name="w")  # w_bp
        C_bp = model.addVars(B, P, lb=0, vtype=GRB.CONTINUOUS, name="C_bp")  # C_bp
        T_jbp = model.addVars(V, B, P, lb=0, vtype=GRB.CONTINUOUS, name="T_jbp")  # T_jbp
        xw = model.addVars(V, V, B, P, vtype=GRB.BINARY, name="xw")  # xw_j1j2bp
        xw_start = model.addVars(V, B, P, vtype=GRB.BINARY, name="xw_start")  # xw_0jbp
        xw_end = model.addVars(V, B, P, vtype=GRB.BINARY, name="xw_end")  # xw_j0bp

        # C_max
        C_max = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="C_max")

        # Cumulative time for each picker
        C_max_p = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0, name="C_max_p")

        # --- Objective ---
        model.setObjective(C_max + gp.quicksum(C_max_p[p] * 0.001 for p in P), GRB.MINIMIZE)

        # --- Constraints ---
        # Each item is picked exactly once
        for j in V:
            model.addConstr(1 == gp.quicksum(gp.quicksum(x[j, j2, b] for j2 in V if j2 != j) + x_end[j, b] for b in B))
            model.addConstr(
                1 == gp.quicksum(gp.quicksum(x[j1, j, b] for j1 in V if j1 != j) + x_start[j, b] for b in B))

        # Each item is connected to the next item in the batch
        for j in V:
            for b in B:
                model.addConstr(
                    gp.quicksum(x[j, j2, b] for j2 in V if j != j2) + x_end[j, b]
                    == gp.quicksum(x[j1, j, b] for j1 in V if j != j1) + x_start[j, b])
        # Each batch is assigned to one picker at most
        for b in B:
            model.addConstr(gp.quicksum(w[b, p] for p in P) <= 1)

        # Only one starting and one ending point per batch
        for b in B:
            model.addConstr(gp.quicksum(x_start[j, b] for j in V) == gp.quicksum(w[b, p] for p in P))
            model.addConstr(gp.quicksum(x_end[j, b] for j in V) == gp.quicksum(w[b, p] for p in P))

        # Each order is assigned to one batch
        for i in O:
            model.addConstr(gp.quicksum(y[i, b] for b in B) == 1)

        # Ensure that the start and end points are only used if the order is assigned to a batch
        for b in B:
            for i in O:
                model.addConstr(gp.quicksum(x_start[j, b] for j in V if u_ij[i, j] > 0) <= y[i, b])
                model.addConstr(gp.quicksum(x_end[j, b] for j in V if u_ij[i, j] > 0) <= y[i, b])

        # # Item assignments based on orders
        for b in B:
            for i in O:
                for p in P:
                    model.addConstr(gp.quicksum(u_ij[i, j] * z[j, b] for j in V) == s_i[i] * y[i, b])

        # Batching constraints
        for j1 in V:
            for b in B:
                model.addConstr(gp.quicksum(x[j1, j2, b] for j2 in V if j1 != j2) <= z[j1, b])

        for j2 in V:
            for b in B:
                model.addConstr(gp.quicksum(x[j1, j2, b] for j1 in V if j1 != j2) <= z[j2, b])

        # Capacity constraint has to be fulfilled only if the batch is assigned to a picker
        for b in B:
            model.addConstr(gp.quicksum(s_i[i] * y[i, b] for i in O) <= gp.quicksum(capa_p[p] * w[b, p] for p in P))

        # Travel time and process time constraints
        for j1 in V:
            for j2 in V:
                for b in B:
                    for p in P:
                        model.addConstr(T_jbp[j1, b, p] + c[j1, j2, p] <= T_jbp[j2, b, p] + M * (1 - xw[j1, j2, b, p]))

        # if self.objective == "minimize_maximum_completion_time":
        # Start and end time constraints
        for j in V:
            for b in B:
                for p in P:
                    # Start time
                    model.addConstr(c_start[j, p] * xw_start[j, b, p] <= T_jbp[j, b, p])
                    # End time
                    model.addConstr(T_jbp[j, b, p] + c_end[j, p] * xw_end[j, b, p] <= C_bp[b, p])

        # non linear constraints
        for j1 in V:
            for b in B:
                for p in P:
                    for j2 in V:
                        model.addConstr(xw[j1, j2, b, p] <= x[j1, j2, b])
                        model.addConstr(xw[j1, j2, b, p] <= w[b, p])
                        model.addConstr(xw[j1, j2, b, p] >= x[j1, j2, b] + w[b, p] - 1)

                    model.addConstr(xw_start[j1, b, p] <= x_start[j1, b])
                    model.addConstr(xw_end[j1, b, p] <= x_end[j1, b])
                    model.addConstr(xw_start[j1, b, p] <= w[b, p])
                    model.addConstr(xw_end[j1, b, p] <= w[b, p])
                    model.addConstr(xw_start[j1, b, p] >= x_start[j1, b] + w[b, p] - 1)
                    model.addConstr(xw_end[j1, b, p] >= x_end[j1, b] + w[b, p] - 1)

        # # Ensure that C_max is at least as large as any picker's cumulative completion time
        for p in P:
            model.addConstr(C_max_p[p] >= gp.quicksum(C_bp[b, p] for b in B))
            model.addConstr(C_max >= C_max_p[p])

        # for p in P:
        #     model.addConstr(C_max >= gp.quicksum(C_bp[b, p] for b in B))

    # --- Optimize ---
    start_time = time.time()
    model.optimize()

    start_end_time = time.time()
    elapsed_time = start_end_time - start_time

    # --- Output ---
    if model.status == GRB.OPTIMAL:
        print("Optimal output found:")
        print("C_max:", C_max.X)
        # print output

        print("Optimal output found")
        for v in model.getVars():
            if v.X > 0:  # Print only decision variables with positive values
                print(f"{v.VarName}: {v.X}")

        # for b in B:
        #     for p in P:
        #         #if w[b, p].X > 0.5:
        #         print(f"Picker {p}, Batch {b} mit Zeit {C_bp[b, p].X}:")
        #         for i in O:
        #             if y[i, b, p].X > 0.5:
        #                 print(f"Order {i}")
        #             for j in V:
        #                 if xw_start[j, b, p].X > 0.5:
        #                     print(f"Start at {j}")
        #             for j in V:
        #                 if xw_end[j, b, p].X > 0.5:
        #                     print(f"End at {j}")
        #             for j in V:
        #                 for j2 in V:
        #                     if xw[j, j2, b, p].X > 0.5:
        #                         print(f"Item {j} to {j2}")

    else:
        print("No optimal output found.")

    #model.write('model_lp')

    print(f"Die Ausführungszeit 'Batching and Routing' beträgt {elapsed_time} Sekunden.")

    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("infeasible_model.ilp")  # Speichere die IIS in einer Datei