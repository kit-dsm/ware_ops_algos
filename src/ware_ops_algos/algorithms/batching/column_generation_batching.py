import gurobipy
import networkx as nx

from ware_ops_algos.algorithms import Batching, Routing, WarehouseOrder, CapacityChecker, BatchingSolution, BatchObject, \
    I, O
from ware_ops_algos.domain_models import PickCart, Articles


class ColumnGenerationBatcher(Batching):
    """
    Implements Branch-Price-and-Cut for JOBPRP following Wahlen and Wahlen, Gschwind 2023.

    Lower bounds are computed by CG and cuts are added dynamically to strengthen linear relaxations.
    """
    def __init__(self,
                 pick_cart: PickCart,
                 articles: Articles,
                 routing_class: type[Routing],
                 routing_class_kwargs: dict,
                 start_batching_class: type[Batching],
                 start_batching_kwargs: dict = None,
                 time_limit: float = 120.0,
                 verbose=False):
        super().__init__(pick_cart, articles)

        self.routing_class = routing_class
        self.routing_class_kwargs = routing_class_kwargs
        self.start_batching_class = start_batching_class
        self.start_batching_kwargs = start_batching_kwargs or {}
        self.time_limit = time_limit
        self._route_cache = {}
        self._router = routing_class(**self.routing_class_kwargs)
        self._start_time = None
        self.algo_name = f"{self.routing_class.algo_name}_{self.start_batching_class.algo_name}_LocalSearchBatching"
        self.verbose = verbose

        self.capacity_checker = CapacityChecker(pick_cart=pick_cart,
                                                articles=articles)


    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        start_batches = self._create_start_batches(input_data)
        """A simple BPC loop"""
        done = False
        while not done:
            # solve the restricted master LP
            self.solve_restricted_master_problem()

            # duals =

            # solve the pricing problem
            new_batches = self.solve_pricing_problem(input_data)

            if not new_batches:
                done = True



    def _create_start_batches(self, order_list: list[WarehouseOrder]) -> list[BatchObject]:
        batching_instance: Batching = self.start_batching_class(
            pick_cart=self.pick_cart,
            articles=self.articles,
            **self.start_batching_kwargs
        )
        batching_sol = batching_instance.solve(order_list)
        return batching_sol.batches

    def __batch_cost_from_orders(self, orders: list[WarehouseOrder]) -> float:
        """Calculate routing cost for a list of orders with caching."""
        if not orders:
            return 0.0

        key = tuple(sorted(o.order_id for o in orders))
        if key not in self._route_cache:
            self._router.reset_parameters()
            pick_list = [pos for order in orders for pos in order.pick_positions]
            sol = self._router.solve(pick_list)
            self._route_cache[key] = sol.route.distance
        return self._route_cache[key]

    def solve_restricted_master_problem(self):
        model = gurobipy.Model("RestrictedMasterProblem")

        obj = (
                gurobipy.quicksum()
        )
        model.setObjective(obj, gurobipy.GRB.MINIMIZE)
        model.addConstr()

    def solve_pricing_problem(self, batches: list[BatchObject]):
        """
        Identifies batches with negative reduced costs. That means a reduction of travel distance.
        SPPRC
        Each node an order (sorted by order number).
        Two arcs: connecting nodes v and v-1 indicating inclusion or not of order v
        Each arc ak v ∈ A, k ∈ {0, 1} is associated with a capacity consumption qk v , a dual price πk v , and a set of orders Ok v .

        (item consumption, dual price pi, singleton order set {})

        Associating sets of orders Ok v with the arcs allows the simultaneous consideration of multiple orders which is needed for the
        incorporation of branching decisions in the pricing

        column generation intuition: We can decompose the JOBRP into a batching and routing problem. The batching problem is the "cheap" master problem
        we can enumerate feasible batches e.g. from O = {1,2,3,4} -> B1 = {1,2} B2 = {1, 3} B3 = {2,4}
        These can be represented as columns with the rows indicate the orders they contain and the distance required to pick them.
                B1 = {1,2}  B2 = {1,3}  B3 = {2,4}
        Order 1     1           1               0
        Order 2     1           0               1
        Order 3     0           1               0
        Order 4     0           0               1
        cost        10          12              9

        We then solve a linear program (LP) which says if we use a constructed batch as part of the solution or not.
        We get e.g. min 10x1 + 12x2 + 9x3

        Constraint would be that every order must occur in exactly one batch.

        The problem is that with growing orders it becomes impossible to enumerate all possible partitions of orders into batches.
        Therefore the master problem is restricted. E.g. starting with only a few batches.

        The LP solution gives us a primal solution x_b, telling us how much of each currently available batch is used.
        It also returns a dual value for every assignment constraint (the dual prices) which correspond to the marginal value associated with satisfying a master constraint.


        :return:
        """
        original_orders = batches[0].orders
        n = len(original_orders)

        sorted_orders = sorted(original_orders, key=lambda o: o.order_id)
        consumptions = {}
        for order in sorted_orders:
            consumption = self.capacity_checker._compute_order_consumption(order)
            consumptions[order.order_id] = consumption

        graph = nx.MultiGraph()
        graph.add_node("source") # artificial souce node

        # init SPPRC multigraph
        for order in sorted_orders:
            graph.add_node(order.order_id)

        from_node = "source"
        graph.add_edge(from_node,
                       sorted_orders[0].order_id,
                       consumption=consumptions[sorted_orders[0].order_id],
                       dual_price=0,
                       order_set={sorted_orders[0].order_id})

        graph.add_edge(from_node,
                       sorted_orders[0].order_id,
                       consumption=0,
                       dual_price=0,
                       order_set={})

        for i, order in enumerate(sorted_orders[1:]):
            from_node = order.order_id
            to_node = sorted_orders[i+1].order_id
            graph.add_edge(from_node,
                           to_node,
                           consumption=consumptions[order.order_id],
                           dual_price=0,
                           order_set={order.order_id})

            graph.add_edge(from_node,
                           to_node,
                           consumption=0,
                           dual_price=0,
                           order_set={})

        selected = ...

        c_b = self.__batch_cost_from_orders(selected)
        reduced_cost = c_b - sum(duals[o] for o in selected)

    def solve(self, input_data: I) -> O:
       solution = self._run(input_data)

