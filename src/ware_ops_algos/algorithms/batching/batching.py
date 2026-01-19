import random
import time
from copy import deepcopy
from enum import Enum
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Optional, Type

import pandas as pd

from ware_ops_algos.algorithms import Algorithm
from ware_ops_algos.algorithms.batching.batching_utils import CapacityChecker
from ware_ops_algos.algorithms.routing.routing import Routing
from ware_ops_algos.algorithms.algorithm import BatchingSolution, BatchObject, WarehouseOrder
from ware_ops_algos.domain_models import Order, PickCart, Articles, DimensionType, Box


class Batching(Algorithm[list[WarehouseOrder], BatchingSolution], ABC):
    """Batching class to batch orders"""

    def __init__(self, pick_cart: PickCart, articles: Articles, **kwargs):
        # instance attributes
        super().__init__(**kwargs)

        # self.picker = picker
        # self.picker_capa = pick_cart.
        self.execution_time = None
        self.pick_cart = pick_cart
        self.articles = articles
        self.capacity_checker = CapacityChecker(pick_cart=pick_cart,
                                                articles=articles)

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        pass


class PriorityBatching(Batching):
    """Priority batching class to batch orders based on sorting criterion."""

    @abstractmethod
    def _sorted_orders(self) -> list[WarehouseOrder]:
        pass

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        self.order_list = input_data
        sorted_orders = self._sorted_orders()
        batched_list: list[BatchObject] = []
        batch_number = 0
        current_batch: list[WarehouseOrder] = []

        for order in sorted_orders:
            # Check if we can add this order to current batch
            if self.capacity_checker.can_add_order(current_batch, order):
                current_batch.append(order)
            else:
                # Current batch is full - start new batch
                if current_batch:
                    batched_list.append(BatchObject(batch_id=batch_number, orders=current_batch))
                    batch_number += 1

                # Check if order fits alone
                if self.capacity_checker.can_add_order([], order):
                    current_batch = [order]
                else:
                    print(f"Order {order.order_id} exceeds capacity, excluded")
                    current_batch = []

        if current_batch:
            batched_list.append(BatchObject(batch_id=batch_number, orders=current_batch))

        return BatchingSolution(batches=batched_list)


class OrderNrFifoBatching(PriorityBatching):
    """First in First out batching based on order number."""
    algo_name = "OrderNrFiFoBatching"

    def _sorted_orders(self) -> list[WarehouseOrder]:
        return sorted(self.order_list, key=lambda o: o.order_id)


class FifoBatching(PriorityBatching):
    """
    First in First out batching class to batch orders
    """
    algo_name = "FiFoBatching"

    # def __init__(self, capacity):
    #     super().__init__(capacity)

    def _sorted_orders(self) -> list[WarehouseOrder]:
        return sorted(self.order_list, key=lambda o: o.order_date)


class RandomBatching(PriorityBatching):
    """
    First in First out batching class to batch orders
    """
    algo_name = "RandomBatching"

    def __init__(self, pick_cart: PickCart, articles: Articles, seed=44):
        super().__init__(pick_cart, articles)

        self.seed = seed
        self.batch_number = 0

    def _sorted_orders(self):
        """
        Sorting the batching list
        """
        shuffled = self.order_list.copy()
        random.Random(self.seed).shuffle(shuffled)
        return shuffled


class DueDateBatching(PriorityBatching):
    """
    First in First out batching class to batch orders
    """
    algo_name = "DueDateBatching"

    def __init__(self, pick_cart: PickCart, articles: Articles):
        super().__init__(pick_cart, articles)

    def _sorted_orders(self) -> list[WarehouseOrder]:
        return sorted(self.order_list, key=lambda o: o.due_date)


class SavingsBatching(Batching, ABC):
    """
    Base class for savings-based batching algorithms.
    """

    def __init__(self,
                 pick_cart: PickCart,
                 articles: Articles,
                 routing_class: Type[Routing],
                 routing_class_kwargs):
        super().__init__(pick_cart, articles)
        self.routing_class = routing_class
        self.routing_class_kwargs = routing_class_kwargs
        self._route_cache = {}
        self.algo_name = f"{self.routing_class.algo_name}_SavingsBatching"

    def _calc_dist_with_routing_algo(self, orders: list[WarehouseOrder]) -> float:
        """Compute route distance for a list of orders, with caching."""
        key = tuple(sorted(o.order_id for o in orders))
        if key not in self._route_cache:
            router = self.routing_class(
                **self.routing_class_kwargs
            )
            batches = [BatchObject(batch_id=0, orders=orders)]
            pick_lists = []
            for batch in batches:
                pick_list = []
                for order in batch.orders:
                    for pos in order.pick_positions:
                        pick_list.append(pos)
                pick_lists.append(pick_list)
            routing_sol = router.solve(pick_lists[0])
            self._route_cache[key] = routing_sol.route.distance
        return self._route_cache[key]

    def _calculate_saving(self, batch_a: BatchObject, batch_b: BatchObject) -> float:
        """Calculate saving from merging two batches."""
        orders_a = batch_a.orders
        orders_b = batch_b.orders
        merged_orders = orders_a + orders_b

        # Capacity check
        # total_amount = sum(pos.in_store for o in merged_orders for pos in o.pick_positions)
        # if total_amount > self.picker_capa:
        #     return 0
        if not self.capacity_checker.orders_fit(merged_orders):
            return 0

        dist_a = self._calc_dist_with_routing_algo(orders_a)
        dist_b = self._calc_dist_with_routing_algo(orders_b)
        dist_comb = self._calc_dist_with_routing_algo(merged_orders)

        return dist_a + dist_b - dist_comb


class ClarkAndWrightBatching(SavingsBatching):
    """
    C&W(ii) batching algorithm using dataclasses.
    """
    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        self.order_list = input_data
        # Start: each order is its own batch
        batches = [BatchObject(batch_id=i, orders=[order]) for i, order in enumerate(self.order_list)]
        batch_counter = len(batches)

        while True:
            # Compute savings for all pairs
            savings = {}
            for batch_a, batch_b in combinations(batches, 2):
                savings[(batch_a.batch_id, batch_b.batch_id)] = self._calculate_saving(batch_a, batch_b)

            # Stop if no positive savings
            if not savings or max(savings.values()) <= 0:
                break

            # Pick best pair
            best_key = max(savings, key=savings.get)
            id_a, id_b = best_key
            a = next(b for b in batches if b.batch_id == id_a)
            b = next(b for b in batches if b.batch_id == id_b)

            # Merge batches
            merged_batch = BatchObject(batch_id=batch_counter, orders=a.orders + b.orders)
            batch_counter += 1

            # Remove old, add new
            batches = [batch for batch in batches if batch.batch_id not in best_key]
            batches.append(merged_batch)

        return BatchingSolution(batches=batches)


class SeedCriteria(str, Enum):
    RANDOM = "random"
    FEWEST_POSITIONS = "fewest_positions"
    MOST_POSITIONS = "most_positions"
    CLOSEST_TO_DEPOT = "closest_to_depot"


class SimilarityMeasure(str, Enum):
    SHARED_ARTICLES = "shared_articles"
    MIN_DISTANCE = "min_distance"


class SeedBatching(Batching):
    def __init__(
        self,
        pick_cart: PickCart,
        articles: Articles,
        distance_matrix: Optional[pd.DataFrame] = None,
        seed_criterion: SeedCriteria = SeedCriteria.RANDOM,
        similarity_measure: SimilarityMeasure = SimilarityMeasure.SHARED_ARTICLES,
        start_node: Optional[tuple[int, int]] = None,
    ):
        super().__init__(pick_cart, articles)
        self.distance_matrix = distance_matrix
        self.similarity_measure = similarity_measure
        self.seed_criterion = seed_criterion
        self.start_node = start_node
        self.algo_name = f"{self.seed_criterion.value}_{self.similarity_measure.value}_SeedBatching"


    @staticmethod
    def shared_article_similarity(seed_order: WarehouseOrder, other_order: WarehouseOrder):
        seed_articles = {pos.article_id for pos in seed_order.pick_positions}
        other_articles = {pos.article_id for pos in other_order.pick_positions}
        return -len(seed_articles & other_articles)

    def min_distance_similarity(self, seed_order: WarehouseOrder, other_order: WarehouseOrder):
        return min(
            self.distance_matrix.at[pos_a.pick_node, pos_b.pick_node]
            for pos_a in seed_order.pick_positions
            for pos_b in other_order.pick_positions
        )

    @staticmethod
    def rand_seed_order(candidate_orders: list[WarehouseOrder]) -> WarehouseOrder:
        return random.choice(candidate_orders)

    @staticmethod
    def most_positions_seed_order(candidate_orders: list[WarehouseOrder]) -> WarehouseOrder:
        return max(candidate_orders, key=lambda o: len(o.pick_positions))

    @staticmethod
    def fewest_positions_seed_order(candidate_orders: list[WarehouseOrder]) -> WarehouseOrder:
        return min(candidate_orders, key=lambda o: len(o.pick_positions))

    def closest_to_depot_seed_order(self, candidate_orders: list[WarehouseOrder]) -> WarehouseOrder:
        return min(
            candidate_orders,
            key=lambda o: min(
                self.distance_matrix.at[self.start_node, pos.pick_node]
                for pos in o.pick_positions
            ),
        )

    def similarity(self, seed_order: WarehouseOrder, other_order: WarehouseOrder):
        if self.similarity_measure == SimilarityMeasure.SHARED_ARTICLES:
            return self.shared_article_similarity(seed_order, other_order)
        elif self.similarity_measure == SimilarityMeasure.MIN_DISTANCE:
            return self.min_distance_similarity(seed_order, other_order)
        else:
            raise ValueError("Invalid similarity measure.")

    def get_seed_order(self,
                       seed_criterion: SeedCriteria,
                       candidate_orders: list[WarehouseOrder]):
        if seed_criterion == SeedCriteria.RANDOM:
            seed_order = self.rand_seed_order(candidate_orders)

        elif seed_criterion == SeedCriteria.MOST_POSITIONS:
            seed_order = self.most_positions_seed_order(candidate_orders)

        elif seed_criterion == SeedCriteria.FEWEST_POSITIONS:
            seed_order = self.fewest_positions_seed_order(candidate_orders)

        elif seed_criterion == SeedCriteria.CLOSEST_TO_DEPOT:
            seed_order = self.closest_to_depot_seed_order(candidate_orders)

        else:
            raise ValueError("Invalid seed criterion.")

        return seed_order

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        self.order_list = input_data
        remaining_orders = self.order_list.copy()
        batched_list: list[BatchObject] = []

        while remaining_orders:
            seed_order = self.get_seed_order(self.seed_criterion, remaining_orders)

            current_batch = [seed_order]
            current_capa = sum(pos.in_store for pos in seed_order.pick_positions)
            remaining_orders.remove(seed_order)
            sorted_remaining = sorted(
                remaining_orders,
                key=lambda o: self.similarity(seed_order, o)
            )

            for candidate in sorted_remaining:
                # demand = sum(pos.in_store for pos in candidate.pick_positions)
                # if current_capa + demand <= self.picker_capa:
                #     current_batch.append(candidate)
                #     current_capa += demand
                if self.capacity_checker.can_add_order(current_batch, candidate):
                    current_batch.append(candidate)

            for o in current_batch:
                if o in remaining_orders:
                    remaining_orders.remove(o)

            batched_list.append(BatchObject(batch_id=len(batched_list), orders=current_batch))

        return BatchingSolution(batches=batched_list)


class LocalSearchBatching(Batching):
    def __init__(self,
                 pick_cart: PickCart,
                 articles: Articles,
                 routing_class: type[Routing],
                 routing_class_kwargs: dict,
                 start_batching_class: type[Batching],
                 start_batching_kwargs: dict = None,
                 time_limit: float = 120.0):
        super().__init__(pick_cart, articles)
        self.routing_class = routing_class
        self.routing_class_kwargs = routing_class_kwargs
        self.start_batching_class = start_batching_class
        self.start_batching_kwargs = start_batching_kwargs or {}
        self.time_limit = time_limit
        self._route_cache = {}
        self._start_time = None
        self.algo_name = f"{self.routing_class.algo_name}_{self.start_batching_class.algo_name}_LocalSearchBatching"

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        self.order_list = input_data
        start_batches = self._create_start_batches()
        batches = self._local_search(start_batches)
        return BatchingSolution(batches=batches)

    def _create_start_batches(self) -> list[BatchObject]:
        batching_instance: Batching = self.start_batching_class(
            pick_cart=self.pick_cart,
            articles=self.articles,
            **self.start_batching_kwargs
        )
        batching_sol = batching_instance.solve(self.order_list)
        return batching_sol.batches

    def _time_limit_exceeded(self) -> bool:
        """Check if time limit has been exceeded."""
        return time.time() - self._start_time > self.time_limit

    def _local_search(self, batches: list[BatchObject]) -> list[BatchObject]:
        """Main local search loop with proper efficiency and time limit checking."""
        self._start_time = time.time()

        initial_cost = sum(self._batch_cost_from_orders(b.orders) for b in batches)
        print(f"\n{'=' * 60}")
        print(f"Local Search Started")
        print(f"{'=' * 60}")
        print(f"Initial solution: {len(batches)} batches, cost: {initial_cost:.2f}")
        print(f"Time limit: {self.time_limit}s")
        print(f"{'=' * 60}\n")

        iteration = 0
        swap_improvements = 0
        shift_improvements = 0

        while True:
            if self._time_limit_exceeded():
                elapsed = time.time() - self._start_time
                print(f"\n⏱️  Time limit exceeded after {elapsed:.2f}s")
                break

            iteration += 1
            overall_improved = False
            iter_start_cost = sum(self._batch_cost_from_orders(b.orders) for b in batches)

            # Exhaust all SWAP improvements
            swap_count = 0
            swap_improved = True
            while swap_improved and not self._time_limit_exceeded():
                batches, swap_improved = self._swap(batches)
                if swap_improved:
                    swap_count += 1
                    swap_improvements += 1
                    overall_improved = True

            # Exhaust all SHIFT improvements
            shift_count = 0
            shift_improved = True
            while shift_improved and not self._time_limit_exceeded():
                batches, shift_improved = self._shift(batches)
                if shift_improved:
                    shift_count += 1
                    shift_improvements += 1
                    overall_improved = True

            if overall_improved:
                iter_end_cost = sum(self._batch_cost_from_orders(b.orders) for b in batches)
                improvement = iter_start_cost - iter_end_cost
                elapsed = time.time() - self._start_time

                print(f"Iteration {iteration}: "
                      f"swaps={swap_count}, shifts={shift_count} | "
                      f"cost: {iter_end_cost:.2f} "
                      f"(Δ {improvement:+.2f}) | "
                      f"{len(batches)} batches | "
                      f"cache: {len(self._route_cache)} | "
                      f"{elapsed:.1f}s")

            if not overall_improved:
                break

        # Final summary
        final_cost = sum(self._batch_cost_from_orders(b.orders) for b in batches)
        total_improvement = initial_cost - final_cost
        elapsed = time.time() - self._start_time

        print(f"\n{'=' * 60}")
        print(f"Local Search Completed")
        print(f"{'=' * 60}")
        print(f"Iterations: {iteration}")
        print(f"Total improvements: {swap_improvements + shift_improvements} "
              f"(swaps: {swap_improvements}, shifts: {shift_improvements})")
        print(f"Initial cost: {initial_cost:.2f}")
        print(f"Final cost: {final_cost:.2f}")
        print(f"Total improvement: {total_improvement:.2f} ({100 * total_improvement / initial_cost:.1f}%)")
        print(f"Final batches: {len(batches)}")
        print(f"Route cache size: {len(self._route_cache)}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"{'=' * 60}\n")

        return batches

    def _swap(self, batches: list[BatchObject]) -> tuple[list[BatchObject], bool]:
        """
        Try to exchange two orders from different batches.
        NO DEEPCOPY - modify in place when improvement found.
        """
        for i in range(len(batches)):
            for j in range(i + 1, len(batches)):
                # Check time limit periodically
                if self._time_limit_exceeded():
                    return batches, False

                batch_i = batches[i]
                batch_j = batches[j]

                for i_idx, order_i in enumerate(batch_i.orders):
                    for j_idx, order_j in enumerate(batch_j.orders):
                        # Check capacity BEFORE computing costs
                        # Create temporary order lists without copying entire batches
                        temp_orders_i = batch_i.orders[:i_idx] + [order_j] + batch_i.orders[i_idx + 1:]
                        temp_orders_j = batch_j.orders[:j_idx] + [order_i] + batch_j.orders[j_idx + 1:]

                        if not self.capacity_checker.orders_fit(temp_orders_i) or \
                                not self.capacity_checker.orders_fit(temp_orders_j):
                            continue

                        # Compute old costs (cached)
                        old_cost_i = self._batch_cost_from_orders(batch_i.orders)
                        old_cost_j = self._batch_cost_from_orders(batch_j.orders)
                        old_total = old_cost_i + old_cost_j

                        # Compute new costs
                        new_cost_i = self._batch_cost_from_orders(temp_orders_i)
                        new_cost_j = self._batch_cost_from_orders(temp_orders_j)
                        new_total = new_cost_i + new_cost_j

                        if new_total < old_total - 1e-6:  # Small epsilon for float comparison
                            # Apply improvement IN PLACE
                            batch_i.orders[i_idx] = order_j
                            batch_j.orders[j_idx] = order_i
                            return batches, True

        return batches, False

    def _shift(self, batches: list[BatchObject]) -> tuple[list[BatchObject], bool]:
        """
        Try to move an order from one batch to another.
        NO DEEPCOPY - modify in place when improvement found.
        """
        for i in range(len(batches)):
            batch_i = batches[i]

            for order_idx, order in enumerate(batch_i.orders):
                for j in range(len(batches)):
                    # Check time limit periodically
                    if self._time_limit_exceeded():
                        return batches, False

                    if i == j:
                        continue

                    batch_j = batches[j]

                    # Skip if source batch would become empty
                    if len(batch_i.orders) <= 1:
                        continue

                    # Check capacity BEFORE computing costs
                    temp_orders_i = batch_i.orders[:order_idx] + batch_i.orders[order_idx + 1:]
                    temp_orders_j = batch_j.orders + [order]

                    if not self.capacity_checker.orders_fit(temp_orders_i) or \
                            not self.capacity_checker.orders_fit(temp_orders_j):
                        continue

                    # Compute costs
                    old_cost_i = self._batch_cost_from_orders(batch_i.orders)
                    old_cost_j = self._batch_cost_from_orders(batch_j.orders)
                    old_total = old_cost_i + old_cost_j

                    new_cost_i = self._batch_cost_from_orders(temp_orders_i)
                    new_cost_j = self._batch_cost_from_orders(temp_orders_j)
                    new_total = new_cost_i + new_cost_j

                    if new_total < old_total - 1e-6:
                        # Apply improvement IN PLACE
                        batch_i.orders.pop(order_idx)
                        batch_j.orders.append(order)

                        # Clean up empty batches if needed
                        if not batch_i.orders:
                            batches.pop(i)

                        return batches, True

        return batches, False

    def _batch_cost_from_orders(self, orders: list[WarehouseOrder]) -> float:
        """Calculate routing cost for a list of orders with caching."""
        if not orders:
            return 0.0

        key = tuple(sorted(o.order_id for o in orders))
        if key not in self._route_cache:
            routing_instance = self.routing_class(**self.routing_class_kwargs)
            pick_list = [pos for order in orders for pos in order.pick_positions]
            routing_instance._run(pick_list)
            self._route_cache[key] = routing_instance.distance
        return self._route_cache[key]

    def _batch_cost(self, batch: BatchObject) -> float:
        """Calculate routing cost for a batch with caching."""
        return self._batch_cost_from_orders(batch.orders)


class LocalSearchBatchingFast(Batching):

    def __init__(self,
                 pick_cart: PickCart,
                 articles: Articles,
                 routing_class: type[Routing],
                 routing_class_kwargs: dict,
                 start_batching_class: type[Batching],
                 start_batching_kwargs: dict = None,
                 time_limit: float = 120.0):
        super().__init__(pick_cart, articles)
        self.routing_class = routing_class
        self.routing_class_kwargs = routing_class_kwargs
        self.start_batching_class = start_batching_class
        self.start_batching_kwargs = start_batching_kwargs or {}
        self.time_limit = time_limit
        self._route_cache = {}
        self._start_time = None
        self.algo_name = (
            f"{self.routing_class.algo_name}_"
            f"{self.start_batching_class.algo_name}_LocalSearchBatching"
        )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        self.order_list = input_data
        batches = self._create_start_batches()

        # 🔹 PRECOMPUTE order consumption ONCE
        self._order_consumption = {
            o.order_id: self.capacity_checker._compute_order_consumption(o)
            for o in self.order_list
        }

        batches = self._local_search(batches)
        return BatchingSolution(batches=batches)

    def _create_start_batches(self) -> list[BatchObject]:
        batching_instance: Batching = self.start_batching_class(
            pick_cart=self.pick_cart,
            articles=self.articles,
            **self.start_batching_kwargs
        )
        return batching_instance.solve(self.order_list).batches

    def _time_limit_exceeded(self) -> bool:
        return time.time() - self._start_time > self.time_limit

    # ------------------------------------------------------------------
    # Local search
    # ------------------------------------------------------------------

    def _local_search(self, batches: list[BatchObject]) -> list[BatchObject]:
        self._start_time = time.time()

        n_dim = self.pick_cart.n_dimension
        max_capacity = [
            cap * self.pick_cart.n_boxes
            for cap in self.pick_cart.capacities
        ]

        # 🔹 INITIAL batch consumptions
        batch_consumption = [
            self.capacity_checker._compute_consumption(b.orders)
            for b in batches
        ]

        # 🔹 INITIAL batch routing costs
        batch_cost = [
            self._batch_cost_from_orders(b.orders)
            for b in batches
        ]

        initial_cost = batch_cost
        print(f"Initial cost: {sum(initial_cost)}")
        while not self._time_limit_exceeded():
            improved = False

            # ---------------- SWAP ----------------
            for i in range(len(batches)):
                bi = batches[i]
                for j in range(i + 1, len(batches)):
                    bj = batches[j]

                    for oi_idx, oi in enumerate(bi.orders):
                        cons_i = self._order_consumption[oi.order_id]

                        for oj_idx, oj in enumerate(bj.orders):
                            cons_j = self._order_consumption[oi.order_id]

                            # Incremental capacity check
                            feasible = True
                            for d in range(n_dim):
                                if (batch_consumption[i][d] - cons_i[d] + cons_j[d] > max_capacity[d] or
                                    batch_consumption[j][d] - cons_j[d] + cons_i[d] > max_capacity[d]):
                                    feasible = False
                                    break
                            if not feasible:
                                continue

                            # Apply swap
                            bi.orders[oi_idx], bj.orders[oj_idx] = oj, oi

                            new_ci = self._batch_cost_from_orders(bi.orders)
                            new_cj = self._batch_cost_from_orders(bj.orders)
                            delta = (new_ci + new_cj) - (batch_cost[i] + batch_cost[j])

                            if delta < -1e-6:
                                # Accept
                                for d in range(n_dim):
                                    batch_consumption[i][d] += cons_j[d] - cons_i[d]
                                    batch_consumption[j][d] += cons_i[d] - cons_j[d]
                                batch_cost[i] = new_ci
                                batch_cost[j] = new_cj
                                improved = True
                                break
                            else:
                                # Revert
                                bi.orders[oi_idx], bj.orders[oj_idx] = oi, oj

                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

            # ---------------- SHIFT ----------------
            for i in range(len(batches)):
                bi = batches[i]
                if len(bi.orders) <= 1:
                    continue

                for idx, o in enumerate(bi.orders):
                    cons_o = self._order_consumption[o.order_id]

                    for j in range(len(batches)):
                        if i == j:
                            continue

                        bj = batches[j]

                        feasible = True
                        for d in range(n_dim):
                            if (batch_consumption[i][d] - cons_o[d] > max_capacity[d] or
                                batch_consumption[j][d] + cons_o[d] > max_capacity[d]):
                                feasible = False
                                break
                        if not feasible:
                            continue

                        # Apply shift
                        bi.orders.pop(idx)
                        bj.orders.append(o)

                        new_ci = self._batch_cost_from_orders(bi.orders)
                        new_cj = self._batch_cost_from_orders(bj.orders)
                        delta = (new_ci + new_cj) - (batch_cost[i] + batch_cost[j])

                        if delta < -1e-6:
                            for d in range(n_dim):
                                batch_consumption[i][d] -= cons_o[d]
                                batch_consumption[j][d] += cons_o[d]
                            batch_cost[i] = new_ci
                            batch_cost[j] = new_cj

                            if not bi.orders:
                                del batches[i]
                                del batch_consumption[i]
                                del batch_cost[i]

                            improved = True
                            break
                        else:
                            # Revert
                            bj.orders.pop()
                            bi.orders.insert(idx, o)

                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break
        print(f"\n{'=' * 60}")
        print(f"Local Search Completed")
        print(f"{'=' * 60}")
        print(f"Final cost: {sum(batch_cost)}")
        print(f"Final batches: {len(batches)}")
        print(f"Route cache size: {len(self._route_cache)}")
        print(f"{'=' * 60}\n")

        return batches

    # ------------------------------------------------------------------
    # Routing cost
    # ------------------------------------------------------------------

    def _batch_cost_from_orders(self, orders: list[WarehouseOrder]) -> float:
        if not orders:
            return 0.0

        key = tuple(sorted(o.order_id for o in orders))
        if key not in self._route_cache:
            routing_instance = self.routing_class(**self.routing_class_kwargs)
            pick_list = [pos for o in orders for pos in o.pick_positions]
            routing_instance._run(pick_list)
            self._route_cache[key] = routing_instance.distance
        return self._route_cache[key]


class IteratedLocalSearchBatching(Batching):
    def __init__(self,
                 capacity: int,
                 routing_class: type[Routing],
                 routing_class_kwargs: dict,
                 start_batching_class: type[Batching],
                 start_batching_kwargs: dict = None,
                 rearrangement_parameter: float = 0.3,
                 threshold_parameter: float = 0.01,
                 time_limit: float = 30.0):
        super().__init__(capacity)
        self.routing_class = routing_class
        self.routing_class_kwargs = routing_class_kwargs
        self.start_batching_class = start_batching_class
        self.start_batching_kwargs = start_batching_kwargs or {}
        self.rearrangement_parameter = rearrangement_parameter
        self.threshold_parameter = threshold_parameter
        self.time_limit = time_limit
        self._route_cache = {}

    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        self.order_list = input_data
        start_batches = self._create_start_batches()
        batches = self._iterated_local_search(start_batches)
        return BatchingSolution(batches=batches)

    def _create_start_batches(self) -> list[BatchObject]:
        batching_instance = self.start_batching_class(
            capacity=self.picker_capa,
            **self.start_batching_kwargs
        )
        batching_sol = batching_instance.solve(self.order_list)
        return batching_sol.batches

    def _iterated_local_search(self, s_start: list[BatchObject]) -> list[BatchObject]:
        s_initial = deepcopy(s_start)
        s_asterisk = self._local_search_phase(s_initial)
        s_incumbent = deepcopy(s_asterisk)
        improvement_found = False

        start_time = time.time()
        while True:
            s = self._perturbation_phase(deepcopy(s_incumbent))
            s = self._local_search_phase(s)

            d_s = self._total_distance(s)
            d_star = self._total_distance(s_asterisk)

            if d_s < d_star:
                s_asterisk = deepcopy(s)
                s_incumbent = deepcopy(s)
                improvement_found = True

            if time.time() - start_time > self.time_limit:
                if improvement_found:
                    improvement_found = False
                    start_time = time.time()
                elif d_s - d_star < self.threshold_parameter * d_star:
                    return s
                else:
                    return s_asterisk

    def _local_search_phase(self, batches: list[BatchObject]) -> list[BatchObject]:
        improved = True
        while improved:
            improved = False
            new_batches = self._local_search_swap(deepcopy(batches))
            assert all(batch.orders for batch in new_batches), "Found empty batch after swap"

            if self._total_distance(new_batches) < self._total_distance(batches):
                batches = new_batches
                improved = True

            new_batches = self._local_search_shift(deepcopy(batches))
            assert all(batch.orders for batch in new_batches), "Found empty batch after shift"

            if self._total_distance(new_batches) < self._total_distance(batches):
                batches = new_batches
                improved = True
        # try:
        #     assert all(batch.orders for batch in batches), "Found empty batch after phase local_search"
        # except AssertionError:
        #     print(batches)
        return batches

    def _local_search_swap(self, batches: list[BatchObject]) -> list[BatchObject]:
        is_optimal = False
        while not is_optimal:
            improvement_found = False
            for i, batch_i in enumerate(batches):
                for j, batch_j in enumerate(batches):
                    if i == j:
                        continue

                    for i_idx, order_i in enumerate(batch_i.orders):
                        for j_idx, order_j in enumerate(batch_j.orders):
                            temp_i = deepcopy(batch_i)
                            temp_j = deepcopy(batch_j)

                            temp_i.orders[i_idx] = order_j
                            temp_j.orders[j_idx] = order_i

                            if not self._batch_within_capacity(temp_i) or not self._batch_within_capacity(temp_j):
                                continue

                            if self._batch_cost(temp_i) + self._batch_cost(temp_j) < \
                               self._batch_cost(batch_i) + self._batch_cost(batch_j):
                                batches[i] = temp_i
                                batches[j] = temp_j
                                improvement_found = True
                                break
                        if improvement_found:
                            break
                    if improvement_found:
                        break
                if improvement_found:
                    break
            if not improvement_found:
                is_optimal = True
        batches = self._remove_empty_batches(batches)
        assert all(batch.orders for batch in batches), "Found empty batch after swap"
        return batches

    def _local_search_shift(self, batches: list[BatchObject]) -> list[BatchObject]:
        is_optimal = False
        while not is_optimal:
            improvement_found = False
            for i, batch_i in enumerate(batches):
                for j, batch_j in enumerate(batches):
                    if i == j:
                        continue

                    for order in list(batch_i.orders):
                        temp_i = deepcopy(batch_i)
                        temp_j = deepcopy(batch_j)

                        temp_i.orders.remove(order)
                        temp_j.orders.append(order)

                        if not temp_i.orders or not temp_j.orders:
                            continue

                        if not self._batch_within_capacity(temp_i) or not self._batch_within_capacity(temp_j):
                            continue
                        if self._batch_cost(temp_i) + self._batch_cost(temp_j) < \
                           self._batch_cost(batch_i) + self._batch_cost(batch_j):
                            batches[i] = temp_i
                            batches[j] = temp_j
                            batches = self._remove_empty_batches(batches)
                            improvement_found = True
                            break
                    if improvement_found:
                        break
                if improvement_found:
                    break
            if not improvement_found:
                is_optimal = True
        batches = self._remove_empty_batches(batches)
        assert all(batch.orders for batch in batches), "Found empty batch after shift"
        return batches

    def _perturbation_phase(self, batches: list[BatchObject]) -> list[BatchObject]:
        iterations = int(len(batches) * self.rearrangement_parameter + 1)
        batches = deepcopy(batches)

        for _ in range(iterations):
            if len(batches) < 2:
                break
            b1, b2 = random.sample(batches, 2)
            q = random.randint(1, min(len(b1.orders), len(b2.orders)))

            selected_1 = b1.orders[:q]
            selected_2 = b2.orders[:q]

            b1.orders = b1.orders[q:]
            b2.orders = b2.orders[q:]

            new_orders = []

            if self._orders_fit(b2.orders + selected_1):
                b2.orders += selected_1
            else:
                new_orders += selected_1

            if self._orders_fit(b1.orders + selected_2):
                b1.orders += selected_2
            else:
                new_orders += selected_2

            if new_orders:
                new_batch_id = max(b.batch_id for b in batches) + 1
                batches.append(BatchObject(batch_id=new_batch_id, orders=new_orders))
        batches = self._remove_empty_batches(batches)
        assert all(batch.orders for batch in batches), "Found empty batch after pertubation"
        return batches

    def _batch_cost(self, batch: BatchObject) -> float:
        key = tuple(sorted(o.order_id for o in batch.orders))
        if key not in self._route_cache:
            routing_instance = self.routing_class(**self.routing_class_kwargs)
            pick_list = [pos for order in batch.orders for pos in order.pick_positions]
            routing_instance._run(pick_list)
            self._route_cache[key] = routing_instance.distance
        return self._route_cache[key]

    def _total_distance(self, batches: list[BatchObject]) -> float:
        return sum(self._batch_cost(batch) for batch in batches)

    def _batch_within_capacity(self, batch: BatchObject) -> bool:
        return self._orders_fit(batch.orders)

    def _orders_fit(self, orders: list[WarehouseOrder]) -> bool:
        return sum(pos.in_store for order in orders for pos in order.pick_positions) <= self.picker_capa

    @staticmethod
    def _remove_empty_batches(batches: list[BatchObject]) -> list[BatchObject]:
        return [batch for batch in batches if batch.orders]
