import heapq
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
                 routing_class_kwargs,
                 time_limit: float | None = None):
        super().__init__(pick_cart, articles)
        self.routing_class = routing_class
        self.routing_class_kwargs = routing_class_kwargs
        self._router = routing_class(**routing_class_kwargs)
        self._route_cache = {}
        self.time_limit = time_limit
        self.algo_name = f"{self.routing_class.algo_name}_SavingsBatching"

    def _calc_dist_with_routing_algo(self, orders: list[WarehouseOrder]) -> float:
        """Compute route distance for a list of orders, with caching."""
        key = tuple(sorted(o.order_id for o in orders))
        if key not in self._route_cache:
            # router = self.routing_class(
            #     **self.routing_class_kwargs
            # )
            self._router.reset_parameters()
            batches = [BatchObject(batch_id=0, orders=orders)]
            pick_lists = []
            for batch in batches:
                pick_list = []
                for order in batch.orders:
                    for pos in order.pick_positions:
                        pick_list.append(pos)
                pick_lists.append(pick_list)
            routing_sol = self._router.solve(pick_lists[0])
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
    def _run(self, input_data: list[WarehouseOrder]) -> BatchingSolution:
        self.order_list = input_data
        start_time = time.time()


        batches = {i: BatchObject(batch_id=i, orders=[order])
                   for i, order in enumerate(self.order_list)}
        batch_counter = len(batches)

        savings_heap = []
        for id_a, id_b in combinations(batches.keys(), 2):
            saving = self._calculate_saving(batches[id_a], batches[id_b])
            if saving > 0:
                pair = (min(id_a, id_b), max(id_a, id_b))
                heapq.heappush(savings_heap, (-saving, pair[0], pair[1]))

        while savings_heap:
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break

            neg_saving, id_a, id_b = heapq.heappop(savings_heap)

            if id_a not in batches or id_b not in batches:
                continue

            if -neg_saving <= 0:
                break

            batch_a = batches.pop(id_a)
            batch_b = batches.pop(id_b)
            merged_batch = BatchObject(batch_id=batch_counter, orders=batch_a.orders + batch_b.orders)
            batches[batch_counter] = merged_batch

            for other_id in batches:
                if other_id == batch_counter:
                    continue
                saving = self._calculate_saving(merged_batch, batches[other_id])
                if saving > 0:
                    pair = (min(batch_counter, other_id), max(batch_counter, other_id))
                    heapq.heappush(savings_heap, (-saving, pair[0], pair[1]))

            batch_counter += 1

        return BatchingSolution(batches=list(batches.values()))


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
        self._router = routing_class(**self.routing_class_kwargs)
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
        for i in range(len(batches)):
            for j in range(i + 1, len(batches)):
                if self._time_limit_exceeded():
                    return batches, False

                batch_i = batches[i]
                batch_j = batches[j]

                old_cost_i = self._batch_cost_from_orders(batch_i.orders)
                old_cost_j = self._batch_cost_from_orders(batch_j.orders)
                old_total = old_cost_i + old_cost_j

                for i_idx, order_i in enumerate(batch_i.orders):
                    for j_idx, order_j in enumerate(batch_j.orders):
                        temp_orders_i = batch_i.orders[:i_idx] + [order_j] + batch_i.orders[i_idx + 1:]
                        temp_orders_j = batch_j.orders[:j_idx] + [order_i] + batch_j.orders[j_idx + 1:]

                        if not self.capacity_checker.orders_fit(temp_orders_i) or \
                                not self.capacity_checker.orders_fit(temp_orders_j):
                            continue

                        # Compute new costs
                        new_cost_i = self._batch_cost_from_orders(temp_orders_i)
                        new_cost_j = self._batch_cost_from_orders(temp_orders_j)
                        new_total = new_cost_i + new_cost_j

                        if new_total < old_total - 1e-6:
                            batch_i.orders[i_idx] = order_j
                            batch_j.orders[j_idx] = order_i
                            return batches, True

        return batches, False

    def _shift(self, batches: list[BatchObject]) -> tuple[list[BatchObject], bool]:
        for i in range(len(batches)):
            batch_i = batches[i]

            for order_idx, order in enumerate(batch_i.orders):
                for j in range(len(batches)):
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
            self._router.reset_parameters()
            pick_list = [pos for order in orders for pos in order.pick_positions]
            sol = self._router.solve(pick_list)
            self._route_cache[key] = sol.route.distance
        return self._route_cache[key]

    def _batch_cost(self, batch: BatchObject) -> float:
        """Calculate routing cost for a batch with caching."""
        return self._batch_cost_from_orders(batch.orders)
