import random
from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

from ware_ops_algos.algorithms import WarehouseOrder, Batching, BatchingSolution, BatchObject
from ware_ops_algos.domain_models import PickCart, Articles


class SimilarityMeasure(ABC):
    name: ClassVar[str]

    @abstractmethod
    def __call__(
            self,
            seed: WarehouseOrder,
            other: WarehouseOrder,
    ) -> float:
        ...


class SeedCriterion(ABC):
    name: ClassVar[str]

    @abstractmethod
    def __call__(
            self,
            candidates: list[WarehouseOrder],
    ) -> WarehouseOrder:
        ...


class SharedArticlesSimilarity(SimilarityMeasure):
    name = "SHARED_ARTICLES"

    def __call__(
            self,
            seed: WarehouseOrder,
            other: WarehouseOrder,
    ) -> float:
        seed_articles = {
            position.article_id
            for position in seed.pick_positions
        }
        other_articles = {
            position.article_id
            for position in other.pick_positions
        }
        return -len(seed_articles & other_articles)


class MinDistanceSimilarity(SimilarityMeasure):
    name = "MIN_DISTANCE"

    def __init__(self, distance_matrix: pd.DataFrame):
        self.distance_matrix = distance_matrix

    def __call__(
            self,
            seed: WarehouseOrder,
            other: WarehouseOrder,
    ) -> float:
        return min(
            self.distance_matrix.at[
                seed_position.pick_node,
                other_position.pick_node,
            ]
            for seed_position in seed.pick_positions
            for other_position in other.pick_positions
        )


class RandomSeed(SeedCriterion):
    name = "RANDOM"

    def __init__(self, seed: int = 43):
        self._rng = random.Random(seed)

    def __call__(
            self,
            candidates: list[WarehouseOrder],
    ) -> WarehouseOrder:
        return self._rng.choice(candidates)


class MostPositionsSeed(SeedCriterion):
    name = "MOST_POSITIONS"

    def __call__(
            self,
            candidates: list[WarehouseOrder],
    ) -> WarehouseOrder:
        return max(
            candidates,
            key=lambda order: len(order.pick_positions),
        )


class FewestPositionsSeed(SeedCriterion):
    name = "FEWEST_POSITIONS"

    def __call__(
            self,
            candidates: list[WarehouseOrder],
    ) -> WarehouseOrder:
        return min(
            candidates,
            key=lambda order: len(order.pick_positions),
        )


class ClosestToDepotSeed(SeedCriterion):
    name = "CLOSEST_TO_DEPOT"

    def __init__(
            self,
            distance_matrix: pd.DataFrame,
            start_node: tuple[int, int],
    ):
        self.distance_matrix = distance_matrix
        self.start_node = start_node

    def __call__(
            self,
            candidates: list[WarehouseOrder],
    ) -> WarehouseOrder:
        return min(
            candidates,
            key=lambda order: min(
                self.distance_matrix.at[
                    self.start_node,
                    position.pick_node,
                ]
                for position in order.pick_positions
            ),
        )


class SeedBatchingModular(Batching):
    def __init__(
            self,
            pick_cart: PickCart,
            articles: Articles,
            seed_criterion: SeedCriterion,
            similarity_measure: SimilarityMeasure,
    ):
        super().__init__(pick_cart, articles)

        self.seed_criterion = seed_criterion
        self.similarity_measure = similarity_measure

        self.algo_name = (
            f"{seed_criterion.name}_"
            f"{similarity_measure.name}_"
            f"SeedBatching"
        )

    def _run(
            self,
            input_data: list[WarehouseOrder],
    ) -> BatchingSolution:
        remaining_orders = input_data.copy()
        batches: list[BatchObject] = []

        while remaining_orders:
            seed_order = self.seed_criterion(remaining_orders)

            current_batch = [seed_order]
            remaining_orders.remove(seed_order)

            sorted_candidates = sorted(
                remaining_orders,
                key=lambda candidate: self.similarity_measure(
                    seed_order,
                    candidate,
                ),
            )

            for candidate in sorted_candidates:
                if self.capacity_checker.can_add_order(
                        current_batch,
                        candidate,
                ):
                    current_batch.append(candidate)

            for order in current_batch:
                if order in remaining_orders:
                    remaining_orders.remove(order)

            batches.append(
                BatchObject(
                    batch_id=len(batches),
                    orders=current_batch,
                )
            )

        return BatchingSolution(batches=batches)
