# ware_ops_algos/algorithms/order_splitting/order_splitting.py

import math
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field

from ware_ops_algos.algorithms import Algorithm
from ware_ops_algos.algorithms.algorithm import AlgorithmSolution
from ware_ops_algos.domain_models import (
    Articles, Order, OrderPosition, Pallet, PalletSpec,
)


@dataclass
class OrderSplittingSolution(AlgorithmSolution):
    orders: list[Order] = field(default_factory=list)


class OrderSplitting(Algorithm[list[Order], OrderSplittingSolution], ABC):
    def __init__(self, articles: Articles, pallet_spec: PalletSpec = None, **kwargs):
        super().__init__(**kwargs)
        self.articles = articles
        self.pallet_spec = pallet_spec or PalletSpec()
        self.article_map = {a.article_id: a for a in articles.articles}

    @abstractmethod
    def _run(self, input_data: list[Order]) -> OrderSplittingSolution:
        ...


class LayerPackingSplitting(OrderSplitting):
    algo_name = "LayerPackingSplitting"

    def _run(self, input_data: list[Order]) -> OrderSplittingSolution:
        out: list[Order] = []
        for source in input_data:
            out.extend(self._split_one(source))
        return OrderSplittingSolution(orders=out)

    def _split_one(self, order: Order) -> list[Order]:
        positions = sorted(
            order.order_positions,
            key=lambda p: -(self.article_map[p.article_id].height or 0),
        )

        pallets_out: list[Order] = []
        pallet = Pallet(spec=self.pallet_spec)
        committed: list[OrderPosition] = []
        idx = 0

        for position in positions:
            art = self.article_map[position.article_id]
            ks = art.kolli_size or 1
            kollis_left = math.ceil(position.amount / ks)
            kollis_here = 0

            while kollis_left > 0:
                if pallet.can_fit_kolli(art):
                    pallet.add_kolli(art)
                    kollis_here += 1
                    kollis_left -= 1
                else:
                    if kollis_here:
                        chunk = copy(position)
                        chunk.amount = kollis_here * ks
                        committed.append(chunk)
                        kollis_here = 0
                    if not committed:
                        raise ValueError(
                            f"Article {art.article_id} doesn't fit on an empty pallet"
                        )
                    idx += 1
                    pallets_out.append(self._make_pallet(order, committed, idx))
                    pallet = Pallet(spec=self.pallet_spec)
                    committed = []

            if kollis_here:
                chunk = copy(position)
                chunk.amount = kollis_here * ks
                committed.append(chunk)

        if committed:
            idx += 1
            pallets_out.append(self._make_pallet(order, committed, idx))

        return pallets_out

    @staticmethod
    def _make_pallet(source: Order, positions: list[OrderPosition], idx: int) -> Order:
        pallet_id = f"{source.order_id}_P{idx}"
        for pos in positions:
            pos.order_number = pallet_id
        return Order(
            order_id=pallet_id,
            order_date=source.order_date,
            due_date=source.due_date,
            order_positions=positions,
            parent_order_id=source.order_id
        )