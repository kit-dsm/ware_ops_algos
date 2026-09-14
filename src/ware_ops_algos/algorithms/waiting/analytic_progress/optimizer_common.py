"""Shared helpers for the analytic-progress waiting optimizer.

Ported from project_4D4L/2_Stochastic_Waiting/Algorithms/analytic_progress/optimizer_common.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol, Sequence

from .models import (
    EXPECTED_INTERARRIVAL_TIME,
    BatchComparisonResult,
    BatchWindow,
    OrderInput,
    WarehouseInstance,
)


EXPECTED_INTERARRIVAL: float = EXPECTED_INTERARRIVAL_TIME
EngineName = Literal["discrete", "continuous"]


class OptimizerEngine(Protocol):
    def analyze_batch_window(
        self,
        window: BatchWindow,
        article_mapping: dict[str, tuple[int, int]],
        *,
        M: int,
        N_L: int,
        w: float,
        L: float,
        v: float,
        t_p: float,
    ) -> BatchComparisonResult: ...

    def analyze_batches(
        self,
        order_json_path: str,
        mapping_json_path: str,
        *,
        M: int,
        N_L: int,
        w: float,
        L: float,
        v: float,
        t_p: float,
        known_size: int = 3,
        total_size: int = 4,
        max_batches: int | None = None,
    ) -> list[BatchComparisonResult]: ...


def load_orders(order_json_path: str) -> list[OrderInput]:
    path = Path(order_json_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_orders = payload.get("orders", [])
    orders: list[OrderInput] = []
    for raw_order in raw_orders:
        orders.append(
            OrderInput(
                order_id=int(raw_order["order_id"]),
                items={str(article_id): int(quantity) for article_id, quantity in raw_order.get("items", {}).items()},
                arrival_time=float(raw_order["arrival_time"]),
                due_date=float(raw_order["due_date"]) if raw_order.get("due_date") is not None else None,
            )
        )

    orders.sort(key=lambda order: (order.arrival_time, order.order_id))
    return orders


def load_article_mapping(mapping_json_path: str) -> dict[str, tuple[int, int]]:
    path = Path(mapping_json_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return {
        str(article_id): (int(position[0]), int(position[1]))
        for article_id, position in payload.items()
    }


def build_batch_windows(
    orders: Sequence[OrderInput],
    known_size: int = 3,
    total_size: int = 4,
) -> list[BatchWindow]:
    if known_size < 1:
        raise ValueError("known_size must be >= 1")
    if total_size <= known_size:
        raise ValueError("total_size must be > known_size")

    windows: list[BatchWindow] = []
    for start_index in range(0, len(orders), total_size):
        current = list(orders[start_index:start_index + total_size])
        if len(current) < total_size:
            break
        windows.append(
            BatchWindow(
                batch_index=len(windows) + 1,
                base_orders=current[:known_size],
                insert_order=current[known_size],
            )
        )
    return windows


def create_warehouse_instance_from_orders(
    orders: Sequence[OrderInput],
    article_mapping: dict[str, tuple[int, int]],
    *,
    M: int,
    N_L: int,
    w: float,
    L: float,
    v: float,
    t_p: float,
) -> WarehouseInstance:
    aisle_pick_counts: dict[int, int] = {}
    pick_positions: list[tuple[int, int]] = []
    arrival_times = [float(order.arrival_time) for order in orders]

    for order in orders:
        for article_id, quantity in order.items.items():
            if article_id not in article_mapping:
                raise KeyError(f"Artikel-ID {article_id} fehlt im Mapping {len(article_mapping)}")
            if quantity < 0:
                raise ValueError(f"Negative Item-Menge fuer Artikel {article_id}: {quantity}")

            aisle, y = article_mapping[article_id]
            aisle_pick_counts[aisle] = aisle_pick_counts.get(aisle, 0) + quantity
            pick_positions.extend([(aisle, y)] * quantity)

    if not aisle_pick_counts:
        raise ValueError("Aus den Orders konnte keine Pick-Instanz aufgebaut werden")

    sorted_aisles = sorted(aisle_pick_counts)
    n_list = [aisle_pick_counts[aisle] for aisle in sorted_aisles]

    return WarehouseInstance(
        M=M,
        N_L=N_L,
        w=w,
        L=L,
        A=sorted_aisles,
        n_list=n_list,
        v=v,
        t_p=t_p,
        P=pick_positions,
        arrival_times=arrival_times,
    )


def expanded_pick_nodes_for_order(
    order: OrderInput,
    article_mapping: dict[str, tuple[int, int]],
) -> list[tuple[int, int]]:
    pick_nodes: list[tuple[int, int]] = []
    for article_id, quantity in order.items.items():
        if article_id not in article_mapping:
            raise KeyError(f"Artikel-ID {article_id} fehlt im Mapping {len(article_mapping)}")
        if quantity < 0:
            raise ValueError(f"Negative Item-Menge fuer Artikel {article_id}: {quantity}")
        pick_nodes.extend([article_mapping[article_id]] * quantity)
    return pick_nodes


def order_pick_service_time(order: OrderInput, t_p: float) -> float:
    total_items = sum(int(qty) for qty in order.items.values())
    return float(total_items) * float(t_p)
