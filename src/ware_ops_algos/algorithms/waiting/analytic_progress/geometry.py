"""Read the paper's symbols directly from the canonical waiting input.

The layout, picker, and resolved batch orders remain the source of truth.
These functions calculate values needed by the analytical formulas without
creating another warehouse instance or copying arrival times into one.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...algorithm_interfaces import WaitingInput


def orders(data: WaitingInput):
    return data.candidates[0].job.route.batch.orders


def M(data: WaitingInput) -> int:
    return int(data.layout.graph_data.n_aisles)


def N_L(data: WaitingInput) -> int:
    return int(data.layout.graph_data.n_pick_locations)


def w(data: WaitingInput) -> float:
    return float(data.layout.graph_data.dist_aisle)


def L(data: WaitingInput) -> float:
    params = data.layout.graph_data
    return float(
        params.dist_bottom_to_pick_location
        + (params.n_pick_locations - 1) * params.dist_pick_locations
        + params.dist_top_to_pick_location
    )


def v(data: WaitingInput) -> float:
    return float(data.picker.speed)


def t_p(data: WaitingInput) -> float:
    return float(data.picker.time_per_pick)


def P(data: WaitingInput) -> list[tuple[int, int]]:
    return [
        (int(pick.pick_node[0]), int(pick.pick_node[1]))
        for order in orders(data)
        for pick in order.pick_positions
        for _ in range(int(pick.in_store or pick.amount))
    ]


def A(data: WaitingInput) -> list[int]:
    return sorted({aisle for aisle, _ in P(data)})


def n_list(data: WaitingInput) -> list[int]:
    counts = Counter(aisle for aisle, _ in P(data))
    return [counts[aisle] for aisle in A(data)]


def k(data: WaitingInput) -> int:
    return len(A(data))


def n_ges(data: WaitingInput) -> int:
    return len(P(data))


def aisle_to_j(data: WaitingInput) -> dict[int, int]:
    return {aisle: index for index, aisle in enumerate(A(data), start=1)}


def arrival_times(data: WaitingInput) -> list[float]:
    return [float(order.order_date) for order in orders(data)]
