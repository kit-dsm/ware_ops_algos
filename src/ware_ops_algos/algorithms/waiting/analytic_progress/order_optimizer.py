"""Engine registry for the analytic-progress waiting optimizer.

Ported from project_4D4L/2_Stochastic_Waiting/Algorithms/analytic_progress/order_optimizer.py.
"""

from __future__ import annotations

from typing import Literal

from .models import BatchComparisonResult, BatchWindow
from .optimizer_common import EngineName
from .order_optimizer_discrete import (
    analyze_batch_window as analyze_batch_window_discrete,
    analyze_batches as analyze_batches_discrete,
    batch_sample_rows,
    batch_summary_rows,
)
from .order_optimizer_continuous import (
    analyze_batch_window as analyze_batch_window_continuous,
    analyze_batches as analyze_batches_continuous,
)


ENGINE_REGISTRY: dict[str, dict] = {
    "discrete": {
        "analyze_batch_window": analyze_batch_window_discrete,
        "analyze_batches": analyze_batches_discrete,
    },
    "continuous": {
        "analyze_batch_window": analyze_batch_window_continuous,
        "analyze_batches": analyze_batches_continuous,
    },
}


def _normalize_engine_name(engine: str | None) -> EngineName:
    if engine is None:
        return "discrete"
    normalized = engine.lower().strip()
    if normalized not in ENGINE_REGISTRY:
        raise ValueError(f"Unbekannte Engine '{engine}'. Erlaubt: {', '.join(ENGINE_REGISTRY)}")
    return normalized  # type: ignore[return-value]


def analyze_batch_window(
    window: BatchWindow,
    article_mapping: dict[str, tuple[int, int]],
    *,
    M: int,
    N_L: int,
    w: float,
    L: float,
    v: float,
    t_p: float,
    engine: str = "discrete",
) -> BatchComparisonResult:
    selected = _normalize_engine_name(engine)
    fn = ENGINE_REGISTRY[selected]["analyze_batch_window"]
    return fn(window, article_mapping, M=M, N_L=N_L, w=w, L=L, v=v, t_p=t_p)


def analyze_batches(
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
    engine: str = "discrete",
) -> list[BatchComparisonResult]:
    selected = _normalize_engine_name(engine)
    fn = ENGINE_REGISTRY[selected]["analyze_batches"]
    return fn(
        order_json_path,
        mapping_json_path,
        M=M,
        N_L=N_L,
        w=w,
        L=L,
        v=v,
        t_p=t_p,
        known_size=known_size,
        total_size=total_size,
        max_batches=max_batches,
    )
