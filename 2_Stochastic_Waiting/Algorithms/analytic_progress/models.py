from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Optimization Constants
# ============================================================================
EXPECTED_INTERARRIVAL_TIME = 28.8  # seconds - time between order arrivals
TIME_TOLERANCE = 1e-9              # tolerance for floating-point comparisons
POSITION_TOLERANCE = 1e-6          # tolerance for position matching
INTEGRATION_THRESHOLD = -1.0       # sentinel value indicating out-of-phase condition
NUMERIC_NAN_SENTINEL = float("nan")
NUMERIC_INF_SENTINEL = float("inf")

# ============================================================================
# Data Models
# ============================================================================

@dataclass(frozen=True)
class OrderInput:
    order_id: int
    items: Dict[str, int]
    arrival_time: float
    due_date: Optional[float] = None


@dataclass(frozen=True)
class BatchWindow:
    batch_index: int
    base_orders: List[OrderInput]
    insert_order: OrderInput

    @property
    def all_orders(self) -> List[OrderInput]:
        return [*self.base_orders, self.insert_order]


@dataclass(frozen=True)
class TimeSampleResult:
    batch_index: int
    idx: int
    t: float
    x: float
    y: float
    phase: str
    return_flag: str
    expected_detour: float
    base_route_duration: float
    predicted_route_duration: float
    waiting_time_to_estimated: float  # Wartezeit auf Basis der geschätzten Ankunft (last_known + 28.8), NICHT t_actual
    predicted_add_time: float
    arrival_time_order_3: float
    predicted_completion_time: float
    mean_predicted_order_completion_time: float
    initial_completion_time: float
    mean_initial_order_completion_time: float
    effective_detour: float
    effective_route_duration: float
    effective_completion_time: float
    effective_policy: str
    detour_route_back: float
    detour_apass: float
    detour_aup: float
    p1_route: float
    p2_backtrack: float
    p3_pass: float
    p4_front: float
    p_sum: float
    v_nodes: float
    r_nodes: float
    g_pass_count: int
    g_front_count: int
    predicted_completion_time_per_order: float
    initial_completion_time_per_order: float
    should_integrate: bool


@dataclass(frozen=True)
class BatchComparisonResult:
    batch_index: int
    base_order_ids: List[int]
    insert_order_id: int
    base_order_arrival_times: List[float]
    insert_order_arrival_time: float
    base_order_positions: List[List[Tuple[int, int]]]
    insert_order_positions: List[Tuple[int, int]]
    base_route_duration: float
    phase1_should_integrate: bool
    phase1_best_integration_t_idx: int
    phase1_best_integration_t: float
    actual_integration_t: float
    actual_waiting_time: float
    mean_initial_completion_time: float
    actual_x: float
    actual_y: float
    actual_case: str
    actual_route_duration: float
    actual_detour: float
    mean_actual_order_completion_time: float
    avg_arrival_time: float
    phase1_best_t: int
    phase1_best_x: float
    phase1_best_y: float
    phase1_predicted_best_detour: float
    phase1_predicted_route_duration: float
    phase1_predicted_completion_time: float
    actual_completion_time: float
    samples: List[TimeSampleResult]
    actual_t_estimated_detour: float
    actual_t_mean_predicted_OCT: float
    estimated_t_insert_idx: Optional[int]


@dataclass
class WarehouseInstance:
    M: int
    N_L: int
    w: float
    L: float
    A: List[int]
    n_list: List[int]
    v: float
    t_p: float
    P: List[Tuple[int, int]]
    arrival_times: List[float]

    def __post_init__(self) -> None:
        if self.M < 1:
            raise ValueError("M must be >= 1")
        if self.N_L < 1:
            raise ValueError("N_L must be >= 1")
        if self.L <= 0:
            raise ValueError("L must be > 0")
        if self.v <= 0:
            raise ValueError("v must be > 0")
        if self.w < 0:
            raise ValueError("w must be >= 0")
        if self.t_p < 0:
            raise ValueError("t_p must be >= 0")
        if len(self.A) == 0:
            raise ValueError("A must not be empty")
        if len(self.A) != len(self.n_list):
            raise ValueError("Length of A and n_list must match")
        if any(self.A[i] >= self.A[i + 1] for i in range(len(self.A) - 1)):
            raise ValueError("A must be strictly increasing")
        if any(a < 1 or a > self.M for a in self.A):
            raise ValueError("All aisle indices in A must be in [1, M]")
        if any(n < 0 for n in self.n_list):
            raise ValueError("n_list must contain non-negative values")

        for aisle, y in self.P:
            if aisle < 1 or aisle > self.M:
                raise ValueError("Pick aisle must be in [1, M]")
            if y < 1 or y > self.N_L:
                raise ValueError("Pick position y must be in [1, N_L]")

        self.k = len(self.A)
        self.n_ges = sum(self.n_list)
        self.aisle_to_j: Dict[int, int] = {a: j for j, a in enumerate(self.A, start=1)}

    def __repr__(self) -> str:
        """Kurze Darstellung für Debugging."""
        return f"WarehouseInstance(M={self.M}, k={self.k}, n_ges={self.n_ges})"

