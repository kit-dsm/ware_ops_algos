from typing import List

from algorithms_timeWindow.algorithm_timeWindow import (
    TimeWindowAlgorithm,
    Warehouse,
    RoutingAlgorithm,
    Batch,
)

class TimeWindowHenn(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, parameter: float) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.parameter = parameter

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement Henn algorithm
        return False


class TimeWindowZhang(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, mode: int) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.mode = mode

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement Zhang algorithm
        return False


class TimeWindowFixArrivePicker(TimeWindowAlgorithm):
    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement fixed picker arrival logic
        return False


class TimeWindowFixBatches(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, num_batches: int) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.num_batches = num_batches

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement fixed number of batches logic
        return False


class TimeWindowFixOrders(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, orders_per_batch: int) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.orders_per_batch = orders_per_batch

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement fixed number of orders per batch logic
        return False


class TimeWindowFixTime(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, time_parameter: int) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.time_parameter = time_parameter

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement fixed time window logic
        return False


class TimeWindowRandom(TimeWindowAlgorithm):
    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement random time window logic
        return False


class TimeWindowProbabilityStart(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, threshold: float) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.threshold = threshold

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement probability-based start logic
        return False


class TimeWindowProbabilityStart2(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, threshold: float) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.threshold = threshold

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement second probability-based start logic
        return False


class TimeWindowProbabilityStartNew(TimeWindowAlgorithm):
    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement new probability-based start logic
        return False


class TimeWindowProbabilityStartPeso(TimeWindowAlgorithm):
    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement weight-based probability start
        return False


class TimeWindowProbabilityStart2Peso(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, threshold: float) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.threshold = threshold

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement second weight-based probability start
        return False


class TimeWindowProbabilityHistogramPeso(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, parameter: float) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.parameter = parameter

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement histogram-based logic using weight
        return False


class TimeWindowProbabilityHistogramGlobal(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, parameter: float) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.parameter = parameter

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement histogram-based logic using global metrics
        return False


class TimeWindowProbabilityStartPesoXTiempo(TimeWindowAlgorithm):
    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement probability start using weight * time
        return False


class TimeWindowProbabilityStartPesoXMejoras(TimeWindowAlgorithm):
    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement probability start using weight * improvements
        return False


class TimeWindowProbabilityStartTiempo(TimeWindowAlgorithm):
    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement probability start using time
        return False


class TimeWindowProbabilityStart2Tiempo(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, threshold: float, mode: int) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.threshold = threshold
        self.mode = mode

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement second time-based probability start
        return False


class TimeWindowProbabilityStartTiempoXMejoras(TimeWindowAlgorithm):
    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement probability start using time * improvements
        return False


class TimeWindowProbabilityStartMejoras(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, threshold: float, mode: int) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.threshold = threshold
        self.mode = mode

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement probability start using improvements
        return False


class TimeWindowProbabilityStartMejorasReset(TimeWindowAlgorithm):
    def __init__(self, warehouse: Warehouse, routing_algorithm: RoutingAlgorithm, threshold: float) -> None:
        super().__init__(warehouse, routing_algorithm)
        self.threshold = threshold

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        # TODO: implement improvements-based start with reset behavior
        return False


# ---------------------------------------------------------------------------
# Selector function (Java: selector_alg_timeWindow_string)
# ---------------------------------------------------------------------------

def select_time_window_algorithm(
    alg_time_window: str,
    warehouse: Warehouse,
    routing_algorithm: RoutingAlgorithm
) -> TimeWindowAlgorithm:
    """
    Select and instantiate a specific time window algorithm based on a string identifier.

    Python equivalent of the Java method:
    `private algoritmos_timeWindow selector_alg_timeWindow_string(String alg_timeWindow, Warehouse w, algoritmos alg_routing)`.

    :param alg_time_window: Name of the time window algorithm, e.g.:
                            "henn_0.0", "zhang_1", "fix_orders_P8", "random", ...
    :param warehouse: Warehouse context.
    :param routing_algorithm: Routing algorithm used by the time window algorithm.
    :return: An instance of a concrete TimeWindowAlgorithm subclass.
    """

    threshold = 0.3  # General threshold used by some probabilistic algorithms

    # Henn variants
    if alg_time_window == "henn_0.0":
        return TimeWindowHenn(warehouse, routing_algorithm, 0.0)
    elif alg_time_window == "henn_0.25":
        return TimeWindowHenn(warehouse, routing_algorithm, 0.25)
    elif alg_time_window == "henn_0.5":
        return TimeWindowHenn(warehouse, routing_algorithm, 0.5)
    elif alg_time_window == "henn_0.75":
        return TimeWindowHenn(warehouse, routing_algorithm, 0.75)
    elif alg_time_window == "henn_1":
        return TimeWindowHenn(warehouse, routing_algorithm, 1.0)

    # Zhang variants
    elif alg_time_window == "zhang_1":
        return TimeWindowZhang(warehouse, routing_algorithm, 1)
    elif alg_time_window == "zhang_2":
        return TimeWindowZhang(warehouse, routing_algorithm, 2)
    elif alg_time_window == "zhang_3":
        return TimeWindowZhang(warehouse, routing_algorithm, 3)

    # Fixed picker arrival
    elif alg_time_window == "fix_arrivePicker":
        return TimeWindowFixArrivePicker(warehouse, routing_algorithm)

    # Fixed number of batches
    elif alg_time_window == "fix_batches_1":
        return TimeWindowFixBatches(warehouse, routing_algorithm, 1)
    elif alg_time_window == "fix_batches_2":
        return TimeWindowFixBatches(warehouse, routing_algorithm, 2)
    elif alg_time_window == "fix_batches_3":
        return TimeWindowFixBatches(warehouse, routing_algorithm, 3)
    elif alg_time_window == "fix_batches_4":
        return TimeWindowFixBatches(warehouse, routing_algorithm, 4)
    elif alg_time_window == "fix_batches_5":
        return TimeWindowFixBatches(warehouse, routing_algorithm, 5)
    elif alg_time_window == "fix_batches_6":
        return TimeWindowFixBatches(warehouse, routing_algorithm, 6)

    # Fixed orders per batch
    elif alg_time_window == "fix_orders_P4":
        return TimeWindowFixOrders(warehouse, routing_algorithm, 4)
    elif alg_time_window == "fix_orders_P8":
        return TimeWindowFixOrders(warehouse, routing_algorithm, 8)
    elif alg_time_window == "fix_orders_P16":
        return TimeWindowFixOrders(warehouse, routing_algorithm, 16)
    elif alg_time_window == "fix_orders_P5":
        return TimeWindowFixOrders(warehouse, routing_algorithm, 5)
    elif alg_time_window == "fix_orders_P15":
        return TimeWindowFixOrders(warehouse, routing_algorithm, 15)
    elif alg_time_window == "fix_orders_P25":
        return TimeWindowFixOrders(warehouse, routing_algorithm, 25)

    # Fixed time window variants
    elif alg_time_window == "fix_time_M3":
        return TimeWindowFixTime(warehouse, routing_algorithm, 3)
    elif alg_time_window == "fix_time_M6":
        return TimeWindowFixTime(warehouse, routing_algorithm, 6)
    elif alg_time_window == "fix_time_M12":
        return TimeWindowFixTime(warehouse, routing_algorithm, 12)
    elif alg_time_window == "fix_time_M10":
        return TimeWindowFixTime(warehouse, routing_algorithm, 10)
    elif alg_time_window == "fix_time_M20":
        return TimeWindowFixTime(warehouse, routing_algorithm, 20)
    elif alg_time_window == "fix_time_M30":
        return TimeWindowFixTime(warehouse, routing_algorithm, 30)

    # Random algorithm
    elif alg_time_window == "random":
        return TimeWindowRandom(warehouse, routing_algorithm)

    # Probability-based start variants
    elif alg_time_window == "probability_start":
        return TimeWindowProbabilityStart(warehouse, routing_algorithm, threshold)
    elif alg_time_window == "probability_start2":
        return TimeWindowProbabilityStart2(warehouse, routing_algorithm, threshold)
    elif alg_time_window == "probability_start_new":
        return TimeWindowProbabilityStartNew(warehouse, routing_algorithm)
    elif alg_time_window == "probability_start_peso":
        return TimeWindowProbabilityStartPeso(warehouse, routing_algorithm)
    elif alg_time_window == "probability_start2_peso":
        return TimeWindowProbabilityStart2Peso(warehouse, routing_algorithm, threshold)

    # Histogram-based (weight) algorithms
    elif alg_time_window == "probability_histograma_peso_P20":
        return TimeWindowProbabilityHistogramPeso(warehouse, routing_algorithm, 0.2)
    elif alg_time_window == "probability_histograma_peso_P30":
        return TimeWindowProbabilityHistogramPeso(warehouse, routing_algorithm, 0.3)
    elif alg_time_window == "probability_histograma_peso_P40":
        return TimeWindowProbabilityHistogramPeso(warehouse, routing_algorithm, 0.4)
    elif alg_time_window == "probability_histograma_peso_P50":
        return TimeWindowProbabilityHistogramPeso(warehouse, routing_algorithm, 0.5)
    elif alg_time_window == "probability_histograma_peso_P60":
        return TimeWindowProbabilityHistogramPeso(warehouse, routing_algorithm, 0.6)
    elif alg_time_window == "probability_histograma_peso_P70":
        return TimeWindowProbabilityHistogramPeso(warehouse, routing_algorithm, 0.7)
    elif alg_time_window == "probability_histograma_peso_P80":
        return TimeWindowProbabilityHistogramPeso(warehouse, routing_algorithm, 0.8)

    # Histogram-based (global) algorithms
    elif alg_time_window == "probability_histograma_global_P10":
        return TimeWindowProbabilityHistogramGlobal(warehouse, routing_algorithm, 0.1)
    elif alg_time_window == "probability_histograma_global_P20":
        return TimeWindowProbabilityHistogramGlobal(warehouse, routing_algorithm, 0.2)
    elif alg_time_window == "probability_histograma_global_P30":
        return TimeWindowProbabilityHistogramGlobal(warehouse, routing_algorithm, 0.3)
    elif alg_time_window == "probability_histograma_global_P40":
        return TimeWindowProbabilityHistogramGlobal(warehouse, routing_algorithm, 0.4)
    elif alg_time_window == "probability_histograma_global_P50":
        return TimeWindowProbabilityHistogramGlobal(warehouse, routing_algorithm, 0.5)
    elif alg_time_window == "probability_histograma_global_P60":
        return TimeWindowProbabilityHistogramGlobal(warehouse, routing_algorithm, 0.6)
    elif alg_time_window == "probability_histograma_global_P70":
        return TimeWindowProbabilityHistogramGlobal(warehouse, routing_algorithm, 0.7)
    elif alg_time_window == "probability_histograma_global_P80":
        return TimeWindowProbabilityHistogramGlobal(warehouse, routing_algorithm, 0.8)
    elif alg_time_window == "probability_histograma_global_P90":
        return TimeWindowProbabilityHistogramGlobal(warehouse, routing_algorithm, 0.9)

    # Probability-based combinations of weight, time, improvements
    elif alg_time_window == "probability_start_pesoXtiempo":
        return TimeWindowProbabilityStartPesoXTiempo(warehouse, routing_algorithm)
    elif alg_time_window == "probability_start_pesoXmejoras":
        return TimeWindowProbabilityStartPesoXMejoras(warehouse, routing_algorithm)
    elif alg_time_window == "probability_start_tiempo":
        return TimeWindowProbabilityStartTiempo(warehouse, routing_algorithm)
    elif alg_time_window == "probability_start2_tiempo_B1":
        return TimeWindowProbabilityStart2Tiempo(warehouse, routing_algorithm, threshold, 1)
    elif alg_time_window == "probability_start_tiempoXmejoras":
        return TimeWindowProbabilityStartTiempoXMejoras(warehouse, routing_algorithm)
    elif alg_time_window == "probability_start_mejoras_B1":
        return TimeWindowProbabilityStartMejoras(warehouse, routing_algorithm, threshold, 1)
    elif alg_time_window == "probability_start_mejoras_reset":
        return TimeWindowProbabilityStartMejorasReset(warehouse, routing_algorithm, threshold)

    # Default: fall back to fixed picker arrival algorithm
    else:
        return TimeWindowFixArrivePicker(warehouse, routing_algorithm)