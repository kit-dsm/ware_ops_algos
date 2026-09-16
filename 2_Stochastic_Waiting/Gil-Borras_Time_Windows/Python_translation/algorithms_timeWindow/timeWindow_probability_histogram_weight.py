# original name: time_window_probability_histogram_peso.py

from typing import List

from algorithm_timeWindow import (
    TimeWindowAlgorithm,
    Warehouse,
    RoutingAlgorithm,
    Batch,
)

# Minimal placeholder for Order; in your real project, import the actual class.
class Order:
    """
    Minimal placeholder for Order.

    Expected interface:
    - getPeso() -> float : returns the weight of the order.

    In the original code, peso is used as a variable name. I assume that it means "weight" in English.
    However, our batches do not track weight.
    #TODO: Adapt this function to our capacity metrics
    """
    def __init__(self, weight: float) -> None:
        self._weight = weight

    def getWeight(self) -> float:
        return self._weight


class TimeWindowProbabilityHistogramWeight(TimeWindowAlgorithm):
    """
    Probabilistic time window algorithm based on a histogram of order weights.

    Python equivalent of the Java class:
        timeWindow_probability_histogram_peso extends algoritmos_timeWindow.

    Conceptual waiting strategy
    ---------------------------
    The algorithm uses historical order weight data to decide whether to
    release a batch or wait for additional orders:

    1. If more than one batch already exists in the system:
         -> release immediately (return True).

    2. Otherwise (0 or 1 existing batch):
         - Compute the remaining capacity in the candidate batch:
               peso_resto = batch_to_route.max_peso - batch_to_route.peso
         - If peso_resto is greater than the maximum observed order weight
           (max_peso):
               -> return False (wait), since the histogram has no data in this range.
         - Use the weight histogram to estimate:
               porcen = fraction of historical orders with weight <= peso_resto
         - If porcen < threshold:
               -> the probability of fitting future orders is low -> release (True)
           else:
               -> high probability of filling the batch -> wait (False).

    Inputs:
    - batches: List[Batch] (existing batches),
    - batch_to_route: Batch with attributes `max_peso` and `peso`.

    Output:
    - bool: True if release is allowed now, False otherwise.

    Edit MB: Original variable names changed to English. Weight is not present in our batches
    peso(s) --> weight
    muestra(s) --> sample(s)
    media --> mean_interarrival_time
    #TODO: Adapt the logic to our capacity metrics instead of weight
    """

    def __init__(
        self,
        warehouse: Warehouse,
        routing_algorithm: RoutingAlgorithm,
        threshold: float,
    ) -> None:
        """
        Initialize the histogram-based weight probability time window algorithm.

        :param warehouse: Warehouse context.
        :param routing_algorithm: Routing algorithm (not used directly here, but
                                  stored for consistency).
        :param threshold: Probability threshold in [0,1] controlling how low the
                          fraction of fitting orders must be to trigger release.
        """
        super().__init__(warehouse, routing_algorithm)

        self.threshold: float = threshold

        # Statistics based on historical orders
        self.avg_weight_n: float = 0.0   # average weight
        self.dv_weight_n: float = 0.0    # mean absolute deviation
        self.n_samples: int = 0        # number of sample orders
        self.max_weight: int = 0          # maximum rounded weight
        self.histogram_weights_n: List[int] = []  # histogram of weights

    def load_stats(self, orders: List[Order]) -> None:
        """
        Load weight statistics from a list of orders.

        Python equivalent of the Java method:
            load_stad(List<Order> LO)

        It computes:
        - avg_pesos_n: average order weight,
        - dv_pesos_n: mean absolute deviation from average,
        - n_muestras: number of orders,
        - max_peso: maximum rounded weight,
        - histograma_pesos_n: histogram of rounded weights from 1..max_peso.

        This method initializes the stats only once, if avg_pesos_n == 0.

        :param orders: List of Order instances with getPeso() method.
        """
        if self.avg_weight_n != 0 or not orders:
            # Already initialized or no data provided; do nothing
            return

        self.avg_weight_n = 0.0
        self.dv_weight_n = 0.0
        self.n_samples = len(orders)
        self.max_weight = 0

        # First pass: compute mean and maximum rounded weight
        for o in orders:
            weight = o.getWeight()
            self.avg_weight_n += weight
            rounded = int(round(weight))
            if self.max_weight < rounded:
                self.max_weight = rounded

        self.avg_weight_n /= len(orders)

        # Second pass: compute mean absolute deviation
        for o in orders:
            weight = o.getWeight()
            self.dv_weight_n += abs(self.avg_weight_n - weight)
        self.dv_weight_n /= len(orders)

        # Initialize histogram: indices 0..max_peso-1 => weights 1..max_peso
        self.histogram_weights_n = [0] * self.max_weight
        for o in orders:
            weight = o.getWeight()
            r = int(round(weight))
            idx = r - 1  # as in Java: Math.round(o.getPeso()) - 1
            if 0 <= idx < len(self.histogram_weights_n):
                self.histogram_weights_n[idx] += 1

    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        """
        Decide whether to release 'batch_to_route' now based on remaining capacity
        and the histogram of order weights.

        Python equivalent of the Java method:
            public boolean run(List<Batch> LB, Batch batchToRouting) throws Exception

        :param batches: Existing batches in the system (LB in Java).
        :param batch_to_route: Candidate batch to be released. Must provide:
                               - max_peso: maximum capacity in weight,
                               - peso: current total weight of items in the batch.
        :return: True if release is allowed, False otherwise.
        """
        # If we already have more than one batch, do not delay further.
        if len(batches) > 1:
            return True

        # Remaining capacity in the batch
        weight_rest = batch_to_route.max_weight - batch_to_route.weight

        # TODO: Adapt to our batch class

        # If remaining capacity exceeds known maximum order weight,
        # histogram cannot provide reliable probability -> wait.
        if weight_rest > self.max_weight:
            return False

        # Count orders that would fit (rounded weight <= peso_resto)
        occurrences = 0
        upper_index = int(round(weight_rest))
        for a in range(upper_index):
            if 0 <= a < len(self.histogram_weights_n):
                occurrences += self.histogram_weights_n[a]

        if self.n_samples <= 0:
            # No samples -> no statistics -> choose a safe behavior (wait)
            # or raise an error. Here we mimic the spirit of Java, which
            # assumes load_stad() has initialized the stats.
            # We choose to raise to signal misconfiguration.
            raise ValueError("n_samples is zero; call load_stats() before run().")

        percentage = occurrences / float(self.n_samples)

        # Release if the fraction of orders that would fit is below the threshold
        return percentage < self.threshold