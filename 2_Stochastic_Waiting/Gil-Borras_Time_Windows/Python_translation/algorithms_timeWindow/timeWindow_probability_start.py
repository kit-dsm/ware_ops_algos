from typing import List
import time as _time

from algorithm_timeWindow import (
    TimeWindowAlgorithm,
    Warehouse,
    RoutingAlgorithm,
    Batch,
)

# Minimal placeholder for Order, used in load_stats.
class Order:
    """
        Minimal placeholder for Order.

        Expected interface:
        - getPeso() -> float

        In the original code, peso is used as a variable name. I assume that it means "weight" in English.
        However, our batches do not track weight.
        #TODO: Adapt this function to our capacity metrics

        """
    def __init__(self, weight: float) -> None:
        self._weight = weight

    def getWeight(self) -> float:
        return self._weight


# Minimal placeholder for BatchCollected; replace with real implementation.
class BatchCollected:
    """
        Minimal placeholder for BatchCollected.

        Expected behavior in this context:
        - initialized with (batch, warehouse)
        - routing(routing_algorithm) -> float (routing time/distance)

        This class is necessary to compute routing time for a batch in the warehouse
        #TODO: Adapt it to our routing logic
        """
    def __init__(self, batch: Batch, warehouse: Warehouse) -> None:
        self.batch = batch
        self.warehouse = warehouse

    def routing(self, routing_algorithm: RoutingAlgorithm) -> float:
        """
        Compute routing time/cost for this batch.

        In the real application, this should call the routing algorithm to compute
        the actual travel distance/time. Here it's a stub.
        """
        # TODO: implement real routing logic
        return 0.0


class TimeWindowProbabilityStart(TimeWindowAlgorithm):
    """
    Probabilistic time-window algorithm based on remaining capacity,
    routing efficiency and elapsed time.

    Python equivalent of the Java class:
        timeWindow_probability_start extends algoritmos_timeWindow.

    Conceptual behavior:
    --------------------
    For each candidate batch, compute a 'score' (porcen in [0, 1]) that reflects:
      - how much capacity remains vs. typical order weights,
      - whether routing efficiency is improving over time,
      - how much time has passed since the last release.

    Then compare this score to a threshold:

      - If score > threshold  -> do NOT release (wait).
      - If score <= threshold -> release.

    The score is updated multiplicatively by several factors, each in [0,1].

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
        Initialize the algorithm.

        :param warehouse: Warehouse context.
        :param routing_algorithm: Routing algorithm used for routing evaluation.
        :param threshold: Probability-like threshold in [0,1]. The smaller this
                          value, the more aggressive the algorithm is in releasing
                          batches (since porcen must fall below this value).
        """
        super().__init__(warehouse, routing_algorithm)

        self.threshold: float = threshold

        # Improvement tracking
        self.avg_improvement: float = 0.0  # number of times tasa <= tasa_old
        self.n_improvement: int = 0        # number of comparisons made

        # Time of last accepted release
        self.time_left: int = int(_time.time() * 1000)

        # Weight statistics
        self.avg_weight_n: float = 0.0
        self.dv_weight_n: float = 0.0

        # Average routing time of accepted batches
        self.avg_routing: float = 0.0

        # Previous efficiency measure (routing/peso)
        self.rate_old: float = float("inf")

    # ----------------------------------------------------------------------
    # Statistics loader (equivalent to load_stad in Java)
    # ----------------------------------------------------------------------
    def load_stats(self, orders: List[Order]) -> None:
        """
        Load weight statistics from a list of orders.

        Python equivalent of the Java method:
            load_stad(List<Order> LO)

        Computes:
        - avg_pesos_n : average order weight,
        - dv_pesos_n  : mean absolute deviation from avg_pesos_n.

        :param orders: List of Order objects with getPeso().
        """
        if not orders:
            self.avg_weight_n = 0.0
            self.dv_weight_n = 0.0
            return

        self.avg_weight_n = 0.0
        self.dv_weight_n = 0.0

        # Compute average weight
        for o in orders:
            self.avg_weight_n += o.getWeight()
        self.avg_weight_n /= len(orders)

        # Compute mean absolute deviation
        for o in orders:
            self.dv_weight_n += abs(self.avg_weight_n - o.getWeight())
        self.dv_weight_n /= len(orders)

    # ----------------------------------------------------------------------
    # Core decision logic (equivalent to run in Java)
    # ----------------------------------------------------------------------
    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        """
        Decide whether to release 'batch_to_route' now or wait.

        Python equivalent of:
            public boolean run(List<Batch> LB, Batch batchToRouting) throws Exception

        Strategy:
        ---------
        1. Start with porcen = 1.0.
        2. Modify porcen based on remaining capacity vs. weight statistics.
        3. Modify porcen based on routing efficiency improvement.
        4. Modify porcen based on elapsed time relative to avg routing and current routing.
        5. If porcen > threshold -> wait (return False).
           Otherwise            -> release (return True).
        6. If we release, update avg_routing, time_left, and reset efficiency history
           if necessary. If we wait, update tasa_old for the next comparison.

        :param batches: List of existing batches (unused in this algorithm).
        :param batch_to_route: Batch to be considered for release.
                               Must provide:
                                - max_peso (maximum capacity),
                                - peso     (current weight).
        :return: True if release is allowed, False otherwise.
        """
        res = True
        percentage = 1.0

        # 1) Capacity-based factor
        weight_rest = batch_to_route.max_weight - batch_to_route.weight

        # TODO: Adapt to our batch class

        # Avoid division by zero if statistics not initialized properly
        if self.avg_weight_n <= 0:
            # If no statistics, we cannot classify capacity meaningfully.
            # Choose neutral behavior (porcen unchanged) or raise.
            # Here we leave porcen as 1.0.
            n_mean = 0.0
        else:
            n_mean = batch_to_route.max_weight / self.avg_weight_n

        expr = self.avg_weight_n + self.dv_weight_n - weight_rest

        if expr < 0:
            percentage *= 1.0
        elif n_mean > 0 and expr < n_mean:
            percentage *= 0.8
        elif n_mean > 0 and expr < 2 * n_mean:
            percentage *= 0.6
        else:
            percentage *= 0.4

        # 2) Routing efficiency factor
        bc = BatchCollected(batch_to_route, self.warehouse)
        routing = bc.routing(self.routing_algorithm)

        # Avoid division by zero on zero-weight batch
        if batch_to_route.weight > 0:
            rate = routing / batch_to_route.weight
        else:
            rate = float("inf")

        if self.rate_old < float("inf"):
            self.n_improvement += 1
            if rate <= self.rate_old:
                self.avg_improvement += 1
            # Multiply porcen by the fraction of improvements
            if self.n_improvement > 0:
                percentage *= (self.avg_improvement / self.n_improvement)

        # 3) Time-based factors
        now_ms = int(_time.time() * 1000)
        if self.time_left + self.avg_routing + routing <= now_ms:
            percentage = 0.0
        elif self.time_left + self.avg_routing <= now_ms:
            percentage *= 0.5
        elif self.time_left + routing <= now_ms:
            percentage *= 0.5
        else:
            # within time window -> no additional penalty
            percentage*= 1.0

        # 4) Final decision: 1 -> no release, 0 -> release (in comments)
        # Here: porcen > threshold -> wait (res = False)
        if percentage > self.threshold:
            res = False

        # 5) Update state depending on decision
        if res:
            # We release: update average routing, reset tasa_old, update time_left
            self.avg_routing = (self.avg_routing + routing) / 2.0
            self.rate_old = float("inf")
            self.time_left = now_ms

            # If improvements are too rare, reset improvement counters
            if self.avg_improvement * 3 < self.n_improvement:
                self.avg_improvement = 1.0
                self.n_improvement = 1
        else:
            # We wait: remember current efficiency to compare next time
            self.rate_old = rate

        return res