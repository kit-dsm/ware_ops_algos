from typing import List
import sys

from algorithm_timeWindow import (
    TimeWindowAlgorithm,
    Warehouse,
    RoutingAlgorithm,
    Batch,
)

# Stub/expected interface for Order and BatchCollected
# In your real project, import these from your actual modules.

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
        Compute routing cost/time for this batch.

        In the real application, this method must:
        - compute a route for 'self.batch' in 'self.warehouse'
        - return a time or distance metric.

        Here we return 0.0 as a stub.
        """
        # TODO: Replace stub with real routing logic
        return 0.0


class TimeWindowProbabilityHistogramGlobal(TimeWindowAlgorithm):
    """
    Probabilistic time window algorithm based on a global histogram of order weights
    and an arrival rate model.

    Python equivalent of the Java class:
        timeWindow_probability_histogram_global extends algoritmos_timeWindow.

    Conceptual behavior (summary):
    ------------------------------
    Uses global statistics about order weights and arrival rate (lambda) to decide
    whether to release a batch now or wait.

    For each candidate batch:
    1. If there are already more than one batch in the system:
         -> always release (return True).
    2. Else:
         * Compute mean inter-arrival time: media = round(1 / lambda).
         * Compute routing time t_routing for the candidate batch.
         * If t_routing < media:
              -> release (True).
           else:
              -> Evaluate remaining capacity (peso_resto) and histogram of weights to
                 estimate how likely it is that future orders fit into this batch:
                 - porcen = fraction of orders with weight <= peso_resto.
                 - If porcen < threshold:
                      -> release (True).
                    else:
                      -> Use an adaptive efficiency measure (t_routing / weight) relative
                         to past cases to decide whether to release or wait.

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
        Initialize the histogram-based global time window algorithm.

        :param warehouse: Warehouse context.
        :param routing_algorithm: Routing algorithm used for route evaluation.
        :param threshold: Probability threshold in [0, 1]. Controls how strict
                          the histogram-based decision is.
        """
        super().__init__(warehouse, routing_algorithm)

        self.threshold: float = threshold

        # Global statistical parameters
        self.avg_weight_n: float = 0.0  # average order weight
        self.dv_weight_n: float = 0.0   # potential dispersion measure (unused)
        self.lambda_: float = 0.0      # arrival rate (orders per time unit)
        self.n_samples: int = 0       # number of sample orders
        self.max_weight: int = 0         # maximum order weight (rounded)

        # Histogram of order weights:
        # histograma_weight_n[w] = number of orders with weight ~ (w + 1)
        self.histogram_weights_n: List[int] = []

        # Adaptive efficiency tracking
        self.counterImprovement: int = 0  # count of cases reaching the final step
        self.efficiency: float = 0.0     # accumulated efficiency (sum of t_routing / weight)

    # ------------------------------------------------------------------
    # Statistical loading (equivalent to load_stad in Java)
    # ------------------------------------------------------------------
    def load_stats(self, orders: List[Order], execution_time: float) -> None:
        """
        Load global statistics (weights histogram and arrival rate).

        Python equivalent of the Java method:
            load_stad(List<Order> LO, long tiempo_ejecucion)

        :param orders: List of orders used as statistical sample.
        :param execution_time: Observation/execution time used to compute
                               the arrival rate (lambda = len(orders) / execution_time).
        """
        if execution_time <= 0:
            # Avoid division by zero, keep lambda_ at 0 which will cause issues
            # if run() is called without valid lambda; caller must ensure correctness.
            self.lambda_ = 0.0
        else:
            self.lambda_ = len(orders) / execution_time

        # Initialize statistics only once (as in Java: if avg_weight_n == 0)
        if self.avg_weight_n == 0 and len(orders) > 0:
            self.avg_weight_n = 0.0
            self.dv_weight_n = 0.0
            self.n_samples = len(orders)
            self.max_weight = 0

            # Compute average weight and maximum rounded weight
            for o in orders:
                weight = o.getWeight()
                self.avg_weight_n += weight
                rounded = int(round(weight))
                if self.max_weight < rounded:
                    self.max_weight = rounded

            self.avg_weight_n /= len(orders)

            # Initialize histogram from 0 to capacity + 1 (inclusive)
            capacity = int(round(self.warehouse.getCapacity()))
            #TODO: Adapt this to our capacity metrics
            self.histogram_weight_n = [0] * (capacity + 1)

            # Fill histogram
            for o in orders:
                weight = o.getWeight()
                r = int(round(weight))
                # This matches Java: histograma_weight_n[Math.round(o.getPeso()) - 1]++
                idx = r - 1  # as in Java: Math.round(o.getPeso()) - 1
                if 0 <= idx < len(self.histogram_weights_n):
                    self.histogram_weights_n[idx] += 1

    # ------------------------------------------------------------------
    # Core decision rule (equivalent to run in Java)
    # ------------------------------------------------------------------
    def run(self, batches: List[Batch], batch_to_route: Batch) -> bool:
        """
        Decide whether to release 'batch_to_route' now, based on global histogram
        and arrival rate statistics.

        Python equivalent of:
            public boolean run(List<Batch> LB, Batch batchToRouting) throws Exception

        Strategy:
        --------
        1. If len(batches) > 1:
             -> return True (immediately allow release).
        2. Compute media = round(1 / lambda_). (mean inter-arrival time)
           If lambda_ <= 0, this may cause division error; ensure load_stats()
           was called with valid data.
        3. Compute t_routing = routing time for batch_to_route.
           If t_routing < media:
             -> return True (release, since route is short).
        4. Compute residual capacity:
             peso_resto = batch_to_route.max_peso - batch_to_route.peso
           Then estimate the fraction 'porcen' of historical orders whose weight
           is <= peso_resto using histograma_pesos_n and n_muestras.
           If porcen < threshold:
             -> return True.
        5. Otherwise:
           - Update efficiency stats:
             effic = t_routing / batch_to_route.peso
             eficacia += effic
             contadorMejoras += 1
             eficaciaMedia = eficacia / contadorMejoras
           - Return:
             not (effic < eficaciaMedia * 1.1)
           i.e., True if effic >= 1.1 * eficaciaMedia, else False.

        :param batches: Existing batches in the system (LB in Java).
        :param batch_to_route: Batch considered for routing/release.
                               Must provide 'max_peso' and 'peso' attributes.
        :return: True if release is allowed, False otherwise.
        """
        # 1. If more than one existing batch: always release
        if len(batches) > 1:
            return True

        # 2. Compute mean inter-arrival time based on lambda_
        if self.lambda_ <= 0:
            # Without a valid lambda_, the model cannot compute 'media'.
            # Java code would throw if lambda was zero; here we can choose
            # a conservative behavior (e.g., do not release) or a default.
            # We'll follow the Java spirit: division by zero would be an error.
            raise ZeroDivisionError(
                "lambda_ is zero or negative; call load_stats with valid data before run()."
            )

        mean_interarrival_time = round(1.0 / self.lambda_)

        # 3. Compute routing time for the candidate batch
        b_collected = BatchCollected(batch_to_route, self.warehouse)
        t_routing = b_collected.routing(self.routing_algorithm)

        if t_routing < mean_interarrival_time:
            # Route is short compared to mean inter-arrival time -> release
            return True

        # 4. Compute remaining capacity and use histogram to estimate probability
        weight_rest = batch_to_route.max_weight - batch_to_route.weight

        #TODO: Adapt to our batch class

        if weight_rest <= 0:
            # No remaining capacity; no room for new items => no benefit in waiting
            # You may choose True or False here; Java continues and uses histogram,
            # but with peso_resto <= 0, loop won't count anything.
            pass

        occurrences = 0
        # Summation over bins corresponding to weights <= peso_resto
        upper_index = int(round(weight_rest))
        for a in range(upper_index):
            if 0 <= a < len(self.histogram_weights_n):
                occurrences+= self.histogram_weights_n[a]

        if self.n_samples <= 0:
            raise ValueError("n_samples is zero; load_stats must be called before run().")

        percentage = occurrences / float(self.n_samples)

        if percentage < self.threshold:
            # Probability of getting small-enough orders is low -> release
            return True

        # 5. Adaptive efficiency check
        self.counterImprovement += 1

        if batch_to_route.weight == 0:
            # Avoid division by zero: if weight is zero, treat efficiency as "bad"
            effic = sys.float_info.max
        else:
            effic = t_routing / batch_to_route.weight

        self.efficiency += effic
        mean_efficiency = self.efficiency / self.counterImprovement

        # Return !(effic < eficaciaMedia * 1.1)
        # i.e. True if effic >= 1.1 * eficacia_media, else False
        return not (effic < mean_efficiency * 1.1)