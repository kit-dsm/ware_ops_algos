"""MDP-based optimal-stopping formulation for the waiting decision.

Ported from project_4D4L/2_Stochastic_Waiting/Algorithms/analytic_progress/mdp_based_formulation.py.
"""

from __future__ import annotations

from .models import TimeSampleResult


def phase_1_discrete(samples: list[TimeSampleResult]) -> TimeSampleResult:
    """Explicit MDP optimal-stopping solution over discrete route indices.

    Minimises ``mean_predicted_order_completion_time`` via Bellman
    backward induction.  Assumes ``samples`` are sorted by ``idx``
    and ``idx = 0, ..., N``.
    """
    if not samples:
        raise ValueError("Samples list must not be empty")

    samples_sorted = sorted(samples, key=lambda s: s.idx)
    N = samples_sorted[-1].idx

    V = [0.0] * (N + 1)
    policy = [""] * (N + 1)

    def stop_cost(sample: TimeSampleResult) -> float:
        return sample.mean_predicted_order_completion_time

    V[N] = stop_cost(samples_sorted[-1])
    policy[N] = "integrate"

    for i in range(N - 1, -1, -1):
        sample_i = samples_sorted[i]
        c_stop = stop_cost(sample_i)
        c_wait = V[i + 1]

        if c_stop <= c_wait:
            V[i] = c_stop
            policy[i] = "integrate"
        else:
            V[i] = c_wait
            policy[i] = "wait"

    best_idx = None
    best_value = float("inf")
    best_sample = None

    for sample in samples_sorted:
        i = sample.idx
        if policy[i] == "integrate":
            c = stop_cost(sample)
            if c < best_value - 1e-9:
                best_value = c
                best_idx = i
                best_sample = sample

    if best_sample is None:
        best_sample = min(samples_sorted, key=stop_cost)

    return best_sample
