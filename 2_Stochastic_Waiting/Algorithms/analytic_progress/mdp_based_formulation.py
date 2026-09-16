from typing import List

try:
    from .models import TimeSampleResult
except ImportError:  # Direktes Ausfuehren als Skript
    from models import TimeSampleResult

def phase_1_discrete(samples: List[TimeSampleResult]) -> TimeSampleResult:
    """
    Explizite MDP-Optimal-Stopping-Lösung über diskrete Routenindizes.
    Minimiert mean_predicted_order_completion_time.

    Annahme: samples sind nach idx sortiert und idx = 0,...,N.
    """
    if not samples:
        raise ValueError("Samples list must not be empty")

    # Sortierung sicherstellen
    samples_sorted = sorted(samples, key=lambda s: s.idx)
    N = samples_sorted[-1].idx

    # Bellman-Werte: V[i] = minimaler Wert ab Zustand i
    # Wir minimieren hier direkt den OCT-Kostenwert
    V = [0.0] * (N + 1)
    policy = [""] * (N + 1)

    # c_I(i): Kosten, wenn wir in i stoppen (= integrate)
    def stop_cost(sample: TimeSampleResult) -> float:
        return sample.mean_predicted_order_completion_time

    # Randbedingung: am letzten Index muss entschieden werden
    V[N] = stop_cost(samples_sorted[-1])
    policy[N] = "integrate"

    # Rückwärtsiteration
    for i in range(N - 1, -1, -1):
        sample_i = samples_sorted[i]
        c_stop = stop_cost(sample_i)
        c_wait = V[i + 1]  # keine unmittelbaren Kosten, nur zukünftige

        if c_stop <= c_wait:
            V[i] = c_stop
            policy[i] = "integrate"
        else:
            V[i] = c_wait
            policy[i] = "wait"

    # Der optimale Startindex ist derjenige i mit policy[i] == "integrate"
    # und minimalem c_stop (das ist äquivalent, aber wir nehmen explizit die optimale Zeit).
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
        # theoretisch nicht möglich, zur Sicherheit:
        best_sample = min(samples_sorted, key=stop_cost)

    return best_sample