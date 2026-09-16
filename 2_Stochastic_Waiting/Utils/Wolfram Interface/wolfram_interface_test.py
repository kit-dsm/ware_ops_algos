from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# Pfade (Nutze Forward Slashes!)
KERNEL_PATH = "C:/Program Files/Wolfram Research/Wolfram/14.3/WolframKernel.exe"
SCRIPT_PATH = "/Algorithms/analytic_progress/Calculate_Intervals/analytic_wait_dynamic.wl"

N_AISLES = 8
N_PICKS  = 3
L        = 17   # Gassenlänge
w        = 1    # Abstand zwischen Gassen
v        = 1    # Geschwindigkeit
t_pick   = 0    # Pickzeit


def get_aisle_combinations(n_picks=N_PICKS, n_aisles=N_AISLES):
    """
    Erzeugt alle Verteilungen von n_picks Picks auf n_aisles Gassen.

    Rückgabe: Liste von (combo, probability)
        combo: Tuple (n1,...,n8), Summe = n_picks
               ni = Anzahl Picks in Gasse i (Index i entspricht Gasse i+1)
    """
    results = []

    def _recurse(remaining, idx, current):
        if idx == n_aisles:
            if remaining == 0:
                combo = tuple(current)
                coeff = factorial(n_picks)
                for k in combo:
                    coeff //= factorial(k)
                prob = coeff * (1 / n_aisles) ** n_picks
                results.append((combo, prob))
            return
        for k in range(remaining + 1):
            _recurse(remaining - k, idx + 1, current + [k])

    _recurse(n_picks, 0, [])
    return results


def combo_to_visited(combo):
    """Gibt sortierte Liste besuchter Gassen (1-indiziert, aufsteigend) zurück."""
    return [i + 1 for i, n in enumerate(combo) if n > 0]


def calculate_intervals(combo, L=L, w=w, v=v, t_pick=t_pick):
    """
    Berechnet kumulative Zeitintervalle der S-Shape Route.

    Rückgabe:
        intervals   : Liste von (t_start, t_end) je Segment
        total_time  : Gesamtroutendauer
        visited     : Besuchte Gassen (aufsteigend sortiert)
        is_return   : True wenn linkste Gasse Return-Gasse ist
    """
    visited = combo_to_visited(combo)
    if not visited:
        return [], 0.0, [], False

    n_visited   = len(visited)
    is_return   = (n_visited % 2 == 1)
    aisle_order = list(reversed(visited))   # rechts → links

    intervals = []
    t = 0.0

    # Segment 0: Depot → rechteste Gasse (horizontal)
    # Depot bei x=0, Gasse x_right bei x=visited[-1]
    dt = visited[-1] * w / v
    intervals.append((round(t, 6), round(t + dt, 6)))
    t += dt

    for i, aisle in enumerate(aisle_order):
        is_last = (i == n_visited - 1)

        if is_last and is_return:
            # Return-Gasse: Hinweg (Picks werden aufgenommen)
            dt = L / v + t_pick
            intervals.append((round(t, 6), round(t + dt, 6)))
            t += dt
            # Return-Gasse: Rückweg (keine Picks)
            dt = L / v
            intervals.append((round(t, 6), round(t + dt, 6)))
            t += dt
        else:
            # Normale vollständige Durchquerung
            dt = L / v + t_pick
            intervals.append((round(t, 6), round(t + dt, 6)))
            t += dt

        # Horizontal zur nächsten Gasse (falls vorhanden)
        if i < n_visited - 1:
            next_aisle = aisle_order[i + 1]
            dt = (aisle - next_aisle) * w / v
            intervals.append((round(t, 6), round(t + dt, 6)))
            t += dt

    # Letztes Segment: linkste Gasse → Depot (horizontal)
    dt = visited[0] * w / v
    intervals.append((round(t, 6), round(t + dt, 6)))
    t += dt

    return intervals, round(t, 6), visited, is_return

def print_combinations(combos, max_rows=15):
    header = f"{'Kombination':<36} {'Gassen':<18} {'P':>8} {'Routenzeit':>12} {'n':>4}"
    print(f"\n{header}")
    print("-" * len(header))           # ─ → -
    for combo, prob in combos[:max_rows]:
        visited = combo_to_visited(combo)
        _, total, _, _ = calculate_intervals(combo)
        print(f"{str(combo):<36} {str(visited):<18} {prob:>8.5f} {total:>12.1f} {len(visited):>4}")
    if len(combos) > max_rows:
        print(f"  ... ({len(combos) - max_rows} weitere nicht angezeigt)")
    print(f"\nGesamt: {len(combos)} Kombinationen  |  SumP = {sum(p for _,p in combos):.8f}")


def print_intervals(combo):
    intervals, total, visited, is_return = calculate_intervals(combo)
    n_visited  = len(visited)
    rs = n_visited / N_AISLES
    print(f"\n{'='*54}")                # ═ → =
    print(f"Kombination    : {combo}")
    print(f"Gassen         : {visited}")
    print(f"Return-Gasse   : {'Ja -> Gasse ' + str(visited[0]) if is_return else 'Nein'}")
    print(f"routeshare     : {n_visited}/{N_AISLES} = {rs:.4f}")
    print(f"candidateshare : {N_AISLES-n_visited}/{N_AISLES} = {1-rs:.4f}")
    print(f"{'-'*54}")                  # ─ → -
    print(f"  {'Seg':<8} {'t_start':>8} {'t_end':>8} {'dt':>8}")
    print(f"{'-'*54}")
    for i, (s, e) in enumerate(intervals):
        print(f"  I{i+1:<7} {s:>8.2f} {e:>8.2f} {e-s:>8.2f}")
    print(f"{'-'*54}")
    print(f"  Routenzeit gesamt: {total:.2f}")
    print(f"{'='*54}")
def compute_optimal_wait(session, combo, delta_exp, L=L, w=w, v=v):
    intervals, total, visited, is_return = calculate_intervals(combo, L, w, v)
    n_visited = len(visited)

    print(f"\nInstanz        : {combo}")
    print(f"Gassen         : {visited}")
    print(f"Routenzeit     : {total:.2f}")
    print(f"routeshare     : {n_visited}/{N_AISLES}")
    print(f"deltaExp       : {delta_exp}")

    wl_visited = [int(a) for a in visited]
    wl_intervals = [[float(s), float(e)] for s, e in intervals]
    wait_sym = wl.Symbol("wait")

    analytic_j = session.evaluate(
        wl.WarehouseOptimization.GetAnalyticJDynamic(
            float(L), float(w), float(v), float(delta_exp),
            wl_visited,
            wl_intervals
        )
    )

    result = session.evaluate(wl.NMinimize([analytic_j, wl.GreaterEqual(wait_sym, 0)], wait_sym))

    if isinstance(result, (list, tuple)):
        min_cost = float(session.evaluate(wl.N(result[0])))
        opt_wait = float(session.evaluate(wl.N(wl.ReplaceAll(wait_sym, result[1]))))
        if opt_wait < 1e-5:
            opt_wait = 0.0
        print(f"Optimales wait : {opt_wait:.4f}")
        print(f"Minimale Kosten: {min_cost:.4f}")
        return opt_wait, min_cost

    print("Fehler: Kein Minimum gefunden.")
    return None, None

if __name__ == "__main__":

    # 1. Alle Kombinationen
    combos = get_aisle_combinations()
    print_combinations(combos)

    # 2. Detailausgabe für konkrete Instanz
    example = (1, 1, 0, 0, 0, 0, 0, 1)   # Gassen 1, 4, 7
    print_intervals(example)

    # Verifikation: Gassen 2 und 8 → Intervalle sollten [0,8],[8,25],... ergeben
    print("\n--- Verifikation (Gassen 2, 8) ---")
    print_intervals((0, 1, 0, 0, 0, 0, 0, 1))

    # Verifikation: Gassen 1, 2, 8 → muss [0,8],[8,25],[25,31],[31,48],[48,49],
    #               [49,66],[66,83],[83,84] ergeben (wie bestehender Wolfram-Code)
    print("\n--- Verifikation (Gassen 1, 2, 8) ---")
    print_intervals((1, 1, 0, 0, 0, 0, 0, 1))

    # 3. Wolfram-Session
    session = WolframLanguageSession(KERNEL_PATH)
    try:
        print("\nStarte Wolfram Kernel...")
        session.start()
        session.evaluate(wl.Get(SCRIPT_PATH))
        print("Kernel online.")
        compute_optimal_wait(session, example, delta_exp=28.8)
    except Exception as e:
        print(f"Fehler: {e}")
    finally:
        session.terminate()
        print("\nKernel beendet.")