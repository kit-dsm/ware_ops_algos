"""
batch_optimizer.py

Loop ueber alle Gassen-Kombinationen -> E[wait*] = Σ p_combo * wait*_combo.

Ablauf pro Kombination:
  1. Python: WarehouseInstance
  2. Python: interval_builder.build_interval_data
             -> p1, p2, p3, p4, D_back-Parameter, D_pass, D_front pro Intervall
  3. Wolfram: GetAnalyticJ -> J(wait) symbolisch
              (Kuerzung L_ges/L_ges fuer vertikale Intervalle durch Simplify)
  4. Wolfram: NMinimize -> wait*, J*
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from math import factorial
from typing import Dict, List, Tuple
from dataclasses import dataclass

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl

import core as core_mod
from models import WarehouseInstance
from Algorithms.analytic_progress.Archive import interval_builder_backup as ib

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

KERNEL_PATH = "C:/Program Files/Wolfram Research/Wolfram/14.3/WolframKernel.exe"
SCRIPT_PATH = ("C:/Users/ya5909/Git-Projekte/project_4D4L/"
               "2_Stochastic_Waiting/Wolfram/analytic_wait_dynamic.wl")


@dataclass(frozen=True)
class LayoutParams:
    M:   int   = 8
    N_L: int   = 17
    w:   float = 1.0
    L:   float = 17.0
    v:   float = 1.0
    t_p: float = 0.0


@dataclass
class CombinationResult:
    combo:       Tuple[int, ...]
    probability: float
    visited:     List[int]
    routenzeit:  float
    wait_opt:    float
    J_opt:       float


# ---------------------------------------------------------------------------
# Kombinationen
# ---------------------------------------------------------------------------

def get_aisle_combinations(
    n_picks: int = 3,
    n_aisles: int = 8,
) -> List[Tuple[Tuple[int, ...], float]]:
    """Multinomial-Verteilungen von n_picks Picks auf n_aisles Gassen."""
    results = []

    def _recurse(remaining, idx, current):
        if idx == n_aisles:
            if remaining == 0:
                combo = tuple(current)
                coeff = factorial(n_picks)
                for k in combo:
                    coeff //= factorial(k)
                prob = coeff * (1.0 / n_aisles) ** n_picks
                results.append((combo, prob))
            return
        for k in range(remaining + 1):
            _recurse(remaining - k, idx + 1, current + [k])

    _recurse(n_picks, 0, [])
    return results


def combo_to_instance(combo: Tuple[int, ...], layout: LayoutParams) -> WarehouseInstance:
    visited = [i + 1 for i, n in enumerate(combo) if n > 0]
    n_list  = [combo[a - 1] for a in visited]
    return WarehouseInstance(
        M=layout.M, N_L=layout.N_L, w=layout.w, L=layout.L,
        A=visited, n_list=n_list, v=layout.v, t_p=layout.t_p,
        P=[], arrival_times=[],
    )


# ---------------------------------------------------------------------------
# Einzel-Optimierung
# ---------------------------------------------------------------------------

def optimize_combination(
    session: WolframLanguageSession,
    combo: Tuple[int, ...],
    probability: float,
    delta_exp: float,
    layout: LayoutParams,
    n_picks_after: int = 3,
) -> CombinationResult:
    """
    Eine Kombination -> (wait*, J*) via Wolfram.

    Schritte:
      1. interval_builder berechnet alle Funktionsparameter (Python)
      2. GetAnalyticJ baut J(wait) symbolisch (Wolfram, inkl. Simplify)
      3. NMinimize findet wait*, J* (Wolfram)
    """
    inst = combo_to_instance(combo, layout)
    _, _, T_end = core_mod.compute_segment_times(inst)

    lambda_val   = 1.0 / delta_exp
    e_cost_after = n_picks_after / lambda_val + T_end

    intervals, _ = ib.build_interval_data(inst)
    wl_intervals  = ib.to_wolfram_list(intervals)

    wait_sym = wl.Symbol("wait")

    analytic_j = session.evaluate(
        wl.WarehouseOptimization.GetAnalyticJ(
            float(layout.L),
            float(lambda_val),
            wl_intervals,
            float(T_end),
            float(e_cost_after),
        )
    )

    result = session.evaluate(
        wl.NMinimize([analytic_j, wl.GreaterEqual(wait_sym, 0)], wait_sym)
    )

    wait_opt = 0.0
    j_opt    = float("inf")

    if isinstance(result, (list, tuple)) and len(result) == 2:
        try:
            j_opt    = float(session.evaluate(wl.N(result[0])))
            wait_opt = float(session.evaluate(
                wl.N(wl.ReplaceAll(wait_sym, result[1]))
            ))
            if wait_opt < 1e-5:
                wait_opt = 0.0
        except Exception:
            pass

    return CombinationResult(
        combo=combo, probability=probability,
        visited=[i + 1 for i, n in enumerate(combo) if n > 0],
        routenzeit=T_end, wait_opt=wait_opt, J_opt=j_opt,
    )


# ---------------------------------------------------------------------------
# Batch-Loop + Aggregation
# ---------------------------------------------------------------------------

def batch_optimize(
    delta_exp: float,
    layout: LayoutParams = None,
    n_picks: int = 3,
    n_picks_after: int = 3,
    verbose: bool = True,
) -> Dict:
    """Loop ueber alle Kombinationen -> E[wait*]."""
    if layout is None:
        layout = LayoutParams()

    combos = get_aisle_combinations(n_picks, layout.M)
    if verbose:
        print(f"Optimiere {len(combos)} Kombinationen  (delta={delta_exp})...\n")

    session = WolframLanguageSession(KERNEL_PATH)
    results: List[CombinationResult] = []

    try:
        session.start()
        session.evaluate(wl.Get(SCRIPT_PATH))
        if verbose:
            print("Wolfram Kernel online.\n")

        for i, (combo, prob) in enumerate(combos, 1):
            try:
                res = optimize_combination(
                    session, combo, prob, delta_exp, layout, n_picks_after
                )
                results.append(res)
                if verbose and (i % 20 == 0 or i == len(combos)):
                    print(f"  [{i:>3}/{len(combos)}]  combo={res.combo}  "
                          f"wait*={res.wait_opt:>7.4f}  J*={res.J_opt:>9.4f}")
            except Exception as e:
                if verbose:
                    print(f"  [{i:>3}/{len(combos)}]  combo={combo}: FEHLER {e}")

    finally:
        session.terminate()
        if verbose:
            print("\nKernel beendet.")

    e_wait = sum(r.probability * r.wait_opt for r in results)
    e_j    = sum(r.probability * r.J_opt
                 for r in results if r.J_opt != float("inf"))

    return {
        "delta_exp" : delta_exp,
        "n_combos"  : len(combos),
        "n_solved"  : len(results),
        "E_wait_opt": e_wait,
        "E_J_opt"   : e_j,
        "details"   : results,
    }


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def print_summary(summary: Dict) -> None:
    print("\n" + "=" * 70)
    print(f"Zusammenfassung  (delta={summary['delta_exp']})")
    print("=" * 70)
    print(f"  Kombinationen : {summary['n_combos']}")
    print(f"  Erfolgreich   : {summary['n_solved']}")
    print(f"  E[wait*]      : {summary['E_wait_opt']:.4f}")
    print(f"  E[J*]         : {summary['E_J_opt']:.4f}")
    print("=" * 70)

    header = (f"  {'Kombination':<32} {'P':>8}  {'Gassen':<16} "
              f"{'wait*':>8}  {'J*':>9}")
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))
    for r in sorted(summary["details"], key=lambda r: -r.probability):
        print(f"  {str(r.combo):<32} {r.probability:>8.5f}  "
              f"{str(r.visited):<16} {r.wait_opt:>8.4f}  {r.J_opt:>9.4f}")


# ---------------------------------------------------------------------------
# Validierung
# ---------------------------------------------------------------------------

def validate_known_case(delta_exp: float = 28.8) -> None:
    """
    Validiert (1,1,0,0,0,0,0,1) -> Gassen [1,2,8].
    Referenz aus manuellem Mathematica-Code: J* = 14.3593.

    Gibt zusaetzlich die Intervall-Tabelle aus (zur Diagnose).
    """
    layout = LayoutParams()
    combo  = (1, 1, 0, 0, 0, 0, 0, 1)
    prob   = 6.0 / (8 ** 3)

    print("Validierung: combo=(1,1,0,0,0,0,0,1)  Gassen=[1,2,8]")
    print(f"delta_exp = {delta_exp}\n")

    inst = combo_to_instance(combo, layout)
    intervals, T_end = ib.build_interval_data(inst)
    ib.print_intervals(intervals)

    session = WolframLanguageSession(KERNEL_PATH)
    try:
        session.start()
        session.evaluate(wl.Get(SCRIPT_PATH))

        res = optimize_combination(session, combo, prob, delta_exp, layout)
        print(f"\n  Routenzeit  : {res.routenzeit:.4f}")
        print(f"  wait*       : {res.wait_opt:.4f}")
        print(f"  J*          : {res.J_opt:.4f}")
        print(f"  Referenz    : J* = 14.3593")
        print(f"  Differenz   : {abs(res.J_opt - 14.3593):.4f}")
    finally:
        session.terminate()


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    validate_known_case(delta_exp=28.8)
    summary = batch_optimize(delta_exp=28.8)
    print_summary(summary)
