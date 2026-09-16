"""
Fehler in letzter Phase - nicht verwenden! Nur Archiv zur Doku
Berechnet E[D](s) = c0 + c1*s + c2*s^2 pro Phasen-Intervall direkt
aus der Routengeometrie.

Formelherleitung
----------------

Vertikale Phasen (Picker in Gasse a_j, Tiefe L_y(s) = v_eff * s):
    p2(s)    = (i*L + L_y) / (M*L)         linear in s
    D_back(s)= N(L_y) / L_ges              rational, aber:
    p2*D_back= N(L_y) / (M*L)             Polynom Grad 2:
               c0_pb = K0/M
               c1_pb = K1 * v_eff / M      (K1 = 2*A_comp, ausser Return-Abstieg: 0)
               c2_pb = v_eff^2 / (M*L)

    p3, p4: konstant (x unveraendert), aus G_pass/G_front-Zaehlung
    D_pass:  konstant  = 2*dx  (+ 2*L falls noetig)
    D_front: nur fuer erste Gasse relevant, sonst 0

Horizontale Phasen (x(s) = x_from - s, dt = x_from - x_to):
    p2:      konstant = i_nach/M
    D_back(s)= K0_from/A_comp + 2*s       (exakt linear)
    p3(s), p4(s): linear, aus antizipatorischem Modell (+1 nur wenn Kandidaten links)
    D_pass(s): linear, aus 2-Punkt-Geometrie
    D_front: 0 (fuer Kandidaten innerhalb Routenspanne)

Antizipatorisches p4-Modell
----------------------------
p4_max = (Anzahl_Kandidaten_links_von_x_max + 1) / M
         NUR wenn ueberhaupt Kandidaten links von x_max existieren,
         sonst p4_max = 0.
Dieses Modell gibt p3 = count(c > x)/M korrekt fuer alle
Integer-x-Positionen im Inneren des Intervalls.
"""

import sys
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from Algorithms.analytic_progress.Calculate_Intervals import continuous_calculation as cc   # nur fuer effective_vertical_speed_for_aisle
    from Algorithms.analytic_progress.Calculate_Intervals import core as core_mod
    from .models import WarehouseInstance
except ImportError:
    import continuous_calculation as cc   # nur fuer effective_vertical_speed_for_aisle
    import core as core_mod
    from models import WarehouseInstance


@dataclass(frozen=True)
class IntervalData:
    """E[D](s) = c0 + c1*s + c2*s^2 in lokaler Zeit s = troute - t_start."""
    t_start:     float
    t_end:       float
    phase_label: str
    ed_coeffs:   List[float]   # [c0, c1, c2]


# ---------------------------------------------------------------------------
# Geometrie-Hilfsfunktionen
# ---------------------------------------------------------------------------

def _completed_right(inst: WarehouseInstance, aisle_x: float) -> List[int]:
    """Besuchte Gassen rechts von aisle_x, aufsteigend sortiert (naechste zuerst)."""
    return sorted(a for a in inst.A if a > aisle_x)


def _k0(inst: WarehouseInstance, aisle_x: float, completed: List[int]) -> float:
    """
    K0 = Σ_idx (L + 2*(dx_idx + idx*L))

    Konstanter Zaehleranteil fuer D_back, summiert ueber
    alle abgeschlossenen Gassen rechts von aisle_x.
    idx ist 0-basiert (naechste Gasse zuerst).
    """
    return sum(
        inst.L + 2.0 * (abs(aisle_x - float(a)) * inst.w + idx * inst.L)
        for idx, a in enumerate(completed)
    )


def _dx_gpass(inst: WarehouseInstance, x: float, candidates: List[int]) -> float:
    """Mittlerer horizontaler Abstand von x zu G_pass (Kandidaten > x)."""
    g = [c for c in candidates if c > x]
    if not g:
        return 0.0
    return sum(abs(c - x) * inst.w for c in g) / len(g)


def _dpass_const(inst: WarehouseInstance, x: float, candidates: List[int],
                 a_ret: Optional[int], in_ret_up: bool = False) -> float:
    """
    Konstantes D_pass fuer vertikale Phasen.

    Gerade k:      2*dx + 2*L
    Ungerades k:
      in_ret_up:   2*dx       (Return-Gasse Aufstieg)
      x > a_ret:  2*dx       (normale Gasse rechts von Return-Gasse)
      x <= a_ret: 2*dx + 2*L (normale Gasse links)
    """
    dx = _dx_gpass(inst, x, candidates)
    if dx == 0.0:
        return 0.0
    if inst.k % 2 == 0:
        return 2.0 * dx + 2.0 * inst.L
    if in_ret_up:
        return 2.0 * dx
    if a_ret is not None and x <= float(a_ret):
        return 2.0 * dx + 2.0 * inst.L
    return 2.0 * dx


def _dpass_horiz(inst: WarehouseInstance, x: float, candidates: List[int],
                 a_ret: Optional[int], after_ret=False) -> float:
    dx = _dx_gpass(inst, x, candidates)
    if dx == 0.0:
        return 0.0
    if inst.k % 2 == 0:
        return 2.0 * dx + 2.0 * inst.L
    # +2L nur NACH Abschluss der Return-Gasse (Phase 4)
    if a_ret is not None and after_ret and x <= float(a_ret):
        return 2.0 * dx + 2.0 * inst.L
    return 2.0 * dx


def _dfront_first_aisle(inst: WarehouseInstance, candidates: List[int]) -> float:
    """
    D_front fuer die erste (rechteste) Gasse (outbound=True).

    Modell fuer Phase 1 gemaess continuous_calculation._continuous_detour_aup:
    - horizontaler Anteil: mittlere Distanz von max(A) zu allen Kandidaten
      rechts von max(A), normiert ueber alle unbesuchten Gassen.
    - Traversal-Anteil 2*L nur falls k gerade ist.
      Bei ungeradem k (Return-Gasse vorhanden) entfaellt dieser Anteil.
    """
    if not candidates:
        return 0.0

    max_a = max(inst.A)
    beyond = [c for c in candidates if c > max_a]
    dx = sum((c - max_a) * inst.w for c in beyond) / len(candidates)
    horizontal = 2.0 * dx
    if inst.k % 2 == 0:
        return 2.0 * inst.L + horizontal
    return horizontal


def _p3p4_horiz_coeffs(inst: WarehouseInstance, x_from: float, x_to: float,
                        candidates: List[int]) -> Tuple[float, float, float, float]:
    """
    Lineare Koeffizienten [a0_p3, a1_p3, a0_p4, a1_p4] fuer horizontale Phase.

    x(s) = x_from - s (Picker bewegt sich nach links).
    Antizipatorisches Modell: p4_max = (count + 1)/M nur wenn
    Kandidaten links von x_from existieren.

    Gibt korrektes p3 = count(c > x)/M fuer alle Ganzzahl-x im Inneren.
    """
    dt = x_from - x_to
    x_min, x_max = x_to, x_from
    cand_share = len(candidates) / inst.M

    cands_left_max = [c for c in candidates if c < x_max]
    cands_left_min = [c for c in candidates if c < x_min]

    p4_max = (len(cands_left_max) + 1) / inst.M if len(cands_left_max) > 0 else 0.0
    p4_min = len(cands_left_min) / inst.M

    # u(s) = 1 - s/dt  (abnehment mit s, da Picker nach links geht)
    # p4(s) = p4_max - (p4_max - p4_min)/dt * s
    a0_p4 = p4_max
    a1_p4 = -(p4_max - p4_min) / dt if dt > 1e-12 else 0.0

    a0_p3 = cand_share - p4_max
    a1_p3 = -a1_p4

    return a0_p3, a1_p3, a0_p4, a1_p4


def _dpass_horiz_coeffs(inst: WarehouseInstance, x_from: float, x_to: float,
                         candidates: List[int], a_ret: Optional[int]) -> Tuple[float, float]:
    """
    Lineare Koeffizienten [c0, c1] fuer D_pass(s) in horizontaler Phase.
    D_pass(s) = D_pass(x_from) + (D_pass(x_to) - D_pass(x_from)) / dt * s.
    """
    dt = x_from - x_to
    # after_ret=True nur wenn x_from == a_ret (Phase 4: Picker hat Return-Gasse verlassen)
    after_ret = (a_ret is not None and int(round(x_from)) == a_ret)
    d0 = _dpass_horiz(inst, x_from, candidates, a_ret, after_ret)
    d1 = _dpass_horiz(inst, x_to, candidates, a_ret, after_ret)
    c0 = d0
    c1 = (d1 - d0) / dt if dt > 1e-12 else 0.0
    return c0, c1


def _completed_horiz(inst: WarehouseInstance, x_from: float, a_ret) -> List[int]:
    """
    Abgeschlossene Gassen am Start der horizontalen Phase.

    Gassen mit a >= x_from werden als 'hinter dem Picker' gezaehlt,
    AUSSER bei Phase_4 (x_from == a_ret): dort wird die Return-Gasse
    selbst ausgeschlossen (Modell-Konvention, vgl. Mathematica).
    """
    if a_ret is not None and int(round(x_from)) == a_ret:
        return sorted(a for a in inst.A if a > x_from)
    return sorted(a for a in inst.A if a >= x_from)


def _dback_horiz(inst: WarehouseInstance, x_from: float,
                 completed: List[int]) -> Tuple[float, float]:
    """
    D_back(s) = b0 + b1*s fuer horizontale Phasen (exakt linear).

    b0 = K0_from / A_comp  (Backtrack-Kosten am Start der Phase)
    b1 = 2                 (jede Einheit Bewegung erhoehe D_back um 2)
    """
    A_comp = len(completed)
    if A_comp == 0:
        return 0.0, 0.0
    K0_from = _k0(inst, x_from, completed)
    return K0_from / A_comp, 2.0


# ---------------------------------------------------------------------------
# Koeffizienten-Funktionen pro Phase
# ---------------------------------------------------------------------------

def _coeffs_phase1(inst: WarehouseInstance, candidates: List[int]) -> List[float]:
    """Phase 1 (Depot → erste Gasse): konstanter Front-Umweg fuer unbesuchte Gassen."""
    p4 = len(candidates) / inst.M if inst.M > 0 else 0.0
    d_front = _dfront_first_aisle(inst, candidates)
    return [p4 * d_front, 0.0, 0.0]


def _coeffs_vertical(inst: WarehouseInstance, j: int, rank: int,
                     candidates: List[int], a_ret: Optional[int]) -> List[float]:
    """
    Vertikale Phase in Gasse A[j-1], Position rank (0-basiert von rechts).

    c0 = K0/M + p3*D_pass + p4*D_front
    c1 = K1*v_eff/M
    c2 = v_eff^2 / (M*L)
    """
    aisle_x = float(inst.A[j - 1])
    comp   = _completed_right(inst, aisle_x)
    A_comp = len(comp)
    K0     = _k0(inst, aisle_x, comp) if comp else 0.0
    K1     = 2.0 * A_comp
    v_eff  = cc.effective_vertical_speed_for_aisle(inst, j)

    c0_pb = K0 / inst.M
    c1_pb = K1 * v_eff / inst.M
    c2_pb = v_eff ** 2 / (inst.M * inst.L)

    p3 = len([c for c in candidates if c > aisle_x]) / inst.M
    p4 = len([c for c in candidates if c < aisle_x]) / inst.M
    d_pass  = _dpass_const(inst, aisle_x, candidates, a_ret)
    d_front = _dfront_first_aisle(inst, candidates) if rank == 0 else 0.0

    return [c0_pb + p3 * d_pass + p4 * d_front, c1_pb, c2_pb]


def _coeffs_ret_up(inst: WarehouseInstance, j: int, rank: int,
                   candidates: List[int], a_ret: Optional[int]) -> List[float]:
    """
    Return-Gasse Aufstieg (phase_n_2_ret_up): L_y = 0 → D_back konstant.

    c0 = K0/M + p3*D_pass
    c1 = c2 = 0
    """
    aisle_x = float(inst.A[j - 1])
    comp   = _completed_right(inst, aisle_x)
    K0     = _k0(inst, aisle_x, comp) if comp else 0.0

    p3 = len([c for c in candidates if c > aisle_x]) / inst.M
    d_pass = _dpass_const(inst, aisle_x, candidates, a_ret, in_ret_up=True)

    return [K0 / inst.M + p3 * d_pass, 0.0, 0.0]


def _coeffs_ret_down(inst: WarehouseInstance, j: int,
                     candidates: List[int], a_ret: Optional[int]) -> List[float]:
    """
    Return-Gasse Abstieg (phase_n_1_ret_down): K1=0, D_pass(s)=2*dx+2*v*s.

    c0 = K0/M + p3 * (2*dx + 2*L)
    c1 = 0
    c2 = v^2 / (M*L)
    """
    aisle_x = float(inst.A[j - 1])
    comp   = _completed_right(inst, aisle_x)
    K0     = _k0(inst, aisle_x, comp) if comp else 0.0

    p3  = len([c for c in candidates if c > aisle_x]) / inst.M
    dx  = _dx_gpass(inst, aisle_x, candidates)
    v   = inst.v

    d_pass = 2.0 * dx + 2.0 * inst.L
    c0 = K0 / inst.M + p3 * d_pass     # D_pass bei L_y=0
    c1 = 0.0
    c2 = v ** 2 / (inst.M * inst.L)

    return [c0, c1, c2]


def _coeffs_horizontal(inst: WarehouseInstance, x_from: float, x_to: float,
                        i_after: int, candidates: List[int],
                        a_ret: Optional[int]) -> List[float]:
    """
    Horizontale Phase von x_from nach x_to (Picker bewegt sich nach links).

    E[D](s) = p2*D_back + p3*D_pass + p4*D_front
    Alle Terme sind polynomiell in s (Grad <= 2).
    """
    p2 = i_after / inst.M          # konstant

    comp = _completed_horiz(inst, x_from, a_ret)
    b0, b1 = _dback_horiz(inst, x_from, comp)     # D_back(s) = b0 + b1*s

    a0_p3, a1_p3, a0_p4, a1_p4 = _p3p4_horiz_coeffs(inst, x_from, x_to, candidates)
    c0_dp, c1_dp = _dpass_horiz_coeffs(inst, x_from, x_to, candidates, a_ret)

    # D_front = 0 fuer Kandidaten innerhalb der Routenspanne
    # (allgemein: s.u. _dfront_first_aisle, hier fuer horizontale Phasen immer 0,
    #  da Kandidaten ausserhalb der Route nur waehrend der ersten Gasse relevant sind)
    d_front = 0.0

    # E[D](s) = p2*(b0+b1*s) + (a0_p3+a1_p3*s)*(c0_dp+c1_dp*s) + (a0_p4+a1_p4*s)*d_front
    c0 = p2 * b0 + a0_p3 * c0_dp + a0_p4 * d_front
    c1 = p2 * b1 + a0_p3 * c1_dp + a1_p3 * c0_dp + a1_p4 * d_front
    c2 = a1_p3 * c1_dp      # (lineare p3) * (lineare D_pass) → quadratisch

    return [c0, c1, c2]


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def build_interval_data(inst: WarehouseInstance):
    """
    Berechnet IntervalData fuer alle Phasen-Intervalle analytisch.

    Gibt (intervals, T_end) zurueck.
    """
    T_in, T_out, T_end = core_mod.compute_segment_times(inst)

    k      = inst.k
    A      = inst.A
    M      = inst.M
    has_ret = (k % 2 == 1)
    a_ret: Optional[int] = int(A[0]) if has_ret else None

    candidates = sorted(set(range(1, M + 1)) - set(A))

    # Besuchsreihenfolge: rechte Gasse zuerst (absteigend nach Entry-Zeit)
    seq = sorted(T_in.keys(), key=lambda j: T_in[j])   # [k, k-1, ..., 1]

    intervals = []

    # ── Phase 1: Depot → erste Gasse ─────────────────────────────────────
    first_j = seq[0]
    intervals.append(IntervalData(
        t_start=0.0, t_end=T_in[first_j],
        phase_label="phase_1_to_first_entry",
        ed_coeffs=_coeffs_phase1(inst, candidates),
    ))

    for rank, j in enumerate(seq):
        aisle_x = float(A[j - 1])
        t_in    = T_in[j]
        t_out   = T_out[j]

        # ── Vertikale Phase ──────────────────────────────────────────────
        if has_ret and j == seq[-1]:
            # Return-Gasse: aufteilen in Aufstieg und Abstieg
            v_eff_up = cc.effective_vertical_speed_for_aisle(inst, j)
            t_split  = t_in + inst.L / v_eff_up

            intervals.append(IntervalData(
                t_start=t_in, t_end=t_split,
                phase_label="phase_n_2_ret_up",
                ed_coeffs=_coeffs_ret_up(inst, j, rank, candidates, a_ret),
            ))
            intervals.append(IntervalData(
                t_start=t_split, t_end=t_out,
                phase_label="phase_n_1_ret_down",
                ed_coeffs=_coeffs_ret_down(inst, j, candidates, a_ret),
            ))
        else:
            intervals.append(IntervalData(
                t_start=t_in, t_end=t_out,
                phase_label="phase_2_vertical",
                ed_coeffs=_coeffs_vertical(inst, j, rank, candidates, a_ret),
            ))

        # ── Horizontale Phase zur naechsten Gasse ───────────────────────
        if rank < k - 1:
            j_next  = seq[rank + 1]
            x_from  = float(A[j - 1])
            x_to    = float(A[j_next - 1])
            i_after = rank + 1      # Anzahl abgeschlossener Gassen nach dem Exit

            intervals.append(IntervalData(
                t_start=t_out, t_end=T_in[j_next],
                phase_label="phase_3_horizontal",
                ed_coeffs=_coeffs_horizontal(
                    inst, x_from, x_to, i_after, candidates, a_ret),
            ))

    # ── Phase 4: letzte Gasse → Depot ────────────────────────────────────
    j_last  = seq[-1]
    x_from  = float(A[j_last - 1])
    x_to    = 0.0

    intervals.append(IntervalData(
        t_start=T_out[j_last], t_end=T_end,
        phase_label="phase_4_after_last_aisle",
        ed_coeffs=_coeffs_horizontal(
            inst, x_from, x_to, k, candidates, a_ret),
    ))

    return intervals, T_end


# ---------------------------------------------------------------------------
# Wolfram-kompatible Liste
# ---------------------------------------------------------------------------

def to_wolfram_list(intervals) -> list:
    """Format: [[t_start, t_end, [c0, c1, c2]], ...]"""
    return [
        [float(iv.t_start), float(iv.t_end),
         [float(c) for c in iv.ed_coeffs]]
        for iv in intervals
    ]


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------

_STATUS_MAP = {
    "phase_1_to_first_entry":   "approach",
    "phase_2_vertical":         "vertical",
    "phase_3_horizontal":       "horizontal",
    "phase_n_2_ret_up":         "ret_up",
    "phase_n_1_ret_down":       "ret_down",
    "phase_4_after_last_aisle": "return",
}


def print_intervals(intervals) -> None:
    header = (f"{'Nr':<5}  {'Status':<12} {'t_start':>7} {'t_end':>7}  "
              f"{'c0':>10}  {'c1':>10}  {'c2':>12}")
    print(f"\n{header}")
    print("-" * len(header))
    for i, iv in enumerate(intervals, start=1):
        c0, c1, c2 = iv.ed_coeffs
        status = _STATUS_MAP.get(iv.phase_label, iv.phase_label)
        print(f"I_{i:<3}  {status:<12} {iv.t_start:>7.2f} {iv.t_end:>7.2f}  "
              f"{c0:>10.5f}  {c1:>10.6f}  {c2:>12.8f}")
