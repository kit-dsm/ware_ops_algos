"""
interval_builder.py
====================

Erzeugt IntervalData-Strukturen (Polynom-Koeffizienten c0, c1, c2)
durch quadratische Interpolation der validierten
continuous_calculation.continuous_expected_detour Funktion.

Strategie:
  Fuer jede Phase [t_start, t_end] werden drei innere Stuetzpunkte
  ausgewertet (kurz nach t_start, in der Mitte, kurz vor t_end).
  Lagrange-Interpolation liefert exakt
      E[D](s) = c0 + c1*s + c2*s^2,    s = t_route - t_start.

Diese Strategie ist exakt, weil E[D] auf jeder Phase analytisch
hoechstens quadratisch in s ist:
  - vertikale Phase: p2(s)*D_back(s) kuerzt sich algebraisch auf
    eine quadratische Form mit c2 = v_eff^2 / (M*L);
    p3*D_pass und p4*D_front sind konstant.
  - horizontale Phase: p3, p4, D_pass sind linear in s; das
    Produkt zweier Linearer ist quadratisch; p2*D_back ist linear.

Phasen-Zuordnung wird 1:1 von continuous_calculation uebernommen.
"""

import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from . import continuous_calculation as cc
    from . import core as core_mod
    from .models import WarehouseInstance
except ImportError:
    import continuous_calculation as cc
    import core as core_mod
    from models import WarehouseInstance


@dataclass(frozen=True)
class IntervalData:
    """E[D](s) = c0 + c1*s + c2*s^2 in lokaler Zeit s = t_route - t_start."""
    t_start:     float
    t_end:       float
    phase_label: str
    ed_coeffs:   List[float]   # [c0, c1, c2]


# Mindestabstand zu Phasenraendern fuer das Sampling.
# Muss > 1e-12 sein (interne eps in continuous_calculation),
# damit die Phasenklassifikation eindeutig im Phaseninneren landet.
_BOUNDARY_OFFSET = 1e-8


# ---------------------------------------------------------------------------
# Quadratische Lagrange-Interpolation
# ---------------------------------------------------------------------------

def _lagrange_quadratic(samples: List[Tuple[float, float]]) -> List[float]:
    """
    Bestimmt [c0, c1, c2] sodass c0 + c1*s + c2*s^2 = y an drei Stuetz-
    stellen exakt erfuellt ist. Robust gegen kollidierende oder fehlende
    Stuetzstellen (fallback auf linear bzw. konstant).
    """
    if len(samples) == 1:
        return [samples[0][1], 0.0, 0.0]

    if len(samples) == 2:
        (s1, y1), (s2, y2) = samples
        if abs(s2 - s1) < 1e-15:
            return [y1, 0.0, 0.0]
        c1 = (y2 - y1) / (s2 - s1)
        c0 = y1 - c1 * s1
        return [c0, c1, 0.0]

    (s1, y1), (s2, y2), (s3, y3) = samples[:3]

    d12 = s1 - s2
    d13 = s1 - s3
    d23 = s2 - s3
    if abs(d12) < 1e-15 or abs(d13) < 1e-15 or abs(d23) < 1e-15:
        return _lagrange_quadratic(samples[:2])

    den1 =  d12 * d13      # (s1-s2)(s1-s3)
    den2 = -d12 * d23      # (s2-s1)(s2-s3)
    den3 =  d13 * d23      # (s3-s1)(s3-s2)

    a1, a2, a3 = y1 / den1, y2 / den2, y3 / den3

    c2 = a1 + a2 + a3
    c1 = -(a1 * (s2 + s3) + a2 * (s1 + s3) + a3 * (s1 + s2))
    c0 =  a1 * s2 * s3 + a2 * s1 * s3 + a3 * s1 * s2

    return [c0, c1, c2]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_phase(
    t_start: float,
    t_end:   float,
    inst:    WarehouseInstance,
    T_in:    Dict[int, float],
    T_out:   Dict[int, float],
) -> List[float]:
    """
    Wertet continuous_expected_detour an drei Stuetzpunkten aus und fittet
    daraus das quadratische E[D]-Polynom.

    Stuetzpunktwahl (ASYMMETRISCH - die Phasen haben Artefakte an
    entgegengesetzten Enden):
      - Start bei 25% dt: umgeht das Eintritts-Artefakt der ersten Gasse
        (G_pass/G_front unmittelbar nach Phasenbeginn noch im Approach-Zustand).
      - Mitte bei 50% dt.
      - Ende bei (dt - eps), direkt am Phasenende: in der letzten Phase
        (Rueckweg zum Depot) ist dort p4 -> 0, sodass der in continuous
        zwischen laengenbasiertem p4 und mengenbasiertem G_front inkonsistente
        D_front-Term verschwindet. Ein Punkt bei 75% dt wuerde dieses
        Tail-Artefakt dagegen einfangen.

    E[D](s) ist auf jeder Phase analytisch quadratisch; drei saubere
    Stuetzpunkte rekonstruieren das Polynom exakt (inkl. c0 via Extrapolation).
    """
    dt = t_end - t_start

    if dt <= 4.0 * _BOUNDARY_OFFSET:
        t_mid = 0.5 * (t_start + t_end)
        est = cc.continuous_expected_detour(t_mid, inst, T_in, T_out)
        return [float(est.expected_detour), 0.0, 0.0]

    eps = _BOUNDARY_OFFSET
    s_vals = [0.25 * dt, 0.5 * dt, dt - eps]

    y_vals = []
    for s in s_vals:
        est = cc.continuous_expected_detour(t_start + s, inst, T_in, T_out)
        y_vals.append(float(est.expected_detour))

    return _lagrange_quadratic(list(zip(s_vals, y_vals)))


# ---------------------------------------------------------------------------
# Phasen-Konstruktion
# ---------------------------------------------------------------------------

def _ordered_aisles_by_entry(T_in: Dict[int, float]) -> List[int]:
    return [j for j, _ in sorted(T_in.items(), key=lambda item: item[1])]


def _build(t_start: float, t_end: float, label: str,
           inst:  WarehouseInstance,
           T_in:  Dict[int, float],
           T_out: Dict[int, float]) -> IntervalData:
    coeffs = _sample_phase(t_start, t_end, inst, T_in, T_out)
    return IntervalData(t_start=float(t_start),
                        t_end=float(t_end),
                        phase_label=label,
                        ed_coeffs=coeffs)


def build_interval_data(inst: WarehouseInstance):
    """
    Liefert (intervals, T_end). Phasenstruktur ist identisch zur frueheren
    analytischen Version; nur die Koeffizienten werden jetzt aus
    continuous_calculation gefittet.
    """
    T_in, T_out, T_end = core_mod.compute_segment_times(inst)

    if inst.k <= 0:
        return [], T_end

    has_ret = (inst.k % 2 == 1)
    seq = _ordered_aisles_by_entry(T_in)

    intervals: List[IntervalData] = []

    # Phase 1: depot -> erste Gasse
    first_j = seq[0]
    intervals.append(_build(0.0, T_in[first_j],
                            "phase_1_to_first_entry", inst, T_in, T_out))

    for rank, j in enumerate(seq):
        t_in_j  = T_in[j]
        t_out_j = T_out[j]

        if has_ret and j == seq[-1]:
            # Return-Gasse: Aufstieg und Abstieg sind getrennte Phasen,
            # da sich der Backtrack-Detour an der Spitze qualitativ aendert.
            v_eff_up = cc.effective_vertical_speed_for_aisle(inst, j)
            t_split  = t_in_j + inst.L / v_eff_up
            intervals.append(_build(t_in_j, t_split,
                                    "phase_n_2_ret_up", inst, T_in, T_out))
            intervals.append(_build(t_split, t_out_j,
                                    "phase_n_1_ret_down", inst, T_in, T_out))
        else:
            intervals.append(_build(t_in_j, t_out_j,
                                    "phase_2_vertical", inst, T_in, T_out))

        # Horizontaler Wechsel zur naechsten Gasse
        if rank < inst.k - 1:
            j_next = seq[rank + 1]
            intervals.append(_build(t_out_j, T_in[j_next],
                                    "phase_3_horizontal", inst, T_in, T_out))

    # Phase 4: letzte Gasse -> depot
    j_last = seq[-1]
    intervals.append(_build(T_out[j_last], T_end,
                            "phase_4_after_last_aisle", inst, T_in, T_out))

    return intervals, T_end


# ---------------------------------------------------------------------------
# Kompatible Hilfsfunktionen (unveraendert zur frueheren API)
# ---------------------------------------------------------------------------

def to_wolfram_list(intervals) -> list:
    """Format fuer Wolfram: [[t_start, t_end, [c0, c1, c2]], ...]"""
    return [
        [float(iv.t_start), float(iv.t_end),
         [float(c) for c in iv.ed_coeffs]]
        for iv in intervals
    ]


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