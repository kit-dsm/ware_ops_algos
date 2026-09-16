"""Generic validator against Wolfram-style reference values.

This module mirrors the checks from validate_a128 but is instance-agnostic.
It supports:
- coefficient checks per interval
- E[D] spot checks per interval
- interval cost contribution checks (Pg/Kg/EDges/J)
- optional Wolfram NMinimize check
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from math import exp
from scipy.optimize import brentq, minimize_scalar
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from . import interval_builder as ib
    from .models import WarehouseInstance
except ImportError:
    import interval_builder as ib
    from models import WarehouseInstance

try:
    from wolframclient.evaluation import WolframLanguageSession
    from wolframclient.language import wl
    HAS_WOLFRAM = True
except Exception:
    HAS_WOLFRAM = False
    WolframLanguageSession = None
    wl = None


KERNEL_PATH = "C:/Program Files/Wolfram Research/Wolfram/14.3/WolframKernel.exe"
SCRIPT_PATH = (
    "C:/Users/ya5909/Git-Projekte/project_4D4L/"
    "2_Stochastic_Waiting/Algorithms/analytic_progress/Calculate_Intervals/analytic_wait_dynamic.wl"
)
TOL = 1e-5


@dataclass(frozen=True)
class ValidationReference:
    wait: float
    delta_exp: float
    expected_coeffs: List[Tuple[str, float, float, float, float, float]]
    spot_checks: Dict[str, List[Tuple[float, float]]]
    ed_i0: float
    pg_refs: Dict[str, float]
    kg_refs: Dict[str, float]
    # J at fixed 'wait' (used by cost table validation)
    ref_j: Optional[float] = None
    # Optional optimizer references for NMinimize (can differ from fixed wait case)
    ref_opt_wait: Optional[float] = None
    ref_opt_j: Optional[float] = None
    n_picks_after: int = 3


def eval_poly(coeffs: Tuple[float, float, float], s: float) -> float:
    c0, c1, c2 = coeffs
    return c0 + c1 * s + c2 * s * s


def _interval_key(iv) -> str:
    return f"{iv.phase_label}:{int(iv.t_start)}-{int(iv.t_end)}"


def test_coefficients(intervals, expected_coeffs) -> bool:
    print("=" * 70)
    print("Test 1: Coefficients")
    print("=" * 70)
    ok_all = True
    for exp, iv in zip(expected_coeffs, intervals):
        _, _, _, c0_ref, c1_ref, c2_ref = exp
        c0, c1, c2 = iv.ed_coeffs
        for name, a, b in (("c0", c0_ref, c0), ("c1", c1_ref, c1), ("c2", c2_ref, c2)):
            ok = abs(a - b) < TOL
            ok_all = ok_all and ok
            print(f"{iv.phase_label:<26} {name:>2}: ref={a:10.6f} got={b:10.6f} {'OK' if ok else 'FAIL'}")
    print("Coefficient test:", "PASS" if ok_all else "FAIL")
    return ok_all


def test_spot(intervals, spot_checks) -> bool:
    print("\n" + "=" * 70)
    print("Test 2: E[D] spot checks")
    print("=" * 70)
    ok_all = True
    for iv in intervals:
        key = _interval_key(iv)
        if key not in spot_checks:
            continue
        for s, ref in spot_checks[key]:
            got = eval_poly(iv.ed_coeffs, s)
            ok = abs(ref - got) < 1e-6
            ok_all = ok_all and ok
            print(f"{key:<34} s={s:5.1f} ref={ref:10.6f} got={got:10.6f} {'OK' if ok else 'FAIL'}")
    print("Spot test:", "PASS" if ok_all else "FAIL")
    return ok_all


def _integral_poly_exp(c0: float, c1: float, c2: float, lam: float, dt: float) -> float:
    e = exp(-lam * dt)
    ld = lam * dt
    i0 = (1.0 - e) / lam
    i1 = (1.0 - e * (1.0 + ld)) / (lam ** 2)
    i2 = (2.0 - e * (2.0 + 2.0 * ld + ld ** 2)) / (lam ** 3)
    return c0 * i0 + c1 * i1 + c2 * i2


def _c_total(lam: float, intervals, T_end: float,
             n_picks_after: int = 3) -> float:
    """
    C(lambda) sodass J(wait) = wait + C(lambda)*exp(-lambda*wait).

    Herleitung:
      J(wait) = wait + sum_i ∫_{wait+ta_i}^{wait+tb_i} E[D_i](t - wait - ta_i) f_T(t) dt
                      + ECostAfter * P(T >= wait + T_end)

    Nach Substitution t = wait + ta_i + u ergibt sich:
      C_i(lambda) = lambda * exp(-lambda*ta_i) * ∫_0^{dt_i} E[D_i](u) * exp(-lambda*u) du

      C(lambda)   = sum_i C_i(lambda) + ECostAfter * exp(-lambda*T_end)
    """
    e_cost_after = n_picks_after / lam + T_end
    c_sum = 0.0
    for iv in intervals:
        dt = iv.t_end - iv.t_start
        c0, c1, c2 = iv.ed_coeffs
        integral = _integral_poly_exp(c0, c1, c2, lam, dt)
        c_sum += lam * exp(-lam * iv.t_start) * integral
    c_sum += e_cost_after * exp(-lam * T_end)
    return c_sum


def compute_J_for_wait(lam: float,
                       wait: float,
                       intervals,
                       T_end: float,
                       n_picks_after: int = 3) -> float:
    """
    Berechnet J(wait) = wait + EDi0*(1-exp(-lam*wait)) + C_routes(lam)*exp(-lam*wait)

    EDi0 ist der erwartete Umweg waehrend des Wartens am Depot (Intervall I_0 = [0, wait]).
    C_routes(lam) enthaelt nur die Routenintervalle I_1..I_n und KgAfter.
    Vollstaendige Formel gemaess Wolfram:
      J = wait + KgDi0 + sum_i KgDi_i + KgAfter
    mit KgDi0 = EDi0 * (1 - exp(-lam*wait)).
    """
    C = _c_total(lam, intervals, T_end, n_picks_after=n_picks_after)
    ed_i0 = _ed_i0_from_intervals(intervals)
    kg_di0 = ed_i0 * (1.0 - exp(-lam * wait))
    return wait + kg_di0 + C * exp(-lam * wait)


def _dj_at_zero(lam: float,
                intervals,
                T_end: float,
                n_picks_after: int = 3) -> float:
    """J'(0) fuer J(wait)=wait+EDi0*(1-exp(-lam*wait))+C*exp(-lam*wait)."""
    C = _c_total(lam, intervals, T_end, n_picks_after=n_picks_after)
    ed_i0 = _ed_i0_from_intervals(intervals)
    return 1.0 - lam * (C - ed_i0)


def find_critical_delta(intervals,
                        T_end: float,
                        n_picks_after: int = 3,
                        delta_lo: float = 1.0,
                        delta_hi: float = 500.0,
                        tol: float = 1e-6) -> float:
    """Findet delta* = 1/lambda* mit J'(0)=0."""
    lam_lo = 1.0 / delta_hi
    lam_hi = 1.0 / delta_lo

    f_lo = _dj_at_zero(lam_lo, intervals, T_end, n_picks_after=n_picks_after)
    f_hi = _dj_at_zero(lam_hi, intervals, T_end, n_picks_after=n_picks_after)

    if f_lo * f_hi > 0:
        raise ValueError(
            "Kein Vorzeichenwechsel fuer J'(0) in "
            f"[delta_lo={delta_lo}, delta_hi={delta_hi}] "
            f"(f_lo={f_lo:.6f}, f_hi={f_hi:.6f})."
        )

    lam_star = brentq(
        lambda lam: _dj_at_zero(lam, intervals, T_end, n_picks_after=n_picks_after),
        lam_lo,
        lam_hi,
        xtol=tol * lam_lo,
        rtol=tol,
    )
    return 1.0 / lam_star

def _ed_i0_from_intervals(intervals) -> float:
    """
    EDi0 = erwarteter Umweg am Depot (Routenstart, s=0 der Approach-Phase).

    Singulaere, generische Quelle: c0 des ERSTEN Intervalls
    (phase_1_to_first_entry). Der Depotpunkt ist physikalisch identisch mit
    s=0 der Approach-Phase, daher EDi0 == phase1.c0. Identisch zur Wolfram-
    Seite (EDi0FromIntervals -> intervals[[1, 3, 1]]); so nutzen Python und
    Wolfram dieselbe Groesse statt zweier getrennter Quellen.
    """
    if not intervals:
        return 0.0
    return float(intervals[0].ed_coeffs[0])

def compute_cost_terms(intervals, ref: ValidationReference) -> Dict[str, float]:
    lam = 1.0 / ref.delta_exp
    t_end = max(iv.t_end for iv in intervals)
    ed_i0 = _ed_i0_from_intervals(intervals)
    out: Dict[str, float] = {}

    # I_0 waiting interval [0, wait]
    out["Pg0"] = 1.0 - exp(-lam * ref.wait)
    out["KgDi0"] = ed_i0 * out["Pg0"]

    # Route intervals I_1..I_n (based on built intervals)
    for idx, iv in enumerate(intervals, start=1):
        dt = iv.t_end - iv.t_start
        c0, c1, c2 = iv.ed_coeffs
        out[f"Pg{idx}"] = exp(-lam * (ref.wait + iv.t_start)) - exp(-lam * (ref.wait + iv.t_end))
        out[f"KgDi{idx}"] = lam * exp(-lam * (ref.wait + iv.t_start)) * _integral_poly_exp(c0, c1, c2, lam, dt)

    e_cost_after = (ref.n_picks_after / lam) + t_end
    out["PgAfter"] = exp(-lam * (ref.wait + t_end))
    out["KgAfter"] = e_cost_after * out["PgAfter"]

    out["Pges"] = sum(v for k, v in out.items() if k.startswith("Pg"))
    out["EDges"] = sum(v for k, v in out.items() if k.startswith("Kg"))
    out["J"] = out["EDges"] + ref.wait
    return out


def test_costs(intervals, ref: ValidationReference, tol: float = 2e-3) -> bool:
    print("\n" + "=" * 70)
    print("Test 3: Cost contributions")
    print("=" * 70)
    got = compute_cost_terms(intervals, ref)

    checks: Dict[str, float] = {}
    checks.update(ref.pg_refs)
    checks.update(ref.kg_refs)
    if ref.ref_j is not None:
        checks["J"] = ref.ref_j

    n = len(intervals)
    keys = (["Pg0"] + [f"Pg{i}" for i in range(1, n + 1)] + ["PgAfter", "Pges"]
            + ["KgDi0"] + [f"KgDi{i}" for i in range(1, n + 1)]
            + ["KgAfter", "EDges", "J"])

    ok_all = True

    for k in keys:
        v_got = got.get(k)
        if v_got is None:
            continue
        if k in checks:
            v_ref = checks[k]
            ok = abs(v_ref - v_got) <= tol
            ok_all = ok_all and ok
            print(f"{k:<8} ref={v_ref:12.6f} got={v_got:12.6f} {'OK' if ok else 'FAIL'}")
        else:
            print(f"{k:<8} {'(kein Ref)':>12} got={v_got:12.6f}")

    print("Cost test:", "PASS" if ok_all else "FAIL")
    return ok_all


def test_wolfram(intervals, ref: ValidationReference) -> Optional[bool]:
    print("\n" + "=" * 70)
    print("Test 4b: Wolfram NMinimize")
    print("=" * 70)

    if not HAS_WOLFRAM:
        print("Wolfram client unavailable -> skipped")
        return None

    lam = 1.0 / ref.delta_exp
    t_end = max(iv.t_end for iv in intervals)
    e_cost_after = (ref.n_picks_after / lam) + t_end
    wait_sym = wl.Symbol("wait")
    wl_intervals = ib.to_wolfram_list(intervals)

    session = WolframLanguageSession(KERNEL_PATH)
    try:
        session.start()
        session.evaluate(wl.Get(SCRIPT_PATH))
        analytic_j = session.evaluate(
            wl.WarehouseOptimization.GetAnalyticJ(
                float(lam), wl_intervals, float(t_end), float(e_cost_after)
            )
        )
        res = session.evaluate(wl.NMinimize([analytic_j, wl.GreaterEqual(wait_sym, 0)], wait_sym))
        j_opt = float(session.evaluate(wl.N(res[0])))
        w_opt = float(session.evaluate(wl.N(wl.ReplaceAll(wait_sym, res[1]))))

        ok = True
        if ref.ref_opt_wait is not None:
            ok_wait = abs(w_opt - ref.ref_opt_wait) < 1e-3
            ok = ok and ok_wait
            print(f"wait*: ref={ref.ref_opt_wait:.6f} got={w_opt:.6f} {'OK' if ok_wait else 'FAIL'}")
        if ref.ref_opt_j is not None:
            ok_j = abs(j_opt - ref.ref_opt_j) < 5e-2
            ok = ok and ok_j
            print(f"J*:    ref={ref.ref_opt_j:.6f} got={j_opt:.6f} {'OK' if ok_j else 'FAIL'}")
        return ok
    finally:
        try:
            session.terminate()
        except Exception:
            pass


def test_critical_delta(intervals,
                        T_end: float,
                        ref: ValidationReference,
                        ref_delta_star: Optional[float] = None,
                        tol: float = 0.1) -> Optional[bool]:
    print("\n" + "=" * 70)
    print("Test 5a: Kritischer Schwellwert delta* (Python)")
    print("=" * 70)
    try:
        delta_star = find_critical_delta(
            intervals,
            T_end,
            n_picks_after=ref.n_picks_after
        )
    except ValueError as exc:
        print(f"Uebersprungen: {exc}")
        return None

    lam_star = 1.0 / delta_star
    dj_star = _dj_at_zero(lam_star, intervals, T_end, n_picks_after=ref.n_picks_after)
    rel = 0.03
    delta_probe_lo = delta_star * (1.0 - rel)
    delta_probe_hi = delta_star * (1.0 + rel)

    dj_left = _dj_at_zero(1.0 / delta_probe_lo, intervals, T_end,
                          n_picks_after=ref.n_picks_after)
    dj_right = _dj_at_zero(1.0 / delta_probe_hi, intervals, T_end,
                           n_picks_after=ref.n_picks_after)

    print(f"delta*             : {delta_star:.6f}")
    print(f"lambda*            : {lam_star:.8f}")
    print(f"J'(0) bei delta*   : {dj_star:+.3e}")
    print(f"Sonde links  ({delta_probe_lo:.4f}): J'(0)={dj_left:+.6f}  (soll > 0 => wait*=0)")
    print(f"Sonde rechts ({delta_probe_hi:.4f}): J'(0)={dj_right:+.6f}  (soll < 0 => wait*>0)")

    ok = (abs(dj_star) < 1e-6) and (dj_left > 0.0) and (dj_right < 0.0)
    if ref_delta_star is not None:
        ok_ref = abs(delta_star - ref_delta_star) < tol
        ok = ok and ok_ref
        print(f"delta*-Ref         : {ref_delta_star:.6f} {'OK' if ok_ref else 'FAIL'}")
    print("Schwellwert-Test (Py):", "PASS" if ok else "FAIL")
    return ok


def test_critical_lambda_wolfram(intervals,
                                 T_end: float,
                                 ref: ValidationReference,
                                 lambda_guess: float = 0.05,
                                 ref_delta_star: Optional[float] = None,
                                 tol: float = 0.1) -> Optional[bool]:
    print("\n" + "=" * 70)
    print("Test 5b: Kritischer Schwellwert lambda* (Wolfram)")
    print("=" * 70)
    if not HAS_WOLFRAM:
        print("Wolfram client unavailable -> skipped")
        return None

    wl_intervals = ib.to_wolfram_list(intervals)
    session = WolframLanguageSession(KERNEL_PATH)
    try:
        session.start()
        session.evaluate(wl.Get(SCRIPT_PATH))
        lambda_star = session.evaluate(
            wl.WarehouseOptimization.FindCriticalLambda(
                wl_intervals,
                float(T_end),
                int(ref.n_picks_after),
                float(lambda_guess),
            )
        )
        lambda_star = float(session.evaluate(wl.N(lambda_star)))
        delta_star = 1.0 / lambda_star
        dj0 = _dj_at_zero(lambda_star, intervals, T_end, n_picks_after=ref.n_picks_after)

        print(f"lambda* (WL)       : {lambda_star:.8f}")
        print(f"delta*  (WL)       : {delta_star:.6f}")
        print(f"J'(0) bei lambda*  : {dj0:+.3e}")

        ok = abs(dj0) < 1e-6
        if ref_delta_star is not None:
            ok_ref = abs(delta_star - ref_delta_star) < tol
            ok = ok and ok_ref
            print(f"delta*-Ref         : {ref_delta_star:.6f} {'OK' if ok_ref else 'FAIL'}")

        print("Schwellwert-Test (WL):", "PASS" if ok else "FAIL")
        return ok
    finally:
        try:
            session.terminate()
        except Exception:
            pass


def test_multi_delta(intervals,
                     T_end: float,
                     ref: ValidationReference,
                     deltas: List[float],
                     skip_wolfram: bool = True) -> Optional[bool]:
    print("\n" + "=" * 70)
    print("Test 6: Delta-Sweep (mehrere delta-Werte)")
    print("=" * 70)

    wl_intervals = ib.to_wolfram_list(intervals)
    wait_sym = wl.Symbol("wait") if HAS_WOLFRAM else None
    session = None
    ok_all = True

    if not skip_wolfram and HAS_WOLFRAM:
        session = WolframLanguageSession(KERNEL_PATH)
        session.start()
        session.evaluate(wl.Get(SCRIPT_PATH))

    try:
        for delta in deltas:
            lam = 1.0 / delta

            def j_py(w: float) -> float:
                if w < 0:
                    return float("inf")
                return compute_J_for_wait(
                    lam,
                    w,
                    intervals,
                    T_end,
                    n_picks_after=ref.n_picks_after
                )

            res = minimize_scalar(j_py, bounds=(0.0, 200.0), method="bounded")
            w_py = float(res.x)
            j_py_star = float(res.fun)
            if w_py < 1e-5:
                w_py = 0.0

            if session is None:
                print(f"delta={delta:7.3f} | wait*_py={w_py:8.4f} | J*_py={j_py_star:10.6f}")
                continue

            e_cost_after = (ref.n_picks_after / lam) + T_end
            analytic_j = session.evaluate(
                wl.WarehouseOptimization.GetAnalyticJ(
                    float(lam), wl_intervals, float(T_end), float(e_cost_after)
                )
            )
            res_wl = session.evaluate(wl.NMinimize([analytic_j, wl.GreaterEqual(wait_sym, 0)], wait_sym))
            j_wl = float(session.evaluate(wl.N(res_wl[0])))
            w_wl = float(session.evaluate(wl.N(wl.ReplaceAll(wait_sym, res_wl[1]))))
            if w_wl < 1e-5:
                w_wl = 0.0

            ok_wait = abs(w_py - w_wl) < 1e-3
            ok_j = abs(j_py_star - j_wl) < 5e-2
            ok_all = ok_all and ok_wait and ok_j
            mark = "OK" if (ok_wait and ok_j) else "FAIL"
            print(
                f"delta={delta:7.3f} | "
                f"wait*_py={w_py:8.4f} wait*_wl={w_wl:8.4f} | "
                f"J*_py={j_py_star:10.6f} J*_wl={j_wl:10.6f}  {mark}"
            )
    finally:
        if session is not None:
            try:
                session.terminate()
            except Exception:
                pass

    if session is None:
        return None
    print("Delta-Sweep:", "PASS" if ok_all else "FAIL")
    return ok_all


def run_validation_for_instance(inst: WarehouseInstance,
                                ref: ValidationReference,
                                skip_wolfram: bool = True,
                                deltas: Optional[List[float]] = None,
                                ref_delta_star: Optional[float] = None):
    intervals, T_end = ib.build_interval_data(inst)
    print("\nComputed intervals:")
    ib.print_intervals(intervals)

    ed_i0 = _ed_i0_from_intervals(intervals)
    ed_i0_ok = abs(ed_i0 - ref.ed_i0) < TOL
    print(f"\nEDi0 (generisch aus phase1.c0): {ed_i0:.6f}  "
          f"(Referenz {ref.ed_i0:.6f}) "
          f"{'OK' if ed_i0_ok else 'FAIL <- phase1.c0 weicht von Referenz ab'}")

    r1 = test_coefficients(intervals, ref.expected_coeffs)
    r2 = test_spot(intervals, ref.spot_checks)
    r3 = test_costs(intervals, ref)

    # Interne Python-J(wait)-Tests (C(lambda)-Form)
    lam = 1.0 / ref.delta_exp
    print("\n" + "=" * 70)
    print(f"Test 4a: internal analytic J(wait) via C(lambda) (lambda={lam:.6f})")
    print("=" * 70)
    # einige feste wait-Werte
    sample_waits = [0.0, 1.0, 5.0, 10.0, 15.0, 18.0, 20.0]
    for w in sample_waits:
        j_val = compute_J_for_wait(lam, w, intervals, T_end,
                                   n_picks_after=ref.n_picks_after)
        print(f"  J_py(wait={w:6.2f}) = {j_val:10.6f}")

    def j_py(w: float) -> float:
        if w < 0:
            return float("inf")
        return compute_J_for_wait(lam, w, intervals, T_end,
                                  n_picks_after=ref.n_picks_after)

    res = minimize_scalar(j_py, bounds=(0.0, 100.0), method="bounded")

    print(f"  -> internal Python optimum: wait* = {float(res.x):.6f}, "
          f"J* = {float(res.fun):.6f}")

    # optionaler Vergleich mit Referenzwert bei ref.wait (z.B. 18.78434)
    if ref.ref_j is not None and ref.wait is not None:
        j_at_ref_wait = j_py(ref.wait)
        diff = abs(j_at_ref_wait - ref.ref_j)
        ok_j_at_wait = diff <= 5e-2  # wie in Wolfram-Test, relativ großzügig
        print(f"  J_py(wait={ref.wait:.3f}) vs ref_j={ref.ref_j:.6f}: "
              f"{j_at_ref_wait:.6f} (diff={diff:.6f}) "
              f"{'OK' if ok_j_at_wait else 'FAIL'}")

    r4 = None if skip_wolfram else test_wolfram(intervals, ref)
    r5 = test_critical_delta(intervals, T_end, ref, ref_delta_star=ref_delta_star)
    r6 = None if skip_wolfram else test_critical_lambda_wolfram(
        intervals,
        T_end,
        ref,
        lambda_guess=1.0 / ref.delta_exp,
        ref_delta_star=ref_delta_star,
    )
    r7 = test_multi_delta(
        intervals,
        T_end,
        ref,
        deltas=deltas or [5.0, 28.8, 50.0, 75.0, 100.0],
        skip_wolfram=skip_wolfram,
    )

    print("\n" + "=" * 70)
    print("Summary")
    print(f"coefficients: {'PASS' if r1 else 'FAIL'}")
    print(f"spots:        {'PASS' if r2 else 'FAIL'}")
    print(f"costs:        {'PASS' if r3 else 'FAIL'}")
    if r4 is not None:
        print(f"wolfram:      {'PASS' if r4 else 'FAIL'}")
    if r5 is not None:
        print(f"delta* (py):  {'PASS' if r5 else 'FAIL'}")
    if r6 is not None:
        print(f"delta* (wl):  {'PASS' if r6 else 'FAIL'}")
    if r7 is not None:
        print(f"delta sweep:  {'PASS' if r7 else 'FAIL'}")
    print("=" * 70)

    return {
        "coefficients": r1,
        "spots": r2,
        "costs": r3,
        "wolfram": r4,
        "critical_delta_py": r5,
        "critical_delta_wl": r6,
        "delta_sweep": r7,
    }


def build_case_a23() -> Tuple[WarehouseInstance, ValidationReference]:
    inst = WarehouseInstance(
        M=8,
        N_L=17,
        w=1.0,
        L=17.0,
        A=[2, 3],
        n_list=[2, 1],
        v=1.0,
        t_p=0.0,
        P=[],
        arrival_times=[],
    )

    # Expected entries for the 5 route intervals produced by builder for this case.
    expected_coeffs = [
        ("phase_1_to_first_entry", 0.0, 3.0, 29.25, 0.0, 0.0),
        ("phase_2_vertical", 3.0, 20.0, 29.25, 0.0, 1.0/136.0),
        ("phase_3_horizontal", 20.0, 21.0, 31.375, 1.5, 0.0),
        ("phase_2_vertical", 21.0, 38.0, 32.875, 0.25, 1.0/136),
        ("phase_4_after_last_aisle", 38.0, 40.0, 153.0/4, 25.0/12, 7.0/48),
    ]

    # Spot checks keyed by built interval labels.
    spot_checks = {
        "phase_1_to_first_entry:0-3": [(0.0, 29.25), (3.0, 29.25)],
        "phase_2_vertical:3-20": [(0.0, 29.25), (17.0, 31.375)],
        "phase_3_horizontal:20-21": [(0.0, 31.375), (1.0, 32.875)],
        "phase_2_vertical:21-38": [(0.0, 32.875), (17.0, 39.25)],
        "phase_4_after_last_aisle:38-40": [(1.0, 1943.0 / 48.0)],
    }

    pg_refs = {
        "Pg0":    0.479119,
        "Pg1":    0.051528,
        "Pg2":    0.209250,
        "Pg3":    0.008876,
        "Pg4":    0.112003,
        "Pg5":    0.009340,
        "PgAfter": 0.129883,
        "Pges":   1.0,
    }

    kg_refs = {
        "KgDi0":   14.014236,
        "KgDi1":    1.507196,
        "KgDi2":    6.247461,
        "KgDi3":    0.285114,
        "KgDi4":    3.964764,
        "KgDi5":    0.378281,
        "KgAfter": 16.417184,
        "EDges":   42.814236,
    }

    ref = ValidationReference(
        wait=18.78434,
        delta_exp=28.8,
        expected_coeffs=expected_coeffs,
        spot_checks=spot_checks,
        ed_i0=29.25,
        pg_refs=pg_refs,
        kg_refs=kg_refs,
        ref_j=61.5986,
        # Optimization reference for full J(wait) including KgDi0.
        ref_opt_wait=0.0,
        ref_opt_j=55.290959,
        n_picks_after=3,
    )
    return inst, ref


def build_case_a128() -> Tuple[WarehouseInstance, ValidationReference]:
    """
    Instanz A = (1, 2, 8), M=8, L=17, k=3 (ungerade -> Return-Gasse).

    Referenzen = analytisch bekannte Werte aus validate_a128. Hier als Soll,
    um zu pruefen, ob der NEUE Sampling-Builder sie reproduziert - speziell:
      - phase_3_horizontal 25-31 (8->2) mit FUENF Candidate-Ueberquerungen,
      - die ret_up / ret_down-Phasen der Return-Gasse.
    Optimierungsreferenz: J* = 14.3593 bei wait* = 0 (delta = 28.8).
    """
    inst = WarehouseInstance(
        M=8, N_L=17, w=1.0, L=17.0,
        A=[1, 2, 8],
        n_list=[1, 1, 1],
        v=1.0, t_p=0.0,
        P=[], arrival_times=[],
    )

    # Reihenfolge identisch zur Builder-Ausgabe (8 Routenintervalle).
    expected_coeffs = [
        ("phase_1_to_first_entry",    0.0,  8.0,  0.0,      0.0,    0.0     ),
        ("phase_2_vertical",          8.0, 25.0,  0.0,      0.0,    1.0/136 ),
        ("phase_3_horizontal",       25.0, 31.0,  17.0/8,   1.0/8,  1.0/8   ),
        ("phase_2_vertical",         31.0, 48.0,  59.0/8,   1.0/4,  1.0/136 ),
        ("phase_3_horizontal",       48.0, 49.0,  13.75,    1.75,   0.0     ),
        ("phase_n_2_ret_up",         49.0, 66.0,  15.5,     0.0,    0.0     ),
        ("phase_n_1_ret_down",       66.0, 83.0,  147.0/4,  0.0,    1.0/136 ),
        ("phase_4_after_last_aisle", 83.0, 84.0,  55.875,     2.0,    0.0     ),
    ]

    spot_checks = {
        "phase_2_vertical:8-25":          [(0.0, 0.0),     (17.0, 289.0/136)           ],
        "phase_3_horizontal:25-31":       [(0.0, 17.0/8),  (6.0,  59.0/8)              ],
        "phase_2_vertical:31-48":         [(0.0, 59.0/8),  (17.0, 13.75)               ],
        "phase_3_horizontal:48-49":       [(0.0, 13.75),   (1.0,  15.5)                ],
        "phase_n_2_ret_up:49-66":         [(0.0, 15.5),    (17.0, 15.5)                ],
        "phase_n_1_ret_down:66-83":       [(0.0, 147.0/4), (17.0, 147.0/4 + 289.0/136) ],
        "phase_4_after_last_aisle:83-84": [(0.0, 55.875),    (1.0,  57.875)                ],
    }

    ref = ValidationReference(
        wait=0.0,
        delta_exp=28.8,
        expected_coeffs=expected_coeffs,
        spot_checks=spot_checks,
        # a128: alle Candidates {3..7} liegen auf der Route ("front") -> D_front=0
        # am Depot -> phase1.c0 = 0. (Anders als a23, wo Candidates abseits liegen.)
        ed_i0=0.0,
        pg_refs={"Pges": 1.0},   # nur die Wahrscheinlichkeits-Invariante als Check
        kg_refs={},              # keine Per-Intervall-Kostenreferenzen vorhanden
        ref_j=14.3858,           # J(wait=0) == Optimum bei delta=28.8
        ref_opt_wait=0.0,
        ref_opt_j=14.3858,
        n_picks_after=3,
    )
    return inst, ref

def build_case_a235() -> Tuple[WarehouseInstance, ValidationReference]:
    """
    Instanz A = (2, 3, 5), M=8, L=17, k=3 (ungerade -> Return-Gasse 2).

    Referenzen aus der unabhaengigen Mathematica-Rechnung (wait=1, delta=28.8,
    J=18.0449). Im Gegensatz zu a128 (a_ret=1, ganz links) hat diese Instanz:
      - eine unbesuchte Gasse (1) LINKS der Return-Gasse -> Front-Detour
        (d_front=34) in ret_down UND Phase 4, also ed_i0 = phase1.c0 = 1.5,
      - die Return-Gasse korrekt im Phase-4-Backtrack (c0_phase4 = 53.0625).
    Beide Effekte fehlten in der alten analytischen a128-Referenz; hier dienen
    sie als Gegenprobe, dass der generische Builder das konsistente Modell
    reproduziert. Jede Kostenkomponente KgDi0..KgDi8/KgAfter deckt sich mit
    der Mathematica-Tabelle.
    Optimierungsreferenz: wait* = 0, J* = 17.594140 (delta = 28.8).
    """
    inst = WarehouseInstance(
        M=8, N_L=16, w=1.0, L=17.0,
        A=[2, 3, 5],
        n_list=[1, 1, 1],
        v=1.0, t_p=0.0,
        P=[], arrival_times=[],
    )

    # Reihenfolge identisch zur Builder-Ausgabe (8 Routenintervalle).
    expected_coeffs = [
        ("phase_1_to_first_entry",    0.0,  5.0,  1.5,      0.0,      0.0      ),
        ("phase_2_vertical",          5.0, 22.0,  1.5,      0.0,      1.0/136  ),
        ("phase_3_horizontal",       22.0, 24.0,  3.125,    1.0625,   0.15625  ),
        ("phase_2_vertical",         24.0, 41.0,  5.875,    0.25,     1.0/136  ),
        ("phase_3_horizontal",       41.0, 42.0,  12.25,    1.5,      0.0      ),
        ("phase_n_2_ret_up",         42.0, 59.0,  13.75,    0.0,      0.0      ),
        ("phase_n_1_ret_down",       59.0, 76.0,  35.0,     0.0,      1.0/136  ),
        ("phase_4_after_last_aisle", 76.0, 78.0,  53.0625,  2.16875,  0.11875  ),
    ]

    # Stuetzwerte = E[D] an den Phasengrenzen (aus Mathematica/continuous,
    # unabhaengig vom Fit). s=0 jeder Phase = deren c0.
    spot_checks = {
        "phase_1_to_first_entry:0-5":     [(0.0, 1.5),     (5.0,  1.5)     ],
        "phase_2_vertical:5-22":          [(0.0, 1.5),     (17.0, 3.625)   ],
        "phase_3_horizontal:22-24":       [(0.0, 3.125),   (2.0,  5.875)   ],
        "phase_2_vertical:24-41":         [(0.0, 5.875),   (17.0, 12.25)   ],
        "phase_3_horizontal:41-42":       [(0.0, 12.25),   (1.0,  13.75)   ],
        "phase_n_2_ret_up:42-59":         [(0.0, 13.75),   (17.0, 13.75)   ],
        "phase_n_1_ret_down:59-76":       [(0.0, 35.0),    (17.0, 37.125)  ],
        "phase_4_after_last_aisle:76-78": [(0.0, 53.0625), (2.0,  57.875)  ],
    }

    # Per-Intervall-Kostenreferenzen aus Mathematica (wait=1). Builder-Index
    # KgDi6=ret_up(I_6a), KgDi7=ret_down(I_6b), KgDi8=phase4(I_7).
    kg_refs = {
        "KgDi0":   0.051189,
        "KgDi1":   0.230906,
        "KgDi2":   0.762498,
        "KgDi3":   0.132215,
        "KgDi4":   1.571740,
        "KgDi5":   0.103167,
        "KgDi6":   1.377350,
        "KgDi7":   1.976580,
        "KgDi8":   0.256283,
        "KgAfter": 10.583000,
        "EDges":   17.044900,
    }

    ref = ValidationReference(
        wait=1.0,
        delta_exp=28.8,
        expected_coeffs=expected_coeffs,
        spot_checks=spot_checks,
        ed_i0=1.5,                 # phase1.c0 = p4 * D_front am Depot = (5/8)*2.4
        pg_refs={"Pges": 1.0},
        kg_refs=kg_refs,
        ref_j=18.0449,             # J(wait=1) == Mathematica-Referenz
        ref_opt_wait=0.0,
        ref_opt_j=17.594140,
        n_picks_after=3,
    )
    return inst, ref


def build_case_a2() -> Tuple[WarehouseInstance, ValidationReference]:
    """
    Instanz A = (2), M=8, L=17, k=1 (ungerade -> Return-Gasse 2).

    Referenzen aus der manuellen Mathematica-Rechnung (wait ~ 0, delta=28.8,
    J=44.7313). Intervallzuordnung im Builder:
      I_1: phase_1_to_first_entry [0,2]
      I_2: phase_n_2_ret_up      [2,19]
      I_3: phase_n_1_ret_down    [19,36]
      I_4: phase_4_after_last    [36,38]

    Hinweis: Der Validator verwendet EDi0 generisch als phase1.c0; bei wait~0
    bleibt KgDi0 numerisch trotzdem ~0 und konsistent zur Mathematica-Tabelle.
    """
    inst = WarehouseInstance(
        M=8, N_L=17, w=1.0, L=17.0,
        A=[2],
        n_list=[1],
        v=1.0, t_p=0.0,
        P=[], arrival_times=[],
    )

    expected_coeffs = [
        ("phase_1_to_first_entry",    0.0,  2.0,  21.0 / 4.0,   0.0,       0.0),
        ("phase_n_2_ret_up",          2.0, 19.0,  21.0 / 4.0,   0.0,       0.0),
        ("phase_n_1_ret_down",       19.0, 36.0,  35.0,         0.0,       1.0 / 136.0),
        ("phase_4_after_last_aisle", 36.0, 38.0,  145.0 / 4.0,  221.0 / 112.0, 19.0 / 112.0),
    ]

    spot_checks = {
        "phase_1_to_first_entry:0-2":     [(0.0, 21.0 / 4.0), (2.0, 21.0 / 4.0)],
        "phase_n_2_ret_up:2-19":          [(0.0, 21.0 / 4.0), (17.0, 21.0 / 4.0)],
        "phase_n_1_ret_down:19-36":       [(0.0, 35.0),       (17.0, 37.125)],
        "phase_4_after_last_aisle:36-38": [(0.0, 36.25),      (2.0, 40.875)],
    }

    pg_refs = {
        "Pg0":     0.0,
        "Pg1":     0.067088,
        "Pg2":     0.415917,
        "Pg3":     0.230490,
        "Pg4":     0.019221,
        "PgAfter": 0.267284,
        "Pges":    1.0,
    }

    kg_refs = {
        "KgDi0":   0.0,
        "KgDi1":   0.352212,
        "KgDi2":   2.183560,
        "KgDi3":   8.206930,
        "KgDi4":   0.738521,
        "KgAfter": 33.250100,
        "EDges":   44.731300,
    }

    ref = ValidationReference(
        wait=0,
        delta_exp=28.8,
        expected_coeffs=expected_coeffs,
        spot_checks=spot_checks,
        ed_i0=21.0 / 4.0,
        pg_refs=pg_refs,
        kg_refs=kg_refs,
        ref_j=44.7313,
        ref_opt_wait=9.085034,
        ref_opt_j=43.135035,
        n_picks_after=3,
    )
    return inst, ref


def build_case_a2_w1() -> Tuple[WarehouseInstance, ValidationReference]:
    """
    Instanz A = (2), gleiche Geometrie wie a2, aber Referenz mit wait=1.

    Kostenreferenzen stammen aus der manuellen Mathematica-Rechnung mit EDi0 = 21/4 im Warteintervall I_0.
    """
    inst = WarehouseInstance(
        M=8, N_L=17, w=1.0, L=17.0,
        A=[2],
        n_list=[1],
        v=1.0, t_p=0.0,
        P=[], arrival_times=[],
    )

    expected_coeffs = [
        ("phase_1_to_first_entry",    0.0,  2.0,  21.0 / 4.0,   0.0,       0.0),
        ("phase_n_2_ret_up",          2.0, 19.0,  21.0 / 4.0,   0.0,       0.0),
        ("phase_n_1_ret_down",       19.0, 36.0,  35.0,         0.0,       1.0 / 136.0),
        ("phase_4_after_last_aisle", 36.0, 38.0,  145.0 / 4.0,  221.0 / 112.0, 19.0 / 112.0),
    ]

    spot_checks = {
        "phase_1_to_first_entry:0-2":     [(0.0, 21.0 / 4.0), (2.0, 21.0 / 4.0)],
        "phase_n_2_ret_up:2-19":          [(0.0, 21.0 / 4.0), (17.0, 21.0 / 4.0)],
        "phase_n_1_ret_down:19-36":       [(0.0, 35.0),       (17.0, 37.125)],
        "phase_4_after_last_aisle:36-38": [(0.0, 36.25),      (2.0, 40.875)],
    }

    pg_refs = {
        "Pg0":     0.0341263,
        "Pg1":     0.0647986,
        "Pg2":     0.4017230,
        "Pg3":     0.2226240,
        "Pg4":     0.0185651,
        "PgAfter": 0.2581620,
        "Pges":    1.0,
    }

    kg_refs = {
        "KgDi0":   0.179163,
        "KgDi1":   0.340193,
        "KgDi2":   2.109050,
        "KgDi3":   7.926860,
        "KgDi4":   0.713318,
        "KgAfter": 32.115400,
        "EDges":   43.384000,
    }

    ref = ValidationReference(
        wait=1.0,
        delta_exp=28.8,
        expected_coeffs=expected_coeffs,
        spot_checks=spot_checks,
        ed_i0=21.0 / 4.0,
        pg_refs=pg_refs,
        kg_refs=kg_refs,
        ref_j=44.3840,
        ref_opt_wait=9.085035,
        ref_opt_j=43.135035,
        n_picks_after=3,
    )
    return inst, ref


def parse_args():
    parser = argparse.ArgumentParser(description="Generic instance validator vs Wolfram references")
    parser.add_argument("--case", choices=["a23", "a128", "a235", "a2", "a2_w1"], default="a128",
                        help="Welche Instanz validiert wird (default: a23)")
    parser.add_argument("--skip-wolfram", action="store_true", help="Skip Wolfram NMinimize check")
    parser.add_argument("--deltas", type=str, default="5,28.8,50,75,100",
                        help="Comma-separated delta values for sweep")
    parser.add_argument("--ref-delta-star", type=float, default=None,
                        help="Optional reference value for critical delta*")
    return parser.parse_args()


def main():
    args = parse_args()
    builders = {
        "a23": build_case_a23,
        "a128": build_case_a128,
        "a235": build_case_a235,
        "a2": build_case_a2,
        "a2_w1": build_case_a2_w1,
    }
    inst, ref =builders[args.case]()
    deltas = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]
    run_validation_for_instance(
        inst,
        ref,
        skip_wolfram=args.skip_wolfram,
        deltas=deltas,
        ref_delta_star=args.ref_delta_star,
    )


if __name__ == "__main__":
    main()

