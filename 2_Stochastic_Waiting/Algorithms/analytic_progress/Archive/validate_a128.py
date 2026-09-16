"""
validate_a128.py  —  Validierungstest fuer Route A = (1, 2, 8)

Vergleicht die analytisch berechneten Polynom-Koeffizienten und die
Wolfram-Optimierung mit den bekannten Werten aus dem manuellen
Mathematica-Notebook (J* = 14.3593 bei delta = 28.8).

Testaufbau
----------
1. Koeffizienten-Check:
   Vergleicht c0, c1, c2 pro Intervall mit den analytisch erwarteten Werten.

2. E[D]-Punktprobe:
   Wertet das Polynom an t_start, t_mid, t_end jedes Intervalls aus und
   vergleicht mit den Mathematica-Formeln.

3. Wolfram-Optimierung:
   Prueft ob J* und wait* mit den bekannten Werten uebereinstimmen.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl

try:
    from Algorithms.analytic_progress.Calculate_Intervals import core as core_mod
    from .models import WarehouseInstance
    from Algorithms.analytic_progress.Archive import interval_builder_backup as ib
except ImportError:
    import core as core_mod
    from models import WarehouseInstance
    import interval_builder_backup as ib

KERNEL_PATH = "C:/Program Files/Wolfram Research/Wolfram/14.3/WolframKernel.exe"
SCRIPT_PATH = ("C:/Users/ya5909/Git-Projekte/project_4D4L/"
               "2_Stochastic_Waiting/Algorithms/analytic_progress/Calculate_Intervals/analytic_wait_dynamic.wl")


EXPECTED = [
    ("phase_1_to_first_entry",  0.0,  8.0,  0.0,        0.0,   0.0        ),
    ("phase_2_vertical",        8.0, 25.0,  0.0,        0.0,   1/136      ),
    ("phase_3_horizontal",     25.0, 31.0, 17/8,        1/8,   1/8        ),
    ("phase_2_vertical",       31.0, 48.0, 59/8,        1/4,   1/136      ),
    ("phase_3_horizontal",     48.0, 49.0, 13.75,       1.75,  0.0        ),
    ("phase_n_2_ret_up",       49.0, 66.0, 15.5,        0.0,   0.0        ),
    ("phase_n_1_ret_down",     66.0, 83.0, 147/4,       0.0,  1/136      ),
    ("phase_4_after_last_aisle",83.0,84.0, 55.875,        2.0,   0.0        ),
]

SPOT = {
    "phase_2_vertical:8-25":         [(0,  0.0),    (17, 289/136)         ],
    "phase_3_horizontal:25-31":      [(0,  17/8),   (6,  59/8)            ],
    "phase_2_vertical:31-48":        [(0,  59/8),   (17, 13.75)           ],
    "phase_3_horizontal:48-49":      [(0,  13.75),  (1,  15.5)            ],
    "phase_n_2_ret_up:49-66":        [(0,  15.5),   (17, 15.5)            ],
    "phase_n_1_ret_down:66-83":      [(0,  147/4),  (17, 147/4 + 289/136)],
    "phase_4_after_last_aisle:83-84":[(0,  42.0),   (1,  44.0)            ],
}

# Toleranz fuer Gleitkomma-Vergleiche
TOL = 1e-9


def make_instance() -> WarehouseInstance:

   #hier Instanzdaten ändern

    return WarehouseInstance(
        M=8, N_L=17, w=1.0, L=17.0, A=[1, 2, 8],
        n_list=[1, 1, 1], v=1.0, t_p=0.0,
        P=[], arrival_times=[],
    )


def eval_poly(coeffs, s):
    c0, c1, c2 = coeffs
    return c0 + c1*s + c2*s*s


# ---------------------------------------------------------------------------
# Test 1: Koeffizienten-Check
# ---------------------------------------------------------------------------

def test_coefficients(inst, intervals):
    print("=" * 70)
    print("Test 1: Polynom-Koeffizienten pro Intervall")
    print("=" * 70)

    header = f"{'Phase':<26} {'Param':>5}  {'Erwartet':>12}  {'Berechnet':>12}  {'OK'}"
    print(header)
    print("-" * 65)

    all_ok = True

    for exp, iv in zip(EXPECTED, intervals):
        label_exp, ts_exp, te_exp, c0_exp, c1_exp, c2_exp = exp
        c0, c1, c2 = iv.ed_coeffs

        for param, expected, computed in [
            ("c0", c0_exp, c0),
            ("c1", c1_exp, c1),
            ("c2", c2_exp, c2),
        ]:
            ok = abs(expected - computed) < TOL
            if not ok:
                all_ok = False
            mark = "✓" if ok else "✗  ← FEHLER"
            print(f"{iv.phase_label:<26} {param:>5}  {expected:>12.8f}  "
                  f"{computed:>12.8f}  {mark}")

    print()
    print("Koeffizienten-Test:", "BESTANDEN ✓" if all_ok else "FEHLGESCHLAGEN ✗")
    return all_ok


# ---------------------------------------------------------------------------
# Test 2: E[D]-Punktprobe
# ---------------------------------------------------------------------------

def test_ed_pointwise(inst, intervals):
    """
    Prueft E[D](s) an t_start, t_mid, t_end jedes Intervalls
    gegen die bekannten Mathematica-Formeln.

    Erwartete Randwerte aus der Herleitung:
      Phase 2 vert Gasse 8: E[D](0)=0, E[D](17)=17^2/136=289/136≈2.125
      Phase 3 horiz 8→2:    E[D](0)=17/8=2.125, E[D](6)=59/8=7.375
      Phase 4 vert Gasse 2: E[D](0)=59/8=7.375, E[D](17)=p2*40+p3*6=13.75
      Phase n_2_ret_up:     E[D]=15.5 (konstant)
      Phase 4_after_last:   E[D](0)=42, E[D](1)=44
    """
    # Bekannte Randwerte (t_start, s_val, expected_E_D)


    print("\n" + "=" * 70)
    print("Test 2: E[D]-Punktprobe an Intervallgrenzen")
    print("=" * 70)

    all_ok = True
    for iv in intervals:
        key = f"{iv.phase_label}:{int(iv.t_start)}-{int(iv.t_end)}"
        if key not in SPOT:
            continue
        for s, expected in SPOT[key]:
            computed = eval_poly(iv.ed_coeffs, s)
            ok = abs(expected - computed) < TOL
            if not ok:
                all_ok = False
            mark = "✓" if ok else "✗  ← FEHLER"
            print(f"  {iv.phase_label:<26} s={s:>4.1f}:  "
                  f"erwartet={expected:>8.5f}  berechnet={computed:>8.5f}  {mark}")

    print()
    print("Punktprobe-Test:", "BESTANDEN ✓" if all_ok else "FEHLGESCHLAGEN ✗")
    return all_ok


# ---------------------------------------------------------------------------
# Test 3: Wolfram-Optimierung
# ---------------------------------------------------------------------------

def test_wolfram(inst,
                intervals,
                T_end,
                delta_exp=28.8,
                ref_j=14.3593,   # None = kein Pass/Fail auf J*
                ref_wait=0.0,
                tol_j=0.05,
                tol_wait=0.001
                 ):
    print("\n" + "=" * 70)
    print(f"Test 3: Wolfram J(wait) Minimierung  (delta={delta_exp})")
    print("=" * 70)

    lambda_val   = 1.0 / delta_exp
    e_cost_after = 3.0 / lambda_val + T_end

    wl_intervals = ib.to_wolfram_list(intervals)
    wait_sym     = wl.Symbol("wait")

    session = WolframLanguageSession(KERNEL_PATH)
    try:
        try:
            session.start()
        except Exception as exc:
            print(f"  Uebersprungen: Wolfram-Kernel konnte nicht gestartet werden ({exc})")
            return None
        session.evaluate(wl.Get(SCRIPT_PATH))

        analytic_j = session.evaluate(
            wl.WarehouseOptimization.GetAnalyticJ(
                float(lambda_val),
                wl_intervals,
                float(T_end),
                float(e_cost_after),
            )
        )

        result = session.evaluate(
            wl.NMinimize([analytic_j, wl.GreaterEqual(wait_sym, 0)], wait_sym)
        )


        j_val = session.evaluate(wl.N(result[0]))
        w_val = session.evaluate(wl.N(wl.ReplaceAll(wait_sym, result[1])))

        if not isinstance(j_val, (int, float)):
            print(f"  FEHLER: Wolfram gibt kein numerisches Ergebnis.")
            print(f"  result = {result}")
            return False

        j_opt = float(j_val)
        wait_opt = float(w_val)

        if wait_opt < 1e-5:
            wait_opt = 0.0

        print(f"  wait*      : {wait_opt:.6f}  (Referenz: {ref_wait:.6f})  ", end="")
        print("✓" if abs(wait_opt - ref_wait) < tol_wait else "✗  ← FEHLER")

        if ref_j is not None:
            ok_j = abs(j_opt - ref_j) < tol_j
            print(f"  J*         : {j_opt:.6f}  (Referenz: {ref_j:.6f})  ", end="")
            print("✓" if ok_j else "✗  ← FEHLER")
            print(f"  |J* - Ref| : {abs(j_opt - ref_j):.6f}")
        else:
            ok_j = True
            print(f"  J*         : {j_opt:.6f}  (kein Referenzwert)")

        ok_wait = abs(wait_opt - ref_wait) < tol_wait
        passed = ok_j and ok_wait
        print()
        print("Wolfram-Test:", "BESTANDEN ✓" if passed else "FEHLGESCHLAGEN ✗")
        return passed

    finally:
        try:
            session.terminate()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Test 4: Kritischer Schwellwert delta* (Option B — rein analytisch)
# ---------------------------------------------------------------------------

from math import exp
from scipy.optimize import brentq


def _integral_poly_exp(c0: float, c1: float, c2: float,
                       lam: float, dt: float) -> float:
    """
    integral_0^dt (c0 + c1*u + c2*u^2) * exp(-lam*u) du

    Geschlossene Formen:
      I0 = (1 - exp(-lam*dt)) / lam
      I1 = (1 - exp(-lam*dt)*(1 + lam*dt)) / lam^2
      I2 = (2 - exp(-lam*dt)*(2 + 2*lam*dt + (lam*dt)^2)) / lam^3
    """
    e  = exp(-lam * dt)
    ld = lam * dt
    I0 = (1.0 - e) / lam
    I1 = (1.0 - e * (1.0 + ld)) / lam**2
    I2 = (2.0 - e * (2.0 + 2.0*ld + ld**2)) / lam**3
    return c0*I0 + c1*I1 + c2*I2


def _c_total(lam: float, intervals, T_end: float,
             n_picks_after: int = 3) -> float:
    """
    C(lambda) sodass J(wait) = wait + C(lambda)*exp(-lambda*wait).

      C_i(lambda) = lambda * exp(-lambda*t_start)
                    * integral_0^dt E[D](u)*exp(-lambda*u) du
      C(lambda)   = sum_i C_i + ECostAfter * exp(-lambda*T_end)
    """
    e_cost_after = n_picks_after / lam + T_end
    c_sum = 0.0
    for iv in intervals:
        dt  = iv.t_end - iv.t_start
        c0, c1, c2 = iv.ed_coeffs
        integral = _integral_poly_exp(c0, c1, c2, lam, dt)
        c_sum   += lam * exp(-lam * iv.t_start) * integral
    c_sum += e_cost_after * exp(-lam * T_end)
    return c_sum


def compute_J_for_wait(lam: float,
                       wait: float,
                       intervals,
                       T_end: float,
                       n_picks_after: int = 3) -> float:
    """
    Berechnet J(wait) = wait + C(lambda)*exp(-lambda*wait)
    fuer gegebene lambda, wait und Intervallstruktur.
    Verwendet dieselbe C(lambda)-Definition wie _c_total.
    """
    C = _c_total(lam, intervals, T_end, n_picks_after=n_picks_after)
    return wait + C * exp(-lam * wait)


def _dj_at_zero(lam: float, intervals, T_end: float) -> float:
    """
    J'(0) = 1 - lambda * C(lambda)

    > 0  =>  wait* = 0   (Randoptimum)
    < 0  =>  wait* > 0   (inneres Optimum)
    = 0  =>  kritischer Schwellwert
    """
    return 1.0 - lam * _c_total(lam, intervals, T_end)


def find_critical_delta(intervals, T_end: float,
                        delta_lo: float = 1.0,
                        delta_hi: float = 500.0,
                        tol: float = 1e-6) -> float:
    """
    Findet delta* = 1/lambda* mit J'(0) = 0 via scipy.brentq.
    Gibt delta* zurueck.
    """
    lam_lo = 1.0 / delta_hi
    lam_hi = 1.0 / delta_lo

    f_lo = _dj_at_zero(lam_lo, intervals, T_end)
    f_hi = _dj_at_zero(lam_hi, intervals, T_end)

    if f_lo * f_hi > 0:
        raise ValueError(
            f"Kein Vorzeichenwechsel in [delta_lo={delta_lo}, delta_hi={delta_hi}]. "
            f"J'(0)={f_hi:.4f} bei delta_lo, J'(0)={f_lo:.4f} bei delta_hi."
        )

    lam_star = brentq(
        lambda lam: _dj_at_zero(lam, intervals, T_end),
        lam_lo, lam_hi,
        xtol=tol * lam_lo,
        rtol=tol,
    )
    return 1.0 / lam_star


def test_critical_delta(intervals, T_end: float,
                        ref_delta_star: float = None,
                        tol: float = 0.1):
    """
    Test 4: Berechnet delta* analytisch und prueft:
    - J'(0) bei delta* ist ~0
    - J'(0) links von delta* ist > 0  (wait*=0)
    - J'(0) rechts von delta* ist < 0  (wait*>0)
    - Optional: Vergleich mit Referenzwert
    """
    print("\n" + "=" * 70)
    print("Test 4: Kritischer Schwellwert delta*  (analytisch, kein Wolfram)")
    print("=" * 70)

    delta_star = find_critical_delta(intervals, T_end)
    lam_star   = 1.0 / delta_star

    dj_star  = _dj_at_zero(lam_star,               intervals, T_end)
    dj_below = _dj_at_zero(1.0 / (delta_star - 1), intervals, T_end)
    dj_above = _dj_at_zero(1.0 / (delta_star + 1), intervals, T_end)

    print(f"  delta*             : {delta_star:.6f}")
    print(f"  lambda*            : {lam_star:.8f}")
    print(f"  J'(0) bei delta*   : {dj_star:.2e}  (soll ~0)")
    print(f"  J'(0) bei delta*-1 : {dj_below:+.6f}  (soll > 0 => wait*=0)")
    print(f"  J'(0) bei delta*+1 : {dj_above:+.6f}  (soll < 0 => wait*>0)")

    ok_zero  = abs(dj_star)  < 1e-6
    ok_below = dj_below > 0
    ok_above = dj_above < 0

    print()
    print(f"  J'(0)~0 bei delta* : {'✓' if ok_zero  else '✗  <- FEHLER'}")
    print(f"  J'(0)>0 links      : {'✓' if ok_below else '✗  <- FEHLER'}")
    print(f"  J'(0)<0 rechts     : {'✓' if ok_above else '✗  <- FEHLER'}")

    if ref_delta_star is not None:
        ok_ref = abs(delta_star - ref_delta_star) < tol
        diff   = abs(delta_star - ref_delta_star)
        print(f"  Referenz delta*    : {ref_delta_star:.4f}  "
              f"{'✓' if ok_ref else f'✗  <- FEHLER (Diff={diff:.4f})'}")
    else:
        ok_ref = True
        print("  (kein Referenzwert angegeben)")

    passed = ok_zero and ok_below and ok_above and ok_ref
    print()
    print("Schwellwert-Test:", "BESTANDEN ✓" if passed else "FEHLGESCHLAGEN ✗")
    return passed, delta_star

def test_critical_lambda_wolfram(inst,
                                 intervals,
                                 T_end,
                                 n_picks_after: int = 3,
                                 lambda_guess: float = 0.05,
                                 ref_delta_star: float = None,
                                 tol: float = 0.1):
    """
    Test 5: Kritischer Schwellwert lambda* via Mathematica (FindRoot auf lambda*C(lambda)=1).

    - Ruft WarehouseOptimization.FindCriticalLambda in Wolfram auf.
    - Konvertiert Intervalldaten ins Wolfram-Format.
    - Berechnet lambda* und delta* = 1/lambda*.
    - Optionaler Vergleich mit einem Referenz-delta*.
    """
    print("\n" + "=" * 70)
    print("Test 5: Kritischer Schwellwert lambda* (Wolfram / FindRoot)")
    print("=" * 70)

    # Intervalle ins Wolfram-Format bringen
    wl_intervals = ib.to_wolfram_list(intervals)

    session = WolframLanguageSession(KERNEL_PATH)
    try:
        try:
            session.start()
        except Exception as exc:
            print(f"  Uebersprungen: Wolfram-Kernel konnte nicht gestartet werden ({exc})")
            return None, None, None
        session.evaluate(wl.Get(SCRIPT_PATH))

        # Aufruf der Mathematica-Funktion
        lambda_star = session.evaluate(
            wl.WarehouseOptimization.FindCriticalLambda(
                wl_intervals,
                float(T_end),
                int(n_picks_after),
                float(lambda_guess),
            )
        )

        try:
            # Falls es ein Wolfram-Ausdruck ist, erst numerisch machen
            lambda_star = session.evaluate(wl.N(lambda_star))
            lambda_star = float(lambda_star)
        except Exception:
            print("  FEHLER: Wolfram gibt kein numerisches lambda* zurueck.")
            print(f"  lambda_star = {lambda_star!r}")
            return False, None, None

        lambda_star = float(lambda_star)
        delta_star = 1.0 / lambda_star

        print(f"  lambda* (Wolfram): {lambda_star:.8f}")
        print(f"  delta*           : {delta_star:.6f}")

        # Konsistenzcheck mit reinem Python-Test (optional):
        dj0 = _dj_at_zero(lambda_star, intervals, T_end)
        print(f"  J'(0) bei lambda*: {dj0:+.3e}  (soll ~0)")

        ok_zero = abs(dj0) < 1e-6

        if ref_delta_star is not None:
            diff = abs(delta_star - ref_delta_star)
            ok_ref = diff < tol
            print(f"  Referenz delta*  : {ref_delta_star:.4f} "
                  f"{'✓' if ok_ref else f'✗  <- FEHLER (Diff={diff:.4f})'}")
        else:
            ok_ref = True
            print("  (kein Referenzwert fuer delta* angegeben)")

        passed = ok_zero and ok_ref
        print()
        print("Wolfram-Lambda*-Test:", "BESTANDEN ✓" if passed else "FEHLGESCHLAGEN ✗")
        return passed, lambda_star, delta_star

    finally:
        try:
            session.terminate()
        except Exception:
            pass


def test_fixed_wait_costs(delta_exp: float,
                          intervals,
                          T_end: float,
                          waits=(0.0, 1.0),
                          n_picks_after: int = 3):
    """
    Test X: Kosten J(wait) fuer dedizierte wait-Werte (z.B. 0, 1).
    Nutzt die analytische Form J(wait) = wait + C(lambda)*exp(-lambda*wait).
    Gibt nur die Werte aus, kein Pass/Fail.
    """
    print("\n" + "=" * 70)
    print(f"Test 6: Kosten für feste wait-Werte (delta={delta_exp})")
    print("=" * 70)

    lam = 1.0 / delta_exp
    for w in waits:
        j_val = compute_J_for_wait(lam, w, intervals, T_end,
                                   n_picks_after=n_picks_after)
        print(f"  J(wait={w:.3f}) = {j_val:.6f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

REFERENCES = {
    28.8: (14.3593, 0.0000),
     5.0: ( 0.0837, 0.0000),
    50.0: (50.4239, 0.4239),
    75.0: (102.0616, 27.0608),
    100.0: (154.2328, 54.2319)
}

def run_validation(delta_exp: float = 28.8,
                   ref_j: float = None,
                   ref_wait: float = None,
                   skip_wolfram: bool = False,
                   ref_delta_star: float = None
                   ):

    print("\n" + "=" * 70)
    print(f"Validierung: Route A = [1, 2, 8], M=8, L=17, delta={delta_exp}")
    print("=" * 70)

    # Referenzwerte aus Tabelle laden (falls nicht explizit uebergeben)
    if ref_j is None or ref_wait is None:
        ref = REFERENCES.get(delta_exp)
        if ref:
            ref_j_use, ref_wait_use = ref
            print(f"  Referenz aus Tabelle: J*={ref_j_use}, wait*={ref_wait_use}")
        else:
            ref_j_use    = ref_j    # bleibt None → kein J*-Check
            ref_wait_use = ref_wait if ref_wait is not None else 0.0
            print("  Kein Referenzwert fuer dieses delta — nur Ausgabe.")
    else:
        ref_j_use, ref_wait_use = ref_j, ref_wait

    inst = make_instance()
    intervals, T_end = ib.build_interval_data(inst)

    print("\nBerechnete Intervalle:")
    ib.print_intervals(intervals)

    ok1 = test_coefficients(inst, intervals)
    ok2 = test_ed_pointwise(inst, intervals)
    ok3 = None
    ok4 = None
    ok5 = None

    if skip_wolfram:
        print("\n(Wolfram-Test uebersprungen)")
    else:
        ok3 = test_wolfram(inst, intervals, T_end,
                           delta_exp=delta_exp,
                           ref_j=ref_j_use,
                           ref_wait=ref_wait_use)

        try:
            ok4, delta_star = test_critical_delta(intervals, T_end)
        except ValueError as exc:
            ok4 = None
            delta_star = None
            print("\n" + "=" * 70)
            print("Test 4: Kritischer Schwellwert delta*  (analytisch, kein Wolfram)")
            print("=" * 70)
            print(f"  Uebersprungen: {exc}")

        # Test 5: Mathematica/FindRoot
        ok5, lambda_star_wl, delta_star_wl = test_critical_lambda_wolfram(
            inst, intervals, T_end,
            n_picks_after=3,
            lambda_guess=1.0 / (delta_exp if delta_exp > 0 else 1.0),
            ref_delta_star=ref_delta_star
        )

    print("\n" + "=" * 70)
    print("Gesamtergebnis:")
    print(f"  Koeffizienten-Test : {'PASS' if ok1 else 'FAIL'}")
    print(f"  Punktprobe-Test    : {'PASS' if ok2 else 'FAIL'}")
    if ok3 is not None or ok4 is not None or ok5 is not None:
        print(f"  Wolfram-Test (J*,wait*) : {'PASS' if ok3 else 'SKIP' if ok3 is None else 'FAIL'}")
        print(f"  Schwellwert-Test (Py)   : {'PASS' if ok4 else 'SKIP' if ok4 is None else 'FAIL'}")
        print(f"  Schwellwert-Test (WL)   : {'PASS' if ok5 else 'SKIP' if ok5 is None else 'FAIL'}")
    print("=" * 70)

    test_fixed_wait_costs(delta_exp, intervals, T_end, waits=(0.0, 1.0))


if __name__ == "__main__":
    # Setze skip_wolfram=True um nur die Python-Seite zu testen
    for delta in [28.8]: #, 5.0, 49.5, 49.592057, 49.6, 50.0, 75.0, 100.0
        run_validation(delta_exp=delta)
        print()
