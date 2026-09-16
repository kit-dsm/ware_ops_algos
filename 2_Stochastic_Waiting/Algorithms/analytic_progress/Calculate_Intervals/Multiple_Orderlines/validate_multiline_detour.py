"""
validate_multiline_detour.py
============================

Referenz-Validierung der Multiline-Umwegoperatoren fuer eine Order mit
beliebig vielen Orderlines (kappa >= 1).

Dieses Skript ist BEWUSST eigenstaendig (nur numpy + math) und haengt NICHT
von core/models/continuous_calculation ab. Es dient als Ground-Truth fuer
  - continuous_calculation_multiline.py  (kontinuierliche Variante)
  - multiple orderlines/detour_multiline.py  (diskrete Variante)

Geprueft wird, dass die drei geschlossenen Operatoren

    E[D(t; kappa)] = psi(kappa, p2) * D_backtrack          (Backtrack-Extremwert)
                   + Lambda_pass(kappa)                     (Layer-Cake-Horizontal)
                   + 2L * E[#2L(M_hit)]                     (geteiltes 2L, vertikal)

mit einer Monte-Carlo-Simulation unter den Threading-Regeln uebereinstimmen,
fuer kappa = 1, 2, 3 und in beiden Regimen (ersetzbar / nicht ersetzbar), und
dass sie die handgerechnete kappa=2-Tabelle (Beispiel A235, Punkt (2,6))
exakt reproduzieren (Summe 54.488).

WICHTIG zur Ersetzbarkeit (replaceable):
    Konsistent zu detour.compute_detour_apass/aup und
    continuous_calculation._continuous_detour_apass/aup gilt:
        x > a_ret  -> vertikaler Anteil 0   (ersetzbar / floor)
        x < a_ret  -> vertikaler Anteil 2L  (nicht ersetzbar / ceil)
    Siehe is_replaceable(); hier wird das Regime als bool uebergeben, damit
    das Skript regime-agnostisch beide Faelle prueft.
"""

from __future__ import annotations

from math import comb, isclose
from typing import List, Sequence, Tuple

import numpy as np

_RNG = np.random.default_rng(12345)


# ===========================================================================
# Die drei geschlossenen Operatoren
# (identisch zu continuous_calculation_multiline.py)
# ===========================================================================

def psi(kappa: int, p: float) -> float:
    """Erwarteter Backtrack-Faktor fuer kappa i.i.d. Linien.

    n Backtrack-Linien -> Round-Trip 2*n/(n+1)*D_backtrack; psi mittelt diesen
    Faktor ueber n ~ Binomial(kappa, p). Es gilt psi(1,p)=p, psi(inf,p)->2.
    """
    if p <= 0.0:
        return 0.0
    return 2.0 - (2.0 / ((kappa + 1) * p)) * (1.0 - (1.0 - p) ** (kappa + 1))


def layer_cake_expected_max(weights: Sequence[float], kappa: int, M: int) -> float:
    """E[2*max(dist) ueber getroffene Gassen] via Layer-Cake.

    weights = Round-Trip-Laengen Weg(a) = 2*dist_x(a) der betroffenen Gassen.
    Pro Gasse Trefferwahrscheinlichkeit 1/M je Linie, also
        P(max >= w_j) = 1 - (1 - N_ge(j)/M)^kappa.
    """
    w = sorted(float(x) for x in weights)
    if not w:
        return 0.0
    n = len(w)
    total = 0.0
    prev = 0.0
    for idx, weight in enumerate(w):
        n_ge = n - idx  # Anzahl Gassen mit Weg >= w[idx] (aufsteigend sortiert)
        prob = 1.0 - (1.0 - (n_ge / M)) ** kappa
        total += (weight - prev) * prob
        prev = weight
    return total


def mhit_prob(j: int, kappa: int, M: int, k: int) -> float:
    """P(M_hit = j): genau j der g=M-k nicht besuchten Gassen werden getroffen.

    kappa Baelle uniform in M Gassen; Inklusions-Exklusion ueber leere Gassen.
    """
    g = M - k
    if j < 0 or j > g:
        return 0.0
    s = 0.0
    for i in range(0, j + 1):
        s += ((-1) ** i) * comb(j, i) * ((k + j - i) / M) ** kappa
    return comb(g, j) * s


def mhit_distribution(M: int, k: int, kappa: int) -> List[float]:
    g = M - k
    return [mhit_prob(j, kappa, M, k) for j in range(0, g + 1)]


def expected_num_2L(M: int, k: int, kappa: int, replaceable: bool) -> float:
    """E[#2L(M_hit)] mit #2L = floor(M_hit/2) (ersetzbar) bzw. ceil(M_hit/2)."""
    g = M - k
    if g <= 0:
        return 0.0
    probs = mhit_distribution(M, k, kappa)

    def num_2l(j: int) -> int:
        return (j // 2) if replaceable else ((j + 1) // 2)

    return sum(probs[j] * num_2l(j) for j in range(0, g + 1))


def is_replaceable(x: float, A: Sequence[int], k: int, is_return_phase: bool,
                   eps: float = 1e-9) -> bool:
    """Referenz-Praedikat fuer die Ersetzbarkeit (korrigierte Ungleichung).

    Ersetzbar (vertikaler Anteil entfaellt) genau dann, wenn eine Return-Gasse
    existiert, der Picker rechts von ihr steht (x > a_ret) und nicht in der
    Abstiegsphase ist -- konsistent zu compute_detour_apass/aup (x>a_ret -> 0).
    """
    a_ret = sorted(A)[0] if (k % 2 == 1) else None
    return (a_ret is not None) and (x > a_ret + eps) and (not is_return_phase)


# ===========================================================================
# Geschlossene Gesamtform
# ===========================================================================

class DetourBreakdown:
    """Container fuer die Komponenten der geschlossenen Form."""

    __slots__ = ("expected", "backtrack", "pass_horizontal", "vertical",
                 "kappa", "replaceable", "mhit")

    def __init__(self, expected, backtrack, pass_horizontal, vertical,
                 kappa, replaceable, mhit):
        self.expected = expected
        self.backtrack = backtrack
        self.pass_horizontal = pass_horizontal
        self.vertical = vertical
        self.kappa = kappa
        self.replaceable = replaceable
        self.mhit = mhit

    def __repr__(self) -> str:
        return (f"DetourBreakdown(E={self.expected:.4f}, bt={self.backtrack:.4f}, "
                f"horiz={self.pass_horizontal:.4f}, vert={self.vertical:.4f}, "
                f"kappa={self.kappa}, replaceable={self.replaceable})")


def closed_form_detour(
    kappa: int,
    *,
    p2: float,
    d_backtrack: float,
    gpass_weg: Sequence[float],
    L: float,
    M: int,
    k: int,
    replaceable: bool,
    n_front: int = 0,  # nur fuer einheitliche Signatur; geht ueber g=M-k ein
) -> DetourBreakdown:
    """Geschlossene Form E[D(t; kappa)] aus den drei Operatoren.

    gpass_weg : Round-Trip-Laengen Weg(a)=2*dist_x(a) der passierten Gassen.
                Front-Gassen liefern keinen Horizontalanteil (Vorwaertsweg),
                zaehlen aber ueber g=M-k in den vertikalen M_hit-Operator hinein.
    n_front   : ungenutzt (die Front-Gassen sind implizit in g=M-k enthalten).
    """
    del n_front  # bewusst ignoriert
    backtrack = psi(kappa, p2) * d_backtrack
    pass_horizontal = layer_cake_expected_max(gpass_weg, kappa, M)
    e_2l = expected_num_2L(M, k, kappa, replaceable)
    vertical = 2.0 * L * e_2l
    expected = backtrack + pass_horizontal + vertical
    return DetourBreakdown(
        expected=expected,
        backtrack=backtrack,
        pass_horizontal=pass_horizontal,
        vertical=vertical,
        kappa=kappa,
        replaceable=replaceable,
        mhit=mhit_distribution(M, k, kappa),
    )


# ===========================================================================
# Monte-Carlo unter den Threading-Regeln (Ground-Truth)
# ===========================================================================

def monte_carlo_detour(
    kappa: int,
    *,
    p2: float,
    d_backtrack: float,
    gpass_weg: Sequence[float],
    n_front: int,
    L: float,
    M: int,
    k: int,
    replaceable: bool,
    n_samples: int = 600_000,
) -> Tuple[float, np.ndarray]:
    """Simuliert kappa unabhaengige Orderlines und mittelt den Umweg.

    Klassen je Linie ~ Multinomial(p1, p2, p3, p4) mit
        p1 = k/M - p2,  p3 = |gpass|/M,  p4 = n_front/M.
    Regeln:
        Backtrack : Round-Trip 2*max(Reichweite), Reichweite ~ U(0, d_backtrack).
        Pass      : Horizontal 2*max(dist) ueber getroffene Pass-Gassen.
        Front     : kein Horizontalanteil.
        Vertikal  : #2L(M_hit)*2L, M_hit = distinkte getroffene nicht besuchte Gassen.
    """
    two_L = 2.0 * L
    weg = np.asarray(gpass_weg, dtype=float)
    n_pass = len(weg)
    g = M - k
    assert n_pass + n_front == g, "‖gpass‖ + n_front muss M-k ergeben"

    p3 = n_pass / M
    p4 = n_front / M
    p1 = k / M - p2
    if p1 < -1e-9:
        raise ValueError(f"p1 < 0 (p2={p2}, k/M={k / M})")
    p1 = max(0.0, p1)
    assert isclose(p1 + p2 + p3 + p4, 1.0, abs_tol=1e-9), "Klassenwahrscheinlichkeiten != 1"

    N = n_samples
    cls = _RNG.choice(4, size=(N, kappa), p=[p1, p2, p3, p4])  # 0=route,1=bt,2=pass,3=front
    reach = _RNG.random((N, kappa)) * d_backtrack
    pass_aisle = _RNG.integers(0, max(1, n_pass), size=(N, kappa))
    front_aisle = _RNG.integers(0, max(1, n_front), size=(N, kappa))

    # Backtrack: Round-Trip = 2*max(Reichweiten der Backtrack-Linien), sonst 0
    reach_bt = np.where(cls == 1, reach, 0.0)
    backtrack = 2.0 * reach_bt.max(axis=1)

    # Pass-Horizontal: 2*max(dist) = max(Weg) ueber getroffene Pass-Gassen, sonst 0
    weg_per_line = np.where(cls == 2, weg[pass_aisle], 0.0) if n_pass > 0 else np.zeros((N, kappa))
    pass_horizontal = weg_per_line.max(axis=1)

    # M_hit: distinkte getroffene nicht besuchte Gassen (pass-Index 0..n_pass-1,
    # front-Index n_pass..n_pass+n_front-1)
    hit = np.zeros((N, g), dtype=bool)
    for c in range(kappa):
        pm = cls[:, c] == 2
        if n_pass > 0:
            hit[pm, pass_aisle[pm, c]] = True
        fm = cls[:, c] == 3
        if n_front > 0:
            hit[fm, n_pass + front_aisle[fm, c]] = True
    mhit = hit.sum(axis=1)

    if replaceable:
        n_2l = mhit // 2
    else:
        n_2l = (mhit + 1) // 2
    vertical = two_L * n_2l

    total = backtrack + pass_horizontal + vertical
    return float(total.mean()), mhit


# ===========================================================================
# Validierter Referenzfall: A235, Punkt (2,6), nicht ersetzbar
# ===========================================================================

EXAMPLE = dict(
    p2=0.3309,
    d_backtrack=31.4,
    gpass_weg=[4.0, 8.0, 10.0, 12.0],   # dist_x {2,4,5,6} -> Weg = 2*dist
    n_front=1,
    L=17.0,                              # 2L = 34
    M=8,
    k=3,
)

# Sollwerte der handgerechneten kappa=2-Tabelle
EXPECTED_KAPPA2 = dict(
    backtrack=18.488,
    pass_horizontal=6.78125,
    vertical=29.21875,
    total=54.488,
)


def _single_line_reference(*, p2, d_backtrack, gpass_weg, n_front, L, M, k,
                           replaceable) -> float:
    """Einzellinien-E[D] (kappa=1) zur Reduktionspruefung.

    Nicht ersetzbar: E = p2*D_back + p3*(meanWeg + 2L) + p4*2L.
    Ersetzbar      : E = p2*D_back + p3*meanWeg            (vertikal 0).
    """
    n_pass = len(gpass_weg)
    p3 = n_pass / M
    p4 = n_front / M
    mean_weg = sum(gpass_weg) / n_pass if n_pass else 0.0
    two_L = 2.0 * L
    if replaceable:
        return p2 * d_backtrack + p3 * mean_weg
    return p2 * d_backtrack + p3 * (mean_weg + two_L) + p4 * two_L


def main() -> None:
    print("=" * 70)
    print("REFERENZFALL A235 (2,6) -- kappa = 2, nicht ersetzbar")
    print("=" * 70)
    bd = closed_form_detour(2, replaceable=False, **EXAMPLE)
    mc, _ = monte_carlo_detour(2, replaceable=False, **EXAMPLE)
    print(f"  Backtrack   geschlossen = {bd.backtrack:8.4f}   (Tabelle {EXPECTED_KAPPA2['backtrack']})")
    print(f"  Pass-Horiz. geschlossen = {bd.pass_horizontal:8.5f}   (Tabelle {EXPECTED_KAPPA2['pass_horizontal']})")
    print(f"  Vertikal    geschlossen = {bd.vertical:8.5f}   (Tabelle {EXPECTED_KAPPA2['vertical']})")
    print(f"  SUMME       geschlossen = {bd.expected:8.4f}   (Tabelle {EXPECTED_KAPPA2['total']})")
    print(f"  SUMME       Monte-Carlo = {mc:8.4f}")
    assert isclose(bd.backtrack, EXPECTED_KAPPA2["backtrack"], abs_tol=1e-2)
    assert isclose(bd.pass_horizontal, EXPECTED_KAPPA2["pass_horizontal"], abs_tol=1e-2)
    assert isclose(bd.vertical, EXPECTED_KAPPA2["vertical"], abs_tol=1e-2)
    assert isclose(bd.expected, EXPECTED_KAPPA2["total"], abs_tol=1e-2)
    assert isclose(bd.expected, mc, rel_tol=5e-3)  # MC-Rauschen ~ 0.5 %

    print()
    print("=" * 70)
    print("GESCHLOSSENE FORM vs MONTE-CARLO  (kappa = 1, 2, 3)")
    print("=" * 70)
    for replaceable in (False, True):
        regime = "ersetzbar (floor)" if replaceable else "nicht ersetzbar (ceil)"
        print(f"\n  Regime: {regime}")
        print(f"  {'kappa':>5} {'geschlossen':>13} {'Monte-Carlo':>13} {'|Diff|':>9}")
        for kappa in (1, 2, 3):
            cf = closed_form_detour(kappa, replaceable=replaceable, **EXAMPLE)
            mc_val, _ = monte_carlo_detour(kappa, replaceable=replaceable, **EXAMPLE)
            diff = abs(cf.expected - mc_val)
            print(f"  {kappa:>5} {cf.expected:>13.4f} {mc_val:>13.4f} {diff:>9.4f}")
            assert isclose(cf.expected, mc_val, rel_tol=5e-3), \
                f"Abweichung zu gross bei kappa={kappa}, {regime}"

    print()
    print("=" * 70)
    print("REDUKTION kappa = 1 == EINZELLINIE")
    print("=" * 70)
    for replaceable in (False, True):
        regime = "ersetzbar" if replaceable else "nicht ersetzbar"
        cf1 = closed_form_detour(1, replaceable=replaceable, **EXAMPLE).expected
        ref = _single_line_reference(replaceable=replaceable, **EXAMPLE)
        print(f"  {regime:>16}:  geschlossen(kappa=1) = {cf1:8.4f}   Einzellinie = {ref:8.4f}")
        assert isclose(cf1, ref, abs_tol=1e-9), "Reduktion kappa=1 stimmt nicht"

    print()
    print("=" * 70)
    print("MONOTONIE  E[D;1] < E[D;2] < E[D;3]  (nicht ersetzbar)")
    print("=" * 70)
    vals = [closed_form_detour(k, replaceable=False, **EXAMPLE).expected for k in (1, 2, 3)]
    print(f"  {vals[0]:.4f} < {vals[1]:.4f} < {vals[2]:.4f}")
    assert vals[0] < vals[1] < vals[2]

    print()
    print("=" * 70)
    print("PRAEDIKAT is_replaceable (korrigierte Ungleichung x > a_ret)")
    print("=" * 70)
    A = [1, 2, 8]
    for x, ret, exp in [(1.5, False, True), (0.0, False, False), (5.0, True, False)]:
        got = is_replaceable(x, A, k=3, is_return_phase=ret)
        print(f"  x={x}, is_return_phase={ret}  -> replaceable={got}  (erwartet {exp})")
        assert got == exp

    print("\nAlle Referenz-Checks bestanden.")


if __name__ == "__main__":
    main()