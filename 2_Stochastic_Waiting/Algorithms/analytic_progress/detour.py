from dataclasses import dataclass
from math import isclose
from typing import List, Optional

try:
    from .models import WarehouseInstance
except ImportError:  # Direktes Ausführen als Skript
    from models import WarehouseInstance


def _to_picking_on_lower_cross_aisle(inst: WarehouseInstance, t: float, yStart: float, isReturnPhase: bool) -> bool:
    """True, wenn der Picker auf y=0 im Hinweg ist und deshalb kein Umweg zählt."""
    return (abs(yStart) < 1e-9) and (not isReturnPhase) and ((t * inst.v) <= inst.M)


def compute_detour_route_back(inst: WarehouseInstance, xStart: float, yStart: float, t: float, isReturnPhase: bool
) -> float:
    """

    Parameter:
        L (float): Total length of aisles
        NL (int): Number of pick positions (nodes) per aisle
        w (float): Horizontal distance between aisles
        A (list[int] or list[float]): aisle coordinates of aisles visited
        xStart (float): horizontal start position
        yStart (float): vertical start position
        isReturnPhase (bool): True = only true if position is in the return phase in retur aisle a_ret

    Returns:
        avg detour for going back(float)
    """

    if _to_picking_on_lower_cross_aisle(inst, t, yStart, isReturnPhase):
        return 0.0

    useCrossStart = False
    L = inst.L
    N_L = inst.N_L
    w = inst.w
    A = inst.A
    k = len(A)
    iRet = None
    aRet = None
    jInput = None

    if yStart in (0.0, L):
        useCrossStart = True

    if not useCrossStart:
        # xStart kann als float übergeben werden, A enthält i.d.R. Ganzzahlen.
        jInput = A.index(int(xStart)) + 1  # 1-basierter Gasenindex


    # Return-Gasse: erste Gasse in A (1-basiert)
    if k % 2 == 1:
        iRet = 1
        aRet = A[iRet - 1]

    eps = 1e-9
    # Spezialfall nach Return-Gasse auf unterer Quergasse im Rueckweg:
    # k ungerade, iRet vorhanden, y=0, rueck-Phase, x < aRet.
    # 'rueck' wird hier robust ueber den Ausschluss des Hinweg-Sonderfalls modelliert.
    special_after_return_on_lower_cross = (
        (k % 2 == 1)
        and (iRet is not None)
        and (aRet is not None)
        and (abs(yStart) < eps)
        and (not _to_picking_on_lower_cross_aisle(inst, t, yStart, isReturnPhase))
        and (xStart < aRet)
    )

    # Vertikale Kantenlänge
    lv = L / (N_L + 1)

    def yLev(c: int) -> float:
        return c * lv

    if special_after_return_on_lower_cross and (iRet is not None) and (aRet is not None):
        # Spezialfall: explizite one-way Mittelung ueber alle besuchten Knoten.
        # Return-Gasse: horiz(x->aRet) + vert(y->node)
        # Andere Gassen: horiz(x->ai) + vert(y->node) + 2L
        total_cost = 0.0
        total_picks = k * N_L
        for i in range(1, k + 1):
            aisle_x = A[i - 1]
            for c in range(1, N_L + 1):
                vert = abs(yLev(c) - yStart)
                horiz = abs(aisle_x - xStart) * w
                if i == iRet:
                    dist = (horiz + vert) / lv
                else:
                    dist = (horiz + vert + (2.0 * L) + (max(0, (i - 2)) * L)) / lv
                total_cost += dist
        return total_cost / total_picks if total_picks > 0 else 0.0

    # Laufrichtung deltaDir[i] je Position i in A, i ist 1-basiert
    def deltaDir(i: int) -> int:
        # Konsistent zu core.delta_j (ohne curr_ret-Sonderfall):
        # k gerade  -> ungerade i aufwaerts (+1), gerade i abwaerts (-1)
        # k ungerade -> gerade i aufwaerts (+1), ungerade i abwaerts (-1)
        if k % 2 == 0:
            return +1 if (i % 2 == 1) else -1
        return +1 if (i % 2 == 0) else -1

    # Rohes Pick-Set je Position (1-basiert): alle Level
    def pickSetRaw(i: int):
        return range(1, N_L + 1)

    # Pick-Set der Startgasse (wenn Start in Gasse und Startgasse != Return-Gasse)
    def startPickSetFiltered(j: int, ys: float):
        if deltaDir(j) == -1:  # down: picks unterhalb ys
            return [c for c in pickSetRaw(j) if yLev(c) < ys]
        else:  # up: picks oberhalb ys
            return [c for c in pickSetRaw(j) if yLev(c) > ys]

    # Pick-Set in der Return-Gasse abhängig von Phase:
    # - Forward-Phase: keine Picks in Return-Gasse
    # - Return-Phase: alle Level mit yLev[c] > ysInput
    def pickSetReturn(ys: float):
        if isReturnPhase or special_after_return_on_lower_cross:
            return [c for c in pickSetRaw(iRet) if yLev(c) > ys]
        else:
            return []

    # Startparameter
    startInAisleQ = False
    j = None
    yStartAisle = None


    if not useCrossStart:
        # Start in Gasse
        if iRet and jInput == iRet:
            # Spezialfall: Start in Return-Gasse -> simuliere Start auf (aRet, 0)
            startInAisleQ = False
            xStartEff = aRet if aRet is not None else xStart
            yStartEff = 0.0

            pos = [i for i in range(1, k + 1) if A[i - 1] > xStartEff]
            if not pos:
                firstIndex = k + 1
                baseFirst = 0.0
                yInFirst = 0.0
            else:
                firstIndex = pos[0]
                if deltaDir(firstIndex) == -1:
                    yInFirst = N_L * lv
                else:
                    yInFirst = lv
                baseFirst = (
                    abs(xStartEff - A[firstIndex - 1]) * w
                    + abs(yStartEff - yInFirst)
                ) / lv
        else:
            # Normalfall: Start in Gasse, die nicht Return-Gasse ist
            startInAisleQ = True
            j = jInput  # 1-basierter Gasenindex (konsistent mit iRet, deltaDir, pickSet)
            yStartAisle = yStart
            firstIndex = j
            baseFirst = 0.0
            yInFirst = yStartAisle
    else:
        # Start auf Quergasse
        startInAisleQ = False
        xStartEff = xStart
        yStartEff = yStart

        exact_idx = next((i for i in range(1, k + 1) if abs(A[i - 1] - xStartEff) < eps), None)

        if exact_idx is not None:
            on_lower_cross = abs(yStartEff) < eps
            on_upper_cross = abs(yStartEff - L) < eps
            # Sonderfall: Top der Return-Gasse vor Return-Phase -> aktuelle Gasse ueberspringen
            if (iRet is not None) and (exact_idx == iRet) and on_upper_cross and (not isReturnPhase):
                pos = [i for i in range(exact_idx + 1, k + 1)]
                yStartEff = 0.0
            else:
                include_current = (
                    (on_lower_cross and deltaDir(exact_idx) == +1)
                    or (on_upper_cross and (deltaDir(exact_idx) == -1 or exact_idx == iRet))
                    )
                if include_current:
                    pos = [i for i in range(exact_idx, k + 1)]
                else:
                    pos = [i for i in range(exact_idx + 1, k + 1)]

        else:
            pos = [i for i in range(1, k + 1) if A[i - 1] > xStartEff]
        if not pos:
            firstIndex = k + 1
            baseFirst = 0.0
            yInFirst = 0.0
        else:
            firstIndex = pos[0]
            if special_after_return_on_lower_cross and (iRet is not None) and (firstIndex == iRet):
                # Nach der Return-Gasse auf y=0 wird a_ret vom unteren Cross aus betreten.
                yInFirst = lv
            elif deltaDir(firstIndex) == -1:
                yInFirst = N_L * lv
            else:
                yInFirst = lv
            baseFirst = (
                abs(xStartEff - A[firstIndex - 1]) * w
                + abs(yStartEff - yInFirst)
            ) / lv

    # Effektives Pick-Set
    def pickSet(i: int):
        if i == iRet:
            return pickSetReturn(yStart)
        if startInAisleQ and (j is not None) and (i == j) and (j != iRet):
            return startPickSetFiltered(j, yStartAisle)
        return list(pickSetRaw(i))

    # Exit-Höhe je Position i (physisch)
    def yOutVal(i: int) -> float:
        if deltaDir(i) == -1:
            return lv
        else:
            return N_L * lv

    # Standard-Transition in Level-Einheiten
    # i: 1..k-1, Differenz A[i] - A[i-1] (Python-Index 1 und 0)
    def dTransVal(i: int) -> float:
        return (2 * lv + abs(A[i] - A[i - 1]) * w) / lv

    # Entry-Höhen und Base-Times (1-basiert, in Listen an Index i-1)
    yInArr = [0.0] * k
    baseArr = [0.0] * k

    if firstIndex is not None and firstIndex <= k:
        yInArr[firstIndex - 1] = yInFirst
        baseArr[firstIndex - 1] = baseFirst

    # nach rechts
    if firstIndex is not None:
        for i in range(firstIndex + 1, k + 1):
            aisleTravel = abs(
                yOutVal(i - 1) - yInArr[i - 2]
            ) / lv  # Vorgänger: i-1 → Index i-2
            transSteps = dTransVal(i - 1)  # Vorgänger-Transition
            yInArr[i - 1] = yOutVal(i - 1)
            baseArr[i - 1] = baseArr[i - 2] + aisleTravel + transSteps

    def deltaStep(i: int, c: int) -> float:
        return abs(c - yInArr[i - 1] / lv)

    def tauPos(i: int, c: int) -> float:
        if i == iRet and isReturnPhase:
            base = abs(c - yStart / lv)
        else:
            base = baseArr[i - 1] + deltaStep(i, c)

        return base

    rightCost = 0.0
    rightPicks = 0

    if firstIndex is None:
        firstIndex = k + 1

    for i in range(firstIndex, k + 1):
        cs = pickSet(i)
        rightCost += sum(tauPos(i, c) for c in cs)
        rightPicks += len(cs)

    ret_in_right_span = (iRet is not None) and (firstIndex <= iRet <= k)
    need_ret_once = isReturnPhase or special_after_return_on_lower_cross
    if need_ret_once and (iRet is not None) and (not ret_in_right_span):
        cs_ret = pickSet(iRet)
        retCost = sum(tauPos(iRet, c) for c in cs_ret)
        retPicks = len(cs_ret)
    else:
        retCost = 0.0
        retPicks = 0

    totalCost = rightCost + retCost
    totalPicks = rightPicks + retPicks
    avgTourPosition = totalCost / totalPicks if totalPicks > 0 else 0.0

    return avgTourPosition


def compute_detour_route_back_exact(
    inst: WarehouseInstance,
    xStart: float,
    yStart: float,
    target_aisle: int,
    target_y_node: int,
    t: Optional[float] = None,
    isReturnPhase: bool = False,
) -> float:
    """Exakter Backtrack-Umweg entlang derselben S-Shape-Route zum Zielknoten und zurueck."""
    if target_aisle < 1 or target_aisle > inst.M:
        raise ValueError(f"target_aisle out of bounds: {target_aisle}")
    if target_y_node < 1 or target_y_node > inst.N_L:
        raise ValueError(f"target_y_node out of bounds: {target_y_node}")
    if target_aisle not in inst.A:
        raise ValueError(f"target_aisle={target_aisle} is not part of visited aisle set A")

    if (t is not None) and _to_picking_on_lower_cross_aisle(inst, t, yStart, isReturnPhase):
        return 0.0

    useCrossStart = False
    L = inst.L
    N_L = inst.N_L
    w = inst.w
    A = inst.A
    k = len(A)
    iRet = None
    aRet = None
    jInput = None

    if yStart in (0.0, L):
        useCrossStart = True

    if not useCrossStart:
        jInput = A.index(int(xStart)) + 1

    if k % 2 == 1:
        iRet = 1
        aRet = A[iRet - 1]

    lv = L / (N_L + 1)

    def yLev(c: int) -> float:
        return c * lv

    def deltaDir(i: int) -> int:
        if k % 2 == 0:
            return +1 if (i % 2 == 1) else -1
        return +1 if (i % 2 == 0) else -1

    def pickSetRaw(i: int):
        return range(1, N_L + 1)

    def startPickSetFiltered(j: int, ys: float):
        if deltaDir(j) == -1:
            return [c for c in pickSetRaw(j) if yLev(c) < ys]
        return [c for c in pickSetRaw(j) if yLev(c) > ys]

    def pickSetReturn(ys: float):
        if isReturnPhase:
            return [c for c in pickSetRaw(iRet) if yLev(c) > ys]
        return []

    startInAisleQ = False
    j = None
    yStartAisle = None

    if not useCrossStart:
        if iRet and jInput == iRet:
            startInAisleQ = False
            xStartEff = aRet if aRet is not None else xStart
            yStartEff = 0.0

            pos = [i for i in range(1, k + 1) if A[i - 1] > xStartEff]
            if not pos:
                firstIndex = k + 1
                baseFirst = 0.0
                yInFirst = 0.0
            else:
                firstIndex = pos[0]
                yInFirst = N_L * lv if deltaDir(firstIndex) == -1 else lv
                baseFirst = (
                    abs(xStartEff - A[firstIndex - 1]) * w
                    + abs(yStartEff - yInFirst)
                ) / lv
        else:
            startInAisleQ = True
            j = jInput
            yStartAisle = yStart
            firstIndex = j
            baseFirst = 0.0
            yInFirst = yStartAisle
    else:
        startInAisleQ = False
        xStartEff = xStart
        yStartEff = yStart

        eps = 1e-9
        exact_idx = next((i for i in range(1, k + 1) if abs(A[i - 1] - xStartEff) < eps), None)

        if exact_idx is not None:
            on_lower_cross = abs(yStartEff) < eps
            on_upper_cross = abs(yStartEff - L) < eps
            if (iRet is not None) and (exact_idx == iRet) and on_upper_cross and (not isReturnPhase):
                pos = [i for i in range(exact_idx + 1, k + 1)]
                yStartEff = 0.0
            else:
                include_current = (
                    (on_lower_cross and deltaDir(exact_idx) == +1)
                    or (on_upper_cross and (deltaDir(exact_idx) == -1 or exact_idx == iRet))
                )
                pos = [i for i in range(exact_idx, k + 1)] if include_current else [i for i in range(exact_idx + 1, k + 1)]
        else:
            pos = [i for i in range(1, k + 1) if A[i - 1] > xStartEff]

        if not pos:
            firstIndex = k + 1
            baseFirst = 0.0
            yInFirst = 0.0
        else:
            firstIndex = pos[0]
            yInFirst = N_L * lv if deltaDir(firstIndex) == -1 else lv
            baseFirst = (
                abs(xStartEff - A[firstIndex - 1]) * w
                + abs(yStartEff - yInFirst)
            ) / lv

    def pickSet(i: int):
        if i == iRet:
            return pickSetReturn(yStart)
        if startInAisleQ and (j is not None) and (i == j) and (j != iRet):
            return startPickSetFiltered(j, yStartAisle)
        return list(pickSetRaw(i))

    def yOutVal(i: int) -> float:
        return lv if deltaDir(i) == -1 else N_L * lv

    def dTransVal(i: int) -> float:
        return (2 * lv + abs(A[i] - A[i - 1]) * w) / lv

    yInArr = [0.0] * k
    baseArr = [0.0] * k

    if firstIndex is not None and firstIndex <= k:
        yInArr[firstIndex - 1] = yInFirst
        baseArr[firstIndex - 1] = baseFirst

    if firstIndex is not None:
        for i in range(firstIndex + 1, k + 1):
            aisleTravel = abs(yOutVal(i - 1) - yInArr[i - 2]) / lv
            transSteps = dTransVal(i - 1)
            yInArr[i - 1] = yOutVal(i - 1)
            baseArr[i - 1] = baseArr[i - 2] + aisleTravel + transSteps

    def deltaStep(i: int, c: int) -> float:
        return abs(c - yInArr[i - 1] / lv)

    def tauPos(i: int, c: int) -> float:
        if i == iRet and isReturnPhase:
            return abs(c - yStart / lv)
        return baseArr[i - 1] + deltaStep(i, c)

    target_i = A.index(target_aisle) + 1
    allowed_nodes = set(pickSet(target_i))
    if target_y_node not in allowed_nodes:
        return 0.0

    one_way = tauPos(target_i, target_y_node)
    return 2.0 * one_way


def compute_detour_apass(
    inst: WarehouseInstance,
    xStart: float,
    yStart: float,
    t: float,
    gPass: List[int],
    isReturnPhase: bool,
    target_aisle: Optional[int] = None,
) -> float:
    """
    Berechnet den zusätzlichen Umweg für einen Pick in einer nicht besuchten,
    aber passierten Gasse (A_pass(a_j)).
    """

    if _to_picking_on_lower_cross_aisle(inst, t, yStart, isReturnPhase):
        return 0.0

    w = inst.w
    L = inst.L
    A = inst.A

    A_sorted = sorted(A)
    k = len(A_sorted)
    eps = 1e-9

    def dist_x(a1: float, a2: int) -> float:
        return abs(a2 - a1) * w

    selected_gpass = gPass
    if target_aisle is not None:
        if target_aisle not in gPass:
            return 0.0
        selected_gpass = [target_aisle]

    # k=1-Sonderfall: auf unterer Quergasse nur im echten Hinweg D_pass=0.
    # Auf dem Rueckweg (nach a_ret) gilt unten D_pass=2(L+dx).
    if k == 1:
        a_ret = A_sorted[0]
        if _to_picking_on_lower_cross_aisle(inst, t, yStart, isReturnPhase) and (xStart < a_ret - eps):
            return 0.0

    def delta_x_pass() -> float:
        n = len(selected_gpass)
        if n == 0:
            return 0.0
        return sum(dist_x(float(xStart), i) for i in selected_gpass) / n

    def return_aisle() -> Optional[int]:
        if k > 1 and (k % 2 == 1):
            return A_sorted[0]
        return None

    if not selected_gpass:
        return 0.0

    dx = delta_x_pass()

    if k == 1 and abs(yStart) < eps and (xStart < A_sorted[0] - eps):
        if not _to_picking_on_lower_cross_aisle(inst, t, yStart, isReturnPhase):
            return 2.0 * (dx + L)

    if k % 2 == 0:
        return 2 * dx + 2.0 * L

    a_ret = A_sorted[0]
    in_return_aisle = abs(xStart - a_ret) < eps
    if in_return_aisle:
        # Return-Gasse: hoch -> 2dx, runter -> 2(dx+L)
        return (2.0 * dx + 2.0 * L) if isReturnPhase else (2.0 * dx)

    a_ret = return_aisle()
    if a_ret is None:
        return dx

    if xStart > a_ret or (xStart == a_ret and yStart == 0 and isReturnPhase is False):
        return 2 * dx

    return 2 * dx + 2.0 * L


def compute_detour_aup(
    inst: WarehouseInstance,
    xStart: float,
    yStart: float,
    gFront: List[int],
    outbound: bool,
    isReturnPhase: bool,
    target_aisle: Optional[int] = None,
) -> float:
    """
    Berechnet den zusätzlichen Umweg für einen Pick in einer nicht besuchten,
    vor dem Picker liegenden Gasse (A_up(a_j), also a_i < a_j).
    """

    L = inst.L
    w = inst.w
    A = inst.A

    A_sorted = sorted(A)
    k = len(A_sorted)
    max_a = A_sorted[-1]
    eps = 1e-9


    def return_aisle() -> Optional[int]:
        if k % 2 == 1:
            return A_sorted[0]
        return None

    selected_gfront = gFront
    if target_aisle is not None:
        if target_aisle not in gFront:
            return 0.0
        selected_gfront = [target_aisle]

    if not selected_gfront:
        return 0.0

    phase1_from_start = outbound

    def phase1_weighted_horizontal_detour(a_k: int) -> float:
        total = len(selected_gfront)
        if total == 0:
            return 0.0
        right_sum = sum(abs(aisle - a_k) * w for aisle in selected_gfront if aisle > a_k)
        return 2.0 * (right_sum / total)

    # k=1-Sonderfall: auf unterer Quergasse ueber outbound unterscheiden.
    # Hinweg (zum a_ret): bisherige Logik; Rueckweg (vom a_ret zum Start): D_front=2L.
    if k == 1:
        a_ret = A_sorted[0]
        if phase1_from_start:
            return phase1_weighted_horizontal_detour(a_ret)
        if (not outbound) and abs(yStart) < eps and (xStart < a_ret - eps):
            return 2.0 * L

    def delta_x_front() -> float:
        relevant_aisles = [aisle for aisle in selected_gfront if aisle > max_a]
        if not relevant_aisles:
            return 0.0
        a_k = A_sorted[-1]
        return sum(abs(aisle - a_k) * w for aisle in relevant_aisles) / len(relevant_aisles)

    if phase1_from_start:
        horizontal_detour = phase1_weighted_horizontal_detour(A_sorted[-1])
    else:
        horizontal_detour = 2 * delta_x_front() if outbound else 0.0


    if k % 2 == 0:
        return (2.0 * L) + horizontal_detour

    a_ret = A_sorted[0]
    # Exakter Eintrittspunkt der Return-Gasse auf der unteren Quergasse:
    # Die Front-Gassen bleiben noch in der Horizontal-Logik, erst innerhalb
    # der Return-Gasse (y>0) wird der Front-Umweg 0.
    if abs(xStart - a_ret) < eps and abs(yStart) < eps and (not isReturnPhase):
        return horizontal_detour

    in_return_aisle = abs(xStart - a_ret) < eps
    if in_return_aisle:
        # Return-Gasse: hoch -> 0, runter -> 2L
        return 2.0 * L if isReturnPhase else 0.0

    a_ret = return_aisle()
    if a_ret is None:
        return (2.0 * L) + horizontal_detour


    if outbound:
        return horizontal_detour

    if a_ret is not None and xStart > a_ret:
        return 0.0

    return 2.0 * L


@dataclass
class DetourEstimate:
    expected_detour: float
    detour_route_back: float
    detour_apass: float
    detour_aup: float
    p1_route: float
    p2_backtrack: float
    p3_pass: float
    p4_front: float
    p_sum: float
    avg_arrival_time: float


def calculate_detour(inst: WarehouseInstance,
                     t: float,
                     xStart: float,
                     yStart: float,
                     gFront: List[int],
                     gPass: List[int],
                     v_nodes_count: int,
                     r_nodes_count: int,
                     isReturnPhase: bool = False) -> float:

    estimate = calculate_detour_estimate(
        inst=inst,
        t=t,
        xStart=xStart,
        yStart=yStart,
        gFront=gFront,
        gPass=gPass,
        v_nodes_count=v_nodes_count,
        r_nodes_count=r_nodes_count,
        isReturnPhase=isReturnPhase,
    )
    return estimate.expected_detour


def calculate_detour_estimate(
    inst: WarehouseInstance,
    t: float,
    xStart: float,
    yStart: float,
    gFront: List[int],
    gPass: List[int],
    v_nodes_count: int,
    r_nodes_count: int,
    isReturnPhase: bool = False,
) -> DetourEstimate:
    """Schaetzt den mittleren Umweg ueber p1..p4 und prueft p1+p2+p3+p4=1."""

    detour_route_back = compute_detour_route_back(inst, xStart, yStart, t, isReturnPhase)
    detour_apass = compute_detour_apass(inst, xStart, yStart, t, gPass, isReturnPhase)
    detour_aup = compute_detour_aup(inst, xStart, yStart, gFront, v_nodes_count == 0, isReturnPhase)

    total_nodes = inst.N_L * inst.M
    if total_nodes <= 0:
        raise ValueError("N = N_L * M must be > 0")

    p1_route = r_nodes_count / total_nodes
    p2_backtrack = v_nodes_count / total_nodes
    p3_pass = (len(gPass) * inst.N_L) / total_nodes
    p4_front = (len(gFront) * inst.N_L) / total_nodes

    p_sum = p1_route + p2_backtrack + p3_pass + p4_front
    if not isclose(p_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Probability sum is not 1.0 (got {p_sum:.12f}). "
            "Check V/R node counts and G_pass/G_front partitioning."
        )

    expected_detour = (
        p2_backtrack * (2.0 * detour_route_back)
        + p3_pass * detour_apass
        + p4_front * detour_aup
    )

    avg_arrival_time = sum(inst.arrival_times) / len(inst.arrival_times) if inst.arrival_times else 0.0

    return DetourEstimate(
        expected_detour=expected_detour,
        detour_route_back=(2 * detour_route_back),
        detour_apass=detour_apass,
        detour_aup=detour_aup,
        p1_route=p1_route,
        p2_backtrack=p2_backtrack,
        p3_pass=p3_pass,
        p4_front=p4_front,
        p_sum=p_sum,
        avg_arrival_time = avg_arrival_time
    )

