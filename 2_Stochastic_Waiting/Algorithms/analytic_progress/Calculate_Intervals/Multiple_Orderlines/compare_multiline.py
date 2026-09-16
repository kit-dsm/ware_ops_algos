from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union
import argparse
import math
import os
import sys
from math import isclose

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)  # ...\Multiple_Orderlines
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..")))       # ...\Calculate_Intervals
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..", ".."))) # ...\analytic_progress
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))) # ...\2_Stochastic_Waiting

try:
    from Algorithms.analytic_progress import continuous_calculation as cc
    from Algorithms.analytic_progress.Calculate_Intervals.Multiple_Orderlines import continuous_calculation_multiline as cc_multi
    from Algorithms.analytic_progress.Calculate_Intervals.Multiple_Orderlines import detour_multiline as detour_multi_mod
    from Algorithms.analytic_progress import core as core_mod
    from Algorithms.analytic_progress.models import WarehouseInstance
    from Algorithms.analytic_progress.Calculate_Intervals.batch_combination_optimizer import (
        LayoutParams, combo_to_instance, get_aisle_combinations
    )
except ImportError:
    import continuous_calculation as cc
    import continuous_calculation_multiline as cc_multi
    import detour_multiline as detour_multi_mod
    import core as core_mod
    from models import WarehouseInstance
    from batch_combination_optimizer import LayoutParams, combo_to_instance, get_aisle_combinations


Row = Dict[str, Union[float, int, str]]


@dataclass(frozen=True)
class CompareParams:
    M: int = 8
    N_L: int = 16
    w: float = 1.0
    L: float = 17.0
    v: float = 1.0
    t_p: float = 0.0


def _parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _make_instance_from_aisles(params: CompareParams, aisles: List[int], n_list: List[int] | None) -> WarehouseInstance:
    if not aisles:
        raise ValueError("aisles must not be empty")
    if n_list is None:
        n_list = [1] * len(aisles)
    if len(n_list) != len(aisles):
        raise ValueError("Length of n_list must match aisles")
    return WarehouseInstance(
        M=params.M,
        N_L=params.N_L,
        w=params.w,
        L=params.L,
        A=aisles,
        n_list=n_list,
        v=params.v,
        t_p=params.t_p,
        P=[],
        arrival_times=[],
    )


def _excel_value(value: Union[float, int, str]) -> Union[float, int, str]:
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return value


def _autosize_worksheet_columns(worksheet) -> None:
    from openpyxl.utils import get_column_letter

    for col_idx, column_cells in enumerate(
        worksheet.iter_cols(min_row=1, max_row=worksheet.max_row, min_col=1, max_col=worksheet.max_column),
        start=1,
    ):
        max_len = 0
        for cell in column_cells:
            if cell.value is None:
                text = ""
            elif isinstance(cell.value, float) and math.isfinite(cell.value):
                text = str(cell.value)
            else:
                text = str(cell.value)
            max_len = max(max_len, len(text))
        worksheet.column_dimensions[get_column_letter(col_idx)].width = max(8, max_len + 2)


def export_multi_sheet_to_xlsx(sheet_rows, output_path):
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise RuntimeError("XLSX-Export nicht moeglich: openpyxl fehlt. pip install openpyxl") from exc

    wb = Workbook()
    wb.remove(wb.active)

    for sheet_name, rows in sheet_rows:
        ws = wb.create_sheet(title=(sheet_name or "sheet")[:31])
        if not rows:
            ws.append(["Hinweis", "Keine Daten fuer den Export vorhanden"])
            continue

        headers = list(rows[0].keys())
        ws.append(headers)
        for row in rows:
            ws.append([_excel_value(row.get(h, "")) for h in headers])

        for col_idx, header in enumerate(headers, start=1):
            for r_idx in range(2, ws.max_row + 1):
                cell = ws.cell(row=r_idx, column=col_idx)
                if isinstance(cell.value, float) and math.isfinite(cell.value):
                    cell.number_format = "0.00%" if header == "rel_diff" else "0.0000"

        _autosize_worksheet_columns(ws)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out))
    print(f"\nXLSX gespeichert: {out}")


def _default_output_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "Results" / f"compare_multiline_{ts}.xlsx"


def _iter_instances(args, params: CompareParams) -> List[Tuple[str, WarehouseInstance]]:
    if args.all_combos:
        combos = get_aisle_combinations(args.n_picks, params.M)
        combos = sorted(combos, key=lambda item: item[1], reverse=True)
        if args.max_combos is not None:
            combos = combos[: args.max_combos]
        instances: List[Tuple[str, WarehouseInstance]] = []
        for combo, _prob in combos:
            inst = combo_to_instance(combo, LayoutParams(M=params.M, N_L=params.N_L, w=params.w, L=params.L, v=params.v, t_p=params.t_p))
            instances.append((str(combo), inst))
        return instances

    aisles = _parse_int_list(args.aisles)
    n_list = _parse_int_list(args.n_list) if args.n_list else None
    inst = _make_instance_from_aisles(params, aisles, n_list)
    label = f"A={inst.A}, n_list={inst.n_list}"
    return [(label, inst)]


def _kappas_from_args(args) -> List[int]:
    if args.kappa_max is not None:
        return list(range(1, int(args.kappa_max) + 1))
    return [int(args.kappa)]


def _compact_rows(rows: Sequence[Row]) -> None:
    header = f"{'kappa':>5} {'t':>8} {'E_disc':>12} {'E_cont':>12} {'abs_diff':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['kappa']):>5} "
            f"{float(row['t']):>8.4f} "
            f"{float(row['E_disc']):>12.4f} "
            f"{float(row['E_cont']):>12.4f} "
            f"{float(row['abs_diff']):>12.4f}"
        )


def _build_summary_rows(rows: Sequence[Row]) -> List[Row]:
    grouped: Dict[int, List[Row]] = {}
    for row in rows:
        grouped.setdefault(int(row["kappa"]), []).append(row)

    summary_rows: List[Row] = []
    for kappa, group in sorted(grouped.items()):
        abs_diffs = [float(r["abs_diff"]) for r in group]
        rel_diffs = [float(r["rel_diff"]) for r in group]
        summary_rows.append(
            {
                "kappa": kappa,
                "n_rows": len(group),
                "mean_abs_diff": sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0,
                "max_abs_diff": max(abs_diffs) if abs_diffs else 0.0,
                "mean_rel_diff": sum(rel_diffs) / len(rel_diffs) if rel_diffs else 0.0,
            }
        )
    return summary_rows


def run_compare(args) -> Tuple[List[Row], List[Row]]:
    params = CompareParams(M=args.M, N_L=args.N_L, w=args.w, L=args.L, v=args.v, t_p=args.t_p)
    kappas = _kappas_from_args(args)
    instances = _iter_instances(args, params)
    rows: List[Row] = []

    for inst_label, inst in instances:
        T_in, T_out, T_end = core_mod.compute_segment_times(inst)
        n_steps = max(1, int(args.n_steps))
        times = [i * T_end / n_steps for i in range(0, n_steps + 1)]

        for kappa in kappas:
            for t in times:
                state = cc.estimate_continuous_route_state(t, T_in, T_out, inst)
                v_cont, r_cont, phase_label, active_j, local = cc._continuous_vr_state(t, T_in, T_out, inst)
                g_pass = core_mod.G_pass(state.x, t, inst)
                g_front = core_mod.G_front(state.x, t, inst)

                total_route_nodes = inst.k * inst.N_L

                eps = 1e-9
                i_full = int(round(v_cont * inst.k - local))
                i_full = max(0, min(inst.k, i_full))
                nodes_in_aisle = int(local * (inst.N_L + 1) + eps)
                nodes_in_aisle = max(0, min(inst.N_L, nodes_in_aisle))
                v_nodes = i_full * inst.N_L + nodes_in_aisle
                v_nodes = max(0, min(total_route_nodes, v_nodes))
                r_nodes = total_route_nodes - v_nodes

                p1, p2, p3, p4, p_sum, *_ = cc.continuous_class_probabilities(t, T_in, T_out, inst)
                if not isclose(p_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
                    raise RuntimeError(
                        f"p_sum ist nicht 1.0 fuer inst={inst_label}, kappa={kappa}, t={t:.6f}: {p_sum:.12f}"
                    )

                cont_est = cc_multi.continuous_expected_detour_multiline(t, inst, T_in, T_out, n_orderlines=kappa)
                disc_est = detour_multi_mod.calculate_detour_multiline(
                    inst,
                    t,
                    state.x,
                    state.y,
                    g_front,
                    g_pass,
                    v_nodes,
                    r_nodes,
                    kappa,
                    state.is_return_phase,
                    outbound=(phase_label == "phase_1_to_first_entry"),
                )

                e_disc = float(disc_est.expected_detour)
                e_cont = float(cont_est.expected_detour)
                abs_diff = abs(e_disc - e_cont)
                rel_diff = abs_diff / e_disc if e_disc != 0.0 else 0.0

                rows.append(
                    {
                        "instance": inst_label,
                        "kappa": kappa,
                        "t": float(t),
                        "x": float(state.x),
                        "y": float(state.y),
                        "phase": phase_label,
                        "return_flag": "R" if state.is_return_phase else "F",
                        "|Gp|": len(g_pass),
                        "|Gf|": len(g_front),
                        "E_disc": e_disc,
                        "E_cont": e_cont,
                        "abs_diff": abs_diff,
                        "rel_diff": rel_diff,
                    }
                )

    summary_rows = _build_summary_rows(rows)
    return rows, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vergleicht multiline diskrete und kontinuierliche Umwegberechnung.")
    parser.add_argument("--M", type=int, default=8, help="Anzahl Gassen")
    parser.add_argument("--N-L", dest="N_L", type=int, default=16, help="Knoten pro Gasse")
    parser.add_argument("--L", type=float, default=17.0, help="Gassenlaenge")
    parser.add_argument("--w", type=float, default=1.0, help="Gassenabstand")
    parser.add_argument("--v", type=float, default=1.0, help="Geschwindigkeit")
    parser.add_argument("--t-p", dest="t_p", type=float, default=0.0, help="Pickzeit")
    parser.add_argument("--aisles", type=str, default="1,2,8", help="Besuchte Gassen, z.B. '1,2,8'")
    parser.add_argument("--n-list", type=str, default="", help="Picks je Gasse, z.B. '1,1,1'")
    parser.add_argument("--kappa", type=int, default=2, help="Anzahl Orderlines")
    parser.add_argument("--kappa-max", type=int, default=None, help="Optionaler Sweep 1..K")
    parser.add_argument("--n-steps", type=int, default=50, help="Anzahl Zeitschritte")
    parser.add_argument("--output", type=str, default=None, help="XLSX-Ausgabepfad")
    parser.add_argument("--all-combos", action="store_true", help="Alle Gassenkombinationen iterieren")
    parser.add_argument("--n-picks", type=int, default=3, help="Picks fuer Kombinationen im All-Combos-Modus")
    parser.add_argument("--max-combos", type=int, default=None, help="Optional nur die ersten N Kombinationen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary_rows = run_compare(args)

    output = Path(args.output) if args.output else _default_output_path()
    export_multi_sheet_to_xlsx(
        [
            ("Multiline Compare", rows),
            ("Summary", summary_rows),
        ],
        output,
    )

    _compact_rows(rows)

    print("\nSummary:")
    for row in summary_rows:
        print(
            f"kappa={int(row['kappa'])} "
            f"mean_abs_diff={float(row['mean_abs_diff']):.4f} "
            f"max_abs_diff={float(row['max_abs_diff']):.4f} "
            f"mean_rel_diff={float(row['mean_rel_diff']):.2%}"
        )


if __name__ == "__main__":
    main()
