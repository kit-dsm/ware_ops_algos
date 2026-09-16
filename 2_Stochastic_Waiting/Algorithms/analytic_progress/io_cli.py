import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Batch analysis for analytic progress")

    parser.add_argument(
        "--orders-json",
        type=str,
        default=str(project_root / "Orders" / "articles_in_order_fixed_value_1" / "instance_1.json"),
        help="Pfad zur Orders-JSON fuer die Batchanalyse",
    )
    parser.add_argument(
        "--mapping-json",
        type=str,
        default=str(project_root / "Data_input" / "128_pick_nodes_16x8_random_article_id_pick_node_mapping.json"),
        help="Pfad zur Artikel-zu-Pickposition-Mapping-JSON",
    )

    # Warehouse-/Zeitparameter, die in main.py an analyze_batches durchgereicht werden.
    parser.add_argument("--M", type=int, default=8, help="Anzahl der Gassen (Standard: 8)")
    parser.add_argument("--N_L", type=int, default=16, help="Anzahl Pick-Nodes pro Gasse (Standard: 16)")
    parser.add_argument("--w", type=float, default=1.0, help="Gassenabstand w (Standard: 1.0)")
    parser.add_argument("--L", type=float, default=17.0, help="Gassenlaenge L (Standard: 17.0)")
    parser.add_argument("--v", type=float, default=1.0, help="Geschwindigkeit v (Standard: 1.0)")
    parser.add_argument("--t_p", type=float, default=1.0, help="Pickzeit t_p (Standard: 1.0)")

    parser.add_argument(
        "--batch-known-size",
        type=int,
        default=3,
        help="Anzahl bereits bekannter Orders pro Batchfenster (Standard: 3)",
    )
    parser.add_argument(
        "--batch-total-size",
        type=int,
        default=4,
        help="Gesamtgroesse des Batchfensters inklusive Insert-Order (Standard: 4)",
    )

    parser.add_argument(
        "--max-batches",
        dest="max_batches",
        type=int,
        default=9999,
        help="Maximale Anzahl auszuwertender Batches, z.B. 1 fuer nur den ersten (Standard: 1)",
    )
    parser.add_argument(
        "--expected-interarrival",
        type=float,
        default=28.8,
        help="Erwartete Interarrival-Zeit in Sekunden (Standard: 28.8)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(project_root / "Results"),
        help="Ausgabeordner fuer Result-Dateien (Standard: <project>/Results)",
    )

    return parser.parse_args()