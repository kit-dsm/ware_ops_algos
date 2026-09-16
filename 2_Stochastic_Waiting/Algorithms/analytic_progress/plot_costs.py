from pathlib import Path
import math
import matplotlib.pyplot as plt

from io_cli import parse_args
from order_optimizer import analyze_batches
from models import TimeSampleResult, BatchComparisonResult


def _plot_single_batch(result: BatchComparisonResult, output_dir: Path) -> Path:
    samples = result.samples  # type: list[TimeSampleResult]
    if not samples:
        raise RuntimeError(f"Batch {result.batch_index}: Keine Zeitstichproben im Batch-Ergebnis.")

    t_values = []
    j_values = []
    oct_pred_values = []

    initial_mean_oct = samples[0].mean_initial_order_completion_time

    for sample in samples:
        t_values.append(sample.t)
        j_values.append(sample.waiting_time_to_estimated + sample.expected_detour)
        oct_pred_values.append(sample.mean_predicted_order_completion_time)

    best_idx = min(
        range(len(samples)),
        key=lambda i: (oct_pred_values[i], t_values[i]),
    )
    best_t = t_values[best_idx]
    best_j = j_values[best_idx]
    best_oct = oct_pred_values[best_idx]

    finite_j = [(t, j) for t, j in zip(t_values, j_values) if math.isfinite(t) and math.isfinite(j)]
    finite_oct = [(t, oct_value) for t, oct_value in zip(t_values, oct_pred_values) if math.isfinite(t) and math.isfinite(oct_value)]
    if not finite_j:
        raise RuntimeError(f"Batch {result.batch_index}: Keine endlichen Werte fuer J(t) vorhanden.")
    if not finite_oct:
        raise RuntimeError(f"Batch {result.batch_index}: Keine endlichen Werte fuer mean OCT vorhanden.")

    t_values_j, j_values_finite = zip(*finite_j)
    t_values_oct, oct_values_finite = zip(*finite_oct)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t_values_j, j_values_finite, label=r"$J(t) = \widehat{W}(t) + \mathbb{E}[D(t)]$", color="tab:blue", linewidth=2.0, zorder=3)
    ax1.set_xlabel("Zeit t entlang der Basisroute [s]")
    ax1.set_ylabel("Zusatzkosten J(t)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axvline(best_t, color="tab:blue", linestyle=":", alpha=0.3)
    ax1.scatter([best_t], [best_j], color="tab:blue", s=40)

    ax2 = ax1.twinx()
    ax2.patch.set_visible(False)
    ax2.plot(t_values_oct, oct_values_finite, label="mean OCT (predicted)", color="tab:orange", linewidth=2.0)
    ax2.axhline(initial_mean_oct, color="tab:gray", linestyle="--", label="initial mean OCT (base only)")
    ax2.set_ylabel("Mean Order Completion Time", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.axvline(best_t, color="tab:red", linestyle=":", label="t* (min OCT) = {:.2f}s".format(best_t))
    ax2.scatter([best_t], [best_oct], color="tab:red", s=40)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(
        f"Batch {result.batch_index}: J(t) und erwartete mean OCT(t) | "
        f"Base={result.base_order_ids}, Insert={result.insert_order_id}"
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"batch_{result.batch_index:03d}_cost_oct.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_cost_and_oct_for_first_batches() -> None:
    """Analysiert die per CLI konfigurierten ersten Batches und speichert je Batch einen Plot."""
    args = parse_args()
    results = analyze_batches(
        args.orders_json,
        args.mapping_json,
        M=args.M,
        N_L=args.N_L,
        w=args.w,
        L=args.L,
        v=args.v,
        t_p=args.t_p,
        known_size=args.batch_known_size,
        total_size=args.batch_total_size,
        max_batches=args.max_batches,
        engine="discrete",
    )
    if not results:
        print("Keine Batch-Ergebnisse vorhanden.")
        return

    output_dir = Path(args.results_dir) / "batch_plots"
    print(f"Erzeuge Plots für {len(results)} Batch(es) nach: {output_dir}")

    for result in results:
        try:
            out_path = _plot_single_batch(result, output_dir)
            print(f"Plot gespeichert: {out_path}")
        except RuntimeError as exc:
            print(f"Uebersprungen: {exc}")


if __name__ == "__main__":
    plot_cost_and_oct_for_first_batches()
