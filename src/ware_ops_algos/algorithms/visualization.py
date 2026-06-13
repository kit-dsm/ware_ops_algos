from datetime import timezone
from matplotlib.patches import Patch
from typing import Any, Callable

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ware_ops_algos.algorithms import ScheduledJob


def render_graph(G, plot: bool = True, out_name=False, draw_edge_labels=False, with_labels=False, dpi=700, font_size=5, node_size=50, node_color='lightblue') -> None:
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos, with_labels=with_labels, node_color=node_color, font_size=font_size, node_size=node_size)
    weight = nx.get_edge_attributes(G, 'weight')
    if draw_edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weight, font_size=font_size)

    if out_name:
        plt.savefig(out_name, dpi=dpi)

    if plot:
        plt.show()


def plot_route(network_graph: nx.Graph, route: list[tuple[int, int]]):
    """Visualizes a picker route"""
    pos = nx.get_node_attributes(network_graph, 'pos')
    plt.figure(figsize=(10, 8))
    plt.title("Route Visualization")

    nx.draw(network_graph, pos, with_labels=False, node_size=30,
            node_color='lightgray', edge_color='lightgray', alpha=0.3)

    edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
    nx.draw_networkx_edges(network_graph, pos, edgelist=edges,
                           edge_color='red', width=6)


    plt.show()


def plot_route_with_directions(network_graph: nx.Graph, route: list[tuple[int, int]]):
    """Visualizes a picker route with direction arrows, including repeated edges."""
    pos = nx.get_node_attributes(network_graph, 'pos')
    plt.figure(figsize=(10, 8))
    plt.title("Route with Directions")

    # Draw the base graph
    nx.draw(network_graph, pos, with_labels=False, node_size=30,
            node_color='lightgray', edge_color='lightgray', alpha=0.3)
    # Count edge directions
    edge_counts = {}
    for i in range(len(route) - 1):
        start, end = route[i], route[i + 1]
        key = (start, end)
        edge_counts[key] = edge_counts.get(key, 0) + 1

    for i in range(len(route) - 1):
        start, end = route[i], route[i + 1]
        x1, y1 = pos[start]
        x2, y2 = pos[end]

        dx = x2 - x1
        dy = y2 - y1
        offset = 0.05 * (edge_counts[(start, end)] - 1) + 0.1
        norm = (dx ** 2 + dy ** 2) ** 0.5
        offset_x = -dy / norm * offset
        offset_y = dx / norm * offset

        plt.arrow(x1 + offset_x, y1 + offset_y,
                  dx * 0.9, dy * 0.9,
                  length_includes_head=True,
                  head_width=0.5,
                  head_length=0.2,
                  fc='red', ec='red')

    plt.axis('off')
    plt.show()


def plot_picker_gantt(assignments, use_datetime=False, tz=timezone.utc, figsize=(12, 6)):
    """
    Visualize batch assignments per picker as a Gantt chart.
    - assignments: list of dicts (output from schedule_routing_batches) or a DataFrame
                   required keys/cols: picker_id, batch_idx, start_time, end_time
    - use_datetime: if True, convert epoch seconds to datetimes for the x-axis
    - tz: timezone for datetime display when use_datetime=True
    """
    df = pd.DataFrame(assignments) if not isinstance(assignments, pd.DataFrame) else assignments.copy()

    df = df.sort_values(["picker_id", "start_time", "batch_idx"]).reset_index(drop=True)

    x_start = df["start_time"].min()
    x_end = df["end_time"].max()
    pickers = sorted(df["picker_id"].unique())
    picker_to_y = {pid: i for i, pid in enumerate(pickers)}

    if use_datetime:
        df["_x_start"] = pd.to_datetime(df["start_time"], unit="s", utc=True).dt.tz_convert(tz)
        df["_x_end"] = pd.to_datetime(df["end_time"], unit="s", utc=True).dt.tz_convert(tz)
        xtitle = f"Time ({tz})"
    else:
        df["_x_start"] = df["start_time"]
        df["_x_end"] = df["end_time"]
        xtitle = "Time (units)"

    fig, ax = plt.subplots(figsize=figsize)

    height = 0.8  # lane height
    for _, row in df.iterrows():
        y = picker_to_y[row["picker_id"]]
        start = row["_x_start"]
        end = row["_x_end"]
        width = end - start
        ax.barh(y=y, width=width, left=start, height=height, align="center", edgecolor="black")
        # Label with batch_idx (small)
        ax.text(start, y, f"#{int(row['batch_idx'])}", va="center", ha="left", fontsize=8)

    ax.set_yticks(list(picker_to_y.values()))
    ax.set_yticklabels([f"Picker {pid}" for pid in pickers])

    ax.set_xlabel(xtitle)
    ax.set_title("Batch Assignment per Picker (Gantt)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    if not use_datetime:
        ax.set_xlim(left=min(x_start, df["_x_start"].min()), right=max(x_end, df["_x_end"].max()))
    fig.tight_layout()
    plt.show()


def plot_gantt(
        jobs: list[ScheduledJob],
        *,
        ax: Axes | None = None,
        row_key: Callable[[Any], Any] = lambda j: j.picker_id,
        color_key: Callable[[Any], str] | None = None,
        row_label: Callable[[Any], str] = str,
        time_scale: float = 1.0,
        bar_height: float = 0.8,
        title: str | None = None,
        xlabel: str = "time",
        ylabel: str = "picker",
) -> Axes:
    """Plot a Gantt chart of ScheduledJob instances.

    Each job is drawn as a horizontal bar from start_time to end_time on the row
    given by row_key(job). Default coloring marks tardy jobs red, on-time jobs green.

    Parameters
    ----------
    jobs : sequence of ScheduledJob
        Must expose start_time, end_time, and the attribute used by row_key/color_key.
    ax : matplotlib Axes, optional
        Axes to draw on. Created if None.
    row_key : callable
        Maps a job to its row identifier. Default: picker_id.
    color_key : callable, optional
        Maps a job to a matplotlib color. Default: red if tardiness > 0 else green.
    row_label : callable
        Formats a row identifier into a y-tick label.
    time_scale : float
        Divisor applied to start_time and end_time (e.g. 3600 for seconds -> hours).
    bar_height : float
        Bar thickness in axis units; rows are spaced 1 apart.
    title, xlabel, ylabel : str
        Plot annotations.
    """
    if color_key is None:
        def color_key(j: Any) -> str:
            return "tab:red" if getattr(j, "tardiness", 0.0) > 0 else "tab:green"

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    rows = sorted({row_key(j) for j in jobs}, key=lambda r: (r is None, r))
    row_to_y = {r: i for i, r in enumerate(rows)}

    for j in jobs:
        y = row_to_y[row_key(j)]
        x0 = j.start_time / time_scale
        width = (j.end_time - j.start_time) / time_scale
        ax.broken_barh(
            [(x0, width)],
            (y - bar_height / 2, bar_height),
            facecolors=color_key(j),
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_yticks(list(row_to_y.values()))
    ax.set_yticklabels([row_label(r) for r in rows])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.invert_yaxis()

    # Legend only meaningful with the default color_key
    if color_key.__name__ == "color_key":
        ax.legend(
            handles=[
                Patch(facecolor="tab:green", edgecolor="black", label="on time"),
                Patch(facecolor="tab:red", edgecolor="black", label="tardy"),
            ],
            loc="upper right",
        )

    return ax



