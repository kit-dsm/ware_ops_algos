from Utils.utils import setup_matplotlib_backend
setup_matplotlib_backend()

import matplotlib.pyplot as plt
import networkx as nx
import pickle


def load_graph_from_pickle(path: str) -> nx.Graph:
    with open(path, 'rb') as f:
        return pickle.load(f)

def scale_positions(pos: dict, scale_x: float = 1.0, scale_y: float = 1.0) -> dict:
    """Skaliert x- und y-Koordinaten separat."""
    return {node: (x * scale_x, y * scale_y) for node, (x, y) in pos.items()}

def print_node_types(graph: nx.Graph):
    pick_nodes = []
    change_aisle_nodes = []
    start_node = None
    end_node = None

    for node, data in graph.nodes(data=True):
        node_type = data.get("type")
        if node_type == "pick_node":
            pick_nodes.append(node)
        elif node_type == "change_aisle_node":
            change_aisle_nodes.append(node)
        elif node_type == "start_node":
            start_node = node
        elif node_type == "end_node":
            end_node = node

    print("\n📦 Pick Nodes:")
    print(sorted(pick_nodes))

    print("\n🔀 Change Aisle Nodes:")
    print(sorted(change_aisle_nodes))

    print("\n🚩 Start Node:", start_node)
    print("🏁 End Node:", end_node)

    return {
        "pick_nodes": pick_nodes,
        "change_aisle_nodes": change_aisle_nodes,
        "start_node": start_node,
        "end_node": end_node
    }

def add_start_end_edge(graph: nx.Graph, start_node, end_node):
    """Fügt eine Kante mit Gewicht 0 zwischen Start- und Endknoten hinzu (falls beide vorhanden)."""
    if start_node is not None and end_node is not None:
        if not graph.has_edge(start_node, end_node):
            graph.add_edge(start_node, end_node, weight=0)
            print(f"\n➕ Edge added: {start_node} ↔ {end_node} (weight=0)")
        else:
            print(f"\nℹ️  Edge between {start_node} and {end_node} already exists.")
    else:
        print("\n⚠️  Start- oder Endknoten nicht gefunden – keine Kante hinzugefügt.")

def visualize_graph_colored(graph: nx.Graph,
                            node_categories: dict,
                            out_file: str = None,
                            font_size=5,
                            node_size=80,
                            dpi=300,
                            scale_x=2.0,
                            scale_y=2.5,
                            figsize: tuple[float, float] = (10, 6)):
    pos_original = nx.get_node_attributes(graph, 'pos')
    pos = scale_positions(pos_original, scale_x=scale_x, scale_y=scale_y)
    edge_labels = nx.get_edge_attributes(graph, 'weight')

    plt.figure(figsize=figsize)

    color_map = {
        "pick_nodes": 'skyblue',
        "change_aisle_nodes": 'lightgray',
        "start_node": 'green',
        "end_node": 'red'
    }

    nx.draw_networkx_nodes(graph, pos,
                           nodelist=node_categories['pick_nodes'],
                           node_color=color_map['pick_nodes'],
                           node_size=node_size,
                           label="Pick Nodes")

    nx.draw_networkx_nodes(graph, pos,
                           nodelist=node_categories['change_aisle_nodes'],
                           node_color=color_map['change_aisle_nodes'],
                           node_size=node_size,
                           label="Cross Aisle Nodes")

    if node_categories['start_node']:
        nx.draw_networkx_nodes(graph, pos,
                               nodelist=[node_categories['start_node']],
                               node_color=color_map['start_node'],
                               node_size=node_size + 30,
                               label="Start Node")

    if node_categories['end_node']:
        nx.draw_networkx_nodes(graph, pos,
                               nodelist=[node_categories['end_node']],
                               node_color=color_map['end_node'],
                               node_size=node_size + 30,
                               label="End Node")

    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos, font_size=font_size)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=font_size)

    # ── Legende ──────────────────────────────────────────────────────────────
    plt.legend(loc="upper center",
               bbox_to_anchor=(0.5, -0.02),
               ncol=2,
               fontsize=8,
               framealpha=0.9,
               title="Node types")

    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    #plt.legend(loc="lower center")

    if out_file:
        plt.savefig(out_file, dpi=dpi)
        print(f"\n🖼️  Graph saved as image at: {out_file}")
    else:
        plt.show()


if __name__ == "__main__":
    path_to_graph = r"U:\Diss\1_Online_Waiting_Strategies\Data_input\128_pick_nodes_16x8.pkl"  # <-- Pfad anpassen

    G = load_graph_from_pickle(path_to_graph)
    node_types = print_node_types(G)

    add_start_end_edge(G, node_types['start_node'], node_types['end_node'])

    visualize_graph_colored(G,
                            node_types,
                            out_file=r"U:\Diss\1_Online_Waiting_Strategies\Data_output\128_pick_nodes_16x8 - legend.png",
                            scale_x=1,
                            scale_y=1,
                            figsize=(16, 8))