import abc
from typing import Tuple, List, Any

from dataclasses import dataclass
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy

from utils import save_to_pickle, query_txt_file


def distance_matrix_generator(G: nx.Graph) -> pd.DataFrame:
    """
    Creates a distance matrix of the pick nodes in the warehouse using the Floyd-Warshall algorithm.
    """
    # Compute the shortest path lengths between all pairs of nodes using Floyd-Warshall
    all_pairs_shortest_path_length = dict(nx.floyd_warshall_predecessor_and_distance(G)[1])

    # Convert the numpy matrix to a pandas DataFrame
    nodes = list(G.nodes())
    matrix = pd.DataFrame(all_pairs_shortest_path_length, index=nodes, columns=nodes)
    matrix.index.name = "index"
    return matrix


def distance_matrix_generator_scipy(G: nx.Graph):
    #distance_mat = nx.floyd_warshall_numpy(self.graph, self.nodes_list)
    nodes = list(G.nodes())  # Extract node labels
    A = nx.adjacency_matrix(G, nodelist=nodes).tolil()  # Ensure node order is consistent
    dist_mat = scipy.sparse.csgraph.floyd_warshall(A, directed=False, unweighted=False)

    # Convert NumPy array to Pandas DataFrame
    df = pd.DataFrame(dist_mat, index=nodes, columns=nodes)
    return df


def distance_matrix_generator_from_shortest_paths(G: nx.Graph, shortest_paths: dict):
    """Generates the distance matrix for a graph based on pre-calculated shortest-paths"""
    nodes = sorted(list(G.nodes))
    n_nodes = len(nodes)

    # Initialize distance matrix
    dist_mat = np.zeros((n_nodes, n_nodes))

    # Fill with shortest-path distances
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i == j:
                dist_mat[i, j] = 0
            else:
                # Use the pre-computed shortest paths if available
                # if (node_i, node_j) in shortest_paths:
                dist_mat[i, j] = shortest_paths[(node_i, node_j)]
                # else:
                #     # Calculate using networkx if not pre-computed
                #     try:
                #         dist_mat[i, j] = nx.shortest_path_length(
                #             G, source=node_i, target=node_j, weight='weight'
                #         )
                #     except nx.NetworkXNoPath:
                #         # No path exists
                #         dist_mat[i, j] = float('inf')

    df = pd.DataFrame(dist_mat, index=nodes, columns=nodes)
    print(df)
    return df


def tour_matrix_generator(G: nx.Graph) -> pd.DataFrame:
    """
    Creates a tour matrix of the pick nodes in the warehouse using the Floyd-Warshall algorithm.
    """
    # Compute the shortest path lengths between all pairs of nodes using Floyd-Warshall
    all_pairs_shortest_paths = dict(nx.floyd_warshall_predecessor_and_distance(G)[0])

    nodes = list(G.nodes())
    matrix = pd.DataFrame(index=nodes, columns=nodes)

    for i in nodes:
        for j in nodes:
            # Reconstruct the path from i to j using the predecessor information
            path = []
            current = j
            while current != i:
                path.insert(0, current)
                current = all_pairs_shortest_paths[i][current]
            path.insert(0, i)

            matrix.at[i, j] = path
    matrix.index.name = "index"
    return matrix

def predecessor_matrix_generator(G: nx.Graph) -> dict[Any, Any] | dict[str, Any] | dict[str, str] | dict[bytes, bytes]:
    """
    Creates a predecessor matrix of the pick nodes in the warehouse using the Floyd-Warshall algorithm.
    """
    # Compute the shortest path lengths between all pairs of nodes using Floyd-Warshall
    all_pairs_shortest_paths = dict(nx.floyd_warshall_predecessor_and_distance(G)[0])

    return all_pairs_shortest_paths


@dataclass
class GraphParameters:
    n_aisles: int
    n_pick_locations: int
    dist_change_aisle: float
    dist_aisle_to_pick_location: float
    dist_pick_locations: float
    dist_start: float
    dist_end: float
    start_location: tuple[int, int]
    end_location: tuple[int, int]
    start_connection_point: tuple[int, int]
    end_connection_point: tuple[int, int]


@dataclass
class GraphExplicitRepresentation:
    vertices: List
    arcs: List


class GraphGeneratorBase(abc.ABC):
    def __init__(self,
                 G: nx.Graph | None = None,
                 **kwargs):

        if G is not None:
            self.G = G
        else:
            self.G = nx.Graph()

    def populate_graph(self):
        ...

    def render(self, plot: bool = True):
        ...

    def dump(self, path_to_folder: str, file_name: str):
        save_to_pickle(path_to_folder, file_name, self.G)


class ExplicitGraphGenerator(GraphGeneratorBase):
    def __init__(self,
                 vertices_coords: dict,
                 arcs: List,
                 G: nx.Graph | None = None,
                 **kwargs):
        super().__init__(G, **kwargs)
        self.vertices_coords = vertices_coords
        self.arcs = arcs

    def _add_nodes(self):
        # Add all vertices
        for vertex_idx in self.vertices_coords:
            x_pos, y_pos, name = self.vertices_coords[vertex_idx]
            # self.G.add_node(vertex_idx, pos=(x_pos, y_pos), type=name)
            self.G.add_node((x_pos, y_pos), pos=(x_pos, y_pos), type=name)

    def _add_edges(self):
        # Add all edges with their distances
        for start, end, distance in self.arcs:
            # self.G.add_edge(start, end, weight=distance)
            self.G.add_edge(
                (self.vertices_coords[start][0],
                 self.vertices_coords[start][1]),
                (self.vertices_coords[end][0],
                 self.vertices_coords[end][1]), weight=distance)

    def populate_graph(self):
        self._add_nodes()
        self._add_edges()


    def render(self, plot: bool = True, out_name=False, dpi=700, font_size=5, node_size=50, node_color='lightblue') -> None:
        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw(self.G, pos=pos, with_labels=True, node_color=node_color, font_size=font_size, node_size=node_size)
        weight = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=weight, font_size=font_size)

        if plot:
            plt.show()

        if out_name:
            plt.savefig(out_name, dpi=dpi)


class ShelfStorageGraphGenerator(GraphGeneratorBase):
    def __init__(
            self,
            n_aisles: int,
            n_pick_locations: int,
            dist_aisle: float,
            dist_pick_locations: float,
            dist_aisle_location: float,
            dist_start: float,  # from start to graph
            dist_end: float,  # from end to graph
            start_location: tuple[int, int] = (0, 0),
            end_location: tuple[int, int] = (1, -1),
            start_connection_point: tuple[int, int] = (1, 0),
            end_connection_point: tuple[int, int] = (1, 0),
            G: nx.Graph | None = None,
            reverse_pick_nodes: bool = None,
            **kwargs):
        super().__init__(G, **kwargs)

        self.n_aisles = n_aisles
        self.n_pick_locations = n_pick_locations
        self.dist_aisle = dist_aisle
        self.dist_pick_locations = dist_pick_locations
        self.dist_aisle_location = dist_aisle_location
        self.dist_start = dist_start
        self.dist_end = dist_end
        self.end_location = end_location
        self.start_location = start_location
        self.start_connection_point = start_connection_point
        self.end_connection_point = end_connection_point
        self.reverse_pick_nodes = reverse_pick_nodes

    def _add_pick_nodes(self) -> None:

        for i in range(1, self.n_aisles + 1):
            for j in range(1, self.n_pick_locations + 1):
                if self.reverse_pick_nodes:
                    self.G.add_node((i, j),
                                    pos=(i, self.n_pick_locations - j + 1), type='pick_node')
                else:
                    self.G.add_node((i, j), pos=(i, j), type='pick_node')

    def _add_start_and_end_nodes(self) -> None:
        self.G.add_node(self.start_location, pos=self.start_location, type='start_node')
        self.G.add_node(self.end_location, pos=self.end_location, type='end_node')

    def _add_change_aisle_nodes(self) -> None:
        # add nodes for changing aisles bottom
        for i in range(1, self.n_aisles + 1):
            self.G.add_node((i, 0), pos=(i, 0), type='change_aisle_node')
        # add nodes for changing aisles top
        for i in range(1, self.n_aisles + 1):
            self.G.add_node((i, self.n_pick_locations + 1), pos=(i, self.n_pick_locations + 1), type='change_aisle_node')

    def _add_edges(self) -> None:

        for aisle in range(1, self.n_aisles + 1):

            # add edges between pick locations (vertical
            for location in range(1, self.n_pick_locations):
                self.G.add_edge((aisle, location),
                                (aisle, location + 1),
                                weight=self.dist_pick_locations)

            # add edges between pick locations and change aisle nodes (bottom)
            self.G.add_edge((aisle, 0),
                            (aisle, 1),
                            weight=self.dist_aisle_location)

            # add edges between pick locations and change aisle nodes (top)
            self.G.add_edge((aisle, self.n_pick_locations ),
                            (aisle, self.n_pick_locations + 1),
                            weight=self.dist_aisle_location)

            if aisle < self.n_aisles:
                # add edges between aisles (horizontal) (bottom)
                self.G.add_edge((aisle, 0),
                                (aisle + 1, 0),
                                weight=self.dist_aisle)

                # add edges between aisles (horizontal) (top)
                self.G.add_edge((aisle, self.n_pick_locations + 1),
                                (aisle + 1, self.n_pick_locations + 1),
                                weight=self.dist_aisle)

        # add edges between start and first aisle
        self.G.add_edge(self.start_location,
                        self.start_connection_point,
                        weight=self.dist_start)

        # add edges between end and last aisle
        self.G.add_edge(self.end_location,
                        self.end_connection_point,
                        weight=self.dist_end)

    def populate_graph(self):
        self._add_pick_nodes()
        self._add_change_aisle_nodes()
        self._add_start_and_end_nodes()
        self._add_edges()

    def render(self, plot: bool = True, out_name=False, dpi=700, font_size=5, node_size=50, node_color='lightblue') -> None:

        plt.figure(figsize=(12, 8))

        pos = nx.get_node_attributes(self.G, 'pos')
        colors = []
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get('type', '')

            if node_type == 'start_node':
                colors.append('green')
            elif node_type == 'end_node':
                colors.append('red')
            elif node_type == 'change_aisle_node':
                colors.append('gray')
            else:
                colors.append(node_color)

        nx.draw(self.G, pos=pos, with_labels=True, node_color=colors, font_size=font_size, node_size=node_size)
        weight = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=weight, font_size=font_size)

        if plot:
            plt.show()

        if out_name:
            plt.savefig(out_name, dpi=dpi, bbox_inches='tight')

def parse_tuple(value: str) -> tuple[int, ...]:
    """Parse a string representation of a tuple into a tuple of integers."""
    try:
        string = value.strip("()")
        values = string.split(",")
        return tuple(map(int, values))

    except Exception as e:
        raise ValueError(f"Fehler beim Parsen des Tupels: {value}") from e

def shelf_storage_graph(folder_input: str = None):
    """
    Create a storage graph based on the parameters in the given text file.

    Args:
        folder_input (str): Relative path to the folder containing the text file.

    Returns:
        ShelfStorageGraphGenerator: A storage graph generator object.
    """

    parameter_values = query_txt_file(folder_input)

    for key, value in parameter_values.items():
        try:
            if key == "start_location":
                start_location = parse_tuple(value)

            elif key == "end_location":
                end_location = parse_tuple(value)

            elif key == "start_connection_point":
                start_connection_point = parse_tuple(value)

            elif key == "end_connection_point":
                end_connection_point = parse_tuple(value)

            elif key == "dist_start":
                dist_start = float(value)

            elif key == "dist_end":
                dist_end = float(value)

            elif key == "dist_change_aisle":
                dist_change_aisle = float(value)

            elif key == "dist_aisle_to_pick_location":
                dist_aisle_to_pick_location = float(value)

            elif key == "dist_pick_locations":
                dist_pick_locations = float(value)

            elif key == "n_aisles":
                n_aisles = int(value)

            elif key == "n_pick_locations":
                n_pick_locations = int(value)

        except ValueError as e:
            print(f"Warnung: Fehler beim Verarbeiten von {key} mit Wert {value}. Grund: {e}")

    return ShelfStorageGraphGenerator(
            n_aisles,
            n_pick_locations,
            dist_change_aisle,
            dist_aisle_to_pick_location,
            dist_pick_locations,
            dist_start,
            dist_end,
            start_location,
            end_location,
            start_connection_point,
            end_connection_point,
    )

def create_graph(render: bool = False, folder_output: str = None, folder_input: str = None, save_dist_matrix: bool = True, save_tour_matrix: bool = True):
    """
    Create a storage graph based on the parameters in the given text file.
    """
    graph = shelf_storage_graph(folder_input)
    graph.populate_graph()
    graph.dump(folder_output, "graph")

    if render:
        graph.render()

    if save_dist_matrix:
        dist_matrix = distance_matrix_generator(graph.G)
        save_to_pickle(folder_output, "distance_matrix", dist_matrix)

    if save_tour_matrix:
        tour_matrix = tour_matrix_generator(graph.G)
        save_to_pickle(folder_output, "tour_matrix", tour_matrix)

    predecessor_matrix = predecessor_matrix_generator(graph.G)
    save_to_pickle(folder_output, "predecessor_matrix", predecessor_matrix)

