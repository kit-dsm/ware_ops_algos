import pickle
import networkx as nx
import json
import pandas as pd
from typing import List, Tuple, Optional, Any, Dict
from datetime import datetime, timedelta
from Domain.entities import Order
from pathlib import Path
import importlib.util
import matplotlib
import os
from glob import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_graph_information(graph_file_path: Path) -> Tuple[nx.Graph, List, Tuple, Tuple, List, pd.DataFrame, pd.DataFrame]:
    """
    Extract information from a graph file and prepare it for simulation.

    Args:
        graph_file_path: Path to the pickle file containing the graph

    Returns:
        A tuple containing:
        - graph: The NetworkX graph
        - pick_nodes: List of nodes where items can be picked
        - start_node: The starting node for routes
        - end_node: The ending node for routes
        - nodes: List of all nodes in the graph
        - dist_matrix: Distance matrix between nodes
        - tour_matrix: Matrix of paths between nodes

    Raises:
        FileNotFoundError: If the graph file does not exist
        ValueError: If the graph does not have a valid structure
    """
    try:
        with open(graph_file_path, 'rb') as file:
            graph = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Graph file not found at {graph_file_path}")
    except Exception as e:
        raise ValueError(f"Error loading graph file: {str(e)}")

    if not isinstance(graph, nx.Graph):
        raise ValueError("The loaded object is not a valid NetworkX graph")

    # Extract pick nodes (nodes where items can be picked)
    pick_nodes = [node for node, data in graph.nodes(data=True) if data.get('type') == 'pick_node']

    # Find start and end nodes
    start_node, end_node = search_for_start_and_end_node(graph)

    # If no start node is found, use the end node as the start node as well
    if start_node is None:
        for node in graph.nodes:
            if graph.nodes[node].get('type') == 'end_node':
                # Add 'start_node' type if it doesn't exist yet
                current_type = graph.nodes[node].get('type')
                if 'start_node' not in (current_type if isinstance(current_type, list) else [current_type]):
                    if isinstance(current_type, list):
                        graph.nodes[node]['type'].append('start_node')
                    else:
                        graph.nodes[node]['type'] = [current_type, 'start_node']

        # Search again for start and end nodes
        start_node, end_node = search_for_start_and_end_node(graph)

    # Verify that start and end nodes were found
    if start_node is None or end_node is None:
        try:
            for node, data in graph.nodes(data=True):
                # Make sure that data.get('type') has a value
                node_type = data.get('type')

                if node_type is None:
                    continue

                # Check if node_type is a string or a list of strings
                if isinstance(node_type, str):
                    if 'depot' in node_type:
                        start_node = node
                        end_node = node

                elif isinstance(node_type, list):
                    if any('depot' in item for item in node_type):
                        start_node = node
                        end_node = node
        except TypeError:
            raise ValueError("Could not identify start and/or end nodes in the graph")

    # Get all nodes
    nodes = list(graph.nodes)

    # Generate distance and tour matrices
    dist_matrix = distance_matrix_generator(graph)
    tour_matrix = tour_matrix_generator(graph)

    return graph, pick_nodes, start_node, end_node, nodes, dist_matrix, tour_matrix

def distance_matrix_generator(G=nx.Graph()) -> pd.DataFrame:
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

def tour_matrix_generator(G=nx.Graph()) -> pd.DataFrame:
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

def search_for_start_and_end_node(graph: nx.Graph) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """
    Search for start and end nodes in a graph.

    Args:
        graph: NetworkX graph to search in

    Returns:
        A tuple containing:
        - start_node: The start node if found, None otherwise
        - end_node: The end node if found, None otherwise
    """
    start_node = None
    end_node = None

    for node, data in graph.nodes(data=True):
        # Make sure that data.get('type') has a value
        node_type = data.get('type')

        if node_type is None:
            continue

        # Check if node_type is a string or a list of strings
        if isinstance(node_type, str):
            if 'start_node' in node_type:
                start_node = node

            if 'end_node' in node_type:
                end_node = node

        elif isinstance(node_type, list):
            if any('start_node' in item for item in node_type):
                start_node = node

            if any('end_node' in item for item in node_type):
                end_node = node

    return start_node, end_node

def read_sku_mapping(sku_mapping_path: Path) -> Dict[int, Tuple]:
    """
    Read SKU to storage location mapping from a JSON file.

    Args:
        sku_mapping_path: Path to the JSON file containing the SKU mapping

    Returns:
        A dictionary mapping SKU IDs (integers) to storage locations (tuples)

    Raises:
        FileNotFoundError: If the mapping file does not exist
        ValueError: If the file is not valid JSON or has an unexpected format
    """
    try:
        with open(sku_mapping_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"SKU mapping file not found at {sku_mapping_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in SKU mapping file: {str(e)}")

    try:
        # Convert keys to integers and values to tuples
        result_dict = {int(k): tuple(v) for k, v in data.items()}
        return result_dict
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unexpected format in SKU mapping file: {str(e)}")

def create_batched_list(batch: 'Batch', sku_storage_mapping: Dict[int, Tuple]) -> pd.DataFrame:
    """
    Creates a pandas DataFrame (batched_list) from a Batch instance
    and an SKU-to-storage mapping.

    Args:
        batch: An instance of the Batch class.
        sku_storage_mapping: A dictionary mapping SKU IDs to pick nodes.

    Returns:
        A pandas DataFrame containing the batched list with columns:
        - order_number: ID of the order
        - batch_number: ID of the batch
        - pick_node: Location where the item can be picked
        - amount: Quantity of the item to pick

    Raises:
        ValueError: If the batch has no orders
    """
    if not hasattr(batch, 'orders') or not batch.orders:
        raise ValueError("Batch has no orders")

    data = []
    missing_pick_nodes = []

    for order in batch.orders:
        for item_id, amount in order.items.items():
            pick_node = sku_storage_mapping.get(item_id)  # Get pick node from mapping
            if pick_node:
                data.append({
                    "order_number": order.id,
                    "batch_number": batch.id,
                    "pick_node": pick_node,
                    "amount": amount
                })
            else:
                missing_pick_nodes.append((order.id, item_id))

    if missing_pick_nodes:
        import logging

        logger = logging.getLogger(__name__)
        for order_id, item_id in missing_pick_nodes:
            logger.warning(f"No pick node found for item {item_id} in order {order_id}")
            print(sku_storage_mapping)

    if not data:
        raise ValueError("No valid items found in batch")

    batched_list = pd.DataFrame(data)
    return batched_list


def get_current_or_next_node(
        tour: List[Any],
        orders: List['Order'],
        dist_matrix: pd.DataFrame,
        time_to_pick: float,
        speed: float,
        start_time: datetime,
        current_time: datetime,
        sku_storage_mapping: Dict[Any, Any]
) -> Any:
    """
    Calculate the current or next node in a tour based on elapsed time.

    Args:
        tour: List of nodes in the tour
        orders: List of Order objects in the batch
        dist_matrix: Distance matrix between nodes
        time_to_pick: Time required to pick one item (in seconds)
        speed: Speed of the vehicle (in units per second)
        start_time: Time when the tour started
        current_time: Current time
        sku_storage_mapping: Mapping from SKU to storage location

    Returns:
        The current or next node in the tour
    """
    batch_articles_quantity_sum = {}

    # Process all orders in the batch
    for order in orders:
        for article_id, quantity in order.items.items():
            # Sum amount if article is already in dict
            if article_id in batch_articles_quantity_sum:
                batch_articles_quantity_sum[article_id] += quantity
            else:
                batch_articles_quantity_sum[article_id] = quantity

    # Berechne die verstrichene Zeit in Sekunden seit Tourbeginn
    elapsed = (current_time - start_time).total_seconds()

    # Starte an der Position des ersten Knotens in der Tour
    cumulative_time = 0.0
    pick_nodes = {}

    for article, amount in batch_articles_quantity_sum.items():
        article_loc = sku_storage_mapping.get(article)
        pick_nodes[article_loc] = amount

    # Iteriere über alle Segmente der Tour
    for i in range(len(tour) - 1):
        start_node = tour[i]
        next_node = tour[i + 1]
        # Hole die Distanz zwischen den Knoten aus der Distanzmatrix
        distance = dist_matrix.at[start_node, next_node]
        # Berechne die Reisezeit für das Segment (in Sekunden)
        travel_time = distance / speed

        # Prüfe, ob das Fahrzeug in diesem Segment noch unterwegs ist
        if cumulative_time + travel_time >= elapsed:
            # Fahrzeug ist auf dem Weg von start_node zu next_node.
            # Wir geben hier den nächsten Knoten als Ziel zurück.
            return next_node

        # Andernfalls wurde dieses Segment bereits vollständig zurückgelegt
        cumulative_time += travel_time

        # Falls der nächste Knoten ein Pick-Node ist, addiere die Pick-Zeit

        if next_node in pick_nodes:
            amount = pick_nodes[next_node]
            del pick_nodes[next_node]
            if cumulative_time + time_to_pick > elapsed:
                # Das Fahrzeug befindet sich aktuell im Pick-Prozess an diesem Knoten
                return next_node
            cumulative_time += time_to_pick * amount

    # Falls alle Segmente bereits durchlaufen wurden, befindet sich das Fahrzeug am Endknoten.
    return tour[-1]

def setup_matplotlib_backend():
    """
    Prüft, ob Tkinter verfügbar ist.
    Wenn nicht, wird matplotlib auf das Agg-Backend umgestellt,
    um Fehler auf headless-Systemen zu vermeiden.
    """
    if importlib.util.find_spec("tkinter") is None:
        print("Tkinter nicht gefunden – matplotlib auf Agg-Backend umstellen.")
        matplotlib.use("Agg")
    else:
        print("Tkinter verfügbar – normales GUI-Backend wird verwendet.")

def query_txt_file(rel_folder_path):

 #   folder_path = os.path.join(os.path.dirname(__file__), rel_folder_path)
 #   folder_path =  rel_folder_path

    txt_files = glob.glob(f"{rel_folder_path}/*.txt")

    if len(txt_files) != 1:
        raise FileNotFoundError("There must be exactly one .txt file in the folder.")
    # all information to be collected in one dict#
#    all_subsystems = {}
 #   for txt_file in txt_files:
#        txt_path = txt_file

    txt_path = txt_files[0]

    parameter_values = {}
    with open(txt_path, 'r') as file:
        for line in file:
            # Remove leading and trailing spaces
            line = line.strip()
            # Check if line is empty; if yes, continue
            if not line:
                continue
            # Separate line at first ':'
            if ':' in line:
                key, value = line.split(':', 1)
                # Remove leading and trailing spaces
                key = key.strip()
                value = value.strip()

                if key.lower() == "end":
                    break

                parameter_values[key] = value


            else:
                # skip line if no ':' is found
                continue

#        all_subsystems[txt_file] = parameter_values

 #   return all_subsystems
    return parameter_values

def save_to_pickle(path, file_name, data):
    filename_with_extension = f"{file_name}.pkl"
    full_path = os.path.join(path, filename_with_extension)
    if os.path.exists(full_path):
        overwrite = True
            #get_bool(str(input(f"Datei {file_name} existiert bereits.Überschreiben? (True / False):")).lower()))
        if not overwrite:
            new_file_name = input("Bitte gib einen neuen Dateinamen an: ")
            save_to_pickle(path, new_file_name, data)
            return

    with open(full_path, 'wb') as opened_file:
        pickle.dump(data, opened_file)
    print(f"Data saved in '{full_path}'.")

def debug_sku_allocation_consistency(warehouse) -> dict:
    """
    Debug helper to check consistency between graph nodes and SKU allocation.

    Args:
        warehouse: The warehouse object to analyze

    Returns:
        Dictionary with diagnostic information
    """
    graph_nodes = set(warehouse.graph.nodes())
    sku_locations = set(warehouse.allocation.values())

    # Check for duplicate locations (multiple SKUs per location)
    from collections import Counter
    location_counts = Counter(warehouse.allocation.values())
    locations_with_multiple_skus = {
        loc: count for loc, count in location_counts.items() if count > 1
    }

    diagnostics = {
        'total_graph_nodes': len(graph_nodes),
        'total_skus': len(warehouse.allocation),
        'unique_pick_locations': len(sku_locations),
        'nodes_without_sku': len(graph_nodes - sku_locations),
        'sku_locations_not_in_graph': len(sku_locations - graph_nodes),
        'locations_with_multiple_skus': len(locations_with_multiple_skus),
        'sample_multi_sku_locations': dict(list(locations_with_multiple_skus.items())[:5])
    }

    # Log Diagnostics Header
    logger.info("=" * 70)
    logger.info("SKU ALLOCATION DIAGNOSTICS")
    logger.info("=" * 70)

    # Log each diagnostic item
    for key, value in diagnostics.items():
        logger.info(f"{key:.<50} {value}")

    logger.info("=" * 70)

    # Warning für mehrere SKUs pro Location
    if diagnostics['locations_with_multiple_skus'] > 0:
        logger.warning(
            f"{diagnostics['locations_with_multiple_skus']} locations have multiple SKUs assigned. "
            f"This is handled correctly by the fixed _make_dummy_order_from_nodes()")

        # Log sample locations
        if diagnostics['sample_multi_sku_locations']:
            logger.info(f"Sample multi-SKU locations: {diagnostics['sample_multi_sku_locations']}")

    # Warnung für Nodes ohne SKU
    if diagnostics['nodes_without_sku'] > 0:
        logger.info(
            f"{diagnostics['nodes_without_sku']} graph nodes have no SKU assignment "
            f"(likely corridor/transit nodes)")

    # Warnung für SKU-Locations nicht im Graph
    if diagnostics['sku_locations_not_in_graph'] > 0:
        logger.warning(
            f"{diagnostics['sku_locations_not_in_graph']} SKU locations are NOT in the warehouse graph! "
            f"This may cause routing issues.")

    return diagnostics





