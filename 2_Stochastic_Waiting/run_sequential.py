from Utils import logging_config
import os
import logging
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Union
from scipy import stats
import numpy as np
import time
import gc
import pyarrow as pa
import pyarrow.parquet as pq

from Simulation.core import reset_global_counters, SimulationEngine, SimulationState, SimulationControl
from Domain.entities import Order, Vehicle, Warehouse
from Utils import utils, sim_visualizer
from Algorithms.batching import BatchingPolicy, FIFOBatching
from Algorithms.dispatching import DispatchingPolicy, GreedyDispatcher
from Algorithms.routing import (Routing, SShapeAnalyticRouting, SShapeAnalyticRoutingAdvanced, SShapeRouting)  # ExactTSPRouting
from Algorithms.waiting import WaitingPolicy, StartImmediatelyPolicy, WaitingOrderCountPolicy
from Algorithms.initial_route import AllNodes

logger = logging.getLogger(__name__)

# Cache for graph data to avoid reloading
GRAPH_CACHE = {}

def extract_instance_metadata(orders_data: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract graph and SKU file paths from instance metadata.

    Args:
        orders_data: Dictionary containing instance data with 'meta' key

    Returns:
        Tuple of (graph_file_path, sku_file_path) or (None, None) if not found
    """
    try:
        meta = orders_data.get('meta', {})
        graph_path = meta.get('layout')
        sku_path = meta.get('storage_assignment')

        if graph_path and sku_path:
            return str(graph_path), str(sku_path)
        else:
            logger.warning("Missing 'layout' or 'storage_assignment' in metadata")
            return None, None

    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return None, None


def validate_instance_files(graph_file: str, sku_file: str, instance_key: str) -> bool:
    """
    Validate that instance-specific files exist and are readable.

    Args:
        graph_file: Path to graph file
        sku_file: Path to SKU file
        instance_key: Instance identifier for logging

    Returns:
        True if files are valid, False otherwise
    """
    try:
        graph_path = Path(graph_file)
        sku_path = Path(sku_file)

        if not graph_path.exists():
            logger.error(f"Graph file not found for {instance_key}: {graph_file}")
            return False

        if not sku_path.exists():
            logger.error(f"SKU file not found for {instance_key}: {sku_file}")
            return False

        # Test if files are readable
        if not graph_path.is_file() or not sku_path.is_file():
            logger.error(f"Files are not regular files for {instance_key}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating files for {instance_key}: {e}")
        return False


def load_orders_from_multiple_folders(orders_folders: List[Union[str, Path]]) -> Dict[str, dict]:
    """
    Load all orders from multiple folders.
    Creates unique keys by prefixing with folder name to avoid conflicts.
    Also validates that each instance has the required metadata.

    Args:
        orders_folders: List of folder paths containing JSON order files

    Returns:
        Dictionary with keys format: "foldername_filename.json"
    """
    orders_dict = {}
    failed_instances = []

    for folder in orders_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(f"Orders folder does not exist: {folder_path}")
            continue

        folder_name = folder_path.name
        json_files = list(folder_path.glob("*.json"))

        logger.info(f"Loading {len(json_files)} files from {folder_path}")

        for file in json_files:
            # Create unique key: foldername_filename
            unique_key = f"{folder_name}_{file.name}"

            # Check for duplicates
            if unique_key in orders_dict:
                logger.warning(f"Duplicate key found: {unique_key}, skipping...")
                continue

            try:
                with open(file, 'r') as f:
                    instance_data = json.load(f)

                # Validate instance structure
                if 'meta' not in instance_data:
                    logger.error(f"Missing 'meta' key in {file}")
                    failed_instances.append(unique_key)
                    continue

                if 'orders' not in instance_data:
                    logger.error(f"Missing 'orders' key in {file}")
                    failed_instances.append(unique_key)
                    continue

                # Extract and validate metadata paths
                graph_file, sku_file = extract_instance_metadata(instance_data)
                if not graph_file or not sku_file:
                    logger.error(f"Invalid metadata in {file}")
                    failed_instances.append(unique_key)
                    continue

                # Validate that files exist
                if not validate_instance_files(graph_file, sku_file, unique_key):
                    failed_instances.append(unique_key)
                    continue

                # Store instance data with additional metadata
                orders_dict[unique_key] = instance_data
                orders_dict[unique_key]['_source_folder'] = str(folder_path)
                orders_dict[unique_key]['_original_filename'] = file.name
                orders_dict[unique_key]['_graph_file'] = graph_file
                orders_dict[unique_key]['_sku_file'] = sku_file

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {file}: {e}")
                failed_instances.append(unique_key)
                continue
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
                failed_instances.append(unique_key)
                continue

    logger.info(f"Successfully loaded: {len(orders_dict)} instances")
    if failed_instances:
        logger.warning(f"Failed to load {len(failed_instances)} instances: {failed_instances}")

    return orders_dict


def _prepare_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for PyArrow/Parquet export by handling problematic column types.
    Converts dict columns with int keys to string representation.
    """
    df_copy = df.copy()

    for col in df_copy.columns:
        # Check if column contains dictionaries
        if df_copy[col].dtype == 'object':
            sample_val = df_copy[col].dropna().iloc[0] if not df_copy[col].dropna().empty else None
            if isinstance(sample_val, dict):
                # Convert dict to string representation or convert int keys to string keys
                if any(isinstance(k, int) for d in df_copy[col].dropna() for k in
                       (d.keys() if isinstance(d, dict) else [])):
                    # Option 1: Convert to string representation (safer)
                    df_copy[col] = df_copy[col].astype(str)

                    # Option 2: Convert int keys to string keys (if you want to keep dict structure)
                    # df_copy[col] = df_copy[col].apply(lambda x: {str(k): v for k, v in x.items()} if isinstance(x, dict) else x)

    return df_copy


def get_orders_folders_config() -> List[Path]:
    """
    Configure which order folders to process.
    Modify this function to add/remove folders.
    """
    base_orders_path = Path(r"Orders")

    # Option 1: Specific folders
    folders = [
        base_orders_path / r"Standard_case",
        # Choose source folders (orders) here
    ]

    # Option 2: All folders matching pattern
    # folders = list(base_orders_path.glob("*_instances_*_orders_*"))

    # Filter existing folders
    existing_folders = [f for f in folders if f.exists()]

    if len(existing_folders) != len(folders):
        missing = [f for f in folders if not f.exists()]
        logger.warning(f"Missing folders: {missing}")

    return existing_folders


def load_instance_graph_data(graph_file: Path, sku_file: Path, instance_key: str) -> Optional[Dict]:
    """Load graph data with caching"""

    cache_key = f"{graph_file}|{sku_file}"

    # Check cache first
    if cache_key in GRAPH_CACHE:
        return GRAPH_CACHE[cache_key]

    try:
        # Load graph data
        graph, pick_nodes, start_node, end_node, nodes, dist_matrix, tour_matrix = (
            utils.extract_graph_information(graph_file)
        )

        # Load SKU mapping
        sku_map = utils.read_sku_mapping(sku_file)

        graph_data = {
            'graph': graph,
            'pick_nodes': pick_nodes,
            'start_node': start_node,
            'end_node': end_node,
            'nodes': nodes,
            'dist_matrix': dist_matrix,
            'tour_matrix': tour_matrix,
            'sku_map': sku_map
        }

        # Cache it
        GRAPH_CACHE[cache_key] = graph_data
        return graph_data

    except Exception as e:
        logger.error(f"Failed to load graph data for {instance_key}: {e}")
        return None


def _write_results_to_file(results: List[dict], output_file: Path, append: bool = False):
    """
    Write results to parquet file with error handling
    """
    if not results:
        return

    try:
        df = pd.DataFrame(results)

        if append and output_file.exists():
            existing_df = pd.read_parquet(output_file, engine="pyarrow")
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(output_file, index=False, engine="pyarrow", compression="snappy")
        else:
            df.to_parquet(output_file, index=False, engine="pyarrow", compression="snappy")

        logger.info(f"Successfully wrote {len(results)} results to {output_file}")

    except Exception as e:
        logger.error(f"Failed to write results to file: {e}")
        try:
            backup_file = output_file.with_suffix('.json')
            with open(backup_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved backup to {backup_file}")
        except Exception as backup_error:
            logger.error(f"Failed to save backup: {backup_error}")


def run_simulation(orders_data: list,
                   batching_strategy: BatchingPolicy,
                   dispatching_strategy: DispatchingPolicy,
                   routing_strategy: Routing,
                   vehicles: List[Vehicle],
                   vis: bool,
                   graph_data: Dict,
                   waiting_strategy: Optional[WaitingPolicy] = None,
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    warehouse = Warehouse(
        allocation=graph_data['sku_map'],
        graph=graph_data['graph'],
        start_node=graph_data['start_node'],
        end_node=graph_data['end_node'],
        dist_matrix=graph_data['dist_matrix'],
        tour_matrix=graph_data['tour_matrix']
    )

    sim_control = SimulationControl()
    sim_control.register_strategy("dispatching", dispatching_strategy)
    sim_control.register_strategy("batching", batching_strategy)
    sim_control.register_strategy("routing", routing_strategy)
    if waiting_strategy:
        sim_control.register_strategy("waiting", waiting_strategy)

    state = SimulationState(
        warehouse=warehouse,
        vehicles=vehicles,
        batcher=sim_control.strategies.get("batching"),
        dispatcher=sim_control.strategies.get("dispatching"),
        router=sim_control.strategies.get("routing"),
        waiting=sim_control.strategies.get("waiting") if "waiting" in sim_control.strategies else None
    )

    engine = SimulationEngine(state, sim_control, vis=vis)

    for json_order in orders_data:
        order_id = json_order['order_id']
        json_items = {int(k): int(v) for k, v in json_order['items'].items()}
        arrival_time_seconds = json_order['arrival_time']
        due_time_seconds = json_order['due_date']
        arrival_time = state.current_time + timedelta(seconds=arrival_time_seconds)
        due_time = state.current_time + timedelta(seconds=due_time_seconds)
        engine.add_order(Order(order_id, json_items, arrival_time, due_time))

    order_df, batch_df = engine.run()

    if vis:
        visualizer = sim_visualizer.SimulationVisualizer(warehouse=warehouse, log=engine.log)
        visualizer.show()

    return order_df, batch_df


def run_single_simulation_with_graph(params: Tuple, graph_data: Dict, orders_dict: Dict[str, dict],
                                     output_folder: Path = None, timestamp: str = None) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    Runs single simulation with preloaded graph and optionally saves detailed data as xlsx
    """
    try:
        order_key, capacity, routing_cls, wait_strat, nr_veh = params
        rc = routing_cls.__name__
        if wait_strat is None:
            wf = 'Waiting'
        elif isinstance(wait_strat, WaitingOrderCountPolicy):
            wf = f"WaitingOrderCountPolicy_{wait_strat.min_orders}"
        else:
            wf = wait_strat.__class__.__name__

        instance_data = orders_dict[order_key]
        meta = instance_data.get('meta', {})

        layout = meta.get('layout')
        storage_assignment = meta.get('storage_assignment')
        exp_num_orders = meta.get('exp_num_orders')
        arrival_config = meta.get('arrival_config')
        num_articles_in_order_config = meta.get('num_articles_in_order_config')
        due_date_config = meta.get('due_date_config')
        article_config = meta.get('article_config')
        storage_policy = meta.get('storage_policy')

        reset_global_counters()
        vehicles = [Vehicle(i + 1, (0, 0)) for i in range(nr_veh)]
        orders_data = instance_data.get('orders', [])

        if not orders_data:
            empty_cols = ['instance', 'layout', 'batch_capacity', 'vehicles', 'waiting_strategy', 'routing_strategy',
                          'storage_assignment', 'exp_num_orders', 'arrival_config',
                          'num_articles_in_order_config', 'due_date_config', 'article_config', 'storage_policy']
            return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

        of, bf = run_simulation(
            orders_data=orders_data,
            batching_strategy=FIFOBatching(max_capacity=capacity),
            dispatching_strategy=GreedyDispatcher(),
            routing_strategy=routing_cls,
            vehicles=vehicles,
            vis=False,                           #Switch vis on here
            graph_data=graph_data,
            waiting_strategy=wait_strat
        )

        # Metadaten hinzufügen
        for df in (of, bf):
            df['instance'] = instance_data.get('_original_filename')
            df['layout'] = layout
            df['batch_capacity'] = capacity
            df['vehicles'] = nr_veh
            df['waiting_strategy'] = wf
            df['routing_strategy'] = rc
            df['storage_assignment'] = storage_assignment
            df['exp_num_orders'] = exp_num_orders
            df['arrival_config'] = str(arrival_config)
            df['num_articles_in_order_config'] = str(num_articles_in_order_config)
            df['due_date_config'] = str(due_date_config)
            df['article_config'] = str(article_config)
            df['storage_policy'] = str(storage_policy)

        # Save as Excel if output folder is provided
        if output_folder and timestamp:
            instance_name = instance_data.get('_original_filename', 'unknown').replace('.json', '')

            # Create safe filenames
            safe_rc = rc.replace('Routing', '').replace('Policy', '')
            safe_wf = wf.replace('WaitingOrderCountPolicy_', 'Wait').replace('Policy', '').replace('Waiting', 'NoWait')

            filename_base = f"{instance_name}_{safe_rc}_{safe_wf}_cap{capacity}_veh{nr_veh}_{timestamp}"

            # Save order data
            if not of.empty:
                order_file = output_folder / "order_details" / f"orders_{filename_base}.xlsx"
                order_file.parent.mkdir(parents=True, exist_ok=True)
                of.to_excel(order_file, index=False, engine='openpyxl')
                logger.info(f"✅ Saved order data: {order_file}")

            # Save batch data
            if not bf.empty:
                batch_file = output_folder / "batch_details" / f"batches_{filename_base}.xlsx"
                batch_file.parent.mkdir(parents=True, exist_ok=True)
                bf.to_excel(batch_file, index=False, engine='openpyxl')
                logger.info(f"✅ Saved batch data: {batch_file}")

        return of, bf

    except Exception as e:
        logger.exception(f"Error in task {params[0]}")
        empty_cols = ['instance', 'layout', 'batch_capacity', 'vehicles', 'waiting_strategy', 'routing_strategy',
                      'storage_assignment', 'exp_num_orders', 'arrival_config',
                      'num_articles_in_order_config', 'due_date_config', 'article_config', 'storage_policy']
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)


def run_single_task_return_kpis(params: Tuple, orders_dict: Dict[str, dict], save_detailed_xlsx: bool = False,
                                output_folder: Path = None, timestamp: str = None) -> Tuple[bool, dict]:
    """
    Runs one simulation and returns success flag and KPIs dict (or None).
    Optionally saves detailed data as xlsx.
    """
    try:
        order_key, capacity, routing_cls, wait_strat, nr_veh = params
        instance_data = orders_dict[order_key]

        # Load and cache graph_data
        graph_file = Path(instance_data['_graph_file'])
        sku_file = Path(instance_data['_sku_file'])
        graph_data = load_instance_graph_data(graph_file, sku_file, order_key)
        if graph_data is None:
            return False, {}

        # Run simulation with optional xlsx saving
        if save_detailed_xlsx and output_folder and timestamp:
            order_df, batch_df = run_single_simulation_with_graph(params, graph_data, orders_dict, output_folder,
                                                                  timestamp)
        else:
            order_df, batch_df = run_single_simulation_with_graph(params, graph_data, orders_dict)

        kpis = calculate_instance_aggregates(order_df, batch_df, params, instance_data)

        del order_df, batch_df
        gc.collect()

        return (True, kpis) if kpis else (False, {})
    except Exception as e:
        logger.error(f"Task {params[0]} failed: {e}")
        return False, {}


def calculate_instance_aggregates(order_df: pd.DataFrame, batch_df: pd.DataFrame,
                                  params: Tuple, instance_data: dict) -> dict:
    """
    Calculates exactly the aggregates needed for analysis_and_report
    """
    order_key, capacity, routing_cls, wait_strat, nr_veh = params

    # Extrahiere Metadaten
    meta = instance_data.get('meta', {})

    result = {
        'instance': instance_data.get('_original_filename'),
        'batch_capacity': capacity,
        'nr_vehicles': nr_veh,
        'waiting_strategy': wait_strat.__class__.__name__ if wait_strat else 'Waiting',
        'routing_strategy': routing_cls.__name__,
        'layout': meta.get('layout'),
        'storage_assignment': meta.get('storage_assignment'),
        'exp_num_orders': meta.get('exp_num_orders'),
        'arrival_config': str(meta.get('arrival_config')),
        'num_articles_in_order_config': str(meta.get('num_articles_in_order_config')),
        'due_date_config': str(meta.get('due_date_config')),
        'article_config': str(meta.get('article_config')),
        'storage_policy': str(meta.get('storage_policy')),
    }

    if hasattr(wait_strat, 'min_orders'):
        result['waiting_strategy'] = f"WaitingOrderCountPolicy_{wait_strat.min_orders}"

    if not batch_df.empty:

        sum_duration = float(batch_df['total_time'].sum())
        sum_dur_walk = float(batch_df['tour_duration'].sum())

        result.update({
            'sum_length': float(batch_df['tour_length'].sum()),
            'sum_duration': float(batch_df['total_time'].sum()),
            'sum_items': float(batch_df['nr_items'].sum()),
            'sum_dur_walk': float(batch_df['tour_duration'].sum()),
            'sum_dur_wait': float(batch_df['waiting_duration'].sum()),
            'mean_dur_walk': float(batch_df['tour_duration'].mean()),
            'std_dur_walk': float(batch_df['tour_duration'].std()),
            'mean_dur_wait': float(batch_df['waiting_duration'].mean()),
            'std_dur_wait': float(batch_df['waiting_duration'].std()),
            'mean_length': float(batch_df['tour_length'].mean()),
            'std_length': float(batch_df['tour_length'].std()),
            'mean_duration': float(batch_df['total_time'].mean()),
            'std_duration': float(batch_df['total_time'].std()),
            'count_batches': int(len(batch_df))
        })

        if sum_duration > 0:
            result['utilisation'] = sum_dur_walk / sum_duration

        if result['sum_items'] > 0:
            result['mean_length_per_item'] = result['sum_length'] / result['sum_items']
            result['mean_duration_per_item'] = result['sum_duration'] / result['sum_items']
            result['mean_walk_per_item'] = result['sum_dur_walk'] / result['sum_items']
            result['mean_wait_per_item'] = result['sum_dur_wait'] / result['sum_items']

    if not order_df.empty:
        if 'dummy_order' in order_df.columns:
            real_orders = order_df[order_df['dummy_order'] == False]
        else:
            real_orders = order_df

        if not real_orders.empty:
            result.update({
                'mean_order_completion_time': float(real_orders['completion_time'].mean()),
                'std_order_completion_time': float(real_orders['completion_time'].std()),
                'mean_tardiness': float(real_orders['tardiness'].mean()),
                'std_tardiness': float(real_orders['tardiness'].std()),
                'count_orders': int(len(real_orders))
            })
    return result


def compute_stats(series: pd.Series, alpha: float = 0.05):
    """Returns mean statistics and 95%-CI for a series."""
    data = series.dropna()
    n = data.size
    if n < 2:
        return pd.Series({
            'mean': data.mean() if n == 1 else np.nan,
            'std': np.nan,
            'sum': data.sum() if n == 1 else np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n': n
        })
    mean = data.mean()
    std = data.std(ddof=1)
    sem = std / np.sqrt(n)
    ci_low, ci_high = stats.t.interval(1 - alpha, df=n - 1, loc=mean, scale=sem)
    return pd.Series({
        'mean': mean,
        'std': std,
        'sum': data.sum(),
        'ci_lower': ci_low,
        'ci_upper': ci_high,
        'n': n
    })


def analysis_and_report(agg_df: pd.DataFrame, output_folder: Path, timestamp: str):
    """Enhanced analysis with instance meta information and 95%-CIs via compute_stats."""

    meta_cols = [
        'layout',
        'storage_assignment',
        'exp_num_orders',
        'arrival_config',
        'num_articles_in_order_config',
        'due_date_config',
        'article_config',
        'storage_policy'
    ]

    agg_merged = agg_df

    # define grouping columns
    stats_group = ['batch_capacity', 'nr_vehicles', 'waiting_strategy', 'routing_strategy'] + meta_cols

    kpi_cols = [
        'mean_order_completion_time', 'mean_tardiness',
        'mean_length', 'mean_duration',
        'mean_length_per_item', 'mean_duration_per_item',
        'mean_dur_walk', 'mean_dur_wait',
        'mean_walk_per_item', 'mean_wait_per_item', 'utilisation',
        'count_orders', 'count_batches'
    ]

    existing_kpi_cols = [col for col in kpi_cols if col in agg_merged.columns]

    # CI- and statistics calculations per strategy
    stats_df = (
        agg_merged
        .groupby(stats_group)[existing_kpi_cols]
        .apply(lambda grp: grp.apply(compute_stats))
        .unstack(level=-1)  # flatten MultiIndex-Spalten
        .reset_index()
    )

    # Export
    summary_file = output_folder / f'summary_kpi_full_{timestamp}.parquet'
    stats_df.to_parquet(summary_file, index=False, engine="pyarrow", compression="snappy")
    print(f"Strategy comparison (incl. CI, std, mean) saved: {summary_file}")

def _convert_parquet_folder_to_xlsx(folder: Path):
    for parquet_file in folder.glob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file, engine="pyarrow")
            xlsx_file = parquet_file.with_suffix(".xlsx")
            df.to_excel(xlsx_file, index=False)
            logger.info(f"Converted {parquet_file.name} -> {xlsx_file.name}")
        except Exception as e:
            logger.error(f"Failed to convert {parquet_file}: {e}")

def main(save_detailed_data: bool = False):
    """
    Main function to run simulations sequentially.

    Args:
        save_detailed_data: If True, saves detailed batch_df and order_df as xlsx files
                           for each simulation run.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    orders_folders = get_orders_folders_config()
    logger.info(f"Processing {len(orders_folders)} order folders")

    for folder in orders_folders:
        print(f"  - {folder}")

    all_orders = load_orders_from_multiple_folders(orders_folders)
    if not all_orders:
        logger.error("No order files found! Check your folder configuration.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = Path(r'Results')
    output_folder = results_base / f"{timestamp}_results"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Output file for aggregated results
    output_file = output_folder / f"aggregated_results_{timestamp}.parquet"

    routing_strategies = [
       SShapeAnalyticRoutingAdvanced] # choose here  SShapeRouting, SShapeAnalyticRouting,

    batching_capacities = [4]
    nr_vehicles = [1]
    shift_end = datetime(2025, 1, 1, 8, 0, 0)

    extra_wait_policies: List[Optional[WaitingPolicy]] = [
       # StartImmediatelyPolicy(initial_route_strategy=AllNodes, shift_end=shift_end),
        None  # NoWaiting
    ]

    all_combinations = []
    for order_key in all_orders.keys():
        for capacity in batching_capacities:
            for vehicle in nr_vehicles:
                # count_policies = [
                #     WaitingOrderCountPolicy(min_orders=k)
                #     for k in range(1, capacity)
                # ]
                waiting_strategies = extra_wait_policies # + count_policies

                for routing_cls in routing_strategies:
                    for wait_strat in waiting_strategies:
                        all_combinations.append((
                            order_key,
                            capacity,
                            routing_cls,
                            wait_strat,
                            vehicle,
                        ))

    tasks = [combo for combo in all_combinations]
    total_tasks = len(tasks)

    write_batch_size = 100

    print(f"Total tasks to execute: {total_tasks}")
    print(f"Saving detailed xlsx data: {save_detailed_data}")

    start_time = time.time()

    # Sequential processing
    logger.info("Starting sequential simulation processing")

    kpi_buffer = []
    kpi_writer = None
    written = 0

    # Process results sequentially
    for i, params in enumerate(tasks):
        try:
            success, kpis = run_single_task_return_kpis(
                params,
                all_orders,
                save_detailed_xlsx=save_detailed_data,
                output_folder=output_folder if save_detailed_data else None,
                timestamp=timestamp if save_detailed_data else None
            )

            if success:
                kpi_buffer.append(kpis)

            # Progress logging
            if (i + 1) % 10 == 0 or i == total_tasks - 1:
                elapsed = (time.time() - start_time) / 60
                progress = (i + 1) / total_tasks * 100
                logger.info(f"Progress: {i + 1}/{total_tasks} ({progress:.1f}%) - {elapsed:.1f}min elapsed")

            # Write KPI results when buffer is full
            if len(kpi_buffer) >= write_batch_size or (i == total_tasks - 1 and kpi_buffer):
                if kpi_writer is None and kpi_buffer:
                    df_initial = pd.DataFrame(kpi_buffer[:min(write_batch_size, len(kpi_buffer))])
                    schema = pa.Table.from_pandas(df_initial).schema
                    kpi_writer = pq.ParquetWriter(str(output_file), schema, compression="snappy")
                    kpi_writer.write_table(pa.Table.from_pandas(df_initial, schema=schema))
                    written += len(df_initial)
                    logger.info(
                        f"✅ Wrote {len(df_initial)} KPI results to {output_file} – cumulated {written}/{total_tasks}")

                    remaining_kpi = kpi_buffer[len(df_initial):]
                    kpi_buffer = remaining_kpi
                elif kpi_writer is not None and kpi_buffer:
                    df_chunk = pd.DataFrame(kpi_buffer)
                    kpi_writer.write_table(pa.Table.from_pandas(df_chunk, schema=kpi_writer.schema))
                    written += len(df_chunk)
                    logger.info(
                        f"✅ Wrote {len(df_chunk)} KPI results to {output_file} – cumulated {written}/{total_tasks}")
                    kpi_buffer.clear()

                gc.collect()

        except Exception as e:
            logger.error(f"Failed to process task {i + 1}: {e}")
            continue

    # Close KPI writer
    if kpi_writer is not None:
        kpi_writer.close()

    elapsed = (time.time() - start_time) / 60
    logger.info(f"All simulations completed in {elapsed:.1f} minutes")

    # Post-processing: analysis_and_report
    if output_file.exists():
        final_df = pd.read_parquet(output_file, engine="pyarrow")
        if final_df.empty:
            logger.warning("Result file is empty, no KPIs to analyze.")
            return

        analysis_and_report(final_df, output_folder, timestamp)
        logger.info(f"Results saved to: {output_folder}")

        if save_detailed_data:
            print(f"Detailed xlsx files saved to: {output_folder}/order_details/ and {output_folder}/batch_details/")
    else:
        logger.warning("No results file created - all simulations may have failed.")

    _convert_parquet_folder_to_xlsx(output_folder)


if __name__ == "__main__":
    save_detailed_data = True
    main(save_detailed_data=save_detailed_data)