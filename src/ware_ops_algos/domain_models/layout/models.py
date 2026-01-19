from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from ware_ops_algos.domain_models import BaseDomainObject

Coord = tuple[int | float, int | float]


class LayoutType(str, Enum):
    CONVENTIONAL = "conventional"
    UNCONVENTIONAL = "unconventional"


@dataclass
class LayoutParameters:
    n_aisles: int
    n_pick_locations: int
    n_blocks: int
    dist_top_to_pick_location: float
    dist_bottom_to_pick_location: float
    dist_pick_locations: float
    dist_aisle: float
    dist_start: float
    start_location: tuple[int, int]
    dist_end: Optional[float] = None
    dist_change_aisle: Optional[float] = None
    dist_cross_aisle: Optional[float] = None
    end_location: Optional[tuple[int, int]] = None
    start_connection_point: Optional[tuple[int, int]] = None
    end_connection_point: Optional[tuple[int, int]] = None,
    depot_location: Optional[str] = None


@dataclass
class LayoutNetwork:
    graph: nx.Graph
    distance_matrix: Optional[pd.DataFrame] = None
    predecessor_matrix: Optional[np.ndarray] = None
    shortest_paths: Optional[dict] = None
    closest_node_to_start: Optional[Coord] = None
    min_aisle_position: Optional[int | float] = None
    max_aisle_position: Optional[int | float] = None
    start_node: Optional[Tuple[int | float, int | float]] = None
    end_node: Optional[Tuple[int | float, int | float]] = None
    node_list: Optional[list[Tuple[int | float, int | float]]] = None


@dataclass
class LayoutData(BaseDomainObject):
    tpe: LayoutType

    graph_data: Optional[LayoutParameters] = None
    layout_network: Optional[LayoutNetwork] = None

    # def get_features(self) -> list[str]:
    #     feats: set[str] = set()
    #
    #     def add_features(obj) -> None:
    #         for f in fields(obj):
    #             val = getattr(obj, f.name)
    #             if val is None:
    #                 continue
    #             # treat empty containers as "absent"
    #             if isinstance(val, (list, tuple, dict, set)) and not val:
    #                 continue
    #             feats.add(f.name)
    #
    #     if self.graph_data is not None:
    #         add_features(self.graph_data)
    #
    #     if self.layout_network is not None:
    #         add_features(self.layout_network)
    #
    #     return sorted(feats)
    def get_features(self) -> dict[str, any]:
        features = {}

        if self.graph_data:
            for f in fields(self.graph_data):
                value = getattr(self.graph_data, f.name)
                if value is not None:
                    features[f.name] = value

        if self.layout_network:
            for f in fields(self.layout_network):
                value = getattr(self.layout_network, f.name)
                if value is not None:
                    features[f.name] = value

        return features




# @dataclass
# class LayoutData(BaseDomainObject):
#     tpe: LayoutType
#     graph_data: Optional[LayoutParameters | GraphTopology] = None
#     graph: Optional[nx.Graph] = None
#     distance_matrix: Optional[pd.DataFrame] = None
#     predecessor_matrix: Optional[np.array] = None
#     start_node: Optional[Tuple[int | float, int | float]] = None
#     end_node: Optional[Tuple[int | float, int | float]] = None
#     shortest_paths: Optional[dict] = None,
#     min_aisle_position: Optional[int | float] = None
#     max_aisle_position: Optional[int | float] = None
#     closest_node_to_start: Optional[Tuple[int | float, int | float]] = None


