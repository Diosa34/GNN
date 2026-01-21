from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .featurization import get_atom_features, gaussian_distance


@dataclass
class GraphData:
    x: np.ndarray
    z: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    y: np.ndarray | None = None
    material_id: str | None = None


def _get_neighbors(structure, cutoff: float) -> Tuple[List[int], List[int], List[float]]:
    senders: List[int] = []
    receivers: List[int] = []
    distances: List[float] = []
    neighbors = structure.get_all_neighbors(cutoff, include_index=True)
    for i, nbrs in enumerate(neighbors):
        for nbr in nbrs:
            j = nbr.index
            dist = float(nbr.nn_distance)
            senders.append(i)
            receivers.append(j)
            distances.append(dist)
    return senders, receivers, distances


def build_graph_from_structure(structure, cutoff: float = 4.0, rbf_dim: int = 16) -> GraphData:
    z = np.asarray([site.specie.number for site in structure], dtype=np.int64)
    node_feats = np.stack(
        [get_atom_features(int(site.specie.number)) for site in structure], axis=0
    )

    senders, receivers, distances = _get_neighbors(structure, cutoff)
    if len(senders) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, rbf_dim + 1), dtype=np.float32)
    else:
        edge_index = np.asarray([senders, receivers], dtype=np.int64)
        dist_arr = np.asarray(distances, dtype=np.float32)
        rbf = np.stack([gaussian_distance(d, cutoff, rbf_dim) for d in distances], axis=0)
        edge_attr = np.concatenate([dist_arr[:, None], rbf], axis=-1).astype(np.float32)

    return GraphData(x=node_feats, z=z, edge_index=edge_index, edge_attr=edge_attr)

