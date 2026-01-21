from typing import Dict, List

import numpy as np


# Minimal periodic table for typical TMDC elements.
# Values: (group, period, electronegativity, covalent_radius_angstrom)
ATOM_FEATURES: Dict[int, List[float]] = {
    16: [16, 3, 2.58, 1.05],  # S
    34: [16, 4, 2.55, 1.20],  # Se
    52: [16, 5, 2.10, 1.38],  # Te
    22: [4, 4, 1.54, 1.60],   # Ti
    23: [5, 4, 1.63, 1.53],   # V
    41: [5, 5, 1.60, 1.64],   # Nb
    42: [6, 5, 2.16, 1.54],   # Mo
    73: [5, 6, 1.50, 1.70],   # Ta
    74: [6, 6, 2.36, 1.62],   # W
    75: [7, 6, 1.90, 1.51],   # Re
    40: [4, 5, 1.33, 1.75],   # Zr
    72: [4, 6, 1.30, 1.59],   # Hf
}


def get_atom_features(atomic_number: int) -> np.ndarray:
    if atomic_number in ATOM_FEATURES:
        group, period, en, radius = ATOM_FEATURES[atomic_number]
        feats = [
            atomic_number / 100.0,
            group / 18.0,
            period / 7.0,
            en / 4.0,
            radius / 2.5,
        ]
    else:
        feats = [
            atomic_number / 100.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    return np.asarray(feats, dtype=np.float32)


def radial_basis(distances: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    return np.exp(-((distances[:, None] - centers[None, :]) ** 2) / (2 * width**2))


def make_rbf(distances: np.ndarray, rbf_dim: int, cutoff: float) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32).reshape(-1)
    if rbf_dim <= 0:
        return distances[:, None]
    centers = np.linspace(0.0, cutoff, rbf_dim, dtype=np.float32)
    width = float(centers[1] - centers[0]) if rbf_dim > 1 else cutoff
    return radial_basis(distances, centers, width).astype(np.float32)


def gaussian_distance(dist: float, cutoff: float, rbf_dim: int) -> np.ndarray:
    return make_rbf(np.asarray([dist], dtype=np.float32), rbf_dim, cutoff).squeeze(0)

