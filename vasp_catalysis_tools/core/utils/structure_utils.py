"""
Structure utility functions for VASP catalysis tools.
"""
from itertools import combinations
from collections import defaultdict
from typing import Optional, Union, Sequence, Tuple

import numpy as np
import warnings
from pathlib import Path
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from pymatgen.core import Structure


def load_structure(struct_source: Union[str, Path, Structure]) -> Structure:
    """Load a structure from a file/dir/Structure object.
    
    Priority for directories: CONTCAR > POSCAR > POSCAR.vasp > *.vasp > *.cif
    """
    if isinstance(struct_source, Structure):
        return struct_source.copy()

    p = Path(struct_source).expanduser().resolve()

    if p.is_file():
        return Structure.from_file(p)
    
    if p.is_dir():
        for fname in ["CONTCAR", "POSCAR", "POSCAR.vasp"]:
            fp = p / fname
            # 增加 st_size > 0 检查，防止读取到空的占位文件报错
            if fp.exists() and fp.stat().st_size > 0:
                return Structure.from_file(fp)
            
        vasp_files = list(p.glob("*.vasp"))
        if len(vasp_files) == 1:
            return Structure.from_file(vasp_files[0])
        elif len(vasp_files) > 1:
            warnings.warn(f"Multiple .vasp files found in {p}; using {vasp_files[0]}")
            return Structure.from_file(vasp_files[0])
        
        cif_files = list(p.glob("*.cif"))
        if len(cif_files) == 1:
            return Structure.from_file(cif_files[0])
        elif len(cif_files) > 1:
            raise FileNotFoundError(f"Multiple CIF files found in {p}. Please specify one explicitly.")

    raise FileNotFoundError(f"No valid structure file found in: {p}")

def parse_supercell_matrix(matrix: Optional[Union[str, Sequence[int], Sequence[Sequence[int]]]]):
    """
    Parses supercell matrix input into a 3x3 scaling matrix.
    
    Args:
        matrix: Can be a string (e.g. "2x2x1"), a list/tuple of integers [2,2,1],
               or a 3x3 matrix [[2,0,0],[0,2,0],[0,0,1]]
    
    Returns:
        A 3x3 scaling matrix as a list of lists
        
    Raises:
        ValueError: If the matrix format is not recognized
    """
    if matrix is None:
        return None
    
    if isinstance(matrix, str):
        factors = matrix.lower().split("x")
        factors = [int(x) for x in factors]
        if len(factors) == 2:
            return [[factors[0], 0, 0], [0, factors[1], 0], [0, 0, 1]]
        elif len(factors) == 3:
            return [[factors[0], 0, 0], [0, factors[1], 0], [0, 0, factors[2]]]
        else:
            raise ValueError(f"Invalid matrix string: {matrix}")
    
    if isinstance(matrix, (list, tuple, np.ndarray)):
        arr = np.array(matrix)
        if arr.shape == (3, 3):
            return arr.tolist()
        if arr.ndim == 1:
            if len(arr) == 2:
                return [[arr[0], 0, 0], [0, arr[1], 0], [0, 0, 1]]
            elif len(arr) == 3:
                return [[arr[0], 0, 0], [0, arr[1], 0], [0, 0, arr[2]]]
    raise ValueError(f"Unsupported supercell matrix format: {matrix}")

def get_atomic_layers(structure: Structure, hcluster_cutoff: float = 0.25):
    """
    Identifies atomic layers in a structure (usually a slab) using hierarchical clustering based on Z-coordinates.
    Returns a list of lists, where each inner list contains indices of atoms in that layer.
    Layers are sorted by Z-height (low to high).
    """
    if not isinstance(structure, Structure):
        raise TypeError("Expected pymatgen Structure object")
    if hcluster_cutoff <= 0:
        raise ValueError(f"hcluster_cutoff must be positive, got {hcluster_cutoff}")
    
    sites = structure.sites
    n_sites = len(sites)
    if n_sites == 0:
        return []
    if n_sites == 1:
        return [[0]]
    
    frac_coords = np.array([s.frac_coords for s in sites])
    frac_z = frac_coords[:, 2]
    c_param = structure.lattice.c
    # Calculate distance matrix considering periodicity in Z (though for slabs usually not periodic in Z, 
    # using periodicity helps robust clustering if atoms are near boundary)
    # For standard slabs with vacuum, simple Z difference is usually enough, but let's be robust.
    dist_matrix = np.zeros((n_sites, n_sites))
    for i, j in combinations(range(n_sites), 2):
        dz = abs(frac_z[i] - frac_z[j])
        dist = dz * c_param
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist
    condensed = squareform(dist_matrix)
    z_linkage = linkage(condensed, method="average")
    clusters = fcluster(z_linkage, hcluster_cutoff, criterion='distance')
    raw_layers = defaultdict(list)
    for idx, cid in enumerate(clusters):
        raw_layers[cid].append(idx)
    # Sort layers by average Z height
    layer_avg_z = {}
    for cid, indices in raw_layers.items():
        layer_avg_z[cid] = np.mean(frac_z[indices])
    
    sorted_cids = sorted(layer_avg_z, key=lambda k: layer_avg_z[k])
    
    return [raw_layers[cid] for cid in sorted_cids]
