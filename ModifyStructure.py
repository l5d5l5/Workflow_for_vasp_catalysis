from __future__ import annotations
import os 
import re
import math
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from itertools import combinations
from collections import defaultdict, Counter
from typing import Optional, Union, Dict, List, Tuple, Sequence, Set, Any

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

#pymatgen
from pymatgen.core import Structure, Element, PeriodicSite, Molecule, Lattice
from pymatgen.core.surface import Slab, SlabGenerator, center_slab
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.adsorption import AdsorbateSiteFinder, plot_slab
from pymatgen.io.ase import AseAtomsAdaptor

def parse_supercell_matrix(matrix: Optional[Union[str, Sequence[int], Sequence[Sequence[int]]]]):
    """Parses supercell matrix input into a 3x3 scaling matrix."""
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
    
    if isinstance(matrix, (list, Tuple, np.ndarray)):
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

class BulkToSlabGenerator:
    """
    Robust generator for creating slabs from bulk, with specific layer counts and fixation.
    """
    def __init__(
            self, 
            structure_source: Union[Structure, str, Path],
            save_dir: Optional[Union[str, Path]] = None,
            standardize: bool=True, 
            log_to_file: bool = True,):

        self.bulk_structure = self._resolve_structure_input(structure_source)
        
        if save_dir:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = Path.cwd() / f"slabs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger = self._setup_logger(self.save_dir, log_to_file)

        if standardize:
            try:
                sga = SpacegroupAnalyzer(self.bulk_structure, symprec=0.1)
                self.bulk_structure = sga.get_conventional_standard_structure()
                self.logger.info("Converted to conventional standard structure.")
            except Exception:
                self.logger.warning("Standardization failed, using original structure.")

    @staticmethod
    def run_from_dict(config: Dict[str, Any]) -> List[Slab]:
        """
        静态方法：通过字典配置执行完整的生成流程。
        
        Config 字典结构示例:
        {
            # --- 初始化参数 ---
            "structure_source": "POSCAR",  (必需)
            "save_dir": "output_slabs",
            "standardize_bulk": True,
            
            # --- 生成参数 (传递给 generate) ---
            "generate_params": {
                "miller_indices": "111",   (必需)
                "target_layers": 4,        (必需)
                "vacuum_thickness": 15.0,
                "supercell_matrix": "2x2",
                "symmetric": True,
                "fix_bottom_layers": 2
            },
            
            # --- 保存参数 ---
            "save_options": {
                "save": True,
                "filename_prefix": "POSCAR_slab" 
            }
        }
        """
        init_kwargs = {
            "structure_source": config.get("structure_source"),
            "save_dir": config.get("save_dir"),
            "standardize": config.get("standardize_bulk", True),
            "log_to_file": config.get("log_to_file", True)
        }
        
        if not init_kwargs["structure_source"]:
            raise ValueError("Config dictionary must contain 'structure_source'.")

        generator = BulkToSlabGenerator(**init_kwargs)
        
        gen_params = config.get("generate_params", {})
        if "miller_indices" not in gen_params or "target_layers" not in gen_params:
            raise ValueError("'generate_params' must contain 'miller_indices' and 'target_layers'.")

        slabs = generator.generate(**gen_params)
        
        save_opts = config.get("save_options", {})
        should_save = save_opts.get("save", True)
        prefix = save_opts.get("filename_prefix", "POSCAR")
        
        if should_save and slabs:
            hkl = gen_params.get("miller_indices")
            layers = gen_params.get("target_layers")
            
            for i, slab in enumerate(slabs):
                # 自动构建文件名: Prefix_HKL_Layers_TermX.vasp
                suffix = f"_term{i}" if len(slabs) > 1 else ""
                fname = f"{prefix}_{hkl}_{layers}L{suffix}.vasp"
                
                generator.save_slab(slab, filename=fname)
                
        return slabs

    def _resolve_structure_input(self, source: Union[Structure, str, Path]) -> Tuple[Structure, str]:
        """内部逻辑：解析输入源并返回 Structure 对象和来源描述。"""
        # case A : Structure type
        if isinstance(source, Structure):
            return source
        
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Structure source not found: {path}")

        # case B : dir type
        if path.is_dir():
            priority_files = ["CONTCAR", "POSCAR", "POSCAR.vasp"]
            for fname in priority_files:
                fpath = path / fname
                if fpath.exists() and fpath.stat().st_size > 0:
                    try:
                        return Structure.from_file(fpath)
                    except Exception:
                        continue
            
            cif_files = list(path.glob("*.cif"))
            if cif_files:
                return Structure.from_file(cif_files[0])
            
            raise FileNotFoundError(f"No valid structure files (CONTCAR/POSCAR/cif) found in {path}")

        # case C : file
        if path.is_file():
            return Structure.from_file(path), str(path)

        raise ValueError("Invalid structure source type.")

    def _setup_logger(self, output_dir: Path, log_to_file: bool) -> logging.Logger:
        logger = logging.getLogger(f"SlabGen_{id(self)}")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        if log_to_file:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(output_dir / "slab_gen.log", mode='w', encoding='utf-8')
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception as e:
                print(f"Warning: Could not create log file: {e}")
        return logger

    def _normalize_miller_indices(self, miller_indices: Union[int, str, Sequence[int]]) -> Tuple[int, int, int]:
        """miller index"""
        try:
            if isinstance(miller_indices, int):
                s = str(miller_indices)
                if len(s) != 3: raise ValueError
                return tuple(int(c) for c in s)
            elif isinstance(miller_indices, str):
                parts = re.findall(r'-?\d+', miller_indices)
                if len(parts) == 3: return tuple(int(p) for p in parts)
                if len(parts) == 1 and len(parts[0]) == 3: return tuple(int(c) for c in parts[0])
            elif isinstance(miller_indices, (list, tuple, np.ndarray)):
                arr = np.array(miller_indices).flatten()
                if len(arr) == 3: return tuple(int(x) for x in arr)
        except Exception:
            pass
        raise ValueError(f"Invalid miller_indices: {miller_indices}")

    def _trim_to_target_layers(self, slab: Slab, target_layers: int, symmetric: bool = False, hcluster_cutoff: float=0.25) -> Slab:
        layers = get_atomic_layers(slab, hcluster_cutoff = hcluster_cutoff)
        n_current = len(layers)
        
        if n_current < target_layers:
            raise ValueError(f"Slab too thin ({n_current} < {target_layers})")
        
        if n_current == target_layers:
            return slab

        layers_to_remove = n_current - target_layers
        
        if symmetric:
            remove_bottom = layers_to_remove // 2
            remove_top = layers_to_remove - remove_bottom
            layers_to_keep = layers[remove_bottom : n_current - remove_top]
        else:
            layers_to_keep = layers[:target_layers]

        keep_indices = set(idx for layer in layers_to_keep for idx in layer)
        remove_indices = list(set(range(len(slab))) - keep_indices)
        
        trimmed = slab.copy()
        trimmed.remove_sites(remove_indices)
        return trimmed
    
    def set_selective_dynamics(self, slab: Slab, fix_bottom: int, fix_top: int, all_fix: bool, hcluster_cutoff: float=0.25) -> Slab:
        n_sites = len(slab)
        if all_fix:
            sd = [[False, False, False]] * n_sites
        else:
            sd = [[True, True, True] for _ in range(n_sites)]
            if fix_bottom > 0 or fix_top > 0:
                layers = get_atomic_layers(slab, hcluster_cutoff=hcluster_cutoff)
                n_layers = len(layers)
                for i in range(min(fix_bottom, n_layers)):
                    for idx in layers[i]: sd[idx] = [False, False, False]
                for i in range(min(fix_top, n_layers)):
                    layer_idx = n_layers - 1 - i
                    if layer_idx >= 0:
                        for idx in layers[layer_idx]: sd[idx] = [False, False, False]
                        
        new_slab = slab.copy()
        new_slab.add_site_property("selective_dynamics", sd)
        return new_slab

    def generate(
        self,
        miller_indices: Union[int, str, Sequence[int]],
        target_layers: int,
        vacuum_thickness: float = 15.0,
        supercell_matrix: Optional[Union[str, Sequence[int]]] = None,
        fix_bottom_layers: int = 0,
        hcluster_cutoff: float = 0.25,
        fix_top_layers: int = 0,
        all_fix: bool = False,
        symmetric: bool = False,
        center: bool = True,
        primitive: bool = True,
        lll_reduce: bool = True,
    ) -> List[Slab]:
        """
        生成 Slab 列表。
        """
        hkl = self._normalize_miller_indices(miller_indices)
        sc_mat = parse_supercell_matrix(supercell_matrix)
        
        estimated_min_size = (target_layers * 2.5) + 8.0
        self.logger.info(f"Generating HKL={hkl}, Layers={target_layers}. Est. Thickness={estimated_min_size}A")

        slabgen = SlabGenerator(
            self.bulk_structure, hkl, 
            min_slab_size=estimated_min_size, 
            min_vacuum_size=vacuum_thickness,
            center_slab=center, primitive=primitive, lll_reduce=lll_reduce
        )
        
        raw_slabs = slabgen.get_slabs(tol=0.1, max_broken_bonds=0)
        
        if not raw_slabs:
            self.logger.warning("Retrying with double thickness...")
            slabgen = SlabGenerator(
                self.bulk_structure, hkl, 
                min_slab_size=estimated_min_size * 2, 
                min_vacuum_size=vacuum_thickness,
                center_slab=center, primitive=primitive, lll_reduce=lll_reduce
            )
            raw_slabs = slabgen.get_slabs(tol=0.1, max_broken_bonds=0)
            if not raw_slabs:
                raise ValueError(f"Failed to generate slabs for {hkl}.")

        processed_slabs = []
        
        for i, raw_slab in enumerate(raw_slabs):
            try:
                slab = self._trim_to_target_layers(raw_slab, target_layers, symmetric=symmetric, hcluster_cutoff = hcluster_cutoff)
                try: slab = slab.get_orthogonal_c_slab()
                except Exception: pass

                if sc_mat: slab.make_supercell(sc_mat)
                if center: slab = center_slab(slab)
                
                slab = self.set_selective_dynamics(slab, fix_bottom_layers, fix_top_layers, all_fix, hcluster_cutoff=hcluster_cutoff)
                processed_slabs.append(slab)
                
            except ValueError as e:
                self.logger.warning(f"Skipping termination {i}: {e}")
                continue

        return processed_slabs

    def save_slab(
        self, 
        slab: Structure, 
        filename: str, 
        fmt: str = "poscar", 
        output_dir: Optional[Union[str, Path]] = None
    ):
        """保存单个 Slab。"""
        target_dir = Path(output_dir) if output_dir else self.save_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / filename
        
        if fmt.lower() == "poscar":
            Poscar(slab).write_file(out_path)
        else:
            slab.to(filename=str(out_path), fmt=fmt)
            
        self.logger.info(f"Saved to {out_path}")

class AdsorptionModify(AdsorbateSiteFinder):
    """
    增强版吸附位点查找与修饰工具。
    """

    def __init__(
        self,
        slab_source: Union[Slab, Structure, str, Path],
        selective_dynamics: bool = False, 
        height: float = 0.9,
        mi_vec: Optional[np.ndarray] = None,
        save_dir: Optional[Union[str, Path]] = None,
        log_to_file: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        self.slab, self.source_info = self._resolve_structure_input(slab_source)
        super().__init__(slab=self.slab, selective_dynamics=selective_dynamics, height=height, mi_vec=mi_vec)

        self.save_dir = Path(save_dir) if save_dir else Path.cwd() / f"adsorption_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if logger:
            self.logger = logger
        else:
            self.logger = self._setup_logger(self.save_dir, log_to_file)
            self.logger.info(f"Initialized AdsorptionModify with slab from: {self.source_info}")

    def _resolve_structure_input(self, source: Union[Structure, Slab, str, Path]) -> Tuple[Structure, str]:
        if isinstance(source, (Structure, Slab)): 
            return source, "In-memory Object"
        
        path = Path(source)
        if not path.exists(): 
            raise FileNotFoundError(f"Source not found: {path}")
        if path.is_file(): 
            return Structure.from_file(path), str(path)

        if path.is_dir():
            search_patterns = ["CONTCAR", "POSCAR", "*CONTCAR*", "*POSCAR*", "*.vasp", "*.cif"]
            for pattern in search_patterns:
                matches = sorted(list(path.glob(pattern)))
                for fpath in matches:
                    if fpath.is_file() and fpath.stat().st_size > 0:
                        try:
                            return Structure.from_file(fpath), str(fpath)
                        except Exception:
                            continue
            raise ValueError(f"No valid structure files found in directory: {path}.")

        raise ValueError(f"Invalid structure source: {source}")

    def _setup_logger(self, output_dir: Path, log_to_file: bool):
        logger = logging.getLogger(f"AdsMod_{id(self)}")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if log_to_file:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(output_dir / "adsorption.log", mode='w', encoding='utf-8')
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception: pass
        return logger

    @staticmethod
    def run_from_dict(config: Dict[str, Any]) -> Union[List[Structure], Dict[str, Any]]:
        """
        执行流程。
        """
        # --- 1. 提取通用参数 ---
        gen_params = config.get("generate_params", {})
        
        init_kwargs = {
            "slab_source": config.get("target_slab_source"),
            "save_dir": config.get("save_dir"),
            "log_to_file": config.get("log_to_file", True),
            "selective_dynamics": gen_params.get("selective_dynamics", False)
        }
        
        if not init_kwargs["slab_source"]: 
            raise ValueError("Config must contain 'target_slab_source'.")

        modifier = AdsorptionModify(**init_kwargs)
        mode = config.get("mode", "analyze")
        modifier.logger.info(f"Running in mode: {mode}")

        # --- Mode A: Relative Placement ---
        if mode == "relative":
            rel_params = config.get("relative_params", {})
            ref_source = rel_params.get("reference_slab_source")
            if not ref_source: raise ValueError("Mode 'relative' requires 'reference_slab_source'.")
            
            ref_struct, _ = modifier._resolve_structure_input(ref_source)
            final_struct, _ = modifier.place_relative_adsorbate(
                relative_slab_structure=ref_struct,
                target_slab_structure=modifier.slab,
                adsorbate_indices=rel_params.get("adsorbate_indices"),
                adsorbate_anchor_indices=rel_params.get("adsorbate_anchor_indices"),
                slab_target_indices=rel_params.get("slab_target_indices"),
                movable_adsorbate_indices=rel_params.get("movable_adsorbate_indices"),
                output_path=None
            )
            save_name = config.get("save_filename", "POSCAR_adsorbed.vasp")
            modifier.save_structure(final_struct, save_name)
            return [final_struct]

        # --- Mode B: Analyze ---
        elif mode == "analyze":
            sites = modifier.find_adsorption_sites()
            results = {"sites": sites}
            target_input = config.get("analyze_params", {}).get("describe_site_index")
            
            if target_input is not None:
                flat_sites = sites['ontop'] + sites['bridge'] + sites['hollow']
                indices_to_process = []
                if isinstance(target_input, str) and target_input.lower() == "all":
                    indices_to_process = range(len(flat_sites))
                elif isinstance(target_input, list):
                    indices_to_process = target_input
                elif isinstance(target_input, int):
                    indices_to_process = [target_input]
                
                if len(indices_to_process) > 0:
                    modifier.logger.info(f"--- Describing {len(indices_to_process)} sites ---")
                    for idx in indices_to_process:
                        if 0 <= idx < len(flat_sites):
                            info = modifier.describe_adsorption_site(modifier.slab, flat_sites[idx])
                            env_str = ", ".join([f"{k}:{v}" for k, v in info['species_count'].items()])
                            modifier.logger.info(f"Site #{idx:<3} | Neighbors: {info['neighbors']:<2} | Env: {env_str}")
                        else:
                            modifier.logger.warning(f"Site #{idx} is out of range (Total: {len(flat_sites)})")

            if config.get("plot", True):
                modifier._plot_and_show(config.get("plot_params", {}), sites)
                
            return results
        
        # --- Mode C: Generate ---
        elif mode == "generate":
            molecule_formula = gen_params.get("molecule_formula")
            if not molecule_formula: raise ValueError("Mode 'generate' requires 'molecule_formula'.")
            
            try:
                molecule = modifier.ase2pmg(molecule_formula)
            except Exception as e:
                if Path(molecule_formula).exists():
                    molecule = Molecule.from_file(molecule_formula)
                else:
                    raise ValueError(f"Failed to load molecule '{molecule_formula}': {e}")

            find_args = gen_params.get("find_args", {}).copy()
            requested_positions = find_args.pop("positions", []) 
            
            all_sites = modifier.find_adsorption_sites(**find_args)
            
            output_collection_name = f"{molecule_formula}_ads_collection"
            base_path = modifier.save_dir / output_collection_name
            base_path.mkdir(parents=True, exist_ok=True)
            
            generated_info = []
            count = 0
            
            modifier.logger.info(f"Start generating structures for {molecule_formula}...")

            for site_type, site_coords_list in all_sites.items():
                if requested_positions and site_type not in requested_positions:
                    continue
                
                for i, site_coords in enumerate(site_coords_list):
                    site_info = modifier.describe_adsorption_site(modifier.slab, site_coords, radius=3.0, top_n=10)
                    coord_counts = site_info.get("species_count", {})
                    adsorbed_structure = modifier.add_adsorbate(
                        molecule, 
                        site_coords,
                        reorient=gen_params.get("reorient", True) 
                    )

                    coord_parts = []
                    for species in sorted(coord_counts.keys()):
                        coord_parts.append(f"{species}{coord_counts[species]}")
                    coord_str = "_".join(coord_parts) if coord_parts else "no_coord"
                    
                    site_dir_name = f"{site_type}_{coord_str}_{i}"
                    full_site_dir = base_path / site_dir_name
                    full_site_dir.mkdir(parents=True, exist_ok=True)
                    
                    out_path = full_site_dir / "POSCAR"
                    Poscar(adsorbed_structure).write_file(out_path)
                    
                    generated_info.append({
                        "path": str(out_path),
                        "site_type": site_type,
                        "coord_env": coord_str
                    })
                    count += 1

            modifier.logger.info(f"Successfully generated {count} structures in {base_path}")
            
            if config.get("plot", True):
                modifier._plot_and_show(config.get("plot_params", {}), all_sites)

            return {"generated_count": count, "output_dir": str(base_path), "details": generated_info}

        return []

    def _plot_and_show(self, plot_params: dict, sites: dict):
        """内部绘图辅助函数"""
        params = plot_params.copy()
        figsize = params.pop("figsize", (6, 6))
        fig, ax = plt.subplots(figsize=figsize)
        self.plot_slab_with_labels(self.slab, ax=ax, ads_sites=sites, **params)
        out_p = self.save_dir / "adsorption_sites.png"
        fig.savefig(out_p, dpi=300, bbox_inches='tight')
        plt.show()

    def save_structure(self, structure: Structure, filename: str):
        out_path = self.save_dir / filename
        Poscar(structure).write_file(out_path)
        self.logger.info(f"Saved structure to {out_path}")

    @classmethod
    def ase2pmg(cls, adsorption_molecule: str) -> Molecule:
        from ase.build import molecule
        ase_molecule = molecule(adsorption_molecule)
        return AseAtomsAdaptor.get_molecule(ase_molecule)

    @staticmethod
    def _get_anchor_coords(structure, indices):
        coords = [structure[i].coords for i in indices]
        return coords[0] if len(coords) == 1 else np.mean(coords, axis=0)

    def place_relative_adsorbate(self, relative_slab_structure, target_slab_structure, adsorbate_indices, adsorbate_anchor_indices, slab_target_indices, movable_adsorbate_indices=None, output_path=None):

        ads_anchor_coords = self._get_anchor_coords(relative_slab_structure, adsorbate_anchor_indices)
        relative_positions = [relative_slab_structure[i].coords - ads_anchor_coords for i in adsorbate_indices]
        adsorbate_species = [relative_slab_structure[i].specie for i in adsorbate_indices]
        slab_target_coords = self._get_anchor_coords(target_slab_structure, slab_target_indices)

        combined = target_slab_structure.copy()
        adsorbate_indices_in_combined = []
        for i, rel_pos in enumerate(relative_positions):
            ads_pos = slab_target_coords + rel_pos
            combined.append(adsorbate_species[i], ads_pos, coords_are_cartesian=True)
            adsorbate_indices_in_combined.append(len(combined) - 1)
        
        # SD 逻辑
        if target_slab_structure.site_properties.get("selective_dynamics"):
            sd = [list(x) for x in target_slab_structure.site_properties["selective_dynamics"]]
        else:
            sd = [[False, False, False] for _ in range(len(target_slab_structure))]

        n_ads = len(adsorbate_species)
        movable_set = set(movable_adsorbate_indices) if movable_adsorbate_indices is not None else set(range(n_ads))
        for i in range(n_ads):
            sd.append([True, True, True] if i in movable_set else [False, False, False])
                
        combined.add_site_property("selective_dynamics", sd)
        return combined, adsorbate_indices_in_combined

    def describe_adsorption_site(self, slab, site_coords, radius=3.0, top_n=10):
        neighbors = slab.get_sites_in_sphere(site_coords, radius, include_index=True)
        neighbors.sort(key=lambda x: x.nn_distance)
        species_list = [str(n.species_string) for n in neighbors[:top_n]]
        species_count = dict(Counter(species_list))
        return {"neighbors": len(neighbors), "species_count": species_count}

    @staticmethod
    def _rotation_matrix_from_vectors(vec1, vec2):
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if np.allclose(v, 0) and c > 0.999999: return np.eye(3)
        if np.allclose(v, 0) and c < -0.999999:
            orth = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9: orth = np.array([0.0, 1.0, 0.0])
            v = np.cross(a, orth)
            v = v / np.linalg.norm(v)
            return -np.eye(3) + 2 * np.outer(v, v)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

    @classmethod
    def _get_rot_matrix_for_slab(cls, slab: Slab) -> np.ndarray:
        if hasattr(slab, "normal"):
            normal = np.array(slab.normal)
            return cls._rotation_matrix_from_vectors(normal, np.array([0.0, 0.0, 1.0]))
        return np.eye(3)

    @classmethod
    def plot_slab_with_labels(
        cls,
        slab: Slab,
        ax: Optional[plt.Axes] = None,
        scale: float = 0.8,
        repeat: Tuple[int, int, int] = (1, 1, 1),
        window: float = 1.5,
        decay: float = 0.2,
        adsorption_sites: bool = True,
        inverse: bool = False,
        label_offset: Tuple[float, float] = (0.0, 0.0),
        ads_sites: Optional[dict] = None,
    ):
        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)

        working = slab.copy()
        rx, ry, rz = 1, 1, 1
        if repeat is not None:
            rx, ry, rz = repeat
            working.make_supercell([rx, ry, rz])

        try:
            plot_slab(
                working, ax=ax, scale=scale, repeat=1, window=window,
                draw_unit_cell=True, decay=decay, adsorption_sites=False, inverse=inverse
            )
        except TypeError:
            plot_slab(working, ax=ax, scale=scale, adsorption_sites=False, inverse=inverse)

        if not adsorption_sites:
            ax.set_aspect("equal")
            return ax

        if ads_sites is None:
            asf = AdsorbateSiteFinder(slab)
            found = asf.find_adsorption_sites()
            ads_sites = found["ontop"] + found["bridge"] + found["hollow"]
        elif isinstance(ads_sites, dict):
            ads_sites = ads_sites.get("ontop", []) + ads_sites.get("bridge", []) + ads_sites.get("hollow", [])

        if not ads_sites: return ax

        R = cls._get_rot_matrix_for_slab(working)
        original_lattice = slab.lattice.matrix
        vec_a = original_lattice[0]
        vec_b = original_lattice[1]

        ads_2d_list = []
        labels_list = []

        for idx, site in enumerate(ads_sites):
            site_coords = np.array(site)
            for i in range(rx):
                for j in range(ry):
                    shifted_coords = site_coords + i * vec_a + j * vec_b
                    rotated = np.dot(R, shifted_coords)
                    ads_2d_list.append(rotated[:2])
                    labels_list.append(str(idx))

        if ads_2d_list:
            xs = [p[0] for p in ads_2d_list]
            ys = [p[1] for p in ads_2d_list]
            ax.plot(xs, ys, linestyle="", marker="x", markersize=6, mew=1.5, color="red", zorder=500)
            for (x, y), label in zip(ads_2d_list, labels_list):
                ax.text(x + label_offset[0], y + label_offset[1], label, 
                        fontsize=9, color="blue", fontweight='bold', zorder=501,
                        ha='center', va='bottom')

        ax.set_aspect("equal")
        return ax

class StructureModify:
    def __init__(self, structure: Structure):
        self.structure = structure.copy()
        self.initial_structure = structure.copy()
        self.fixed_frac_coords: Set[Tuple[float, float, float]] = set()
        self._initialize_fixed_coords(self.structure)
    
    def _initialize_fixed_coords(self, structure: Structure):
        self.fixed_frac_coords.clear()
        for site in structure:
            sdyn = getattr(site, "selective_dynamics", None)
            if sdyn is not None and not np.any(sdyn):
                self.fixed_frac_coords.add(tuple(site.frac_coords))
    
    def _parse_dopant(self, dopant: Union[Element, str, None]) -> Optional[Element]:
        if isinstance(dopant, str) and dopant.lower() not in ["vac", "vacancy", "none", ""]:
            return Element(dopant)
        return dopant if isinstance(dopant, Element) else None

    def _sanitize_structure(self, structure: Structure) -> Structure:
        """
        当进行 replace 操作时，新原子的 selective_dynamics 往往为 None。
        此方法会检查结构是否开启了 selective_dynamics，如果是，则将 None 补全为 [True, True, True]。
        """
        if "selective_dynamics" in structure.site_properties:
            new_sd = []
            for site in structure:
                sd = site.properties.get("selective_dynamics")
                if sd is None:
                    new_sd.append([True, True, True])
                else:
                    new_sd.append(sd)
            structure.add_site_property("selective_dynamics", new_sd)
        return structure

    def _apply_defects(self, structure: Structure, indices: Sequence[int], dopant: Optional[Element]) -> Structure:
        """核心修改逻辑：应用空位或掺杂，并自动修复属性。"""
        new_struct = structure.copy()
        if dopant is None:
            new_struct.remove_sites(sorted(indices, reverse=True))
        else:
            for idx in indices:
                new_struct.replace(idx, dopant)
        return self._sanitize_structure(new_struct)

    def _find_candidate_indices(self, substitute_element: Element, is_slab: bool = False,
                                target_layer_indices: Optional[List[int]] = None,
                                z_range: Optional[Tuple[float, float]] = None,
                                top_layers: Optional[int] = None,
                                target_cn: Optional[int] = None, cn_cutoff: Optional[float] = None) -> List[int]:
        
        candidates = [i for i, s in enumerate(self.structure) 
                      if s.specie == substitute_element and tuple(s.frac_coords) not in self.fixed_frac_coords]
        
        if not candidates:
            raise ValueError(f"No candidate sites found for {substitute_element} (excluding fixed sites).")

        if target_layer_indices is not None:
            candidates = [i for i in candidates if i in target_layer_indices]
        elif z_range is not None:
            z_min, z_max = z_range
            c_len = self.structure.lattice.c
            if z_max > 1.0: z_min, z_max = z_min / c_len, z_max / c_len
            candidates = [i for i in candidates if z_min <= self.structure[i].frac_coords[2] <= z_max]
        elif top_layers is not None:
            sorted_by_z = sorted(candidates, key=lambda i: self.structure[i].frac_coords[2], reverse=True)
            candidates = sorted_by_z[:top_layers]

        if not candidates:
            raise ValueError("No candidate sites found after geometric filtering.")

        if target_cn is not None:
            cutoff = cn_cutoff or 2.0
            filtered = []
            for i in candidates:
                try:
                    cn = len(self.structure.get_neighbors(self.structure[i], cutoff))
                    if abs(cn - target_cn) <= 0.1: filtered.append(i)
                except Exception: continue
            candidates = filtered
            if not candidates:
                raise ValueError(f"No sites with CN={target_cn} found.")

        if is_slab: return candidates
        
        try:
            sga = SpacegroupAnalyzer(self.structure)
            sym_struct = sga.get_symmetrized_structure()
            reduced = []
            for group in sym_struct.equivalent_indices:
                intersection = [idx for idx in group if idx in candidates]
                if intersection: reduced.append(intersection[0])
            return reduced
        except Exception as e:
            logging.warning(f"Symmetry analysis failed: {e}. Using all candidates.")
            return candidates

    def get_layers(self, hcluster_cutoff: float = 0.25, target_element: Union[Element, str] = None) -> List[int]:

        layers = get_atomic_layers(self.structure, hcluster_cutoff)
        
        if target_element:
            target_el = Element(target_element) if isinstance(target_element, str) else target_element
            filtered_layers = []
            for layer in layers:
                filtered_layer = [idx for idx in layer if self.structure[idx].specie == target_el]
                if filtered_layer:
                    filtered_layers.append(filtered_layer)
            return filtered_layers
            
        return layers

    def make_supercell(self, supercell_matrix: Union[str, Sequence[int], int] = 1, structure: Structure = None) -> Structure:
        matrix = parse_supercell_matrix(supercell_matrix)
        target = structure.copy() if structure else self.structure
        target.make_supercell(matrix)
        if structure is None:
            self.structure = target
            self._initialize_fixed_coords(self.structure)
        return target

    def replace_species_all(self, species_map: Dict) -> Structure:
        self.structure.replace_species(species_map)
        self._initialize_fixed_coords(self.structure)
        return self.structure

    def insert_atom(self, element: Union[Element, str], frac_coord: Sequence[float]) -> Structure:
        new_s = self.structure.copy()
        new_s.insert(len(new_s), Element(element) if isinstance(element, str) else element, frac_coord)
        return self._sanitize_structure(new_s) # 插入原子也需要修复属性

    @classmethod
    def generate(cls, structure: Structure, substitute_element: Union[Element, str], 
                 dopant: Union[Element, str, None] = None, dopant_num: Union[int, float] = 1, 
                 num_structs: int = 1, random_seed: int = None, **kwargs):
        if random_seed: random.seed(random_seed)
        modifier = cls(structure)
        sub_el = Element(substitute_element) if isinstance(substitute_element, str) else substitute_element
        dopant_el = modifier._parse_dopant(dopant)

        candidates = modifier._find_candidate_indices(sub_el, **kwargs)
        n_max = len(candidates)
        if isinstance(dopant_num, float): n_sub = max(1, math.ceil(dopant_num * n_max))
        else: n_sub = min(dopant_num, n_max)

        results = []
        seen = set()
        for _ in range(num_structs * 10):
            if len(results) >= num_structs: break
            chosen = tuple(sorted(random.sample(candidates, n_sub)))
            fingerprint = tuple(tuple(modifier.structure[i].frac_coords) for i in chosen)
            if fingerprint in seen: continue
            seen.add(fingerprint)
            results.append(modifier._apply_defects(modifier.structure, chosen, dopant_el))
        return results

    @classmethod
    def step_by_step(cls, structure: Structure, substitute_element: Union[Element, str],
                     dopant: Union[Element, str, None] = None, max_steps: int = 3,
                     max_structures_num: int = 15, return_all_steps: bool = False, 
                     random_seed: int = None, **kwargs):
        if random_seed: random.seed(random_seed)
        modifier = cls(structure)
        sub_el = Element(substitute_element) if isinstance(substitute_element, str) else substitute_element
        dopant_el = modifier._parse_dopant(dopant)

        try:
            candidates = modifier._find_candidate_indices(sub_el, **kwargs)
        except ValueError as e:
            logging.warning(f"Initialization failed: {e}")
            return []

        if len(candidates) < max_steps:
            logging.warning(f"Not enough candidates ({len(candidates)}) for {max_steps} steps.")
            return []

        all_paths = []
        final_structs = []
        seen_paths = set()

        for _ in range(max_structures_num * 50):
            if len(all_paths) >= max_structures_num: break
            chosen_seq = random.sample(candidates, max_steps)
            path_fingerprint = frozenset(chosen_seq)
            if path_fingerprint in seen_paths: continue
            seen_paths.add(path_fingerprint)

            current_path = []
            for step in range(1, max_steps + 1):
                indices_subset = chosen_seq[:step]
                step_struct = modifier._apply_defects(modifier.initial_structure, indices_subset, dopant_el)
                current_path.append(step_struct)
            
            all_paths.append(current_path)
            final_structs.append(current_path[-1])

        return all_paths if return_all_steps else final_structs