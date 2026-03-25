"""
Adsorption site identification and modification for VASP catalysis.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Optional, Union, Dict, List, Tuple, Any, Set

from pymatgen.core import Structure, Element, Molecule, PeriodicSite
from pymatgen.core.surface import Slab, center_slab
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.adsorption import AdsorbateSiteFinder, plot_slab
from pymatgen.io.ase import AseAtomsAdaptor

from ..utils.structure_utils import get_atomic_layers


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