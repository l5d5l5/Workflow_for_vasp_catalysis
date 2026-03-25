"""
Bulk to Slab Generator for VASP catalysis simulations.
"""
import os
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict, List, Tuple, Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.surface import Slab, SlabGenerator, center_slab
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ..utils.structure_utils import get_atomic_layers, parse_supercell_matrix


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

    def _resolve_structure_input(self, source: Union[Structure, str, Path]) -> Structure:
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
            return Structure.from_file(path)

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

    def _normalize_miller_indices(self, miller_indices: Union[int, str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
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
        layers = get_atomic_layers(slab, hcluster_cutoff=hcluster_cutoff)
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
        miller_indices: Union[int, str, Tuple[int, int, int]],
        target_layers: int,
        vacuum_thickness: float = 15.0,
        supercell_matrix: Optional[Union[str, Tuple[int, int, int]]] = None,
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
                slab = self._trim_to_target_layers(raw_slab, target_layers, symmetric=symmetric, hcluster_cutoff=hcluster_cutoff)
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