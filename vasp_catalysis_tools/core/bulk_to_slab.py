"""
Bulk to Slab Generator for VASP catalysis simulations.
"""
import os
import warnings
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict, List, Tuple, Any, Sequence

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.surface import Slab, SlabGenerator, center_slab
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ..utils.structure_utils import get_atomic_layers, parse_supercell_matrix, load_structure

class BulkToSlabGenerator:
    """
    Robust generator for creating slabs from bulk, with specific layer counts and fixation.
    支持传统的单步调用，也支持 Fluent API 链式调用。
    """
    def __init__(
            self, 
            structure_source: Union[Structure, str, Path],
            save_dir: Optional[Union[str, Path]] = None,
            standardize: bool = True):

        # 直接调用外部的通用工具函数，代码变得极其清爽
        self.bulk_structure = load_structure(structure_source)
        
        self.save_dir = Path(save_dir) if save_dir else None
        if standardize:
            try:
                sga = SpacegroupAnalyzer(self.bulk_structure, symprec=0.1)
                self.bulk_structure = sga.get_conventional_standard_structure()
            except Exception as e:
                warnings.warn(f"Standardization failed ({e}), using original structure.")
                
        self._slabs: List[Slab] = []

    @staticmethod
    def run_from_dict(config: Dict[str, Any]) -> List['Slab']:
        """静态方法：通过字典配置执行完整的生成流程，带有严格的参数检查和智能路径推导。"""
        
        # 1. 基础参数检查
        source = config.get("structure_source")
        if not source:
            raise ValueError("Config error: 'structure_source' is required.")

        gen_params = config.get("generate_params")
        if not gen_params:
            raise ValueError("Config error: 'generate_params' dictionary is required.")

        required_gen_keys = ["miller_indices", "target_layers"]
        for key in required_gen_keys:
            if key not in gen_params or gen_params[key] is None:
                raise ValueError(f"Config error: '{key}' is required in 'generate_params'.")

        # 2. 智能处理保存路径
        save_dir = config.get("save_dir")
        save_opts = config.get("save_options", {})
        should_save = save_opts.get("save", True)

        if should_save and not save_dir:
            if isinstance(source, (str, Path)):
                # 如果用户没给路径，但给了文件源：保存在 bulk 结构同级目录下，并带上参数信息
                base_dir = Path(source).parent
                
                # 解析参数用于命名 (例如: 111, 4L, 2x2)
                hkl = gen_params["miller_indices"]
                hkl_str = "".join(map(str, hkl)) if isinstance(hkl, (list, tuple)) else str(hkl)
                layers = gen_params["target_layers"]
                
                sc = gen_params.get("supercell_matrix")
                if sc:
                    if isinstance(sc, (list, tuple)):
                        sc_str = f"{sc[0]}x{sc[1]}"
                    else:
                        sc_str = str(sc)
                else:
                    sc_str = "1x1"
                    
                save_dir = base_dir / f"slab_{hkl_str}_{layers}L_{sc_str}"
                print(f"未指定 save_dir，自动推导保存路径为: {save_dir}")
            else:
                # 如果传入的是内存 Structure 对象且没给路径，直接报错
                raise ValueError("Config error: 'save_dir' must be provided when 'structure_source' is an in-memory object.")

        # 3. 初始化并执行
        init_kwargs = {
            "structure_source": source,
            "save_dir": save_dir,
            "standardize": config.get("standardize_bulk", True)
        }

        generator = BulkToSlabGenerator(**init_kwargs)
        
        # 执行生成逻辑
        slabs = generator.generate(**gen_params).get_slabs()
        
        # 4. 保存逻辑
        if should_save and slabs:
            fmt = save_opts.get("fmt", "poscar")
            user_filename = save_opts.get("filename", save_opts.get("filename_prefix", "POSCAR"))
            
            for i, slab in enumerate(slabs):
                if len(slabs) == 1:
                    fname = user_filename
                else:
                    fname = f"{user_filename}_term{i}"
                
                generator.save_slab(slab, filename=fname, fmt=fmt)
                
        return slabs

    def _normalize_miller_indices(self, miller_indices: Union[int, str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
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

    def save_slab(self, slab: Structure, filename: Union[str, Path], fmt: str = "poscar", output_dir: Optional[Union[str, Path]] = None):
        """
        保存 Slab 结构。支持智能路径推导。
        """
        file_path = Path(filename)
        
        # 1. 如果用户显式指定了 output_dir，优先使用
        if output_dir:
            target_dir = Path(output_dir)
            out_path = target_dir / file_path.name
            
        # 2. 如果用户把路径写在了 filename 里 (例如: output_path / "POSCAR")
        elif file_path.parent != Path("") and str(file_path.parent) != ".":
            target_dir = file_path.parent
            out_path = file_path
            
        # 3. 如果在类初始化时指定了全局 save_dir
        elif self.save_dir:
            target_dir = self.save_dir
            out_path = target_dir / file_path.name
            
        # 4. 兜底方案：如果什么都没指定，默认保存到当前运行目录 (cwd)
        else:
            target_dir = Path.cwd()
            out_path = target_dir / file_path.name
            
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if fmt.lower() == "poscar":
            Poscar(slab).write_file(out_path)
        else:
            slab.to(filename=str(out_path), fmt=fmt)

    def generate(
        self,
        miller_indices: Union[int, str, Tuple[int, int, int]],
        target_layers: int,
        vacuum_thickness: float = 15.0,
        shift: Optional[float] = None,
        supercell_matrix: Optional[Union[str, Tuple[int, int, int]]] = None,
        fix_bottom_layers: int = 0,
        hcluster_cutoff: float = 0.25,
        fix_top_layers: int = 0,
        all_fix: bool = False,
        symmetric: bool = False,
        center: bool = True,
        primitive: bool = True,
        lll_reduce: bool = True,
    ) -> 'BulkToSlabGenerator':
        hkl = self._normalize_miller_indices(miller_indices)
        sc_mat = parse_supercell_matrix(supercell_matrix) if supercell_matrix else None
        
        estimated_min_size = (target_layers * 2.5) + 8.0

        slabgen = SlabGenerator(
            self.bulk_structure, hkl, 
            min_slab_size=estimated_min_size, 
            min_vacuum_size=vacuum_thickness,
            center_slab=center, primitive=primitive, lll_reduce=lll_reduce
        )
        
        if shift is not None:
            raw_slabs = [slabgen.get_slab(shift=shift)]
        else:
            raw_slabs = slabgen.get_slabs(tol=0.1, max_broken_bonds=0)
        
            if not raw_slabs:
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
                warnings.warn(f"Skipping termination {i}: {e}")
                continue

        self._slabs = processed_slabs
        return self

    def get_slabs(self) -> List[Slab]:
        return [slab.copy() for slab in self._slabs]

    def get_slab(self, termination_index: int = 0) -> Slab:
        if not self._slabs:
            raise ValueError("No slabs generated yet. Call generate() first.")
        if termination_index >= len(self._slabs) or termination_index < -len(self._slabs):
            raise IndexError(f"Termination index {termination_index} out of range.")
        return self._slabs[termination_index].copy()

    def select_termination(self, index: int) -> 'BulkToSlabGenerator':
        if not self._slabs:
            raise ValueError("No slabs to select from. Call generate() first.")
        try:
            self._slabs = [self._slabs[index]]
        except IndexError:
            raise IndexError(f"Termination index {index} out of range.")
        return self

    def make_supercell(self, supercell_matrix: Union[str, Sequence[int], int]) -> 'BulkToSlabGenerator':
        if not self._slabs:
            return self
        matrix = parse_supercell_matrix(supercell_matrix)
        for slab in self._slabs:
            slab.make_supercell(matrix)
        return self

    def set_fixation(self, fix_bottom_layers: int = 0, fix_top_layers: int = 0, all_fix: bool = False, hcluster_cutoff: float = 0.25) -> 'BulkToSlabGenerator':
        if not self._slabs:
            return self
        for i in range(len(self._slabs)):
            self._slabs[i] = self.set_selective_dynamics(
                self._slabs[i], fix_bottom_layers, fix_top_layers, all_fix, hcluster_cutoff
            )
        return self