"""
Structure modification tools for VASP catalysis simulations.
Optimized for AI-driven platforms, supporting both Fluent API (chained) and step-by-step usage.
"""
import random
import math
import logging
import numpy as np
from typing import Optional, Union, Dict, List, Tuple, Sequence, Set

from pymatgen.core import Structure, Element
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp

from ..utils.structure_utils import get_atomic_layers, parse_supercell_matrix


class StructureModify:
    """
    结构修改工具类。
    
    支持两种调用方式：
    1. 链式调用 (Fluent API):
       struct = StructureModify(s).make_supercell("2x2x1").insert_atom("Pt", [0,0,0]).get_structure()
       
    2. 分步调用:
       modifier = StructureModify(s)
       modifier.make_supercell("2x2x1")
       modifier.insert_atom("Pt", [0,0,0])
       struct = modifier.get_structure()
    """
    def __init__(self, structure: Structure):
        # 内部始终维护一份正在修改的结构
        self._structure = structure.copy()
        # 备份初始传入的结构
        self._initial_structure = structure.copy()
        self._fixed_frac_coords: Set[Tuple[float, float, float]] = set()
        self._refresh_fixed_coords()
    
    # ==========================================
    # 状态获取与重置
    # ==========================================
    def get_structure(self) -> Structure:
        """获取当前修改完成的结构"""
        return self._sanitize_structure(self._structure).copy()

    def get_initial_structure(self) -> Structure:
        """获取未经过任何修改的初始原始结构"""
        return self._initial_structure.copy()

    def reset_to_initial(self) -> 'StructureModify':
        """重置内部状态为初始导入的结构 (返回 self 以支持链式调用)"""
        self._structure = self._initial_structure.copy()
        self._refresh_fixed_coords()
        return self

    # ==========================================
    # 内部辅助方法 (Private)
    # ==========================================
    def _refresh_fixed_coords(self):
        """刷新被固定的原子坐标集合（用于掺杂/空位筛选时避开固定层）"""
        self._fixed_frac_coords.clear()
        for site in self._structure:
            sdyn = getattr(site, "selective_dynamics", None)
            if sdyn is not None and not np.any(sdyn):
                self._fixed_frac_coords.add(tuple(site.frac_coords))
    
    def _parse_element(self, el: Union[Element, str, None]) -> Optional[Element]:
        """统一解析元素输入，兼容字符串、Element对象和空位"""
        if isinstance(el, str):
            if el.lower() in ["vac", "vacancy", "none", ""]:
                return None
            return Element(el)
        return el

    def _sanitize_structure(self, structure: Structure) -> Structure:
        """检查并修复 selective_dynamics 属性，防止 VASP 报错"""
        if "selective_dynamics" in structure.site_properties:
            new_sd = []
            for site in structure:
                sd = site.properties.get("selective_dynamics")
                new_sd.append(sd if sd is not None else [True, True, True])
            structure.add_site_property("selective_dynamics", new_sd)
        return structure

    def _apply_defects(self, structure: Structure, indices: Sequence[int], dopant: Optional[Element]) -> Structure:
        """核心修改逻辑：应用空位或掺杂，并自动修复属性"""
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
        """寻找可替换的候选原子索引"""
        candidates = [i for i, s in enumerate(self._structure) 
                      if s.specie == substitute_element and tuple(s.frac_coords) not in self._fixed_frac_coords]
        
        if not candidates:
            raise ValueError(f"No candidate sites found for {substitute_element} (excluding fixed sites).")

        if target_layer_indices is not None:
            candidates = [i for i in candidates if i in target_layer_indices]
        elif z_range is not None:
            z_min, z_max = z_range
            c_len = self._structure.lattice.c
            if z_max > 1.0: 
                z_min, z_max = z_min / c_len, z_max / c_len
            candidates = [i for i in candidates if z_min <= self._structure[i].frac_coords[2] <= z_max]
        elif top_layers is not None:
            sorted_by_z = sorted(candidates, key=lambda i: self._structure[i].frac_coords[2], reverse=True)
            candidates = sorted_by_z[:top_layers]

        if not candidates:
            raise ValueError("No candidate sites found after geometric filtering.")

        if target_cn is not None:
            cutoff = cn_cutoff or 2.0
            filtered = []
            for i in candidates:
                try:
                    cn = len(self._structure.get_neighbors(self._structure[i], cutoff))
                    if abs(cn - target_cn) <= 0.1: 
                        filtered.append(i)
                except Exception: 
                    continue
            candidates = filtered
            if not candidates:
                raise ValueError(f"No sites with CN={target_cn} found.")

        if is_slab: 
            return candidates
        
        try:
            sga = SpacegroupAnalyzer(self._structure)
            sym_struct = sga.get_symmetrized_structure()
            reduced = []
            for group in sym_struct.equivalent_indices:
                intersection = [idx for idx in group if idx in candidates]
                if intersection: 
                    reduced.append(intersection[0])
            return reduced
        except Exception as e:
            logging.warning(f"Symmetry analysis failed: {e}. Using all candidates.")
            return candidates

    # ==========================================
    # 查询方法 (返回查询结果，不改变结构)
    # ==========================================
    def get_layers(self, hcluster_cutoff: float = 0.25, target_element: Union[Element, str] = None) -> List[List[int]]:
        """获取结构中的原子层"""
        layers = get_atomic_layers(self._structure, hcluster_cutoff)
        
        if target_element:
            target_el = self._parse_element(target_element)
            filtered_layers = []
            for layer in layers:
                filtered_layer = [idx for idx in layer if self._structure[idx].specie == target_el]
                if filtered_layer:
                    filtered_layers.append(filtered_layer)
            return filtered_layers
            
        return layers

    # ==========================================
    # 结构修改方法 (返回 self 支持链式调用)
    # ==========================================
    def make_supercell(self, supercell_matrix: Union[str, Sequence[int], int] = 1) -> 'StructureModify':
        """创建超胞"""
        matrix = parse_supercell_matrix(supercell_matrix)
        self._structure.make_supercell(matrix)
        self._refresh_fixed_coords()
        return self

    def replace_species_all(self, species_map: Dict) -> 'StructureModify':
        """替换所有指定元素"""
        clean_map = {self._parse_element(k): self._parse_element(v) for k, v in species_map.items()}
        self._structure.replace_species(clean_map)
        self._refresh_fixed_coords()
        return self

    def insert_atom(self, element: Union[Element, str], frac_coord: Sequence[float]) -> 'StructureModify':
        """在指定分数坐标插入原子"""
        el = self._parse_element(element)
        self._structure.insert(len(self._structure), el, frac_coord)
        self._structure = self._sanitize_structure(self._structure)
        return self

    def modify_atom_element(self, index: int, new_element: Union[Element, str]) -> 'StructureModify':
        """修改单个原子的元素"""
        if index < 0 or index >= len(self._structure):
            raise IndexError(f"Atom index {index} out of range (0-{len(self._structure)-1})")
        el = self._parse_element(new_element)
        self._structure.replace(index, el)
        self._structure = self._sanitize_structure(self._structure)
        return self

    def modify_atom_coords(self, index: int, coords: Sequence[float], frac_coords: bool = True) -> 'StructureModify':
        """修改单个原子的坐标"""
        if index < 0 or index >= len(self._structure):
            raise IndexError(f"Atom index {index} out of range (0-{len(self._structure)-1})")
        if len(coords) != 3:
            raise ValueError("Coordinates must be a sequence of length 3")
            
        site = self._structure[index]
        
        # pymatgen 的 replace 方法默认接收分数坐标
        # 如果传入的是笛卡尔坐标，需要先转换为分数坐标
        f_coords = coords if frac_coords else self._structure.lattice.get_fractional_coords(coords)
        
        # 使用 replace 替换该位置的原子（保持元素和属性不变，只改变坐标）
        self._structure.replace(index, species=site.species, coords=f_coords, properties=site.properties)
        return self


    def batch_modify_coords(self, indices: List[int], coords_list: List[Sequence[float]], frac_coords: bool = True) -> 'StructureModify':
        """批量修改多个原子的坐标"""
        if len(indices) != len(coords_list):
            raise ValueError(f"Number of indices ({len(indices)}) must match number of coordinates ({len(coords_list)})")
        
        for i, coords in zip(indices, coords_list):
            if i < 0 or i >= len(self._structure):
                raise IndexError(f"Atom index {i} out of range (0-{len(self._structure)-1})")
            if len(coords) != 3:
                raise ValueError(f"Coordinates for atom {i} must be length 3, got {len(coords)}")
            
            site = self._structure[i]
            
            f_coords = coords if frac_coords else self._structure.lattice.get_fractional_coords(coords)
            self._structure.replace(i, species=site.species, coords=f_coords, properties=site.properties)
            
        return self

    def modify_lattice(self, a: Optional[float] = None, b: Optional[float] = None, 
                      c: Optional[float] = None, alpha: Optional[float] = None, 
                      beta: Optional[float] = None, gamma: Optional[float] = None, 
                      scale_sites: bool = True) -> 'StructureModify':
        """修改晶胞参数"""
        lattice = self._structure.lattice
        new_a = a if a is not None else lattice.a
        new_b = b if b is not None else lattice.b
        new_c = c if c is not None else lattice.c
        new_alpha = alpha if alpha is not None else lattice.alpha
        new_beta = beta if beta is not None else lattice.beta
        new_gamma = gamma if gamma is not None else lattice.gamma
        
        new_lattice = Lattice.from_parameters(new_a, new_b, new_c, new_alpha, new_beta, new_gamma)
        self._structure.lattice = new_lattice
        return self

    def modify_lattice_vectors(self, matrix: Union[np.ndarray, List[List[float]]]) -> 'StructureModify':
        """通过提供新的晶格矩阵来修改晶格"""
        matrix_array = np.array(matrix)
        if matrix_array.shape != (3, 3):
            raise ValueError(f"Lattice matrix must be 3x3, got shape {matrix_array.shape}")
        
        self._structure.lattice = Lattice(matrix_array)
        return self

    def transform_structure(self, transformation_matrix: Union[np.ndarray, List[List[float]]]) -> 'StructureModify':
        """应用变换矩阵到结构上 (修复了原版坐标不联动的 Bug)"""
        matrix = np.array(transformation_matrix)
        if matrix.shape != (3, 3):
            raise ValueError(f"Transformation matrix must be 3x3, got {matrix.shape}")
        
        op = SymmOp.from_rotation_and_translation(rotation_matrix=matrix, translation_vec=np.zeros(3))
        self._structure.apply_operation(op, fractional=False)
        return self

    # ==========================================
    # 缺陷/掺杂批量生成模块 (返回 List[Structure])
    # ==========================================
    def generate_defects_batch(
        self, 
        substitute_element: Union[Element, str], 
        dopant: Union[Element, str, None] = None, 
        dopant_num: Union[int, float] = 1, 
        num_structs: int = 1, 
        random_seed: Optional[int] = None, 
        **kwargs
    ) -> List[Structure]:
        """批量生成随机掺杂或空位结构"""
        if random_seed is not None:
            random.seed(random_seed)
            
        sub_el = self._parse_element(substitute_element)
        dopant_el = self._parse_element(dopant)

        candidates = self._find_candidate_indices(sub_el, **kwargs)
        n_max = len(candidates)
        if n_max == 0:
            return []

        n_sub = max(1, math.ceil(dopant_num * n_max)) if isinstance(dopant_num, float) else min(dopant_num, n_max)

        results = []
        seen = set()
        for _ in range(num_structs * 10):
            if len(results) >= num_structs: 
                break
                
            chosen = tuple(sorted(random.sample(candidates, n_sub)))
            fingerprint = tuple(tuple(self._structure[i].frac_coords) for i in chosen)
            
            if fingerprint in seen: 
                continue
            seen.add(fingerprint)
            
            new_struct = self._apply_defects(self._structure, chosen, dopant_el)
            results.append(new_struct)
            
        return results

    def generate_defects_step_by_step(
        self, 
        substitute_element: Union[Element, str],
        dopant: Union[Element, str, None] = None, 
        max_steps: int = 3,
        max_structures_num: int = 15, 
        return_all_steps: bool = False, 
        random_seed: Optional[int] = None, 
        **kwargs
    ) -> Union[List[Structure], List[List[Structure]]]:
        """逐步生成掺杂或空位结构"""
        if random_seed is not None:
            random.seed(random_seed)
            
        sub_el = self._parse_element(substitute_element)
        dopant_el = self._parse_element(dopant)

        try:
            candidates = self._find_candidate_indices(sub_el, **kwargs)
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
            if len(all_paths) >= max_structures_num: 
                break
            chosen_seq = random.sample(candidates, max_steps)
            path_fingerprint = frozenset(chosen_seq)
            if path_fingerprint in seen_paths: 
                continue
            seen_paths.add(path_fingerprint)

            current_path = []
            for step in range(1, max_steps + 1):
                indices_subset = chosen_seq[:step]
                # 基于当前状态的结构进行缺陷累加
                step_struct = self._apply_defects(self._structure, indices_subset, dopant_el)
                current_path.append(step_struct)
            
            all_paths.append(current_path)
            final_structs.append(current_path[-1])

        return all_paths if return_all_steps else final_structs
