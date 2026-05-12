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

from .utils.structure_utils import get_atomic_layers, parse_supercell_matrix

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
        """应用空位或掺杂，并自动修复属性"""
        new_struct = structure.copy()
        if dopant is None:
            new_struct.remove_sites(sorted(indices, reverse=True))
        else:
            for idx in indices:
                site = new_struct[idx]
                new_struct.replace(idx, species=dopant, properties=site.properties)
        return self._sanitize_structure(new_struct)

    def _find_candidate_indices(self, substitute_element: Element, is_slab: bool = False,
                                target_layer_indices: Optional[List[int]] = None,
                                z_range: Optional[Tuple[float, float]] = None,
                                top_layers: Optional[int] = None,
                                target_cn: Optional[int] = None, cn_cutoff: Optional[float] = None,
                                use_symmetry: bool = False) -> List[int]:
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

        if is_slab or not use_symmetry: 
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
        ) -> 'StructureModify':
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
    
    def _get_dopant_distance_fingerprint(self, indices: Sequence[int]) -> tuple:
        """
        计算掺杂原子的距离指纹 (极速物理去重核心)
        """
        if len(indices) < 2:
            return tuple()
            
        # 获取选中原子的分数坐标
        frac_coords = [self._structure[i].frac_coords for i in indices]
        
        # 利用 pymatgen 的 lattice 计算考虑周期性边界条件的最短距离矩阵
        dist_matrix = self._structure.lattice.get_all_distances(frac_coords, frac_coords)
        
        # 提取上三角矩阵的距离值（即两两之间的距离）
        distances = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                # 保留 2 位小数，容忍微小的数值误差
                distances.append(round(dist_matrix[i, j], 2))
                
        # 排序后作为元组返回，这就是该构型的唯一指纹
        return tuple(sorted(distances))
    
    def generate_defects_batch(
        self, 
        substitute_element: Union[Element, str], 
        dopant: Union[Element, str, None] = None, 
        dopant_num: Union[int, float] = 1, 
        num_structs: int = 1, 
        random_seed: Optional[int] = None, 
        **kwargs
    ) -> List[Structure]:
        """批量生成随机掺杂或空位结构 (采用极速距离指纹去重)"""
        if random_seed is not None:
            random.seed(random_seed)
            
        sub_el = self._parse_element(substitute_element)
        dopant_el = self._parse_element(dopant)

        # 1. 如果只掺杂 1 个原子，直接使用对称性消除，这是最快且最完美的
        kwargs['use_symmetry'] = False
        candidates = self._find_candidate_indices(sub_el, **kwargs)
        n_max = len(candidates)
        
        if n_max == 0:
            logging.warning(f"No candidates found for {substitute_element}.")
            return []

        n_sub = max(1, math.ceil(dopant_num * n_max)) if isinstance(dopant_num, float) else min(dopant_num, n_max)
        
        # --- 特殊极速通道：单原子掺杂 ---
        if n_sub == 1:
            kwargs['use_symmetry'] = True # 开启对称性消除
            sym_candidates = self._find_candidate_indices(sub_el, **kwargs)
            results = []
            for idx in sym_candidates[:num_structs]:
                results.append(self._apply_defects(self._structure, [idx], dopant_el))
            return results
        # -------------------------------

        logging.info(f"Total {substitute_element} atoms: {n_max}. Will replace {n_sub} of them with {dopant}.")

        max_comb = math.comb(n_max, n_sub)
        
        unique_structures = []
        seen_fingerprints = set()
        
        # 放大尝试次数，因为会有很多物理等价的结构被拒绝
        max_attempts = num_structs * 100 
        attempts = 0
        
        while len(unique_structures) < num_structs and attempts < max_attempts:
            attempts += 1
            chosen = tuple(sorted(random.sample(candidates, n_sub)))
            
            # 计算极速距离指纹
            fingerprint = self._get_dopant_distance_fingerprint(chosen)
            
            # 如果指纹已经存在，说明物理构型重复，直接跳过 (耗时几乎为 0)
            if fingerprint in seen_fingerprints:
                continue
                
            seen_fingerprints.add(fingerprint)
            
            # 只有当构型真正独特时，才去执行耗时的结构复制和替换操作
            new_struct = self._apply_defects(self._structure, chosen, dopant_el)
            unique_structures.append(new_struct)
                
        if len(unique_structures) < num_structs:
            logging.warning(f"Only found {len(unique_structures)} physically unique structures after {attempts} attempts.")
            
        return unique_structures

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
        """
        按层级 (Level-by-Level) 逐步生成掺杂结构 (广度优先搜索)。
        彻底消除早期步骤的冗余结构。
        """
        if random_seed is not None:
            random.seed(random_seed)
            
        sub_el = self._parse_element(substitute_element)
        dopant_el = self._parse_element(dopant)

        # 获取所有候选位点
        kwargs['use_symmetry'] = False
        all_candidates = self._find_candidate_indices(sub_el, **kwargs)
        
        # 第一步：强制使用对称性消除，获取物理上唯一的起点
        kwargs['use_symmetry'] = True
        sym_candidates = self._find_candidate_indices(sub_el, **kwargs)
        
        # current_paths 记录当前层级的所有路径，每条路径是一个元组 (idx1, idx2, ...)
        current_paths = [ (idx,) for idx in sym_candidates ]
        
        # 如果初始对称性较低，导致第一步就有多种可能，我们限制其数量
        if len(current_paths) > max_structures_num:
            current_paths = random.sample(current_paths, max_structures_num)
            
        all_levels_paths = [current_paths]
        
        # 从第 2 步开始逐层扩展
        for step in range(2, max_steps + 1):
            next_paths = []
            seen_fingerprints = set()
            
            # 为了防止每次都选到相同的近邻原子，随机打乱候选列表
            shuffled_candidates = list(all_candidates)
            random.shuffle(shuffled_candidates)
            
            # 遍历上一层的所有有效路径，尝试添加一个新原子
            for path in current_paths:
                for c in shuffled_candidates:
                    if c in path:
                        continue
                        
                    # 组合成新的路径并排序，确保指纹计算的一致性
                    new_path = tuple(sorted(path + (c,)))
                    fp = self._get_dopant_distance_fingerprint(new_path)
                    
                    if fp not in seen_fingerprints:
                        seen_fingerprints.add(fp)
                        next_paths.append(new_path)
                        
                        # 如果当前层级已经收集到了足够多的唯一结构，提前结束当前层的搜索
                        if len(next_paths) >= max_structures_num:
                            break
                if len(next_paths) >= max_structures_num:
                    break
                    
            current_paths = next_paths
            all_levels_paths.append(current_paths)
            
        # 将原子索引序列转换为真实的 Structure 对象
        if return_all_steps:
            result_structures = []
            for level_paths in all_levels_paths:
                level_structs = []
                for path in level_paths:
                    level_structs.append(self._apply_defects(self._structure, path, dopant_el))
                result_structures.append(level_structs)
            return result_structures
        else:
            # 如果不返回所有步骤，只返回最后一步的结构
            final_paths = all_levels_paths[-1]
            return [self._apply_defects(self._structure, path, dopant_el) for path in final_paths]

    def to_string(self, fmt: str = "poscar") -> str:
        """
            将当前结构序列化为字符串。
            
            Args:
                fmt: 'xyz' | 'cif' | 'poscar'
            
            Returns:
                对应格式的字符串内容
        """
        structure = self.get_structure()
        fmt = fmt.lower()
        
        if fmt == "xyz":
            from pymatgen.io.xyz import XYZ
            return str(XYZ(structure))
        elif fmt == "cif":
            from pymatgen.io.cif import CifWriter
            return str(CifWriter(structure))
        elif fmt in ["poscar", "vasp"]:
            from pymatgen.io.vasp import Poscar
            return str(Poscar(structure))
        else:
            raise ValueError(f"Unsupported format: {fmt}. Supported formats are: poscar, xyz, cif.")
        
    def remove_atom(self, index: int) -> 'StructureModify':
        """移除指定索引的原子"""
        if index < 0 or index >= len(self._structure):
            raise IndexError(f"Atom index {index} out of range (0-{len(self._structure)-1})")
        self._structure.remove_sites([index])
        self._structure = self._sanitize_structure(self._structure)
        self._refresh_fixed_coords()
        return self
    
    def get_structure_info(self) -> dict:
        """
        返回当前结构的晶格参数、原子统计等摘要信息。
        设计为 FastAPI 响应体的直接数据源。
        
        Returns:
            包含以下字段的字典：
            - a, b, c (float): 晶格长度，单位 Å
            - alpha, beta, gamma (float): 晶格角度，单位 °
            - volume (float): 晶胞体积，单位 Å³
            - num_atoms (int): 原子总数
            - formula (str): 化学式（如 "Pt6"）
            - reduced_formula (str): 最简化学式（如 "Pt"）
            - cell_type (str): 'bulk' 或 'slab'（根据 c 轴判断）
        """
        lattice = self._structure.lattice
        composition = self._structure.composition
        
        # 简单启发式判断：c 轴显著大于 a/b 时认为是 slab（含真空层）
        cell_type = "slab" if lattice.c > max(lattice.a, lattice.b) * 1.5 else "bulk"
        
        return {
            "a": round(lattice.a, 4),
            "b": round(lattice.b, 4),
            "c": round(lattice.c, 4),
            "alpha": round(lattice.alpha, 4),
            "beta": round(lattice.beta, 4),
            "gamma": round(lattice.gamma, 4),
            "volume": round(lattice.volume, 4),
            "num_atoms": len(self._structure),
            "formula": composition.formula.replace(" ", ""),
            "reduced_formula": composition.reduced_formula,
            "cell_type": cell_type,
        }